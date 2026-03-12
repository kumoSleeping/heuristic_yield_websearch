"""
core/entari_plugin_hyw.py - entari 插件 (保留原项目绝大部分功能)

功能:
- .q 命令触发问答
- 引用消息 + 图片处理
- 配置覆盖 (entari 插件配置 → config.yml)
- 搜索进度/里程碑推送
- Markdown 图片提取
- 完整的事件捕获与展示
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger

from arclet.alconna import Alconna, AllParam, Args, Arparma
from arclet.entari import (
    At,
    BasicConfModel,
    Image,
    MessageChain,
    MessageCreatedEvent,
    Quote,
    Session,
    Text,
    command,
    listen,
    metadata,
    plugin_config,
)
from arclet.entari.event.command import CommandReceive
from arclet.entari.event.lifespan import Cleanup, Startup

from .main import Stats, load_config, run, startup_tools, shutdown_tools, cfg_get, CONFIG_PATH

try:
    from importlib.metadata import version as get_version

    for _dist_name in ("hyw", "entari-plugin-hyw-ai"):
        try:
            __version__ = get_version(_dist_name)
            break
        except Exception:
            continue
    else:
        __version__ = "6.0.0"
except Exception:
    __version__ = "6.0.0"


_IMG_TAG_RE = re.compile(r"<img[^>]+>", flags=re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)", flags=re.IGNORECASE)
_SEARCH_TIME_RANGE = {"a": "全时段", "d": "近1日", "w": "近1周", "m": "近1月", "y": "近1年"}


# ── 工具函数 ──────────────────────────────────────────────────

def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = str(value or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        try:
            parsed = yaml.safe_load(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _format_web_search_query(arguments: Any) -> str:
    payload = _as_dict(arguments)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    query_list = data.get("queries") if isinstance(data.get("queries"), list) else []
    if query_list:
        query = " || ".join(str(item or "").strip() for item in query_list if str(item or "").strip())
    else:
        query = str(data.get("query") or data.get("user_message") or data.get("description") or "").strip()
    if not query:
        return ""
    time_key = str(data.get("time_range") or "").strip().lower()
    if time_key and time_key != "a":
        return f"{query} | [{_SEARCH_TIME_RANGE.get(time_key, time_key)}]"
    return query


def _format_search_progress_lines(queries: list[str]) -> str:
    if not queries:
        return ""
    rows = ["🔍 正在搜索:"]
    for i, text in enumerate(queries[:8], start=1):
        rows.append(f"  {i}. {text}")
    return "\n".join(rows).strip()


# ── 插件配置 ──────────────────────────────────────────────────

@dataclass
class PluginConfig(BasicConfModel):
    question_command: str = ".q"
    quote: bool = False

    # 配置穿透字段
    api_key: str | None = None
    api_base: str | None = None
    model: str | None = None
    active_model: str | None = None
    models: Any | None = None
    system_prompt: str | None = None
    language: str | None = None
    language_style: str | None = None
    abilitys_headless: bool | None = None
    max_rounds: int | None = None

    # 旧字段兼容
    headless: bool = False
    theme_color: str = "#ef4444"
    activate_global_config: bool = True
    base_url: str | None = None
    model_name: str | None = None
    temperature: float | None = None


conf = plugin_config(PluginConfig)


def _normalize_conf_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    return value


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(value or "").strip())
    safe = re.sub(r"\s+", "_", safe).strip("._")
    return (safe or "query")[:48]


def _extract_markdown_images(answer: str) -> tuple[list[str], str]:
    text = str(answer or "")
    image_sources: list[str] = []
    for raw in _MD_IMAGE_RE.findall(text):
        src = str(raw or "").strip().strip("<>").strip()
        if not src:
            continue
        image_sources.append(src)
    cleaned = _MD_IMAGE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return image_sources, cleaned


def _build_core_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    def _pick(primary_field: str, legacy_field: str | None = None) -> Any | None:
        primary_value = _normalize_conf_value(getattr(conf, primary_field, None))
        if primary_value is not None:
            return primary_value
        if legacy_field:
            return _normalize_conf_value(getattr(conf, legacy_field, None))
        return None

    key_map = {
        "api_key": _pick("api_key"),
        "api_base": _pick("api_base", "base_url"),
        "model": _pick("model", "model_name"),
        "active_model": _pick("active_model"),
        "models": _normalize_conf_value(getattr(conf, "models", None)),
        "system_prompt": _pick("system_prompt"),
        "language": _pick("language"),
        "language_style": _pick("language_style"),
        "headless": _pick("abilitys_headless", "headless"),
        "max_rounds": _pick("max_rounds"),
    }

    for key, value in key_map.items():
        if value is not None:
            overrides[key] = value

    return overrides


def _warn_ignored_legacy_fields() -> None:
    ignored = {
        "temperature": _normalize_conf_value(conf.temperature),
    }
    used = sorted([k for k, v in ignored.items() if v is not None])
    if used:
        logger.warning("以下 entari 旧配置字段已忽略: {}", ", ".join(used))
    if conf.headless:
        logger.warning("配置项 abilitys_headless/headless 已映射为 headless。")
    if str(conf.theme_color or "").strip() not in {"", "#ef4444"}:
        logger.warning("配置项 theme_color 已忽略。")


def _apply_overrides_to_core_config() -> None:
    overrides = _build_core_overrides()
    if conf.headless:
        if "headless" not in overrides:
            overrides["headless"] = True

    if not overrides:
        return

    if not conf.activate_global_config:
        logger.warning("activate_global_config=false, 忽略覆盖字段。")
        return

    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        raw = {}

    merged = raw if isinstance(raw, dict) else {}
    merged.update(overrides)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Activated entari overrides into {}: {}", CONFIG_PATH, ", ".join(sorted(overrides.keys())))


# ── 图片处理 (复用原有逻辑) ──────────────────────────────────
try:
    from entari_plugin_hyw.entari_misc import process_images
except ImportError:
    # fallback: 内联简单版
    import base64
    import io
    import httpx
    from PIL import Image as PILImage

    def _compress_b64(b64: str, quality: int = 85) -> str:
        img = PILImage.open(io.BytesIO(base64.b64decode(b64)))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    async def process_images(mc: MessageChain, vision_model=None):
        if vision_model == "off":
            return [], None
        images = []
        if mc.get(Image):
            urls = mc[Image].map(lambda item: item.src)
            async with httpx.AsyncClient(timeout=30.0) as client:
                for url in urls:
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            b64 = base64.b64encode(resp.content).decode()
                            images.append(_compress_b64(b64))
                    except Exception as e:
                        logger.warning("download image failed: {}", e)
        return images, None


# ── 命令处理 ──────────────────────────────────────────────────

@listen(CommandReceive)
async def remove_at(content: MessageChain):
    return content.lstrip(At)


alc_q = Alconna(conf.question_command, Args["content;?", AllParam])


@command.on(alc_q)
async def handle_question(session: Session[MessageCreatedEvent], result: Arparma):
    content = result.all_matched_args.get("content")
    mc = MessageChain(content) if content else MessageChain()

    if session.reply:
        try:
            mc.extend(MessageChain(" ") + session.reply.origin.message)
        except Exception:
            try:
                mc.extend(session.reply.origin.message)
            except Exception:
                logger.debug("Reply message append skipped.")

    user_input = str(mc.get(Text)).strip() if mc.get(Text) else ""
    user_input = _IMG_TAG_RE.sub("", user_input).strip()

    if not user_input and not mc.get(Image):
        return

    images, _ = await process_images(mc, None)
    question = user_input or "请根据图片内容进行分析并回答。"

    # 搜索/进度跟踪
    sent_progress: set[str] = set()
    progress_lines: list[str] = []
    progress_lock = threading.Lock()
    search_queries: list[str] = []
    search_seen: set[str] = set()
    search_lock = threading.Lock()

    def _push_progress(text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        with progress_lock:
            if msg in sent_progress:
                return
            sent_progress.add(msg)
            progress_lines.append(msg)

    def _on_tool(name: str, args: dict):
        if name == "web_search":
            line = _format_web_search_query(args)
            if line:
                with search_lock:
                    if line not in search_seen:
                        search_seen.add(line)
                        search_queries.append(line)

    config = load_config()
    stats = Stats()

    try:
        final_content = await asyncio.to_thread(
            run, question, config=config, stats=stats, on_tool=_on_tool,
        )
        with search_lock:
            search_hint = _format_search_progress_lines(list(search_queries))
        if search_hint:
            _push_progress(search_hint)

        with progress_lock:
            combined_progress = "\n\n".join(progress_lines).strip()
        if combined_progress:
            await session.send(MessageChain(Text(combined_progress)))

        answer_text = str(final_content or "").strip()
        if not answer_text:
            await session.send("未生成有效回答。")
            return

        chain = MessageChain()
        image_sources, cleaned_text = _extract_markdown_images(answer_text)
        if image_sources:
            if cleaned_text:
                chain.append(Text(cleaned_text))
            for image_src in image_sources[:6]:
                chain.append(Image(src=image_src))
        else:
            chain.append(Text(answer_text))

        if conf.quote:
            chain = MessageChain(Quote(session.event.message.id)) + chain

        await session.send(chain)

    except Exception as exc:
        logger.exception("Error in hyw execution: {}", exc)
        await session.send(f"执行出错: {exc}")


# ── 生命周期 ──────────────────────────────────────────────────

@listen(Startup)
async def on_startup():
    logger.info("HYW plugin startup: loading configuration...")

    _warn_ignored_legacy_fields()

    try:
        _apply_overrides_to_core_config()
    except Exception as exc:
        logger.warning("Failed to apply entari overrides: {}", exc)

    try:
        cfg = load_config()
        logger.success(
            "HYW plugin ready. model={}, api_base={}, headless={}",
            cfg.get("model"),
            cfg.get("api_base"),
            cfg.get("headless"),
        )
        headless = cfg.get("headless") not in (False, "false", "0", 0)
        try:
            logger.info("tools startup (headless={})", headless)
            startup_tools(headless=headless)
            logger.info("tools startup done")
        except Exception as exc:
            logger.warning("tools startup skipped: {}", exc)
    except Exception as exc:
        logger.warning("HYW plugin ready with unknown config state: {}", exc)


@listen(Cleanup)
async def cleanup_resources():
    logger.info("Cleaning up hyw resources (tools)...")
    shutdown_tools()


__plugin__ = metadata(
    "hyw",
    author=[{"name": "kumo", "email": "dev@example.com"}],
    version=__version__,
    config=PluginConfig,
)
