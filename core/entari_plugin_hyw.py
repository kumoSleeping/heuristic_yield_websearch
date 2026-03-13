"""
core/entari_plugin_hyw.py - entari 插件 (保留原项目绝大部分功能)

功能:
- .q 命令触发问答
- 引用消息 + 图片处理
- 读取 ~/.hyw/config.yml
- 搜索进度/里程碑推送
- Markdown 图片提取
- 完整的事件捕获与展示
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
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

from .config import load_config
from .main import Stats, run_stream, shutdown_tools, startup_tools
from .render import render_markdown_result

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


conf = plugin_config(PluginConfig)


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


def _clean_answer(text: str) -> str:
    cleaned = re.sub(
        r"<summary>\s*(.*?)\s*</summary>",
        lambda m: "\n".join(
            f"> {line}"
            for line in [row.strip() for row in m.group(1).splitlines()]
            if line
        ),
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _image_to_link(match: re.Match[str]) -> str:
        alt = str(match.group(1) or "").strip()
        raw_target = str(match.group(2) or "").strip()
        if not raw_target:
            return match.group(0)
        label = alt or "图片"
        if raw_target.lower().startswith("data:image"):
            return f"{label}: [inline image omitted]"
        return f"{label}: [{raw_target}]({raw_target})"

    cleaned = _MD_IMAGE_RE.sub(_image_to_link, cleaned)
    cleaned = re.sub(r"</?(?:search|wiki|page|tool_results|result)\b[^>]*>", "", cleaned)
    return cleaned.strip()


def _format_tool_argument(name: str, arguments: Any) -> str:
    payload = _as_dict(arguments)
    if name in ("web_search", "web_search_wiki"):
        return _format_web_search_query(payload)
    if name == "page_extract":
        return str(payload.get("url") or "").strip()
    return json.dumps(payload, ensure_ascii=False) if payload else ""


def _format_tool_trace_line(name: str, arguments: Any) -> str:
    payload = _as_dict(arguments)
    argument = _format_tool_argument(name, payload)
    line = f"> {name}"
    if argument:
        line += f"({argument})"

    extras: list[str] = []
    count = payload.get("_count")
    ok = payload.get("_ok")
    jina_tokens = payload.get("_jina_tokens")
    elapsed_s = payload.get("_elapsed_s")
    if ok is False:
        extras.append("!")
    elif count not in (None, ""):
        extras.append(f"{count} ✓")
    elif ok is True:
        extras.append("✓")
    if jina_tokens not in (None, ""):
        extras.append(f"jina {jina_tokens}tok")
    if elapsed_s not in (None, ""):
        try:
            extras.append(f"{float(elapsed_s):.1f}s")
        except Exception:
            pass

    if extras:
        line += " " + " ".join(extras)
    return line.strip()


def _render_result_to_src(payload: dict[str, Any]) -> str:
    base64_data = str(payload.get("base64") or "").strip()
    if not base64_data:
        return ""
    mime_type = str(payload.get("mime_type") or "image/png").strip() or "image/png"
    return f"data:{mime_type};base64,{base64_data}"


def _build_notice_chain(text: str, quote_id: str | None = None) -> MessageChain:
    chain = MessageChain(Text(text))
    if quote_id:
        chain = MessageChain(Quote(quote_id)) + chain
    return chain


def _tool_text_line(name: str, args: dict[str, Any]) -> str:
    display_name = str(args.get("_display_name") or name).strip() or str(name or "").strip()
    argument = _format_tool_argument(name, args)
    line = f"> {display_name}"
    if argument:
        line += f"({argument})"
    return line


def _planned_tool_block(tools: list[tuple[str, dict[str, Any]]] | None) -> str:
    if not tools:
        return ""
    rows: list[str] = []
    for name, args in tools:
        line = _tool_text_line(name, args)
        if line:
            rows.append(line)
    return "\n".join(rows).strip()


async def _build_answer_chain(
    answer_text: str,
    *,
    config: dict[str, Any],
    quote_id: str | None = None,
) -> tuple[MessageChain, MessageChain, bool]:
    image_sources, cleaned_text = _extract_markdown_images(answer_text)
    render_text = _clean_answer(cleaned_text if image_sources else answer_text)

    fallback_chain = _build_notice_chain("渲染出错", quote_id)

    primary_chain = fallback_chain
    rendered_ok = False
    if render_text:
        try:
            payload = await render_markdown_result(
                render_text,
                title="HYW",
                config=config,
                headless=bool(config.get("headless") not in (False, "false", "0", 0)),
            )
            rendered_src = _render_result_to_src(payload if isinstance(payload, dict) else {})
            if rendered_src:
                primary_chain = MessageChain(Image(src=rendered_src))
                rendered_ok = True
        except Exception as exc:
            logger.warning("Markdown render failed in entari plugin: {}", exc)
    if quote_id and rendered_ok:
        primary_chain = MessageChain(Quote(quote_id)) + primary_chain

    return primary_chain, fallback_chain, rendered_ok


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
    quote_id = session.event.message.id if conf.quote else None
    loop = asyncio.get_running_loop()
    sent_round_messages: set[str] = set()
    sent_round_lock = threading.Lock()

    def _build_round_chain(text: str) -> MessageChain:
        chain = MessageChain(Text(text))
        if quote_id:
            chain = MessageChain(Quote(quote_id)) + chain
        return chain

    def _on_rewind(thinking: str, tools: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        clean = _clean_answer(thinking)
        planned_block = _planned_tool_block(tools)
        if planned_block:
            clean = f"{clean}\n{planned_block}".strip() if clean else planned_block
        if not clean:
            return
        with sent_round_lock:
            if clean in sent_round_messages:
                return
            sent_round_messages.add(clean)
        try:
            future = asyncio.run_coroutine_threadsafe(
                session.send(_build_round_chain(clean)),
                loop,
            )
            future.result()
        except Exception as exc:
            logger.warning("Round message send failed in entari plugin: {}", exc)

    config = load_config()
    stats = Stats()

    try:
        final_content = await asyncio.to_thread(
            run_stream,
            question,
            config=config,
            stats=stats,
            on_rewind=_on_rewind,
            images=images,
        )

        answer_text = str(final_content or "").strip()
        if not answer_text:
            return

        primary_chain, fallback_chain, rendered_ok = await _build_answer_chain(
            answer_text,
            config=config,
            quote_id=quote_id,
        )
        try:
            await session.send(primary_chain)
        except Exception as exc:
            if rendered_ok:
                logger.warning("Rendered message send failed in entari plugin: {}", exc)
                await session.send(fallback_chain)
            else:
                raise

    except Exception as exc:
        logger.exception("Error in hyw execution: {}", exc)
        return


# ── 生命周期 ──────────────────────────────────────────────────

@listen(Startup)
async def on_startup():
    logger.info("HYW plugin startup: loading configuration...")

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
            startup_info = startup_tools(headless=headless, config=cfg) or {}
            logger.info("tools startup done")
            render_info = startup_info.get("render") if isinstance(startup_info, dict) else None
            if isinstance(render_info, dict):
                logger.success(
                    "render prewarm done. platform={}, prewarmed={}, paths={}",
                    render_info.get("platform"),
                    render_info.get("prewarmed"),
                    render_info.get("paths"),
                )
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
