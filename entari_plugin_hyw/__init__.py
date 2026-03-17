"""
entari_plugin_hyw/__init__.py - entari 插件 (保留原项目绝大部分功能)

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
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

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

from core.config import DEFAULT_TOOL_SELECTIONS, cfg_get, load_config, normalize_tool_provider_name
from core.main import Stats, run, run_stream, shutdown_tools, startup_tools
from core.render import render_markdown_result

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
_TASK_LIST_BLOCK_RE = re.compile(r"<task_list\b[^>]*>.*?</task_list>", flags=re.IGNORECASE | re.DOTALL)
_ARTICLE_SKELETON_BLOCK_RE = re.compile(r"<article_skeleton\b[^>]*>.*?</article_skeleton>", flags=re.IGNORECASE | re.DOTALL)
_SEARCH_REWRITE_BLOCK_RE = re.compile(r"<search_rewrite\b[^>]*>.*?</search_rewrite>", flags=re.IGNORECASE | re.DOTALL)
_TOOL_BLOCK_RE = re.compile(r"<(?:search|wiki|sub_agent|page)\b[^>]*>.*?</(?:search|wiki|sub_agent|page)>", flags=re.IGNORECASE | re.DOTALL)
_TOOL_SELF_CLOSING_RE = re.compile(r"<(?:search|wiki|sub_agent|page)\b[^>]*/>", flags=re.IGNORECASE)
_ENTARI_RENDER_PROVIDER = "md2png_lite"


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
    render_theme: str = "paper"


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


def _render_article_skeleton_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    title_match = re.search(r"<title\b[^>]*>(.*?)</title>", block, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", str(title_match.group(1) or "").strip()) if title_match else "未命名骨架"
    lines = [f"文章骨架\n{title}"]
    section_matches = list(re.finditer(r"<section\b([^>]*)>(.*?)</section>", block, flags=re.IGNORECASE | re.DOTALL))
    if section_matches:
        for section_match in section_matches:
            attrs = _as_dict({})
            for key, dq, sq in re.findall(r'([:\w-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', str(section_match.group(1) or "")):
                attrs[str(key or "").strip().lower()] = str((dq if dq != "" else sq) or "").strip()
            section_name = str(attrs.get("name") or attrs.get("title") or "").strip() or "未命名章节"
            lines.append(f"- {section_name}")
            for claim_match in re.finditer(r"^\s*\[(\d+)\]\s+(.+?)\s*$", str(section_match.group(2) or ""), flags=re.MULTILINE):
                lines.append(f"  [{claim_match.group(1)}] {claim_match.group(2).strip()}")
    else:
        for claim_match in re.finditer(r"^\s*\[(\d+)\]\s+(.+?)\s*$", block, flags=re.MULTILINE):
            lines.append(f"- [{claim_match.group(1)}] {claim_match.group(2).strip()}")
    return "\n".join(lines).strip()


def _render_search_rewrite_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    terms = [
        re.sub(r"\s+", " ", str(term_match.group(1) or "").strip())
        for term_match in re.finditer(r"<term\b[^>]*>(.*?)</term>", block, flags=re.IGNORECASE | re.DOTALL)
    ]
    if not terms:
        inner = re.sub(r"</?search_rewrite\b[^>]*>", "", block, flags=re.IGNORECASE)
        for line in inner.splitlines():
            candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(line or "").strip())
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                terms.append(candidate)
    if not terms:
        return ""
    return "搜索词重绘\n" + "\n".join(f"- {term}" for term in terms)


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
    cleaned = _TASK_LIST_BLOCK_RE.sub("", cleaned)
    cleaned = _SEARCH_REWRITE_BLOCK_RE.sub(_render_search_rewrite_xml, cleaned)
    cleaned = _ARTICLE_SKELETON_BLOCK_RE.sub(_render_article_skeleton_xml, cleaned)
    cleaned = _TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = _TOOL_SELF_CLOSING_RE.sub("", cleaned)
    cleaned = re.sub(r"</?(?:search|wiki|sub_agent|page|tool_results|result|article_skeleton|search_rewrite|section|title|term)\b[^>]*>", "", cleaned)
    return cleaned.strip()


def _format_tool_argument(name: str, arguments: Any) -> str:
    payload = _as_dict(arguments)
    if name in ("web_search", "web_search_wiki"):
        return _format_web_search_query(payload)
    if name == "page_extract":
        url = str(payload.get("url") or "").strip()
        host = ""
        if url:
            parsed = urlparse(url)
            host = str(parsed.netloc or parsed.path or "").strip()
            if host.startswith("www."):
                host = host[4:]
        query = re.sub(r"\s+", " ", str(payload.get("query") or "").strip())
        lines = str(payload.get("lines") or "").strip()
        line_label = "all" if lines.lower() == "all" else (f"{lines}line" if lines else "")
        parts = [part for part in (host, query, line_label) if part]
        return ", ".join(parts)
    if name == "sub_agent_task":
        tools = str(payload.get("tools") or "").strip()
        url = str(payload.get("url") or "").strip()
        task = re.sub(r"\s+", " ", str(payload.get("task") or "").strip())
        host = ""
        if url:
            parsed = urlparse(url)
            host = str(parsed.netloc or parsed.path or "").strip()
            if host.startswith("www."):
                host = host[4:]
        binding = ""
        if tools and host:
            binding = f"{tools}({host})"
        elif tools:
            binding = tools
        elif host:
            binding = host
        if task:
            task = task[:120]
        if binding and task:
            return f"{binding}, {task}"
        return binding or task
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


def _flatten_provider_tokens(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = str(raw or "").strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if str(part or "").strip()]
        return [text]
    if isinstance(raw, (list, tuple, set)):
        items: list[str] = []
        for item in raw:
            items.extend(_flatten_provider_tokens(item))
        return items
    return []


def _provider_summary(config: dict[str, Any], capability: str) -> str:
    candidates = [
        cfg_get(config, f"tools.use.{capability}"),
        cfg_get(config, f"tools.{capability}.providers"),
        cfg_get(config, f"tools.{capability}.provider"),
    ]
    providers: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for item in _flatten_provider_tokens(candidate):
            normalized = normalize_tool_provider_name(str(item or "").strip())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            providers.append(normalized)
    if not providers:
        default = str(DEFAULT_TOOL_SELECTIONS.get(capability, "") or "").strip()
        if default:
            providers.append(default)
            seen.add(default)
    return " / ".join(providers).strip()


def _planned_notice_line(name: str, args: dict[str, Any], config: dict[str, Any]) -> str:
    payload = _as_dict(args)
    if str(name or "").strip() == "web_search":
        provider = _provider_summary(config, "search")
        suffix = f"[{provider}]" if provider else ""
        query = re.sub(r"\s+", " ", str(payload.get("query") or "").strip())
        return f"Search{suffix}: {query}".strip()

    if str(name or "").strip() == "page_extract":
        provider = _provider_summary(config, "page_extract")
        suffix = f"[{provider}]" if provider else ""
        url = str(payload.get("url") or "").strip()
        query = re.sub(r"\s+", " ", str(payload.get("query") or "").strip())
        host = ""
        if url:
            parsed = urlparse(url)
            host = str(parsed.netloc or parsed.path or "").strip()
        host_block = f"<{host}> " if host else ""
        return f"Page{suffix}: {host_block}{query}".strip()

    tools = str(payload.get("tools") or "").strip().lower()
    task = re.sub(r"\s+", " ", str(payload.get("task") or "").strip())
    if task:
        task = task[:200]

    if tools == "websearch":
        provider = _provider_summary(config, "search")
        suffix = f"[{provider}]" if provider else ""
        return f"Sub Agent: Web Search{suffix}: {task}".strip()

    if tools == "page":
        provider = _provider_summary(config, "page_extract")
        suffix = f"[{provider}]" if provider else ""
        url = str(payload.get("url") or "").strip()
        host = ""
        if url:
            parsed = urlparse(url)
            host = str(parsed.netloc or parsed.path or "").strip()
        host_block = f"<{host}> " if host else ""
        return f"Sub Agent: Page{suffix}: {host_block}{task}".strip()

    return ""


def _planned_sub_agent_block(
    tools: list[tuple[str, dict[str, Any]]] | None,
    *,
    config: dict[str, Any],
) -> str:
    if not tools:
        return ""
    rows: list[str] = []
    seen: set[str] = set()
    for name, args in tools:
        line = _planned_notice_line(name, args, config)
        if not line or line in seen:
            continue
        seen.add(line)
        rows.append(line)
    return "\n".join(rows).strip()


def _build_entari_render_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config) if isinstance(config, dict) else {}

    tools = merged.get("tools")
    if not isinstance(tools, dict):
        tools = {}
    else:
        tools = deepcopy(tools)
    merged["tools"] = tools

    tool_index = tools.get("index")
    if not isinstance(tool_index, dict):
        tool_index = {}
    else:
        tool_index = deepcopy(tool_index)
    tools["index"] = tool_index

    md2png_provider = tool_index.get(_ENTARI_RENDER_PROVIDER)
    if not isinstance(md2png_provider, dict):
        md2png_provider = {}
    else:
        md2png_provider = deepcopy(md2png_provider)
    md2png_provider.setdefault("render", "md2png_lite.provider:render_md2png_lite_result")
    tool_index[_ENTARI_RENDER_PROVIDER] = md2png_provider

    md2png_config = merged.get("md2png_lite")
    if not isinstance(md2png_config, dict):
        md2png_config = {}
    else:
        md2png_config = deepcopy(md2png_config)
    md2png_config["theme"] = str(conf.render_theme or md2png_config.get("theme") or "paper").strip() or "paper"
    md2png_config["font_pack"] = "auto"
    merged["md2png_lite"] = md2png_config
    return merged


def _entari_render_font_status(config: dict[str, Any]) -> str:
    try:
        from md2png_lite.font_sync import resolved_font_pack, synced_font_dirs
    except Exception as exc:
        return f"Entari render fonts: md2png-lite font sync unavailable ({exc})."

    effective = str(resolved_font_pack("auto") or "system").strip().lower() or "system"
    font_dirs = [str(item).strip() for item in synced_font_dirs("auto") if str(item).strip()]

    if effective == "noto" and font_dirs:
        return f"Entari render fonts: effective=noto, cache={font_dirs[0]}"
    return "Entari render fonts: effective=system, using system fonts. Install `hyw[notosans]` to enable Noto sync."


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
        headless = bool(config.get("headless") not in (False, "false", "0", 0))
        try:
            payload = await render_markdown_result(
                render_text,
                title="",
                config=_build_entari_render_config(config),
                provider=_ENTARI_RENDER_PROVIDER,
                headless=headless,
            )
            rendered_src = _render_result_to_src(payload if isinstance(payload, dict) else {})
            if rendered_src:
                primary_chain = MessageChain(Image(src=rendered_src))
                rendered_ok = True
        except Exception as exc:
            logger.warning("Markdown render with {} failed in entari plugin: {}", _ENTARI_RENDER_PROVIDER, exc)
            try:
                payload = await render_markdown_result(
                    render_text,
                    title="",
                    config=config,
                    headless=headless,
                )
                rendered_src = _render_result_to_src(payload if isinstance(payload, dict) else {})
                if rendered_src:
                    primary_chain = MessageChain(Image(src=rendered_src))
                    rendered_ok = True
            except Exception as fallback_exc:
                logger.warning("Markdown render fallback failed in entari plugin: {}", fallback_exc)
    if quote_id and rendered_ok:
        primary_chain = MessageChain(Quote(quote_id)) + primary_chain

    return primary_chain, fallback_chain, rendered_ok


# ── 图片处理 (复用原有逻辑) ──────────────────────────────────
try:
    from .entari_misc import process_images
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
        clean = _planned_sub_agent_block(tools, config=config) if tools else ""
        if not clean:
            clean = _clean_answer(thinking)
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
        runner = run_stream if config.get("stream") not in (False, "false", "0", 0) else run
        kwargs: dict[str, Any] = {
            "config": config,
            "stats": stats,
            "images": images,
        }
        if runner is run_stream:
            kwargs["on_rewind"] = _on_rewind
        final_content = await asyncio.to_thread(
            runner,
            question,
            **kwargs,
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
        logger.info(_entari_render_font_status(cfg))
        headless = cfg.get("headless") not in (False, "false", "0", 0)
        try:
            logger.info("tools startup (headless={})", headless)
            startup_tools(headless=headless, config=cfg)
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
