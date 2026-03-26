"""
core/entari_plugin_hyw.py - entari 插件 (保留原项目绝大部分功能)

功能:
- /q 命令触发问答
- /stop 命令停止该用户最近一次提问
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

from .config import DEFAULT_TOOL_SELECTIONS, cfg_get, load_config, normalize_tool_provider_name
from .main import Stats, StopRequestedError, build_compact_context, run, run_stream, shutdown_tools, startup_tools
from .render import render_markdown_result
from .tool_view import format_tool_view_argument, format_tool_view_text

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
_TOOL_BLOCK_RE = re.compile(
    r"<(?:search|wiki|sub_agent|page|context_keep|plan_update|context_update|context_delete)\b[^>]*>.*?</(?:search|wiki|sub_agent|page|context_keep|plan_update|context_update|context_delete)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TOOL_SELF_CLOSING_RE = re.compile(
    r"<(?:search|wiki|sub_agent|page|context_keep|plan_update|context_update|context_delete)\b[^>]*/>",
    flags=re.IGNORECASE,
)
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
    question_command: str = "/q"
    stop_command: str = "/stop"
    quote: bool = False
    verbose: bool = True
    render_theme: str = "paper"


conf = plugin_config(PluginConfig)


@dataclass
class _ActiveRequest:
    request_id: str
    task: asyncio.Task[Any]
    stop_event: threading.Event


_ACTIVE_REQUESTS: dict[str, list[_ActiveRequest]] = {}
_ACTIVE_REQUESTS_LOCK = threading.Lock()
_TURN_MEMORY: dict[str, list[dict[str, str]]] = {}
_TURN_MEMORY_LOCK = threading.Lock()
_MAX_TURN_MEMORY_ITEMS = 12


def _session_request_scope(session: Session[MessageCreatedEvent]) -> str:
    platform = str(getattr(getattr(session, "account", None), "platform", "") or "").strip() or "unknown"
    self_id = str(getattr(getattr(session, "account", None), "self_id", "") or "").strip() or "unknown"
    user_id = ""
    channel_id = ""
    try:
        user_id = str(session.user.id or "").strip()
    except Exception:
        user_id = str(getattr(getattr(session.event, "user", None), "id", "") or "").strip()
    try:
        channel_id = str(session.channel.id or "").strip()
    except Exception:
        channel_id = str(getattr(getattr(session.event, "channel", None), "id", "") or "").strip()
    target = f"user:{user_id}" if user_id else f"channel:{channel_id or 'unknown'}"
    return f"{platform}:{self_id}:{target}"


def _register_active_request(scope_key: str, request: _ActiveRequest) -> None:
    with _ACTIVE_REQUESTS_LOCK:
        bucket = [item for item in _ACTIVE_REQUESTS.get(scope_key, []) if not item.task.done()]
        bucket.append(request)
        _ACTIVE_REQUESTS[scope_key] = bucket


def _unregister_active_request(scope_key: str, request_id: str) -> None:
    with _ACTIVE_REQUESTS_LOCK:
        bucket = [
            item
            for item in _ACTIVE_REQUESTS.get(scope_key, [])
            if item.request_id != request_id and not item.task.done()
        ]
        if bucket:
            _ACTIVE_REQUESTS[scope_key] = bucket
        else:
            _ACTIVE_REQUESTS.pop(scope_key, None)


def _stop_latest_active_request(scope_key: str) -> bool:
    with _ACTIVE_REQUESTS_LOCK:
        bucket = [item for item in _ACTIVE_REQUESTS.get(scope_key, []) if not item.task.done()]
        if not bucket:
            _ACTIVE_REQUESTS.pop(scope_key, None)
            return False
        _ACTIVE_REQUESTS[scope_key] = bucket
        active = bucket[-1]

    active.stop_event.set()
    active.task.cancel()
    return True


def _message_chain_text(chain: Any) -> str:
    if chain is None:
        return ""
    try:
        text_value = chain.get(Text)
    except Exception:
        text_value = None
    text = str(text_value).strip() if text_value else ""
    return _IMG_TAG_RE.sub("", text).strip()


def _get_turn_memory(scope_key: str) -> list[dict[str, str]]:
    with _TURN_MEMORY_LOCK:
        bucket = _TURN_MEMORY.get(scope_key, [])
        return [dict(item) for item in bucket if isinstance(item, dict)]


def _append_turn_memory(scope_key: str, user_text: str, assistant_text: str) -> None:
    clean_user = str(user_text or "").strip()
    clean_assistant = str(assistant_text or "").strip()
    if not clean_user and not clean_assistant:
        return
    with _TURN_MEMORY_LOCK:
        bucket = [dict(item) for item in _TURN_MEMORY.get(scope_key, []) if isinstance(item, dict)]
        if clean_user:
            bucket.append({"role": "user", "content": clean_user})
        if clean_assistant:
            bucket.append({"role": "assistant", "content": clean_assistant})
        _TURN_MEMORY[scope_key] = bucket[-_MAX_TURN_MEMORY_ITEMS:]


def _clear_runtime_state() -> None:
    with _ACTIVE_REQUESTS_LOCK:
        _ACTIVE_REQUESTS.clear()
    with _TURN_MEMORY_LOCK:
        _TURN_MEMORY.clear()


def _merge_context_fragments(*parts: str | None) -> str | None:
    rows = [str(item or "").strip() for item in parts if str(item or "").strip()]
    if not rows:
        return None
    return "\n\n".join(rows)


def _format_quoted_text_context(text: str) -> str | None:
    clean = str(text or "").strip()
    if not clean:
        return None
    if len(clean) > 480:
        clean = clean[:479].rstrip() + "…"
    return (
        "Quoted Message Summary\n"
        "Treat the quoted text below only as local turn context. Quoted images or attachments are not automatically resent unless this is an explicit fresh image-analysis turn.\n"
        f"Quoted: {clean}"
    )


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
    cleaned = re.sub(
        r"</?(?:search|wiki|sub_agent|page|context_keep|plan_update|context_update|context_delete|tool_results|result|article_skeleton|search_rewrite|section|title|term)\b[^>]*>",
        "",
        cleaned,
    )
    return cleaned.strip()


def _format_tool_argument(name: str, arguments: Any) -> str:
    return format_tool_view_argument(name, _as_dict(arguments), max_items=12, max_chars=160)


def _format_tool_trace_line(name: str, arguments: Any) -> str:
    payload = _as_dict(arguments)
    line = f"> {format_tool_view_text(name, payload, max_chars=200)}"

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
        extras.append(f"{jina_tokens}tok")
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
    return f"> {format_tool_view_text(name, args, max_chars=200)}"


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
    del config
    return format_tool_view_text(str(name or "").strip(), _as_dict(args), max_chars=200)


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


def _compose_round_notice(
    thinking: str,
    tools: list[tuple[str, dict[str, Any]]] | None,
    *,
    config: dict[str, Any],
) -> str:
    parts: list[str] = []
    clean_thinking = _clean_answer(thinking)
    if clean_thinking:
        parts.append(clean_thinking)
    tool_block = _planned_sub_agent_block(tools, config=config) if tools else ""
    if tool_block:
        parts.append(tool_block)
    return "\n\n".join(part for part in parts if part).strip()


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
alc_stop = Alconna(conf.stop_command)


@command.on(alc_stop)
async def handle_stop(session: Session[MessageCreatedEvent]):
    scope_key = _session_request_scope(session)
    quote_id = session.event.message.id if conf.quote else None
    notice = "Stopped your most recent request." if _stop_latest_active_request(scope_key) else "No active recent request."
    await session.send(_build_notice_chain(notice, quote_id))


@command.on(alc_q)
async def handle_question(session: Session[MessageCreatedEvent], result: Arparma):
    content = result.all_matched_args.get("content")
    current_mc = MessageChain(content) if content else MessageChain()
    reply_mc = None
    if session.reply:
        try:
            reply_mc = session.reply.origin.message
        except Exception:
            logger.debug("Reply message read skipped.")

    user_input = _message_chain_text(current_mc)
    quoted_text = _message_chain_text(reply_mc)

    quote_id = session.event.message.id if conf.quote else None
    stop_event = threading.Event()
    current_task = asyncio.current_task()
    if current_task is None:
        logger.warning("Failed to register entari request: current task missing.")
        return
    scope_key = _session_request_scope(session)
    request_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{id(current_task)}"
    _register_active_request(
        scope_key,
        _ActiveRequest(
            request_id=request_id,
            task=current_task,
            stop_event=stop_event,
        ),
    )

    def _build_round_chain(text: str) -> MessageChain:
        chain = MessageChain(Text(text))
        if quote_id:
            chain = MessageChain(Quote(quote_id)) + chain
        return chain

    async def _send_round_notice(text: str) -> None:
        if stop_event.is_set():
            return
        await session.send(_build_round_chain(text))

    try:
        if not conf.verbose:
            try:
                await _send_round_notice("Thinking...")
            except Exception as exc:
                logger.warning("Thinking notice send failed in entari plugin: {}", exc)

        loop = asyncio.get_running_loop()
        sent_round_messages: set[str] = set()
        sent_round_lock = threading.Lock()
        config = load_config()
        stats = Stats()
        turn_memory = _get_turn_memory(scope_key)
        current_has_images = bool(current_mc.get(Image))
        quoted_has_images = bool(reply_mc and reply_mc.get(Image))
        use_quoted_images = bool(not current_has_images and quoted_has_images and not turn_memory)
        image_chain = reply_mc if use_quoted_images and reply_mc is not None else current_mc
        images, _ = await process_images(image_chain, None)

        question = user_input or (
            "请根据图片内容进行分析并回答。"
            if images
            else ""
        )
        if not question and not images:
            return

        memory_context = build_compact_context(turn_memory, current_user_text=question or quoted_text)
        quoted_context = None
        if quoted_text and (not memory_context or quoted_text not in memory_context):
            quoted_context = _format_quoted_text_context(quoted_text)
        context = _merge_context_fragments(memory_context, quoted_context)

        def _on_rewind(thinking: str, tools: list[tuple[str, dict[str, Any]]] | None = None) -> None:
            if not conf.verbose or stop_event.is_set():
                return
            clean = _compose_round_notice(thinking, tools, config=config)
            if not clean:
                return
            with sent_round_lock:
                if clean in sent_round_messages:
                    return
                sent_round_messages.add(clean)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _send_round_notice(clean),
                    loop,
                )
                future.result()
            except Exception as exc:
                logger.warning("Round message send failed in entari plugin: {}", exc)

        runner = run_stream if config.get("stream") not in (False, "false", "0", 0) else run
        kwargs: dict[str, Any] = {
            "config": config,
            "stats": stats,
            "images": images,
            "context": context,
            "stop_checker": stop_event.is_set,
        }
        if runner is run_stream and conf.verbose:
            kwargs["on_rewind"] = _on_rewind
        final_content = await asyncio.to_thread(
            runner,
            question,
            **kwargs,
        )

        if stop_event.is_set():
            return
        answer_text = str(final_content or "").strip()
        if not answer_text:
            return
        _append_turn_memory(scope_key, question, answer_text)

        primary_chain, fallback_chain, rendered_ok = await _build_answer_chain(
            answer_text,
            config=config,
            quote_id=quote_id,
        )
        try:
            if stop_event.is_set():
                return
            await session.send(primary_chain)
        except Exception as exc:
            if rendered_ok:
                logger.warning("Rendered message send failed in entari plugin: {}", exc)
                if not stop_event.is_set():
                    await session.send(fallback_chain)
            else:
                raise

    except asyncio.CancelledError:
        stop_event.set()
        return
    except StopRequestedError:
        stop_event.set()
        return
    except Exception as exc:
        logger.exception("Error in hyw execution: {}", exc)
        return
    finally:
        _unregister_active_request(scope_key, request_id)


# ── 生命周期 ──────────────────────────────────────────────────

@listen(Startup)
async def on_startup():
    logger.info("HYW plugin startup: loading configuration...")
    _clear_runtime_state()

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
    _clear_runtime_state()
    shutdown_tools()


__plugin__ = metadata(
    "hyw",
    author=[{"name": "kumo", "email": "dev@example.com"}],
    version=__version__,
    config=PluginConfig,
)
