"""
hyw/main.py - 极简 LLM 对话循环 + LiteLLM 原生工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web runtime (自带)
配置: ~/.hyw/config.yml, 兼容单模型与多模型写法

工具调用方式: 使用 LiteLLM 原生 tools/function calling.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from importlib import import_module
import json
import logging
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .codex_transport import (
    message_payload as codex_transport_message_payload,
    model_response as codex_transport_model_response,
    should_use_codex_mirror_transport,
    stream_response as codex_transport_stream_response,
)
from .config import (
    CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    LOG_DIR,
    build_model_config,
    cfg_get,
    load_config,
    resolve_tool_handlers,
)
from .prompt import (
    ACTIVE_PAGE_STATE_PROMPT,
    BASE_SYSTEM_PROMPT,
    DEFAULT_NO_RESULTS_TEXT,
    DEFAULT_NO_TITLE_TEXT,
    DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT,
    DEFAULT_PAGE_NO_MATCHING_TEXT,
    DUPLICATE_QUERY_SKIPPED_TEXT,
    FIRST_SEARCH_PROMPT,
    LATE_ROUND_FINAL_REPLY_PROMPT,
    LATEST_RAW_EMPTY_TEXT,
    LATEST_RAW_HEADING,
    PAGE_MARKDOWN_CACHE_HIT_TEXT,
    PAGE_MARKDOWN_CACHE_MISS_TEXT,
    PAGE_MARKDOWN_CACHE_PREFIX,
    PAGE_MARKDOWN_EMPTY_TITLE,
    PAGE_MARKDOWN_SCOPE_PREFIX,
    PAGE_MARKDOWN_MATCHED_LINES_TEXT,
    PAGE_MARKDOWN_TITLE_PREFIX,
    POST_SEARCH_PROMPT,
    SEARCH_RESULT_REMINDER,
    litellm_tool_config_for_phase,
)
from .search_document import (
    build_search_document_markdown,
    build_search_open_lines,
    build_search_result_lines,
    coerce_search_filters as _coerce_search_filters,
    format_search_filters as _format_search_filters,
    format_search_request_label as _format_search_request_label,
    normalize_search_request_key,
)


class _LazyLiteLLM:
    def __init__(self) -> None:
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_overrides", {})

    def _load(self):
        module = object.__getattribute__(self, "_module")
        if module is not None:
            return module

        lock = object.__getattribute__(self, "_lock")
        with lock:
            module = object.__getattribute__(self, "_module")
            if module is not None:
                return module

            module = import_module("litellm")
            module.suppress_debug_info = True
            for logger_name in ("LiteLLM", "litellm", "litellm.utils", "httpx", "httpcore"):
                logging.getLogger(logger_name).setLevel(logging.ERROR)

            overrides = object.__getattribute__(self, "_overrides")
            for attr_name, value in overrides.items():
                setattr(module, attr_name, value)

            object.__setattr__(self, "_module", module)
            return module

    def load(self):
        return self._load()

    def is_loaded(self) -> bool:
        return object.__getattribute__(self, "_module") is not None

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        overrides = object.__getattribute__(self, "_overrides")
        overrides[name] = value

        module = object.__getattribute__(self, "_module")
        if module is not None:
            setattr(module, name, value)


litellm = _LazyLiteLLM()


def _strip_model_routing_suffix(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    head, sep, tail = text.rpartition("/")
    if not sep:
        return text.partition(":")[0].strip()
    clean_tail = tail.partition(":")[0].strip()
    if not clean_tail:
        return head
    return f"{head}/{clean_tail}"


def _strip_openrouter_model_prefix(value: str) -> str:
    text = _strip_model_routing_suffix(value)
    if not text.startswith("openrouter/"):
        return text
    stripped = text[len("openrouter/"):].strip()
    return stripped or text


STATUS_PREPARING = "Preparing..."
STATUS_THINKING = "Thinking..."
STATUS_SEARCHING = "Searching..."
STATUS_READY = "\u2713"
_PREWARM_LOCK = threading.Lock()
_PREWARM_THREADS: dict[str, threading.Thread] = {}
_PREWARM_DONE: set[str] = set()


class StopRequestedError(RuntimeError):
    pass


def _raise_if_stop_requested(stop_checker: Any | None = None) -> None:
    if not callable(stop_checker):
        return
    try:
        should_stop = bool(stop_checker())
    except Exception:
        should_stop = False
    if should_stop:
        raise StopRequestedError("run aborted by caller")


def _get_litellm(*, on_status: Any | None = None):
    if callable(on_status) and not litellm.is_loaded():
        try:
            on_status(STATUS_PREPARING)
        except Exception:
            pass
    return litellm.load()


def _prewarm_runtime(config: dict[str, Any] | None = None) -> None:
    cfg = build_model_config(config or load_config())
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    litellm_mod = _get_litellm()

    token_counter = getattr(litellm_mod, "token_counter", None)
    if callable(token_counter):
        try:
            token_counter(model=model, messages=[{"role": "user", "content": "ping"}])
        except Exception:
            pass

    get_model_info = getattr(litellm_mod, "get_model_info", None)
    if callable(get_model_info):
        try:
            get_model_info(model)
        except Exception:
            pass

    _try_cost(model, 1, 1)


def start_runtime_prewarm(config: dict[str, Any] | None = None) -> None:
    cfg = build_model_config(config or load_config())
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    if not model:
        return

    with _PREWARM_LOCK:
        if model in _PREWARM_DONE:
            return
        existing = _PREWARM_THREADS.get(model)
        if existing is not None and existing.is_alive():
            return

        def _worker() -> None:
            try:
                _prewarm_runtime(cfg)
            finally:
                with _PREWARM_LOCK:
                    _PREWARM_DONE.add(model)
                    _PREWARM_THREADS.pop(model, None)

        thread = threading.Thread(
            target=_worker,
            name=f"hyw-prewarm-{model}",
            daemon=True,
        )
        _PREWARM_THREADS[model] = thread
        thread.start()


def get_runtime_prewarm_label(config: dict[str, Any] | None = None) -> str:
    cfg = build_model_config(config or load_config())
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    if not model:
        return ""

    with _PREWARM_LOCK:
        thread = _PREWARM_THREADS.get(model)
        if thread is not None and thread.is_alive():
            return STATUS_PREPARING
        if model in _PREWARM_DONE:
            return STATUS_READY

    if litellm.is_loaded():
        return STATUS_READY
    return ""

# ── 统计 ──────────────────────────────────────────────────────

def _try_cost(model: str, pt: int, ct: int) -> float | None:
    """尝试多种模型名变体查询价格，返回总 cost 或 None."""
    litellm_mod = _get_litellm()
    variants: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        variants.append(text)

    def _strip_routing_suffix(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        head, sep, tail = text.rpartition("/")
        if not sep:
            return text.partition(":")[0].strip()
        clean_tail = tail.partition(":")[0].strip()
        if not clean_tail:
            return head
        return f"{head}/{clean_tail}"

    def _expand(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        _add(text)

        stripped = _strip_routing_suffix(text)
        if stripped and stripped != text:
            _add(stripped)

        for candidate in (text, stripped):
            if not candidate:
                continue
            if candidate.startswith("openrouter/google/"):
                rest = candidate[len("openrouter/google/"):].strip()
                _add("gemini/" + rest)
                _add(rest)
                continue
            if candidate.startswith("google/"):
                rest = candidate[len("google/"):].strip()
                _add("gemini/" + rest)
                _add(rest)
                continue
            if candidate.startswith("openrouter/"):
                rest = candidate[len("openrouter/"):].strip()
                _add(rest)
                if "/" in rest:
                    _add(rest.split("/", 1)[-1])

    _expand(model)
    for v in variants:
        try:
            pc, cc = litellm_mod.cost_per_token(model=v, prompt_tokens=pt, completion_tokens=ct)
            return pc + cc
        except Exception:
            continue
    return None


@dataclass
class Stats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    jina_tokens: int = 0
    jina_calls: int = 0
    _has_cost: bool = False
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    def record(self, usage: dict[str, Any], cost: float | None):
        with self._lock:
            prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            completion_details = usage.get("completion_tokens_details")
            reasoning_tokens = 0
            if isinstance(completion_details, dict):
                reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.total_tokens += total_tokens if total_tokens > 0 else prompt_tokens + completion_tokens
            if cost is not None:
                self.cost_usd += cost
                self._has_cost = True

    def record_jina(self, *, tokens: int = 0, requests: int = 1):
        with self._lock:
            self.jina_tokens += max(0, int(tokens or 0))
            self.jina_calls += max(0, int(requests or 0))

    def summary(self) -> str:
        with self._lock:
            cost = f"${self.cost_usd:.6f}" if self._has_cost else "N/A"
            return (
                f"{self.calls}次 | "
                f"{self.prompt_tokens}+{self.completion_tokens}={self.total_tokens}tok | "
                f"think {self.reasoning_tokens}tok | "
                f"{cost} | "
                f"jina {self.jina_tokens}tok"
            )


def _tool_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _tool_payload_for_model(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def _tool_markdown_for_model(name: str, args: dict[str, Any], payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "工具返回了无效结果。"

    direct = str(payload.get("_model_markdown") or "").strip()
    if direct:
        return direct

    if name == "navigate":
        title = str(payload.get("title") or "").strip()
        url = str(payload.get("url") or args.get("url") or "").strip()
        if not payload.get("ok", True):
            error = str(payload.get("error") or "navigate failed").strip()
            lines = [f"# Navigate: {url}" if url else "# Navigate", "", error]
            return "\n".join(lines).strip()

        lines = []
        if title:
            lines.append(f"# Navigate: {title}")
        elif url:
            lines.append(f"# Navigate: {url}")
        if url:
            if lines:
                lines.append("")
            lines.append(f"Source: {url}")
        return "\n".join(lines).strip() or "Navigated."

    public = _tool_payload_for_model(payload)
    return json.dumps(public, ensure_ascii=False, indent=2)


def _record_tool_stats(stats: Stats | None, payload: dict[str, Any]) -> None:
    if stats is None or not isinstance(payload, dict):
        return
    meta = _tool_meta(payload)
    provider = str(meta.get("provider") or payload.get("provider") or "").strip()
    if provider != "jina_ai":
        return
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return
    stats.record_jina(
        tokens=int(usage.get("tokens") or 0),
        requests=int(usage.get("requests") or 1),
    )


# ── 调用日志 ──────────────────────────────────────────────────
_log_lock = threading.Lock()


def _safe_name(s: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|\r\n\t]+', '_', s.strip())
    return re.sub(r'\s+', '_', safe).strip('._')[:48] or 'call'


def _log_dir(config: dict[str, Any] | None = None) -> Path:
    d = Path(str((config or {}).get("log_dir") or "").strip() or str(LOG_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_log_id(question: str) -> str:
    """生成对话日志文件名: YYYYMMDD_HHMMSS_{safe_question}.md"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_safe_name(question)}.md"


def _log_title_from_messages(messages: list[dict[str, Any]] | None) -> str:
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = _format_log_message_content(message.get("content"))
        text = re.sub(r"\s+", " ", str(content or "").strip()).strip()
        if text:
            return text[:200]
    return "unknown"


def _log_turn_label(label: str, *, turn: int | None = None) -> str:
    if turn is not None:
        return str(max(1, int(turn)))
    match = re.search(r"\bround\s+(\d+)\b", str(label or "").strip(), re.IGNORECASE)
    if match is not None:
        return match.group(1)
    text = str(label or "").strip()
    return text or "unknown"


def _format_log_input_messages(messages: list[dict[str, Any]] | None) -> list[str]:
    lines: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "?").strip() or "?"
        content = _format_log_message_content(message.get("content"))
        lines.append(f"**[{role}]**")
        lines.append(f"```\n{content}\n```")
    if not lines:
        lines.append("无")
    return lines


def _format_log_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[str]:
    rows = [dict(item) for item in (tool_calls or []) if isinstance(item, dict)]
    if not rows:
        return ["无"]
    return [
        "```json",
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        "```",
    ]


def _extract_raw_tool_calls(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_tool_calls = payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    calls: list[dict[str, Any]] = []
    for raw_call in raw_tool_calls:
        call_payload = _to_dict(raw_call)
        if not isinstance(call_payload, dict):
            continue
        function_payload = call_payload.get("function")
        if function_payload is None and isinstance(call_payload.get("function_call"), dict):
            function_payload = call_payload.get("function_call")
        function_payload = _to_dict(function_payload) if function_payload is not None else {}
        if not isinstance(function_payload, dict):
            function_payload = {}

        call_id = str(call_payload.get("id") or "").strip()
        name = str(function_payload.get("name") or call_payload.get("name") or "").strip()
        raw_arguments = function_payload.get("arguments")
        if raw_arguments in (None, ""):
            raw_arguments = call_payload.get("arguments")
        arguments = _tool_arguments_to_dict(raw_arguments)

        row: dict[str, Any] = {}
        if call_id:
            row["id"] = call_id
        if name:
            row["name"] = name
        if isinstance(arguments, dict) and arguments:
            row["args"] = arguments
        elif isinstance(raw_arguments, str) and raw_arguments.strip():
            row["args_raw"] = raw_arguments.strip()
        if row:
            calls.append(row)
    return calls


def _append_log_lines(path: Path, lines: list[str]) -> None:
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _tool_payload_log_summary(
    *,
    name: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    duration_ms: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"tool": str(name or "").strip()}
    if duration_ms is not None:
        row["duration_ms"] = int(max(0.0, float(duration_ms)))

    if name == "navigate":
        search = str(args.get("search") or "").strip()
        ref = str(args.get("ref") or "").strip()
        url = str(args.get("url") or "").strip()
        keep = args.get("keep")
        if search:
            row["search"] = search
        if ref:
            row["ref"] = ref
        if url:
            row["url"] = url
        if isinstance(keep, str) and keep.strip():
            row["keep"] = keep.strip()
        elif isinstance(keep, list):
            rendered = [str(item or "").strip() for item in keep if str(item or "").strip()]
            if rendered:
                row["keep"] = rendered
        for key in ("ok", "provider", "title", "count", "error"):
            value = payload.get(key)
            if value not in (None, "", []):
                row[key] = value
        replaced_count = payload.get("_replaced_count")
        if replaced_count not in (None, "", []):
            row["replaced"] = replaced_count
        created_count = payload.get("_created_count")
        if created_count not in (None, "", []):
            row["kept"] = created_count
        return row

    return row


def log_model_call(
    *,
    label: str,
    model: str,
    messages: list[dict],
    output: str,
    tool_calls: list[dict[str, Any]] | None = None,
    turn: int | None = None,
    title: str | None = None,
    usage: dict[str, Any] | None = None,
    cost: float | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    config: dict[str, Any] | None = None,
    log_id: str | None = None,
):
    """每次 LLM 调用后追加写入日志文件 (每个对话一份)."""
    try:
        d = _log_dir(config)
        fname = log_id or _make_log_id("unknown")
        path = d / fname

        ts = datetime.now().strftime("%H:%M:%S")
        u = usage or {}
        log_title = re.sub(r"\s+", " ", str(title or "").strip()) or _log_title_from_messages(messages)
        turn_label = _log_turn_label(label, turn=turn)
        lines: list[str] = []

        with _log_lock:
            need_header = (not path.exists()) or path.stat().st_size <= 0
            if need_header:
                lines.extend([f"# {log_title}", ""])

            lines.extend(
                [
                    f"## Turn: {turn_label}",
                    "",
                    "### 模型Input",
                    *_format_log_input_messages(messages),
                    "",
                    "### 模型Output",
                ]
            )
            if output:
                lines.append(f"```\n{output}\n```")
            else:
                lines.append("无")
            lines.extend(
                [
                    "",
                    "### 模型工具Output",
                    *_format_log_tool_calls(tool_calls),
                    "",
                    "### 模型时间、开销",
                    f"- time: {ts}",
                    f"- model: `{model}`",
                ]
            )
            if duration_ms is not None:
                lines.append(f"- duration: {duration_ms:.0f}ms")
            if u:
                lines.append(
                    f"- tokens: prompt={u.get('prompt_tokens',0)} completion={u.get('completion_tokens',0)} total={u.get('total_tokens',0)}"
                )
            if cost is not None:
                lines.append(f"- cost: ${cost:.6f}")
            if error:
                lines.append(f"- error: {error}")
            lines.extend(["", "---", ""])

            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
    except Exception:
        pass  # 日志不应影响主流程


def log_tool_call(
    *,
    name: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    duration_ms: float | None = None,
    turn: int | None = None,
    config: dict[str, Any] | None = None,
    log_id: str | None = None,
):
    if str(name or "").strip() not in {"navigate"}:
        return
    try:
        d = _log_dir(config)
        fname = log_id or _make_log_id("unknown")
        path = d / fname
        summary = _tool_payload_log_summary(
            name=str(name or "").strip(),
            args=dict(args or {}),
            payload=dict(payload or {}),
            duration_ms=duration_ms,
        )
        lines = ["", "### 工具执行摘要"]
        if turn is not None:
            lines.append(f"- turn: {int(turn)}")
        lines.extend(
            [
                "```json",
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ]
        )
        with _log_lock:
            _append_log_lines(path, lines)
    except Exception:
        pass


def log_tool_selection(
    *,
    round_i: int,
    calls: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    log_id: str | None = None,
):
    del round_i, calls, config, log_id
    return


_DIRECT_SEARCH_TIMEOUT_S = 12.0
_DIRECT_PAGE_TIMEOUT_S = 60.0
_EXTERNAL_TOOL_TIMEOUT_S = 45.0


def _run_async(coro):
    """在同步上下文中运行 async 函数."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result, error = {}, {}
    def _go():
        try:
            result["v"] = asyncio.run(coro)
        except BaseException as e:
            error["v"] = e
    t = threading.Thread(target=_go, daemon=True)
    t.start(); t.join()
    if "v" in error:
        raise error["v"]
    return result.get("v")


def startup_tools(headless: bool = True, config: dict[str, Any] | None = None):
    """Warm up the built-in retrieval runtime."""
    from .web_runtime import on_startup

    return on_startup(headless=headless, config=config)


def shutdown_tools(config: dict[str, Any] | None = None):
    del config
    from .web_runtime import on_shutdown

    on_shutdown()


def _search_backend(
    query: str,
    *,
    kl: str = "",
    df: str = "",
    t: str = "",
    ia: str = "",
    config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    **_,
) -> dict[str, Any]:
    """调用 WebToolSuite 执行搜索."""
    try:
        return _run_async(
            asyncio.wait_for(
                _search_backend_async(
                    query=query,
                    kl=kl,
                    df=df,
                    t=t,
                    ia=ia,
                    config=config,
                    progress_callback=progress_callback,
                ),
                timeout=_DIRECT_SEARCH_TIMEOUT_S,
            )
        )
    except asyncio.TimeoutError:
        timeout_text = (
            str(int(_DIRECT_SEARCH_TIMEOUT_S))
            if float(_DIRECT_SEARCH_TIMEOUT_S).is_integer()
            else f"{_DIRECT_SEARCH_TIMEOUT_S:.1f}"
        )
        return {"ok": False, "error": f"search timed out after {timeout_text}s", "query": query, "results": []}


async def _search_backend_async(
    query: str,
    *,
    kl: str = "",
    df: str = "",
    t: str = "",
    ia: str = "",
    config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    **_,
) -> dict[str, Any]:
    """异步调用 WebToolSuite 执行搜索."""
    from .web_runtime import search_web

    cfg = build_model_config(config or load_config())
    try:
        payload = await search_web(
            query=query, mode="text",
            kl=kl,
            df=df,
            t=t,
            ia=ia,
            max_results=5,
            config=cfg,
            headless=bool(cfg.get("headless") not in (False, "false", "0", 0)),
            progress_callback=progress_callback,
        )
    except Exception as e:
        return {"ok": False, "error": f"search provider failed: {str(e).strip() or type(e).__name__}", "query": query, "results": []}

    if not isinstance(payload, dict):
        return {"ok": False, "error": "search provider failed: invalid payload type", "query": query, "results": []}
    if not bool(payload.get("ok", True)):
        provider = str(payload.get("provider") or "").strip()
        raw_error = str(payload.get("error") or "").strip() or "unknown provider error"
        error_text = raw_error if not provider or raw_error.startswith(f"{provider}:") else f"{provider}: {raw_error}"
        result = {"ok": False, "error": error_text, "query": query, "results": []}
        if isinstance(payload.get("_meta"), dict):
            result["_meta"] = dict(payload.get("_meta") or {})
        return result

    rows = payload.get("results", []) if isinstance(payload, dict) else []
    results = [
        {"title": str(r.get("title", ""))[:200], "url": str(r.get("url", ""))[:400], "snippet": str(r.get("snippet") or r.get("intro") or "")[:240]}
        for r in rows if isinstance(r, dict)
    ]
    result = {"ok": True, "query": query, "count": len(results), "results": results}
    if isinstance(payload, dict) and isinstance(payload.get("_meta"), dict):
        result["_meta"] = dict(payload.get("_meta") or {})
    return result


def _page_extract(
    url: str,
    *,
    config: dict[str, Any] | None = None,
    max_chars: int = 8000,
    progress_callback: Any | None = None,
    **_,
) -> dict[str, Any]:
    from .web_runtime import page_extract

    cfg = build_model_config(config or load_config())
    try:
        payload = _run_async(
            asyncio.wait_for(
                page_extract(
                    url=url,
                    max_chars=max_chars,
                    config=cfg,
                    headless=cfg.get("headless") not in (False, "false", "0", 0),
                    progress_callback=progress_callback,
                ),
                timeout=_DIRECT_PAGE_TIMEOUT_S,
            )
        )
    except asyncio.TimeoutError:
        timeout_text = (
            str(int(_DIRECT_PAGE_TIMEOUT_S))
            if float(_DIRECT_PAGE_TIMEOUT_S).is_integer()
            else f"{_DIRECT_PAGE_TIMEOUT_S:.1f}"
        )
        return {"ok": False, "error": f"page extract timed out after {timeout_text}s", "url": url}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid extract payload", "url": url}

    return {
        "ok": bool(payload.get("ok")),
        "error": str(payload.get("error") or "").strip(),
        "provider": str(payload.get("provider") or "").strip(),
        "title": str(payload.get("title") or "")[:200],
        "url": str(payload.get("url") or url)[:400],
        "content": str(payload.get("content") or "")[: max(1, int(max_chars))],
        "_meta": dict(payload.get("_meta") or {}) if isinstance(payload.get("_meta"), dict) else {},
    }


def _public_search_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        public_rows.append(
            {
                "title": str(item.get("title") or "").strip() or DEFAULT_NO_TITLE_TEXT,
                "url": str(item.get("url") or "").strip(),
                "snippet": str(item.get("snippet") or "").strip(),
                "provider": str(item.get("provider") or "").strip(),
                "domain": str(item.get("domain") or "").strip(),
                "matched_queries": [
                    str(query or "").strip()
                    for query in item.get("matched_queries", [])
                    if str(query or "").strip()
                ],
            }
        )
    return public_rows


def _provider_for_capability(capability: str, config: dict[str, Any] | None = None) -> str:
    try:
        handlers = resolve_tool_handlers(config, capability)
    except Exception:
        return ""
    if not handlers:
        return ""
    return str(handlers[0].provider or "").strip()


def execute_tool_payload(
    name: str,
    args: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    user_question: str = "",
    log_id: str | None = None,
    runtime_state: _SessionRuntimeState | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    cfg = build_model_config(config or load_config())
    if name == "navigate":
        r = _run_navigate(
            url=args.get("url", ""),
            ref=args.get("ref", ""),
            search=args.get("search", ""),
            search_filters=args,
            cfg=cfg,
            runtime_state=runtime_state,
            progress_callback=progress_callback,
        )
    else:
        r = {"ok": False, "error": f"unknown tool: {name}"}
    return r


def execute_tool(name: str, args: dict[str, Any]) -> str:
    r = execute_tool_payload(name, args)
    r = _tool_payload_for_model(r if isinstance(r, dict) else {"ok": False, "error": "invalid tool payload"})
    return json.dumps(r, ensure_ascii=False, indent=2)


def _tool_capability(name: str) -> str:
    if name == "navigate":
        return "page_extract"
    return ""


def _tool_display_name(name: str, *, provider: str = "") -> str:
    del provider
    return {
        "navigate": "Navigate",
    }.get(name, str(name or "").strip())


def _emit_tool_progress(callback: Any | None, name: str, args: dict[str, Any]) -> None:
    if not callable(callback):
        return
    try:
        callback(name, args)
    except Exception:
        pass


def _tool_provider_from_payload(payload: dict[str, Any]) -> str:
    meta = _tool_meta(payload)
    provider = str(meta.get("provider") or payload.get("provider") or "").strip()
    if provider:
        return provider
    rows = payload.get("results")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                provider = str(row.get("provider") or "").strip()
                if provider:
                    return provider
    return ""


def _tool_provider_from_config(name: str, config: dict[str, Any] | None = None) -> str:
    capability = _tool_capability(name)
    if not capability:
        return ""
    if capability not in {"page_extract", "render"}:
        return ""
    return _provider_for_capability(capability, config)


def _tool_callback_args(
    name: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    *,
    elapsed_s: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(args)
    provider = ""
    if isinstance(payload, dict):
        provider = _tool_provider_from_payload(payload)
        if "count" in payload:
            merged["_count"] = payload.get("count")
            merged["count"] = payload.get("count")
        if "ok" in payload:
            merged["_ok"] = payload.get("ok")
        if payload.get("from_cache"):
            merged["_from_cache"] = True
        if name == "navigate":
            for key in ("title", "url", "total_lines", "total_chars"):
                if payload.get(key) not in (None, ""):
                    merged[key] = payload.get(key)
        meta = _tool_meta(payload)
        usage = meta.get("usage")
        if isinstance(usage, dict) and usage.get("tokens") not in (None, ""):
            merged["_jina_tokens"] = usage.get("tokens")
            if name == "navigate":
                merged["_page_usage_tokens"] = usage.get("tokens")
                merged["_page_usage_requests"] = usage.get("requests")
        billing = meta.get("billing")
        if name == "navigate" and isinstance(billing, dict):
            merged["_page_billing_mode"] = billing.get("mode")
            merged["_page_cost_usd"] = billing.get("cost_usd")
    if not provider:
        if name == "navigate" and str(args.get("search") or "").strip():
            provider = _provider_for_capability("search", config)
        else:
            provider = _tool_provider_from_config(name, config)
    if provider:
        merged["_provider"] = provider
    merged["_display_name"] = _tool_display_name(name, provider=provider)
    if elapsed_s is not None:
        merged["_elapsed_s"] = max(0.0, float(elapsed_s))
    return merged


def _normalize_native_tool_name(name: str) -> str:
    normalized = str(name or "").strip()
    aliases = {
        "page": "navigate",
        "page_extract": "navigate",
        "read": "navigate",
        "navigate_page": "navigate",
    }
    return aliases.get(normalized, normalized)


def _tool_arguments_to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _coerce_page_extract_search_arg(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return []

    raw_search = payload.get("search")
    if not isinstance(raw_search, list):
        return []
    return _split_page_extract_search_terms(raw_search)


def _coerce_page_extract_mode(payload: dict[str, Any]) -> str:
    raw_mode = str((payload or {}).get("mode") or "").strip().lower()
    if raw_mode in {"sample", "range"}:
        return raw_mode
    start_line = _coerce_page_extract_line_number((payload or {}).get("start_line"))
    end_line = _coerce_page_extract_line_number((payload or {}).get("end_line"))
    if start_line or end_line:
        return "range"
    return "sample"


def _coerce_page_extract_line_number(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 0
    return max(0, parsed)


def _normalize_keep_argument(value: Any) -> str | list[str] | None:
    if value is None:
        return None

    raw_parts: list[str] = []
    input_is_list = isinstance(value, list)
    if isinstance(value, str):
        raw_parts.extend(segment.strip() for segment in value.split(","))
    elif isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            raw_parts.extend(segment.strip() for segment in text.split(","))
    else:
        return None

    normalized_parts: list[str] = []
    saw_any = False
    for part in raw_parts:
        if not part:
            continue
        saw_any = True
        match = _LINE_REF_RE.fullmatch(part)
        if match is None:
            return None
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            start, end = end, start
        normalized_parts.append(f"L{start}" if start == end else f"L{start}-L{end}")

    if not saw_any or not normalized_parts:
        return None
    if input_is_list or len(normalized_parts) > 1:
        return normalized_parts
    return normalized_parts[0]


def _sanitize_native_tool_call(name: str, args: dict[str, Any], *, call_id: str = "") -> dict[str, Any] | None:
    normalized_name = _normalize_native_tool_name(name)
    payload = dict(args or {})
    if normalized_name == "navigate":
        url = str(payload.get("url") or payload.get("target") or "").strip()
        ref = _normalize_context_ref(str(payload.get("ref") or "").strip())
        search = str(payload.get("search") or payload.get("query") or "").strip()
        keep = payload.get("keep")
        normalized_keep = _normalize_keep_argument(keep)
        if not url and not ref and not search:
            return None
        if keep is None or normalized_keep is None:
            return None
        clean_args = {}
        if url:
            clean_args["url"] = url
        if ref:
            clean_args["ref"] = ref
        if search:
            clean_args["search"] = search
            clean_args.update(_coerce_search_filters(payload))
        if normalized_keep is not None:
            clean_args["keep"] = normalized_keep
    else:
        return None

    if call_id:
        clean_args["_call_id"] = call_id
    return {"name": normalized_name, "args": clean_args}


def _invalid_tool_call_reason(name: str, args: dict[str, Any]) -> str:
    normalized_name = _normalize_native_tool_name(name)
    payload = dict(args or {})
    if normalized_name == "navigate":
        url = str(payload.get("url") or payload.get("target") or "").strip()
        ref = _normalize_context_ref(str(payload.get("ref") or "").strip())
        search = str(payload.get("search") or payload.get("query") or "").strip()
        keep = payload.get("keep")
        normalized_keep = _normalize_keep_argument(keep)
        if not url and not ref and not search:
            return "navigate requires `url`, `ref`, or `search`."
        if keep is None:
            return "navigate requires `keep`; use `L0` when nothing should be preserved."
        if normalized_keep is None:
            return "navigate `keep` must use exact visible line refs such as `L0`, `L12`, `L12-L23`, or a list of them."
        return "navigate arguments are invalid."
    if normalized_name == "open":
        return "open is no longer available. Use `navigate(..., keep=[...])` instead."
    if normalized_name == "close":
        return "close is no longer available. Use `navigate(..., keep=[...])` instead."
    return f"unknown tool `{str(name or '').strip() or '<empty>'}`."


def _format_invalid_tool_call_message(errors: list[str]) -> str:
    clean = [str(item or "").strip() for item in errors if str(item or "").strip()]
    if not clean:
        return ""
    rows = ["Tool call rejected:"]
    rows.extend(f"- {item}" for item in clean[:4])
    return "\n".join(rows).strip()


def _parse_native_tool_calls_detailed(payload: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(payload, dict):
        return [], []
    raw_tool_calls = payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return [], []

    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw_call in raw_tool_calls:
        call_payload = _to_dict(raw_call)
        if not isinstance(call_payload, dict):
            continue
        call_id = str(call_payload.get("id") or "").strip()
        function_payload = call_payload.get("function")
        if function_payload is None and isinstance(call_payload.get("function_call"), dict):
            function_payload = call_payload.get("function_call")
        function_payload = _to_dict(function_payload) if function_payload is not None else {}
        if not isinstance(function_payload, dict):
            function_payload = {}
        name = str(function_payload.get("name") or call_payload.get("name") or "").strip()
        arguments = _tool_arguments_to_dict(function_payload.get("arguments") or call_payload.get("arguments"))
        call = _sanitize_native_tool_call(name, arguments, call_id=call_id)
        if not isinstance(call, dict):
            reason = _invalid_tool_call_reason(name, arguments)
            if reason:
                errors.append(f"`{str(name or '').strip() or '<empty>'}`: {reason}")
            continue
        signature = (
            str(call.get("name") or "").strip(),
            tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in (call.get("args") or {}).items()
                    if str(key) != "_call_id"
                )
            ),
        )
        if signature in seen:
            continue
        seen.add(signature)
        calls.append(call)
    return calls, errors


def _parse_native_tool_calls(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    calls, _ = _parse_native_tool_calls_detailed(payload)
    return calls


def _merge_stream_tool_call_deltas(
    state: dict[int, dict[str, Any]],
    delta_payload: dict[str, Any],
) -> None:
    raw_tool_calls = delta_payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return
    for fallback_index, raw_call in enumerate(raw_tool_calls):
        call_payload = _to_dict(raw_call)
        if not isinstance(call_payload, dict):
            continue
        raw_index = call_payload.get("index")
        try:
            index = int(raw_index)
        except Exception:
            index = fallback_index
            while index in state:
                index += 1
        current = state.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        call_id = str(call_payload.get("id") or "").strip()
        if call_id:
            current["id"] = call_id
        function_payload = _to_dict(call_payload.get("function")) if call_payload.get("function") is not None else {}
        if not isinstance(function_payload, dict):
            function_payload = {}
        name_delta = str(function_payload.get("name") or "")
        if name_delta:
            current["function"]["name"] = str(current["function"].get("name") or "") + name_delta
        arguments_delta = function_payload.get("arguments")
        if arguments_delta is not None:
            current["function"]["arguments"] = str(current["function"].get("arguments") or "") + str(arguments_delta)


def _finalize_stream_tool_calls(state: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if not state:
        return []
    ordered = [state[index] for index in sorted(state)]
    return _parse_native_tool_calls({"tool_calls": ordered})


def _normalize_context_ref(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"\d+(?::\d+)?", text):
        return text
    return ""


def _parse_context_ref(value: str) -> tuple[int, int | None]:
    text = _normalize_context_ref(value)
    if not text:
        return 0, None
    if ":" not in text:
        try:
            return int(text), None
        except Exception:
            return 0, None
    head, _, tail = text.partition(":")
    try:
        return int(head), int(tail)
    except Exception:
        return 0, None


def _resolve_page_call_refs(calls: list[dict[str, Any]], phase_state: "_PhaseRuntimeState") -> None:
    if not calls:
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("name") or "").strip() not in {"navigate"}:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else None
        if not isinstance(args, dict):
            continue
        if str(args.get("url") or "").strip():
            continue
        ref = _normalize_context_ref(str(args.get("ref") or "").strip())
        if not ref:
            continue
        item_id, link_index = _parse_context_ref(ref)
        if item_id < 0:
            continue
        item = _find_context_item(phase_state, item_id)
        if item is None:
            continue
        resolved_url = ""
        if link_index is not None:
            link_rows = _context_item_link_rows(item)
            for row in link_rows:
                if not isinstance(row, dict):
                    continue
                if int(row.get("index") or 0) != int(link_index):
                    continue
                resolved_url = _normalize_state_url(str(row.get("url") or "").strip()) or str(row.get("url") or "").strip()
                break
        if not resolved_url:
            resolved_url = _normalize_state_url(str(item.data.get("url") or "").strip()) or str(item.data.get("url") or "").strip()
        if resolved_url:
            args["url"] = resolved_url
_TOOL_LIMIT = 8


def _clean_model_text(text: str) -> str:
    return str(text or "").strip()


def _normalize_tool_turn_message(text: str, calls: list[dict[str, Any]]) -> str:
    del calls
    return _clean_model_text(text)

def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
                continue
            if item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "")


def _append_context_messages(msgs: list[dict[str, Any]], context: Any) -> None:
    if not context:
        return
    if isinstance(context, str):
        text = context.strip()
        if text:
            msgs.append({"role": "assistant", "content": text})
        return
    if not isinstance(context, list):
        return
    for item in context:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        text = _message_content_text(content)
        if not text:
            continue
        msgs.append({"role": role, "content": content})


def _context_has_prior_turns_summary(context: Any) -> bool:
    marker = "Previous Turns Summary"
    if not context:
        return False
    if isinstance(context, str):
        return marker in context
    if not isinstance(context, list):
        return False
    for item in context:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if marker in _message_content_text(content):
            return True
    return False


def build_compact_context(
    memory: list[dict[str, Any]] | None,
    *,
    current_user_text: str = "",
    max_pairs: int = 3,
) -> str | None:
    rows = [dict(item) for item in (memory or []) if isinstance(item, dict)]
    if not rows:
        return None

    def _trim(text: str, limit: int) -> str:
        raw = _clean_model_text(text)
        cap = max(1, int(limit))
        if len(raw) <= cap:
            return raw
        return raw[: cap - 1].rstrip() + "…"

    def _looks_like_choice_reply(text: str) -> bool:
        raw = _clean_model_text(text)
        if not raw:
            return False
        compact = re.sub(r"\s+", "", raw).casefold()
        if compact in {
            "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f",
            "前者", "后者", "就这个",
            "第1个", "第2个", "第3个", "第4个", "第5个",
            "第一", "第二", "第三", "第四", "第五",
            "第一个", "第二个", "第三个", "第四个", "第五个",
        }:
            return True
        return re.fullmatch(r"(?:第)?\d+(?:个|项|步)?", compact) is not None

    def _looks_like_option_menu(text: str) -> bool:
        raw = _clean_model_text(text)
        if not raw:
            return False
        patterns = (
            r"(?m)^\s*[1-9][\.\)．、]\s+\S+",
            r"(?m)^\s*[A-Fa-f][\.\)]\s+\S+",
            r"(?m)^\s*\d+\s+\S+",
            r"(?m)^\s*选\s*[1-9A-Fa-f]",
            r"(?m)^\s*你回复.*(?:1|2|3|4|5|A|B|C)",
        )
        return any(re.search(pattern, raw) for pattern in patterns)

    pairs: list[tuple[str, str]] = []
    pending_user = ""
    for item in rows:
        role = str(item.get("role") or "").strip().lower()
        content = _clean_model_text(str(item.get("content") or "").strip())
        if not content:
            continue
        if role == "user":
            pending_user = content
            continue
        if role == "assistant":
            if pending_user:
                pairs.append((pending_user, content))
                pending_user = ""
            else:
                pairs.append(("", content))
    if not pairs:
        return None

    lines = [
        "Previous Turns Summary",
        "Use the summary below only as background. This turn is still a fresh tool/runtime session, so do not reuse prior search history, cached pages, or stale assumptions unless the user repeats them.",
    ]
    preserve_last_menu = _looks_like_choice_reply(current_user_text)
    last_index = len(pairs) - 1
    pair_limit = max(1, int(max_pairs))
    start_index = max(0, len(pairs) - pair_limit)
    for idx, (user_text, assistant_text) in enumerate(pairs[start_index:], start=start_index):
        if user_text:
            lines.append(f"User: {_trim(user_text, 160)}")
        if preserve_last_menu and idx == last_index and _looks_like_option_menu(assistant_text):
            lines.append("Assistant (full because current user reply looks like an option selection):")
            lines.append(assistant_text)
        else:
            lines.append(f"Assistant: {_trim(assistant_text, 220)}")
    return "\n".join(lines).strip()


def _group_page_matches(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    groups: list[tuple[int, int, list[tuple[int, str]]]] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        try:
            line_no = int(item.get("line") or 0)
        except Exception:
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "").strip())
        if not line_no or not text:
            continue
        if groups and line_no <= groups[-1][1] + 1:
            start, _, texts = groups[-1]
            groups[-1] = (start, line_no, [*texts, (line_no, text)])
        else:
            groups.append((line_no, line_no, [(line_no, text)]))
    rendered: list[tuple[str, str]] = []
    width = max(2, len(str(max((int(item.get("line") or 0) for item in matches if isinstance(item, dict)), default=1))))
    for start, end, texts in groups:
        line_span = f"{start}" if start == end else f"{start}-{end}"
        rendered.append((line_span, "\n".join(f"L{line_no:0{width}d} | {text}" for line_no, text in texts)))
    return rendered


_SAMPLE_ACTION_LINK_TEXTS = (
    "skip to content",
    "see faq",
    "see faqs",
    "see checklist",
    "see list of endorsements",
    "endorse the osaid",
    "add your name",
)


def _normalize_sample_preview_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _extract_markdown_link_texts(value: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"\[([^\]]+)\]\([^)]+\)", str(value or ""))]


_CONTEXT_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CONTEXT_ENUMERATED_LINK_PREFIX_RE = re.compile(r"^\s*(?:L\d+\s*\|\s*)?\[(\d+)\]\s+")


def _extract_context_link_rows(value: Any) -> list[dict[str, Any]]:
    text = str(value or "")
    if not text.strip():
        return []

    rows: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    next_index = 1

    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        prefix_match = _CONTEXT_ENUMERATED_LINK_PREFIX_RE.match(line)
        explicit_index = int(prefix_match.group(1)) if prefix_match is not None else 0
        link_matches = list(_CONTEXT_MARKDOWN_LINK_RE.finditer(line))
        if not link_matches:
            continue
        for match_no, match in enumerate(link_matches, start=1):
            label = re.sub(r"\s+", " ", str(match.group(1) or "").strip()) or "link"
            raw_url = str(match.group(2) or "").strip()
            normalized_url = _normalize_state_url(raw_url) or raw_url
            if not normalized_url:
                continue

            if match_no == 1 and explicit_index > 0 and explicit_index not in used_indexes:
                index = explicit_index
                if index >= next_index:
                    next_index = index + 1
            else:
                while next_index in used_indexes:
                    next_index += 1
                index = next_index
                next_index += 1

            used_indexes.add(index)
            rows.append({"index": index, "url": normalized_url, "text": label})
    return rows


def _context_item_link_rows(item: "_ContextItem | None") -> list[dict[str, Any]]:
    if item is None:
        return []

    existing = item.data.get("links")
    if isinstance(existing, list):
        normalized_rows: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for row in existing:
            if not isinstance(row, dict):
                continue
            try:
                index = int(row.get("index") or 0)
            except Exception:
                index = 0
            raw_url = str(row.get("url") or "").strip()
            normalized_url = _normalize_state_url(raw_url) or raw_url
            if index <= 0 or not normalized_url or index in seen_indexes:
                continue
            seen_indexes.add(index)
            normalized_rows.append(
                {
                    "index": index,
                    "url": normalized_url,
                    "text": str(row.get("text") or "").strip(),
                    "line": int(row.get("line") or 0),
                }
            )
        if normalized_rows:
            normalized_rows.sort(key=lambda row: (int(row.get("index") or 0), int(row.get("line") or 0)))
            return normalized_rows

    text = _context_value_text(item.data.get("text") or item.data.get("snippet"))
    return _extract_context_link_rows(text)


def _remember_context_item_ref_targets(
    runtime_state: "_SessionRuntimeState | None",
    item: "_ContextItem | None",
) -> None:
    if runtime_state is None or item is None:
        return

    remembered: dict[str, str] = {}
    item_url = _normalize_state_url(str(item.data.get("url") or "").strip()) or str(item.data.get("url") or "").strip()
    if item_url:
        remembered[str(item.item_id)] = item_url

    for row in _context_item_link_rows(item):
        try:
            index = int(row.get("index") or 0)
        except Exception:
            index = 0
        raw_url = str(row.get("url") or "").strip()
        normalized_url = _normalize_state_url(raw_url) or raw_url
        if index <= 0 or not normalized_url:
            continue
        remembered[f"{item.item_id}:{index}"] = normalized_url

    if not remembered:
        return
    with runtime_state.lock:
        runtime_state.ref_targets.update(remembered)


def _looks_like_sample_noise_line(text: str, *, line_no: int = 0, total_lines: int = 0) -> bool:
    clean = _normalize_sample_preview_text(text)
    if not clean:
        return True
    if re.fullmatch(r"(?:[\*\-=_#.`~]\s*){2,}", clean):
        return True
    if re.fullmatch(r"#+\s*\[\]\([^)]+\)", clean):
        return True

    markdown_links = _extract_markdown_link_texts(clean)
    link_only = bool(markdown_links) and re.fullmatch(
        r"(?:[*\-+]\s+|\d+\.\s+|#+\s+)?(?:\[[^\]]+\]\([^)]+\)\s*)+",
        clean,
    ) is not None
    if not link_only:
        return False

    plain = re.sub(r"(?:[*\-+]\s+|\d+\.\s+|#+\s+)", "", clean)
    plain = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", plain)
    plain = _normalize_sample_preview_text(plain)
    if plain:
        return False

    link_text = _normalize_sample_preview_text(" ".join(markdown_links)).casefold()
    if any(link_text.startswith(prefix) for prefix in _SAMPLE_ACTION_LINK_TEXTS):
        return True
    if total_lines > 0 and line_no >= max(1, int(total_lines * 0.7)):
        return True
    return len(link_text) <= 32


def _append_page_memory_items(
    call: dict[str, Any],
    payload: dict[str, Any],
    state: _PhaseRuntimeState,
    *,
    round_no: int,
    question_text: str,
) -> list[int]:
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    url = str(payload.get("url") or args.get("url") or "").strip()
    title = str(payload.get("title") or "").strip()
    question = str(payload.get("question") or question_text or "").strip()
    search = payload.get("search")
    if search in (None, ""):
        search = args.get("search")
    call_id = str(args.get("_call_id") or "").strip()
    matches = payload.get("_matched_lines") if isinstance(payload.get("_matched_lines"), list) else []
    mode = str(payload.get("mode") or args.get("mode") or "").strip().lower()
    if mode == "sample":
        total_lines = int(payload.get("total_lines") or 0)
        matches = [
            item
            for item in matches
            if isinstance(item, dict)
            and not _looks_like_sample_noise_line(
                str(item.get("text") or ""),
                line_no=int(item.get("line") or 0),
                total_lines=total_lines,
            )
        ]
    created_ids: list[int] = []
    grouped = _group_page_matches(matches)
    if not grouped:
        item = _add_context_item(
            state,
            "memory.page",
            {
                "title": title,
                "url": url,
                "text": str(payload.get("page_error") or payload.get("error") or DEFAULT_PAGE_NO_MATCHING_TEXT).strip() or DEFAULT_PAGE_NO_MATCHING_TEXT,
                "question": question,
                "search": search,
                "round": round_no,
                "call_id": call_id,
            },
            pending=True,
        )
        created_ids.append(item.item_id)
        return created_ids

    for source_lines, text in grouped:
        item = _add_context_item(
            state,
            "memory.page",
            {
                "title": title,
                "url": url,
                "question": question,
                "search": search,
                "window": source_lines,
                "text": text,
                "round": round_no,
                "call_id": call_id,
            },
            pending=True,
        )
        created_ids.append(item.item_id)
    return created_ids


def _set_active_page_item(
    state: _PhaseRuntimeState,
    payload: dict[str, Any],
    *,
    round_no: int,
    call_id: str = "",
) -> int:
    raw_lines = [str(line or "") for line in payload.get("_raw_lines", [])] if isinstance(payload.get("_raw_lines"), list) else []
    item = _add_context_item(
        state,
        "page.active",
        {
            "title": str(payload.get("title") or "").strip(),
            "url": str(payload.get("url") or "").strip(),
            "round": round_no,
            "call_id": call_id,
            "raw_lines": raw_lines,
            "from_cache": bool(payload.get("from_cache")),
        },
    )
    markdown, link_rows, total_lines, total_chars = _render_active_page_markdown(
        page_item_id=item.item_id,
        url=str(payload.get("url") or "").strip(),
        title=str(payload.get("title") or "").strip(),
        raw_lines=raw_lines,
        from_cache=bool(payload.get("from_cache")),
    )
    item.data["markdown"] = markdown
    item.data["links"] = link_rows
    item.data["count"] = int(payload.get("count") or 0)
    item.data["total_lines"] = total_lines
    item.data["total_chars"] = total_chars
    state.active_page_ids.append(int(item.item_id))
    return item.item_id


def _build_active_page_message(state: _PhaseRuntimeState) -> str:
    items = _active_page_items(state)
    if not items:
        return ""
    return "\n\n".join(
        str(item.data.get("markdown") or "").strip()
        for item in items
        if str(item.data.get("markdown") or "").strip()
    ).strip()


_LINE_REF_RE = re.compile(r"^L?(?P<start>\d+)(?:\s*-\s*L?(?P<end>\d+))?$", re.IGNORECASE)


def _parse_keep_specs(value: Any) -> list[tuple[int, int]]:
    parts: list[str] = []
    if isinstance(value, str):
        parts.extend(segment.strip() for segment in value.split(","))
    elif isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            parts.extend(segment.strip() for segment in text.split(","))

    ranges: list[tuple[int, int]] = []
    for part in parts:
        if not part:
            continue
        match = _LINE_REF_RE.fullmatch(part)
        if match is None:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def _keep_and_clear_active_page(
    *,
    args: dict[str, Any],
    state: _PhaseRuntimeState,
    round_no: int,
) -> tuple[dict[str, Any], list[int]]:
    ref = _normalize_context_ref(str(args.get("ref") or "").strip())
    item_id, link_index = _parse_context_ref(ref)
    item = _find_context_item(state, int(item_id)) if ref and item_id >= 0 and link_index is None else None
    if item is not None and str(item.item_type or "").strip() != "page.active":
        item = None
    if item is not None and int(item.item_id) not in {int(page_id) for page_id in state.active_page_ids}:
        item = None
    if item is None:
        payload = {
            "ok": False,
            "error": "no matching active page for keep",
            "count": 0,
            "_model_markdown": "# Keep\n\nNo matching active page for keep.",
        }
        return payload, []

    keep_ranges = _parse_keep_specs(args.get("keep"))
    raw_lines = [str(line or "") for line in item.data.get("raw_lines", [])] if isinstance(item.data.get("raw_lines"), list) else []
    source_url = str(item.data.get("url") or "").strip()
    title = str(item.data.get("title") or "").strip()
    created_ids: list[int] = []
    matches: list[dict[str, Any]] = []
    if keep_ranges and raw_lines:
        for start, end in keep_ranges:
            for line_no in range(max(1, start), min(len(raw_lines), end) + 1):
                text = str(raw_lines[line_no - 1] or "").rstrip()
                if not text.strip():
                    continue
                matches.append({"line": line_no, "text": text})
        grouped = _group_page_matches(matches)
        for source_lines, text in grouped:
            memory_item = _add_context_item(
                state,
                "memory.page",
                {
                    "title": title,
                    "url": source_url,
                    "window": source_lines,
                    "text": text,
                    "links": _extract_context_link_rows(text),
                    "round": round_no,
                    "call_id": str(args.get("_call_id") or "").strip(),
                },
            )
            created_ids.append(memory_item.item_id)
    state.items = [entry for entry in state.items if int(entry.item_id) != int(item.item_id)]
    state.active_page_ids = [page_id for page_id in state.active_page_ids if int(page_id) != int(item.item_id)]
    payload = {
        "ok": True,
        "ref": str(item.item_id),
        "url": source_url,
        "title": title,
        "count": len(created_ids),
        "_created_ids": created_ids,
        "_model_markdown": (
            f"# Keep [{item.item_id}]\n\n"
            + (f"Kept {len(created_ids)} compact evidence block(s)." if created_ids else "Cleared active page without keeping lines.")
        ),
    }
    return payload, created_ids


def _replace_active_pages(
    *,
    args: dict[str, Any],
    state: _PhaseRuntimeState,
    round_no: int,
) -> tuple[list[int], list[int]]:
    active_items = list(_active_page_items(state))
    if not active_items:
        return [], []

    replace_args: dict[str, Any] = {}
    normalized_keep = _normalize_keep_argument(args.get("keep"))
    if normalized_keep is not None:
        replace_args["keep"] = normalized_keep
    call_id = str(args.get("_call_id") or "").strip()
    if call_id:
        replace_args["_call_id"] = call_id

    replaced_ids: list[int] = []
    created_ids: list[int] = []
    for item in active_items:
        payload, new_ids = _keep_and_clear_active_page(
            args={**replace_args, "ref": str(item.item_id)},
            state=state,
            round_no=round_no,
        )
        if not payload.get("ok"):
            continue
        replaced_ids.append(int(item.item_id))
        created_ids.extend(int(item_id) for item_id in new_ids)
    return replaced_ids, created_ids


def _build_latest_raw_message(state: "_PhaseRuntimeState") -> str:
    return _build_context_items_message(
        heading=LATEST_RAW_HEADING,
        empty_text=LATEST_RAW_EMPTY_TEXT,
        items=_latest_raw_items(state),
    )


def _late_round_warning_message(*, round_i: int) -> str:
    if int(round_i) >= 15:
        return LATE_ROUND_FINAL_REPLY_PROMPT
    return ""


def _build_round_brief_message(
    state: "_PhaseRuntimeState",
    *,
    round_i: int,
    user_message: str = "",
    has_prior_turns_summary: bool = False,
) -> str:
    prompts: list[str] = []
    if state.disclosure_step <= 0 and not has_prior_turns_summary:
        prompts.append(FIRST_SEARCH_PROMPT.format(user_message=user_message).strip())
    elif state.disclosure_step == 1 and not state.disclosure_refine_consumed:
        prompts.append(POST_SEARCH_PROMPT.strip())
    active_items = _active_page_items(state)
    if active_items:
        active_ids = ", ".join(str(item.item_id) for item in active_items)
        prompts.append(ACTIVE_PAGE_STATE_PROMPT.format(count=len(active_items), ids=active_ids))
    late_round_warning = _late_round_warning_message(round_i=round_i)
    if late_round_warning:
        prompts.append(late_round_warning)
    return "\n\n".join(part for part in prompts if str(part).strip()).strip()


def _build_loop_system_prompt(
    cfg: dict[str, Any],
    user_message: str = "",
    *,
    disclosure_step: int = 0,
    round_i: int = 0,
) -> str:
    del disclosure_step
    custom = str(cfg.get("system_prompt") or "").strip()
    late_round_warning = _late_round_warning_message(round_i=round_i)
    if late_round_warning:
        custom = f"{custom}\n{late_round_warning}".strip() if custom else late_round_warning
    name = str(cfg.get("name") or DEFAULT_NAME).strip()
    return BASE_SYSTEM_PROMPT.format(
        name=name,
        language=cfg.get("language") or "zh-CN",
        time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        custom=(custom + "\n") if custom else "",
        user_message=user_message,
    ).strip()


def _build_loop_messages(
    *,
    cfg: dict[str, Any],
    prompt_text: str,
    history: list[dict[str, Any]],
    state: _PhaseRuntimeState,
    round_i: int,
    context: Any = None,
) -> list[dict[str, Any]]:
    has_prior_turns_summary = _context_has_prior_turns_summary(context)
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_loop_system_prompt(
                cfg,
                prompt_text,
                disclosure_step=state.disclosure_step,
                round_i=round_i,
            ),
        }
    ]
    _append_context_messages(msgs, context)
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if not _message_content_text(content):
            continue
        msgs.append({"role": role, "content": deepcopy(content)})
    round_brief = _build_round_brief_message(
        state,
        round_i=round_i,
        user_message=prompt_text,
        has_prior_turns_summary=has_prior_turns_summary,
    )
    if round_brief:
        msgs.append({"role": "user", "content": round_brief})
        if state.disclosure_step == 1:
            state.disclosure_refine_consumed = True
    latest_raw_message = _build_latest_raw_message(state)
    if latest_raw_message:
        msgs.append({"role": "user", "content": latest_raw_message})
    active_page_message = _build_active_page_message(state)
    if active_page_message:
        msgs.append({"role": "user", "content": active_page_message})
    if state.pending_user_note:
        msgs.append({"role": "user", "content": state.pending_user_note})
    return msgs


def _format_model_error_message(exc: Exception) -> str:
    err = str(exc or "").strip()
    for frag in ("APIError:", "Exception -"):
        if frag in err:
            err = err[err.index(frag) + len(frag):].strip()
            break
    if len(err) > 200:
        err = err[:200]

    lowered = err.lower()
    missing_key = (
        "api_key" in lowered
        or "openai_api_key" in lowered
        or "api key" in lowered
    )
    if missing_key:
        config_path = str(CONFIG_PATH.expanduser())
        return (
            f"[模型调用失败] {err}\n"
            f"可通过设置环境变量 `OPENAI_API_KEY`，或在配置文件 `{config_path}` 中填写 `api_key:`。"
        )
    lowered = err.lower()
    if (
        "midstreamfallbackerror" in lowered
        or "tool choice is none, but model called a tool" in lowered
        or ("provider_unavailable" in lowered and "groq" in lowered)
    ):
        config_path = str(CONFIG_PATH.expanduser())
        return (
            f"[模型流式调用失败] {err}\n"
            f"当前上游 provider 的流式返回存在兼容问题。"
            f"请在配置文件 `{config_path}` 中设置 `stream: false` 后重试；"
            f"单次问答也可以直接使用非流模式。"
        )
    if any(token in lowered for token in ("image_url", "image input", "multimodal", "vision", "image")):
        return (
            f"[图片输入不可用] {err}\n"
            "当前第一阶段会直接把图片发送给主模型，不再使用单独的视觉总结模型。"
            "请切换到支持图片输入的阶段一模型后重试。"
        )
    return f"[模型调用失败] {err}"


_RETRYABLE_MODEL_ERROR_FRAGMENTS = (
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "server disconnected",
    "remote end closed connection",
    "connection reset",
    "connection aborted",
    "connection timed out",
    "connect timeout",
    "read timed out",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "too many requests",
    "rate limit",
    "provider_unavailable",
    "apiconnectionerror",
    "remoteprotocolerror",
    "stream error",
    "ssl:",
)

_NON_RETRYABLE_MODEL_ERROR_FRAGMENTS = (
    "invalid api key",
    "incorrect api key",
    "authentication",
    "unauthorized",
    "permission denied",
    "context_length_exceeded",
    "maximum context length",
    "invalid_request_error",
    "unsupported parameter",
    "tool choice is none, but model called a tool",
    "guardrail",
    "safeguard",
    "content policy",
    "safety system",
    "does not support",
)


def _model_retry_count(cfg: dict[str, Any]) -> int:
    value = cfg.get("model_retries")
    if value is None:
        value = cfg.get("llm_retries")
    try:
        parsed = int(value)
    except Exception:
        parsed = 2
    return max(0, min(parsed, 5))


def _model_retry_base_delay_s(cfg: dict[str, Any]) -> float:
    value = cfg.get("model_retry_base_delay_s")
    try:
        parsed = float(value)
    except Exception:
        parsed = 1.0
    return max(0.1, min(parsed, 30.0))


def _model_retry_max_delay_s(cfg: dict[str, Any]) -> float:
    value = cfg.get("model_retry_max_delay_s")
    try:
        parsed = float(value)
    except Exception:
        parsed = 8.0
    return max(_model_retry_base_delay_s(cfg), min(parsed, 120.0))


def _model_retry_delay_s(cfg: dict[str, Any], retry_index: int) -> float:
    retry_no = max(1, int(retry_index))
    base = _model_retry_base_delay_s(cfg)
    cap = _model_retry_max_delay_s(cfg)
    return min(cap, base * (2 ** (retry_no - 1)))


def _is_retryable_model_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in _NON_RETRYABLE_MODEL_ERROR_FRAGMENTS):
        return False
    return any(token in text for token in _RETRYABLE_MODEL_ERROR_FRAGMENTS)


def _model_retry_status_text(retry_index: int, retry_count: int, delay_s: float) -> str:
    total = max(1, int(retry_count))
    current = max(1, min(int(retry_index), total))
    return f"模型重试({current}/{total}) · {delay_s:.1f}s"


def _call_model_with_retries(
    invoke: Any,
    *,
    cfg: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
    trace_label: str,
    log_id: str | None = None,
    on_status: Any | None = None,
) -> Any:
    retry_count = _model_retry_count(cfg)
    for retry_index in range(retry_count + 1):
        attempt_started_at = time.perf_counter()
        try:
            return invoke()
        except StopRequestedError:
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - attempt_started_at) * 1000
            log_model_call(
                label=trace_label,
                model=model,
                messages=messages,
                output="",
                error=str(e)[:300],
                duration_ms=duration_ms,
                config=cfg,
                log_id=log_id,
            )
            if retry_index >= retry_count or not _is_retryable_model_error(e):
                raise
            delay_s = _model_retry_delay_s(cfg, retry_index + 1)
            logging.warning(
                "Retrying model call %s (%s) after error: %s",
                trace_label,
                model,
                str(e).strip() or type(e).__name__,
            )
            if callable(on_status):
                try:
                    on_status(_model_retry_status_text(retry_index + 1, retry_count, delay_s))
                except Exception:
                    pass
            time.sleep(delay_s)


def _looks_like_guardrail_model(model: str) -> bool:
    text = str(model or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in ("safeguard", "guardrail", "moderation"))


def _format_empty_output_message(cfg: dict[str, Any], *, round_i: int) -> str:
    model_name = str(cfg.get("model") or DEFAULT_MODEL).strip()
    if round_i == 0 and _looks_like_guardrail_model(model_name):
        config_path = str(CONFIG_PATH.expanduser())
        return (
            f"[主模型未产生可用输出] 当前主模型 `{model_name}` 看起来是 safeguard / guardrail 类模型，"
            "更适合安全过滤，不适合作为主控制模型。\n"
            f"请在配置文件 `{config_path}` 中把 `models[0]` 换成常规对话模型，"
            "例如 `openrouter/openai/gpt-oss-120b` 或 `openrouter/google/gemini-3.1-flash-lite-preview` 后重试。"
        )
    return "未获得有效回答。"


def _format_log_message_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        image_index = 0
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
                continue
            if item_type == "image_url":
                image_index += 1
                parts.append(f"[Image #{image_index}]")
                continue
            parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "")


def _default_image_prompt(image_count: int) -> str:
    if image_count <= 0:
        return ""
    if image_count == 1:
        return "请根据图片内容进行分析并回答。"
    return "请结合这些图片内容进行分析并回答。"


def _effective_prompt_text(question: str, image_count: int = 0) -> str:
    text = str(question or "").strip()
    if not text:
        return _default_image_prompt(image_count)
    normalized = re.sub(r"\[Image #\d+\]", "", text)
    if normalized.strip():
        return text
    return _default_image_prompt(image_count) or text


def _resolve_max_rounds(cfg: dict[str, Any]) -> int:
    try:
        value = int(str(cfg.get("max_rounds") or "").strip())
    except Exception:
        value = 20
    return max(2, min(value or 20, 32))


def _resolve_external_tool_timeout_s(cfg: dict[str, Any]) -> float:
    raw = cfg.get("external_tool_timeout_s")
    if raw in (None, "", False):
        return float(_EXTERNAL_TOOL_TIMEOUT_S)
    try:
        value = float(str(raw).strip())
    except Exception:
        return float(_EXTERNAL_TOOL_TIMEOUT_S)
    if value <= 0:
        return float(_EXTERNAL_TOOL_TIMEOUT_S)
    return max(5.0, min(value, 300.0))


def _format_timeout_seconds(value: float) -> str:
    try:
        normalized = float(value)
    except Exception:
        normalized = 0.0
    if normalized.is_integer():
        return str(int(normalized))
    return f"{normalized:.1f}"


_INLINE_IMAGE_BASE64_PREFIXES = (
    "/9j/",  # jpeg
    "iVBORw0KGgo",  # png
    "R0lGOD",  # gif
    "UklGR",  # webp
    "Qk",  # bmp
)


def _guess_image_mime_type(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


def _inline_image_data_url(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("data:image/"):
        return text
    if len(text) <= 240 and not text.startswith(_INLINE_IMAGE_BASE64_PREFIXES):
        return ""
    try:
        raw = base64.b64decode(text, validate=True)
    except Exception:
        return ""
    if not raw:
        return ""
    mime_type = _guess_image_mime_type(raw)
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"


def _build_multimodal_content(question: str, image_paths: list[str] | None = None) -> str | list[dict[str, Any]]:
    paths = [str(path).strip() for path in (image_paths or []) if str(path).strip()]
    text = _effective_prompt_text(question, len(paths))
    if not paths:
        return text

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path_str in paths:
        raw_value = str(path_str or "").strip()
        data_url = _inline_image_data_url(raw_value)
        if not data_url:
            path = Path(raw_value).expanduser()
            try:
                if not path.exists():
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            mime_type = str(mimetypes.guess_type(path.name)[0] or "image/png").strip() or "image/png"
            data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def build_stage_model_config(
    config: dict[str, Any] | None,
    stage: str,
    *,
    stage1_model_index: int | None = None,
    stage2_model_index: int | None = None,
) -> dict[str, Any]:
    raw_cfg = build_model_config(config or load_config())
    profiles = raw_cfg.get("models") or []
    profile_count = len(profiles) if isinstance(profiles, list) else 0
    if profile_count <= 0:
        return raw_cfg

    picked_stage1 = int(
        raw_cfg.get("stage1_model_index")
        if stage1_model_index is None
        else stage1_model_index
    ) % profile_count
    picked_stage2 = int(
        raw_cfg.get("stage2_model_index")
        if stage2_model_index is None
        else stage2_model_index
    ) % profile_count
    stage_name = str(stage or "").strip().lower()
    picked_index = picked_stage2 if stage_name in {"stage2", "execute", "final"} else picked_stage1
    stage_cfg = build_model_config(raw_cfg, model_index=picked_index)
    stage_cfg["stage1_model_index"] = picked_stage1
    stage_cfg["stage2_model_index"] = picked_stage2
    return stage_cfg


@dataclass
class _SessionRuntimeState:
    search_history_raw: list[str] = field(default_factory=list)
    search_history_normalized: list[str] = field(default_factory=list)
    search_results_deduped: list[dict[str, Any]] = field(default_factory=list)
    visited_page_urls: set[str] = field(default_factory=set)
    ref_targets: dict[str, str] = field(default_factory=dict)
    page_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock, repr=False)


@dataclass
class _ContextItem:
    item_id: int
    item_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PhaseRuntimeState:
    items: list[_ContextItem] = field(default_factory=list)
    next_item_id: int = 0
    latest_raw_ids: list[int] = field(default_factory=list)
    latest_raw_round: int = 0
    active_page_ids: list[int] = field(default_factory=list)
    pending_user_note: str = ""
    input_has_images: bool = False
    disclosure_step: int = 0
    disclosure_refine_consumed: bool = False


def _truncate_text(text: Any, limit: int = 240) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max(1, int(limit)):
        return raw
    return raw[: max(1, int(limit)) - 1].rstrip() + "…"


def _normalize_state_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("//"):
        text = "https:" + text
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    try:
        parsed = urlsplit(text)
    except Exception:
        return ""
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower().strip()
    if not netloc:
        return ""
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _url_domain(url: str) -> str:
    normalized = _normalize_state_url(url)
    if not normalized:
        return ""
    try:
        host = urlsplit(normalized).netloc.lower().strip()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_search_query(query: str) -> str:
    text = str(query or "").strip().strip("`'\"")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _next_context_item_id(state: _PhaseRuntimeState) -> int:
    item_id = int(state.next_item_id)
    state.next_item_id += 1
    return item_id


def _add_context_item(
    state: _PhaseRuntimeState,
    item_type: str,
    data: dict[str, Any],
    *,
    pending: bool = False,
) -> _ContextItem:
    del pending
    item = _ContextItem(item_id=_next_context_item_id(state), item_type=str(item_type).strip(), data=dict(data))
    state.items.append(item)
    return item


def _find_context_item(state: _PhaseRuntimeState, item_id: int) -> _ContextItem | None:
    for item in state.items:
        if int(item.item_id) == int(item_id):
            return item
    return None


def _active_page_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    items: list[_ContextItem] = []
    alive_ids: list[int] = []
    for raw_item_id in list(state.active_page_ids):
        item = _find_context_item(state, int(raw_item_id))
        if item is None or str(item.item_type or "").strip() != "page.active":
            continue
        alive_ids.append(int(item.item_id))
        items.append(item)
    if alive_ids != state.active_page_ids:
        state.active_page_ids = alive_ids
    return items


def _current_active_page_item(state: _PhaseRuntimeState) -> _ContextItem | None:
    items = _active_page_items(state)
    return items[-1] if items else None


def _iter_context_items(
    state: _PhaseRuntimeState,
    prefix: str = "",
    *,
    include_pending: bool = False,
) -> list[_ContextItem]:
    del include_pending
    source = state.items
    items = sorted(source, key=lambda item: int(item.item_id))
    if not prefix:
        return items
    return [item for item in items if item.item_type.startswith(prefix)]


def _memory_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    return _iter_context_items(state, "memory.")


def _items_for_ids(state: _PhaseRuntimeState, ids: list[int]) -> list[_ContextItem]:
    if not ids:
        return []
    by_id = {int(item.item_id): item for item in _memory_items(state)}
    selected: list[_ContextItem] = []
    for item_id in ids:
        item = by_id.get(int(item_id))
        if item is not None:
            selected.append(item)
    return selected


def _latest_raw_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    return _items_for_ids(state, state.latest_raw_ids)


def _context_value_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item or "").strip() for item in value]
        filtered = [item for item in parts if item]
        return " | ".join(filtered)
    return str(value or "").strip()


def _context_item_label(item: _ContextItem) -> str:
    label = str(item.item_type or "").split(".", 1)[-1] or str(item.item_type or "").strip()
    round_no = int(item.data.get("round") or 0)
    if round_no > 0:
        return f"{label} | round {round_no}"
    return label


def _append_context_item_lines(
    lines: list[str],
    item: _ContextItem,
    *,
    text_limit: int = 800,
) -> None:
    data = dict(item.data)
    lines.append(f"[{item.item_id}] {_context_item_label(item)}")
    visible_keys = ("query", "question", "title", "url", "search")
    for key in visible_keys:
        value = _context_value_text(data.get(key))
        if value:
            lines.append(f"{key}: {value}")
    text = _context_value_text(data.get("text") or data.get("snippet"))
    if text:
        lines.append(f"text: {_truncate_text(text, text_limit)}")
    lines.append("")


def _build_context_items_message(
    *,
    heading: str,
    empty_text: str,
    items: list[_ContextItem],
) -> str:
    if not items:
        return ""
    lines = [heading, ""]
    for item in items:
        _append_context_item_lines(lines, item)
    return "\n".join(line for line in lines if str(line).strip()).strip()


def _merge_search_results(
    existing: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    query: str,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    next_order = 0

    for item in existing:
        if not isinstance(item, dict):
            continue
        url = _normalize_state_url(str(item.get("url") or "").strip())
        key = url or f"fallback::{next_order}"
        normalized = dict(item)
        if url:
            normalized["url"] = url
        normalized.setdefault("_order", next_order)
        normalized.setdefault("matched_queries", [])
        by_key[key] = normalized
        ordered.append(normalized)
        next_order = max(next_order, int(normalized.get("_order") or 0) + 1)

    matched_query = str(query or "").strip()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_url = str(row.get("url") or "").strip()
        url = _normalize_state_url(raw_url)
        key = url or f"fallback::{next_order}"
        title = str(row.get("title") or "").strip() or DEFAULT_NO_TITLE_TEXT
        snippet = str(row.get("snippet") or row.get("intro") or "").strip()
        provider = str(row.get("provider") or "").strip()
        domain = _url_domain(url)
        existing_row = by_key.get(key)
        if existing_row is None:
            matched_queries = [matched_query] if matched_query else []
            normalized = {
                "title": title,
                "url": url or raw_url,
                "snippet": snippet,
                "provider": provider,
                "domain": domain,
                "matched_queries": matched_queries,
                "_order": next_order,
            }
            by_key[key] = normalized
            ordered.append(normalized)
            next_order += 1
            continue
        if title and len(title) > len(str(existing_row.get("title") or "")):
            existing_row["title"] = title
        if snippet and len(snippet) > len(str(existing_row.get("snippet") or "")):
            existing_row["snippet"] = snippet
        if provider and not str(existing_row.get("provider") or "").strip():
            existing_row["provider"] = provider
        if domain and not str(existing_row.get("domain") or "").strip():
            existing_row["domain"] = domain
        queries = [
            str(item or "").strip()
            for item in existing_row.get("matched_queries", [])
            if str(item or "").strip()
        ]
        if matched_query and matched_query not in queries:
            queries.append(matched_query)
        existing_row["matched_queries"] = queries[:8]

    ordered.sort(key=lambda item: int(item.get("_order") or 0))
    return ordered


def _search_history_contains(
    state: "_SessionRuntimeState | None",
    query: str,
    *,
    filters: dict[str, Any] | None = None,
) -> bool:
    normalized = normalize_search_request_key(query, filters)
    if not normalized or state is None:
        return False
    with state.lock:
        if normalized in state.search_history_normalized:
            return True
        state.search_history_raw.append(_format_search_request_label(query, filters))
        state.search_history_normalized.append(normalized)
        state.search_history_raw[:] = state.search_history_raw[-96:]
        state.search_history_normalized[:] = state.search_history_normalized[-96:]
    return False


def _search_results_for_request(
    runtime_state: "_SessionRuntimeState | None",
    *,
    query: str,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if runtime_state is None:
        return []
    search_label = _format_search_request_label(query, filters)
    if not search_label:
        return []
    with runtime_state.lock:
        rows = runtime_state.search_results_deduped[:]
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        matched_queries = [
            str(item or "").strip()
            for item in row.get("matched_queries", [])
            if str(item or "").strip()
        ]
        if search_label in matched_queries:
            selected.append(dict(row))
    return _public_search_results(selected)


def _build_search_payload_markdown(
    *,
    query: str,
    search_filters: dict[str, Any] | None = None,
    public_results: list[dict[str, Any]],
    skipped_duplicate: bool,
    direction_brief: str = "",
    expanded_queries: list[str] | None = None,
    duplicate_queries: list[str] | None = None,
    summary_text: str = "",
    error: str = "",
) -> str:
    return build_search_document_markdown(
        query=query,
        search_filters=search_filters,
        public_results=public_results,
        skipped_duplicate=skipped_duplicate,
        reminder=SEARCH_RESULT_REMINDER,
        error=error,
        direction_brief=direction_brief,
        expanded_queries=expanded_queries,
        duplicate_queries=duplicate_queries,
        summary_text=summary_text,
        default_no_results_text=DEFAULT_NO_RESULTS_TEXT,
        duplicate_query_skipped_text=DUPLICATE_QUERY_SKIPPED_TEXT,
        default_no_title_text=DEFAULT_NO_TITLE_TEXT,
    )


def _normalize_page_window(value: Any, default: int = 20) -> int | str:
    text = str(value or "").strip().lower()
    if text == "all":
        return "all"
    try:
        parsed = int(text)
    except Exception:
        parsed = default
    return max(10, min(parsed, 80))


def _clean_page_probe_pattern(value: str) -> str:
    text = str(value or "").strip().strip("`'\"")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_page_probe_task(value: Any) -> str:
    return _clean_page_probe_pattern(str(value or ""))


_PAGE_PROBE_QUOTED_SEGMENT_RE = re.compile(r"[\"“”'‘’「『《〈（【]([^\"“”'‘’」』》〉）】]{1,48})[\"“”'‘’」』》〉）】]")
_PAGE_PROBE_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:円|日元|%|人|年)?")
_PAGE_PROBE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._/:+-]{1,31}|[0-9][0-9,./:+-]{0,31}(?:円|日元|%|人|年)?|[一-龥ぁ-ゟ゠-ヿー・]{2,24}[!！]?")
_PAGE_PROBE_GENERIC_ANCHORS = {
    "html",
    "body",
    "div",
    "span",
    "img",
    "svg",
    "href",
    "src",
    "alt",
    "p",
    "h1",
    "h2",
    "h3",
    "li",
    "ul",
    "ol",
    "a",
    "main",
    "article",
    "section",
    "header",
    "footer",
    "nav",
    "menu",
    "sidebar",
    "button",
    "link",
    "links",
    "image",
    "images",
    "script",
    "style",
    "class",
    "id",
    "title",
    "task",
    "page",
    "summary",
    "explanation",
    "提取",
    "确认",
    "说明",
    "重点",
    "明确",
    "如果",
    "请",
    "页面",
    "页",
    "段落",
    "行",
    "关键词",
    "搜索",
    "搜索词",
    "页面结构",
    "网页结构",
    "个人转述",
    "本页",
    "该页",
    "任务",
    "support",
    "evidence",
}
_PAGE_PROBE_GENERIC_FRAGMENTS = (
    "html",
    "body",
    "div",
    "span",
    "img",
    "svg",
    "href",
    "src",
    "header",
    "footer",
    "button",
    "script",
    "style",
    "header",
    "提取",
    "确认",
    "说明",
    "重点",
    "明确",
    "页面",
    "关键词",
    "搜索",
    "结构",
    "本页",
    "该页",
    "任务",
)


def _page_probe_anchor_key(value: str) -> str:
    text = _clean_page_probe_pattern(value).strip("[](){}<>\"'`“”‘’「」『』《》〈〉（）。，、；：！？")
    return text.casefold()


def _looks_like_page_probe_anchor(value: str) -> bool:
    clean = _clean_page_probe_pattern(value).strip("[](){}<>\"'`“”‘’「」『』《》〈〉（）。，、；：！？")
    if not clean or len(clean) < 2 or len(clean) > 32:
        return False
    if clean.count(" ") > 2:
        return False
    normalized = clean.casefold()
    if normalized in _PAGE_PROBE_GENERIC_ANCHORS:
        return False
    if not re.search(r"[A-Za-z0-9一-龥ぁ-ゟ゠-ヿ]", clean):
        return False
    if not re.search(r"[A-Za-z0-9ぁ-ゟ゠-ヿ]", clean):
        pure_han = re.fullmatch(r"[一-龥]+", clean) is not None
        if pure_han and any(fragment in clean for fragment in _PAGE_PROBE_GENERIC_FRAGMENTS):
            return False
    return True


def _extract_page_probe_anchors(text: str, *, max_items: int = 4) -> list[str]:
    source = _normalize_page_probe_task(text)
    if not source:
        return []

    anchors: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        clean = _clean_page_probe_pattern(candidate).strip("[](){}<>\"'`“”‘’「」『』《》〈〉（）。，、；：！？")
        key = _page_probe_anchor_key(clean)
        if not _looks_like_page_probe_anchor(clean) or not key or key in seen:
            return
        seen.add(key)
        anchors.append(clean)

    for match in _PAGE_PROBE_QUOTED_SEGMENT_RE.finditer(source):
        _push(match.group(1))
        if len(anchors) >= max_items:
            return anchors

    for match in _PAGE_PROBE_NUMBER_RE.finditer(source):
        _push(match.group(0))
        if len(anchors) >= max_items:
            return anchors

    for match in _PAGE_PROBE_TOKEN_RE.finditer(source):
        _push(match.group(0))
        if len(anchors) >= max_items:
            return anchors

    return anchors


def _derive_page_probe_patterns(task: str) -> list[str]:
    return _extract_page_probe_anchors(task, max_items=4)


def _normalize_page_extract_search(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                parts.append(text)
    else:
        text = str(value or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return ""
    text = " | ".join(parts)
    text = text.replace("\n", " | ")
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s+", " ", text).strip(" |")
    return text


def _split_page_extract_search_terms(value: Any, *, max_items: int = 8) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    chunks: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            if "|" in text:
                chunks.extend(segment.strip() for segment in text.split("|"))
            else:
                chunks.append(text)
    else:
        raw = _normalize_page_extract_search(value)
        if raw:
            chunks.extend(segment.strip() for segment in raw.split("|"))
    for chunk in chunks:
        clean = _clean_page_probe_pattern(chunk)
        key = _page_probe_anchor_key(clean)
        if not key or key in seen or not _looks_like_page_probe_anchor(clean):
            continue
        seen.add(key)
        terms.append(clean)
        if len(terms) >= max(1, int(max_items)):
            break
    return terms


def _page_cache_lookup(
    runtime_state: "_SessionRuntimeState | None",
    normalized_url: str,
) -> dict[str, Any] | None:
    if runtime_state is None or not normalized_url:
        return None
    with runtime_state.lock:
        payload = runtime_state.page_cache.get(normalized_url)
        return deepcopy(payload) if isinstance(payload, dict) else None


def _page_cache_store(
    runtime_state: "_SessionRuntimeState | None",
    normalized_url: str,
    payload: dict[str, Any],
) -> None:
    if runtime_state is None or not normalized_url or not isinstance(payload, dict):
        return
    with runtime_state.lock:
        runtime_state.page_cache[normalized_url] = deepcopy(payload)


def _load_page_payload(
    *,
    url: str,
    cfg: dict[str, Any],
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], bool]:
    normalized_url = _normalize_state_url(url)
    cached = _page_cache_lookup(runtime_state, normalized_url)
    if isinstance(cached, dict):
        cached["_from_cache"] = True
        return cached, True

    payload = _page_extract(
        url,
        config=cfg,
        max_chars=12000,
        progress_callback=progress_callback,
    )
    if not isinstance(payload, dict):
        payload = {"ok": False, "error": "invalid page payload", "url": url, "content": ""}
    if normalized_url and payload.get("ok"):
        stored = dict(payload)
        stored["_from_cache"] = False
        _page_cache_store(runtime_state, normalized_url, stored)
    payload["_from_cache"] = False
    return payload, False


def _extract_page_line_matches(
    content: str,
    *,
    patterns: list[str],
    window: int | str,
    per_pattern_limit: int = 5,
) -> list[dict[str, Any]]:
    raw_lines = str(content or "").splitlines()
    if not raw_lines:
        return []

    if window == "all":
        matched: list[dict[str, Any]] = []
        for line_no, raw_line in enumerate(raw_lines, start=1):
            text = str(raw_line).rstrip()
            if not text.strip():
                continue
            matched.append({"line": line_no, "text": text})
        return matched

    lowered_patterns = [pattern.casefold() for pattern in patterns if pattern]
    if not lowered_patterns:
        return []

    ranges: list[tuple[int, int]] = []
    radius = max(1, int(window))
    for pattern in lowered_patterns:
        hits = 0
        for index, raw_line in enumerate(raw_lines, start=1):
            if pattern not in raw_line.casefold():
                continue
            start = max(1, index - radius)
            end = min(len(raw_lines), index + radius)
            ranges.append((start, end))
            hits += 1
            if hits >= max(1, int(per_pattern_limit)):
                break

    if not ranges:
        return []

    merged_ranges: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1] + 1:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    matched: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for start, end in merged_ranges:
        for line_no in range(start, end + 1):
            if line_no in seen_lines:
                continue
            seen_lines.add(line_no)
            text = str(raw_lines[line_no - 1]).rstrip()
            if not text.strip():
                continue
            matched.append({"line": line_no, "text": text})
            if len(matched) >= 120:
                return matched
    return matched


def _sample_page_lines(
    content: str,
    *,
    max_lines: int = 30,
) -> tuple[list[dict[str, Any]], int, int, int]:
    raw_lines = str(content or "").splitlines()
    if not raw_lines:
        return [], 0, 0, 1
    total_lines = len(raw_lines)
    total_chars = len(str(content or ""))
    candidates = [
        (index, raw_line.rstrip())
        for index, raw_line in enumerate(raw_lines)
        if not _looks_like_sample_noise_line(raw_line, line_no=index + 1, total_lines=total_lines)
    ]
    if not candidates:
        return [], total_lines, total_chars, max(1, total_lines)
    preview_count = max(1, min(int(max_lines), len(candidates)))
    if preview_count <= 1:
        picked_indexes = [0]
    else:
        picked_indexes = []
        seen_indexes: set[int] = set()
        last_index = len(candidates) - 1
        for i in range(preview_count):
            index = round((last_index * i) / (preview_count - 1))
            if index in seen_indexes:
                continue
            seen_indexes.add(index)
            picked_indexes.append(index)
        if len(picked_indexes) < preview_count:
            for index in range(len(candidates)):
                if index in seen_indexes:
                    continue
                picked_indexes.append(index)
                if len(picked_indexes) >= preview_count:
                    break
        picked_indexes.sort()
    sample_step = max(1, (total_lines + preview_count - 1) // preview_count)
    sampled: list[dict[str, Any]] = []
    for index in picked_indexes:
        source_index, source_text = candidates[index]
        line_no = source_index + 1
        text = _truncate_text(str(source_text).rstrip(), 30)
        sampled.append({"line": line_no, "text": text})
    return sampled, total_lines, total_chars, sample_step


def _extract_page_line_range(
    content: str,
    *,
    start_line: int,
    end_line: int,
    max_lines: int = 240,
) -> list[dict[str, Any]]:
    raw_lines = str(content or "").splitlines()
    if not raw_lines:
        return []
    start = max(1, int(start_line))
    end = max(start, int(end_line))
    end = min(end, len(raw_lines))
    matched: list[dict[str, Any]] = []
    for line_no in range(start, end + 1):
        text = str(raw_lines[line_no - 1]).rstrip()
        if not text.strip():
            continue
        matched.append({"line": line_no, "text": text})
        if len(matched) >= max(1, int(max_lines)):
            break
    return matched


def _rewrite_open_page_line(
    text: str,
    *,
    page_item_id: int,
    next_link_index: int,
) -> tuple[str, list[dict[str, Any]], int]:
    value = str(text or "").rstrip()
    links: list[dict[str, Any]] = []
    current_index = max(1, int(next_link_index))

    def _img_link_repl(match: re.Match[str]) -> str:
        nonlocal current_index
        alt = re.sub(r"\s+", " ", str(match.group(1) or "").strip()) or "image"
        url = str(match.group(2) or "").strip()
        ref = f"{page_item_id}:{current_index}"
        links.append({"index": current_index, "url": url, "text": f"IMG:{alt}"})
        current_index += 1
        return f"IMG:{alt} [{ref}]"

    def _img_repl(match: re.Match[str]) -> str:
        alt = re.sub(r"\s+", " ", str(match.group(1) or "").strip()) or "image"
        return f"IMG:{alt}"

    def _link_repl(match: re.Match[str]) -> str:
        nonlocal current_index
        label = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
        url = str(match.group(2) or "").strip()
        rendered = label or "link"
        ref = f"{page_item_id}:{current_index}"
        links.append({"index": current_index, "url": url, "text": rendered})
        current_index += 1
        return f"{rendered} [{ref}]"

    value = re.sub(r"\[!\[([^\]]*)\]\([^)]+\)\]\(([^)]+)\)", _img_link_repl, value)
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", _img_repl, value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link_repl, value)
    value = re.sub(r"<https?://[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value, links, current_index


def _render_active_page_markdown(
    *,
    page_item_id: int,
    url: str,
    title: str,
    raw_lines: list[str],
    from_cache: bool,
) -> tuple[str, list[dict[str, Any]], int, int]:
    total_lines = len(raw_lines)
    total_chars = len("\n".join(raw_lines))
    width = max(2, len(str(max(total_lines, 1))))
    lines = [f"# Page [{page_item_id}]", ""]
    if title:
        lines.append(f"{PAGE_MARKDOWN_TITLE_PREFIX}{title}")
    if url:
        lines.append(f"Source: {url}")
    lines.append(f"Ref usage: use `navigate(ref=\"{page_item_id}:2\")` to follow page [{page_item_id}] link #2.")
    lines.append(
        f"{PAGE_MARKDOWN_CACHE_PREFIX}{PAGE_MARKDOWN_CACHE_HIT_TEXT if from_cache else PAGE_MARKDOWN_CACHE_MISS_TEXT}"
    )
    lines.append(f"Preview: chars={total_chars} lines={total_lines} opened=all")
    lines.extend(["", PAGE_MARKDOWN_MATCHED_LINES_TEXT])

    link_index = 1
    link_rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(raw_lines, start=1):
        clean = str(raw_line or "").rstrip()
        if not clean.strip():
            continue
        rewritten, discovered, link_index = _rewrite_open_page_line(
            clean,
            page_item_id=page_item_id,
            next_link_index=link_index,
        )
        if not rewritten:
            continue
        for row in discovered:
            link_rows.append(
                {
                    "index": int(row.get("index") or 0),
                    "url": str(row.get("url") or "").strip(),
                    "text": str(row.get("text") or "").strip(),
                    "line": line_no,
                }
            )
        lines.append(f"L{line_no:0{width}d} | {rewritten}")
    return "\n".join(lines).strip(), link_rows, total_lines, total_chars


def _build_page_probe_markdown(
    *,
    url: str,
    title: str,
    scope: str,
    matches: list[dict[str, Any]],
    preview_meta: dict[str, Any] | None = None,
    page_error: str = "",
    from_cache: bool = False,
) -> str:
    lines = [f"# Page: {url}" if url else PAGE_MARKDOWN_EMPTY_TITLE, ""]
    if title:
        lines.append(f"{PAGE_MARKDOWN_TITLE_PREFIX}{title}")
    if scope:
        lines.append(f"{PAGE_MARKDOWN_SCOPE_PREFIX}{scope}")
    if isinstance(preview_meta, dict) and preview_meta:
        meta_parts: list[str] = []
        total_lines = int(preview_meta.get("total_lines") or 0)
        total_chars = int(preview_meta.get("total_chars") or 0)
        shown = int(preview_meta.get("shown_lines") or 0)
        sample_step = int(preview_meta.get("sample_step") or 0)
        if total_chars > 0:
            meta_parts.append(f"chars={total_chars}")
        if total_lines > 0:
            meta_parts.append(f"lines={total_lines}")
        if shown > 0:
            meta_parts.append(f"shown={shown}")
        if sample_step > 0:
            meta_parts.append(f"step~={sample_step}")
        if meta_parts:
            lines.append("Preview: " + " ".join(meta_parts))
    lines.append(
        f"{PAGE_MARKDOWN_CACHE_PREFIX}{PAGE_MARKDOWN_CACHE_HIT_TEXT if from_cache else PAGE_MARKDOWN_CACHE_MISS_TEXT}"
    )
    if page_error:
        lines.extend(["", page_error])
        return "\n".join(lines).strip()
    if not matches:
        lines.extend(["", DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT if from_cache else DEFAULT_PAGE_NO_MATCHING_TEXT])
        return "\n".join(lines).strip()
    lines.extend(["", PAGE_MARKDOWN_MATCHED_LINES_TEXT])
    width = max(2, len(str(max(int(item.get("line") or 0) for item in matches if isinstance(item, dict)) or 1)))
    for item in matches:
        if not isinstance(item, dict):
            continue
        try:
            line_no = int(item.get("line") or 0)
        except Exception:
            line_no = 0
        text = str(item.get("text") or "").rstrip()
        if not line_no or not text:
            continue
        lines.append(f"L{line_no:0{width}d} | {text}")
    return "\n".join(lines).strip()


def _run_page_probe(
    *,
    url: str,
    mode: str,
    start_line: int,
    end_line: int,
    question: Any,
    cfg: dict[str, Any],
    stats: Stats | None = None,
    log_id: str | None = None,
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    source_url = str(url or "").strip()
    if not source_url:
        return {
            "ok": False,
            "error": "page url is empty",
            "url": "",
            "question": _normalize_page_probe_task(question),
            "mode": str(mode or "sample").strip() or "sample",
        }

    normalized_url = _normalize_state_url(source_url) or source_url
    if runtime_state is not None and normalized_url:
        with runtime_state.lock:
            runtime_state.visited_page_urls.add(normalized_url)

    normalized_question = _normalize_page_probe_task(question)
    normalized_mode = str(mode or "sample").strip().lower()
    if normalized_mode not in {"sample", "range"}:
        normalized_mode = "sample"
    preview_meta: dict[str, Any] = {}
    normalized_scope = ""
    range_start = max(1, int(start_line))
    range_end = max(range_start, int(end_line))
    if normalized_mode == "range":
        normalized_scope = f"lines={range_start}-{range_end}"
    page_payload, from_cache = _load_page_payload(
        url=source_url,
        cfg=cfg,
        runtime_state=runtime_state,
        progress_callback=progress_callback,
    )
    title = str(page_payload.get("title") or "").strip()
    content = str(page_payload.get("content") or "").strip()
    page_error = str(page_payload.get("error") or "").strip()
    matches = []
    if content and not page_error:
        if normalized_mode == "range":
            matches = _extract_page_line_range(
                content,
                start_line=range_start,
                end_line=range_end,
            )
        else:
            matches, total_lines, total_chars, sample_step = _sample_page_lines(
                content,
                max_lines=30,
            )
            preview_meta = {
                "total_lines": total_lines,
                "total_chars": total_chars,
                "shown_lines": len(matches),
                "sample_step": sample_step,
            }

    payload = {
        "ok": bool(page_payload.get("ok")) and not page_error and bool(matches),
        "provider": str(page_payload.get("provider") or _tool_provider_from_config("page_extract", cfg)).strip(),
        "url": str(page_payload.get("url") or normalized_url).strip(),
        "title": title,
        "question": normalized_question,
        "mode": normalized_mode,
        "sample_step": int(preview_meta.get("sample_step") or 0) if normalized_mode == "sample" else 0,
        "start_line": range_start if normalized_mode == "range" else 0,
        "end_line": range_end if normalized_mode == "range" else 0,
        "total_lines": int(preview_meta.get("total_lines") or 0),
        "total_chars": int(preview_meta.get("total_chars") or 0),
        "search": normalized_scope,
        "_matched_lines": matches,
        "count": len(matches),
        "page_error": page_error,
        "error": page_error,
        "from_cache": from_cache,
        "_model_markdown": _build_page_probe_markdown(
            url=str(page_payload.get("url") or normalized_url).strip(),
            title=title,
            scope=normalized_scope,
            matches=matches,
            preview_meta=preview_meta,
            page_error=page_error,
            from_cache=from_cache,
        ),
    }
    meta = _tool_meta(page_payload)
    if meta:
        payload["_meta"] = meta
    if page_error and not page_payload.get("ok"):
        payload["error"] = page_error
    return payload


def _navigate_url(
    *,
    url: str,
    cfg: dict[str, Any],
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    source_url = str(url or "").strip()
    if not source_url:
        return {"ok": False, "error": "navigate url is empty", "url": ""}

    normalized_url = _normalize_state_url(source_url) or source_url
    if runtime_state is not None and normalized_url:
        with runtime_state.lock:
            runtime_state.visited_page_urls.add(normalized_url)

    page_payload, from_cache = _load_page_payload(
        url=source_url,
        cfg=cfg,
        runtime_state=runtime_state,
        progress_callback=progress_callback,
    )
    title = str(page_payload.get("title") or "").strip()
    content = str(page_payload.get("content") or "").strip()
    page_error = str(page_payload.get("error") or "").strip()
    raw_lines = str(content or "").splitlines()
    count = sum(1 for line in raw_lines if str(line or "").strip())
    payload = {
        "ok": bool(page_payload.get("ok")) and not page_error and bool(count),
        "provider": str(page_payload.get("provider") or _tool_provider_from_config("navigate", cfg)).strip(),
        "url": str(page_payload.get("url") or normalized_url).strip(),
        "title": title,
        "count": count,
        "total_lines": len(raw_lines),
        "total_chars": len(content),
        "from_cache": from_cache,
        "page_error": page_error,
        "error": page_error,
        "_raw_lines": raw_lines,
    }
    if page_error:
        payload["_model_markdown"] = _build_page_probe_markdown(
            url=str(page_payload.get("url") or normalized_url).strip(),
            title=title,
            scope="",
            matches=[],
            page_error=page_error,
            from_cache=from_cache,
        )
    meta = _tool_meta(page_payload)
    if meta:
        payload["_meta"] = meta
    return payload


def _run_navigate(
    *,
    url: str,
    ref: str = "",
    search: str = "",
    search_filters: dict[str, Any] | None = None,
    cfg: dict[str, Any],
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    search_text = str(search or "").strip()
    ref_text = _normalize_context_ref(str(ref or "").strip())
    normalized_search_filters = _coerce_search_filters(search_filters)
    if search_text:
        search_label = _format_search_request_label(search_text, normalized_search_filters)
        cached_results = _search_results_for_request(
            runtime_state,
            query=search_text,
            filters=normalized_search_filters,
        )
        provider = ""
        meta: dict[str, Any] = {}
        page_error = ""
        from_cache = bool(cached_results)
        public_results = cached_results
        if not public_results:
            _search_history_contains(runtime_state, search_text, filters=normalized_search_filters)
            search_payload = _search_backend(
                search_text,
                kl=normalized_search_filters.get("kl", ""),
                df=normalized_search_filters.get("df", ""),
                t=normalized_search_filters.get("t", ""),
                ia=normalized_search_filters.get("ia", ""),
                config=cfg,
                progress_callback=progress_callback,
            )
            if not isinstance(search_payload, dict):
                search_payload = {"ok": False, "error": "invalid search payload", "results": []}
            rows = search_payload.get("results")
            deduped_rows = _merge_search_results(
                [],
                rows if isinstance(rows, list) else [],
                query=search_label,
            )
            public_results = _public_search_results(deduped_rows)
            if runtime_state is not None and public_results:
                with runtime_state.lock:
                    runtime_state.search_results_deduped = _merge_search_results(
                        runtime_state.search_results_deduped,
                        public_results,
                        query=search_label,
                    )
            provider = _tool_provider_from_payload(search_payload) or _provider_for_capability("search", cfg)
            meta = _tool_meta(search_payload)
            page_error = str(search_payload.get("error") or "").strip()
        else:
            provider = str(public_results[0].get("provider") or _provider_for_capability("search", cfg)).strip()

        raw_lines = build_search_open_lines(
            public_results,
            default_no_title_text=DEFAULT_NO_TITLE_TEXT,
        )
        filter_text = _format_search_filters(normalized_search_filters)
        title = f"Search: {search_text}"
        if filter_text:
            title += f" [{filter_text}]"
        payload = {
            "ok": bool(raw_lines) and not page_error,
            "provider": provider,
            "url": "",
            "title": title,
            "search": search_label,
            "filters": normalized_search_filters,
            "results": public_results,
            "count": len(raw_lines),
            "total_lines": len(raw_lines),
            "total_chars": len("\n".join(raw_lines)),
            "page_error": page_error,
            "error": page_error or (DEFAULT_NO_RESULTS_TEXT if not raw_lines else ""),
            "from_cache": from_cache,
            "_raw_lines": raw_lines,
            "_model_markdown": _build_search_payload_markdown(
                query=search_text,
                search_filters=normalized_search_filters,
                public_results=public_results,
                skipped_duplicate=False,
                error=page_error if page_error else (DEFAULT_NO_RESULTS_TEXT if not raw_lines else ""),
            ),
        }
        if meta:
            payload["_meta"] = meta
        return payload

    source_url = str(url or "").strip()
    if not source_url and ref_text and runtime_state is not None:
        with runtime_state.lock:
            source_url = str(runtime_state.ref_targets.get(ref_text) or "").strip()
    if not source_url:
        return {
            "ok": False,
            "error": f"navigate ref `{ref_text}` could not be resolved" if ref_text else "navigate requires `url`, `ref`, or `search`",
            "url": "",
        }

    normalized_url = _normalize_state_url(source_url) or source_url
    if runtime_state is not None and normalized_url:
        with runtime_state.lock:
            runtime_state.visited_page_urls.add(normalized_url)

    page_payload, from_cache = _load_page_payload(
        url=source_url,
        cfg=cfg,
        runtime_state=runtime_state,
        progress_callback=progress_callback,
    )
    title = str(page_payload.get("title") or "").strip()
    content = str(page_payload.get("content") or "")
    page_error = str(page_payload.get("error") or "").strip()
    raw_lines = str(content or "").splitlines()
    total_lines = len(raw_lines)
    total_chars = len(content)

    payload = {
        "ok": bool(page_payload.get("ok")) and not page_error and bool(raw_lines),
        "provider": str(page_payload.get("provider") or _tool_provider_from_config("navigate", cfg)).strip(),
        "url": str(page_payload.get("url") or normalized_url).strip(),
        "title": title,
        "count": sum(1 for line in raw_lines if str(line or "").strip()),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "page_error": page_error,
        "error": page_error,
        "from_cache": from_cache,
        "_raw_lines": raw_lines,
    }
    meta = _tool_meta(page_payload)
    if meta:
        payload["_meta"] = meta
    if page_error and not page_payload.get("ok"):
        payload["_model_markdown"] = _build_page_probe_markdown(
            url=str(page_payload.get("url") or normalized_url).strip(),
            title=title,
            scope="opened=all",
            matches=[],
            preview_meta={"total_lines": total_lines, "total_chars": total_chars, "shown_lines": 0, "sample_step": 0},
            page_error=page_error,
            from_cache=from_cache,
        )
    return payload


def _tool_timeout_payload(
    *,
    name: str,
    args: dict[str, Any],
    cfg: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    timeout_text = _format_timeout_seconds(elapsed_s)
    provider = _tool_provider_from_config(name, cfg)
    error = f"{name} batch timed out after {timeout_text}s"
    if name == "navigate":
        url = str(args.get("url") or "").strip()
        search = str(args.get("search") or "").strip()
        if search:
            search_filters = _coerce_search_filters(args)
            return {
                "ok": False,
                "provider": _provider_for_capability("search", cfg),
                "url": "",
                "title": f"Search: {search}",
                "search": _format_search_request_label(search, search_filters),
                "filters": search_filters,
                "count": 0,
                "total_lines": 0,
                "total_chars": 0,
                "page_error": error,
                "error": error,
                "timed_out": True,
                "_raw_lines": [],
                "_model_markdown": _build_search_payload_markdown(
                    query=search,
                    search_filters=search_filters,
                    public_results=[],
                    skipped_duplicate=False,
                    error=error,
                ),
            }
        return {
            "ok": False,
            "provider": provider,
            "url": url,
            "title": "",
            "count": 0,
            "total_lines": 0,
            "total_chars": 0,
            "page_error": error,
            "error": error,
            "timed_out": True,
            "_model_markdown": _build_page_probe_markdown(
                url=url,
                title="",
                matches=[],
                page_error=error,
                from_cache=False,
            ),
        }
    return {
        "ok": False,
        "error": error,
        "timed_out": True,
    }


def _record_runtime_warning(
    message: str,
    warning_messages: list[str] | None = None,
    *,
    on_status: Any | None = None,
) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    if warning_messages is not None and text not in warning_messages:
        warning_messages.append(text)
    logging.warning(text)
    if callable(on_status):
        try:
            on_status(text)
        except Exception:
            pass
    return text


def _prepend_runtime_warnings(message: str, warning_messages: list[str] | None = None) -> str:
    rows: list[str] = []
    for item in warning_messages or []:
        text = str(item or "").strip()
        if text and text not in rows:
            rows.append(text)
    final_message = str(message or "").strip()
    if final_message:
        rows.append(final_message)
    return "\n".join(rows).strip()


def _prepare_user_input_content(
    question: str,
    image_paths: list[str],
    *,
    config: dict[str, Any],
    stats: Stats | None = None,
    log_id: str | None = None,
    warning_messages: list[str] | None = None,
    on_status: Any | None = None,
) -> tuple[str | list[dict[str, Any]], str]:
    prompt_text = _effective_prompt_text(question, len(image_paths)) or "images"
    if not image_paths:
        return prompt_text, ""
    del config, stats, log_id, warning_messages, on_status
    return _build_multimodal_content(question, image_paths), ""


# ── LLM 调用 ─────────────────────────────────────────────────
def _to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, list):
        return list(obj)
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try: return fn()
            except Exception: pass
    return vars(obj) if hasattr(obj, "__dict__") else str(obj)


def _completion_extra_body(cfg: dict[str, Any]) -> dict[str, Any] | None:
    extra_body = cfg.get("extra_body")
    return deepcopy(extra_body) if isinstance(extra_body, dict) else None


def _completion_api_key(cfg: dict[str, Any]) -> str:
    api_key = str(cfg.get("api_key") or "").strip()
    if api_key.startswith("os.environ/"):
        env_name = api_key.partition("/")[2].strip()
        if env_name:
            return str(os.getenv(env_name) or "").strip()
    if api_key:
        return api_key
    env_name = str(cfg.get("api_key_env") or "").strip()
    if env_name:
        return str(os.getenv(env_name) or "").strip()
    return ""


def _apply_completion_transport_options(cfg: dict[str, Any], kw: dict[str, Any]) -> None:
    api_base = str(cfg.get("api_base") or "").strip()
    if api_base:
        kw["api_base"] = api_base
    api_key = _completion_api_key(cfg)
    if api_key:
        kw["api_key"] = api_key
    custom_llm_provider = str(cfg.get("custom_llm_provider") or "").strip()
    if custom_llm_provider:
        kw["custom_llm_provider"] = custom_llm_provider


def _normalized_reasoning_effort(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("reasoning_effort") or "").strip().lower()
    return value if value in ("minimal", "low", "medium", "high", "xhigh", "none") else ""


def _should_use_openrouter_reasoning(cfg: dict[str, Any]) -> bool:
    model = str(cfg.get("model") or "").strip().lower()
    api_base = str(cfg.get("api_base") or "").strip().lower()
    return model.startswith("openrouter/") or "openrouter.ai" in api_base


def _apply_reasoning_options(cfg: dict[str, Any], kw: dict[str, Any]) -> None:
    effort = _normalized_reasoning_effort(cfg)
    if not effort:
        return

    kw["reasoning_effort"] = effort
    if not _should_use_openrouter_reasoning(cfg):
        return

    extra_body = kw.get("extra_body")
    merged = deepcopy(extra_body) if isinstance(extra_body, dict) else {}
    reasoning = merged.get("reasoning")
    reasoning_cfg = deepcopy(reasoning) if isinstance(reasoning, dict) else {}
    reasoning_cfg.setdefault("effort", effort)
    reasoning_cfg.setdefault("exclude", False)
    merged["reasoning"] = reasoning_cfg
    kw["extra_body"] = merged


def _usage_cost(usage: dict[str, Any]) -> float | None:
    if not isinstance(usage, dict):
        return None
    raw = usage.get("cost")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except Exception:
            return None
    return None


def _usage_reasoning_tokens(usage: dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return max(0, int(details.get("reasoning_tokens") or 0))
    except Exception:
        return 0


def _estimated_completion_tokens_for_cost(usage: dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    reasoning_tokens = _usage_reasoning_tokens(usage)
    if reasoning_tokens <= 0:
        return completion_tokens
    if completion_tokens >= reasoning_tokens:
        return completion_tokens
    return completion_tokens + reasoning_tokens


def _estimated_usage_cost(model: str, usage: dict[str, Any]) -> float | None:
    if not isinstance(usage, dict):
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = _estimated_completion_tokens_for_cost(usage)
    if not (prompt_tokens or completion_tokens):
        return None
    return _try_cost(model, prompt_tokens, completion_tokens)


def _estimate_text_tokens(model: str, text: str) -> int:
    body = str(text or "").strip()
    if not body:
        return 0
    litellm_mod = _get_litellm()
    token_counter = getattr(litellm_mod, "token_counter", None)
    if callable(token_counter):
        try:
            counted = token_counter(model=model, messages=[{"role": "assistant", "content": body}])
            return max(0, int(counted or 0))
        except Exception:
            pass
    return max(1, len(body) // 4)


def _reasoning_details_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    details = payload.get("reasoning_details")
    if isinstance(details, list):
        return [dict(item) for item in details if isinstance(item, dict)]
    provider_fields = payload.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        nested = provider_fields.get("reasoning_details")
        if isinstance(nested, list):
            return [dict(item) for item in nested if isinstance(item, dict)]
    return []


def _reasoning_text_from_payload(payload: dict[str, Any]) -> str:
    direct = str(payload.get("reasoning") or payload.get("reasoning_content") or "").strip()
    if direct:
        return direct
    provider_fields = payload.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        direct = str(provider_fields.get("reasoning") or provider_fields.get("reasoning_content") or "").strip()
        if direct:
            return direct
    return ""


def _render_reasoning_display(
    *,
    payload: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    reasoning_text_parts: list[str] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    merged_payload = payload if isinstance(payload, dict) else {}
    details = list(reasoning_details or _reasoning_details_from_payload(merged_payload))
    text_parts: list[str] = []
    direct = _reasoning_text_from_payload(merged_payload)
    if direct:
        text_parts.append(direct)
    for item in reasoning_text_parts or []:
        text = str(item or "").strip()
        if text:
            text_parts.append(text)

    encrypted_blocks = 0
    for detail in details:
        kind = str(detail.get("type") or "").strip().lower()
        if kind == "reasoning.encrypted" or (kind == "reasoning" and str(detail.get("encrypted_content") or "").strip()):
            encrypted_blocks += 1
            continue
        if kind == "reasoning.text":
            text = str(detail.get("text") or "").strip()
            if text:
                text_parts.append(text)
            continue
        summary = detail.get("summary")
        if isinstance(summary, list):
            for entry in summary:
                if isinstance(entry, dict):
                    text = str(entry.get("text") or "").strip()
                else:
                    text = str(entry or "").strip()
                if text:
                    text_parts.append(text)
            continue
        text = str(detail.get("text") or "").strip()
        if text:
            text_parts.append(text)

    deduped_parts: list[str] = []
    seen_parts: set[str] = set()
    for item in text_parts:
        normalized = re.sub(r"\s+", " ", str(item or "").strip())
        if not normalized or normalized in seen_parts:
            continue
        seen_parts.add(normalized)
        deduped_parts.append(str(item).strip())

    reasoning_tokens = _usage_reasoning_tokens(usage or {})
    if reasoning_tokens <= 0 and encrypted_blocks > 0:
        reasoning_tokens = encrypted_blocks

    meta = {
        "reasoning_tokens": reasoning_tokens,
        "encrypted_blocks": encrypted_blocks,
        "detail_count": len(details),
        "plaintext_available": bool(deduped_parts),
    }
    return "\n\n".join(deduped_parts).strip(), meta


def _assistant_message_with_reasoning(
    msg: Any,
    *,
    content: str,
    reasoning_text_parts: list[str] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    payload = _to_dict(msg) if msg is not None else {}
    if not isinstance(payload, dict):
        payload = {}
    direct = _reasoning_text_from_payload(payload)
    if not direct and reasoning_text_parts:
        direct = "\n\n".join(str(item).strip() for item in reasoning_text_parts if str(item).strip()).strip()
    if direct:
        assistant_message["reasoning"] = direct
    raw_details = reasoning_details or _reasoning_details_from_payload(payload)
    details = [
        dict(item)
        for item in raw_details
        if isinstance(item, dict) and str(item.get("type") or "").strip().lower() != "reasoning.encrypted"
    ]
    if details:
        assistant_message["reasoning_details"] = details
    return assistant_message


def _apply_completion_limits(cfg: dict[str, Any], kw: dict[str, Any]) -> None:
    max_completion_tokens = cfg.get("max_completion_tokens")
    if isinstance(max_completion_tokens, int) and max_completion_tokens > 0:
        kw["max_completion_tokens"] = max_completion_tokens
        return

    max_tokens = cfg.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        kw["max_tokens"] = max_tokens


def llm_call(
    messages,
    *,
    config,
    stats=None,
    trace_label="Model",
    log_id=None,
    phase: str = "loop",
    disclosure_step: int | None = None,
    log_turn: int | None = None,
    log_title: str | None = None,
    on_status: Any | None = None,
):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    # SDK retries stay disabled; the app performs model-only retries outside tool execution.
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "drop_params": True, "max_retries": 0}
    tool_cfg = litellm_tool_config_for_phase(phase, disclosure_step=disclosure_step)
    _apply_completion_transport_options(cfg, kw)
    _apply_completion_limits(cfg, kw)
    for key, value in tool_cfg.items():
        if key == "tools" and not value:
            continue
        kw[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    extra_body = _completion_extra_body(cfg)
    if extra_body:
        kw["extra_body"] = extra_body
    _apply_reasoning_options(cfg, kw)

    t0 = time.perf_counter()
    if should_use_codex_mirror_transport(cfg):
        def _invoke() -> Any:
            return codex_transport_model_response(
                codex_transport_stream_response(
                    cfg=cfg,
                    messages=messages,
                    tools=tool_cfg.get("tools") if isinstance(tool_cfg.get("tools"), list) else [],
                )
            )
        resp = _call_model_with_retries(
            _invoke,
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    else:
        litellm_mod = _get_litellm()
        resp = _call_model_with_retries(
            lambda: litellm_mod.completion(**kw),
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    duration_ms = (time.perf_counter() - t0) * 1000

    # 提取 usage / cost
    usage: dict[str, Any] = {}
    cost: float | None = None
    u_raw = getattr(resp, "usage", None)
    usage = _to_dict(u_raw) if u_raw else {}
    if not isinstance(usage, dict): usage = {}
    pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cost = _usage_cost(usage)
    if cost is None and (pt or ct or _usage_reasoning_tokens(usage)):
        cost = _estimated_usage_cost(model, usage)
    if stats:
        stats.record(usage, cost)

    # 提取输出内容
    choices = getattr(resp, "choices", None) or []
    msg = choices[0].message if choices else None
    output_text = _text(msg) if msg else ""
    raw_tool_calls = _extract_raw_tool_calls(_to_dict(msg) if msg is not None else {})

    # 写日志
    log_model_call(
        label=trace_label, model=model, messages=messages,
        output=output_text,
        tool_calls=raw_tool_calls,
        turn=log_turn,
        title=log_title,
        usage=usage, cost=cost, duration_ms=duration_ms, config=cfg,
        log_id=log_id,
    )

    return resp


def llm_text_call(
    messages,
    *,
    config,
    stats=None,
    trace_label="Sub-agent Text",
    log_id=None,
    temperature: float = 0.1,
    on_status: Any | None = None,
):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "drop_params": True,
        "max_retries": 0,
    }
    _apply_completion_transport_options(cfg, kw)
    _apply_completion_limits(cfg, kw)
    extra_body = _completion_extra_body(cfg)
    if extra_body:
        kw["extra_body"] = extra_body
    _apply_reasoning_options(cfg, kw)

    t0 = time.perf_counter()
    if should_use_codex_mirror_transport(cfg):
        def _invoke() -> Any:
            return codex_transport_model_response(
                codex_transport_stream_response(
                    cfg=cfg,
                    messages=messages,
                    tools=[],
                )
            )
        resp = _call_model_with_retries(
            _invoke,
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    else:
        litellm_mod = _get_litellm()
        resp = _call_model_with_retries(
            lambda: litellm_mod.completion(**kw),
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    duration_ms = (time.perf_counter() - t0) * 1000

    usage: dict[str, Any] = {}
    cost: float | None = None
    u_raw = getattr(resp, "usage", None)
    usage = _to_dict(u_raw) if u_raw else {}
    if not isinstance(usage, dict):
        usage = {}
    pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cost = _usage_cost(usage)
    if cost is None and (pt or ct or _usage_reasoning_tokens(usage)):
        cost = _estimated_usage_cost(model, usage)
    if stats:
        stats.record(usage, cost)

    choices = getattr(resp, "choices", None) or []
    msg = choices[0].message if choices else None
    output_text = _text(msg) if msg else ""
    log_model_call(
        label=trace_label,
        model=model,
        messages=messages,
        output=output_text,
        usage=usage,
        cost=cost,
        duration_ms=duration_ms,
        config=cfg,
        log_id=log_id,
    )
    return resp


def llm_custom_tool_call(
    messages,
    *,
    config,
    tools: list[dict[str, Any]],
    stats=None,
    trace_label="Sub-agent Tool",
    log_id=None,
    temperature: float = 0.1,
    tool_choice: str = "auto",
    parallel_tool_calls: bool = False,
    on_status: Any | None = None,
):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "drop_params": True,
        "max_retries": 0,
        "tools": deepcopy(tools),
        "tool_choice": tool_choice,
        "parallel_tool_calls": bool(parallel_tool_calls),
    }
    _apply_completion_transport_options(cfg, kw)
    _apply_completion_limits(cfg, kw)
    extra_body = _completion_extra_body(cfg)
    if extra_body:
        kw["extra_body"] = extra_body
    _apply_reasoning_options(cfg, kw)

    t0 = time.perf_counter()
    if should_use_codex_mirror_transport(cfg):
        def _invoke() -> Any:
            return codex_transport_model_response(
                codex_transport_stream_response(
                    cfg=cfg,
                    messages=messages,
                    tools=tools,
                )
            )
        resp = _call_model_with_retries(
            _invoke,
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    else:
        litellm_mod = _get_litellm()
        resp = _call_model_with_retries(
            lambda: litellm_mod.completion(**kw),
            cfg=cfg,
            model=model,
            messages=messages,
            trace_label=trace_label,
            log_id=log_id,
            on_status=on_status,
        )
    duration_ms = (time.perf_counter() - t0) * 1000

    usage: dict[str, Any] = {}
    cost: float | None = None
    u_raw = getattr(resp, "usage", None)
    usage = _to_dict(u_raw) if u_raw else {}
    if not isinstance(usage, dict):
        usage = {}
    pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cost = _usage_cost(usage)
    if cost is None and (pt or ct or _usage_reasoning_tokens(usage)):
        cost = _estimated_usage_cost(model, usage)
    if stats:
        stats.record(usage, cost)

    choices = getattr(resp, "choices", None) or []
    msg = choices[0].message if choices else None
    output_text = _text(msg) if msg else ""
    log_model_call(
        label=trace_label,
        model=model,
        messages=messages,
        output=output_text,
        usage=usage,
        cost=cost,
        duration_ms=duration_ms,
        config=cfg,
        log_id=log_id,
    )
    return resp


# ── 对话循环 ──────────────────────────────────────────────────
def _text(msg) -> str:
    c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(c, str): return c.strip()
    if isinstance(c, list):
        return "\n".join(str(getattr(p, "text", "") or (p.get("text") if isinstance(p, dict) else "") or "") for p in c).strip()
    return ""


def _select_loop_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_open_targets: set[str] = set()
    allowed_tools = {"navigate"}

    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name not in allowed_tools or not isinstance(args, dict):
            continue

        search = str(args.get("search") or "").strip()
        ref = _normalize_context_ref(str(args.get("ref") or "").strip())
        normalized_url = _normalize_state_url(str(args.get("url") or "").strip())
        normalized_keep = _normalize_keep_argument(args.get("keep"))
        target_key = ""
        payload: dict[str, Any] = {}

        if search:
            search_filters = _coerce_search_filters(args)
            target_key = "search::" + normalize_search_request_key(search, search_filters)
            if not target_key or target_key in seen_open_targets:
                continue
            payload["search"] = search
            payload.update(search_filters)
        elif normalized_url:
            target_key = "url::" + normalized_url
            if target_key in seen_open_targets:
                continue
            payload["url"] = normalized_url
            if ref:
                payload["ref"] = ref
        elif ref:
            target_key = "ref::" + ref
            if target_key in seen_open_targets:
                continue
            payload["ref"] = ref
        else:
            continue

        if normalized_keep is not None:
            payload["keep"] = normalized_keep

        seen_open_targets.add(target_key)
        if str(args.get("_call_id") or "").strip():
            payload["_call_id"] = str(args.get("_call_id") or "").strip()
        selected.append({"name": "navigate", "args": payload})
        if len(selected) >= _TOOL_LIMIT:
            break

    return selected


def _collect_loop_calls(
    *,
    text: str,
    message_payload: dict[str, Any],
    state: _PhaseRuntimeState,
    round_i: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_calls, invalid_errors = _parse_native_tool_calls_detailed(message_payload)
    _resolve_page_call_refs(raw_calls, state)
    calls = _select_loop_calls(raw_calls)
    _ensure_call_ids(calls, round_i=round_i)
    return calls, invalid_errors


def _ensure_call_ids(calls: list[dict[str, Any]], *, round_i: int) -> None:
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else None
        if args is None:
            continue
        if str(args.get("_call_id") or "").strip():
            continue
        args["_call_id"] = f"r{round_i + 1}c{index}"


def _execute_loop_calls(
    *,
    calls: list[dict[str, Any]],
    round_i: int,
    cfg: dict[str, Any],
    stats: Stats | None,
    prompt_text: str,
    log_id: str | None,
    runtime_state: _SessionRuntimeState | None,
    context_state: _PhaseRuntimeState,
    started_at: float,
    on_tool: Any | None = None,
    progress_callback: Any | None = None,
    stop_checker: Any | None = None,
) -> dict[int, dict[str, Any]]:
    payload_map: dict[int, dict[str, Any]] = {}
    note_parts: list[str] = []
    external_calls: list[tuple[int, dict[str, Any]]] = []
    batch_created_ids: list[int] = []

    def _emit_tool_progress_event(name: str, args: dict[str, Any], event: dict[str, Any], *, started_at_s: float) -> None:
        if not callable(on_tool):
            return
        merged = dict(args)
        if isinstance(event, dict):
            status = str(event.get("status") or "").strip().lower()
            if status:
                merged["_progress_status"] = status
                # Progress events come from nested provider work; the outer tool is
                # still running until the final payload is emitted below.
                merged["_pending"] = True
            if "query" in event and not merged.get("query"):
                merged["query"] = event.get("query")
            if "url" in event and not merged.get("url"):
                merged["url"] = event.get("url")
            if "error" in event:
                merged["_error"] = event.get("error")
        merged["_elapsed_s"] = max(0.0, time.perf_counter() - started_at_s)
        try:
            on_tool(name, merged)
        except Exception:
            pass

    for index, call in enumerate(calls):
        _raise_if_stop_requested(stop_checker)
        external_calls.append((index, call))

    if external_calls:
        _raise_if_stop_requested(stop_checker)
        futures: dict[Any, tuple[int, dict[str, Any], float]] = {}
        external_payloads: dict[int, tuple[dict[str, Any], float, float, dict[str, Any]]] = {}
        tool_timeout_s = _resolve_external_tool_timeout_s(cfg)
        pool = ThreadPoolExecutor(max_workers=len(external_calls))
        try:
            for index, call in external_calls:
                _raise_if_stop_requested(stop_checker)
                name = str(call.get("name") or "").strip()
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                if not isinstance(args, dict):
                    args = {}
                call_started_at = time.perf_counter()
                wrapped_progress = None
                if callable(on_tool) or callable(progress_callback):
                    def _callback(event: dict[str, Any], *, _name=name, _args=dict(args), _started=call_started_at) -> None:
                        if callable(progress_callback):
                            try:
                                progress_callback(event)
                            except Exception:
                                pass
                        _emit_tool_progress_event(_name, _args, event if isinstance(event, dict) else {}, started_at_s=_started)
                    wrapped_progress = _callback
                futures[
                    pool.submit(
                        execute_tool_payload,
                        name,
                        args,
                        config=cfg,
                        stats=stats,
                        user_question=prompt_text,
                        log_id=log_id,
                        runtime_state=runtime_state,
                        progress_callback=wrapped_progress,
                    )
                ] = (index, call, call_started_at)
            pending = set(futures)
            while pending:
                _raise_if_stop_requested(stop_checker)
                done, not_done = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                now = time.perf_counter()

                for future in done:
                    pending.discard(future)
                    index, call, call_started_at = futures[future]
                    try:
                        payload = future.result()
                    except StopRequestedError:
                        raise
                    except Exception as e:
                        payload = {
                            "ok": False,
                            "error": f"tool execution failed: {str(e).strip() or type(e).__name__}",
                        }
                    if not isinstance(payload, dict):
                        payload = {"ok": False, "error": "invalid tool payload"}
                    completed_elapsed_s = max(0.0, now - call_started_at)
                    external_payloads[index] = (payload, call_started_at, completed_elapsed_s, call)

                for future in list(not_done):
                    index, call, call_started_at = futures[future]
                    elapsed_s = max(0.0, now - call_started_at)
                    if elapsed_s < tool_timeout_s:
                        continue
                    pending.discard(future)
                    future.cancel()
                    name = str(call.get("name") or "").strip()
                    args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    if not isinstance(args, dict):
                        args = {}
                    payload = _tool_timeout_payload(
                        name=name,
                        args=args,
                        cfg=cfg,
                        elapsed_s=elapsed_s,
                    )
                    external_payloads[index] = (payload, call_started_at, elapsed_s, call)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        for index, call in external_calls:
            _raise_if_stop_requested(stop_checker)
            payload, call_started_at, completed_elapsed_s, original_call = external_payloads.get(
                index,
                ({"ok": False, "error": "missing tool payload"}, time.perf_counter(), 0.0, call),
            )
            name = str(original_call.get("name") or "").strip()
            args = original_call.get("args") if isinstance(original_call.get("args"), dict) else {}
            if not isinstance(args, dict):
                args = {}
            if name == "navigate" and payload.get("ok"):
                replaced_ids, created_ids = _replace_active_pages(
                    args=args,
                    state=context_state,
                    round_no=round_i + 1,
                )
                if replaced_ids:
                    payload["_replaced_ids"] = replaced_ids
                    payload["_replaced_count"] = len(replaced_ids)
                if created_ids:
                    payload["_created_ids"] = created_ids
                    payload["_created_count"] = len(created_ids)
                    batch_created_ids.extend(created_ids)
                    for created_id in created_ids:
                        _remember_context_item_ref_targets(runtime_state, _find_context_item(context_state, int(created_id)))
                active_page_id = _set_active_page_item(
                    context_state,
                    payload,
                    round_no=round_i + 1,
                    call_id=str(args.get("_call_id") or "").strip(),
                )
                _remember_context_item_ref_targets(runtime_state, _find_context_item(context_state, int(active_page_id)))
            _record_tool_stats(stats, payload)
            payload_map[index] = payload
            log_tool_call(
                name=name,
                args=args,
                payload=payload,
                duration_ms=completed_elapsed_s * 1000.0,
                turn=round_i + 1,
                config=cfg,
                log_id=log_id,
            )
            if callable(on_tool):
                try:
                    on_tool(
                        name,
                        _tool_callback_args(
                            name,
                            args,
                            payload,
                            elapsed_s=completed_elapsed_s,
                            config=cfg,
                        ),
                    )
                except Exception:
                    pass
    context_state.latest_raw_ids = list(batch_created_ids)
    context_state.latest_raw_round = int(round_i + 1) if batch_created_ids else 0
    context_state.pending_user_note = "\n\n".join(part for part in note_parts if part).strip()
    return payload_map


def _validate_progressive_tool_requirements(
    state: _PhaseRuntimeState,
    calls: list[dict[str, Any]],
) -> str:
    # Intentionally disabled for now: rely on prompt-level control rather than
    # hard rejection at the tool-validation layer.
    del state, calls
    return ""
    return ""


def _advance_progressive_disclosure(
    state: _PhaseRuntimeState,
    calls: list[dict[str, Any]],
) -> None:
    tool_names = {str(call.get("name") or "").strip() for call in calls if isinstance(call, dict)}
    if state.disclosure_step <= 0 and "navigate" in tool_names:
        state.disclosure_step = 1
        state.disclosure_refine_consumed = False
        state.pending_user_note = ""
        return
    if state.disclosure_step == 1 and state.disclosure_refine_consumed:
        state.disclosure_step = 2


def run(
    question: str,
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    on_tool: Any | None = None,
    on_reasoning: Any | None = None,
    images: list[str] | None = None,
    context: str | list[dict[str, Any]] | None = None,
    stop_checker: Any | None = None,
) -> str:
    """单次问答: 正常多轮循环 + LiteLLM 原生工具调用 + 可删除的累积上下文."""
    image_paths = [str(path).strip() for path in (images or []) if str(path).strip()]
    if not question.strip() and not image_paths:
        return ""
    cfg = build_model_config(config or load_config())
    st = stats or Stats()
    max_rounds = _resolve_max_rounds(cfg)
    prompt_text = _effective_prompt_text(question, len(image_paths)) or "images"
    lid = _make_log_id(prompt_text)
    started_at = time.perf_counter()
    warning_messages: list[str] = []
    runtime_state = _SessionRuntimeState()
    loop_state = _PhaseRuntimeState()
    loop_state.input_has_images = bool(image_paths)
    user_content, input_error = _prepare_user_input_content(
        question,
        image_paths,
        config=cfg,
        stats=st,
        log_id=lid,
        warning_messages=warning_messages,
    )
    if input_error:
        return input_error

    history: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    last = ""

    for round_i in range(max_rounds):
        _raise_if_stop_requested(stop_checker)
        msgs = _build_loop_messages(
            cfg=cfg,
            prompt_text=prompt_text,
            history=history,
            state=loop_state,
            round_i=round_i,
            context=context,
        )
        loop_state.pending_user_note = ""
        try:
            resp = llm_call(
                msgs,
                config=cfg,
                stats=st,
                trace_label=f"round {round_i + 1}",
                log_id=lid,
                phase="loop",
                disclosure_step=loop_state.disclosure_step,
                log_turn=round_i + 1,
                log_title=prompt_text,
            )
        except Exception as e:
            return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)

        choices = getattr(resp, "choices", None) or []
        msg = choices[0].message if choices else None
        if msg is None:
            break
        msg_payload = _to_dict(msg)
        if not isinstance(msg_payload, dict):
            msg_payload = {}
        usage = _to_dict(getattr(resp, "usage", None) or {})
        if not isinstance(usage, dict):
            usage = {}

        text = _text(msg)
        if text:
            last = text

        calls, invalid_tool_errors = _collect_loop_calls(
            text=text,
            message_payload=msg_payload,
            state=loop_state,
            round_i=round_i,
        )
        reasoning_text, reasoning_meta = _render_reasoning_display(
            payload=msg_payload,
            usage=usage,
        )
        if callable(on_reasoning):
            try:
                on_reasoning(reasoning_text, reasoning_meta)
            except Exception:
                pass
        assistant_content = _normalize_tool_turn_message(text, calls) if calls else _clean_model_text(text)
        invalid_tool_message = _format_invalid_tool_call_message(invalid_tool_errors)
        if invalid_tool_message:
            loop_state.pending_user_note = invalid_tool_message
            continue

        progressive_error = _validate_progressive_tool_requirements(loop_state, calls)
        if progressive_error:
            loop_state.pending_user_note = progressive_error
            continue

        if assistant_content.strip():
            history.append({"role": "assistant", "content": assistant_content})

        if not calls:
            _raise_if_stop_requested(stop_checker)
            final_message = _clean_model_text(text) or last or _format_empty_output_message(cfg, round_i=round_i)
            return _prepend_runtime_warnings(final_message, warning_messages)

        log_tool_selection(
            round_i=round_i,
            calls=calls,
            config=cfg,
            log_id=lid,
        )
        _raise_if_stop_requested(stop_checker)
        _execute_loop_calls(
            calls=calls,
            round_i=round_i,
            cfg=cfg,
            stats=st,
            prompt_text=prompt_text,
            log_id=lid,
            runtime_state=runtime_state,
            context_state=loop_state,
            started_at=started_at,
            on_tool=on_tool,
            progress_callback=None,
            stop_checker=stop_checker,
        )
        _advance_progressive_disclosure(loop_state, calls)

    final_message = _clean_model_text(last) or _format_empty_output_message(cfg, round_i=max_rounds - 1)
    _raise_if_stop_requested(stop_checker)
    return _prepend_runtime_warnings(final_message, warning_messages)


# ── 流式对话循环 ──────────────────────────────────────────────
def run_stream(
    question: str,
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    on_chunk: Any | None = None,
    on_tool: Any | None = None,
    on_reasoning: Any | None = None,
    on_status: Any | None = None,
    on_rewind: Any | None = None,
    images: list[str] | None = None,
    context: str | list[dict[str, Any]] | None = None,
    stop_checker: Any | None = None,
) -> str:
    """流式对话循环: 正常多轮循环 + LiteLLM 原生工具调用 + 可删除的累积上下文."""
    image_paths = [str(path).strip() for path in (images or []) if str(path).strip()]
    if not question.strip() and not image_paths:
        return ""
    cfg = build_model_config(config or load_config())
    st = stats or Stats()
    max_rounds = _resolve_max_rounds(cfg)
    prompt_text = _effective_prompt_text(question, len(image_paths)) or "images"
    lid = _make_log_id(prompt_text)
    started_at = time.perf_counter()
    warning_messages: list[str] = []
    runtime_state = _SessionRuntimeState()
    loop_state = _PhaseRuntimeState()
    loop_state.input_has_images = bool(image_paths)
    user_content, input_error = _prepare_user_input_content(
        question,
        image_paths,
        config=cfg,
        stats=st,
        log_id=lid,
        warning_messages=warning_messages,
        on_status=on_status,
    )
    if input_error:
        return input_error

    history: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    last = ""

    for round_i in range(max_rounds):
        _raise_if_stop_requested(stop_checker)
        msgs = _build_loop_messages(
            cfg=cfg,
            prompt_text=prompt_text,
            history=history,
            state=loop_state,
            round_i=round_i,
            context=context,
        )
        loop_state.pending_user_note = ""
        if callable(on_status):
            on_status(STATUS_THINKING)

        model = str(cfg.get("model") or DEFAULT_MODEL).strip()
        tool_cfg = litellm_tool_config_for_phase("loop", disclosure_step=loop_state.disclosure_step)
        t0 = time.perf_counter()
        retry_count = _model_retry_count(cfg)
        retry_index = 0
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        reasoning_text_parts: list[str] = []
        reasoning_details: list[dict[str, Any]] = []
        stream_tool_calls: dict[int, dict[str, Any]] = {}
        msg_payload: dict[str, Any] = {}

        while True:
            content_parts = []
            usage = {}
            reasoning_text_parts = []
            reasoning_details = []
            stream_tool_calls = {}
            msg_payload = {}
            last_reasoning_emit_tokens = [0]
            attempt_started_at = time.perf_counter()
            try:
                if should_use_codex_mirror_transport(cfg):
                    def _handle_codex_text_delta(delta: str) -> None:
                        _raise_if_stop_requested(stop_checker)
                        content_parts.append(delta)
                        if callable(on_chunk):
                            on_chunk(delta)

                    response_payload = codex_transport_stream_response(
                        cfg=cfg,
                        messages=msgs,
                        tools=tool_cfg.get("tools") if isinstance(tool_cfg.get("tools"), list) else [],
                        on_text_delta=_handle_codex_text_delta,
                    )
                    usage = _to_dict(response_payload.get("usage") or {})
                    msg_payload = codex_transport_message_payload(response_payload)
                    reasoning_details = [
                        dict(item)
                        for item in response_payload.get("output") or []
                        if isinstance(item, dict) and str(item.get("type") or "").strip() == "reasoning"
                    ]
                else:
                    litellm_mod = _get_litellm(on_status=on_status)
                    kw: dict[str, Any] = {
                        "model": model, "messages": msgs, "temperature": 0.2,
                        "stream": True, "drop_params": True,
                        "stream_options": {"include_usage": True},
                        "max_retries": 0,
                    }
                    _apply_completion_transport_options(cfg, kw)
                    _apply_completion_limits(cfg, kw)
                    for key, value in tool_cfg.items():
                        if key == "tools" and not value:
                            continue
                        kw[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
                    extra_body = _completion_extra_body(cfg)
                    if extra_body:
                        kw["extra_body"] = extra_body
                    _apply_reasoning_options(cfg, kw)

                    stream = litellm_mod.completion(**kw)
                    for chunk in stream:
                        _raise_if_stop_requested(stop_checker)
                        u = getattr(chunk, "usage", None)
                        if u:
                            usage = _to_dict(u) if not isinstance(u, dict) else u

                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if not delta:
                            continue
                        delta_payload = _to_dict(delta)
                        if isinstance(delta_payload, dict):
                            _merge_stream_tool_call_deltas(stream_tool_calls, delta_payload)
                            delta_reasoning_text, _ = _render_reasoning_display(payload=delta_payload)
                            if delta_reasoning_text:
                                reasoning_text_parts.append(delta_reasoning_text)
                                if callable(on_reasoning):
                                    estimated_tokens = _estimate_text_tokens(model, "\n\n".join(reasoning_text_parts))
                                    if estimated_tokens > last_reasoning_emit_tokens[0]:
                                        last_reasoning_emit_tokens[0] = estimated_tokens
                                        try:
                                            on_reasoning(
                                                "",
                                                {
                                                    "reasoning_tokens": estimated_tokens,
                                                    "streaming": True,
                                                },
                                            )
                                        except Exception:
                                            pass
                            for detail in _reasoning_details_from_payload(delta_payload):
                                reasoning_details.append(detail)

                        c = getattr(delta, "content", None)
                        if c:
                            _raise_if_stop_requested(stop_checker)
                            content_parts.append(c)
                            if callable(on_chunk):
                                try:
                                    on_chunk(c)
                                except Exception:
                                    pass
                    msg_payload = {"tool_calls": [stream_tool_calls[index] for index in sorted(stream_tool_calls)]} if stream_tool_calls else {}
                break
            except StopRequestedError:
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - attempt_started_at) * 1000
                partial = "".join(content_parts)
                log_model_call(
                    label=f"round {round_i + 1}",
                    model=model,
                    messages=msgs,
                    output=partial,
                    tool_calls=_extract_raw_tool_calls({"tool_calls": [stream_tool_calls[index] for index in sorted(stream_tool_calls)]}),
                    turn=round_i + 1,
                    title=prompt_text,
                    error=str(e)[:300],
                    duration_ms=duration_ms,
                    config=cfg,
                    log_id=lid,
                )
                can_retry = (
                    retry_index < retry_count
                    and _is_retryable_model_error(e)
                    and not partial.strip()
                    and not stream_tool_calls
                )
                if not can_retry:
                    return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)
                retry_index += 1
                delay_s = _model_retry_delay_s(cfg, retry_index)
                if callable(on_status):
                    on_status(_model_retry_status_text(retry_index, retry_count, delay_s))
                time.sleep(delay_s)
                if callable(on_status):
                    on_status(STATUS_THINKING)
                continue

        duration_ms = (time.perf_counter() - t0) * 1000
        full_text = "".join(content_parts)
        payload_text = _text(msg_payload) if isinstance(msg_payload, dict) else ""
        effective_text = full_text or payload_text
        if effective_text:
            last = effective_text

        # Stats & cost
        if not isinstance(usage, dict):
            usage = {}
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost = _usage_cost(usage)
        if cost is None and (pt or ct or _usage_reasoning_tokens(usage)):
            cost = _estimated_usage_cost(model, usage)
        if st:
            st.record(usage, cost)

        raw_tool_calls = _extract_raw_tool_calls(
            {"tool_calls": [stream_tool_calls[index] for index in sorted(stream_tool_calls)]}
            if stream_tool_calls
            else msg_payload
        )
        logged_text = effective_text
        log_output = logged_text

        # Log
        log_model_call(
            label=f"round {round_i + 1}", model=model, messages=msgs,
            output=log_output,
            tool_calls=raw_tool_calls,
            turn=round_i + 1,
            title=prompt_text,
            usage=usage, cost=cost, duration_ms=duration_ms, config=cfg,
            log_id=lid,
        )

        reasoning_text, reasoning_meta = _render_reasoning_display(
            usage=usage,
            reasoning_text_parts=reasoning_text_parts,
            reasoning_details=reasoning_details,
        )
        if callable(on_reasoning):
            try:
                on_reasoning(reasoning_text, reasoning_meta)
            except Exception:
                pass

        calls, invalid_tool_errors = _collect_loop_calls(
            text=effective_text,
            message_payload=msg_payload,
            state=loop_state,
            round_i=round_i,
        )
        assistant_content = _normalize_tool_turn_message(effective_text, calls) if calls else _clean_model_text(effective_text)
        invalid_tool_message = _format_invalid_tool_call_message(invalid_tool_errors)
        if invalid_tool_message:
            if callable(on_rewind):
                try:
                    on_rewind(assistant_content or effective_text, None)
                except Exception:
                    pass
            loop_state.pending_user_note = invalid_tool_message
            continue

        progressive_error = _validate_progressive_tool_requirements(loop_state, calls)
        if progressive_error:
            if callable(on_rewind):
                try:
                    on_rewind(assistant_content or effective_text, None)
                except Exception:
                    pass
            loop_state.pending_user_note = progressive_error
            continue

        if assistant_content.strip():
            history.append({"role": "assistant", "content": assistant_content})

        if not calls:
            _raise_if_stop_requested(stop_checker)
            final_message = _clean_model_text(effective_text) or last or _format_empty_output_message(cfg, round_i=round_i)
            return _prepend_runtime_warnings(final_message, warning_messages)

        if callable(on_rewind):
            try:
                on_rewind(assistant_content or effective_text, None)
            except Exception:
                pass

        if callable(on_status):
            on_status(STATUS_SEARCHING)
        log_tool_selection(
            round_i=round_i,
            calls=calls,
            config=cfg,
            log_id=lid,
        )
        _raise_if_stop_requested(stop_checker)
        _execute_loop_calls(
            calls=calls,
            round_i=round_i,
            cfg=cfg,
            stats=st,
            prompt_text=prompt_text,
            log_id=lid,
            runtime_state=runtime_state,
            context_state=loop_state,
            started_at=started_at,
            on_tool=on_tool,
            progress_callback=None,
            stop_checker=stop_checker,
        )
        _advance_progressive_disclosure(loop_state, calls)

    final_message = _clean_model_text(last) or _format_empty_output_message(cfg, round_i=max_rounds - 1)
    _raise_if_stop_requested(stop_checker)
    return _prepend_runtime_warnings(final_message, warning_messages)
