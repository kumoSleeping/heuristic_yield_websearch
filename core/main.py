"""
hyw/main.py - 极简 LLM 对话循环 + LiteLLM 原生工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web_search (自带)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .tool_view import format_tool_view_text
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
    BASE_SYSTEM_PROMPT,
    CANDIDATE_EVIDENCE_HEADING,
    CANDIDATE_EVIDENCE_REPLACEMENT_TEXT,
    CANDIDATE_EVIDENCE_TEXT,
    CONTEXT_KEEP_FOLLOWUP_REMINDER_TEXT,
    CONTEXT_MANAGEMENT_TOOL_PROMPT,
    DEFAULT_NO_RESULTS_TEXT,
    DEFAULT_NO_TITLE_TEXT,
    DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT,
    DEFAULT_PAGE_NO_MATCHING_TEXT,
    DUPLICATE_QUERY_SKIPPED_TEXT,
    EVIDENCE_CONTEXT_EMPTY_TEXT,
    EVIDENCE_CONTEXT_HEADING,
    EVIDENCE_CONTEXT_ID_TEXT,
    EVIDENCE_CONTEXT_KEPT_TEXT,
    FIRST_SEARCH_PROMPT,
    NORMAL_LOOP_PROMPT,
    POST_SEARCH_SKELETON_PROMPT,
    POST_SKELETON_REFINE_PROMPT,
    PAGE_MARKDOWN_CACHE_HIT_TEXT,
    PAGE_MARKDOWN_CACHE_MISS_TEXT,
    PAGE_MARKDOWN_CACHE_PREFIX,
    PAGE_MARKDOWN_EMPTY_TITLE,
    PAGE_MARKDOWN_KEYWORDS_PREFIX,
    PAGE_MARKDOWN_MATCHED_LINES_TEXT,
    PAGE_MARKDOWN_TITLE_PREFIX,
    PAGE_MARKDOWN_WINDOW_ALL_TEXT,
    PAGE_MARKDOWN_WINDOW_PREFIX,
    PAGE_MARKDOWN_WINDOW_SUFFIX,
    format_context_keep_markdown,
    litellm_tool_config_for_phase,
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

    if name in ("web_search", "web_search_wiki"):
        query = str(args.get("query") or payload.get("query") or "").strip()
        results = payload.get("results")
        lines: list[str] = []
        if query:
            lines.append(f"# Search: {query}")
            lines.append("")

        if not payload.get("ok", True):
            error = str(payload.get("error") or "search failed").strip()
            lines.append(error)
            return "\n".join(lines).strip()

        rows = results if isinstance(results, list) else []
        if not rows:
            lines.append(DEFAULT_NO_RESULTS_TEXT)
            return "\n".join(lines).strip()

        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or DEFAULT_NO_TITLE_TEXT).strip() or DEFAULT_NO_TITLE_TEXT
            url = str(row.get("url") or "").strip()
            snippet = str(row.get("snippet") or row.get("intro") or "").strip()
            if url:
                lines.append(f"{idx}. [{title}]({url})")
            else:
                lines.append(f"{idx}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    if name == "page_extract":
        title = str(payload.get("title") or "").strip()
        url = str(payload.get("url") or args.get("url") or "").strip()
        content = str(payload.get("content") or "").strip()
        if not payload.get("ok", True):
            error = str(payload.get("error") or "page extract failed").strip()
            lines = [f"# Page: {url}" if url else "# Page Extract", "", error]
            return "\n".join(lines).strip()

        lines = []
        if title:
            lines.append(f"# {title}")
        elif url:
            lines.append(f"# Page: {url}")
        if url:
            if lines:
                lines.append("")
            lines.append(f"Source: {url}")
        if content:
            if lines:
                lines.append("")
            lines.append(content)
        return "\n".join(lines).strip() or "No content."

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


def log_model_call(
    *,
    label: str,
    model: str,
    messages: list[dict],
    output: str,
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
        lines = [
            f"## [{ts}] {label}",
            f"- model: `{model}`",
        ]
        if duration_ms is not None:
            lines.append(f"- duration: {duration_ms:.0f}ms")
        if u:
            lines.append(f"- tokens: prompt={u.get('prompt_tokens',0)} completion={u.get('completion_tokens',0)} total={u.get('total_tokens',0)}")
        if cost is not None:
            lines.append(f"- cost: ${cost:.6f}")
        if error:
            lines.append(f"- **error**: {error}")

        # input
        lines.append("")
        lines.append("### Input")
        for m in messages:
            role = m.get("role", "?")
            content = _format_log_message_content(m.get("content"))
            lines.append(f"**[{role}]**")
            lines.append(f"```\n{content}\n```")

        # output
        lines.append("")
        lines.append("### Output")
        if output:
            lines.append(f"```\n{output}\n```")

        lines.append("")
        lines.append("---")
        lines.append("")

        with _log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
    except Exception:
        pass  # 日志不应影响主流程


# ── 工具: web_search (hyw 内置) ──────────────────────────


def get_web_search_backend(config: dict[str, Any] | None = None) -> str:
    handlers = resolve_tool_handlers(config, "search")
    if not handlers:
        return "search:none"
    return "search:" + ",".join(handler.provider for handler in handlers)


_DIRECT_SEARCH_TIMEOUT_S = 12.0
_DIRECT_PAGE_TIMEOUT_S = 60.0


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
    """Warm up the built-in websearch service."""
    from .web_search import on_startup

    return on_startup(headless=headless, config=config)


def shutdown_tools(config: dict[str, Any] | None = None):
    del config
    from .web_search import on_shutdown

    on_shutdown()


def _web_search(
    query: str,
    *,
    config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    **_,
) -> dict[str, Any]:
    """调用 WebToolSuite 执行搜索."""
    try:
        return _run_async(
            asyncio.wait_for(
                _web_search_async(
                    query=query,
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


async def _web_search_async(
    query: str,
    *,
    config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    **_,
) -> dict[str, Any]:
    """异步调用 WebToolSuite 执行搜索."""
    from .web_search import web_search

    cfg = build_model_config(config or load_config())
    try:
        payload = await web_search(
            query=query, mode="text",
            max_results=5,
            config=cfg,
            headless=bool(cfg.get("headless") not in (False, "false", "0", 0)),
            progress_callback=progress_callback,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
    from .web_search import page_extract

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
        "content": str(payload.get("content") or "")[:8000],
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
    if name in ("web_search", "web_search_wiki"):
        r = _run_direct_web_search(
            query=args.get("query", ""),
            cfg=cfg,
            runtime_state=runtime_state,
            progress_callback=progress_callback,
        )
    elif name == "page_extract":
        r = _run_page_probe(
            url=args.get("url", ""),
            query=args.get("query", ""),
            lines=args.get("lines", ""),
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
    if name in ("web_search", "web_search_wiki"):
        return "search"
    if name == "page_extract":
        return "page_extract"
    if name == "plan_update":
        return "plan"
    if name == "context_keep":
        return "context"
    return ""


def _tool_display_name(name: str, *, provider: str = "") -> str:
    del provider
    return {
        "web_search": "Web Search",
        "web_search_wiki": "Web Search",
        "page_extract": "Read",
        "plan_update": "Plan",
        "context_keep": "Context",
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
    if capability not in {"search", "page_extract", "render"}:
        return ""
    try:
        handlers = resolve_tool_handlers(config, capability)
    except Exception:
        return ""
    if not handlers:
        return ""
    return str(handlers[0].provider or "").strip()


def _tool_preview_callback_args(name: str, args: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(args)
    provider = _tool_provider_from_config(name, config)
    if provider:
        merged["_provider"] = provider
    merged["_display_name"] = _tool_display_name(name, provider=provider)
    merged["_planned"] = True
    return merged


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
        if "ok" in payload:
            merged["_ok"] = payload.get("ok")
        if name == "plan_update":
            merged["_created_count"] = len(payload.get("created_ids") or []) if isinstance(payload.get("created_ids"), list) else 0
            merged["_updated_count"] = len(payload.get("updated_ids") or []) if isinstance(payload.get("updated_ids"), list) else 0
        if payload.get("from_cache"):
            merged["_from_cache"] = True
        meta = _tool_meta(payload)
        usage = meta.get("usage")
        if isinstance(usage, dict) and usage.get("tokens") not in (None, ""):
            merged["_jina_tokens"] = usage.get("tokens")
            if name == "page_extract":
                merged["_page_usage_tokens"] = usage.get("tokens")
                merged["_page_usage_requests"] = usage.get("requests")
        billing = meta.get("billing")
        if name == "page_extract" and isinstance(billing, dict):
            merged["_page_billing_mode"] = billing.get("mode")
            merged["_page_cost_usd"] = billing.get("cost_usd")
    if not provider:
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
        "search": "web_search",
        "page": "page_extract",
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


def _sanitize_native_tool_call(name: str, args: dict[str, Any], *, call_id: str = "") -> dict[str, Any] | None:
    normalized_name = _normalize_native_tool_name(name)
    payload = dict(args or {})
    if normalized_name == "web_search":
        query = str(payload.get("query") or payload.get("q") or "").strip()
        if not query:
            return None
        clean_args: dict[str, Any] = {"query": query}
    elif normalized_name == "page_extract":
        url = str(payload.get("url") or payload.get("target") or "").strip()
        ref = _normalize_context_ref(str(payload.get("ref") or "").strip())
        query = str(payload.get("query") or payload.get("keyword") or payload.get("keywords") or "").strip()
        lines = str(payload.get("lines") or "").strip()
        if not url and not ref:
            return None
        clean_args = {}
        if url:
            clean_args["url"] = url
        if ref:
            clean_args["ref"] = ref
        if query:
            clean_args["query"] = query
        if lines:
            clean_args["lines"] = lines
    elif normalized_name == "plan_update":
        clean_args = {}
        create = payload.get("create")
        update = payload.get("update")
        if isinstance(create, list):
            create_items: list[dict[str, Any]] = []
            for item in create:
                if not isinstance(item, dict):
                    continue
                normalized_type = _normalize_context_item_type(str(item.get("type") or "").strip())
                if not normalized_type.startswith("skeleton."):
                    continue
                create_items.append({**dict(item), "type": normalized_type})
            if create_items:
                clean_args["create"] = create_items
        if isinstance(update, list):
            update_items = [dict(item) for item in update if isinstance(item, dict) and item.get("id") not in (None, "")]
            if update_items:
                clean_args["update"] = update_items
        if not clean_args:
            return None
    elif normalized_name == "context_keep":
        ids = _normalize_context_keep_args(payload)
        if not ids:
            return None
        clean_args = {}
        if ids:
            clean_args["ids"] = ids
    else:
        return None

    if call_id:
        clean_args["_call_id"] = call_id
    return {"name": normalized_name, "args": clean_args}


def _parse_native_tool_calls(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_tool_calls = payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    calls: list[dict[str, Any]] = []
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


def _tool_calls_preview_text(calls: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        line = format_tool_view_text(name, args)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _normalize_context_ref(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"\d+", text):
        return text
    return ""


def _resolve_page_call_refs(calls: list[dict[str, Any]], phase_state: "_PhaseRuntimeState") -> None:
    if not calls:
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("name") or "").strip() != "page_extract":
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else None
        if not isinstance(args, dict):
            continue
        if str(args.get("url") or "").strip():
            continue
        ref = _normalize_context_ref(str(args.get("ref") or "").strip())
        if not ref:
            continue
        try:
            item_id = int(ref)
        except Exception:
            continue
        item = _find_context_item(phase_state, item_id)
        if item is None:
            continue
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

def _append_search_evidence_items(call: dict[str, Any], payload: dict[str, Any], state: _PhaseRuntimeState) -> list[int]:
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    query = str(args.get("query") or payload.get("query") or "").strip()
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    created_ids: list[int] = []
    if not rows:
        item = _add_context_item(
            state,
            "evidence.search",
            {
                "query": query,
                "text": str(payload.get("error") or DEFAULT_NO_RESULTS_TEXT).strip() or DEFAULT_NO_RESULTS_TEXT,
                "snippet": str(payload.get("error") or DEFAULT_NO_RESULTS_TEXT).strip() or DEFAULT_NO_RESULTS_TEXT,
            },
            pending=True,
        )
        created_ids.append(item.item_id)
        return created_ids

    for row in rows:
        if not isinstance(row, dict):
            continue
        item = _add_context_item(
            state,
            "evidence.search",
            {
                "query": query,
                "title": str(row.get("title") or DEFAULT_NO_TITLE_TEXT).strip() or DEFAULT_NO_TITLE_TEXT,
                "url": str(row.get("url") or "").strip(),
                "snippet": re.sub(r"\s+", " ", str(row.get("snippet") or row.get("intro") or "").strip()),
            },
            pending=True,
        )
        created_ids.append(item.item_id)
    return created_ids


def _group_page_matches(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    groups: list[tuple[int, int, list[str]]] = []
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
            groups[-1] = (start, line_no, [*texts, text])
        else:
            groups.append((line_no, line_no, [text]))
    rendered: list[tuple[str, str]] = []
    for start, end, texts in groups:
        line_span = f"{start}" if start == end else f"{start}-{end}"
        rendered.append((line_span, "\n".join(texts)))
    return rendered


def _append_page_evidence_items(call: dict[str, Any], payload: dict[str, Any], state: _PhaseRuntimeState) -> list[int]:
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    url = str(payload.get("url") or args.get("url") or "").strip()
    title = str(payload.get("title") or "").strip()
    query = str(payload.get("query") or args.get("query") or "").strip()
    matches = payload.get("matched_lines") if isinstance(payload.get("matched_lines"), list) else []
    created_ids: list[int] = []
    grouped = _group_page_matches(matches)
    if not grouped:
        item = _add_context_item(
            state,
            "evidence.page",
            {
                "title": title,
                "url": url,
                "query": query,
                "text": str(payload.get("page_error") or payload.get("error") or DEFAULT_PAGE_NO_MATCHING_TEXT).strip() or DEFAULT_PAGE_NO_MATCHING_TEXT,
            },
            pending=True,
        )
        created_ids.append(item.item_id)
        return created_ids

    for source_lines, text in grouped:
        item = _add_context_item(
            state,
            "evidence.page",
            {
                "title": title,
                "url": url,
                "query": query,
                "source_lines": source_lines,
                "text": text,
            },
            pending=True,
        )
        created_ids.append(item.item_id)
    return created_ids


def _normalize_context_ids(value: Any) -> list[int]:
    raw_items: list[Any] = []
    if isinstance(value, list):
        raw_items = list(value)
    elif isinstance(value, str):
        raw_items = [chunk.strip() for chunk in re.split(r"[,\n|]+", value)]

    ids: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            normalized = int(item)
        except Exception:
            continue
        if normalized in seen or normalized < 0:
            continue
        seen.add(normalized)
        ids.append(normalized)
    return ids


def _normalize_context_keep_args(args: dict[str, Any] | None) -> list[int]:
    payload = dict(args or {}) if isinstance(args, dict) else {}
    return _normalize_context_ids(payload.get("ids", []))


def _context_keep_markdown(
    *,
    kept_ids: list[int],
    already_kept_ids: list[int],
    missing_ids: list[int],
    reason: str = "",
) -> str:
    return format_context_keep_markdown(
        kept_ids=kept_ids,
        already_kept_ids=already_kept_ids,
        missing_ids=missing_ids,
        reason=reason,
    )


def _keep_session_context_refs(
    state: "_PhaseRuntimeState",
    ids: list[int],
    *,
    reason: str = "",
) -> dict[str, Any]:
    requested_ids: list[int] = []
    seen: set[int] = set()
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        requested_ids.append(item_id)

    if not requested_ids:
        return {
            "ok": False,
            "error": "no valid ids to keep",
            "kept_ids": [],
            "already_kept_ids": [],
            "missing_ids": [],
            "_model_markdown": _context_keep_markdown(
                kept_ids=[],
                already_kept_ids=[],
                missing_ids=[],
                reason=reason,
            ),
        }

    kept_map = {
        int(item.item_id): item
        for item in state.items
        if item.item_type.startswith("evidence.")
    }
    pending_map = {
        int(item.item_id): item
        for item in state.pending_items
        if item.item_type.startswith("evidence.")
    }

    kept_ids: list[int] = []
    already_kept_ids: list[int] = []
    missing_ids: list[int] = []
    moved_items: list[_ContextItem] = []

    for item_id in requested_ids:
        if item_id in kept_map:
            already_kept_ids.append(item_id)
            continue
        item = pending_map.get(item_id)
        if item is None:
            missing_ids.append(item_id)
            continue
        kept_ids.append(item_id)
        moved_items.append(item)

    if moved_items:
        moved_id_set = {int(item.item_id) for item in moved_items}
        state.pending_items = [item for item in state.pending_items if int(item.item_id) not in moved_id_set]
        state.items.extend(moved_items)

    return {
        "ok": bool(kept_ids),
        "kept_ids": kept_ids,
        "already_kept_ids": already_kept_ids,
        "missing_ids": missing_ids,
        "reason": reason,
        "count": len(kept_ids),
        "_model_markdown": _context_keep_markdown(
            kept_ids=kept_ids,
            already_kept_ids=already_kept_ids,
            missing_ids=missing_ids,
            reason=reason,
        ),
    }


def _build_accumulated_context_message(state: "_PhaseRuntimeState") -> str:
    evidence_items = _evidence_items(state)
    candidate_items = _candidate_evidence_items(state)
    if not evidence_items and not candidate_items and not state.pending_user_note:
        return ""
    parts = [EVIDENCE_CONTEXT_HEADING]
    if evidence_items:
        parts.extend(
            [
                EVIDENCE_CONTEXT_KEPT_TEXT,
                EVIDENCE_CONTEXT_ID_TEXT,
            ]
        )
    else:
        parts.append(EVIDENCE_CONTEXT_EMPTY_TEXT)
    for item in evidence_items:
        data = dict(item.data)
        parts.append(f"[{item.item_id}] {item.item_type}")
        for key in ("query", "title", "url", "snippet", "source_lines", "text"):
            value = str(data.get(key) or "").strip()
            if value:
                parts.append(f"{key}: {value}")
        parts.append("")
    if candidate_items:
        parts.extend(
            [
                "",
                CANDIDATE_EVIDENCE_HEADING,
                CANDIDATE_EVIDENCE_TEXT,
                CANDIDATE_EVIDENCE_REPLACEMENT_TEXT,
            ]
        )
        for item in candidate_items:
            data = dict(item.data)
            parts.append(f"[{item.item_id}] {item.item_type}")
            for key in ("query", "title", "url", "snippet", "source_lines", "text"):
                value = str(data.get(key) or "").strip()
                if value:
                    parts.append(f"{key}: {value}")
            parts.append("")
    if state.pending_user_note:
        parts.extend(["", state.pending_user_note])
    if candidate_items:
        parts.extend(["", CONTEXT_KEEP_FOLLOWUP_REMINDER_TEXT])
    return "\n".join(part for part in parts if str(part).strip()).strip()


def _render_skeleton_markdown(state: "_PhaseRuntimeState") -> str:
    lines = ["# Skeleton Context"]
    skeleton_items = _skeleton_items(state)
    if not skeleton_items:
        lines.extend(["", "- Skeleton not built yet."])
        return "\n".join(lines).strip()
    for item in skeleton_items:
        data = dict(item.data)
        lines.extend(["", f"[{item.item_id}] {item.item_type}"])
        if str(data.get("claim_id") or "").strip():
            lines.append(f"claim_id: {str(data.get('claim_id') or '').strip()}")
        text = str(data.get("text") or "").strip()
        if text:
            lines.append(f"text: {text}")
    return "\n".join(lines).strip()


def _build_skeleton_context_message(state: "_PhaseRuntimeState") -> str:
    return _render_skeleton_markdown(state)


def _build_round_brief_message(state: "_PhaseRuntimeState") -> str:
    prompts: list[str] = []
    if state.disclosure_step <= 0:
        prompts.append(FIRST_SEARCH_PROMPT.strip())
    elif state.disclosure_step == 1:
        prompts.append(POST_SEARCH_SKELETON_PROMPT.strip())
    elif state.disclosure_step == 2 and not state.disclosure_refine_consumed:
        prompts.append(POST_SKELETON_REFINE_PROMPT.strip())
    return "\n\n".join(part for part in prompts if str(part).strip()).strip()


def _build_loop_system_prompt(
    cfg: dict[str, Any],
    user_message: str = "",
    *,
    disclosure_step: int = 0,
) -> str:
    custom = str(cfg.get("system_prompt") or "").strip()
    name = str(cfg.get("name") or DEFAULT_NAME).strip()
    context_management_tool_prompt = ""
    if int(disclosure_step) > 0:
        context_management_tool_prompt = CONTEXT_MANAGEMENT_TOOL_PROMPT.strip()
    return "\n\n".join(
        part
        for part in (
            BASE_SYSTEM_PROMPT.format(
                name=name,
                language=cfg.get("language") or "zh-CN",
                time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                custom=(custom + "\n") if custom else "",
                context_management_tool_prompt=(context_management_tool_prompt + "\n") if context_management_tool_prompt else "",
                user_message=user_message,
            ).strip(),
            NORMAL_LOOP_PROMPT.strip(),
        )
        if part
    ).strip()


def _build_loop_messages(
    *,
    cfg: dict[str, Any],
    prompt_text: str,
    history: list[dict[str, Any]],
    state: _PhaseRuntimeState,
    context: Any = None,
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_loop_system_prompt(
                cfg,
                prompt_text,
                disclosure_step=state.disclosure_step,
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
    round_brief = _build_round_brief_message(state)
    if round_brief:
        msgs.append({"role": "user", "content": round_brief})
        if state.disclosure_step == 2:
            state.disclosure_refine_consumed = True
    skeleton_message = _build_skeleton_context_message(state)
    if skeleton_message:
        msgs.append({"role": "user", "content": skeleton_message})
    context_message = _build_accumulated_context_message(state)
    if context_message:
        msgs.append({"role": "user", "content": context_message})
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
        value = 15
    return max(2, min(value or 15, 32))


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


def _block_has_model_overrides(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    for key in (
        "model",
        "model_provider",
        "custom_llm_provider",
        "api_key",
        "api_key_env",
        "api_base",
        "wire_api",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
    ):
        if block.get(key) not in (None, ""):
            return True
    return isinstance(block.get("extra_body"), dict)


def _build_override_model_config(cfg: dict[str, Any], *paths: str) -> dict[str, Any]:
    child_cfg = deepcopy(cfg)
    block: dict[str, Any] | None = None
    for path in paths:
        candidate = cfg_get(cfg, path, {})
        if _block_has_model_overrides(candidate):
            block = candidate
            break
    if not isinstance(block, dict):
        return child_cfg

    has_override = False
    explicit_transport_override = False
    explicit_reasoning_override = False
    explicit_limit_override = False
    for key in (
        "model",
        "model_provider",
        "custom_llm_provider",
        "api_key",
        "api_key_env",
        "api_base",
        "wire_api",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
    ):
        value = block.get(key)
        if value in (None, ""):
            continue
        child_cfg[key] = deepcopy(value)
        has_override = True
        if key in {"model", "model_provider", "custom_llm_provider", "api_key", "api_key_env", "api_base", "wire_api"}:
            explicit_transport_override = True
        if key == "reasoning_effort":
            explicit_reasoning_override = True
        if key in {"max_tokens", "max_completion_tokens"}:
            explicit_limit_override = True
    extra_body = block.get("extra_body")
    if isinstance(extra_body, dict):
        child_cfg["extra_body"] = deepcopy(extra_body)
        has_override = True
    elif explicit_transport_override:
        # Do not leak the parent model's provider routing/body overrides into a child
        # that explicitly switched model/provider credentials unless it asked for them.
        child_cfg.pop("extra_body", None)
    if explicit_transport_override and not explicit_reasoning_override:
        child_cfg.pop("reasoning_effort", None)
    if explicit_transport_override and not explicit_limit_override:
        child_cfg.pop("max_tokens", None)
        child_cfg.pop("max_completion_tokens", None)
    if not has_override:
        return child_cfg

    child_profile: dict[str, Any] = {
        "model": str(child_cfg.get("model") or cfg.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    }
    for key in (
        "model_provider",
        "custom_llm_provider",
        "api_key",
        "api_key_env",
        "api_base",
        "wire_api",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
    ):
        value = child_cfg.get(key)
        if value in (None, ""):
            continue
        child_profile[key] = deepcopy(value)
    if isinstance(child_cfg.get("extra_body"), dict):
        child_profile["extra_body"] = deepcopy(child_cfg["extra_body"])

    child_cfg["models"] = [child_profile]
    child_cfg["stage1_model_index"] = 0
    child_cfg["stage2_model_index"] = 0
    return child_cfg


def _build_sub_agent_model_config(cfg: dict[str, Any], path: str) -> dict[str, Any]:
    return _build_override_model_config(cfg, path)


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
    pending_items: list[_ContextItem] = field(default_factory=list)
    next_item_id: int = 0
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
    item = _ContextItem(item_id=_next_context_item_id(state), item_type=str(item_type).strip(), data=dict(data))
    target = state.pending_items if pending else state.items
    target.append(item)
    return item


def _find_context_item(state: _PhaseRuntimeState, item_id: int) -> _ContextItem | None:
    for item in [*state.items, *state.pending_items]:
        if int(item.item_id) == int(item_id):
            return item
    return None


def _iter_context_items(
    state: _PhaseRuntimeState,
    prefix: str = "",
    *,
    include_pending: bool = False,
) -> list[_ContextItem]:
    source = [*state.items, *state.pending_items] if include_pending else state.items
    items = sorted(source, key=lambda item: int(item.item_id))
    if not prefix:
        return items
    return [item for item in items if item.item_type.startswith(prefix)]


def _skeleton_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    return _iter_context_items(state, "skeleton.")


def _evidence_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    return _iter_context_items(state, "evidence.")


def _candidate_evidence_items(state: _PhaseRuntimeState) -> list[_ContextItem]:
    return sorted(
        [item for item in state.pending_items if item.item_type.startswith("evidence.")],
        key=lambda item: int(item.item_id),
    )


def _normalize_context_item_type(value: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "project_language": "skeleton.project_language",
        "keyword": "skeleton.keyword",
        "user_need": "skeleton.user_need",
        "claim": "skeleton.claim",
        "search": "evidence.search",
        "page": "evidence.page",
    }
    return aliases.get(text, text)


def _clean_context_item_data(item_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in ("text", "title", "url", "snippet", "source_lines", "claim_id", "query"):
        value = str(payload.get(key) or "").strip()
        if value:
            cleaned[key] = value
    if item_type == "skeleton.project_language" and "text" not in cleaned:
        value = str(payload.get("project_language") or "").strip()
        if value:
            cleaned["text"] = value
    return cleaned


def _plan_update_markdown(
    *,
    created: list[int],
    updated: list[int],
    missing: list[int],
) -> str:
    lines = ["# Plan Update", ""]
    lines.append(f"Created: {', '.join(str(item) for item in created) if created else 'none'}")
    lines.append(f"Updated: {', '.join(str(item) for item in updated) if updated else 'none'}")
    if missing:
        lines.append(f"Missing: {', '.join(str(item) for item in missing)}")
    return "\n".join(lines).strip()


def _apply_plan_update(
    state: _PhaseRuntimeState,
    args: dict[str, Any],
) -> dict[str, Any]:
    created_ids: list[int] = []
    updated_ids: list[int] = []
    missing_ids: list[int] = []

    for item in args.get("create", []) if isinstance(args.get("create"), list) else []:
        if not isinstance(item, dict):
            continue
        item_type = _normalize_context_item_type(str(item.get("type") or "").strip())
        if not item_type or not item_type.startswith("skeleton."):
            continue
        data = _clean_context_item_data(item_type, item)
        if not data:
            continue
        created = _add_context_item(state, item_type, data)
        created_ids.append(created.item_id)

    for item in args.get("update", []) if isinstance(args.get("update"), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
        except Exception:
            continue
        current = _find_context_item(state, item_id)
        if current is None or not current.item_type.startswith("skeleton."):
            missing_ids.append(item_id)
            continue
        data = _clean_context_item_data(current.item_type, item)
        if not data:
            continue
        current.data.update(data)
        updated_ids.append(item_id)

    return {
        "ok": bool(created_ids or updated_ids),
        "created_ids": created_ids,
        "updated_ids": updated_ids,
        "missing_ids": missing_ids,
        "_model_markdown": _plan_update_markdown(
            created=created_ids,
            updated=updated_ids,
            missing=missing_ids,
        ),
    }


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


def _search_history_contains(state: "_SessionRuntimeState | None", query: str) -> bool:
    normalized = _normalize_search_query(query)
    if not normalized or state is None:
        return False
    with state.lock:
        if normalized in state.search_history_normalized:
            return True
        state.search_history_raw.append(str(query or "").strip())
        state.search_history_normalized.append(normalized)
        state.search_history_raw[:] = state.search_history_raw[-96:]
        state.search_history_normalized[:] = state.search_history_normalized[-96:]
    return False


def _build_search_payload_markdown(
    *,
    query: str,
    public_results: list[dict[str, Any]],
    skipped_duplicate: bool,
    error: str = "",
) -> str:
    lines = [
        f"# Search: {query}",
        "",
    ]
    if skipped_duplicate:
        lines.append(DUPLICATE_QUERY_SKIPPED_TEXT)
        return "\n".join(lines).strip()
    if error:
        lines.append(error)
        return "\n".join(lines).strip()
    if not public_results:
        lines.append(DEFAULT_NO_RESULTS_TEXT)
        return "\n".join(lines).strip()
    for index, item in enumerate(public_results, start=1):
        title = str(item.get("title") or DEFAULT_NO_TITLE_TEXT).strip() or DEFAULT_NO_TITLE_TEXT
        url = str(item.get("url") or "").strip()
        snippet = _truncate_text(item.get("snippet") or "", 180)
        matched_queries = [
            str(matched or "").strip()
            for matched in item.get("matched_queries", [])
            if str(matched or "").strip()
        ]
        if url:
            lines.append(f"{index}. [{title}]({url})")
        else:
            lines.append(f"{index}. {title}")
        if matched_queries:
            lines.append(f"   Matched queries: {' | '.join(matched_queries)}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def _run_direct_web_search(
    *,
    query: str,
    cfg: dict[str, Any],
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    query_text = str(query or "").strip()
    if not query_text:
        return {"ok": False, "error": "search query is empty", "query": ""}

    skipped_duplicate = _search_history_contains(runtime_state, query_text)
    if skipped_duplicate:
        return {
            "ok": False,
            "provider": _tool_provider_from_config("web_search", cfg),
            "query": query_text,
            "count": 0,
            "results": [],
            "skipped_duplicate": True,
            "error": "duplicate query skipped in this session",
            "_model_markdown": _build_search_payload_markdown(
                query=query_text,
                public_results=[],
                skipped_duplicate=True,
            ),
        }

    payload = _web_search(
        query_text,
        config=cfg,
        progress_callback=progress_callback,
    )
    if not isinstance(payload, dict):
        payload = {"ok": False, "error": "invalid search payload", "results": []}

    rows = payload.get("results")
    deduped_rows = _merge_search_results([], rows if isinstance(rows, list) else [], query=query_text)
    public_results = _public_search_results(deduped_rows)
    if runtime_state is not None and public_results:
        with runtime_state.lock:
            runtime_state.search_results_deduped = _merge_search_results(
                runtime_state.search_results_deduped,
                public_results,
                query=query_text,
            )

    error = str(payload.get("error") or "").strip()
    result = {
        "ok": bool(payload.get("ok", True)),
        "provider": _tool_provider_from_payload(payload) or _tool_provider_from_config("web_search", cfg),
        "query": query_text,
        "count": len(public_results),
        "results": public_results,
        "skipped_duplicate": False,
        "_model_markdown": _build_search_payload_markdown(
            query=query_text,
            public_results=public_results,
            skipped_duplicate=False,
            error=error if not payload.get("ok", True) else "",
        ),
    }
    meta = _tool_meta(payload)
    if meta:
        result["_meta"] = meta
    if error and not result["ok"]:
        result["error"] = error
    return result


def _normalize_page_window(value: Any, default: int = 20) -> int | str:
    text = str(value or "").strip().lower()
    if text == "all":
        return "all"
    try:
        parsed = int(text)
    except Exception:
        parsed = default
    return max(10, min(parsed, 80))


def _clean_page_probe_keyword(value: str) -> str:
    text = str(value or "").strip().strip("`'\"")
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_page_probe(task: str, *, lines: Any = None) -> tuple[list[str], int | str]:
    raw_task = str(task or "").strip()
    window = _normalize_page_window(lines, default=20)

    if window == "all":
        return [], "all"

    line_match = re.search(r"(\d+)\s*line", raw_task, flags=re.IGNORECASE)
    if line_match:
        window = _normalize_page_window(line_match.group(1), default=window)

    cleaned = re.sub(r"\b\d+\s*line\b", "", raw_task, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bin\s*page\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("，", "|").replace(",", "|").replace("、", "|")

    keywords: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[|\n]+", cleaned):
        keyword = _clean_page_probe_keyword(chunk)
        normalized = keyword.lower()
        if not keyword or normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(keyword)

    if not keywords and raw_task:
        fallback = _clean_page_probe_keyword(raw_task)
        if fallback:
            keywords.append(fallback)
    return keywords[:8], window


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
    keywords: list[str],
    window: int | str,
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

    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    if not lowered_keywords:
        lowered_keywords = [""]

    hit_lines: list[int] = []
    for index, raw_line in enumerate(raw_lines, start=1):
        lowered_line = raw_line.lower()
        if any(keyword and keyword in lowered_line for keyword in lowered_keywords):
            hit_lines.append(index)

    if not hit_lines:
        return []

    radius = max(5, window // 2)
    ranges: list[tuple[int, int]] = []
    for line_no in hit_lines:
        start = max(1, line_no - radius)
        end = min(len(raw_lines), line_no + radius)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    matched: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for start, end in ranges:
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


def _build_page_probe_markdown(
    *,
    url: str,
    title: str,
    keywords: list[str],
    window: int | str,
    matches: list[dict[str, Any]],
    page_error: str = "",
    from_cache: bool = False,
) -> str:
    lines = [f"# Page: {url}" if url else PAGE_MARKDOWN_EMPTY_TITLE, ""]
    if title:
        lines.append(f"{PAGE_MARKDOWN_TITLE_PREFIX}{title}")
    if keywords and window != "all":
        lines.append(f"{PAGE_MARKDOWN_KEYWORDS_PREFIX}{' | '.join(keywords)}")
    lines.append(
        PAGE_MARKDOWN_WINDOW_ALL_TEXT
        if window == "all"
        else f"{PAGE_MARKDOWN_WINDOW_PREFIX}{window}{PAGE_MARKDOWN_WINDOW_SUFFIX}"
    )
    lines.append(
        f"{PAGE_MARKDOWN_CACHE_PREFIX}{PAGE_MARKDOWN_CACHE_HIT_TEXT if from_cache else PAGE_MARKDOWN_CACHE_MISS_TEXT}"
    )
    if page_error:
        lines.extend(["", page_error])
        return "\n".join(lines).strip()
    if not matches:
        lines.extend(["", DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT])
        return "\n".join(lines).strip()
    lines.extend(["", PAGE_MARKDOWN_MATCHED_LINES_TEXT])
    for item in matches:
        if not isinstance(item, dict):
            continue
        line_no = int(item.get("line") or 0)
        text = str(item.get("text") or "").rstrip()
        if not text:
            continue
        lines.append(f"{line_no} | {text}")
    return "\n".join(lines).strip()


def _run_page_probe(
    *,
    url: str,
    query: str,
    lines: Any,
    cfg: dict[str, Any],
    runtime_state: "_SessionRuntimeState | None",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    source_url = str(url or "").strip()
    if not source_url:
        return {"ok": False, "error": "page url is empty", "url": "", "query": str(query or "").strip()}

    normalized_url = _normalize_state_url(source_url) or source_url
    if runtime_state is not None and normalized_url:
        with runtime_state.lock:
            runtime_state.visited_page_urls.add(normalized_url)

    keywords, window = _parse_page_probe(query, lines=lines)
    page_payload, from_cache = _load_page_payload(
        url=source_url,
        cfg=cfg,
        runtime_state=runtime_state,
        progress_callback=progress_callback,
    )
    title = str(page_payload.get("title") or "").strip()
    content = str(page_payload.get("content") or "").strip()
    page_error = str(page_payload.get("error") or "").strip()
    matches = _extract_page_line_matches(content, keywords=keywords, window=window) if content else []

    payload = {
        "ok": bool(page_payload.get("ok")) and bool(matches),
        "provider": str(page_payload.get("provider") or _tool_provider_from_config("page_extract", cfg)).strip(),
        "url": str(page_payload.get("url") or normalized_url).strip(),
        "title": title,
        "query": str(query or "").strip(),
        "keywords": keywords,
        "window": window,
        "matched_lines": matches,
        "count": len(matches),
        "page_error": page_error,
        "from_cache": from_cache,
        "_model_markdown": _build_page_probe_markdown(
            url=str(page_payload.get("url") or normalized_url).strip(),
            title=title,
            keywords=keywords,
            window=window,
            matches=matches,
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
):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    # Fail fast on provider throttling instead of waiting through SDK retries.
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
        try:
            resp = codex_transport_model_response(
                codex_transport_stream_response(
                    cfg=cfg,
                    messages=messages,
                    tools=tool_cfg.get("tools") if isinstance(tool_cfg.get("tools"), list) else [],
                )
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log_model_call(
                label=trace_label, model=model, messages=messages,
                output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                log_id=log_id,
            )
            raise
    else:
        litellm_mod = _get_litellm()
        try:
            resp = litellm_mod.completion(**kw)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log_model_call(
                label=trace_label, model=model, messages=messages,
                output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                log_id=log_id,
            )
            raise
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
    native_calls = _parse_native_tool_calls(_to_dict(msg) if msg is not None else {})
    if native_calls:
        tool_preview = _tool_calls_preview_text(native_calls)
        if output_text and tool_preview:
            output_text = f"{output_text}\n\n{tool_preview}".strip()
        elif tool_preview:
            output_text = tool_preview

    # 写日志
    log_model_call(
        label=trace_label, model=model, messages=messages,
        output=output_text,
        usage=usage, cost=cost, duration_ms=duration_ms, config=cfg,
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
    seen_search_queries: set[str] = set()
    seen_page_keys: set[tuple[str, str, str]] = set()
    seen_keep_ids: set[tuple[int, ...]] = set()
    allowed_tools = {"web_search", "page_extract", "plan_update", "context_keep"}

    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name not in allowed_tools or not isinstance(args, dict):
            continue

        if name == "web_search":
            query = str(args.get("query") or "").strip()
            normalized_query = _normalize_search_query(query)
            if not normalized_query or normalized_query in seen_search_queries:
                continue
            seen_search_queries.add(normalized_query)
            selected.append({"name": "web_search", "args": {"query": query}})
            if len(selected) >= _TOOL_LIMIT:
                break
            continue

        if name == "page_extract":
            normalized_url = _normalize_state_url(str(args.get("url") or "").strip())
            query = str(args.get("query") or "").strip()
            lines = str(args.get("lines") or "").strip()
            page_key = (normalized_url, _normalize_search_query(query), str(_normalize_page_window(lines)))
            if not normalized_url or page_key in seen_page_keys:
                continue
            seen_page_keys.add(page_key)
            payload = {
                "url": normalized_url,
                "query": query,
                "lines": str(_normalize_page_window(lines)),
            }
            if str(args.get("_call_id") or "").strip():
                payload["_call_id"] = str(args.get("_call_id") or "").strip()
            selected.append({"name": "page_extract", "args": payload})
            if len(selected) >= _TOOL_LIMIT:
                break
            continue

        if name == "context_keep":
            ids = _normalize_context_keep_args(args)
            if not ids:
                continue
            signature = tuple(ids)
            if signature in seen_keep_ids:
                continue
            seen_keep_ids.add(signature)
            payload: dict[str, Any] = {"ids": ids}
            if str(args.get("_call_id") or "").strip():
                payload["_call_id"] = str(args.get("_call_id") or "").strip()
            selected.append({"name": "context_keep", "args": payload})
            if len(selected) >= _TOOL_LIMIT:
                break
            continue

        if name == "plan_update":
            payload = dict(args)
            selected.append({"name": "plan_update", "args": payload})
            if len(selected) >= _TOOL_LIMIT:
                break

    return selected


def _collect_loop_calls(
    *,
    text: str,
    message_payload: dict[str, Any],
    state: _PhaseRuntimeState,
    round_i: int,
) -> list[dict[str, Any]]:
    raw_calls = _parse_native_tool_calls(message_payload)
    _resolve_page_call_refs(raw_calls, state)
    calls = _select_loop_calls(raw_calls)
    _ensure_call_ids(calls, round_i=round_i)
    return calls


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
                merged["_pending"] = status not in {"done", "failed"}
            if "count" in event:
                merged["_count"] = event.get("count")
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
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if not isinstance(args, dict):
            args = {}
        call_started_at = time.perf_counter()

        if name == "context_keep":
            ids = _normalize_context_keep_args(args)
            payload = _keep_session_context_refs(
                context_state,
                ids,
            )
        elif name == "plan_update":
            payload = _apply_plan_update(context_state, args)
        else:
            external_calls.append((index, call))
            continue

        if not isinstance(payload, dict):
            payload = {"ok": False, "error": "invalid tool payload"}
        _record_tool_stats(stats, payload)
        payload_map[index] = payload
        model_note = str(payload.get("_model_markdown") or "").strip()
        if name == "context_keep" and model_note:
            note_parts.append(model_note)
        if callable(on_tool):
            try:
                on_tool(
                    name,
                    _tool_callback_args(
                        name,
                        args,
                        payload,
                        elapsed_s=time.perf_counter() - call_started_at,
                        config=cfg,
                    ),
                )
            except Exception:
                pass

    if external_calls:
        _raise_if_stop_requested(stop_checker)
        context_state.pending_items = []
        futures: dict[Any, tuple[int, dict[str, Any], float]] = {}
        with ThreadPoolExecutor(max_workers=len(external_calls)) as pool:
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

            external_payloads: dict[int, tuple[dict[str, Any], float, dict[str, Any]]] = {}
            for future in as_completed(futures):
                index, call, call_started_at = futures[future]
                payload = future.result()
                if not isinstance(payload, dict):
                    payload = {"ok": False, "error": "invalid tool payload"}
                external_payloads[index] = (payload, call_started_at, call)

        for index, call in external_calls:
            _raise_if_stop_requested(stop_checker)
            payload, call_started_at, original_call = external_payloads.get(index, ({"ok": False, "error": "missing tool payload"}, time.perf_counter(), call))
            name = str(original_call.get("name") or "").strip()
            args = original_call.get("args") if isinstance(original_call.get("args"), dict) else {}
            if not isinstance(args, dict):
                args = {}
            if name == "web_search":
                created_ids = _append_search_evidence_items(original_call, payload, context_state)
                batch_created_ids.extend(created_ids)
            elif name == "page_extract":
                created_ids = _append_page_evidence_items(original_call, payload, context_state)
                batch_created_ids.extend(created_ids)
            _record_tool_stats(stats, payload)
            payload_map[index] = payload
            if callable(on_tool):
                try:
                    on_tool(
                        name,
                        _tool_callback_args(
                            name,
                            args,
                            payload,
                            elapsed_s=time.perf_counter() - call_started_at,
                            config=cfg,
                        ),
                    )
                except Exception:
                    pass
    context_state.pending_user_note = "\n\n".join(part for part in note_parts if part).strip()
    return payload_map


def _validate_progressive_tool_requirements(
    state: _PhaseRuntimeState,
    calls: list[dict[str, Any]],
) -> str:
    del state, calls
    return ""


def _calls_include_plan_update(state: _PhaseRuntimeState, calls: list[dict[str, Any]]) -> bool:
    del state
    for call in calls:
        if not isinstance(call, dict) or str(call.get("name") or "").strip() != "plan_update":
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if not isinstance(args, dict):
            continue
        create_items = args.get("create", []) if isinstance(args.get("create"), list) else []
        if any(isinstance(item, dict) for item in create_items):
            return True
        update_items = args.get("update", []) if isinstance(args.get("update"), list) else []
        if any(isinstance(item, dict) for item in update_items):
            return True
    return False


def _advance_progressive_disclosure(
    state: _PhaseRuntimeState,
    calls: list[dict[str, Any]],
) -> None:
    tool_names = {str(call.get("name") or "").strip() for call in calls if isinstance(call, dict)}
    if state.disclosure_step <= 0 and tool_names.intersection({"web_search", "page_extract"}):
        state.disclosure_step = 1
        state.pending_user_note = ""
        return
    if state.disclosure_step == 1 and _calls_include_plan_update(state, calls):
        state.disclosure_step = 2
        state.disclosure_refine_consumed = False
        state.pending_user_note = ""
        return
    if state.disclosure_step == 2 and state.disclosure_refine_consumed:
        state.disclosure_step = 3


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

        reasoning_text, reasoning_meta = _render_reasoning_display(
            payload=msg_payload,
            usage=usage,
        )
        if callable(on_reasoning):
            try:
                on_reasoning(reasoning_text, reasoning_meta)
            except Exception:
                pass

        calls = _collect_loop_calls(
            text=text,
            message_payload=msg_payload,
            state=loop_state,
            round_i=round_i,
        )
        assistant_content = _normalize_tool_turn_message(text, calls) if calls else _clean_model_text(text)

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

        _raise_if_stop_requested(stop_checker)
        _execute_loop_calls(
            calls=calls,
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
            context=context,
        )
        loop_state.pending_user_note = ""
        if callable(on_status):
            on_status(STATUS_THINKING)

        model = str(cfg.get("model") or DEFAULT_MODEL).strip()
        tool_cfg = litellm_tool_config_for_phase("loop", disclosure_step=loop_state.disclosure_step)
        t0 = time.perf_counter()
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        reasoning_text_parts: list[str] = []
        reasoning_details: list[dict[str, Any]] = []
        stream_tool_calls: dict[int, dict[str, Any]] = {}
        last_reasoning_emit_tokens = [0]
        if should_use_codex_mirror_transport(cfg):
            def _handle_codex_text_delta(delta: str) -> None:
                _raise_if_stop_requested(stop_checker)
                content_parts.append(delta)
                if callable(on_chunk):
                    on_chunk(delta)

            try:
                response_payload = codex_transport_stream_response(
                    cfg=cfg,
                    messages=msgs,
                    tools=tool_cfg.get("tools") if isinstance(tool_cfg.get("tools"), list) else [],
                    on_text_delta=_handle_codex_text_delta,
                )
            except StopRequestedError:
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                log_model_call(
                    label=f"round {round_i + 1}", model=model, messages=msgs,
                    output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                    log_id=lid,
                )
                return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)
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

            try:
                stream = litellm_mod.completion(**kw)
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                log_model_call(
                    label=f"round {round_i + 1}", model=model, messages=msgs,
                    output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                    log_id=lid,
                )
                return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)

            try:
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
            except StopRequestedError:
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                partial = "".join(content_parts)
                log_model_call(
                    label=f"round {round_i + 1}",
                    model=model,
                    messages=msgs,
                    output=partial,
                    error=str(e)[:300],
                    duration_ms=duration_ms,
                    config=cfg,
                    log_id=lid,
                )
                return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)
            msg_payload = {"tool_calls": [stream_tool_calls[index] for index in sorted(stream_tool_calls)]} if stream_tool_calls else {}

        duration_ms = (time.perf_counter() - t0) * 1000
        full_text = "".join(content_parts)
        if full_text:
            last = full_text

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

        native_calls = _finalize_stream_tool_calls(stream_tool_calls)
        if not native_calls and isinstance(msg_payload, dict):
            native_calls = _parse_native_tool_calls(msg_payload)
        logged_text = full_text or (_text(msg_payload) if isinstance(msg_payload, dict) else "")
        tool_preview = _tool_calls_preview_text(native_calls)
        if logged_text and tool_preview:
            log_output = f"{logged_text}\n\n{tool_preview}".strip()
        elif tool_preview:
            log_output = tool_preview
        else:
            log_output = logged_text

        # Log
        log_model_call(
            label=f"round {round_i + 1}", model=model, messages=msgs,
            output=log_output,
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

        calls = _collect_loop_calls(
            text=full_text,
            message_payload=msg_payload,
            state=loop_state,
            round_i=round_i,
        )
        assistant_content = _normalize_tool_turn_message(full_text, calls) if calls else _clean_model_text(full_text)

        progressive_error = _validate_progressive_tool_requirements(loop_state, calls)
        if progressive_error:
            if callable(on_rewind) and calls:
                rewind_tools = [
                    (
                        call["name"],
                        _tool_preview_callback_args(call["name"], call["args"], config=cfg),
                    )
                    for call in calls
                ]
                try:
                    on_rewind(assistant_content or full_text, rewind_tools)
                except Exception:
                    pass
            loop_state.pending_user_note = progressive_error
            continue

        if assistant_content.strip():
            history.append({"role": "assistant", "content": assistant_content})

        if not calls:
            _raise_if_stop_requested(stop_checker)
            final_message = _clean_model_text(full_text) or last or _format_empty_output_message(cfg, round_i=round_i)
            return _prepend_runtime_warnings(final_message, warning_messages)

        if callable(on_rewind):
            rewind_tools = [
                (
                    call["name"],
                    _tool_preview_callback_args(call["name"], call["args"], config=cfg),
                )
                for call in calls
            ]
            try:
                on_rewind(assistant_content or full_text, rewind_tools)
            except Exception:
                pass

        if callable(on_status):
            on_status(STATUS_SEARCHING)
        _raise_if_stop_requested(stop_checker)
        _execute_loop_calls(
            calls=calls,
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
