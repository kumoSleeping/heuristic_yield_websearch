"""
hyw/main.py - 极简 LLM 对话循环 + XML 标签工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web_search (自带)
配置: ~/.hyw/config.yml, 兼容单模型与多模型写法

工具调用方式: 模型在文本中输出 <search>/<page> XML 标签, 解析后执行工具, 注入结果再让模型继续.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from importlib import import_module
import json
import logging
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
from .prompts import (
    BASE_SYSTEM_PROMPT,
    CONTEXT_SWITCH_PROMPT,
    EMPTY_VERIFICATION_OUTLINE_PROMPT,
    FINAL_PHASE_PROMPT,
    HEADING_KEYWORD_REWRITE,
    HEADING_USER_NEED,
    HEADING_VERIFICATION_CLAIMS,
    HEADING_VERIFICATION_OUTLINE,
    STAGE1_IMAGE_GUIDANCE,
    STAGE1_PAGE_MODE_GUIDANCE,
    STAGE1_PHASE_PROMPT,
    STAGE1_RETRY_PROMPT,
    STAGE1_SKELETON_RETRY_PROMPT,
    STAGE1_WEBSEARCH_MODE_GUIDANCE,
    STAGE2_FINAL_REPLY_PROMPT,
    STAGE2_KICKOFF_PROMPT,
    STAGE2_PHASE_PROMPT,
    STAGE2_RETRY_PROMPT,
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
            lines.append("No results.")
            return "\n".join(lines).strip()

        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "No Title").strip() or "No Title"
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
_DIRECT_PAGE_TIMEOUT_S = 18.0


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
                "title": str(item.get("title") or "").strip() or "No Title",
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
    return ""


def _tool_provider_label(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    aliases = {
        "jina_ai": "jina",
        "ddgs": "duckduckgo",
    }
    return aliases.get(normalized, normalized)


def _tool_display_name(name: str, *, provider: str = "") -> str:
    suffix = {
        "web_search": "websearch",
        "web_search_wiki": "wiki",
        "page_extract": "page",
    }.get(name, str(name or "").strip())
    prefix = _tool_provider_label(provider)
    return f"{prefix}_{suffix}" if prefix else suffix


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
    handlers = resolve_tool_handlers(config, capability)
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


# ── XML 标签工具调用解析 ─────────────────────────────────────
_TOOL_TAG_RE = re.compile(r"<(search|wiki|page)\b([^>]*)>(.*?)</\1>", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_TASK_LIST_BLOCK_RE = re.compile(r"<task_list\b[^>]*>.*?</task_list>", re.IGNORECASE | re.DOTALL)
_ARTICLE_SKELETON_BLOCK_RE = re.compile(r"<article_skeleton\b[^>]*>.*?</article_skeleton>", re.IGNORECASE | re.DOTALL)
_SEARCH_REWRITE_BLOCK_RE = re.compile(r"<search_rewrite\b[^>]*>.*?</search_rewrite>", re.IGNORECASE | re.DOTALL)
_KEYWORD_REWRITE_BLOCK_RE = re.compile(r"<keyword_rewrite\b[^>]*>.*?</keyword_rewrite>", re.IGNORECASE | re.DOTALL)
_USER_NEED_BLOCK_RE = re.compile(r"<user_need\b[^>]*>.*?</user_need>", re.IGNORECASE | re.DOTALL)
_VERIFICATION_OUTLINE_BLOCK_RE = re.compile(r"<verification_outline\b[^>]*>.*?</verification_outline>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_USER_NEED_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?:user need reconstruction|user need restore|用户需求复原|需求复原|user need)\s*$", re.IGNORECASE)
_MARKDOWN_SEARCH_REWRITE_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?:keyword rewrite|关键词重绘|搜索词重绘|search rewrite)\s*$", re.IGNORECASE)
_MARKDOWN_SKELETON_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?:verification outline|verification skeleton|核验骨架|文章骨架|article skeleton|skeleton)\s*$", re.IGNORECASE)


def _parse_tag_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    text = str(raw or "")
    for key, dq, sq in re.findall(r'([:\w-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', text):
        value = dq if dq != "" else sq
        attrs[str(key or "").strip().lower()] = str(value or "").strip()
    return attrs


def _parse_tool_tags(text: str) -> list[dict]:
    """提取主模型允许的工具 XML 标签。"""
    # 先去掉 `...` 内联代码，避免匹配到模型解释文本中的标签
    cleaned = _INLINE_CODE_RE.sub("", text)
    calls: list[dict] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for m in _TOOL_TAG_RE.finditer(cleaned):
        tag = str(m.group(1) or "").strip().lower()
        attrs = _parse_tag_attrs(m.group(2))
        query = m.group(3).strip()
        if tag == "search":
            task = query
            if not task:
                continue
            name = "web_search"
            args = {"query": task}
        elif tag == "page":
            task = query
            url = str(attrs.get("url") or attrs.get("target") or "").strip()
            if not task or not url:
                continue
            name = "page_extract"
            args = {
                "url": url,
                "query": task,
                "lines": str(attrs.get("lines") or "").strip(),
            }
        else:
            continue
        signature = (
            name,
            tuple(sorted((str(key), str(value)) for key, value in args.items())),
        )
        if signature in seen:
            continue
        seen.add(signature)
        calls.append({"name": name, "args": args})
    return calls


def _strip_tool_tags(text: str) -> str:
    """从最终回答中去掉工具标签"""
    cleaned = _TASK_LIST_BLOCK_RE.sub("", str(text or ""))
    cleaned = _ARTICLE_SKELETON_BLOCK_RE.sub("", cleaned)
    cleaned = _SEARCH_REWRITE_BLOCK_RE.sub("", cleaned)
    cleaned = _KEYWORD_REWRITE_BLOCK_RE.sub("", cleaned)
    cleaned = _USER_NEED_BLOCK_RE.sub("", cleaned)
    cleaned = _VERIFICATION_OUTLINE_BLOCK_RE.sub("", cleaned)
    return _TOOL_TAG_RE.sub("", cleaned).strip()


def _format_tool_call_xml(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "").strip()
    args = call.get("args") if isinstance(call, dict) else {}
    if not isinstance(args, dict):
        args = {}
    if name == "web_search":
        return f"<search>{str(args.get('query') or '').strip()}</search>"
    if name == "page_extract":
        attrs: list[str] = []
        url = str(args.get("url") or "").strip()
        lines = str(args.get("lines") or "").strip()
        if url:
            attrs.append(f'url="{url.replace(chr(34), "&quot;")}"')
        if lines:
            attrs.append(f'lines="{lines.replace(chr(34), "&quot;")}"')
        attr_text = (" " + " ".join(attrs)) if attrs else ""
        return f"<page{attr_text}>{str(args.get('query') or '').strip()}</page>"
    return ""


def _tool_turn_preamble(text: str) -> str:
    match = _TOOL_TAG_RE.search(str(text or ""))
    if not match:
        return _strip_tool_tags(text)
    return str(text[:match.start()]).strip()


def _normalize_tool_turn_message(text: str, calls: list[dict[str, Any]]) -> str:
    preamble = _tool_turn_preamble(text)
    tags = "\n".join(_format_tool_call_xml(call) for call in calls).strip()
    if preamble and tags:
        return f"{preamble}\n\n{tags}"
    return preamble or tags


def _tool_tags_only_message(calls: list[dict[str, Any]]) -> str:
    return "\n".join(_format_tool_call_xml(call) for call in calls).strip()


def _strip_tool_xml_only(text: str) -> str:
    return _TOOL_TAG_RE.sub("", str(text or "")).strip()


def _extract_markdown_section_block(
    text: str,
    *,
    heading_re: re.Pattern[str],
    stop_heading_res: tuple[re.Pattern[str], ...] = (),
    stop_on_tool_tag: bool = False,
) -> str:
    lines = str(text or "").splitlines()
    start_index: int | None = None
    for index, raw_line in enumerate(lines):
        if heading_re.match(str(raw_line or "")):
            start_index = index
            break
    if start_index is None:
        return ""

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        raw_line = str(lines[index] or "")
        if stop_on_tool_tag and _TOOL_TAG_RE.search(raw_line):
            end_index = index
            break
        if any(pattern.match(raw_line) for pattern in stop_heading_res):
            end_index = index
            break
    return "\n".join(lines[start_index:end_index]).strip()


def _extract_article_skeleton(text: str) -> str:
    match = _VERIFICATION_OUTLINE_BLOCK_RE.search(str(text or ""))
    if match:
        return str(match.group(0) or "").strip()
    match = _ARTICLE_SKELETON_BLOCK_RE.search(str(text or ""))
    if not match:
        return _extract_markdown_section_block(
            str(text or ""),
            heading_re=_MARKDOWN_SKELETON_HEADING_RE,
            stop_on_tool_tag=True,
        )
    return str(match.group(0) or "").strip()


def _extract_search_rewrite(text: str) -> str:
    match = _KEYWORD_REWRITE_BLOCK_RE.search(str(text or ""))
    if match:
        return str(match.group(0) or "").strip()
    match = _SEARCH_REWRITE_BLOCK_RE.search(str(text or ""))
    if not match:
        return _extract_markdown_section_block(
            str(text or ""),
            heading_re=_MARKDOWN_SEARCH_REWRITE_HEADING_RE,
            stop_heading_res=(_MARKDOWN_USER_NEED_HEADING_RE, _MARKDOWN_SKELETON_HEADING_RE),
            stop_on_tool_tag=True,
        )
    return str(match.group(0) or "").strip()


def _extract_user_need_restore(text: str) -> str:
    match = _USER_NEED_BLOCK_RE.search(str(text or ""))
    if match:
        return str(match.group(0) or "").strip()
    return _extract_markdown_section_block(
        str(text or ""),
        heading_re=_MARKDOWN_USER_NEED_HEADING_RE,
        stop_heading_res=(_MARKDOWN_SKELETON_HEADING_RE,),
        stop_on_tool_tag=True,
    )


_PHASE_STAGE1 = "stage1"
_PHASE_STAGE2 = "stage2"
_PHASE_FINAL = "final"
_PHASE_TOOL_LIMIT = 8

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


def _tool_results_for_next_round(results_xml: str, *, round_i: int, max_rounds: int) -> str:
    del round_i, max_rounds
    return str(results_xml or "").strip()


def _stage1_mode_guidance(phase_state: _PhaseRuntimeState) -> str:
    mode = str(phase_state.first_collection_mode or "").strip().lower()
    if mode == "page":
        return STAGE1_PAGE_MODE_GUIDANCE.strip()
    return STAGE1_WEBSEARCH_MODE_GUIDANCE.strip()


def _stage1_image_guidance(phase_state: _PhaseRuntimeState) -> str:
    if not phase_state.input_has_images:
        return ""
    return STAGE1_IMAGE_GUIDANCE.strip()


def _build_phase_prompt(
    phase: str,
    *,
    phase_state: _PhaseRuntimeState,
) -> str:
    if phase == _PHASE_STAGE1:
        parts = [STAGE1_PHASE_PROMPT.strip()]
        image_guidance = _stage1_image_guidance(phase_state)
        if image_guidance:
            parts.append(image_guidance)
        parts.append(_stage1_mode_guidance(phase_state))
        return "\n".join(part for part in parts if part).strip()
    if phase == _PHASE_FINAL:
        return FINAL_PHASE_PROMPT.strip()

    skeleton_parts: list[str] = []
    user_need_context = _user_need_context_text(phase_state)
    if user_need_context:
        skeleton_parts.append(user_need_context)
    search_rewrite_context = _search_rewrite_context_text(phase_state)
    if search_rewrite_context:
        skeleton_parts.append(search_rewrite_context)
    if phase_state.skeleton_xml:
        skeleton_parts.append(HEADING_VERIFICATION_OUTLINE)
        skeleton_parts.append(phase_state.skeleton_xml)
    claim_checklist = _claim_checklist_text(phase_state)
    if claim_checklist:
        skeleton_parts.append(claim_checklist)
    context = "\n\n".join(part for part in skeleton_parts if part).strip()
    parts = [
        STAGE2_PHASE_PROMPT.format(
            skeleton_context=context or EMPTY_VERIFICATION_OUTLINE_PROMPT,
        ).strip()
    ]
    if int(phase_state.stage2_turns or 0) >= 2:
        parts.append(STAGE2_FINAL_REPLY_PROMPT.strip())
    return "\n\n".join(part for part in parts if part).strip()


def _build_system_prompt(
    cfg: dict,
    user_message: str = "",
    *,
    phase: str = _PHASE_STAGE1,
    phase_state: _PhaseRuntimeState | None = None,
) -> str:
    custom = str(cfg.get("system_prompt") or "").strip()
    name = str(cfg.get("name") or DEFAULT_NAME).strip()
    current_phase_state = phase_state or _PhaseRuntimeState()
    display_user_message = user_message
    if phase in {_PHASE_STAGE2, _PHASE_FINAL} and current_phase_state.search_rewrite_terms:
        display_user_message = CONTEXT_SWITCH_PROMPT
    return "\n\n".join(
        part
        for part in (
            BASE_SYSTEM_PROMPT.format(
                name=name,
                language=cfg.get("language") or "zh-CN",
                time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                custom=(custom + "\n") if custom else "",
                user_message=display_user_message,
            ).strip(),
            _build_phase_prompt(phase, phase_state=current_phase_state),
        )
        if part
    ).strip()


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
        value = 8
    return max(2, min(value or 8, 16))


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
    for key in ("model", "api_key", "api_base", "reasoning_effort", "max_tokens", "max_completion_tokens"):
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
    for key in ("model", "api_key", "api_base", "reasoning_effort", "max_tokens", "max_completion_tokens"):
        value = block.get(key)
        if value in (None, ""):
            continue
        child_cfg[key] = deepcopy(value)
        has_override = True
        if key in {"model", "api_key", "api_base"}:
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
    for key in ("api_key", "api_base", "reasoning_effort", "max_tokens", "max_completion_tokens"):
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
class _SkeletonClaim:
    claim_id: str
    text: str
    section: str = ""


@dataclass
class _PhaseRuntimeState:
    phase: str = _PHASE_STAGE1
    skeleton_xml: str = ""
    skeleton_title: str = ""
    claims: list[_SkeletonClaim] = field(default_factory=list)
    user_need_markdown: str = ""
    user_need_items: list[str] = field(default_factory=list)
    search_rewrite_xml: str = ""
    search_rewrite_terms: list[str] = field(default_factory=list)
    first_collection_mode: str = ""
    input_has_images: bool = False
    stage2_turns: int = 0
    execute_tool_rounds: int = 0
    execute_search_rounds: int = 0
    execute_page_rounds: int = 0


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


def _parse_article_skeleton(block: str) -> tuple[str, list[_SkeletonClaim]]:
    text = str(block or "").strip()
    if not text:
        return "", []

    claims: list[_SkeletonClaim] = []
    seen_ids: set[str] = set()

    if "<verification_outline" in text.lower():
        for claim_match in re.finditer(r"<i\b[^>]*\bid\s*=\s*['\"]?(\d+)['\"]?[^>]*>(.*?)</i>", text, flags=re.IGNORECASE | re.DOTALL):
            claim_id = str(claim_match.group(1) or "").strip()
            claim_text = re.sub(r"\s+", " ", str(claim_match.group(2) or "").strip())
            if not claim_id or not claim_text or claim_id in seen_ids:
                continue
            seen_ids.add(claim_id)
            claims.append(_SkeletonClaim(claim_id=claim_id, text=claim_text))
        return "", claims

    if "<article_skeleton" in text.lower():
        title_match = re.search(r"<title\b[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", str(title_match.group(1) or "").strip()) if title_match else ""

        section_matches = list(
            re.finditer(r"<section\b([^>]*)>(.*?)</section>", text, flags=re.IGNORECASE | re.DOTALL)
        )
        if section_matches:
            for match in section_matches:
                attrs = _parse_tag_attrs(match.group(1))
                section_name = str(attrs.get("name") or attrs.get("title") or "").strip()
                body = str(match.group(2) or "")
                for claim_match in re.finditer(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s+(.+?)\s*$", body, flags=re.MULTILINE):
                    claim_id = str(claim_match.group(1) or "").strip()
                    claim_text = re.sub(r"\s+", " ", str(claim_match.group(2) or "").strip())
                    if not claim_id or not claim_text or claim_id in seen_ids:
                        continue
                    seen_ids.add(claim_id)
                    claims.append(_SkeletonClaim(claim_id=claim_id, text=claim_text, section=section_name))
            return title, claims

        for claim_match in re.finditer(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s+(.+?)\s*$", text, flags=re.MULTILINE):
            claim_id = str(claim_match.group(1) or "").strip()
            claim_text = re.sub(r"\s+", " ", str(claim_match.group(2) or "").strip())
            if not claim_id or not claim_text or claim_id in seen_ids:
                continue
            seen_ids.add(claim_id)
            claims.append(_SkeletonClaim(claim_id=claim_id, text=claim_text))
        return title, claims

    title = ""
    current_section = ""
    lines = text.splitlines()
    if lines and _MARKDOWN_SKELETON_HEADING_RE.match(lines[0]):
        lines = lines[1:]
    for raw_line in lines:
        line = str(raw_line or "").rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            heading_text = re.sub(r"\s+", " ", str(heading_match.group(2) or "").strip())
            if not title:
                title = heading_text
            else:
                current_section = heading_text
            continue
        title_match = re.match(r"^\s*title\s*:\s*(.+?)\s*$", stripped, flags=re.IGNORECASE)
        if title_match and not title:
            title = re.sub(r"\s+", " ", str(title_match.group(1) or "").strip())
            continue
        claim_match = re.match(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s+(.+?)\s*$", line)
        if not claim_match:
            continue
        claim_id = str(claim_match.group(1) or "").strip()
        claim_text = re.sub(r"\s+", " ", str(claim_match.group(2) or "").strip())
        if not claim_id or not claim_text or claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)
        claims.append(_SkeletonClaim(claim_id=claim_id, text=claim_text, section=current_section))
    return title, claims


def _update_skeleton_state(phase_state: _PhaseRuntimeState, block: str) -> bool:
    title, claims = _parse_article_skeleton(block)
    if not block.strip():
        return False
    if not claims:
        return False
    phase_state.skeleton_xml = block.strip()
    phase_state.skeleton_title = title
    phase_state.claims = claims
    return True


def _parse_user_need_items(block: str) -> list[str]:
    text = str(block or "").strip()
    if not text:
        return []
    if "<user_need" in text.lower():
        items: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"<u\b[^>]*>(.*?)</u>", text, flags=re.IGNORECASE | re.DOTALL):
            item = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
            normalized = item.lower()
            if not item or normalized in seen:
                continue
            seen.add(normalized)
            items.append(item)
        return items[:6]
    items: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    if lines and _MARKDOWN_USER_NEED_HEADING_RE.match(lines[0]):
        lines = lines[1:]
    for line in lines:
        candidate = re.sub(r"^\s*(?:[-*]|\d+[.)]|\[[^\]]+\])\s*", "", str(line or "").strip())
        candidate = re.sub(r"\s+", " ", candidate).strip()
        normalized = candidate.lower()
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        items.append(candidate)
    return items[:6]


def _update_user_need_state(phase_state: _PhaseRuntimeState, block: str) -> bool:
    items = _parse_user_need_items(block)
    if not block.strip() or not items:
        return False
    phase_state.user_need_markdown = block.strip()
    phase_state.user_need_items = items
    return True


def _user_need_context_text(phase_state: _PhaseRuntimeState) -> str:
    if not phase_state.user_need_items:
        return ""
    lines = ["# User Need Reconstruction"]
    for index, item in enumerate(phase_state.user_need_items, start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines).strip()


def _claim_checklist_text(phase_state: _PhaseRuntimeState) -> str:
    if not phase_state.claims:
        return ""
    lines = ["# Verification Claims"]
    for claim in phase_state.claims:
        section_prefix = f"{claim.section} / " if claim.section else ""
        lines.append(f"[{claim.claim_id}] {section_prefix}{claim.text}")
    return "\n".join(lines).strip()


def _parse_search_rewrite_terms(block: str) -> list[str]:
    text = str(block or "").strip()
    if not text:
        return []

    terms: list[str] = []
    seen: set[str] = set()

    if "<keyword_rewrite" in text.lower():
        for match in re.finditer(r"<t\b[^>]*>(.*?)</t>", text, flags=re.IGNORECASE | re.DOTALL):
            term = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
            normalized = term.lower()
            if not term or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term)
        return terms[:12]

    if "<search_rewrite" in text.lower():
        for match in re.finditer(r"<term\b[^>]*>(.*?)</term>", text, flags=re.IGNORECASE | re.DOTALL):
            term = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
            normalized = term.lower()
            if not term or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term)
        if terms:
            return terms[:12]
        text = re.sub(r"</?search_rewrite\b[^>]*>", "", text, flags=re.IGNORECASE)

    lines = text.splitlines()
    if lines and _MARKDOWN_SEARCH_REWRITE_HEADING_RE.match(lines[0]):
        lines = lines[1:]
    for line in lines:
        if _MARKDOWN_SKELETON_HEADING_RE.match(str(line or "")):
            break
        candidate = re.sub(r"^\s*(?:[-*]|\d+[.)]|\[[^\]]+\])\s*", "", str(line or "").strip())
        candidate = re.sub(r"\s+", " ", candidate).strip()
        normalized = candidate.lower()
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(candidate)
    return terms[:12]


def _update_search_rewrite_state(phase_state: _PhaseRuntimeState, block: str) -> bool:
    terms = _parse_search_rewrite_terms(block)
    if not block.strip() or not terms:
        return False
    phase_state.search_rewrite_xml = block.strip()
    phase_state.search_rewrite_terms = terms
    return True


def _search_rewrite_context_text(phase_state: _PhaseRuntimeState) -> str:
    if not phase_state.search_rewrite_terms:
        return ""
    lines = ["# Keyword Rewrite"]
    for index, term in enumerate(phase_state.search_rewrite_terms, start=1):
        lines.append(f"{index}. {term}")
    return "\n".join(lines).strip()


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
        title = str(row.get("title") or "").strip() or "No Title"
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
        lines.append("Skipped duplicate query from this session.")
        return "\n".join(lines).strip()
    if error:
        lines.append(error)
        return "\n".join(lines).strip()
    if not public_results:
        lines.append("No results.")
        return "\n".join(lines).strip()
    for index, item in enumerate(public_results, start=1):
        title = str(item.get("title") or "No Title").strip() or "No Title"
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
            "ok": True,
            "provider": _tool_provider_from_config("web_search", cfg),
            "query": query_text,
            "count": 0,
            "results": [],
            "skipped_duplicate": True,
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
    lines = [
        f"# Page: {url}" if url else "# Page",
        "",
    ]
    if title:
        lines.append(f"Title: {title}")
    if keywords and window != "all":
        lines.append(f"Keywords: {' | '.join(keywords)}")
    lines.append("Window: all lines" if window == "all" else f"Window: {window} lines")
    lines.append(f"Cache: {'hit' if from_cache else 'miss'}")
    if page_error:
        lines.extend(["", page_error])
        return "\n".join(lines).strip()
    if not matches:
        lines.extend(["", "No matching lines found in cached page content."])
        return "\n".join(lines).strip()
    lines.extend(["", "Matched lines:"])
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
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try: return fn()
            except Exception: pass
    return vars(obj) if hasattr(obj, "__dict__") else str(obj)


def _completion_extra_body(cfg: dict[str, Any]) -> dict[str, Any] | None:
    extra_body = cfg.get("extra_body")
    return deepcopy(extra_body) if isinstance(extra_body, dict) else None


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
        if kind == "reasoning.encrypted":
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

    meta = {
        "reasoning_tokens": _usage_reasoning_tokens(usage or {}),
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


def llm_call(messages, *, config, stats=None, trace_label="Model", log_id=None):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    # Fail fast on provider throttling instead of waiting through SDK retries.
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "drop_params": True, "max_retries": 0}
    if cfg.get("api_base"): kw["api_base"] = cfg["api_base"]
    if cfg.get("api_key"): kw["api_key"] = cfg["api_key"]
    _apply_completion_limits(cfg, kw)
    extra_body = _completion_extra_body(cfg)
    if extra_body:
        kw["extra_body"] = extra_body
    _apply_reasoning_options(cfg, kw)
    litellm_mod = _get_litellm()

    t0 = time.perf_counter()
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


def _build_main_loop_user_message(
    user_content: str | list[dict[str, Any]],
    *,
    round_i: int,
    max_rounds: int,
    tool_results_text: str = "",
) -> str | list[dict[str, Any]]:
    if tool_results_text:
        return tool_results_text
    del round_i, max_rounds
    prompt_body = _message_content_text(user_content) or "images"
    text_block = "\n\n".join(["# 原始用户问题", prompt_body]).strip()
    if isinstance(user_content, list):
        images = [
            deepcopy(item)
            for item in user_content
            if isinstance(item, dict) and str(item.get("type") or "").strip() == "image_url"
        ]
        if images:
            return [{"type": "text", "text": text_block}, *images]
    return text_block


def _select_main_loop_calls(
    calls: list[dict[str, Any]],
    *,
    round_i: int,
    max_rounds: int,
    phase: str,
) -> list[dict[str, Any]]:
    current = min(round_i + 1, max_rounds)
    if current >= max_rounds or phase == _PHASE_FINAL:
        return []

    selected: list[dict[str, Any]] = []
    seen_page_keys: set[tuple[str, str, str]] = set()
    seen_search_queries: set[str] = set()
    # Search phase still defaults to search-first by prompt, but direct-link
    # questions may legitimately start with a page fetch in round 1.
    allowed_tools = {"web_search", "page_extract"}

    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name not in allowed_tools:
            continue

        if name == "web_search":
            query = str(args.get("query") or "").strip()
            normalized_query = _normalize_search_query(query)
            if not normalized_query or normalized_query in seen_search_queries:
                continue
            seen_search_queries.add(normalized_query)
            selected.append({"name": "web_search", "args": {"query": query}})
            if len(selected) >= _PHASE_TOOL_LIMIT:
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
            selected.append(
                {
                    "name": "page_extract",
                    "args": {
                        "url": normalized_url,
                        "query": query,
                        "lines": str(_normalize_page_window(lines)),
                    },
                }
            )
            if len(selected) >= _PHASE_TOOL_LIMIT:
                break
            continue

    return selected


def _tool_result_tag_xml(call: dict[str, Any], result_text: str) -> str:
    name = str(call.get("name") or "").strip()
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    attrs = [f'name="{name}"']
    tools = str(args.get("tools") or "").strip()
    url = str(args.get("url") or "").strip()
    lines = str(args.get("lines") or "").strip()
    if tools:
        attrs.append(f'tools="{tools.replace(chr(34), "&quot;")}"')
    if url:
        attrs.append(f'url="{url.replace(chr(34), "&quot;")}"')
    if lines:
        attrs.append(f'lines="{lines.replace(chr(34), "&quot;")}"')
    return f"<result {' '.join(attrs)}>\n{result_text}\n</result>"


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


def _execute_calls_parallel(
    *,
    calls: list[dict[str, Any]],
    cfg: dict[str, Any],
    stats: Stats | None,
    prompt_text: str,
    log_id: str | None,
    runtime_state: _SessionRuntimeState | None,
    started_at: float,
    on_tool: Any | None = None,
    progress_callback: Any | None = None,
) -> dict[int, dict[str, Any]]:
    if not calls:
        return {}

    results_map: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {
            pool.submit(
                execute_tool_payload,
                call["name"],
                call["args"],
                config=cfg,
                stats=stats,
                user_question=prompt_text,
                log_id=log_id,
                runtime_state=runtime_state,
                progress_callback=progress_callback,
            ): (idx, call)
            for idx, call in enumerate(calls)
        }
        for fut in as_completed(futures):
            idx, call = futures[fut]
            payload = fut.result()
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": "invalid tool payload"}
            _record_tool_stats(stats, payload)
            results_map[idx] = payload
            if callable(on_tool):
                try:
                    on_tool(
                        call["name"],
                        _tool_callback_args(
                            call["name"],
                            call["args"],
                            payload,
                            elapsed_s=time.perf_counter() - started_at,
                            config=cfg,
                        ),
                    )
                except Exception:
                    pass
    return results_map


def _results_xml_from_payloads(
    calls: list[dict[str, Any]],
    payload_map: dict[int, dict[str, Any]],
) -> str:
    parts: list[str] = []
    for index, call in enumerate(calls):
        payload = payload_map.get(index) or {"ok": False, "error": "missing tool payload"}
        parts.append(_tool_result_tag_xml(call, _tool_markdown_for_model(call["name"], call["args"], payload)))
    return "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"


@dataclass
class _TurnDecision:
    kind: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    assistant_content: str = ""
    user_content: str = ""
    final_text: str = ""
    next_phase: str = _PHASE_STAGE1
    store_assistant_turn: bool = True


def _phase_for_round(
    phase_state: _PhaseRuntimeState,
    *,
    round_i: int,
    max_rounds: int,
) -> str:
    if phase_state.phase == _PHASE_STAGE2 and round_i + 1 >= max_rounds:
        return _PHASE_FINAL
    return phase_state.phase


def _search_retry_message() -> str:
    return STAGE1_RETRY_PROMPT.strip()


def _skeleton_retry_message() -> str:
    return STAGE1_SKELETON_RETRY_PROMPT.strip()


def _execute_kickoff_message(phase_state: _PhaseRuntimeState) -> str:
    parts = [line for line in STAGE2_KICKOFF_PROMPT.strip().splitlines() if line.strip()]
    user_need_context = _user_need_context_text(phase_state)
    if user_need_context:
        parts.extend(["", user_need_context])
    search_rewrite_context = _search_rewrite_context_text(phase_state)
    if search_rewrite_context:
        parts.extend(["", search_rewrite_context])
    if phase_state.skeleton_xml:
        parts.extend(["", phase_state.skeleton_xml])
    checklist = _claim_checklist_text(phase_state)
    if checklist:
        parts.extend(["", checklist])
    return "\n".join(parts).strip()


def _execute_retry_message(*, final_soon: bool) -> str:
    message = STAGE2_RETRY_PROMPT.strip()
    if final_soon:
        message += " 下一轮将进入 final，请在这轮尽量补齐关键证据。"
    return message


def _reset_runtime_state_for_execute(runtime_state: _SessionRuntimeState | None) -> None:
    if runtime_state is None:
        return
    with runtime_state.lock:
        runtime_state.search_history_raw.clear()
        runtime_state.search_history_normalized.clear()
        runtime_state.search_results_deduped.clear()


def _build_stage2_phase_messages(
    *,
    cfg: dict[str, Any],
    prompt_text: str,
    phase_state: _PhaseRuntimeState,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": _build_system_prompt(cfg, prompt_text, phase=_PHASE_STAGE2, phase_state=phase_state),
        },
        {
            "role": "user",
            "content": _execute_kickoff_message(phase_state),
        },
    ]


def _record_first_collection_mode(
    phase_state: _PhaseRuntimeState,
    calls: list[dict[str, Any]],
) -> None:
    if str(phase_state.first_collection_mode or "").strip():
        return
    has_search = any(str(call.get("name") or "").strip() == "web_search" for call in calls if isinstance(call, dict))
    has_page = any(str(call.get("name") or "").strip() == "page_extract" for call in calls if isinstance(call, dict))
    if has_search and has_page:
        phase_state.first_collection_mode = "mix"
    elif has_page:
        phase_state.first_collection_mode = "page"
    elif has_search:
        phase_state.first_collection_mode = "websearch"


def _store_assistant_turn(
    msgs: list[dict[str, Any]],
    msg: Any,
    *,
    content: str,
    reasoning_text: str = "",
    reasoning_details: list[dict[str, Any]] | None = None,
    reasoning_text_parts: list[str] | None = None,
) -> None:
    if reasoning_text_parts is not None:
        msgs.append(
            _assistant_message_with_reasoning(
                None,
                content=content,
                reasoning_text_parts=reasoning_text_parts,
                reasoning_details=reasoning_details,
            )
        )
        return
    msgs.append(
        _assistant_message_with_reasoning(
            msg,
            content=content,
        )
    )


def _decide_turn(
    *,
    text: str,
    phase: str,
    round_i: int,
    max_rounds: int,
    phase_state: _PhaseRuntimeState,
    cfg: dict[str, Any],
) -> _TurnDecision:
    skeleton_block = _extract_article_skeleton(text)
    user_need_block = _extract_user_need_restore(text)
    search_rewrite_block = _extract_search_rewrite(text)
    raw_calls = _parse_tool_tags(text)
    calls = _select_main_loop_calls(raw_calls, round_i=round_i, max_rounds=max_rounds, phase=phase)
    _ensure_call_ids(calls, round_i=round_i)
    clean = _strip_tool_tags(text)
    nonfinal_content = _strip_tool_xml_only(text) or clean or text

    if phase == _PHASE_STAGE1:
        if calls:
            _record_first_collection_mode(phase_state, calls)
            return _TurnDecision(
                kind="tools",
                calls=calls,
                assistant_content=_normalize_tool_turn_message(text, calls),
                next_phase=_PHASE_STAGE1,
            )
        has_user_need = _update_user_need_state(phase_state, user_need_block) if user_need_block else bool(phase_state.user_need_items)
        has_rewrite = _update_search_rewrite_state(phase_state, search_rewrite_block) if search_rewrite_block else bool(phase_state.search_rewrite_terms)
        has_skeleton = _update_skeleton_state(phase_state, skeleton_block) if skeleton_block else bool(phase_state.claims)
        if has_user_need and has_rewrite and has_skeleton:
            return _TurnDecision(
                kind="continue",
                assistant_content=nonfinal_content,
                user_content=_execute_kickoff_message(phase_state),
                next_phase=_PHASE_STAGE2,
            )
        if not phase_state.first_collection_mode and round_i == 0 and clean:
            return _TurnDecision(
                kind="final",
                final_text=clean,
                next_phase=_PHASE_FINAL,
            )
        return _TurnDecision(
            kind="continue",
            assistant_content="",
            user_content=_skeleton_retry_message(),
            next_phase=_PHASE_STAGE1,
            store_assistant_turn=False,
        )

    if phase == _PHASE_STAGE2:
        if user_need_block:
            _update_user_need_state(phase_state, user_need_block)
        if search_rewrite_block:
            _update_search_rewrite_state(phase_state, search_rewrite_block)
        if skeleton_block:
            _update_skeleton_state(phase_state, skeleton_block)
            return _TurnDecision(
                kind="continue",
                assistant_content=nonfinal_content,
                user_content=_execute_kickoff_message(phase_state),
                next_phase=_PHASE_STAGE2,
            )
        if calls:
            return _TurnDecision(
                kind="tools",
                calls=calls,
                assistant_content=_normalize_tool_turn_message(text, calls),
                next_phase=_PHASE_FINAL if round_i + 2 >= max_rounds else _PHASE_STAGE2,
            )
        if not clean:
            final_soon = round_i + 2 >= max_rounds
            return _TurnDecision(
                kind="continue",
                assistant_content=nonfinal_content,
                user_content=_execute_retry_message(final_soon=final_soon),
                next_phase=_PHASE_FINAL if final_soon else _PHASE_STAGE2,
            )
        return _TurnDecision(
            kind="final",
            final_text=clean or _format_empty_output_message(cfg, round_i=round_i),
            next_phase=_PHASE_FINAL,
        )

    return _TurnDecision(
        kind="final",
        final_text=clean or _format_empty_output_message(cfg, round_i=round_i),
        next_phase=_PHASE_FINAL,
    )


def run(
    question: str,
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    on_tool: Any | None = None,
    on_reasoning: Any | None = None,
    images: list[str] | None = None,
    context: str | list[dict[str, Any]] | None = None,
) -> str:
    """单次问答: 多轮 XML 标签工具调用循环, 返回最终文本.

    context — 上一轮对话消息, 用于多轮对话上下文传递.
    """
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
    phase_state = _PhaseRuntimeState()
    phase_state.input_has_images = bool(image_paths)
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

    msgs: list[dict] = [
        {
            "role": "system",
            "content": _build_system_prompt(cfg, prompt_text, phase=phase_state.phase, phase_state=phase_state),
        }
    ]
    _append_context_messages(msgs, context)
    msgs.append(
        {
            "role": "user",
            "content": _build_main_loop_user_message(
                user_content,
                round_i=0,
                max_rounds=max_rounds,
            ),
        }
    )
    last = ""

    for round_i in range(max_rounds):
        phase = _phase_for_round(phase_state, round_i=round_i, max_rounds=max_rounds)
        if phase == _PHASE_STAGE2:
            phase_state.stage2_turns += 1
        phase_cfg = build_stage_model_config(cfg, phase)
        msgs[0]["content"] = _build_system_prompt(cfg, prompt_text, phase=phase, phase_state=phase_state)
        try:
            resp = llm_call(msgs, config=phase_cfg, stats=st, trace_label=f"round {round_i + 1}", log_id=lid)
        except Exception as e:
            return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)

        choices = getattr(resp, "choices", None) or []
        msg = choices[0].message if choices else None
        if msg is None: break
        usage = _to_dict(getattr(resp, "usage", None) or {})
        if not isinstance(usage, dict):
            usage = {}

        text = _text(msg)
        if text: last = text

        reasoning_text, reasoning_meta = _render_reasoning_display(
            payload=_to_dict(msg),
            usage=usage,
        )
        if callable(on_reasoning):
            try:
                on_reasoning(reasoning_text, reasoning_meta)
            except Exception:
                pass

        decision = _decide_turn(
            text=text,
            phase=phase,
            round_i=round_i,
            max_rounds=max_rounds,
            phase_state=phase_state,
            cfg=cfg,
        )

        if decision.kind == "final":
            final_message = decision.final_text or last or _format_empty_output_message(phase_cfg, round_i=round_i)
            return _prepend_runtime_warnings(final_message, warning_messages)

        if decision.store_assistant_turn:
            _store_assistant_turn(
                msgs,
                msg,
                content=decision.assistant_content or (text or last or "继续"),
            )
        phase_transition_to_stage2 = phase != _PHASE_STAGE2 and decision.next_phase == _PHASE_STAGE2
        phase_state.phase = decision.next_phase

        if phase_transition_to_stage2:
            _reset_runtime_state_for_execute(runtime_state)
            phase_state.stage2_turns = 0
            msgs = _build_stage2_phase_messages(
                cfg=cfg,
                prompt_text=prompt_text,
                phase_state=phase_state,
            )
            continue

        if decision.kind != "tools":
            if decision.user_content:
                msgs.append({"role": "user", "content": decision.user_content})
            continue

        payload_map = _execute_calls_parallel(
            calls=decision.calls,
            cfg=cfg,
            stats=st,
            prompt_text=prompt_text,
            log_id=lid,
            runtime_state=runtime_state,
            started_at=started_at,
            on_tool=on_tool,
            progress_callback=None,
        )
        results_xml = _results_xml_from_payloads(decision.calls, payload_map)
        next_tool_results_text = _tool_results_for_next_round(results_xml, round_i=round_i, max_rounds=max_rounds)
        msgs.append({"role": "user", "content": next_tool_results_text})

    final_message = _strip_tool_tags(last) or _format_empty_output_message(cfg, round_i=max_rounds - 1)
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
) -> str:
    """流式对话循环, 通过回调驱动 CLI 显示.

    on_chunk(delta)  — 每个 token 到达时调用 (实时流式)
    on_rewind(thinking, tools) — 工具轮检测到后调用, thinking 为去标签文本, tools 为即将执行的工具列表
    on_tool(name, args) — 工具调用回调
    on_status(text)  — 状态回调: "Preparing...", "Thinking...", "Searching..."
    context — 上一轮对话消息, 用于多轮对话上下文传递.

    Returns 最终清理后的回答文本.
    """
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
    phase_state = _PhaseRuntimeState()
    phase_state.input_has_images = bool(image_paths)
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

    msgs: list[dict] = [
        {
            "role": "system",
            "content": _build_system_prompt(cfg, prompt_text, phase=phase_state.phase, phase_state=phase_state),
        },
    ]
    _append_context_messages(msgs, context)
    msgs.append(
        {
            "role": "user",
            "content": _build_main_loop_user_message(
                user_content,
                round_i=0,
                max_rounds=max_rounds,
            ),
        }
    )
    for round_i in range(max_rounds):
        phase = _phase_for_round(phase_state, round_i=round_i, max_rounds=max_rounds)
        if phase == _PHASE_STAGE2:
            phase_state.stage2_turns += 1
        phase_cfg = build_stage_model_config(cfg, phase)
        msgs[0]["content"] = _build_system_prompt(cfg, prompt_text, phase=phase, phase_state=phase_state)
        litellm_mod = _get_litellm(on_status=on_status)
        if callable(on_status):
            on_status(STATUS_THINKING)

        model = str(phase_cfg.get("model") or DEFAULT_MODEL).strip()
        kw: dict[str, Any] = {
            "model": model, "messages": msgs, "temperature": 0.2,
            "stream": True, "drop_params": True,
            "stream_options": {"include_usage": True},
            "max_retries": 0,
        }
        if phase_cfg.get("api_base"):
            kw["api_base"] = phase_cfg["api_base"]
        if phase_cfg.get("api_key"):
            kw["api_key"] = phase_cfg["api_key"]
        _apply_completion_limits(phase_cfg, kw)
        extra_body = _completion_extra_body(phase_cfg)
        if extra_body:
            kw["extra_body"] = extra_body
        _apply_reasoning_options(phase_cfg, kw)

        t0 = time.perf_counter()
        try:
            stream = litellm_mod.completion(**kw)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log_model_call(
                label=f"round {round_i + 1}", model=model, messages=msgs,
                output="", error=str(e)[:300], duration_ms=duration_ms, config=phase_cfg,
                log_id=lid,
            )
            return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)

        # ── 实时流式: 边收 chunk 边推给 CLI ──
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        reasoning_text_parts: list[str] = []
        reasoning_details: list[dict[str, Any]] = []
        last_reasoning_emit_tokens = [0]
        try:
            for chunk in stream:
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
                    content_parts.append(c)
                    if callable(on_chunk):
                        try:
                            on_chunk(c)
                        except Exception:
                            pass
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
                config=phase_cfg,
                log_id=lid,
            )
            return _prepend_runtime_warnings(_format_model_error_message(e), warning_messages)

        duration_ms = (time.perf_counter() - t0) * 1000
        full_text = "".join(content_parts)

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

        # Log
        log_model_call(
            label=f"round {round_i + 1}", model=model, messages=msgs,
            output=full_text,
            usage=usage, cost=cost, duration_ms=duration_ms, config=phase_cfg,
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

        decision = _decide_turn(
            text=full_text,
            phase=phase,
            round_i=round_i,
            max_rounds=max_rounds,
            phase_state=phase_state,
            cfg=cfg,
        )

        if decision.kind == "final":
            final_message = decision.final_text or _format_empty_output_message(phase_cfg, round_i=round_i)
            return _prepend_runtime_warnings(final_message, warning_messages)

        if callable(on_rewind):
            rewind_tools = None
            if decision.kind == "tools":
                rewind_tools = [
                    (
                        call["name"],
                        _tool_preview_callback_args(call["name"], call["args"], config=cfg),
                    )
                    for call in decision.calls
                ]
            try:
                on_rewind(decision.assistant_content or full_text, rewind_tools)
            except Exception:
                pass

        if decision.store_assistant_turn:
            _store_assistant_turn(
                msgs,
                None,
                content=decision.assistant_content or (full_text or "继续"),
                reasoning_text_parts=reasoning_text_parts,
                reasoning_details=reasoning_details,
            )
        phase_transition_to_stage2 = phase != _PHASE_STAGE2 and decision.next_phase == _PHASE_STAGE2
        phase_state.phase = decision.next_phase

        if phase_transition_to_stage2:
            _reset_runtime_state_for_execute(runtime_state)
            phase_state.stage2_turns = 0
            msgs = _build_stage2_phase_messages(
                cfg=cfg,
                prompt_text=prompt_text,
                phase_state=phase_state,
            )
            continue

        if decision.kind != "tools":
            if decision.user_content:
                msgs.append({"role": "user", "content": decision.user_content})
            continue

        if callable(on_status):
            on_status(STATUS_SEARCHING)
        payload_map = _execute_calls_parallel(
            calls=decision.calls,
            cfg=cfg,
            stats=st,
            prompt_text=prompt_text,
            log_id=lid,
            runtime_state=runtime_state,
            started_at=started_at,
            on_tool=on_tool,
            progress_callback=on_tool if callable(on_tool) else None,
        )
        results_xml = _results_xml_from_payloads(decision.calls, payload_map)
        next_tool_results_text = _tool_results_for_next_round(results_xml, round_i=round_i, max_rounds=max_rounds)
        msgs.append({"role": "user", "content": next_tool_results_text})

    return _prepend_runtime_warnings(_format_empty_output_message(cfg, round_i=max_rounds - 1), warning_messages)
