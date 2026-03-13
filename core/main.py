"""
hyw/main.py - 极简 LLM 对话循环 + XML 标签工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web_search (自带)
配置: ~/.hyw/config.yml, 兼容单模型与多模型写法

工具调用方式: 模型在文本中输出 <search>/<wiki> XML 标签, 解析后执行工具, 注入结果再让模型继续.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    LOG_DIR,
    build_model_config,
    load_config,
    resolve_tool_handlers,
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
    variants = [model]
    # openrouter/google/xxx → gemini/xxx
    if model.startswith("openrouter/google/"):
        variants.append("gemini/" + model.split("/", 2)[-1])
        variants.append(model.split("/", 2)[-1])
    # openrouter/provider/xxx → provider/xxx, xxx
    elif model.startswith("openrouter/"):
        rest = model[len("openrouter/"):]
        variants.append(rest)
        if "/" in rest:
            variants.append(rest.split("/", 1)[-1])
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
    total_tokens: int = 0
    cost_usd: float = 0.0
    jina_tokens: int = 0
    jina_calls: int = 0
    _has_cost: bool = False

    def record(self, usage: dict[str, Any], cost: float | None):
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        t = int(usage.get("total_tokens") or 0)
        self.total_tokens += max(t, self.prompt_tokens + self.completion_tokens - (self.total_tokens - t if t else 0))
        if cost is not None:
            self.cost_usd += cost
            self._has_cost = True

    def record_jina(self, *, tokens: int = 0, requests: int = 1):
        self.jina_tokens += max(0, int(tokens or 0))
        self.jina_calls += max(0, int(requests or 0))

    def summary(self) -> str:
        cost = f"${self.cost_usd:.6f}" if self._has_cost else "N/A"
        return (
            f"{self.calls}次 | "
            f"{self.prompt_tokens}+{self.completion_tokens}={self.total_tokens}tok | "
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
            if len(content) > 500:
                content = content[:500] + "..."
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


def _web_search(query: str, **_) -> dict[str, Any]:
    """调用 WebToolSuite 执行搜索."""
    from .web_search import web_search

    cfg = load_config()
    try:
        payload = _run_async(web_search(
            query=query, mode="text",
            max_results=5,
            config=cfg,
            headless=bool(cfg.get("headless") not in (False, "false", "0", 0)),
        ))
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


def _page_extract(url: str, **_) -> dict[str, Any]:
    from .web_search import page_extract

    cfg = load_config()
    try:
        payload = _run_async(page_extract(
            url=url,
            max_chars=8000,
            config=cfg,
            headless=cfg.get("headless") not in (False, "false", "0", 0),
        ))
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid extract payload", "url": url}

    return {
        "ok": bool(payload.get("ok")),
        "provider": str(payload.get("provider") or "").strip(),
        "title": str(payload.get("title") or "")[:200],
        "url": str(payload.get("url") or url)[:400],
        "content": str(payload.get("content") or "")[:8000],
        "_meta": dict(payload.get("_meta") or {}) if isinstance(payload.get("_meta"), dict) else {},
    }


def execute_tool_payload(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name in ("web_search", "web_search_wiki"):
        r = _web_search(query=args.get("query", ""))
    elif name == "page_extract":
        r = _page_extract(url=args.get("url", ""))
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
        "web_search": "search",
        "web_search_wiki": "wiki",
        "page_extract": "page",
    }.get(name, str(name or "").strip())
    prefix = _tool_provider_label(provider)
    return f"{prefix}_{suffix}" if prefix else suffix


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
        meta = _tool_meta(payload)
        usage = meta.get("usage")
        if isinstance(usage, dict) and usage.get("tokens") not in (None, ""):
            merged["_jina_tokens"] = usage.get("tokens")
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


def _parse_tool_tags(text: str) -> list[dict]:
    """提取 <search>/<wiki>/<page> XML 标签，最多4个"""
    # 先去掉 `...` 内联代码，避免匹配到模型解释文本中的标签
    cleaned = _INLINE_CODE_RE.sub("", text)
    calls: list[dict] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for m in _TOOL_TAG_RE.finditer(cleaned):
        tag, _, query = m.group(1), m.group(2), m.group(3).strip()
        if not query:
            continue
        if tag == "page":
            name = "page_extract"
            args = {"url": query}
        else:
            name = "web_search_wiki" if tag == "wiki" else "web_search"
            args = {"query": query}
        signature = (
            name,
            tuple(sorted((str(key), str(value)) for key, value in args.items())),
        )
        if signature in seen:
            continue
        seen.add(signature)
        calls.append({"name": name, "args": args})
        if len(calls) >= 4:
            break
    return calls


def _strip_tool_tags(text: str) -> str:
    """从最终回答中去掉工具标签"""
    return _TOOL_TAG_RE.sub("", text).strip()


SYSTEM_PROMPT = """\
# 你的身份
- You are {name}.
- Current time: {time}, 这是一个很重要的信息, 请务必在回答中考虑到时间因素带来的信息滞后问题, 你是一个互联网实时信息查询助手, 可以使用工具来获取最新的信息, 以弥补你的知识截止日期带来的信息滞后问题.
- 你需要分析用户发给你的这句话并从中识别任务或问题, 若无明显任务则默认任务为: 解释这句话 / 这句话中的关键词.
- 拒绝: 不合理、任何违法、违规、政治敏感、伦理道德有争议的内容.
- 最终产出: 一份简洁精准、资料完全来源于用户计算机返回的报告.

## 输出规范(2选1)
- 在没完成信息整理之前:
    - 输出一个工具调用前对用户的说明: 
    - 2-3句话, 不使用如何 markdown 格式, 不使用粗体, 语言简介不啰嗦, 分享欲望低.
    - 随后输出 XML 格式的工具调用:
        - 基于待解决的每个子问题：分别进行多条不同的、更适合搜索引擎的简短关键词组合查询查询，以扩大搜索的召回率，去除多余的助词，搜索核心实体名词.
        - 在回复后嵌入以下 XML 标签调用工具（本轮对话调用最多 4 个搜索工具):
            - <wiki>关键词</wiki>         — 去搜索用户原文词汇确认此词语最新的意义, 关键词必须原封不动的来自用户消息.
            - <search>搜索词</search>     — 弥补知识截止日期带来的信息滞后问题. 
            - <page>https://example.com/article</page> — 提取单个页面正文, 当搜索结果里已经拿到可信 URL, 需要精确内容时必须使用.
            - 规范:
                - wiki、search 规则1: 用户的原话为: {user_message}, 在构建、切分搜索词的时候禁止改变用户原文中任何一个词, 包括 语意扩充、翻译、擅自添加领域... 防止滚雪球效应发送, 从最开始就偏离了用户意图.
                - wiki、search 规则2: 每次搜索需要保证搜索词交错: 减少多次搜索指向相同结果
                - wiki、search 规则3: 不搜索低质量、描述模糊、敏感词、忽略用户消息图片中的角色、对用户图片消息谨慎判断重点搜索内容
                - wiki、search 规则4: 对于专业性知识推荐搜索的时候额外追加一条搜索 带有相关专业网站的, 例如查询工具类: github、动漫类: 萌娘百科、我的世界: mcwiki/mcmod...以此类推
                - page 规则1: 在 wiki、search 后使用, 开销大但精准的 page 工具获取精准信息, 优先选择可信、有消信息多的页面.
                - page 规则2: 永远不访问知乎、csdn 等二手转载平台、相信官方页面、正规 wiki、文章平台.
        - 随后停止输出.
    - 推荐三种工具按顺序合理混合使用以最大化回答精准度, 下面是一些不同情况的最佳实践:
    - eg:
        我会先从用户消息中提取关键词xxx、yyy，然后分别 wiki 下xxx、yyy以确定在今天这个词语的最新含义是什么, 同时为了不浪费时间, 我会一并 search xxx和yyy的联系, 以确认我的理解是否正确...
        <wiki>xxx</wiki>
        <search>xxx 最新情报</search>
        <search>xxx 官网</search>
    - eg:
        用户的消息中存在多个待确认词汇, 我会先使用多个 wiki 工具确定...、...以及...的意思, 随后继续探究...和...在...场景的关系
        <wiki>xxx</wiki>
        <wiki>yyy</wiki>
        <wiki>zzz</wiki>
    - eg:
        我会先看您发给我的链接, 随后觉得需不需要补充知识..
        <page>https://example.com/article</page>
    - eg:
        通过上一轮的搜索, 我确定了xxx是xxx, 用户想要了解...接下来我将访问...获取最精准的消息...
        <page>https://example.com/article</page>
                
- 在完成信息整理之后, 最终正式回复:
    - 以 `# 标题` 开头
    - 标题下方包含：1-2 句核心摘要
    - 随后是 markdown 格式的丰富报告样式的正文, 简洁精准不啰嗦
    - Preferred language: {language}
    - Custom prompt: {custom}
    - 对怀疑的内容分享欲望较低, 绝不分享拿不准的消息.
    - 随后不再调用任何工具, 停止输出.

"""

TOOL_RESULTS_GUIDE = """\
以上是工具返回的 Markdown 结果。请分析这些结果。
"""


def _build_system_prompt(cfg: dict, user_message: str = "") -> str:
    custom = str(cfg.get("system_prompt") or "").strip()
    name = str(cfg.get("name") or DEFAULT_NAME).strip()
    return SYSTEM_PROMPT.format(
        name=name,
        language=cfg.get("language") or "zh-CN",
        time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        custom=(custom + "\n") if custom else "",
        user_message=user_message,
    )


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
    return f"[模型调用失败] {err}"


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
    re_ = str(cfg.get("reasoning_effort") or "").strip().lower()
    if re_ in ("minimal", "low", "medium", "high"):
        kw["reasoning_effort"] = re_
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
    if pt or ct:
        cost = _try_cost(model, pt, ct)
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


def run(
    question: str,
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    on_tool: Any | None = None,
    images: list[str] | None = None,
    context: str | None = None,
) -> str:
    """单次问答: 多轮 XML 标签工具调用循环, 返回最终文本.

    context — 上一轮 AI 总结, 用于多轮对话上下文传递.
    """
    image_paths = [str(path).strip() for path in (images or []) if str(path).strip()]
    if not question.strip() and not image_paths:
        return ""
    cfg = build_model_config(config or load_config())
    st = stats or Stats()
    max_rounds = max(1, int(cfg.get("max_rounds") or 6))
    prompt_text = _effective_prompt_text(question, len(image_paths)) or "images"
    lid = _make_log_id(prompt_text)
    started_at = time.perf_counter()

    msgs: list[dict] = [{"role": "system", "content": _build_system_prompt(cfg, prompt_text)}]
    if context:
        msgs.append({"role": "assistant", "content": context})
    msgs.append({"role": "user", "content": _build_multimodal_content(question, image_paths)})
    last = ""

    for round_i in range(max_rounds):
        try:
            resp = llm_call(msgs, config=cfg, stats=st, trace_label=f"round {round_i + 1}", log_id=lid)
        except Exception as e:
            return _format_model_error_message(e)

        choices = getattr(resp, "choices", None) or []
        msg = choices[0].message if choices else None
        if msg is None: break

        text = _text(msg)
        if text: last = text

        # 解析 XML 工具标签
        calls = _parse_tool_tags(text)

        if not calls:
            return _strip_tool_tags(text) or last or "未获得有效回答。"

        # assistant 原文
        msgs.append({"role": "assistant", "content": text})

        # 执行工具，构建结果 (并发)
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {
                pool.submit(execute_tool_payload, call["name"], call["args"]): (idx, call)
                for idx, call in enumerate(calls)
            }
            results_map: dict[int, str] = {}
            callbacks_map: dict[int, tuple[str, dict[str, Any]]] = {}
            for fut in as_completed(futures):
                idx, c = futures[fut]
                payload = fut.result()
                if not isinstance(payload, dict):
                    payload = {"ok": False, "error": "invalid tool payload"}
                _record_tool_stats(st, payload)
                result_text = _tool_markdown_for_model(c["name"], c["args"], payload)
                results_map[idx] = result_text
                callbacks_map[idx] = (
                    c["name"],
                    _tool_callback_args(
                        c["name"],
                        c["args"],
                        payload,
                        elapsed_s=time.perf_counter() - started_at,
                        config=cfg,
                    ),
                )
            if callable(on_tool):
                for idx in range(len(calls)):
                    callback = callbacks_map.get(idx)
                    if not callback:
                        continue
                    try:
                        on_tool(*callback)
                    except Exception:
                        pass
        parts = []
        for i in range(len(calls)):
            attr_key = "url" if "url" in calls[i]["args"] else "query"
            attr_value = calls[i]["args"].get(attr_key, "")
            parts.append(
                f'<result name="{calls[i]["name"]}" {attr_key}="{attr_value}">\n{results_map[i]}\n</result>'
            )
        results_xml = "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"
        msgs.append({"role": "user", "content": results_xml + "\n\n" + TOOL_RESULTS_GUIDE})

    return _strip_tool_tags(last) or "未获得有效回答。"


# ── 流式对话循环 ──────────────────────────────────────────────
def run_stream(
    question: str,
    *,
    config: dict[str, Any] | None = None,
    stats: Stats | None = None,
    on_chunk: Any | None = None,
    on_tool: Any | None = None,
    on_status: Any | None = None,
    on_rewind: Any | None = None,
    images: list[str] | None = None,
    context: str | None = None,
) -> str:
    """流式对话循环, 通过回调驱动 CLI 显示.

    on_chunk(delta)  — 每个 token 到达时调用 (实时流式)
    on_rewind(thinking, tools) — 工具轮检测到后调用, thinking 为去标签文本, tools 为即将执行的工具列表
    on_tool(name, args) — 工具调用回调
    on_status(text)  — 状态回调: "Preparing...", "Thinking...", "Searching..."
    context — 上一轮 AI 总结, 用于多轮对话上下文传递.

    Returns 最终清理后的回答文本.
    """
    image_paths = [str(path).strip() for path in (images or []) if str(path).strip()]
    if not question.strip() and not image_paths:
        return ""
    cfg = build_model_config(config or load_config())
    st = stats or Stats()
    max_rounds = max(1, int(cfg.get("max_rounds") or 6))
    prompt_text = _effective_prompt_text(question, len(image_paths)) or "images"
    lid = _make_log_id(prompt_text)
    started_at = time.perf_counter()

    msgs: list[dict] = [
        {"role": "system", "content": _build_system_prompt(cfg, prompt_text)},
    ]
    if context:
        msgs.append({"role": "assistant", "content": context})
    msgs.append({"role": "user", "content": _build_multimodal_content(question, image_paths)})

    for round_i in range(max_rounds):
        litellm_mod = _get_litellm(on_status=on_status)
        if callable(on_status):
            on_status(STATUS_THINKING)

        model = str(cfg.get("model") or DEFAULT_MODEL).strip()
        kw: dict[str, Any] = {
            "model": model, "messages": msgs, "temperature": 0.2,
            "stream": True, "drop_params": True,
            "stream_options": {"include_usage": True},
            "max_retries": 0,
        }
        if cfg.get("api_base"):
            kw["api_base"] = cfg["api_base"]
        if cfg.get("api_key"):
            kw["api_key"] = cfg["api_key"]
        _apply_completion_limits(cfg, kw)
        extra_body = _completion_extra_body(cfg)
        if extra_body:
            kw["extra_body"] = extra_body
        re_ = str(cfg.get("reasoning_effort") or "").strip().lower()
        if re_ in ("minimal", "low", "medium", "high"):
            kw["reasoning_effort"] = re_

        t0 = time.perf_counter()
        try:
            stream = litellm_mod.completion(**kw)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log_model_call(
                label=f"round {round_i + 1}", model=model, messages=msgs,
                output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                log_id=lid,
            )
            return _format_model_error_message(e)

        # ── 实时流式: 边收 chunk 边推给 CLI ──
        content_parts: list[str] = []
        usage: dict[str, Any] = {}

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

            c = getattr(delta, "content", None)
            if c:
                content_parts.append(c)
                if callable(on_chunk):
                    try:
                        on_chunk(c)
                    except Exception:
                        pass

        duration_ms = (time.perf_counter() - t0) * 1000
        full_text = "".join(content_parts)

        # Stats & cost
        if not isinstance(usage, dict):
            usage = {}
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost: float | None = None
        if pt or ct:
            cost = _try_cost(model, pt, ct)
        if st:
            st.record(usage, cost)

        # Log
        log_model_call(
            label=f"round {round_i + 1}", model=model, messages=msgs,
            output=full_text,
            usage=usage, cost=cost, duration_ms=duration_ms, config=cfg,
            log_id=lid,
        )

        # 解析工具标签
        calls = _parse_tool_tags(full_text)

        if not calls:
            # ── 最终回答 ──
            clean = _strip_tool_tags(full_text)
            return clean or "未获得有效回答。"

        # ── 工具轮: 通知 CLI 回退已显示的内容 ──
        planned_callbacks = [
            (
                call["name"],
                _tool_preview_callback_args(call["name"], call["args"], config=cfg),
            )
            for call in calls
        ]
        if callable(on_rewind):
            try:
                on_rewind(_strip_tool_tags(full_text), planned_callbacks)
            except Exception:
                pass

        msgs.append({"role": "assistant", "content": full_text})

        if callable(on_status):
            on_status(STATUS_SEARCHING)
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {
                pool.submit(execute_tool_payload, call["name"], call["args"]): (idx, call)
                for idx, call in enumerate(calls)
            }
            results_map: dict[int, str] = {}
            callbacks_map: dict[int, tuple[str, dict[str, Any]]] = {}
            for fut in as_completed(futures):
                idx, tc = futures[fut]
                payload = fut.result()
                if not isinstance(payload, dict):
                    payload = {"ok": False, "error": "invalid tool payload"}
                _record_tool_stats(st, payload)
                result_text = _tool_markdown_for_model(tc["name"], tc["args"], payload)
                results_map[idx] = result_text
                callbacks_map[idx] = (
                    tc["name"],
                    _tool_callback_args(
                        tc["name"],
                        tc["args"],
                        payload,
                        elapsed_s=time.perf_counter() - started_at,
                        config=cfg,
                    ),
                )
            if callable(on_tool):
                for idx in range(len(calls)):
                    callback = callbacks_map.get(idx)
                    if not callback:
                        continue
                    try:
                        on_tool(*callback)
                    except Exception:
                        pass
        parts = []
        for i in range(len(calls)):
            attr_key = "url" if "url" in calls[i]["args"] else "query"
            attr_value = calls[i]["args"].get(attr_key, "")
            parts.append(
                f'<result name="{calls[i]["name"]}" {attr_key}="{attr_value}">\n{results_map[i]}\n</result>'
            )
        results_xml = "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"
        msgs.append({"role": "user", "content": results_xml + "\n\n" + TOOL_RESULTS_GUIDE})

    return "未获得有效回答。"
