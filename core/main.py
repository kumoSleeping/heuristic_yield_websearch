"""
hyw/main.py - 极简 LLM 对话循环 + XML 标签工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web_search (自带)
配置: ~/.hyw/config.yml 纯穿透读取, 不做校验

工具调用方式: 模型在文本中输出 <search>/<wiki> XML 标签, 解析后执行工具, 注入结果再让模型继续.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
import yaml

# ── 静默 litellm ─────────────────────────────────────────────
litellm.suppress_debug_info = True
for _n in ("LiteLLM", "litellm", "litellm.utils", "httpx", "httpcore"):
    logging.getLogger(_n).setLevel(logging.ERROR)

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_NAME = "hyw"
CONFIG_PATH = Path.home() / ".hyw" / "config.yml"
LOG_DIR = Path.home() / ".hyw" / "logs"


# ── 配置: 纯穿透 ─────────────────────────────────────────────
def load_config() -> dict[str, Any]:
    """原样读 yaml, 不做校验/默认值填充."""
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text("utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        raw = {}
    return raw if isinstance(raw, dict) else {}


def cfg_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """点分路径取值: cfg_get(c, 'tools.web_search.enabled', True)"""
    cur: Any = config
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


# ── 统计 ──────────────────────────────────────────────────────

def _try_cost(model: str, pt: int, ct: int) -> float | None:
    """尝试多种模型名变体查询价格，返回总 cost 或 None."""
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
            pc, cc = litellm.cost_per_token(model=v, prompt_tokens=pt, completion_tokens=ct)
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

    def summary(self) -> str:
        cost = f"${self.cost_usd:.6f}" if self._has_cost else "N/A"
        return f"{self.calls}次 | {self.prompt_tokens}+{self.completion_tokens}={self.total_tokens}tok | {cost}"


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
            content = str(m.get("content") or "")
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
_suite_lock = threading.RLock()
_suite = None


def _get_suite(headless: bool = True):
    global _suite
    with _suite_lock:
        if _suite is None:
            from .xml_tools.web_search.server import WebToolSuite
            _suite = WebToolSuite(headless=headless)
        return _suite


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


def startup_tools(headless: bool = True):
    """预热浏览器截图服务."""
    from .xml_tools.web_search.runtime import on_startup
    on_startup(headless=headless)


def shutdown_tools():
    from .xml_tools.web_search.runtime import on_shutdown
    on_shutdown()


def _web_search(query: str, df: str = "", **_) -> dict[str, Any]:
    """调用 WebToolSuite 执行搜索."""
    suite = _get_suite()
    try:
        payload = _run_async(suite.web_search(
            query=query, mode="text",
            time_range=df or "",
            max_results=5,
        ))
    except Exception as e:
        return {"ok": False, "error": str(e)}

    rows = payload.get("results", []) if isinstance(payload, dict) else []
    results = [
        {"title": str(r.get("title", ""))[:200], "url": str(r.get("url", ""))[:400], "snippet": str(r.get("snippet") or r.get("intro") or "")[:240]}
        for r in rows if isinstance(r, dict)
    ]
    return {"ok": True, "query": query, "count": len(results), "results": results}


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name in ("web_search", "web_search_wiki"):
        r = _web_search(
            query=args.get("query", ""),
            df=args.get("df", "") if name == "web_search" else "",
        )
    else:
        r = {"ok": False, "error": f"unknown tool: {name}"}
    return json.dumps(r, ensure_ascii=False, indent=2)


# ── XML 标签工具调用解析 ─────────────────────────────────────
_TOOL_TAG_RE = re.compile(r"<(search|wiki)\b([^>]*)>(.*?)</\1>", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


def _parse_tool_tags(text: str) -> list[dict]:
    """提取 <search df="...">query</search> 和 <wiki>query</wiki>，最多4个"""
    # 先去掉 `...` 内联代码，避免匹配到模型解释文本中的标签
    cleaned = _INLINE_CODE_RE.sub("", text)
    calls: list[dict] = []
    for m in _TOOL_TAG_RE.finditer(cleaned):
        tag, attrs_str, query = m.group(1), m.group(2), m.group(3).strip()
        if not query:
            continue
        name = "web_search_wiki" if tag == "wiki" else "web_search"
        args: dict[str, str] = {"query": query}
        if tag == "search" and attrs_str.strip():
            try:
                el = ET.fromstring(f"<{tag}{attrs_str}></{tag}>")
                df = (el.get("df") or "").strip()
                if df:
                    args["df"] = df
            except ET.ParseError:
                pass
        calls.append({"name": name, "args": args})
        if len(calls) >= 4:
            break
    return calls


def _strip_tool_tags(text: str) -> str:
    """从最终回答中去掉工具标签"""
    return _TOOL_TAG_RE.sub("", text).strip()


SYSTEM_PROMPT = """\
# 你的身份
You are {name}.
Current time: {time}, 这是一个很重要的信息, 请务必在回答中考虑到时间因素带来的信息滞后问题, 你是一个互联网实时信息查询助手, 可以使用工具来获取最新的信息, 以弥补你的知识截止日期带来的信息滞后问题.
你需要分析用户发给你的这句话并从中识别任务或问题, 若无明显任务则默认任务为解释这句话 / 这句话中的关键词.
拒绝不合理、任何违法、违规、政治敏感、伦理道德有争议的内容.


## 工作流程:
1.默认描述任务拆解+搜索计划, 随后像用户通过xml标签的发送调用请求, 等待用户计算机执行工具并返回结果.
- 工具调用过程对用户的回复: 2-3句话, 不使用如何 markdown 格式, 包括粗体, 语言简介不啰嗦, 分享欲望低.
2.验证工具结果, 同时必须继续拆解更细的搜索词进行第二轮工具调用, 向用户解释后, 加入xml调用工具, 继续等待用户计算机执行工具并返回结果.
3.在第三轮若判断已经能解决问题, 开始最终回复:
- 以 `# 标题` 开头
- 标题下方包含：<summary>1-2 句核心摘要</summary>
- Preferred language: {language}
- Custom prompt: {custom}
- Never 在最终缓解推测.
- 对怀疑的内容分享欲望较低, 绝不分享拿不准的消息.

## 工具
> 硬性规则1: 用户的原话为: {user_message}, 在构建、切分搜索词的时候禁止改变用户原文中任何一个词, 包括 语意扩充、翻译、擅自添加领域... 防止滚雪球效应发送, 从最开始就偏离了用户意图.
> 硬性规则2: 每次搜索需要保证搜索词交错: 减少多次搜索指向相同结果
> 硬性规则3: 不搜索低质量内容、敏感词、忽略用户消息图片中的角色
由于模型被设定为: 对怀疑的内容分享欲望较低, 在第一轮工具给出合并结果后, 如果信息不足, 可以自主继续切分更细的搜索词进行第二轮工具调用, 以提升召回率.

工具调用过程回复: 
基于待解决的每个子问题：分别进行多条不同的、更适合搜索引擎的简短关键词组合查询查询 (本轮对话调用最多4个搜索工具)，以扩大搜索的召回率，去除多余的助词，搜索核心实体名词。
同时用2-3句话使用介绍要做的工具调用, 以让用户了解你的思路和计划, 这也有助于你自己理清思路, 例如:
- 我理解用户给我发送了...我会先搜索...绝不改变用户的原话中的词语, 以确认...最新含义是什么..., 同时为了不浪费时间, 我会一并搜索...以确认我的理解是否正确...
- 我几乎确定了...是表示...我再做一次交叉验证... 同时为了不浪费时间, 我会在...方向一并搜索...提升搜索的
- 已经很明确了...了, 但我还是想再确认一下...以避免错误回答。

在回复后嵌入以下 XML 标签调用工具（一次最多4个）：
<wiki>关键词</wiki>         — 去搜索用户原文词汇确认此词语最新的意义, 关键词必须原封不动的来自用户消息.
<search>搜索词</search>     — 弥补知识截止日期带来的信息滞后问题. 
<search df="2026-02-24..2026-03-12">搜索词</search>  — 带日期过滤, 推荐至少使用一次.

随后等待工具结果返回.

eg:
我会先从用户消息中提取关键词xxx、yyy，然后分别wiki一下xxx、yyy以确定在今天这个词语的最新含义是什么, 同时为了不浪费时间, 我会一并搜索xxx和yyy的联系, 以确认我的理解是否正确...
<wiki>xxx</wiki>
<search df="2026-02-24..2026-03-12">xxx 最新情报</search>
<search>xxx 剧情简介</search>

## 工作流程注意
通常情况下一轮的搜索结果往往不足以支持模型做出准确的回答, 根据工具返回结果, 继续分析是否继续工具调用, 直到你认为信息充分了才给出最终回答.
"""

TOOL_RESULTS_GUIDE = """\
以上是工具返回的结果。请分析这些结果.
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


# ── LLM 调用 ─────────────────────────────────────────────────
def _to_dict(obj: Any) -> Any:
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try: return fn()
            except Exception: pass
    return vars(obj) if hasattr(obj, "__dict__") else str(obj)


def llm_call(messages, *, config, stats=None, trace_label="Model", log_id=None):
    model = str(config.get("model") or DEFAULT_MODEL).strip()
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "drop_params": True}
    if config.get("api_base"): kw["api_base"] = config["api_base"]
    if config.get("api_key"): kw["api_key"] = config["api_key"]
    re_ = str(config.get("reasoning_effort") or "").strip().lower()
    if re_ in ("minimal", "low", "medium", "high"):
        kw["reasoning_effort"] = re_

    t0 = time.perf_counter()
    try:
        resp = litellm.completion(**kw)
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        log_model_call(
            label=trace_label, model=model, messages=messages,
            output="", error=str(e)[:300], duration_ms=duration_ms, config=config,
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
        usage=usage, cost=cost, duration_ms=duration_ms, config=config,
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
    context: str | None = None,
) -> str:
    """单次问答: 多轮 XML 标签工具调用循环, 返回最终文本.

    context — 上一轮 AI 总结, 用于多轮对话上下文传递.
    """
    if not question.strip():
        return ""
    cfg = config or load_config()
    st = stats or Stats()
    max_rounds = max(1, int(cfg.get("max_rounds") or 6))
    lid = _make_log_id(question)

    msgs: list[dict] = [{"role": "system", "content": _build_system_prompt(cfg, question)}]
    if context:
        msgs.append({"role": "assistant", "content": context})
    msgs.append({"role": "user", "content": question})
    last = ""

    for round_i in range(max_rounds):
        try:
            resp = llm_call(msgs, config=cfg, stats=st, trace_label=f"round {round_i + 1}", log_id=lid)
        except Exception as e:
            err = str(e or "").strip()
            for frag in ("APIError:", "Exception -"):
                if frag in err:
                    err = err[err.index(frag) + len(frag):].strip()
                    break
            if len(err) > 200:
                err = err[:200]
            return f"[模型调用失败] {err}"

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
        for c in calls:
            if callable(on_tool):
                try: on_tool(c["name"], c["args"])
                except Exception: pass
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(execute_tool, c["name"], c["args"]): c for c in calls}
            results_map: dict[int, str] = {}
            for fut in as_completed(futures):
                c = futures[fut]
                idx = calls.index(c)
                results_map[idx] = fut.result()
        parts = [
            f'<result name="{calls[i]["name"]}" query="{calls[i]["args"]["query"]}">\n{results_map[i]}\n</result>'
            for i in range(len(calls))
        ]
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
    context: str | None = None,
) -> str:
    """流式对话循环, 通过回调驱动 CLI 显示.

    on_chunk(delta)  — 每个 token 到达时调用 (实时流式)
    on_rewind()      — 工具轮检测到后调用, CLI 应丢弃已显示的文本
    on_tool(name, args) — 工具调用回调
    on_status(text)  — 状态回调: "模型思考中...", "搜索中..."
    context — 上一轮 AI 总结, 用于多轮对话上下文传递.

    Returns 最终清理后的回答文本.
    """
    if not question.strip():
        return ""
    cfg = config or load_config()
    st = stats or Stats()
    max_rounds = max(1, int(cfg.get("max_rounds") or 6))
    lid = _make_log_id(question)

    msgs: list[dict] = [
        {"role": "system", "content": _build_system_prompt(cfg, question)},
    ]
    if context:
        msgs.append({"role": "assistant", "content": context})
    msgs.append({"role": "user", "content": question})

    for round_i in range(max_rounds):
        if callable(on_status):
            on_status("模型思考中...")

        model = str(cfg.get("model") or DEFAULT_MODEL).strip()
        kw: dict[str, Any] = {
            "model": model, "messages": msgs, "temperature": 0.2,
            "stream": True, "drop_params": True,
            "stream_options": {"include_usage": True},
        }
        if cfg.get("api_base"):
            kw["api_base"] = cfg["api_base"]
        if cfg.get("api_key"):
            kw["api_key"] = cfg["api_key"]
        re_ = str(cfg.get("reasoning_effort") or "").strip().lower()
        if re_ in ("minimal", "low", "medium", "high"):
            kw["reasoning_effort"] = re_

        t0 = time.perf_counter()
        try:
            stream = litellm.completion(**kw)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log_model_call(
                label=f"round {round_i + 1}", model=model, messages=msgs,
                output="", error=str(e)[:300], duration_ms=duration_ms, config=cfg,
                log_id=lid,
            )
            err = str(e or "").strip()
            for frag in ("APIError:", "Exception -"):
                if frag in err:
                    err = err[err.index(frag) + len(frag):].strip()
                    break
            if len(err) > 200:
                err = err[:200]
            return f"[模型调用失败] {err}"

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
        if callable(on_rewind):
            try:
                on_rewind(_strip_tool_tags(full_text))
            except Exception:
                pass

        msgs.append({"role": "assistant", "content": full_text})

        for tc in calls:
            if callable(on_tool):
                try:
                    on_tool(tc["name"], tc["args"])
                except Exception:
                    pass
        if callable(on_status):
            on_status("搜索中...")
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(execute_tool, tc["name"], tc["args"]): tc for tc in calls}
            results_map: dict[int, str] = {}
            for fut in as_completed(futures):
                tc = futures[fut]
                idx = calls.index(tc)
                results_map[idx] = fut.result()
        parts = [
            f'<result name="{calls[i]["name"]}" query="{calls[i]["args"]["query"]}">\n{results_map[i]}\n</result>'
            for i in range(len(calls))
        ]
        results_xml = "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"
        msgs.append({"role": "user", "content": results_xml + "\n\n" + TOOL_RESULTS_GUIDE})

    return "未获得有效回答。"
