"""
hyw/main.py - 极简 LLM 对话循环 + XML 标签工具调用 + 统计 + 调用日志

依赖: litellm, hyw/web_search (自带)
配置: ~/.hyw/config.yml, 兼容单模型与多模型写法

工具调用方式: 模型在文本中输出 <search>/<wiki> XML 标签, 解析后执行工具, 注入结果再让模型继续.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
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
def _clean_cfg_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _model_defaults(config: dict[str, Any]) -> dict[str, str]:
    saved = config.get("_model_defaults")
    if isinstance(saved, dict):
        defaults = {
            "model": _clean_cfg_text(saved.get("model")),
            "api_key": _clean_cfg_text(saved.get("api_key")),
            "api_base": _clean_cfg_text(saved.get("api_base") or saved.get("base_url")),
            "reasoning_effort": _clean_cfg_text(saved.get("reasoning_effort")),
        }
    else:
        defaults = {
            "model": _clean_cfg_text(config.get("model") or config.get("model_name")),
            "api_key": _clean_cfg_text(config.get("api_key")),
            "api_base": _clean_cfg_text(config.get("api_base") or config.get("base_url")),
            "reasoning_effort": _clean_cfg_text(config.get("reasoning_effort")),
        }
    if not defaults["model"]:
        defaults["model"] = DEFAULT_MODEL
    return defaults


def _normalize_model_profile(
    entry: Any,
    *,
    defaults: dict[str, str],
    name_hint: str = "",
) -> dict[str, Any] | None:
    if isinstance(entry, str):
        entry = {"model": entry}
    if not isinstance(entry, dict):
        return None

    model = _clean_cfg_text(entry.get("model") or entry.get("model_name") or defaults.get("model"))
    if not model:
        model = DEFAULT_MODEL

    profile: dict[str, Any] = {"model": model}
    alias = _clean_cfg_text(entry.get("name") or entry.get("label") or name_hint)
    if alias:
        profile["name"] = alias

    api_key = _clean_cfg_text(entry.get("api_key") or defaults.get("api_key"))
    api_base = _clean_cfg_text(entry.get("api_base") or entry.get("base_url") or defaults.get("api_base"))
    reasoning_effort = _clean_cfg_text(entry.get("reasoning_effort") or defaults.get("reasoning_effort"))

    if api_key:
        profile["api_key"] = api_key
    if api_base:
        profile["api_base"] = api_base
    if reasoning_effort:
        profile["reasoning_effort"] = reasoning_effort
    return profile


def _normalize_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = _model_defaults(config)
    raw_models = config.get("models")
    profiles: list[dict[str, Any]] = []

    if isinstance(raw_models, list):
        for entry in raw_models:
            profile = _normalize_model_profile(entry, defaults=defaults)
            if profile:
                profiles.append(profile)
    elif isinstance(raw_models, dict):
        for name, entry in raw_models.items():
            profile = _normalize_model_profile(entry, defaults=defaults, name_hint=_clean_cfg_text(name))
            if profile:
                profiles.append(profile)

    if not profiles:
        fallback = _normalize_model_profile(config, defaults=defaults)
        if fallback:
            profiles.append(fallback)

    return profiles or [{"model": DEFAULT_MODEL}]


def _pick_active_model_index(config: dict[str, Any], profiles: list[dict[str, Any]] | None = None) -> int:
    items = profiles or _normalize_models(config)
    if not items:
        return 0

    raw_index = config.get("active_model_index")
    if isinstance(raw_index, str) and raw_index.strip().isdigit():
        raw_index = int(raw_index.strip())
    if isinstance(raw_index, int):
        return max(0, min(raw_index, len(items) - 1))

    active = _clean_cfg_text(config.get("active_model"))
    if active:
        lowered = active.lower()
        for idx, profile in enumerate(items):
            model = _clean_cfg_text(profile.get("model")).lower()
            name = _clean_cfg_text(profile.get("name")).lower()
            if lowered in {model, name}:
                return idx
    return 0


def build_model_config(config: dict[str, Any] | None = None, model_index: int | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    profiles = _normalize_models(raw)
    index = _pick_active_model_index(raw, profiles) if model_index is None else int(model_index or 0)
    if profiles:
        index %= len(profiles)
    else:
        index = 0

    active = profiles[index]
    cfg = dict(raw)
    cfg["models"] = profiles
    cfg["_model_defaults"] = _model_defaults(raw)
    cfg["active_model_index"] = index
    cfg["active_model"] = active.get("model") or active.get("name") or DEFAULT_MODEL
    cfg["model"] = active.get("model") or DEFAULT_MODEL

    for key in ("api_key", "api_base", "reasoning_effort"):
        if key in active:
            cfg[key] = active[key]
        else:
            cfg.pop(key, None)
    return cfg


def get_model_profiles(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = build_model_config(config)
    return [dict(profile) for profile in cfg.get("models") or []]


def load_config() -> dict[str, Any]:
    """读取 yaml，并归一化为向后兼容的多模型配置."""
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text("utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        raw = {}
    return build_model_config(raw if isinstance(raw, dict) else {})


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
    del config
    return "websearch"


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
    del config
    from .web_search import on_startup

    on_startup(headless=headless)


def shutdown_tools(config: dict[str, Any] | None = None):
    del config
    from .web_search import on_shutdown

    on_shutdown()


def _web_search(query: str, df: str = "", **_) -> dict[str, Any]:
    """调用 WebToolSuite 执行搜索."""
    from .web_search import web_search

    try:
        payload = _run_async(web_search(
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


def _tool_callback_args(args: dict[str, Any], result_text: str, *, elapsed_s: float | None = None) -> dict[str, Any]:
    merged = dict(args)
    try:
        payload = json.loads(result_text)
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        if "count" in payload:
            merged["_count"] = payload.get("count")
        if "ok" in payload:
            merged["_ok"] = payload.get("ok")
    if elapsed_s is not None:
        merged["_elapsed_s"] = max(0.0, float(elapsed_s))
    return merged


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
> 技巧1: 对于专业性知识推荐搜索的时候额外追加一条搜索 带有相关专业网站的, 例如查询工具类: github、动漫类: 萌娘百科、我的世界: mcwiki/mcmod...以此类推
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


def _build_multimodal_content(question: str, image_paths: list[str] | None = None) -> str | list[dict[str, Any]]:
    paths = [str(path).strip() for path in (image_paths or []) if str(path).strip()]
    text = _effective_prompt_text(question, len(paths))
    if not paths:
        return text

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path_str in paths:
        path = Path(path_str).expanduser()
        raw = path.read_bytes()
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


def llm_call(messages, *, config, stats=None, trace_label="Model", log_id=None):
    cfg = build_model_config(config)
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "drop_params": True}
    if cfg.get("api_base"): kw["api_base"] = cfg["api_base"]
    if cfg.get("api_key"): kw["api_key"] = cfg["api_key"]
    re_ = str(cfg.get("reasoning_effort") or "").strip().lower()
    if re_ in ("minimal", "low", "medium", "high"):
        kw["reasoning_effort"] = re_

    t0 = time.perf_counter()
    try:
        resp = litellm.completion(**kw)
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
            futures = {pool.submit(execute_tool, c["name"], c["args"]): c for c in calls}
            results_map: dict[int, str] = {}
            for fut in as_completed(futures):
                c = futures[fut]
                idx = calls.index(c)
                result_text = fut.result()
                results_map[idx] = result_text
                if callable(on_tool):
                    try:
                        on_tool(
                            c["name"],
                            _tool_callback_args(
                                c["args"],
                                result_text,
                                elapsed_s=time.perf_counter() - started_at,
                            ),
                        )
                    except Exception:
                        pass
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
    images: list[str] | None = None,
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
        if callable(on_rewind):
            try:
                on_rewind(_strip_tool_tags(full_text))
            except Exception:
                pass

        msgs.append({"role": "assistant", "content": full_text})

        if callable(on_status):
            on_status("搜索中...")
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(execute_tool, tc["name"], tc["args"]): tc for tc in calls}
            results_map: dict[int, str] = {}
            for fut in as_completed(futures):
                tc = futures[fut]
                idx = calls.index(tc)
                result_text = fut.result()
                results_map[idx] = result_text
                if callable(on_tool):
                    try:
                        on_tool(
                            tc["name"],
                            _tool_callback_args(
                                tc["args"],
                                result_text,
                                elapsed_s=time.perf_counter() - started_at,
                            ),
                        )
                    except Exception:
                        pass
        parts = [
            f'<result name="{calls[i]["name"]}" query="{calls[i]["args"]["query"]}">\n{results_map[i]}\n</result>'
            for i in range(len(calls))
        ]
        results_xml = "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"
        msgs.append({"role": "user", "content": results_xml + "\n\n" + TOOL_RESULTS_GUIDE})

    return "未获得有效回答。"
