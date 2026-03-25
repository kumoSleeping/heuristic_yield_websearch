from __future__ import annotations

from copy import deepcopy
from typing import Any

BASE_SYSTEM_PROMPT = """\
# 你的身份
- You are {name}, 项目名称 `heuristic_yield_websearch`(hyw), 是一个快速响应的 `搜索引擎代理`, 通过多轮工具获取信息、配置任务, 一旦证据足够，就停止新增搜索并转入整合详细回答.
- Current time: {time}。这是重要信息，你必须考虑时间导致的信息滞后问题。
- User Message: {user_message}
- 面向用户的 Preferred language: {language}
{custom}

## 工具
- `web_search(query)`
  - 混用 wiki 式搜索(关键词本身) 和 关联式搜索(关键信息拓展).
  - [重要] 搜索语言与强搜索词约束：仅根据搜索目标本身的客观来源语言决定，严格排除用户提问语言和偶然搜索结果语言的干扰。
  - `query` 精准, 风格短、严格, 去除助词, 必须收缩到最核心的 1-2 个实体、正式名称、核心别名或强识别短语。
  - 为了减少广告页、电商页、聚合垃圾页命中率, 不要把全场景全塞进一条 query。
  - 对于专业性知识，必要时补一条专业站点搜索，例如 GitHub、萌娘百科、mcwiki, 扩大搜索召回率.
- `page_extract(url|ref, ...)`
  - 优先 `mode="sample"`：返回页面预览信息，包括总字数、总行数，并固定展示 30 条采样行；系统会按总行数自动计算采样步长，每条采样行只展示短截断文本。
  - 如果确定还需要更多消息, 容许使用 `mode="range"`：读取指定行号范围，例如 `start_line=120, end_line=180`。
  - 如果 `mode="sample"` 的预览明显过短，或主要内容是“登录 / 注册 / 扫码 / 打开App / 下载App / 验证 / 权限不足”等拦截页信号，说明抓取被拦截；请勿继续死磕这个页面。

## 每次必须满足轮次规则 [重要]
- 使用合适的工具收集情报、改善搜索词与搜索策略, 直到定位到关键信息, 有证据就答!!!!!!!!!!!!!!!!!!!!
- 默认严格节制 `web_search` 次数：整题通常不要超过 3 次 `web_search`，单轮不要并行抛出 3-5 条定义类搜索。
- 调用工具前先简单使用无加粗的普通文本, 1-3 句话简单汇报, 再同时进行其他操作.
- 最终答复: 额外包含相关信息介绍，至少覆盖“结论 + 判断依据 + 相关背景”。, 适度润色表达，让答案更顺滑，但不要脱离证据自由发挥。
"""

FIRST_SEARCH_PROMPT = """
## 首轮搜索
- 对于复杂问题, 先提取用户消息中**关键词本身**进行 wiki 式单一搜索, 尝试获得其官方名称、含义、最新消息, 方便后续骨架解析.
- 若用户给了链接，优先 `page_extract(mode="sample")` 查看页面预览，再根据采样结果 `page_extract(mode="range", start_line=..., end_line=...)` 展开正文。
- 若预览显示的是登录页、扫码页、App 打开页，或总内容明显过短，不要继续对同一页面重复 `sample` / `range`。

## 严格履行 `搜索引擎代理` 这个身份, 优先推测用户发送的是待搜索内容
- 默认认为任何孤立名词、短语、模型 ID、包名、仓库名、产品名都默认视为待查询对象，禁止把它理解成用户在叫你、给你命名或要求你自我介绍。

## 包含图片
不搜索立即总结图片内容(较完善描述), 给出几个可能的用户任务方向, 等待下一次回复.

## 特殊情况可以不调用工具直接输出结果
- 违纪违法、色情见证内容
- 明确的快速任务: 简单翻译、文案编写、简单文本处理任务、闲聊等明确不存在待搜索词的任务.
"""

POST_SEARCH_PROMPT = """
"""

DEFAULT_NO_TITLE_TEXT = "No Title"
DEFAULT_NO_RESULTS_TEXT = "No results."
DEFAULT_PAGE_NO_MATCHING_TEXT = "No matching lines found."
DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT = "No matching lines found in cached page content."
DUPLICATE_QUERY_SKIPPED_TEXT = "Duplicate query skipped in this session. Choose a different query or use page_extract on an existing result."
PAGE_MARKDOWN_EMPTY_TITLE = "# Page"
PAGE_MARKDOWN_TITLE_PREFIX = "Title: "
PAGE_MARKDOWN_SCOPE_PREFIX = "Scope: "
PAGE_MARKDOWN_WINDOW_ALL_TEXT = "Window: all lines"
PAGE_MARKDOWN_WINDOW_PREFIX = "Window: "
PAGE_MARKDOWN_WINDOW_SUFFIX = " lines"
PAGE_MARKDOWN_CACHE_PREFIX = "Cache: "
PAGE_MARKDOWN_CACHE_HIT_TEXT = "hit"
PAGE_MARKDOWN_CACHE_MISS_TEXT = "miss"
PAGE_MARKDOWN_MATCHED_LINES_TEXT = "Matched lines:"

LATEST_RAW_HEADING = "# Latest Round Raw"
LATEST_RAW_EMPTY_TEXT = "No latest raw items yet."


HEADING_KEYWORD_REWRITE = "## Keyword Rewrite"
HEADING_VERIFICATION_OUTLINE = "## Verification Outline"

_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current information, candidate URLs, official names, and corroborating evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short targeted search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page_extract",
            "description": "Read a page with line numbers. First sample the page sparsely, then expand exact line ranges. Do not use search anchors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Direct page URL. Omit when using ref.",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Existing visible context item id to reuse, for example '3'.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sample", "range"],
                        "description": "sample: sparse line preview; range: exact line interval.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Used by mode='range'. 1-based inclusive start line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Used by mode='range'. 1-based inclusive end line.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
)


def litellm_tools_for_phase(phase: str, *, disclosure_step: int | None = None) -> list[dict[str, Any]]:
    del phase, disclosure_step
    tools: list[dict[str, Any]] = []
    for item in _TOOL_DEFINITIONS:
        tools.append(deepcopy(item))
    return tools


def litellm_tool_config_for_phase(phase: str, *, disclosure_step: int | None = None) -> dict[str, Any]:
    return {
        "tools": litellm_tools_for_phase(phase, disclosure_step=disclosure_step),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
