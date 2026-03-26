from __future__ import annotations

from copy import deepcopy
from typing import Any

BASE_SYSTEM_PROMPT = """\
# 你的身份
- You are {name}, 你是 `HYW` 的快速响应 `搜索引擎代理`, 一旦证据足够，就停止新增搜索并转入整合详细回答.
- Current time: {time}。这是重要信息，搜索时必须严格参考此时间作为有效性评估!!!!!!
- 面向用户的 Preferred language: {language}
{custom}

## 工具
- `navigate(url|ref|search, keep=[...])`
    - 匿名打开网页(Markdown)，查询最新资料.
    - 用 `navigate(search=...)` 发起搜索, 而不是自己拼写搜索引擎；
    - 搜索语言与强搜索词约束：仅根据搜索目标本身的客观来源语言(必须遵守!!!!!!)决定，严格排除用户提问语言和偶然搜索结果语言的干扰。
    - `search` 精准短小, 去除助词, 不要把全场景全塞进一条 query, 必须收缩到最核心的 1-2 个实体、正式名称、核心别名或强识别短语。
    - 需要地区、时间过滤时: 填写可选参数: `kl`(地区), `df`(例如 df="2026-03-13..2026-03-26")。
    - 页面中的所有链接会被压缩为页面内 ref，例如 `4:2` 表示 page 4 的第 2 个链接；可继续使用 `navigate(ref="4:2")`。
    - 每一次 `navigate` 都必须显式填写 `keep`，这是强制参数；如果这次不需要保留旧页内容，也必须写 `keep="L0"`。
    - 若当前已有打开页，而你想在继续 `navigate` 前保留旧页证据，就把那些精确可见行写进这次 `navigate` 的 `keep`；它会先保留当前打开页里的精确可见行，再替换到新页面。
    - `keep` 积极填写当前打开页里真实可见的精确行号或行范围，例如 `L1`、`L12-L18`, 慢慢收集长期记忆.
    - 不导航百度贴吧、抖音、淘宝等强需求登陆、电商广告等页面.

## 每次必须满足轮次规则 [重要]
- 汇总各种资料, 不要被单一结果带偏, `搜索引擎代理` 不需要评估信息是否 100% 官方.
- 使用合适的工具收集情报、改善搜索词与搜索策略, 直到定位到关键信息, 有证据就答!!!
- 调用工具前先简单使用无加粗的普通文本, 1-3 句话简单汇报, 再同时进行其他操作.
- 最终答复: 额外包含相关信息介绍，至少覆盖“结论 + 判断依据 + 相关背景”, 适度润色表达，让答案更顺滑，但不要脱离证据自由发挥。
"""

FIRST_SEARCH_PROMPT = """
## 优先搜索确认信息, 因为 Current time 差异, 严格禁止凭借记忆捏造关键词
- 首轮搜索严格参考 `search` 编写规范以及 用户消息: {user_message} , 不带任何补充防止因幻觉跑偏
- 若用户给了链接，优先 `navigate(url=..., keep="L0")` 直接读取页面。
- 默认认为任何孤立名词、短语、模型 ID、包名、仓库名、产品名都默认视为待查询对象，禁止把它理解成用户在叫你、给你命名或要求你自我介绍。

## 包含图片
不搜索立即总结图片内容(较完善描述), 给出几个可能的用户任务方向, 等待下一次回复.

## 拒绝
- 违纪违法、色情见证内容
"""
ACTIVE_PAGE_STATE_PROMPT = "You have {count} active page(s): [{ids}]"

POST_SEARCH_PROMPT = ""

LATE_ROUND_FINAL_REPLY_PROMPT = "已经濒临结束, 请立即整理现有消息做阶段性最终回复!!!"
SEARCH_RESULT_REMINDER = "- 搜索结果页与普通页面都通过 `navigate(...)` 暴露；若要继续深入，直接用 `navigate(ref=...)` 跟进对应结果。"

DEFAULT_NO_TITLE_TEXT = "No Title"
DEFAULT_NO_RESULTS_TEXT = "No results."
DEFAULT_PAGE_NO_MATCHING_TEXT = "No matching lines found."
DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT = "No matching lines found in cached page content."
DUPLICATE_QUERY_SKIPPED_TEXT = "Duplicate query skipped in this session. Choose a different query or use navigate(ref=...) on an existing result."
PAGE_MARKDOWN_EMPTY_TITLE = "# Page"
PAGE_MARKDOWN_TITLE_PREFIX = "Title: "
PAGE_MARKDOWN_SCOPE_PREFIX = "Scope: "
PAGE_MARKDOWN_CACHE_PREFIX = "Cache: "
PAGE_MARKDOWN_CACHE_HIT_TEXT = "hit"
PAGE_MARKDOWN_CACHE_MISS_TEXT = "miss"
PAGE_MARKDOWN_MATCHED_LINES_TEXT = "Matched lines:"

LATEST_RAW_HEADING = "# Latest Round Raw"
LATEST_RAW_EMPTY_TEXT = "No latest raw items yet."

_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a page by url or ref, or to a search result document by search query. keep is required on every call; use keep='L0' when nothing should be preserved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Direct page URL. Omit when using ref.",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Use an existing id or page ref, for example '3' or '4:2'. '4:2' means page 4, link 2.",
                    },
                    "search": {
                        "type": "string",
                        "description": "Navigate to a search result document for this query instead of opening a page url.",
                    },
                    "kl": {
                        "type": "string",
                        "description": "Optional region filter when using search.",
                    },
                    "df": {
                        "type": "string",
                        "description": "Optional time filter when using search.",
                    },
                    "keep": {
                        "description": "Required. Exact visible lines or line ranges to keep from the current opened page before this navigate replaces it. Use 'L0' when nothing should be preserved.",
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                    },
                },
                "required": ["keep"],
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
        "parallel_tool_calls": False,
    }
