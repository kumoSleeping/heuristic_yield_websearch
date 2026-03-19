from __future__ import annotations

from copy import deepcopy
from typing import Any

BASE_SYSTEM_PROMPT = """\
# 你的身份
- You are {name}, 是一个快速响应的搜索专家, 通过多轮工具获取信息、配置任务、节约精选上下文, 直到找到核心的证据的一瞬间，立即停止搜索并直接回答。
- Current time: {time}。这是重要信息，你必须考虑时间导致的信息滞后问题。
- 当前任务锚点: {user_message}
- Preferred language: {language}
- 你的工作记忆会以两个独立区域提供给你：
  - `Skeleton Context`：规划层，记录项目语言、关键词重写、用户需求、待核验 claim。
  - `Evidence Context`：证据层，记录搜索结果与页面摘录。
{custom}

## 工具
- `web_search(query)`
  - 混用 wiki 式搜索(关键词本身) 和 关联式搜索(关键信息拓展).
  - `query` 风格: 1-2词, 短、严格, 去除助词, 遵循 `skeleton.project_language`, 优先使用 `skeleton.keyword` 组合
  - 对于专业性知识，必要时补一条专业站点搜索，例如 GitHub、萌娘百科、mcwiki, 扩大搜索召回率.
- `page_extract(url|ref, query, lines)`
  - `page_extract` 易失败, 需一次抓取多个可能的页面, 搜索目标
  - 优先读取已经出现在 `Evidence Context` 中的结果，必要时使用 `ref`。
  - `lines`: 关键词附近的上下文窗口行；使用 15 - 30, 用户明确查看整页时才使用 `all`.
  - `query` 使用 `xxx | yyy | zzz` 同时匹配多个片段时, 必须包含至少一个页面独有的短语、精确模型名或强识别词，禁止只用泛词作为 query，例如 `available | designed | list | docs`；这类词会命中大量导航/模板垃圾。
{context_management_tool_prompt}
- 积极的使用这些工具, 直到定位到关键信息, 立即进行最终回复.

## 每次必须满足轮次规则 (重要!)
- 调用工具前先简单使用无加粗的普通文本, 1-3 句话简单汇报, 再同时进行其他操作.
- 在最终答复之前，每个工具轮次都必须一次性调用多个工具，优先 3-6 个；不要把本应同轮完成的动作拆成多轮单工具执行。
- 必须积极调用 `context_keep(ids=[...])`、 `plan_update(create=[...], update=[...])`, 他们会辅助你产生更好的结果.
- 有证据就答，不再验证, 严禁为了“更完美”“更稳妥”“再验证一下”而继续扩张检索；你能找到的资料通常已经基本找过了, 带着不确定性直接回答。

"""

CONTEXT_MANAGEMENT_TOOL_PROMPT = """\
- `context_keep(ids)`
  - [重要] 你会在下一轮遗忘本次搜索/抽取得到的新证, 必须使用本工具保留才能继续出现在后续轮次的 `Evidence Context`。
  - 由于 kept evidence 不能再删除，必须只 keep 最小必要集合, 电商页、无关页、重复页、弱相关页、误命中页不 keep。
- `plan_update(...)`
  - 用来新增或修改 `Skeleton Context` 里的 item。
  - 一旦获取新有用信息, 语言、关键词、用户需求或 claim 均需立即使用 `plan_update` 使用最新知识修正。
  - 最常见的 skeleton 建法示例：
    - project_language 须仅根据搜索目标本身的原始客观属性（如原产地或文化背景）来决定搜索语言，严格排除用户提问语言或现有网络搜索结果语言的干扰。
    - `create=[{{"type":"skeleton.project_language","text":"日语"}},{{"type":"skeleton.keyword","text":"超時空要塞マクロス"}},{{"type":"skeleton.user_need","text":"介绍"}},{{"type":"skeleton.claim","claim_id":"1","text":"wiki网页"}}]`
"""

FIRST_SEARCH_PROMPT = """
## 首轮搜索
- 对于复杂问题, 先提取用户消息中**关键词本身**进行 wiki 式单一搜索, 尝试获得其官方名称、含义、最新消息, 方便后续骨架解析.
- 若用户给了链接，优先 `page_extract`, 若无任何要求立即进入总结.

## 特殊情况可以不调用工具直接输出结果
- 违纪违法、色情见证内容
- 明确的快速任务: 简单翻译、文案编写、简单文本处理任务、闲聊、图片识别等明确不存在待搜索词的任务.
"""

POST_SEARCH_SKELETON_PROMPT = """
"""

POST_SKELETON_REFINE_PROMPT = """
"""

NORMAL_LOOP_PROMPT = """\
## 方法
- 一旦已有证据已经足够回答用户问题，就立刻结束，不要再进入任何额外验证流程。
"""

CONTEXT_KEEP_FOLLOWUP_REMINDER_TEXT = """\
如果你不打算现在直接进行最终回复，请立刻调用 `context_keep(ids=[...])`，先保留本轮有价值的 Candidate Evidence。
然后再同时调用你接下来需要的其他操作。
"""

DEFAULT_NO_TITLE_TEXT = "No Title"
DEFAULT_NO_RESULTS_TEXT = "No results."
DEFAULT_PAGE_NO_MATCHING_TEXT = "No matching lines found."
DEFAULT_PAGE_NO_MATCHING_CACHED_TEXT = "No matching lines found in cached page content."
DUPLICATE_QUERY_SKIPPED_TEXT = "Duplicate query skipped in this session. Choose a different query or use page_extract on an existing result."
PAGE_MARKDOWN_EMPTY_TITLE = "# Page"
PAGE_MARKDOWN_TITLE_PREFIX = "Title: "
PAGE_MARKDOWN_KEYWORDS_PREFIX = "Keywords: "
PAGE_MARKDOWN_WINDOW_ALL_TEXT = "Window: all lines"
PAGE_MARKDOWN_WINDOW_PREFIX = "Window: "
PAGE_MARKDOWN_WINDOW_SUFFIX = " lines"
PAGE_MARKDOWN_CACHE_PREFIX = "Cache: "
PAGE_MARKDOWN_CACHE_HIT_TEXT = "hit"
PAGE_MARKDOWN_CACHE_MISS_TEXT = "miss"
PAGE_MARKDOWN_MATCHED_LINES_TEXT = "Matched lines:"

EVIDENCE_CONTEXT_HEADING = "# Evidence Context"
EVIDENCE_CONTEXT_EMPTY_TEXT = "No kept evidence yet."
EVIDENCE_CONTEXT_KEPT_TEXT = "Kept items below stay active across rounds."
EVIDENCE_CONTEXT_ID_TEXT = "Each evidence item has a stable integer id starting at 0."
CANDIDATE_EVIDENCE_HEADING = "# Candidate Evidence"
CANDIDATE_EVIDENCE_TEXT = "Items below came from the latest tool batch only."
CANDIDATE_EVIDENCE_REPLACEMENT_TEXT = "They will be replaced by future tool results unless you keep them with context_keep."



def format_context_keep_markdown(
    *,
    kept_ids: list[int],
    already_kept_ids: list[int],
    missing_ids: list[int],
    reason: str = "",
) -> str:
    lines = ["# Context Keep", ""]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append(f"Kept: {', '.join(str(item) for item in kept_ids) if kept_ids else 'none'}")
    if already_kept_ids:
        lines.append(f"Already kept: {', '.join(str(item) for item in already_kept_ids)}")
    if missing_ids:
        lines.append(f"Missing: {', '.join(str(item) for item in missing_ids)}")
    return "\n".join(lines).strip()


HEADING_USER_NEED = "## User Need Reconstruction"
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
            "description": "Extract evidence from a page, either by direct url or by an existing evidence ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Direct page URL. Omit when using ref.",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Existing evidence id to reuse, for example '3'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Keywords for locating relevant content. Can be empty when lines='all'.",
                    },
                    "lines": {
                        "type": "string",
                        "description": "Extraction window, such as '15-30'.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "context_keep",
            "description": "Promote high-value candidate evidence into persistent Evidence Context for future rounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Candidate evidence ids to keep.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_update",
            "description": "Create or update plan skeleton items in Skeleton Context only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "create": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                                "claim_id": {"type": "string"},
                            },
                            "required": ["type"],
                            "additionalProperties": False,
                        },
                    },
                    "update": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "text": {"type": "string"},
                                "claim_id": {"type": "string"},
                            },
                            "required": ["id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
    },
)

_CONTEXT_MANAGEMENT_TOOL_NAMES = frozenset({"context_keep", "plan_update"})


def litellm_tools_for_phase(phase: str, *, disclosure_step: int | None = None) -> list[dict[str, Any]]:
    del phase
    include_context_management_tools = disclosure_step is None or int(disclosure_step) > 0
    tools: list[dict[str, Any]] = []
    for item in _TOOL_DEFINITIONS:
        name = str(item.get("function", {}).get("name") or "").strip()
        if not include_context_management_tools and name in _CONTEXT_MANAGEMENT_TOOL_NAMES:
            continue
        tools.append(deepcopy(item))
    return tools


def litellm_tool_config_for_phase(phase: str, *, disclosure_step: int | None = None) -> dict[str, Any]:
    return {
        "tools": litellm_tools_for_phase(phase, disclosure_step=disclosure_step),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
