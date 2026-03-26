from __future__ import annotations

import json
import re
from typing import Any

FILTER_KEYS: tuple[str, ...] = ("kl", "df")


def coerce_search_filters(payload: dict[str, Any] | None) -> dict[str, str]:
    data = payload if isinstance(payload, dict) else {}
    filters: dict[str, str] = {}

    region = str(data.get("kl") or data.get("region") or "").strip().lower()
    if region:
        filters["kl"] = region

    date_filter = str(data.get("df") or data.get("timelimit") or "").strip()
    if date_filter:
        filters["df"] = date_filter

    return filters


def format_search_filters(filters: dict[str, Any] | None) -> str:
    data = coerce_search_filters(filters)
    parts: list[str] = []
    for key in FILTER_KEYS:
        value = str(data.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def format_search_request_label(query: str, filters: dict[str, Any] | None = None) -> str:
    clean_query = str(query or "").strip()
    filter_text = format_search_filters(filters)
    if clean_query and filter_text:
        return f"{clean_query} [{filter_text}]"
    return clean_query


def normalize_search_request_key(query: str, filters: dict[str, Any] | None = None) -> str:
    clean_query = re.sub(r"\s+", " ", str(query or "").strip().strip("`'\"")).lower()
    if not clean_query:
        return ""
    normalized_filters = coerce_search_filters(filters)
    if not normalized_filters:
        return clean_query
    return clean_query + "\n" + json.dumps(normalized_filters, ensure_ascii=False, sort_keys=True)


def build_search_result_lines(
    public_results: list[dict[str, Any]],
    *,
    default_no_title_text: str = "No Title",
    snippet_limit: int = 180,
) -> list[str]:
    rows = [item for item in public_results if isinstance(item, dict)]
    if not rows:
        return []
    width = max(2, len(str(len(rows))))
    rendered: list[str] = []
    for index, item in enumerate(rows, start=1):
        title = str(item.get("title") or default_no_title_text).strip() or default_no_title_text
        domain = str(item.get("domain") or "").strip()
        snippet = str(item.get("snippet") or item.get("intro") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > max(1, int(snippet_limit)):
            snippet = snippet[: max(1, int(snippet_limit)) - 1].rstrip() + "…"
        parts = [f"[{index}] {title}"]
        if domain:
            parts.append(domain)
        matched_queries = [
            str(item_query or "").strip()
            for item_query in item.get("matched_queries", [])
            if str(item_query or "").strip()
        ]
        if matched_queries:
            parts.append("matched=" + " | ".join(matched_queries[:3]))
        if snippet:
            parts.append(snippet)
        rendered.append(f"L{index:0{width}d} | " + " | ".join(parts))
    return rendered


def build_search_open_lines(
    public_results: list[dict[str, Any]],
    *,
    default_no_title_text: str = "No Title",
    snippet_limit: int = 180,
) -> list[str]:
    rows = [item for item in public_results if isinstance(item, dict)]
    rendered: list[str] = []
    for index, item in enumerate(rows, start=1):
        title = str(item.get("title") or default_no_title_text).strip() or default_no_title_text
        url = str(item.get("url") or "").strip()
        domain = str(item.get("domain") or "").strip()
        snippet = str(item.get("snippet") or item.get("intro") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > max(1, int(snippet_limit)):
            snippet = snippet[: max(1, int(snippet_limit)) - 1].rstrip() + "…"
        linked_title = f"[{title}]({url})" if url else title
        parts = [f"[{index}] {linked_title}"]
        if domain:
            parts.append(domain)
        matched_queries = [
            str(item_query or "").strip()
            for item_query in item.get("matched_queries", [])
            if str(item_query or "").strip()
        ]
        if matched_queries:
            parts.append("matched=" + " | ".join(matched_queries[:3]))
        if snippet:
            parts.append(snippet)
        rendered.append(" | ".join(parts))
    return rendered


def build_search_document_markdown(
    *,
    query: str,
    public_results: list[dict[str, Any]],
    skipped_duplicate: bool,
    reminder: str,
    error: str = "",
    search_filters: dict[str, Any] | None = None,
    direction_brief: str = "",
    expanded_queries: list[str] | None = None,
    duplicate_queries: list[str] | None = None,
    summary_text: str = "",
    default_no_results_text: str = "No results.",
    duplicate_query_skipped_text: str = "Duplicate query skipped in this session.",
    default_no_title_text: str = "No Title",
) -> str:
    lines = [f"# Search: {query}", ""]
    filter_text = format_search_filters(search_filters)
    if filter_text:
        lines.append(f"Filters: {filter_text}")
    clean_direction = str(direction_brief or "").strip()
    if clean_direction:
        lines.append(f"Direction: {clean_direction}")
    clean_queries = [str(item or "").strip() for item in expanded_queries or [] if str(item or "").strip()]
    if clean_queries:
        lines.append("Executed queries: " + ", ".join(clean_queries))
    clean_duplicates = [str(item or "").strip() for item in duplicate_queries or [] if str(item or "").strip()]
    if clean_duplicates:
        lines.append("Skipped duplicate queries: " + ", ".join(clean_duplicates))
    clean_summary = str(summary_text or "").strip()
    if clean_summary:
        lines.append(f"Search summary: {clean_summary}")

    if len(lines) > 2:
        lines.append("")

    if skipped_duplicate and not public_results and not error:
        lines.append(duplicate_query_skipped_text)
    elif error:
        lines.append(error)
    elif not public_results:
        lines.append(default_no_results_text)
    else:
        result_lines = build_search_result_lines(
            public_results,
            default_no_title_text=default_no_title_text,
        )
        lines.append(f"Preview: results={len(result_lines)}")
        lines.extend(["", "Result lines:"])
        lines.extend(result_lines)

    if reminder:
        if lines and str(lines[-1]).strip():
            lines.append("")
        lines.append(str(reminder).strip())
    return "\n".join(lines).strip()


__all__ = [
    "build_search_document_markdown",
    "build_search_open_lines",
    "build_search_result_lines",
    "coerce_search_filters",
    "format_search_filters",
    "format_search_request_label",
    "normalize_search_request_key",
]
