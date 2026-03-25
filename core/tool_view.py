from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse


def sanitize_tool_view_args(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key or key.startswith("_"):
                continue
            cleaned[key] = sanitize_tool_view_args(raw_value)
        return cleaned
    if isinstance(value, list):
        return [sanitize_tool_view_args(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_tool_view_args(item) for item in value]
    return value


def _short_host(url: str) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    host = str(parsed.netloc or parsed.path or "").strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _short_page_target(url: str, *, max_segments: int = 4, max_chars: int = 72) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    host = _short_host(raw_url)
    path = str(parsed.path or "").strip("/")
    if not host:
        return _truncate_line(path, max_chars) if path else ""
    if not path:
        return host
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) > max(1, int(max_segments)):
        path = "/".join(segments[: max(1, int(max_segments))]) + "/…"
    else:
        path = "/".join(segments)
    return _truncate_line(f"{host}/{path}", max_chars)


def _clean_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clean_inline_list(value: Any, *, limit: int = 2) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_inline_text(item)
        normalized = text.casefold()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        items.append(text)
        if len(items) >= max(1, int(limit)):
            break
    return items


def _pattern_summary(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(_clean_inline_list(value, limit=8))
    text = _clean_inline_text(value)
    if not text:
        return ""
    parts = [part.strip() for part in text.split("|")]
    return " | ".join(part for part in parts if part)


def _truncate_line(text: str, max_chars: int) -> str:
    line = str(text or "").strip()
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 1].rstrip() + "…"


def _window_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() == "all":
        return "All"
    match = re.search(r"\d+", text)
    if not match:
        return ""
    return f"L{int(match.group(0))}"


def _numeric_value(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _page_extract_mode_label(payload: dict[str, Any]) -> str:
    mode = str(payload.get("mode") or "").strip().lower()
    if mode == "range":
        start_line = _numeric_value(payload.get("start_line"))
        end_line = _numeric_value(payload.get("end_line"))
        if start_line is not None and end_line is not None and start_line > 0 and end_line >= start_line:
            return f"L{start_line}-L{end_line}"
        return "range"
    if mode == "sample":
        return "sample"
    return ""


def _page_extract_sample_stats(payload: dict[str, Any]) -> str:
    if str(payload.get("mode") or "").strip().lower() != "sample":
        return ""
    shown = _numeric_value(payload.get("count"))
    if shown is None:
        shown = _list_count(payload.get("_matched_lines"))
    total_lines = _numeric_value(payload.get("total_lines"))
    if shown is not None and shown > 0 and total_lines is not None and total_lines > 0:
        return f"{shown}/{total_lines} lines"
    if shown is not None and shown > 0:
        return f"{shown} lines"
    if total_lines is not None and total_lines > 0:
        return f"{total_lines} lines"
    return ""


def _page_extract_title(payload: dict[str, Any], *, max_chars: int = 72) -> str:
    title = _clean_inline_text(payload.get("title"))
    if not title:
        return ""
    return _truncate_line(title, max_chars)


def _plan_counts(payload: dict[str, Any]) -> tuple[int, int]:
    create_count = _numeric_value(payload.get("_created_count"))
    if create_count is None:
        create_count = _list_count(payload.get("create"))
    update_count = _numeric_value(payload.get("_updated_count"))
    if update_count is None:
        update_count = _list_count(payload.get("update"))
    return max(0, int(create_count or 0)), max(0, int(update_count or 0))


def format_tool_view_argument(name: str, arguments: Any, *, max_items: int = 12, max_chars: int = 160) -> str:
    payload = sanitize_tool_view_args(arguments)
    if not isinstance(payload, dict):
        return ""

    if name in ("web_search", "web_search_wiki"):
        query = _clean_inline_text(payload.get("query"))
        return f"\"{query}\"" if query else ""

    if name == "page_extract":
        host = _short_host(str(payload.get("url") or "").strip())
        ref = str(payload.get("ref") or "").strip()
        search = _pattern_summary(payload.get("search"))
        target = host or ref
        if target and search:
            return f"\"{search}\" in \"{target}\""
        return f"\"{target}\"" if target else (f"\"{search}\"" if search else "")

    if name == "context_delete":
        ids = payload.get("ids")
        if isinstance(ids, list) and ids:
            return f"Delete {len(ids)}"
        return ""

    if not payload:
        return ""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def format_tool_view_text(name: str, arguments: Any, *, max_chars: int = 160) -> str:
    payload = arguments if isinstance(arguments, dict) else {}
    tool_name = str(name or "").strip()

    if tool_name in ("web_search", "web_search_wiki"):
        query = _clean_inline_text(payload.get("query"))
        line = f"Web Search \"{query}\"" if query else "Web Search"
        count = _numeric_value(payload.get("_count"), payload.get("count"))
        if count is not None:
            line += f" · {count}"
        return _truncate_line(line, max_chars)

    if tool_name == "page_extract":
        target = _short_page_target(str(payload.get("url") or "").strip())
        ref = str(payload.get("ref") or "").strip()
        if not target and ref:
            target = f"#{ref}"
        mode_label = _page_extract_mode_label(payload)
        sample_stats = _page_extract_sample_stats(payload)
        title = _page_extract_title(payload)
        line = "Read"
        if mode_label:
            line += f" {mode_label}"
        if target:
            if sample_stats:
                line += f" {sample_stats}"
            if title:
                line += f" \"{title}\""
            line += f" in \"{target}\""
        else:
            if sample_stats:
                line += f" {sample_stats}"
            if title:
                line += f" \"{title}\""
        return _truncate_line(line, max_chars)

    if tool_name == "context_delete":
        count = _numeric_value(payload.get("_count"), payload.get("count"))
        if count is None:
            ids = payload.get("ids")
            if isinstance(ids, list) and ids:
                count = len(ids)
        if count is not None:
            return f"Context: Delete {count}"
        return "Context: Delete"

    fallback = format_tool_view_argument(tool_name, payload, max_chars=max_chars)
    if fallback:
        return _truncate_line(f"{tool_name} {fallback}", max_chars)
    return _truncate_line(tool_name, max_chars)
