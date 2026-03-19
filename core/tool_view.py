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


def _clean_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


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
        query = _clean_inline_text(payload.get("query"))
        base = ""
        if query:
            base = f"\"{query}\""
        target = host or ref
        if target:
            base = f"{base} in \"{target}\"".strip()
        extras: list[str] = []
        line_label = _window_label(payload.get("lines"))
        if line_label:
            extras.append(line_label)
        if extras:
            return f"{base} · {' · '.join(extras)}".strip()
        return base

    if name == "context_keep":
        ids = payload.get("ids")
        if isinstance(ids, list) and ids:
            return f"Keep {len(ids)}"
        return ""

    if name == "context_delete":
        ids = payload.get("ids")
        if isinstance(ids, list) and ids:
            return f"Delete {len(ids)}"
        return ""

    if name == "plan_update":
        create_count, update_count = _plan_counts(payload)
        parts: list[str] = []
        if create_count:
            parts.append(f"Create {create_count}")
        if update_count:
            parts.append(f"Update {update_count}")
        return ", ".join(parts)

    if name == "sub_agent_task":
        tools = str(payload.get("tools") or "").strip()
        host = _short_host(str(payload.get("url") or "").strip())
        task = _clean_inline_text(payload.get("task"))
        if task:
            task = task[:120]
        binding = ""
        if tools and host:
            binding = f"{tools}({host})"
        elif tools:
            binding = tools
        elif host:
            binding = host
        if binding and task:
            return f"{binding}, {task}"
        return binding or task

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
        query = _clean_inline_text(payload.get("query"))
        host = _short_host(str(payload.get("url") or "").strip())
        ref = str(payload.get("ref") or "").strip()
        target = host or (f"#{ref}" if ref else "")
        line = f"Read \"{query}\"" if query else "Read"
        if target:
            line += f" in \"{target}\""
        extras: list[str] = []
        window = _window_label(payload.get("lines") or payload.get("window"))
        if window:
            extras.append(window)
        count = _numeric_value(payload.get("_count"), payload.get("count"))
        if count is not None:
            extras.append(str(count))
        if extras:
            line += " · " + " · ".join(extras)
        return _truncate_line(line, max_chars)

    if tool_name == "context_keep":
        count = _numeric_value(payload.get("_count"), payload.get("count"))
        if count is None:
            ids = payload.get("ids")
            if isinstance(ids, list) and ids:
                count = len(ids)
        if count is not None:
            return f"Context: Keep {count}"
        return "Context: Keep"

    if tool_name == "context_delete":
        count = _numeric_value(payload.get("_count"), payload.get("count"))
        if count is None:
            ids = payload.get("ids")
            if isinstance(ids, list) and ids:
                count = len(ids)
        if count is not None:
            return f"Context: Delete {count}"
        return "Context: Delete"

    if tool_name == "plan_update":
        create_count, update_count = _plan_counts(payload)
        actions: list[str] = []
        if create_count:
            actions.append(f"Create {create_count}")
        if update_count:
            actions.append(f"Update {update_count}")
        if not actions:
            actions.append("Update")
        return _truncate_line("Plan: " + " · ".join(actions), max_chars)

    if tool_name == "sub_agent_task":
        tools = str(payload.get("tools") or "").strip().lower()
        task = _clean_inline_text(payload.get("task"))
        host = _short_host(str(payload.get("url") or "").strip())
        if tools == "websearch" and task:
            return _truncate_line(f"Web Search \"{task}\"", max_chars)
        if tools == "page":
            line = f"Read \"{task}\"" if task else "Read"
            if host:
                line += f" in \"{host}\""
            return _truncate_line(line, max_chars)

    fallback = format_tool_view_argument(tool_name, payload, max_chars=max_chars)
    if fallback:
        return _truncate_line(f"{tool_name} {fallback}", max_chars)
    return _truncate_line(tool_name, max_chars)
