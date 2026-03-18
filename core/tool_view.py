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


def format_tool_view_argument(name: str, arguments: Any, *, max_items: int = 12, max_chars: int = 160) -> str:
    payload = sanitize_tool_view_args(arguments)
    if not isinstance(payload, dict):
        return ""

    if name in ("web_search", "web_search_wiki"):
        return re.sub(r"\s+", " ", str(payload.get("query") or "").strip())

    if name == "page_extract":
        host = _short_host(str(payload.get("url") or "").strip())
        ref = str(payload.get("ref") or "").strip()
        query = re.sub(r"\s+", " ", str(payload.get("query") or "").strip())
        lines = str(payload.get("lines") or "").strip()
        line_label = "all" if lines.lower() == "all" else (f"{lines}line" if lines else "")
        parts = [part for part in (host or ref, query, line_label) if part]
        return ", ".join(parts)

    if name in ("context_keep", "context_delete"):
        ids = payload.get("ids")
        last_block = bool(payload.get("last_block"))
        parts: list[str] = []
        if isinstance(ids, list) and ids:
            rendered_ids = ", ".join(str(item) for item in ids[:max_items])
            if len(ids) > max_items:
                rendered_ids += ", …"
            parts.append(rendered_ids)
        if last_block:
            parts.append("last_block")
        return ", ".join(parts)

    if name == "context_update":
        create = payload.get("create")
        update = payload.get("update")
        parts: list[str] = []
        if isinstance(create, list) and create:
            parts.append(f"create {len(create)}")
        if isinstance(update, list) and update:
            parts.append(f"update {len(update)}")
        return ", ".join(parts)

    if name == "sub_agent_task":
        tools = str(payload.get("tools") or "").strip()
        host = _short_host(str(payload.get("url") or "").strip())
        task = re.sub(r"\s+", " ", str(payload.get("task") or "").strip())
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

