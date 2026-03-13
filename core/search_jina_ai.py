from __future__ import annotations

from collections import deque
import os
import re
import threading
import time
from typing import Any, Dict, List

from .config import cfg_get

_JINA_CONFIG_RESERVED_KEYS = {"headers", "search", "page_extract", "prefer_free"}
_JINA_CAPABILITY_RESERVED_KEYS = {"headers", "prefer_free"}
_JINA_PAGE_EXTRACT_FREE_RPM = 20
_JINA_PAGE_EXTRACT_WINDOW_S = 60.0
_PAGE_EXTRACT_CALL_TIMES: deque[float] = deque()
_PAGE_EXTRACT_CALL_LOCK = threading.Lock()


def _load_httpx_module() -> Any:
    try:
        import httpx

        return httpx
    except Exception as exc:
        raise RuntimeError("Jina AI support requires `httpx`; reinstall `hyw`.") from exc


def _stringify_header_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _normalize_header_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    headers: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key or raw_value is None:
            continue
        text = _stringify_header_value(raw_value)
        if text:
            headers[key] = text
    return headers


def _collect_header_block(value: Any, *, reserved_keys: set[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    headers = _normalize_header_map(
        {key: val for key, val in value.items() if str(key) not in reserved_keys}
    )
    nested_headers = value.get("headers")
    if isinstance(nested_headers, dict):
        headers.update(_normalize_header_map(nested_headers))
    return headers


def _to_int(value: Any) -> int:
    try:
        return max(0, int(str(value).strip()))
    except Exception:
        return 0


def _config_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _jina_config_roots(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    roots = [
        cfg_get(config or {}, "jina_ai", {}),
        cfg_get(config or {}, "tools.config.jina_ai", {}),
    ]
    return [root for root in roots if isinstance(root, dict)]


def _jina_prefer_free(config: dict[str, Any] | None, capability: str) -> bool:
    if capability != "page_extract":
        return False
    for root in _jina_config_roots(config):
        capability_block = root.get(capability)
        if isinstance(capability_block, dict) and "prefer_free" in capability_block:
            return _config_bool(capability_block.get("prefer_free"))
        if "prefer_free" in root:
            return _config_bool(root.get("prefer_free"))
    return True


def _get_header_key(headers: dict[str, str], name: str) -> str | None:
    target = str(name or "").strip().lower()
    for key in headers:
        if str(key).strip().lower() == target:
            return key
    return None


def _has_auth_header(headers: dict[str, str]) -> bool:
    return _get_header_key(headers, "Authorization") is not None


def _use_page_extract_api_key(config: dict[str, Any] | None = None) -> bool:
    del config
    now = time.monotonic()
    with _PAGE_EXTRACT_CALL_LOCK:
        while _PAGE_EXTRACT_CALL_TIMES and (now - _PAGE_EXTRACT_CALL_TIMES[0]) >= _JINA_PAGE_EXTRACT_WINDOW_S:
            _PAGE_EXTRACT_CALL_TIMES.popleft()
        use_api_key = len(_PAGE_EXTRACT_CALL_TIMES) >= _JINA_PAGE_EXTRACT_FREE_RPM
        _PAGE_EXTRACT_CALL_TIMES.append(now)
        return use_api_key


def _jina_headers(
    *,
    config: dict[str, Any] | None = None,
    capability: str,
    default_accept: str,
) -> dict[str, str]:
    headers = {
        "User-Agent": "hyw/0.1 (+https://github.com/kumoSleeping/heuristic_yield_websearch)",
    }
    api_key = str(os.environ.get("JINA_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for root in _jina_config_roots(config):
        headers.update(_collect_header_block(root, reserved_keys=_JINA_CONFIG_RESERVED_KEYS))
        capability_block = root.get(capability) if isinstance(root, dict) else None
        headers.update(_collect_header_block(capability_block, reserved_keys=_JINA_CAPABILITY_RESERVED_KEYS))

    headers.setdefault("Accept", default_accept)
    if capability == "search":
        headers.setdefault("X-Respond-With", "no-content")
    if capability == "page_extract":
        headers.setdefault("X-Engine", "direct")
        if _jina_prefer_free(config, capability) and _has_auth_header(headers) and not _use_page_extract_api_key(config):
            auth_key = _get_header_key(headers, "Authorization")
            if auth_key:
                headers.pop(auth_key, None)
    return headers


async def _count_tokens_via_segment(client: Any, content: str) -> int:
    text = str(content or "")
    if not text:
        return 0
    try:
        response = await client.post(
            "https://segment.jina.ai/",
            json={"content": text},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return 0

    if not isinstance(payload, dict):
        return 0
    return _to_int(payload.get("num_tokens"))


async def _build_usage_meta(
    client: Any,
    *,
    response: Any,
    raw_text: str,
    capability: str,
) -> dict[str, Any]:
    tokens = _to_int(response.headers.get("x-usage-tokens"))
    source = "header" if tokens > 0 else ""
    if tokens <= 0 and raw_text:
        tokens = await _count_tokens_via_segment(client, raw_text)
        if tokens > 0:
            source = "segment"

    return {
        "provider": "jina_ai",
        "capability": capability,
        "usage": {
            "requests": 1,
            "tokens": tokens,
            "source": source or "unknown",
        },
    }


def _normalize_search_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    url = str(
        row.get("href")
        or row.get("url")
        or row.get("link")
        or row.get("source")
        or row.get("document_url")
        or ""
    ).strip()
    if url.startswith("//"):
        url = "https:" + url
    if not url.startswith(("http://", "https://")):
        return None

    title = str(
        row.get("title")
        or row.get("name")
        or row.get("headline")
        or row.get("text")
        or ""
    ).strip() or "No Title"
    snippet = str(
        row.get("body")
        or row.get("snippet")
        or row.get("description")
        or row.get("content")
        or row.get("text")
        or ""
    ).strip()
    if snippet.startswith(title):
        snippet = snippet[len(title) :].strip(" -:\n")

    return {
        "title": title,
        "url": url,
        "snippet": snippet[:500],
        "provider": "jina_ai",
    }


def _collect_jina_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in (
        "data",
        "results",
        "items",
        "topk",
        "organic",
        "organic_results",
        "searchResults",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return list(value)
    return []


def _parse_markdown_links(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    link_re = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)]+)\)")
    for match in link_re.finditer(str(text or "")):
        title = str(match.group("title") or "").strip()
        url = str(match.group("url") or "").strip()
        if not title or not url or url in seen:
            continue
        seen.add(url)
        rows.append({"title": title, "url": url, "snippet": "", "provider": "jina_ai"})
    return rows


def _looks_like_json_text(text: str) -> bool:
    stripped = str(text or "").strip()
    return stripped.startswith("{") or stripped.startswith("[")


async def jina_ai_search(
    query: str,
    kl: str | None = None,
    max_results: int = 5,
    config: dict[str, Any] | None = None,
    **_: Any,
) -> Dict[str, Any]:
    del kl
    httpx = _load_httpx_module()
    url = "https://s.jina.ai/"
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(
            url,
            params={"q": query},
            headers=_jina_headers(
                config=config,
                capability="search",
                default_accept="application/json",
            ),
        )
        response.raise_for_status()
        raw_text = response.text
        meta = await _build_usage_meta(
            client,
            response=response,
            raw_text=raw_text,
            capability="search",
        )

    rows: list[dict[str, Any]] = []
    try:
        payload = response.json()
        raw_rows = _collect_jina_rows(payload)
        for row in raw_rows:
            normalized = _normalize_search_row(row)
            if normalized:
                rows.append(normalized)
    except Exception:
        rows.extend(_parse_markdown_links(response.text))

    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        url_text = str(row.get("url") or "").strip()
        if not url_text or url_text in seen:
            continue
        seen.add(url_text)
        unique_rows.append(row)
        if len(unique_rows) >= max(1, int(max_results)):
            break
    return {
        "results": unique_rows,
        "_meta": meta,
        "_model_markdown": "" if _looks_like_json_text(raw_text) else raw_text.strip(),
    }


def _normalize_reader_source_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        raise ValueError("url is empty")
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    return text


def _parse_jina_reader_text(text: str, fallback_url: str) -> dict[str, Any]:
    title = ""
    source_url = fallback_url
    body_lines: list[str] = []
    in_body = False

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if not in_body and line.startswith("Title:"):
            title = line.split(":", 1)[1].strip()
            continue
        if not in_body and line.startswith("URL Source:"):
            source_url = line.split(":", 1)[1].strip() or fallback_url
            continue
        if line.strip() == "Markdown Content:":
            in_body = True
            continue
        if in_body:
            body_lines.append(line)

    content = "\n".join(body_lines).strip() or str(text or "").strip()
    return {
        "ok": bool(content),
        "provider": "jina_ai",
        "title": title,
        "url": source_url,
        "content": content,
        "html": "",
    }


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_jina_reader_json(payload: Any, fallback_url: str) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            parsed = _parse_jina_reader_json(item, fallback_url)
            if parsed.get("content"):
                return parsed
        return {
            "ok": False,
            "provider": "jina_ai",
            "title": "",
            "url": fallback_url,
            "content": "",
            "html": "",
        }

    if not isinstance(payload, dict):
        return _parse_jina_reader_text(str(payload or ""), fallback_url)

    title = _coerce_text(payload.get("title") or payload.get("name") or payload.get("headline"))
    source_url = _coerce_text(
        payload.get("url")
        or payload.get("source_url")
        or payload.get("source")
        or payload.get("href")
    ) or fallback_url
    content = _coerce_text(
        payload.get("content")
        or payload.get("text")
        or payload.get("markdown")
        or payload.get("body")
        or payload.get("md")
    )
    html = _coerce_text(payload.get("html"))

    if not content and isinstance(payload.get("data"), str):
        content = _coerce_text(payload.get("data"))

    if content or html:
        return {
            "ok": bool(content or html),
            "provider": "jina_ai",
            "title": title,
            "url": source_url,
            "content": content or html,
            "html": html,
        }

    for key in ("data", "result", "results", "item"):
        nested = payload.get(key)
        if isinstance(nested, (dict, list)):
            parsed = _parse_jina_reader_json(nested, source_url)
            if parsed.get("content"):
                if not parsed.get("title") and title:
                    parsed["title"] = title
                if not parsed.get("url"):
                    parsed["url"] = source_url
                return parsed

    for value in payload.values():
        if isinstance(value, (dict, list)):
            parsed = _parse_jina_reader_json(value, source_url)
            if parsed.get("content"):
                if not parsed.get("title") and title:
                    parsed["title"] = title
                if not parsed.get("url"):
                    parsed["url"] = source_url
                return parsed

    return {
        "ok": False,
        "provider": "jina_ai",
        "title": title,
        "url": source_url,
        "content": "",
        "html": html,
    }


async def jina_ai_page_extract(
    url: str,
    max_chars: int = 8000,
    timeout: float = 20.0,
    config: dict[str, Any] | None = None,
    **_: Any,
) -> Dict[str, Any]:
    httpx = _load_httpx_module()
    source_url = _normalize_reader_source_url(url)
    target_url = f"https://r.jina.ai/{source_url}"
    headers = _jina_headers(
        config=config,
        capability="page_extract",
        default_accept="text/plain",
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(target_url, headers=headers)
        response.raise_for_status()
        raw_text = response.text
        meta = await _build_usage_meta(
            client,
            response=response,
            raw_text=raw_text,
            capability="page_extract",
        )

    content_type = _coerce_text(response.headers.get("content-type")).lower()
    if "json" in content_type:
        payload = _parse_jina_reader_json(response.json(), source_url)
    else:
        try:
            payload = _parse_jina_reader_json(response.json(), source_url)
        except Exception:
            payload = _parse_jina_reader_text(response.text, source_url)
    payload["content"] = str(payload.get("content") or "")[: max(1, int(max_chars))]
    payload["_meta"] = meta
    payload["_model_markdown"] = str(payload.get("content") or "").strip()
    return payload


__all__ = ["jina_ai_page_extract", "jina_ai_search"]
