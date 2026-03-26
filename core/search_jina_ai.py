from __future__ import annotations

from typing import Any, Dict

from .jina_common import (
    build_usage_meta,
    has_auth_header,
    jina_headers,
    load_httpx_module,
)


def _jina_headers(
    *,
    config: dict[str, Any] | None = None,
    capability: str,
    default_accept: str,
) -> dict[str, str]:
    default_headers = {}
    if capability == "page_extract":
        default_headers = {
            "X-Engine": "browser",
            "X-Return-Format": "markdown",
        }
    return jina_headers(
        config=config,
        provider_name="jina_ai",
        capability=capability,
        default_accept=default_accept,
        default_headers=default_headers,
    )


def _normalize_reader_source_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        raise ValueError("url is empty")
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    return text


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


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
    httpx = load_httpx_module()
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
        meta = await build_usage_meta(
            client,
            response=response,
            raw_text=raw_text,
            provider="jina_ai",
            capability="page_extract",
            billed=has_auth_header(headers),
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


__all__ = ["jina_ai_page_extract"]
