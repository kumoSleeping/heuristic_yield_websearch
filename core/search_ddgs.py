from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional


def _normalize_region(kl: Optional[str]) -> str:
    raw = str(kl or "").strip().lower()
    return raw or "wt-wt"


def _load_ddgs_class() -> Any:
    try:
        from ddgs import DDGS

        return DDGS
    except Exception as ddgs_exc:
        try:
            from duckduckgo_search import DDGS

            return DDGS
        except Exception:
            raise RuntimeError(
                "ddgs is not installed; reinstall `hyw` or install `ddgs` manually."
            ) from ddgs_exc


def _normalize_search_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        "provider": "ddgs",
    }


async def ddgs_search(
    query: str,
    kl: Optional[str] = None,
    max_results: int = 5,
    **_: Any,
) -> List[Dict[str, Any]]:
    def _run() -> List[Dict[str, Any]]:
        DDGS = _load_ddgs_class()
        kwargs = {
            "region": _normalize_region(kl),
            "safesearch": "moderate",
            "max_results": max(1, int(max_results)),
            "page": 1,
            "backend": "duckduckgo",
        }
        try:
            return list(DDGS().text(query, **kwargs) or [])
        except TypeError:
            kwargs.pop("backend", None)
            return list(DDGS().text(query, **kwargs) or [])

    raw_rows = await asyncio.to_thread(_run)
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_search_row(row)
        if not normalized:
            continue
        url = normalized["url"]
        if url in seen:
            continue
        seen.add(url)
        results.append(normalized)
        if len(results) >= max(1, int(max_results)):
            break
    return results


__all__ = ["ddgs_search"]
