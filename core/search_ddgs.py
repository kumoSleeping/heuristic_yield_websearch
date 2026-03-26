from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, unquote, urlsplit

from .jina_common import (
    build_usage_meta,
    has_auth_header,
    jina_headers,
    load_httpx_module,
)

_DDG_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
_JINA_DDGS_DEFAULT_HEADERS = {
    "X-Engine": "browser",
    "X-Return-Format": "markdown",
}


def _normalize_region(kl: Optional[str]) -> str:
    raw = str(kl or "").strip().lower()
    return raw or "wt-wt"


def _normalize_search_filters(
    *,
    kl: Any = None,
    region: Any = None,
    df: Any = None,
    timelimit: Any = None,
    t: Any = None,
    ia: Any = None,
) -> dict[str, str]:
    filters: dict[str, str] = {}

    region_text = str(kl or region or "").strip().lower()
    if region_text:
        filters["kl"] = region_text

    time_text = str(df or timelimit or "").strip()
    if time_text:
        filters["df"] = time_text

    source_tag = str(t or "").strip()
    if source_tag:
        filters["t"] = source_tag

    interface = str(ia or "").strip().lower()
    if interface:
        filters["ia"] = interface

    return filters


def build_duckduckgo_html_url(
    query: str,
    *,
    kl: Any = None,
    region: Any = None,
    df: Any = None,
    timelimit: Any = None,
    t: Any = None,
    ia: Any = None,
) -> str:
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("query is empty")

    filters = _normalize_search_filters(
        kl=kl,
        region=region,
        df=df,
        timelimit=timelimit,
        t=t,
        ia=ia,
    )
    params: list[tuple[str, str]] = [("q", clean_query)]
    if filters.get("kl"):
        params.append(("l", filters["kl"]))
    if filters.get("df"):
        params.append(("df", filters["df"]))
    if filters.get("t"):
        params.append(("t", filters["t"]))
    params.append(("ia", filters.get("ia") or "web"))
    return f"{_DDG_HTML_SEARCH_URL}?{urlencode(params)}"


def _build_ddgs_kwargs(
    *,
    kl: Any = None,
    region: Any = None,
    df: Any = None,
    timelimit: Any = None,
    max_results: int = 5,
) -> dict[str, Any]:
    filters = _normalize_search_filters(
        kl=kl,
        region=region,
        df=df,
        timelimit=timelimit,
    )
    return {
        "region": _normalize_region(filters.get("kl")),
        "safesearch": "moderate",
        "timelimit": filters.get("df") or None,
        "max_results": max(1, int(max_results)),
        "page": 1,
        "backend": "duckduckgo",
    }


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


def _unwrap_duckduckgo_redirect(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except Exception:
        return text

    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target).strip()
    return text


def _normalize_search_row(row: Dict[str, Any], *, provider: str) -> Optional[Dict[str, Any]]:
    url = _unwrap_duckduckgo_redirect(
        str(
            row.get("href")
            or row.get("url")
            or row.get("link")
            or row.get("source")
            or row.get("document_url")
            or ""
        ).strip()
    )
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
    snippet = re.sub(r"\s+", " ", snippet).strip()

    return {
        "title": title,
        "url": url,
        "snippet": snippet[:500],
        "provider": provider,
    }


def _dedupe_rows(
    raw_rows: list[dict[str, Any]],
    *,
    provider: str,
    max_results: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_search_row(row, provider=provider)
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


def _clean_markdown_line(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", clean)
    clean = re.sub(r"[*_`#>]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _looks_like_linkish_stub(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    lowered = clean.lower()
    if lowered == "duckduckgo":
        return True
    if clean.startswith(("http://", "https://")):
        return True
    if " " not in clean and ("/" in clean or "." in clean):
        return True
    return False


def _extract_markdown_snippet(text: str, *, title: str) -> str:
    parts: list[str] = []
    clean_title = str(title or "").strip()
    for raw_line in str(text or "").splitlines():
        clean = _clean_markdown_line(raw_line)
        if not clean:
            continue
        if clean_title and clean == clean_title:
            continue
        if _looks_like_linkish_stub(clean):
            continue
        parts.append(clean)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:500]


def _parse_markdown_links(text: str, *, provider: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    link_re = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)]+)\)")
    for match in link_re.finditer(str(text or "")):
        title = str(match.group("title") or "").strip()
        url = _unwrap_duckduckgo_redirect(str(match.group("url") or "").strip())
        if not title or not url or url in seen:
            continue
        if title.lower().startswith("image "):
            continue
        seen.add(url)
        rows.append({"title": title, "url": url, "snippet": "", "provider": provider})
    return rows


def _parse_jina_ddgs_markdown(text: str, *, max_results: int) -> list[dict[str, Any]]:
    heading_re = re.compile(
        r"^##\s+\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)]+)\)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(heading_re.finditer(str(text or "")))
    rows: list[dict[str, Any]] = []

    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(str(text or ""))
        title = str(match.group("title") or "").strip()
        url = _unwrap_duckduckgo_redirect(str(match.group("url") or "").strip())
        snippet = _extract_markdown_snippet(str(text or "")[body_start:body_end], title=title)
        rows.append(
            {
                "title": title or "No Title",
                "url": url,
                "snippet": snippet,
                "provider": "jina_ddgs",
            }
        )

    if not rows:
        rows = _parse_markdown_links(text, provider="jina_ddgs")
    return _dedupe_rows(rows, provider="jina_ddgs", max_results=max_results)


async def ddgs_search(
    query: str,
    kl: Optional[str] = None,
    max_results: int = 5,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    def _run() -> List[Dict[str, Any]]:
        DDGS = _load_ddgs_class()
        params = _build_ddgs_kwargs(
            kl=kl,
            region=kwargs.get("region"),
            df=kwargs.get("df"),
            timelimit=kwargs.get("timelimit"),
            max_results=max_results,
        )
        try:
            return list(DDGS().text(query, **params) or [])
        except TypeError:
            params.pop("backend", None)
            return list(DDGS().text(query, **params) or [])

    raw_rows = await asyncio.to_thread(_run)
    return _dedupe_rows(
        [row for row in raw_rows if isinstance(row, dict)],
        provider="ddgs",
        max_results=max_results,
    )


async def jina_ddgs_search(
    query: str,
    kl: Optional[str] = None,
    max_results: int = 5,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    httpx = load_httpx_module()
    search_url = build_duckduckgo_html_url(
        query,
        kl=kl,
        region=kwargs.get("region"),
        df=kwargs.get("df"),
        timelimit=kwargs.get("timelimit"),
        t=kwargs.get("t"),
        ia=kwargs.get("ia"),
    )
    headers = jina_headers(
        config=config,
        provider_name="jina_ddgs",
        capability="search",
        default_accept="text/plain",
        default_headers=_JINA_DDGS_DEFAULT_HEADERS,
    )
    target_url = f"https://r.jina.ai/{search_url}"

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(target_url, headers=headers)
        response.raise_for_status()
        raw_text = response.text
        meta = await build_usage_meta(
            client,
            response=response,
            raw_text=raw_text,
            provider="jina_ddgs",
            capability="search",
            billed=has_auth_header(headers),
        )

    return {
        "results": _parse_jina_ddgs_markdown(raw_text, max_results=max_results),
        "_meta": meta,
    }


__all__ = [
    "build_duckduckgo_html_url",
    "ddgs_search",
    "jina_ddgs_search",
]
