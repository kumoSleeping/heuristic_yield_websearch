from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .render import RenderCallable, RenderResult, render_markdown_base64, render_markdown_result

_VALID_SEARCH_MODES = {"text"}
_VALID_TIME_RANGE_CODES = {"a", "d", "w", "m", "y"}
_suite_lock = threading.RLock()
_suite: "WebToolSuite | None" = None


def _normalize_query(query: str) -> str:
    return str(query or "").replace('"', " ").replace("“", " ").replace("”", " ").strip()


def _normalize_time_range(time_range: str) -> Optional[str]:
    raw = str(time_range or "").strip().lower()
    if not raw or raw == "a":
        return None
    if raw in _VALID_TIME_RANGE_CODES:
        return raw
    return None


def _normalize_mode(mode: str) -> str:
    raw = str(mode or "text").strip().lower() or "text"
    if raw not in _VALID_SEARCH_MODES:
        raise ValueError(f"unsupported mode for web_search: {mode}")
    return raw


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
                "ddgs is not installed; install `ddgs` (preferred) or keep "
                "`duckduckgo-search` during the transition."
            ) from ddgs_exc


def _normalize_text_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = str(
        row.get("href")
        or row.get("url")
        or row.get("link")
        or row.get("source")
        or ""
    ).strip()
    if not url.startswith(("http://", "https://")):
        return None

    title = str(row.get("title") or row.get("name") or "").strip() or "No Title"
    snippet = str(
        row.get("body")
        or row.get("snippet")
        or row.get("description")
        or row.get("content")
        or ""
    ).strip()[:500]

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
    }


class WebToolSuite:
    def __init__(self, headless: bool = False, render_markdown: Optional[RenderCallable] = None):
        self._headless = bool(headless)
        self._render_markdown = render_markdown or render_markdown_base64
        self._search_index_counter = 0
        logger.info("WebToolSuite(websearch): initialized")

    def _next_search_index(self) -> int:
        self._search_index_counter += 1
        return self._search_index_counter

    @staticmethod
    def _notify_progress(
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(payload)
        except Exception:
            pass

    async def _search_text(
        self,
        query: str,
        kl: Optional[str],
        time_range: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        def _run() -> List[Dict[str, Any]]:
            DDGS = _load_ddgs_class()
            kwargs = {
                "region": _normalize_region(kl),
                "safesearch": "moderate",
                "timelimit": time_range,
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
            normalized = _normalize_text_row(row)
            if not normalized:
                continue
            url = str(normalized.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(normalized)
            if len(results) >= max(1, int(max_results)):
                break

        logger.info("WebToolSuite(websearch): mapped {} result(s) for '{}'", len(results), query)
        return results

    def set_render_markdown(self, render_markdown: Optional[RenderCallable]) -> None:
        self._render_markdown = render_markdown or render_markdown_base64

    async def render_markdown(
        self,
        markdown_text: str,
        title: str = "",
        theme_color: str = "#ef4444",
        **kwargs: Any,
    ) -> Optional[str]:
        if self._render_markdown is None:
            raise RuntimeError("render_markdown is not configured")

        result = self._render_markdown(
            markdown_text=markdown_text,
            title=title,
            theme_color=theme_color,
            **kwargs,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    async def markdown_llm_render(
        self,
        markdown_text: str,
        title: str = "Assistant Response",
        theme_color: str = "#ef4444",
        **_: Any,
    ) -> Dict[str, Any]:
        if self._render_markdown is render_markdown_base64:
            payload = await render_markdown_result(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
            )
        else:
            raw = await self.render_markdown(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
            )
            if isinstance(raw, dict):
                payload = raw
            else:
                payload = {
                    "ok": bool(raw),
                    "renderer": "custom",
                    "mime_type": "image/jpeg",
                    "base64": str(raw or "").strip(),
                }

        base64_data = str(payload.get("base64") or "").strip()
        mime_type = str(payload.get("mime_type") or "image/jpeg").strip() or "image/jpeg"
        if not base64_data:
            return {
                "ok": False,
                "message": "render failed",
                "renderer": str(payload.get("renderer") or "unknown"),
            }

        return {
            "ok": True,
            "message": "render success",
            "renderer": str(payload.get("renderer") or "non-browser").strip() or "non-browser",
            "image": {
                "mime_type": mime_type,
                "base64": base64_data,
            },
        }

    async def web_search(
        self,
        query: str,
        mode: str = "text",
        kl: str = "",
        time_range: str = "a",
        max_results: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("query is empty")

        normalized_mode = _normalize_mode(mode)
        region = str(kl or "").strip() or None
        normalized_time = _normalize_time_range(time_range)
        limit = max(1, min(int(max_results), 10))

        self._notify_progress(
            progress_callback,
            {
                "phase": "search",
                "status": "start",
                "query": normalized_query,
                "mode": normalized_mode,
            },
        )

        rows = await self._search_text(normalized_query, region, normalized_time, limit)
        results = [
            {
                "index": self._next_search_index(),
                "title": str(row.get("title") or "").strip() or "No Title",
                "url": str(row.get("url") or "").strip(),
                "intro": str(row.get("snippet") or "").strip(),
                "snippet": str(row.get("snippet") or "").strip(),
                "images": [],
                "images_local": [],
                "image_details": [],
                "image_count": 0,
            }
            for row in rows
            if isinstance(row, dict)
        ]

        self._notify_progress(
            progress_callback,
            {
                "phase": "search",
                "status": "done",
                "query": normalized_query,
                "mode": normalized_mode,
                "count": len(results),
            },
        )

        return {
            "query": normalized_query,
            "mode": normalized_mode,
            "filters": {
                "kl": region,
                "time_range": normalized_time,
            },
            "count": len(results),
            "results": results,
        }


def _get_suite(headless: bool = False) -> WebToolSuite:
    global _suite
    with _suite_lock:
        if _suite is None:
            _suite = WebToolSuite(headless=headless)
        return _suite


def on_startup(headless: bool = False) -> None:
    _get_suite(headless=headless)


def on_shutdown() -> None:
    global _suite
    with _suite_lock:
        _suite = None


async def wait_until_ready(timeout: float | None = None) -> bool:
    del timeout
    return True


async def web_search(
    query: str,
    mode: str = "text",
    kl: str = "",
    time_range: str = "a",
    max_results: int = 5,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    suite = _get_suite()
    return await suite.web_search(
        query=query,
        mode=mode,
        kl=kl,
        time_range=time_range,
        max_results=max_results,
        progress_callback=progress_callback,
    )


__all__ = [
    "RenderCallable",
    "RenderResult",
    "WebToolSuite",
    "on_shutdown",
    "on_startup",
    "render_markdown_base64",
    "render_markdown_result",
    "wait_until_ready",
    "web_search",
]
