from __future__ import annotations

import inspect
import threading
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .config import resolve_tool_handlers
from .render import RenderCallable, RenderResult, render_markdown_result
from .search_ddgs import ddgs_search
from .search_jina_ai import jina_ai_page_extract, jina_ai_search

_VALID_SEARCH_MODES = {"text"}
_suite_lock = threading.RLock()
_suite: "WebToolSuite | None" = None
_suite_signature: tuple[Any, ...] | None = None


def _normalize_query(query: str) -> str:
    return str(query or "").replace('"', " ").replace("“", " ").replace("”", " ").strip()


def _normalize_mode(mode: str) -> str:
    raw = str(mode or "text").strip().lower() or "text"
    if raw not in _VALID_SEARCH_MODES:
        raise ValueError(f"unsupported mode for web_search: {mode}")
    return raw


def _normalize_reader_source_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        raise ValueError("url is empty")
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    return text


def _normalize_search_row(row: Dict[str, Any], provider: str) -> Optional[Dict[str, Any]]:
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
        snippet = snippet[len(title):].strip(" -:\n")

    return {
        "title": title,
        "url": url,
        "snippet": snippet[:500],
        "provider": provider,
    }


def _extract_handler_meta(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _extract_search_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _looks_like_payment_required_error(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return "402" in lowered and "payment required" in lowered


class WebToolSuite:
    def __init__(
        self,
        headless: bool = False,
        render_markdown: Optional[RenderCallable] = None,
        config: dict[str, Any] | None = None,
    ):
        self._headless = bool(headless)
        self._render_markdown = render_markdown
        self._config = dict(config or {})
        self._search_handlers = resolve_tool_handlers(self._config, "search")
        search_providers = {handler.provider for handler in self._search_handlers}
        self._emergency_search_handlers = (
            []
            if "ddgs" in search_providers
            else resolve_tool_handlers(self._config, "search", selection="ddgs")
        )
        self._extract_handlers = resolve_tool_handlers(self._config, "page_extract")
        self._render_handlers = resolve_tool_handlers(self._config, "render")
        self._search_index_counter = 0
        logger.info(
            "WebToolSuite: initialized search={} extract={} render={}",
            [handler.provider for handler in self._search_handlers],
            [handler.provider for handler in self._extract_handlers],
            [handler.provider for handler in self._render_handlers],
        )

    def warm_up(self) -> dict[str, Any]:
        return {}

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

    async def _call_handler(self, handler: Any, **kwargs: Any) -> Any:
        result = handler.callable(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _search_via_handlers(
        self,
        handlers: list[Any],
        *,
        query: str,
        kl: Optional[str],
        max_results: int,
        errors: list[str],
    ) -> tuple[List[Dict[str, Any]], dict[str, Any]] | None:
        for handler in handlers:
            try:
                raw = await self._call_handler(
                    handler,
                    query=query,
                    kl=kl,
                    max_results=max_results,
                    headless=self._headless,
                    config=self._config,
                )
            except Exception as exc:
                errors.append(f"{handler.provider}: {exc}")
                continue

            rows = _extract_search_rows(raw)
            if not rows:
                errors.append(f"{handler.provider}: invalid result type")
                continue

            meta = _extract_handler_meta(raw)
            results: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_search_row(row, handler.provider)
                if not normalized:
                    continue
                url = normalized["url"]
                if url in seen:
                    continue
                seen.add(url)
                results.append(normalized)
                if len(results) >= max(1, int(max_results)):
                    break
            if results:
                logger.info("WebToolSuite(search): {} -> {} result(s)", handler.provider, len(results))
                return results, meta
            errors.append(f"{handler.provider}: empty result")
        return None

    async def _search_text(
        self,
        query: str,
        kl: Optional[str],
        max_results: int,
    ) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
        errors: list[str] = []
        primary = await self._search_via_handlers(
            self._search_handlers,
            query=query,
            kl=kl,
            max_results=max_results,
            errors=errors,
        )
        if primary is not None:
            return primary

        if self._emergency_search_handlers and any(_looks_like_payment_required_error(item) for item in errors):
            logger.warning("WebToolSuite(search): configured provider hit 402, falling back to {}", [handler.provider for handler in self._emergency_search_handlers])
            fallback = await self._search_via_handlers(
                self._emergency_search_handlers,
                query=query,
                kl=kl,
                max_results=max_results,
                errors=errors,
            )
            if fallback is not None:
                return fallback

        raise RuntimeError("; ".join(errors) if errors else "no search provider is configured")

    def set_render_markdown(self, render_markdown: Optional[RenderCallable]) -> None:
        self._render_markdown = render_markdown

    async def render_markdown(
        self,
        markdown_text: str,
        title: str = "",
        theme_color: str = "#ef4444",
        **kwargs: Any,
    ) -> Optional[str]:
        if self._render_markdown is None:
            payload = await render_markdown_result(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
                config=self._config,
                headless=self._headless,
            )
            return str(payload.get("base64") or "").strip()

        result = self._render_markdown(
            markdown_text=markdown_text,
            title=title,
            theme_color=theme_color,
            headless=self._headless,
            config=self._config,
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
        if self._render_markdown is None:
            payload = await render_markdown_result(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
                config=self._config,
                headless=self._headless,
            )
        else:
            raw = await self.render_markdown(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
            )
            payload = raw if isinstance(raw, dict) else {
                "ok": bool(raw),
                "renderer": "custom",
                "mime_type": "image/png",
                "base64": str(raw or "").strip(),
            }

        base64_data = str(payload.get("base64") or "").strip()
        mime_type = str(payload.get("mime_type") or "image/png").strip() or "image/png"
        if not base64_data:
            return {
                "ok": False,
                "message": "render failed",
                "renderer": str(payload.get("renderer") or "unknown"),
            }

        return {
            "ok": True,
            "message": "render success",
            "renderer": str(payload.get("renderer") or "unknown"),
            "image": {
                "mime_type": mime_type,
                "base64": base64_data,
            },
        }

    async def page_extract(
        self,
        url: str,
        max_chars: int = 8000,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        source_url = _normalize_reader_source_url(url)
        self._notify_progress(
            progress_callback,
            {
                "phase": "page_extract",
                "status": "start",
                "url": source_url,
            },
        )

        errors: list[str] = []
        for handler in self._extract_handlers:
            try:
                payload = await self._call_handler(
                    handler,
                    url=source_url,
                    max_chars=max_chars,
                    headless=self._headless,
                    config=self._config,
                )
            except Exception as exc:
                errors.append(f"{handler.provider}: {exc}")
                continue

            if not isinstance(payload, dict):
                errors.append(f"{handler.provider}: invalid result type")
                continue
            content = str(payload.get("content") or "").strip()
            if not content:
                errors.append(f"{handler.provider}: empty content")
                continue

            result = {
                "ok": True,
                "provider": handler.provider,
                "title": str(payload.get("title") or "").strip(),
                "url": str(payload.get("url") or source_url).strip() or source_url,
                "content": content[: max(1, int(max_chars))],
                "html": str(payload.get("html") or ""),
                "_meta": _extract_handler_meta(payload),
            }
            self._notify_progress(
                progress_callback,
                {
                    "phase": "page_extract",
                    "status": "done",
                    "url": result["url"],
                    "provider": handler.provider,
                },
            )
            return result

        error_text = "; ".join(errors) if errors else "no page extract provider is configured"
        self._notify_progress(
            progress_callback,
            {
                "phase": "page_extract",
                "status": "failed",
                "url": source_url,
                "error": error_text,
            },
        )
        return {
            "ok": False,
            "provider": "",
            "title": "",
            "url": source_url,
            "content": "",
            "html": "",
            "error": error_text,
        }

    async def web_search(
        self,
        query: str,
        mode: str = "text",
        kl: str = "",
        max_results: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("query is empty")

        normalized_mode = _normalize_mode(mode)
        region = str(kl or "").strip() or None
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

        rows, meta = await self._search_text(normalized_query, region, limit)
        results = [
            {
                "index": self._next_search_index(),
                "title": str(row.get("title") or "").strip() or "No Title",
                "url": str(row.get("url") or "").strip(),
                "intro": str(row.get("snippet") or "").strip(),
                "snippet": str(row.get("snippet") or "").strip(),
                "provider": str(row.get("provider") or "").strip(),
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
            },
            "count": len(results),
            "results": results,
            "_meta": meta,
        }


def _suite_signature_for(headless: bool, config: dict[str, Any] | None) -> tuple[Any, ...]:
    cfg = dict(config or {})
    groups: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for capability in ("search", "page_extract", "render"):
        handlers = resolve_tool_handlers(cfg, capability)
        groups.append((capability, tuple((handler.provider, handler.target) for handler in handlers)))
    return (bool(headless), tuple(groups))


def _get_suite(headless: bool = False, config: dict[str, Any] | None = None) -> WebToolSuite:
    global _suite, _suite_signature
    signature = _suite_signature_for(headless, config)
    with _suite_lock:
        if _suite is None or _suite_signature != signature:
            _suite = WebToolSuite(headless=headless, config=config)
            _suite_signature = signature
        return _suite


def on_startup(headless: bool = False, config: dict[str, Any] | None = None) -> dict[str, Any]:
    suite = _get_suite(headless=headless, config=config)
    return suite.warm_up()


def on_shutdown() -> None:
    global _suite, _suite_signature
    with _suite_lock:
        _suite = None
        _suite_signature = None


async def wait_until_ready(timeout: float | None = None) -> bool:
    del timeout
    return True


async def web_search(
    query: str,
    mode: str = "text",
    kl: str = "",
    max_results: int = 5,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    config: dict[str, Any] | None = None,
    headless: bool = False,
) -> Dict[str, Any]:
    suite = _get_suite(headless=headless, config=config)
    return await suite.web_search(
        query=query,
        mode=mode,
        kl=kl,
        max_results=max_results,
        progress_callback=progress_callback,
    )


async def page_extract(
    url: str,
    max_chars: int = 8000,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    config: dict[str, Any] | None = None,
    headless: bool = False,
) -> Dict[str, Any]:
    suite = _get_suite(headless=headless, config=config)
    return await suite.page_extract(
        url=url,
        max_chars=max_chars,
        progress_callback=progress_callback,
    )


__all__ = [
    "RenderCallable",
    "RenderResult",
    "WebToolSuite",
    "ddgs_search",
    "jina_ai_page_extract",
    "jina_ai_search",
    "on_shutdown",
    "on_startup",
    "page_extract",
    "wait_until_ready",
    "web_search",
]
