from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING, Any

from loguru import logger


if TYPE_CHECKING:
    from .server import WebToolSuite


_LOCK = threading.RLock()
_SUITE: "WebToolSuite | None" = None
_HEADLESS = False
_PRESTART_LOCK = threading.RLock()
_PRESTART_THREAD: threading.Thread | None = None
_PRESTART_DONE = threading.Event()
_PRESTART_FAILED = False
_PRESTART_ERROR = ""
_VALID_TIME_RANGE = {"a", "d", "w", "m", "y"}


def _tool_logs_enabled() -> bool:
    raw = str(os.environ.get("HYW_SHOW_TOOL_LOGS", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


if not _tool_logs_enabled():
    logger.disable("tools.web_search")


def _run_awaitable(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    out: dict[str, Any] = {}
    err: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            out["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # noqa: BLE001
            err["value"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in err:
        raise err["value"]
    return out.get("value")


def _get_suite() -> WebToolSuite:
    global _SUITE
    if _SUITE is None:
        from .server import WebToolSuite

        _SUITE = WebToolSuite(headless=_HEADLESS)
    return _SUITE


def _normalize_text(value: Any, *, max_chars: int = 160) -> str:
    text = " ".join(str(value or "").replace("　", " ").split()).strip()
    return text[:max_chars].strip()


def _normalize_time_range(value: Any) -> str:
    code = str(value or "a").strip().lower() or "a"
    return code if code in _VALID_TIME_RANGE else "a"


def _resolve_query(description: Any, user_message: Any, explicit_query: Any) -> str:
    for candidate in (explicit_query, user_message, description):
        text = _normalize_text(candidate, max_chars=120)
        if text:
            return text
    return ""


def _compact_text_result(row: dict[str, Any]) -> dict[str, Any] | None:
    url = _normalize_text(row.get("url"), max_chars=400)
    title = _normalize_text(row.get("title") or "No Title", max_chars=200)
    snippet = _normalize_text(row.get("snippet") or row.get("intro"), max_chars=240)
    if not (url or title or snippet):
        return None
    return {"title": title or "No Title", "url": url, "snippet": snippet}


def _load_web_search_config() -> dict[str, Any]:
    try:
        from ...main import load_config

        conf = load_config()
    except Exception:
        return {}
    tools_conf = conf.get("tools") if isinstance(conf, dict) else {}
    if not isinstance(tools_conf, dict):
        return {}
    web_conf = tools_conf.get("web_search")
    return web_conf if isinstance(web_conf, dict) else {}


def _ensure_prestart_background(headless: bool) -> None:
    global _PRESTART_THREAD, _PRESTART_FAILED, _PRESTART_ERROR

    def _runner() -> None:
        global _PRESTART_FAILED, _PRESTART_ERROR
        err_msg = ""
        try:
            from ._public.browser.service import get_screenshot_service

            svc = get_screenshot_service(headless=bool(headless))
            svc._ensure_ready()
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc or "").strip() or exc.__class__.__name__
        finally:
            with _PRESTART_LOCK:
                _PRESTART_FAILED = bool(err_msg)
                _PRESTART_ERROR = err_msg
            _PRESTART_DONE.set()

    with _PRESTART_LOCK:
        if _PRESTART_THREAD is not None and _PRESTART_THREAD.is_alive():
            return
        _PRESTART_DONE.clear()
        _PRESTART_FAILED = False
        _PRESTART_ERROR = ""
        _PRESTART_THREAD = threading.Thread(target=_runner, name="hyw-web-search-prestart", daemon=True)
        _PRESTART_THREAD.start()


def on_startup(headless: bool = False) -> None:
    global _HEADLESS, _PRESTART_FAILED, _PRESTART_ERROR
    effective_headless = bool(headless)
    with _LOCK:
        _HEADLESS = effective_headless
    if str(os.environ.get("HYW_WEB_PRESTART", "1")).strip().lower() not in {"1", "true", "yes", "on"}:
        _PRESTART_DONE.set()
        return
    _ensure_prestart_background(_HEADLESS)
    with _PRESTART_LOCK:
        thread = _PRESTART_THREAD
    if thread is not None:
        thread.join()
    _PRESTART_DONE.wait()
    with _PRESTART_LOCK:
        failed = bool(_PRESTART_FAILED)
        err = str(_PRESTART_ERROR or "").strip()
    if failed:
        try:
            from ._public.browser.service import get_screenshot_service

            svc = get_screenshot_service(headless=effective_headless)
            svc._ensure_ready()
            with _PRESTART_LOCK:
                _PRESTART_FAILED = False
                _PRESTART_ERROR = ""
        except Exception as exc:  # noqa: BLE001
            msg = str(exc or "").strip() or err or "web_search prestart failed"
            raise RuntimeError(msg) from exc


def wait_until_ready(headless: bool = False) -> None:
    on_startup(headless=headless)


def _close_renderer_if_exists() -> None:
    try:
        from ._public.browser import renderer as renderer_module

        renderer = getattr(renderer_module, "_content_renderer", None)
        if renderer is None:
            return
        _run_awaitable(renderer.close())
        renderer_module._content_renderer = None
    except Exception:
        pass


def on_shutdown() -> None:
    global _SUITE
    with _LOCK:
        _SUITE = None
        _close_renderer_if_exists()
        try:
            from ._public.browser.service import close_screenshot_service

            _run_awaitable(close_screenshot_service())
        except Exception:
            pass
        from ._public.browser.manager import close_shared_browser

        close_shared_browser()


def web_search(
    description: str,
    user_message: str = "",
    query: str = "",
    kl: str = "",
    time_range: str = "a",
    max_results: int = 5,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Search the web for current information and return normalized text results."""
    web_conf = _load_web_search_config()
    defaults = web_conf.get("defaults") if isinstance(web_conf.get("defaults"), dict) else {}

    desc = _normalize_text(description, max_chars=240)
    msg = _normalize_text(user_message, max_chars=240) or desc
    resolved_query = _resolve_query(desc, msg, query)
    if not resolved_query:
        raise ValueError("Missing required parameter: description")

    resolved_kl = str(kl if kl not in (None, "") else defaults.get("kl") or "").strip()
    time_range_value = time_range if time_range not in (None, "") else defaults.get("time_range") or "a"
    resolved_time_range = _normalize_time_range(time_range_value)
    max_results_value = max_results if max_results not in (None, "") else defaults.get("max_results", 5)
    try:
        resolved_max_results = max(1, min(int(max_results_value or 5), 8))
    except Exception:
        resolved_max_results = 5

    payload = _run_awaitable(
        _get_suite().web_search(
            query=resolved_query,
            mode="text",
            kl=resolved_kl,
            time_range=resolved_time_range,
            max_results=resolved_max_results,
            progress_callback=progress if callable(progress) else None,
        )
    )
    rows = payload.get("results") if isinstance(payload, dict) else []
    results = [item for item in (_compact_text_result(row) for row in rows if isinstance(row, dict)) if item]
    return {
        "query": resolved_query,
        "search_hint": "use explicit query if provided, otherwise search the user message directly",
        "kl": resolved_kl,
        "time_range": resolved_time_range,
        "max_results": resolved_max_results,
        "results": results,
        "count": len(results),
    }


__all__ = ["on_shutdown", "on_startup", "wait_until_ready", "web_search"]
