from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from .config import resolve_tool_handlers

RenderCallable = Callable[..., Any]
RenderResult = Dict[str, Any]


def _normalize_render_payload(payload: Any, provider: str) -> RenderResult:
    if isinstance(payload, dict):
        base64_data = str(payload.get("base64") or "").strip()
        mime_type = str(payload.get("mime_type") or "image/png").strip() or "image/png"
        renderer = str(payload.get("renderer") or provider).strip() or provider
        return {
            "ok": bool(payload.get("ok", bool(base64_data))),
            "renderer": renderer,
            "mime_type": mime_type,
            "base64": base64_data,
        }
    return {
        "ok": bool(payload),
        "renderer": provider,
        "mime_type": "image/png",
        "base64": str(payload or "").strip(),
    }


async def render_markdown_result(
    markdown_text: str,
    title: str = "Assistant Response",
    theme_color: str = "#ef4444",
    *,
    config: dict[str, Any] | None = None,
    provider: str | None = None,
    providers: list[str] | tuple[str, ...] | str | None = None,
    headless: bool = True,
) -> RenderResult:
    selection = providers if providers is not None else provider
    handlers = resolve_tool_handlers(config, "render", selection=selection)
    if not handlers:
        raise RuntimeError("no render provider is configured")

    errors: list[str] = []
    for handler in handlers:
        try:
            result = handler.callable(
                markdown_text=markdown_text,
                title=title,
                theme_color=theme_color,
                headless=headless,
                config=config,
            )
            if inspect.isawaitable(result):
                result = await result
            payload = _normalize_render_payload(result, handler.provider)
            if payload.get("ok"):
                return payload
            errors.append(f"{handler.provider}: empty render result")
        except Exception as exc:
            errors.append(f"{handler.provider}: {exc}")

    raise RuntimeError("; ".join(errors) if errors else "render failed")


async def render_markdown_base64(
    markdown_text: str,
    title: str = "Assistant Response",
    theme_color: str = "#ef4444",
    *,
    config: dict[str, Any] | None = None,
    provider: str | None = None,
    providers: list[str] | tuple[str, ...] | str | None = None,
    headless: bool = True,
) -> str:
    payload = await render_markdown_result(
        markdown_text=markdown_text,
        title=title,
        theme_color=theme_color,
        config=config,
        provider=provider,
        providers=providers,
        headless=headless,
    )
    return str(payload.get("base64") or "").strip()


__all__ = [
    "RenderCallable",
    "RenderResult",
    "render_markdown_base64",
    "render_markdown_result",
]
