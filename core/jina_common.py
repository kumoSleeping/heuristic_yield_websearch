from __future__ import annotations

import os
from typing import Any

from .config import cfg_get

_JINA_CONFIG_RESERVED_KEYS = {"headers", "search", "page_extract", "prefer_free"}
_JINA_CAPABILITY_RESERVED_KEYS = {"headers", "prefer_free"}


def load_httpx_module() -> Any:
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


def _jina_config_roots(config: dict[str, Any] | None, provider_name: str) -> list[dict[str, Any]]:
    roots = [
        cfg_get(config or {}, provider_name, {}),
        cfg_get(config or {}, f"tools.config.{provider_name}", {}),
    ]
    return [root for root in roots if isinstance(root, dict)]


def _get_header_key(headers: dict[str, str], name: str) -> str | None:
    target = str(name or "").strip().lower()
    for key in headers:
        if str(key).strip().lower() == target:
            return key
    return None


def has_auth_header(headers: dict[str, str]) -> bool:
    return _get_header_key(headers, "Authorization") is not None


def jina_headers(
    *,
    config: dict[str, Any] | None = None,
    provider_name: str,
    capability: str,
    default_accept: str,
    default_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "User-Agent": "hyw/0.1 (+https://github.com/kumoSleeping/heuristic_yield_websearch)",
    }
    api_key = str(os.environ.get("JINA_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for root in _jina_config_roots(config, provider_name):
        headers.update(_collect_header_block(root, reserved_keys=_JINA_CONFIG_RESERVED_KEYS))
        capability_block = root.get(capability) if isinstance(root, dict) else None
        headers.update(
            _collect_header_block(
                capability_block,
                reserved_keys=_JINA_CAPABILITY_RESERVED_KEYS,
            )
        )

    headers.setdefault("Accept", default_accept)
    for key, value in (default_headers or {}).items():
        text = str(value or "").strip()
        if text:
            headers.setdefault(str(key).strip(), text)
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


async def build_usage_meta(
    client: Any,
    *,
    response: Any,
    raw_text: str,
    provider: str,
    capability: str,
    billed: bool | None = None,
) -> dict[str, Any]:
    tokens = _to_int(response.headers.get("x-usage-tokens"))
    source = "header" if tokens > 0 else ""
    if tokens <= 0 and raw_text:
        tokens = await _count_tokens_via_segment(client, raw_text)
        if tokens > 0:
            source = "segment"

    return {
        "provider": str(provider or "").strip(),
        "capability": capability,
        "billing": {
            "mode": "paid" if billed else "free",
            "cost_usd": 0.0 if billed is False else None,
        },
        "usage": {
            "requests": 1,
            "tokens": tokens,
            "source": source or "unknown",
        },
    }


__all__ = [
    "build_usage_meta",
    "has_auth_header",
    "jina_headers",
    "load_httpx_module",
]
