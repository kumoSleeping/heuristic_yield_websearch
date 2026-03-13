from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import yaml

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_NAME = "hyw"
CONFIG_PATH = Path.home() / ".hyw" / "config.yml"
LOG_DIR = Path.home() / ".hyw" / "logs"

_TOOL_CAPABILITIES = {"search", "page_extract", "render"}
_TOOL_PROVIDER_ALIASES = {
    "ddg": "ddgs",
    "duckduckgo": "ddgs",
    "jina": "jina_ai",
    "jina-ai": "jina_ai",
    "jinaai": "jina_ai",
    "non-browser": "render",
    "non_browser": "render",
    "weasyprint": "render",
}
_TOOL_CAPABILITY_ALIASES = {
    "extract": "page_extract",
    "page": "page_extract",
    "page_extract": "page_extract",
    "page-search": "page_extract",
    "page_search": "page_extract",
    "render": "render",
    "search": "search",
}

BUILTIN_TOOL_PROVIDER_INDEX: dict[str, dict[str, str]] = {
    "ddgs": {
        "search": "core.search_ddgs:ddgs_search",
    },
    "jina_ai": {
        "search": "core.search_jina_ai:jina_ai_search",
        "page_extract": "core.search_jina_ai:jina_ai_page_extract",
    },
    "render": {
        "render": "core.render_non_browser:render_markdown_non_browser_result",
    },
}

DEFAULT_TOOL_SELECTIONS: dict[str, str] = {
    "search": "ddgs",
    "page_extract": "jina_ai",
    "render": "render",
}


@dataclass(frozen=True)
class ConfigFileState:
    path: Path
    exists: bool
    valid: bool
    raw: dict[str, Any]
    error: str | None = None


def _clean_cfg_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _clean_cfg_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        parsed = int(value)
        return parsed if parsed > 0 else None

    text = _clean_cfg_text(value)
    if not text:
        return None
    if text.isdigit():
        parsed = int(text)
        return parsed if parsed > 0 else None
    return None


def _clean_cfg_mapping(value: Any) -> dict[str, Any] | None:
    return deepcopy(value) if isinstance(value, dict) else None


def _merge_cfg_mapping(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if base is None and override is None:
        return None
    if base is None:
        return deepcopy(override)
    if override is None:
        return deepcopy(base)

    merged = deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_cfg_mapping(existing, value) or {}
        else:
            merged[key] = deepcopy(value)
    return merged


def _model_defaults(config: dict[str, Any]) -> dict[str, Any]:
    saved = config.get("_model_defaults")
    if isinstance(saved, dict):
        defaults = {
            "model": _clean_cfg_text(saved.get("model")),
            "api_key": _clean_cfg_text(saved.get("api_key")),
            "api_base": _clean_cfg_text(saved.get("api_base") or saved.get("base_url")),
            "reasoning_effort": _clean_cfg_text(saved.get("reasoning_effort")),
            "max_tokens": _clean_cfg_positive_int(saved.get("max_tokens")),
            "max_completion_tokens": _clean_cfg_positive_int(saved.get("max_completion_tokens")),
            "extra_body": _clean_cfg_mapping(saved.get("extra_body")),
        }
    else:
        defaults = {
            "model": _clean_cfg_text(config.get("model") or config.get("model_name")),
            "api_key": _clean_cfg_text(config.get("api_key")),
            "api_base": _clean_cfg_text(config.get("api_base") or config.get("base_url")),
            "reasoning_effort": _clean_cfg_text(config.get("reasoning_effort")),
            "max_tokens": _clean_cfg_positive_int(config.get("max_tokens")),
            "max_completion_tokens": _clean_cfg_positive_int(config.get("max_completion_tokens")),
            "extra_body": _clean_cfg_mapping(config.get("extra_body")),
        }
    if not defaults["model"]:
        defaults["model"] = DEFAULT_MODEL
    return defaults


def _normalize_model_profile(
    entry: Any,
    *,
    defaults: dict[str, Any],
    name_hint: str = "",
) -> dict[str, Any] | None:
    if isinstance(entry, str):
        entry = {"model": entry}
    if not isinstance(entry, dict):
        return None

    model = _clean_cfg_text(entry.get("model") or entry.get("model_name") or defaults.get("model"))
    if not model:
        model = DEFAULT_MODEL

    profile: dict[str, Any] = {"model": model}
    alias = _clean_cfg_text(entry.get("name") or entry.get("label") or name_hint)
    if alias:
        profile["name"] = alias

    api_key = _clean_cfg_text(entry.get("api_key") or defaults.get("api_key"))
    api_base = _clean_cfg_text(entry.get("api_base") or entry.get("base_url") or defaults.get("api_base"))
    reasoning_effort = _clean_cfg_text(entry.get("reasoning_effort") or defaults.get("reasoning_effort"))
    max_tokens = _clean_cfg_positive_int(entry.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _clean_cfg_positive_int(defaults.get("max_tokens"))
    max_completion_tokens = _clean_cfg_positive_int(entry.get("max_completion_tokens"))
    if max_completion_tokens is None:
        max_completion_tokens = _clean_cfg_positive_int(defaults.get("max_completion_tokens"))
    extra_body = _merge_cfg_mapping(
        _clean_cfg_mapping(defaults.get("extra_body")),
        _clean_cfg_mapping(entry.get("extra_body")),
    )

    if api_key:
        profile["api_key"] = api_key
    if api_base:
        profile["api_base"] = api_base
    if reasoning_effort:
        profile["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        profile["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        profile["max_completion_tokens"] = max_completion_tokens
    if extra_body:
        profile["extra_body"] = extra_body
    return profile


def _normalize_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = _model_defaults(config)
    raw_models = config.get("models")
    profiles: list[dict[str, Any]] = []

    if isinstance(raw_models, list):
        for entry in raw_models:
            profile = _normalize_model_profile(entry, defaults=defaults)
            if profile:
                profiles.append(profile)
    elif isinstance(raw_models, dict):
        for name, entry in raw_models.items():
            profile = _normalize_model_profile(entry, defaults=defaults, name_hint=_clean_cfg_text(name))
            if profile:
                profiles.append(profile)

    if not profiles:
        fallback = _normalize_model_profile(config, defaults=defaults)
        if fallback:
            profiles.append(fallback)

    return profiles or [{"model": DEFAULT_MODEL}]


def _pick_active_model_index(config: dict[str, Any], profiles: list[dict[str, Any]] | None = None) -> int:
    items = profiles or _normalize_models(config)
    if not items:
        return 0

    raw_index = config.get("active_model_index")
    if isinstance(raw_index, str) and raw_index.strip().isdigit():
        raw_index = int(raw_index.strip())
    if isinstance(raw_index, int):
        return max(0, min(raw_index, len(items) - 1))

    active = _clean_cfg_text(config.get("active_model"))
    if active:
        lowered = active.lower()
        for idx, profile in enumerate(items):
            model = _clean_cfg_text(profile.get("model")).lower()
            name = _clean_cfg_text(profile.get("name")).lower()
            if lowered in {model, name}:
                return idx
    return 0


def _normalize_tool_capability(value: str) -> str:
    raw = _clean_cfg_text(value).lower()
    normalized = _TOOL_CAPABILITY_ALIASES.get(raw, raw)
    if normalized not in _TOOL_CAPABILITIES:
        raise ValueError(f"unsupported tool capability: {value}")
    return normalized


def normalize_tool_provider_name(value: str) -> str:
    raw = _clean_cfg_text(value).lower()
    return _TOOL_PROVIDER_ALIASES.get(raw, raw)


def cfg_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _flatten_provider_tokens(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    if isinstance(raw, (list, tuple, set)):
        items: list[str] = []
        for item in raw:
            items.extend(_flatten_provider_tokens(item))
        return items
    return []


def _selection_candidates(config: dict[str, Any], capability: str) -> list[Any]:
    tool_node = cfg_get(config, f"tools.{capability}")
    return [
        cfg_get(config, f"tools.use.{capability}"),
        cfg_get(config, f"tools.{capability}.providers"),
        cfg_get(config, f"tools.{capability}.provider"),
        tool_node,
        config.get(f"{capability}_providers"),
        config.get(f"{capability}_provider"),
    ]


def get_tool_provider_index(config: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    index = {
        provider: dict(capabilities)
        for provider, capabilities in BUILTIN_TOOL_PROVIDER_INDEX.items()
    }
    cfg = config or {}
    custom_index = cfg_get(cfg, "tools.index", cfg_get(cfg, "tools.providers", {}))
    if not isinstance(custom_index, dict):
        return index

    for provider_name, mapping in custom_index.items():
        normalized_provider = normalize_tool_provider_name(str(provider_name))
        if not isinstance(mapping, dict):
            continue
        slot = index.setdefault(normalized_provider, {})
        for capability_name, target in mapping.items():
            if target is None:
                continue
            try:
                capability = _normalize_tool_capability(str(capability_name))
            except ValueError:
                continue
            target_text = _clean_cfg_text(target)
            if target_text:
                slot[capability] = target_text
    return index


def get_tool_provider_name(
    config: dict[str, Any] | None,
    capability: str,
    selection: Any = None,
) -> str:
    normalized_capability = _normalize_tool_capability(capability)
    if selection is not None:
        raw_items = _flatten_provider_tokens(selection)
    else:
        raw_items = []
        for candidate in _selection_candidates(config or {}, normalized_capability):
            if isinstance(candidate, dict):
                raw_items = _flatten_provider_tokens(
                    candidate.get("providers")
                    or candidate.get("provider")
                    or candidate.get("use")
                )
            else:
                raw_items = _flatten_provider_tokens(candidate)
            if raw_items:
                break

    normalized = [
        item if ":" in item else normalize_tool_provider_name(item)
        for item in raw_items
        if _clean_cfg_text(item)
    ]
    if normalized:
        return normalized[0]
    return str(DEFAULT_TOOL_SELECTIONS.get(normalized_capability, "") or "")


@dataclass(frozen=True)
class ToolHandler:
    provider: str
    capability: str
    target: str
    callable: Callable[..., Any]


@lru_cache(maxsize=64)
def _load_callable(target: str) -> Callable[..., Any]:
    module_name, _, attr_name = str(target).partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"invalid tool target: {target}")
    module = import_module(module_name)
    fn = getattr(module, attr_name, None)
    if not callable(fn):
        raise TypeError(f"target is not callable: {target}")
    return fn


def resolve_tool_handlers(
    config: dict[str, Any] | None,
    capability: str,
    selection: Any = None,
) -> list[ToolHandler]:
    normalized_capability = _normalize_tool_capability(capability)
    index = get_tool_provider_index(config)
    name = get_tool_provider_name(config, normalized_capability, selection=selection)
    if not name:
        return []
    if ":" in name:
        provider = name
        target = name
    else:
        provider = normalize_tool_provider_name(name)
        target = _clean_cfg_text(index.get(provider, {}).get(normalized_capability))
        if not target:
            return []
    return [
        ToolHandler(
            provider=provider,
            capability=normalized_capability,
            target=target,
            callable=_load_callable(target),
        )
    ]


def build_model_config(config: dict[str, Any] | None = None, model_index: int | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    profiles = _normalize_models(raw)
    index = _pick_active_model_index(raw, profiles) if model_index is None else int(model_index or 0)
    if profiles:
        index %= len(profiles)
    else:
        index = 0

    active = profiles[index]
    cfg = dict(raw)
    cfg["models"] = profiles
    cfg["_model_defaults"] = _model_defaults(raw)
    cfg["active_model_index"] = index
    cfg["active_model"] = active.get("model") or active.get("name") or DEFAULT_MODEL
    cfg["model"] = active.get("model") or DEFAULT_MODEL

    for key in ("api_key", "api_base", "reasoning_effort", "max_tokens", "max_completion_tokens"):
        if key in active:
            cfg[key] = active[key]
        else:
            cfg.pop(key, None)
    if "extra_body" in active:
        cfg["extra_body"] = deepcopy(active["extra_body"])
    else:
        cfg.pop("extra_body", None)
    return cfg


def get_model_profiles(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = build_model_config(config)
    return [dict(profile) for profile in cfg.get("models") or []]


def inspect_config_file(path: str | Path | None = None) -> ConfigFileState:
    target = Path(path or CONFIG_PATH).expanduser()
    if not target.exists():
        return ConfigFileState(path=target, exists=False, valid=False, raw={})

    try:
        parsed = yaml.safe_load(target.read_text("utf-8"))
    except Exception as exc:
        return ConfigFileState(
            path=target,
            exists=True,
            valid=False,
            raw={},
            error=str(exc).strip() or exc.__class__.__name__,
        )

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return ConfigFileState(
            path=target,
            exists=True,
            valid=False,
            raw={},
            error=f"配置文件根节点必须是对象，当前是 {type(parsed).__name__}",
        )

    return ConfigFileState(path=target, exists=True, valid=True, raw=parsed)


def load_config() -> dict[str, Any]:
    state = inspect_config_file()
    cfg = build_model_config(state.raw if state.valid else {})
    cfg["_config_path"] = str(state.path)
    cfg["_config_exists"] = state.exists
    cfg["_config_valid"] = state.valid
    if state.error:
        cfg["_config_error"] = state.error
    return cfg


__all__ = [
    "BUILTIN_TOOL_PROVIDER_INDEX",
    "CONFIG_PATH",
    "ConfigFileState",
    "DEFAULT_MODEL",
    "DEFAULT_NAME",
    "DEFAULT_TOOL_SELECTIONS",
    "LOG_DIR",
    "ToolHandler",
    "build_model_config",
    "cfg_get",
    "get_model_profiles",
    "get_tool_provider_index",
    "get_tool_provider_name",
    "inspect_config_file",
    "load_config",
    "normalize_tool_provider_name",
    "resolve_tool_handlers",
]
