from __future__ import annotations

from importlib.util import find_spec


def cli_install_hint() -> str:
    return (
        "Install the default package with `pip install hyw` "
        "or, from a source checkout, run `uv run hyw`."
    )


def require_cli_support() -> None:
    missing = [name for name in ("rich", "prompt_toolkit") if find_spec(name) is None]
    if not missing:
        return
    names = ", ".join(missing)
    raise RuntimeError(
        f"CLI support is not installed ({names} missing). {cli_install_hint()}"
    )
