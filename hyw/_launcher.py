from __future__ import annotations

from typing import Sequence

from core._optional import require_cli_support


def main(argv: Sequence[str] | None = None):
    require_cli_support()

    from core.cli import main as cli_main

    return cli_main(list(argv) if argv is not None else None)
