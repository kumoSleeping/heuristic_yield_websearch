from __future__ import annotations

import os
import sys
from pathlib import Path


def _usage() -> str:
    return "Usage: entari run [config-file]"


def _default_config_path() -> Path:
    cwd = Path.cwd()
    local = cwd / "entari.yml"
    if local.exists():
        return local

    repo_dev = cwd / "dev.entari" / "entari.yml"
    if repo_dev.exists():
        return repo_dev

    return local


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0

    if argv[0] != "run":
        print(_usage(), file=sys.stderr)
        return 2

    config_path = Path(argv[1]).expanduser() if len(argv) > 1 else _default_config_path()
    os.environ.setdefault("ENTARI_CONFIG_FILE", str(config_path))

    from arclet.entari.core import Entari

    app = Entari.load(config_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
