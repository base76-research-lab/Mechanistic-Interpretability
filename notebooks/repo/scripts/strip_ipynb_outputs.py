#!/usr/bin/env python3
"""
strip_ipynb_outputs.py

Clears cell outputs + execution counts to keep notebooks reviewable in git.

Usage:
  python3 scripts/strip_ipynb_outputs.py path/to/notebook.ipynb [more.ipynb...]
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbformat
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: nbformat\n"
        "Install dev deps:\n"
        "  pip install -r requirements-dev.txt\n"
    ) from exc


def strip_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        nbformat.write(nb, path)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("Usage: strip_ipynb_outputs.py <notebook.ipynb> [more.ipynb...]\n")
        return 2

    for raw in argv[1:]:
        path = Path(raw)
        if not path.exists():
            sys.stderr.write(f"Not found: {path}\n")
            return 2
        if path.suffix.lower() != ".ipynb":
            sys.stderr.write(f"Not a notebook: {path}\n")
            return 2
        strip_notebook(path)
        print(f"Stripped: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
