#!/usr/bin/env python
"""Build the attendee notebooks (`notebooks/`) and completed versions (`solutions/`) from one source.

Each notebook is defined in `scripts/notebook_src/nbXX_*.py` as a list `CELLS` of `md(...)` / `code(...)`
entries. A `code(solution, todo=...)` cell is written with `todo` (a `# TODO` exercise) to `notebooks/`
and with `solution` to `solutions/`. Edit the source, then run:

    uv run python scripts/build_notebooks.py [nb04_agent ...]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "notebook_src"
sys.path.insert(0, str(SRC))

KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.11"}}


def _cell(spec, solution: bool):
    kind, text, todo = spec
    if kind == "md":
        return new_markdown_cell(text.strip("\n"))
    src = text if (solution or todo is None) else todo
    return new_code_cell(src.strip("\n"))


def build(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    for folder, solution in (("notebooks", False), ("solutions", True)):
        nb = new_notebook(cells=[_cell(c, solution) for c in mod.CELLS], metadata=KERNEL)
        out = ROOT / folder / mod.FILENAME
        out.parent.mkdir(exist_ok=True)
        nbformat.write(nb, out)
        print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in SRC.glob("nb*.py"))
    for name in names:
        build(name)
