"""Cell constructors for notebook sources."""


def md(text: str):
    return ("md", text, None)


def code(solution: str, todo: str | None = None):
    """A code cell. If `todo` is given, attendees get `todo`; `solutions/` gets `solution`."""
    return ("code", solution, todo)


SETUP = '''import sys
from pathlib import Path

_root = Path.cwd()
while _root != _root.parent and not (_root / "utils").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root))

from utils.setup import init_lab

LAB = init_lab()  # loads .env, fails loudly if WORKSHOP_USER is unset, turns tracing on'''
