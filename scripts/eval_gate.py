#!/usr/bin/env python
"""CI regression gate: run the golden set through a system and fail (exit 1) if any metric drops below
`evals/thresholds.json`. Run it on every change to prompts, tools, retrieval, models or the graph.

    uv run python scripts/eval_gate.py [workflow agent rag] [--reps N]

Each run is a normal scoped LangSmith experiment, so a failing gate links straight to the traces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.setup import init_lab  # noqa: E402


def main() -> int:
    init_lab()
    from evals.harness import SYSTEMS, gate, run_system, scorecard

    thresholds = {k: v for k, v in json.loads((ROOT / "evals" / "thresholds.json").read_text()).items() if not k.startswith("_")}
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    systems = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()] or ["workflow"]
    failed = False
    for system in systems:
        if system not in SYSTEMS:
            sys.exit(f"unknown system {system!r}; choose from {list(SYSTEMS)}")
        results = run_system(system, num_repetitions=reps)
        problems = gate(scorecard(results), thresholds.get(system, {}))
        status = "PASS" if not problems else "FAIL"
        print(f"{'✅' if not problems else '⛔'} {status} {system} ({results.experiment_name})" + "".join(f"\n     {p}" for p in problems))
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
