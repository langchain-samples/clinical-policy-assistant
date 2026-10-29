#!/usr/bin/env python
"""Facilitator / CI helper: run the golden set through each milestone system and compare.

    uv run python scripts/run_experiments.py [rag agent workflow] [--reps N]
    uv run python scripts/run_experiments.py baseline improved-v1 improved-v2   # optional retrieval notebook

Each run is a scoped LangSmith experiment (e.g. `agent-jane-doe-1a2b`) on your scoped dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.setup import init_lab  # noqa: E402


def main() -> None:
    init_lab()
    from evals.harness import SYSTEMS, compare, print_scorecard, run_experiment, run_system, scorecard
    from utils.retrieval import CONFIGS

    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    names = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()] or list(SYSTEMS)
    unknown = [n for n in names if n not in SYSTEMS and n not in CONFIGS]
    if unknown:
        sys.exit(f"unknown system/config {unknown}; choose from {list(SYSTEMS) + list(CONFIGS)}")
    cards = {}
    for name in names:
        results = run_system(name, num_repetitions=reps) if name in SYSTEMS else run_experiment(CONFIGS[name], num_repetitions=reps)
        cards[name] = scorecard(results)
        print_scorecard(cards[name], f"{name} → {results.experiment_name}")
        for r in (r for r in cards[name]["rows"] if r["category"] == "failure_mode"):
            print(f"    {r['failure_mode']:22s} correctness={r.get('correctness')} tool_selection={r.get('tool_selection')}")
    print("\n=== comparison ===")
    compare(cards)
    out = ROOT / ".scratch" / "experiment_scores.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cards, indent=2, default=str))
    print(f"\nsaved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
