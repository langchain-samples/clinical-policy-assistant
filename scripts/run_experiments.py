#!/usr/bin/env python
"""Facilitator / CI helper: run baseline → improved-v1 → improved-v2 on the golden set and compare.

    uv run python scripts/run_experiments.py [baseline improved-v1 improved-v2] [--reps N]

Each run is a scoped LangSmith experiment (e.g. `baseline-jane-doe-1a2b`) on your scoped dataset.
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
    from evals.harness import compare, print_scorecard, run_experiment, scorecard
    from utils.retrieval import CONFIGS

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    args = [a for a in args if not a.isdigit()]
    names = args or list(CONFIGS)
    cards = {}
    for name in names:
        results = run_experiment(CONFIGS[name], num_repetitions=reps)
        cards[name] = scorecard(results)
        print_scorecard(cards[name], f"{name} → {results.experiment_name}")
        fm = [r for r in cards[name]["rows"] if r["category"] == "failure_mode"]
        for r in fm:
            print(f"    {r['failure_mode']:22s} correctness={r.get('correctness')} citation={r.get('citation_validity')}")
    print("\n=== comparison ===")
    compare(cards)
    out = ROOT / ".scratch" / "experiment_scores.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} | {"rows": v["rows"]} for k, v in cards.items()}, indent=2, default=str))
    print(f"\nsaved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
