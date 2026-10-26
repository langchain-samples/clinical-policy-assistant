"""Seed the chart Q&A golden dataset into LangSmith: idempotent and scoped per attendee.

The committed `dataset_chart_qa.json` is the source of truth. This pushes it to a LangSmith dataset named
`scoped("chart-qa-golden")` (e.g. `chart-qa-golden-jane-doe`), so 30 attendees never share or clobber
a dataset. Re-running never duplicates. Pass `refresh=True` (or `--refresh`) after editing the JSON.

    uv run python evals/seed_chart_qa.py [--refresh]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.workshop import scoped  # noqa: E402

DATASET_BASENAME = "chart-qa-golden"
DATASET_PATH = Path(__file__).resolve().parent / "dataset_chart_qa.json"


def dataset_name() -> str:
    return scoped(DATASET_BASENAME)


def load_examples() -> list[dict]:
    """Committed examples as LangSmith example dicts (inputs / outputs / metadata / split)."""
    raw = json.loads(DATASET_PATH.read_text())["examples"]
    out = []
    for e in raw:
        meta = {"category": e["category"], "failure_mode": e.get("failure_mode")}
        out.append({
            "inputs": {"question": e["question"], "patient_id": e["patient_id"]},
            "outputs": {
                "reference_answer": e["reference_answer"],
                "reference_doc_ids": e["reference_doc_ids"],
                "category": e["category"],
                "failure_mode": e.get("failure_mode"),
                "expects_safety_event": e.get("expects_safety_event", False),
            },
            "metadata": meta,
            "split": e["category"],
        })
    return out


def seed_dataset(name: str | None = None, *, refresh: bool = False) -> str | None:
    """Create/refresh the scoped LangSmith dataset. Returns its name (None if no LangSmith key)."""
    name = name or dataset_name()
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set: skipping seed (the dataset JSON is committed locally).")
        return None
    from langsmith import Client

    client = Client()
    examples = load_examples()
    if refresh and client.has_dataset(dataset_name=name):
        client.delete_dataset(dataset_name=name)
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
        existing = sum(1 for _ in client.list_examples(dataset_id=ds.id))
        if existing:
            if existing != len(examples):
                print(f"⚠️  {name!r} has {existing} examples but the JSON has {len(examples)}; "
                      "run seed_dataset(refresh=True) to rebuild.")
            else:
                print(f"Dataset {name!r} already has {existing} examples: leaving as is.")
            return name
    else:
        ds = client.create_dataset(
            dataset_name=name,
            description="SYNTHETIC Northfield Health chart Q&A golden set (fact lookup, synthesis, failure modes, "
            "not-in-record, refusals).",
        )
    client.create_examples(dataset_id=ds.id, examples=examples)
    print(f"Seeded {len(examples)} examples into LangSmith dataset {name!r}.")
    return name


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    seed_dataset(refresh="--refresh" in sys.argv)
