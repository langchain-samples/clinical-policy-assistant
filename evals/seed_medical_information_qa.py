"""Seed the medical-information QA eval dataset into LangSmith (idempotent, offline-safe).

The committed ``dataset_medical_information_qa.json`` is the source of truth (versioned, reviewable). This
helper pushes it to a LangSmith dataset so you can run experiments in the UI + via ``evaluate()``.

Idempotent: it creates the dataset if missing and only adds examples when the dataset is empty, so
re-running never duplicates. To refresh after editing the JSON, delete the dataset in the LangSmith UI
(or pass ``refresh=True``) and re-run. Offline-safe: no ``LANGSMITH_API_KEY`` → no-op with a note.

Run from repository root/content root:
    uv run python evals/seed_medical_information_qa.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATASET_NAME = "medical-information-qa-eval"
DATASET_PATH = Path(__file__).resolve().parent / "dataset_medical_information_qa.json"


def load_examples() -> list[dict]:
    """The committed examples as LangSmith example dicts (inputs / outputs / metadata)."""
    raw = json.loads(DATASET_PATH.read_text())
    return [
        {"inputs": e["inputs"], "outputs": e["outputs"], "metadata": e.get("metadata", {})}
        for e in raw
    ]


def seed_dataset(name: str = DATASET_NAME, *, refresh: bool = False) -> str | None:
    """Create/refresh the LangSmith dataset from the committed JSON. Returns the dataset name (or None offline)."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set — skipping seed (the dataset JSON is committed locally).")
        return None

    from langsmith import Client

    client = Client()

    if refresh and client.has_dataset(dataset_name=name):
        client.delete_dataset(dataset_name=name)

    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
        existing = sum(1 for _ in client.list_examples(dataset_id=ds.id))
        if existing:
            print(f"Dataset {name!r} already has {existing} examples — leaving as is (pass refresh=True to rebuild).")
            return name
    else:
        ds = client.create_dataset(
            dataset_name=name,
            description="Cited, safety-gated medical-information QA over synthetic prescribing-information fixtures.",
        )

    examples = load_examples()
    client.create_examples(dataset_id=ds.id, examples=examples)
    print(f"Seeded {len(examples)} examples into LangSmith dataset {name!r}.")
    return name


if __name__ == "__main__":
    seed_dataset()
