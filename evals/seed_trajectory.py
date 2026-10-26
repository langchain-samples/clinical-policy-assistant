"""Small trajectory dataset for notebook 04: which tools should the agent call, and must it file a safety event?

Seeded idempotently as `scoped("chart-qa-trajectory")`. SYNTHETIC teaching data.
"""

from __future__ import annotations

import os

from utils.workshop import scoped

DATASET_BASENAME = "chart-qa-trajectory"

EXAMPLES = [
    {"question": "What anticoagulant is NH-1042 currently on, and since when?", "patient_id": "NH-1042",
     "expected_tools": ["lookup_structured_record"], "expects_filing": False},
    {"question": "What is the most recent creatinine on file for NH-1017, with its date?", "patient_id": "NH-1017",
     "expected_tools": ["lookup_structured_record"], "expects_filing": False},
    {"question": "Did NH-1023 have breathing trouble overnight during the September 2025 admission? Check the nursing notes.",
     "patient_id": "NH-1023", "expected_tools": ["search_chart_notes"], "expects_filing": False},
    {"question": "Were there any medication administration problems during NH-1004's May 2025 stay? If so, file it for patient-safety review.",
     "patient_id": "NH-1004", "expected_tools": ["search_chart_notes", "file_safety_event"], "expects_filing": True},
    {"question": "Should I increase NH-1042's Thrombelix dose?", "patient_id": "NH-1042",
     "expected_tools": [], "expects_filing": False},
]


def dataset_name() -> str:
    return scoped(DATASET_BASENAME)


def reference_trajectory(ex: dict) -> list[dict]:
    """OpenAI-format reference: the user turn plus one assistant turn calling each expected tool."""
    msgs = [{"role": "user", "content": ex["question"]}]
    if ex["expected_tools"]:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": name, "arguments": "{}"}} for name in ex["expected_tools"]
        ]})
    return msgs


def seed_trajectory_dataset() -> str | None:
    name = dataset_name()
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set: skipping seed.")
        return None
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
        if any(True for _ in client.list_examples(dataset_id=ds.id, limit=1)):
            print(f"Dataset {name!r} already seeded.")
            return name
    else:
        ds = client.create_dataset(dataset_name=name, description="SYNTHETIC chart Q&A agent trajectories.")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": {"question": e["question"], "patient_id": e["patient_id"]},
         "outputs": {"trajectory": reference_trajectory(e), "expected_tools": e["expected_tools"],
                     "expects_filing": e["expects_filing"]}}
        for e in EXAMPLES
    ])
    print(f"Seeded {len(EXAMPLES)} trajectory examples into {name!r}.")
    return name
