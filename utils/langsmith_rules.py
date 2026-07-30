"""Offline-safe helper for LangSmith online-eval **run rules** (automations), used by the `05_observe` lab.

A run rule scores every new trace that matches a filter with an LLM-as-judge (or routes it to an
annotation queue). Creating one MUTATES the workspace and requires org enablement, so this helper is
**guarded**: it only calls the API when `ENABLE_ONLINE_EVALS=1` is set in the environment. During the
headless notebook build (`build_lab.py` / nbconvert) that variable is unset, so the call is a no-op and
returns a `{"skipped": ...}` dict instead of touching LangSmith. Set it explicitly to run the cell live.

Secrets are referenced by name only — the LangSmith client reads `LANGSMITH_API_KEY` from the env itself.
"""

from __future__ import annotations

import os
from typing import Any


def online_evals_enabled() -> bool:
    """True only when the operator has explicitly opted in. Unset during headless builds → cells no-op."""
    return os.environ.get("ENABLE_ONLINE_EVALS") == "1"


def create_judge_run_rule(
    *,
    project_name: str,
    display_name: str,
    judge_prompt: str,
    filter_: str = "eq(is_root, true)",
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
    """Create an LLM-as-judge run rule that scores matching new traces in `project_name`.

    Guarded: returns `{"skipped": "set ENABLE_ONLINE_EVALS=1 to create the rule live"}` unless opted in,
    so it is safe to execute in a headless build and never mutates the workspace by accident.

    `filter_` is a LangSmith trace-filter expression (default: root runs only). `sampling_rate` is the
    fraction of matching runs scored. Returns the created rule payload (or the skip marker).
    """
    if not online_evals_enabled():
        return {"skipped": "set ENABLE_ONLINE_EVALS=1 to create the rule live"}

    from langsmith import Client

    client = Client()  # reads LANGSMITH_API_KEY / endpoint from the environment
    project = client.read_project(project_name=project_name)
    payload = {
        "display_name": display_name,
        "session_id": str(project.id),
        "is_enabled": True,
        "sampling_rate": sampling_rate,
        "filter": filter_,
        "evaluators": [{"structured": {"prompt": judge_prompt}}],
    }
    # The run-rules (automations) endpoint isn't in the typed SDK surface across versions; post directly.
    resp = client.request_with_retries("POST", "/runs/rules", data=payload)
    resp.raise_for_status()
    return resp.json()


def route_low_scores_to_queue(
    *,
    project_name: str,
    display_name: str,
    annotation_queue_id: str,
    filter_: str = "eq(is_root, true)",
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
    """Create a run rule that adds matching runs to an annotation queue for human review.

    Same guard as `create_judge_run_rule`. Pair this with a score filter (e.g. runs whose judge score is
    low) to build the online-eval → review → next-dataset loop.
    """
    if not online_evals_enabled():
        return {"skipped": "set ENABLE_ONLINE_EVALS=1 to create the rule live"}

    from langsmith import Client

    client = Client()
    project = client.read_project(project_name=project_name)
    payload = {
        "display_name": display_name,
        "session_id": str(project.id),
        "is_enabled": True,
        "sampling_rate": sampling_rate,
        "filter": filter_,
        "add_to_annotation_queue_id": annotation_queue_id,
    }
    resp = client.request_with_retries("POST", "/runs/rules", data=payload)
    resp.raise_for_status()
    return resp.json()
