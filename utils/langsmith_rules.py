"""Guarded helpers for LangSmith online-eval run rules and annotation queues (notebook 05).

Creating rules or queues MUTATES the workspace and may need org enablement, so every helper only calls the
API when `ENABLE_ONLINE_EVALS=1` is set. Otherwise it returns `{"skipped": ...}` and touches nothing.
Callers pass names through `utils.workshop.scoped()` so 30 attendees never collide.
"""

from __future__ import annotations

import os
from typing import Any

_SKIP = {"skipped": "set ENABLE_ONLINE_EVALS=1 to create this live"}


def online_evals_enabled() -> bool:
    return os.environ.get("ENABLE_ONLINE_EVALS") == "1"


def ensure_annotation_queue(name: str, description: str = "") -> dict[str, Any]:
    """Create (or reuse) an annotation queue called `name`. Returns {"id", "name"} or the skip marker."""
    if not online_evals_enabled():
        return dict(_SKIP)
    from langsmith import Client

    client = Client()
    for q in client.list_annotation_queues(name=name):
        if q.name == name:
            return {"id": str(q.id), "name": q.name}
    q = client.create_annotation_queue(name=name, description=description)
    return {"id": str(q.id), "name": q.name}


def create_judge_run_rule(
    *,
    project_name: str,
    display_name: str,
    judge_prompt: str,
    filter_: str = "eq(is_root, true)",
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
    """Create an LLM-as-judge run rule that scores new matching traces in `project_name` (guarded)."""
    if not online_evals_enabled():
        return dict(_SKIP)
    from langsmith import Client

    client = Client()
    project = client.read_project(project_name=project_name)
    payload = {
        "display_name": display_name,
        "session_id": str(project.id),
        "is_enabled": True,
        "sampling_rate": sampling_rate,
        "filter": filter_,
        "evaluators": [{"structured": {"prompt": judge_prompt}}],
    }
    resp = client.request_with_retries("POST", "/runs/rules", data=payload)
    resp.raise_for_status()
    return resp.json()


def route_to_queue(
    *,
    project_name: str,
    display_name: str,
    annotation_queue_id: str,
    filter_: str = "eq(is_root, true)",
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
    """Create a run rule that adds matching runs to an annotation queue for human review (guarded)."""
    if not online_evals_enabled():
        return dict(_SKIP)
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
