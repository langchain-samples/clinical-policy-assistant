"""Invoke an agent and print a clickable LangSmith trace link, optionally masking identifiers first.

`ask(graph, question, patient_id=...)` returns the typed `ChartAnswer` (or the final text). With
`mask=True` the question goes through `utils.phi.mask_text` BEFORE the run starts. That means raw names,
MRNs and dates of birth never reach the model, the tools, or the trace inputs.
"""

from __future__ import annotations

import os
import re

from langchain_core.tracers.context import tracing_v2_enabled

# The SDK's run URL (/r/{run}) can 500 for some orgs; the project "peek" route renders reliably.
_RUN_URL_RE = re.compile(r"^(?P<base>https?://[^/]+/o/[^/]+/projects/p/(?P<proj>[0-9a-fA-F-]+))/r/(?P<run>[0-9a-fA-F-]+)")


def _peek_url(cb) -> str:
    raw = cb.get_run_url()
    m = _RUN_URL_RE.match(raw)
    if not m:
        return raw
    proj, run = m.group("proj"), m.group("run")
    return f"{m.group('base')}?peek={run}&peek_project={proj}&peeked_trace={run}"


def _tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"


def invoke_traced(graph, inputs, config: dict | None = None, context: dict | None = None) -> dict:
    """Invoke `graph` and return the FULL result dict (messages, __interrupt__, structured_response),
    printing a trace link when tracing is on."""
    kwargs = {"config": config}
    if context is not None:
        kwargs["context"] = context
    if _tracing_on():
        with tracing_v2_enabled(project_name=os.getenv("LANGSMITH_PROJECT")) as cb:
            result = graph.invoke(inputs, **kwargs)
        try:
            print("🔗 trace:", _peek_url(cb))
        except Exception:  # noqa: BLE001: a missing link must never break the lab
            pass
        return result
    return graph.invoke(inputs, **kwargs)


def ask(graph, question: str, *, patient_id: str | None = None, config: dict | None = None, mask: bool = False):
    """Ask the chart assistant one question. Returns `ChartAnswer` when the graph has a response_format."""
    if mask:
        from utils.phi import mask_text

        question, kinds = mask_text(question)
        if kinds:
            print(f"🛡️  masked before sending: {', '.join(kinds)}")
    payload = {"messages": [{"role": "user", "content": question}]}
    context = {"patient_id": patient_id} if patient_id else None
    result = invoke_traced(graph, payload, config=config, context=context)
    if result.get("structured_response") is not None:
        return result["structured_response"]
    return result["messages"][-1].content


def show(answer) -> None:
    """Pretty-print a ChartAnswer."""
    if isinstance(answer, str):
        print(answer)
        return
    print(f"disposition : {answer.disposition}")
    print(f"safety_event: {answer.safety_event}" + (f" ({answer.safety_event_summary})" if answer.safety_event else ""))
    print(f"\n{answer.answer}\n")
    for c in answer.citations:
        print(f"  • {c.doc_id}#{c.section}: “{c.quote}”")
