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


def trace_url(run_id) -> str | None:
    """Clickable LangSmith link for a run ID in the current project (None if it can't be resolved yet)."""
    from types import SimpleNamespace

    from langsmith import Client

    try:
        raw = Client().get_run_url(run=SimpleNamespace(id=run_id, trace_id=run_id), project_name=os.getenv("LANGSMITH_PROJECT"))
    except Exception:  # noqa: BLE001: a missing link must never break the lab
        return None
    m = _RUN_URL_RE.match(raw)
    return f"{m.group('base')}?peek={m.group('run')}&peek_project={m.group('proj')}&peeked_trace={m.group('run')}" if m else raw


def run_traced(fn, *args, **kwargs):
    """Call a `@traceable` function and print its trace link (Lab 1's pipeline is a function, not a graph)."""
    import uuid

    if not _tracing_on():
        return fn(*args, **kwargs)
    run_id = uuid.uuid4()
    result = fn(*args, langsmith_extra={"run_id": run_id}, **kwargs)
    url = trace_url(run_id)
    print("🔗 trace:", url or f"(open project {os.getenv('LANGSMITH_PROJECT')} in LangSmith)")
    return result


def print_steps(result: dict, width: int = 160) -> None:
    """Replay an agent run as its reasoning loop: model decision → tool call → tool result → … → answer."""
    step = 0
    for m in result.get("messages", []):
        kind = getattr(m, "type", "")
        if kind == "human":
            print(f"👤 {str(m.content)[:width]}")
        elif kind == "ai" and getattr(m, "tool_calls", None):
            step += 1
            for tc in m.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"🧠 step {step}: model calls {tc['name']}({args})")
        elif kind == "tool":
            print(f"   🔧 {m.name} → {' '.join(str(m.content).split())[:width]}…")
    answer = result.get("structured_response")
    final = answer.answer if answer is not None else str(result["messages"][-1].content)
    print(f"✅ answer after {step} tool step(s): {final[:width * 2]}")


def print_deep_steps(result: dict, width: int = 150) -> None:
    """Replay a deep-agent run: plans, delegations, file I/O, sandbox commands, and the final answer."""
    for m in result.get("messages", []):
        if getattr(m, "type", "") == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                a, name = tc["args"], tc["name"]
                if name == "write_todos":
                    items = " · ".join(f"{'✓' if t.get('status') == 'completed' else '○'} {t.get('content')}" for t in a.get("todos", []))
                    print(f"📝 plan: {items[:width * 2]}")
                elif name == "task":
                    print(f"🤝 delegate → {a.get('subagent_type')}: {str(a.get('description'))[:width]}")
                elif name in ("read_file", "write_file", "edit_file", "ls", "glob", "grep"):
                    print(f"📁 {name} {a.get('file_path') or a.get('path') or a.get('pattern') or ''}")
                elif name == "execute":
                    print(f"⚙️  execute: {a.get('command')}")
                else:
                    print(f"🔧 {name}({', '.join(f'{k}={v!r}' for k, v in a.items())[:width]})")
        elif getattr(m, "type", "") == "tool" and m.name in ("execute", "task"):
            print(f"   ↳ {m.name} returned: {' '.join(str(m.content).split())[:width]}…")
    answer = result.get("structured_response")
    final = answer.answer if answer is not None else str(result["messages"][-1].content)
    print(f"✅ answer: {final[:width * 3]}")


def show(answer) -> None:
    """Pretty-print a ChartAnswer (or its dict form, as stored in the Lab 3 workflow state)."""
    if isinstance(answer, str):
        print(answer)
        return
    if isinstance(answer, dict):
        from chart_qa_graph import ChartAnswer

        answer = ChartAnswer.model_validate(answer)
    print(f"disposition : {answer.disposition}")
    print(f"safety_event: {answer.safety_event}" + (f" ({answer.safety_event_summary})" if answer.safety_event else ""))
    print(f"\n{answer.answer}\n")
    for c in answer.citations:
        print(f"  • {c.doc_id}#{c.section}: “{c.quote}”")
