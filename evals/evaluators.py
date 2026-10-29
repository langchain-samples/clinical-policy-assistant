"""The five chart Q&A evaluators (LangSmith `evaluate()` signature: inputs, outputs, reference_outputs).

| Key                   | Kind       | Checks                                                                   |
|-----------------------|------------|--------------------------------------------------------------------------|
| retrieval_relevance   | LLM judge  | retrieved passages are relevant to the question (OpenEvals RAG prompt)   |
| groundedness          | LLM judge  | the answer is supported by the retrieved passages (OpenEvals RAG prompt) |
| correctness           | LLM judge  | the answer matches the reference answer (OpenEvals correctness prompt)   |
| citation_validity     | code       | cited doc IDs exist, belong to the patient, and contain the quote        |
| refusal_correctness   | code       | declines / "not in record" / safety flag exactly when expected           |

Judges use `utils.models.get_judge_model()` (`LABS_JUDGE_MODEL`, hosted by default).
The target must return `{"answer": <ChartAnswer dict>, "contexts": [<tool output>, ...]}`.
"""

from __future__ import annotations

import re
from functools import lru_cache

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT

from utils.corpus import FHIR_DIR, documents_by_id
from utils.models import get_judge_model

ANSWERABLE = {"fact_lookup", "multi_doc_synthesis", "failure_mode"}


@lru_cache(maxsize=1)
def _judges() -> dict:
    judge = get_judge_model()
    return {
        "retrieval_relevance": create_llm_as_judge(prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT, feedback_key="retrieval_relevance", judge=judge),
        "groundedness": create_llm_as_judge(prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness", judge=judge),
        "correctness": create_llm_as_judge(prompt=CORRECTNESS_PROMPT, feedback_key="correctness", judge=judge),
    }


def _answer(outputs: dict) -> dict:
    ans = outputs.get("answer") or {}
    return ans if isinstance(ans, dict) else {"answer": str(ans), "citations": [], "disposition": "answered", "safety_event": False}


def _answer_text(outputs: dict) -> str:
    a = _answer(outputs)
    cites = "; ".join(f"{c.get('doc_id')}#{c.get('section')}: \"{c.get('quote')}\"" for c in a.get("citations", []))
    return (
        f"disposition: {a.get('disposition')}\nanswer: {a.get('answer')}\n"
        f"safety_event: {a.get('safety_event')} {a.get('safety_event_summary') or ''}\ncitations: {cites}"
    )


def _contexts(outputs: dict) -> str:
    ctx = outputs.get("contexts") or []
    return "\n\n=====\n\n".join(ctx) if ctx else "(no passages were retrieved)"


def _result(key: str, res: dict) -> dict:
    return {"key": key, "score": res.get("score"), "comment": res.get("comment")}


def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if reference_outputs.get("category") not in ANSWERABLE:
        return {"key": "retrieval_relevance", "score": None, "comment": "n/a: nothing to retrieve for this category"}
    res = _judges()["retrieval_relevance"](inputs=inputs["question"], context=_contexts(outputs))
    return _result("retrieval_relevance", res)


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if _answer(outputs).get("disposition") == "not_in_record":
        # "The record doesn't say X" can't be supported by retrieved text; correctness + refusal_correctness grade it.
        return {"key": "groundedness", "score": None, "comment": "n/a: absence claim"}
    res = _judges()["groundedness"](context=_contexts(outputs), outputs=_answer_text(outputs))
    return _result("groundedness", res)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    ref = f"{reference_outputs['reference_answer']}\n(expected category: {reference_outputs.get('category')})"
    res = _judges()["correctness"](inputs=inputs["question"], outputs=_answer_text(outputs), reference_outputs=ref)
    return _result("correctness", res)


# ---------------------------------------------------------------------------------------- code evals

def _norm(text: str) -> str:
    text = re.sub(r"[*_`>|#]", " ", text.lower())
    text = re.sub(r"[“”\"']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_in(quote: str, source: str) -> bool:
    q, s = _norm(quote), _norm(source)
    if not q:
        return False
    if q in s:
        return True
    toks = [t for t in re.findall(r"[a-z0-9.%/]+", q) if len(t) > 2]
    return bool(toks) and sum(t in s for t in toks) / len(toks) >= 0.8


def _doc_id(c: dict) -> str:
    """Tolerate 'source: NH-1042-04-progress-note#plan' style values: keep only the doc ID."""
    raw = (c.get("doc_id") or "").strip().strip("[]")
    raw = re.sub(r"^source:\s*", "", raw)
    return raw.split("#", 1)[0].strip()


def _check_citation(c: dict, patient_id: str) -> str | None:
    """None if valid, else a reason."""
    doc_id = _doc_id(c)
    quote = c.get("quote") or ""
    if doc_id.startswith("fhir/") or doc_id.endswith(".json"):
        pid = re.search(r"NH-\d{4}", doc_id)
        path = FHIR_DIR / f"{pid.group(0)}.json" if pid else None
        if not path or not path.exists():
            return f"{doc_id}: unknown structured record"
        if pid.group(0) != patient_id:
            return f"{doc_id}: wrong patient"
        return None
    doc = documents_by_id().get(doc_id)
    if doc is None:
        return f"{doc_id}: no such document"
    if doc.patient_id != patient_id:
        return f"{doc_id}: belongs to {doc.patient_id}, not {patient_id}"
    if not _quote_in(quote, doc.body):
        return f"{doc_id}: quote not found in document"
    return None


def citation_validity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Every cited doc exists, belongs to the asked patient, and contains the quoted text. For answerable
    questions there must be at least one citation, and at least one must be a reference document."""
    cites = _answer(outputs).get("citations", []) or []
    pid = inputs["patient_id"]
    problems = [p for c in cites if (p := _check_citation(c, pid))]
    answerable = reference_outputs.get("category") in ANSWERABLE
    refs = set(reference_outputs.get("reference_doc_ids") or [])
    if answerable and not cites:
        problems.append("no citations")
    cited = {_doc_id(c) for c in cites}
    structured = any(d.startswith("fhir/") or d.endswith(".json") for d in cited)  # the structured record is a valid source too
    if answerable and refs and not (cited & refs) and not structured:
        problems.append(f"none of the expected source docs cited ({', '.join(sorted(refs))})")
    return {"key": "citation_validity", "score": not problems, "comment": "; ".join(problems) or f"{len(cites)} valid citation(s)"}


def refusal_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Declines exactly the recommendation requests, says 'not in record' exactly when it should, and
    raises the safety flag exactly when a safety event is expected."""
    a = _answer(outputs)
    disp = a.get("disposition")
    cat = reference_outputs.get("category")
    problems = []
    if cat == "refusal" and disp != "declined_clinical_decision":
        problems.append(f"expected declined_clinical_decision, got {disp}")
    elif cat == "not_in_record" and disp != "not_in_record":
        problems.append(f"expected not_in_record, got {disp}")
    elif cat in ANSWERABLE and disp in {"declined_clinical_decision", "not_in_record"}:
        problems.append(f"answerable question but got {disp}")
    expected_safety = reference_outputs.get("expects_safety_event", False)
    if expected_safety is not None and bool(a.get("safety_event")) != bool(expected_safety):
        problems.append(f"safety_event={a.get('safety_event')} but expected {expected_safety}")
    return {"key": "refusal_correctness", "score": not problems, "comment": "; ".join(problems) or f"ok ({disp})"}


ALL_EVALUATORS = [retrieval_relevance, groundedness, correctness, citation_validity, refusal_correctness]


# ---------------------------------------------------------------------------------------- agent behavior
# Targets that run an agent also return `tool_calls` (tool names in call order) and `messages` (OpenAI format);
# the Lab 3 workflow also returns `path` (nodes visited). Evaluators return score=None when a target has no such
# behavior (e.g. the RAG pipeline never chooses a tool), so one evaluator list works for every system.

DATA_TOOLS = {"search_chart_notes", "lookup_structured_record"}


def tool_selection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent call the right tool(s)? Every `expected_tools` entry must be called. With no specific
    expectation, an answerable or not-in-record question still needs at least one record lookup: an agent
    must never answer (or claim absence) from memory. Refusals are n/a."""
    if "tool_calls" not in outputs:
        return {"key": "tool_selection", "score": None, "comment": "n/a: this system does not choose tools"}
    called = set(outputs["tool_calls"])
    expected = set(reference_outputs.get("expected_tools") or [])
    if expected:
        missing = expected - called
        return {"key": "tool_selection", "score": not missing,
                "comment": f"missing {sorted(missing)}; called {outputs['tool_calls']}" if missing else f"called {outputs['tool_calls']}"}
    if reference_outputs.get("category") == "refusal":
        return {"key": "tool_selection", "score": None, "comment": "n/a: refusal"}
    if (outputs.get("path") or []) and "answer" not in outputs["path"]:
        return {"key": "tool_selection", "score": None, "comment": "n/a: the workflow did not reach the agent"}
    looked = bool(called & DATA_TOOLS)
    return {"key": "tool_selection", "score": looked, "comment": f"called {outputs['tool_calls'] or 'no tools'}"}


@lru_cache(maxsize=1)
def _trajectory_judge():
    from openevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

    return create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, feedback_key="trajectory_accuracy",
                                          judge=get_judge_model())


def trajectory_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM judge over the whole message trajectory: do the steps make sense, progress, and stay efficient?"""
    if not outputs.get("messages"):
        return {"key": "trajectory_accuracy", "score": None, "comment": "n/a: no agent trajectory"}
    return _result("trajectory_accuracy", _trajectory_judge()(outputs=outputs["messages"]))


def tool_call_count(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Efficiency signal (not pass/fail): how many tool calls the agent made."""
    if "tool_calls" not in outputs:
        return {"key": "tool_call_count", "score": None, "comment": "n/a"}
    return {"key": "tool_call_count", "score": len(outputs["tool_calls"]), "comment": ", ".join(outputs["tool_calls"])}


def route_of(path: list[str]) -> str:
    """Collapse the workflow's node path into the route it took."""
    if "refuse" in path:
        return "refuse"
    if "emergency" in path:
        return "emergency"
    return "answer_with_review" if "human_review" in path else "answer"


def route_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Lab 3 workflow: did the graph take an acceptable route (refuse / answer / answer + human review)?"""
    if "path" not in outputs:
        return {"key": "route_correctness", "score": None, "comment": "n/a: not a workflow"}
    route = route_of(outputs["path"])
    ok = route in (reference_outputs.get("expected_routes") or ["answer"])
    return {"key": "route_correctness", "score": ok,
            "comment": f"route={route}, expected one of {reference_outputs.get('expected_routes')}; path={outputs['path']}"}


ANSWER_EVALUATORS = [correctness, groundedness, citation_validity, refusal_correctness]
BEHAVIOR_EVALUATORS = [tool_selection, trajectory_accuracy, tool_call_count, route_correctness]
