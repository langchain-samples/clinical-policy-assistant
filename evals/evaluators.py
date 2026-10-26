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
    if answerable and refs and not ({_doc_id(c) for c in cites} & refs):
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
