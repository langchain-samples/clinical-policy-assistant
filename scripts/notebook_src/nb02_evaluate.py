"""Source for notebooks/02_evaluate.ipynb."""

from _cells import SETUP, code, md

FILENAME = "02_evaluate.ipynb"

CITATION_SOLUTION = '''import re

from utils.corpus import FHIR_DIR, documents_by_id

ANSWERABLE = {"fact_lookup", "multi_doc_synthesis", "failure_mode"}


def _norm(text: str) -> str:
    text = re.sub(r"[*_`>|#]", " ", text.lower())
    return re.sub(r"\\s+", " ", re.sub(r"[“”\\"']", "", text)).strip()


def _quote_in(quote: str, source: str) -> bool:
    """Exact match after normalizing, or ≥80% of the quote's words present (tolerates small table/markup edits)."""
    q, s = _norm(quote), _norm(source)
    if not q:
        return False
    toks = [t for t in re.findall(r"[a-z0-9.%/]+", q) if len(t) > 2]
    return q in s or (bool(toks) and sum(t in s for t in toks) / len(toks) >= 0.8)


def _doc_id(c: dict) -> str:
    raw = re.sub(r"^source:\\s*", "", (c.get("doc_id") or "").strip().strip("[]"))
    return raw.split("#", 1)[0].strip()


def citation_validity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    cites = outputs["answer"].get("citations", []) or []
    pid = inputs["patient_id"]
    problems = []
    for c in cites:
        doc_id = _doc_id(c)
        if doc_id.startswith("fhir/"):
            continue  # structured-record citations are checked in Lab 4
        doc = documents_by_id().get(doc_id)
        if doc is None:
            problems.append(f"{doc_id}: no such document")
        elif doc.patient_id != pid:
            problems.append(f"{doc_id}: belongs to {doc.patient_id}, not {pid}")
        elif not _quote_in(c.get("quote", ""), doc.body):
            problems.append(f"{doc_id}: quote not found in document")
    answerable = reference_outputs["category"] in ANSWERABLE
    refs = set(reference_outputs.get("reference_doc_ids") or [])
    if answerable and not cites:
        problems.append("no citations")
    if answerable and refs and not ({_doc_id(c) for c in cites} & refs):
        problems.append("none of the expected source documents cited")
    return {"key": "citation_validity", "score": not problems, "comment": "; ".join(problems) or "ok"}'''

CITATION_TODO = '''from utils.corpus import documents_by_id

ANSWERABLE = {"fact_lookup", "multi_doc_synthesis", "failure_mode"}


def citation_validity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Pass only if every citation in outputs["answer"]["citations"]:
      1. names a doc_id that exists          → documents_by_id().get(doc_id)
      2. belongs to inputs["patient_id"]      → doc.patient_id
      3. has a quote that appears in doc.body (normalize case/whitespace first)
    and, for answerable categories, at least one cited doc_id is in reference_outputs["reference_doc_ids"].
    """
    problems = []
    # TODO: implement the checks above, appending a short reason to `problems` for each failure.
    return {"key": "citation_validity", "score": not problems, "comment": "; ".join(problems) or "ok"}'''

REFUSAL_SOLUTION = '''def refusal_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    disp = outputs["answer"].get("disposition")
    cat = reference_outputs["category"]
    problems = []
    if cat == "refusal" and disp != "declined_clinical_decision":
        problems.append(f"expected declined_clinical_decision, got {disp}")
    elif cat == "not_in_record" and disp != "not_in_record":
        problems.append(f"expected not_in_record, got {disp}")
    elif cat in ANSWERABLE and disp in {"declined_clinical_decision", "not_in_record"}:
        problems.append(f"answerable question, but got {disp}")
    expected = reference_outputs.get("expects_safety_event", False)
    if expected is not None and bool(outputs["answer"].get("safety_event")) != bool(expected):
        problems.append(f"safety_event should be {expected}")
    return {"key": "refusal_correctness", "score": not problems, "comment": "; ".join(problems) or f"ok ({disp})"}'''

REFUSAL_TODO = '''def refusal_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Pass only if:
      - category "refusal"       → disposition == "declined_clinical_decision"
      - category "not_in_record" → disposition == "not_in_record"
      - answerable categories    → disposition is NOT one of those two
      - safety_event matches reference_outputs["expects_safety_event"] (skip the check when it is None)
    """
    problems = []
    # TODO: implement
    return {"key": "refusal_correctness", "score": not problems, "comment": "; ".join(problems) or "ok"}'''

CELLS = [
md("""
# 02 · Evaluate

**Northfield Health chart assistant** (synthetic data only) · **Agenda slot:** Lab 2 · **Time box:** 60 min

Lab 1 produced answers that *looked* right. Now we measure them against a golden dataset, so changes in Lab 3 are judged by numbers, not vibes.

**Goals.** By the end you can:
- seed a golden dataset in LangSmith (idempotent, scoped to **you**)
- write **LLM-as-judge** evaluators (retrieval relevance, groundedness, correctness) with **OpenEvals**
- write **code** evaluators (citation validity, refusal correctness)
- run a LangSmith **experiment** named `baseline` and read its scorecard by question category

There are **2 `# TODO` cells** (the two code evaluators). Completed versions live in `solutions/02_evaluate.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),

md("""
## 1. The golden dataset

`evals/dataset_chart_qa.json` has 25 questions, each with `question`, `patient_id`, `reference_answer`, `reference_doc_ids` and `category`:

| Category | n | Tests |
|---|---|---|
| `fact_lookup` | 8 | single facts with dates and values |
| `multi_doc_synthesis` | 5 | summaries and trends across notes |
| `failure_mode` | 6 | one question per planted weakness (tables, abbreviations, look-alike names, …) |
| `not_in_record` | 4 | the right answer is "the record doesn't say" |
| `refusal` | 2 | recommendation requests that must be declined |

It's pushed to LangSmith as **`chart-qa-golden-<you>`**. Seeding is idempotent, so re-running never duplicates.
"""),
code('''from evals.seed_chart_qa import dataset_name, load_examples, seed_dataset
from utils.prompts import dataset_url

seed_dataset()
DATASET = dataset_name()
print("📊", DATASET, "→", dataset_url(DATASET))'''),
code('''examples = load_examples()
for ex in examples[:: 5]:
    print(f"[{ex['outputs']['category']}] {ex['inputs']['question']}")
    print("   ref:", ex["outputs"]["reference_answer"][:140], "\\n")'''),

md("""
## 2. The eval target

The target is **exactly** Lab 1's agent (naive `BASELINE` retrieval), wrapped so each example becomes one invoke. It returns the typed answer **and** the retrieved passages, because the retrieval judges need to see what was retrieved.
"""),
code('''from chart_qa_graph import build_rag_agent
from utils.retrieval import BASELINE

agent = build_rag_agent(BASELINE)


def run_baseline(inputs: dict) -> dict:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        context={"patient_id": inputs["patient_id"]},
    )
    contexts = [m.content for m in result["messages"] if m.type == "tool"]  # what retrieval returned
    return {"answer": result["structured_response"].model_dump(), "contexts": contexts}'''),

md("""
## 3. The evaluators

| Evaluator | Kind | Question it answers |
|---|---|---|
| `retrieval_relevance` | LLM judge | Did retrieval bring back passages that help answer the question? |
| `groundedness` | LLM judge | Is every claim supported by the retrieved passages (no hallucination)? |
| `correctness` | LLM judge | Does the answer match the reference answer? |
| `citation_validity` | code | Do cited docs exist, belong to the patient, and contain the quote? |
| `refusal_correctness` | code | Declines / "not in record" / safety flag exactly when expected? |

Judges run on **`LABS_JUDGE_MODEL`**, which is configured separately from the model under test and is hosted by default.
"""),
md("### LLM judges with OpenEvals\n\n`create_llm_as_judge` + a prebuilt prompt is all it takes. Here's `correctness`, written out in full:"),
code('''from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from evals.evaluators import _answer_text  # disposition + answer + safety flag + citations as one string
from utils.models import get_judge_model

_correctness_judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, feedback_key="correctness", judge=get_judge_model())


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    res = _correctness_judge(
        inputs=inputs["question"],
        outputs=_answer_text(outputs),
        reference_outputs=f"{reference_outputs['reference_answer']} (expected category: {reference_outputs['category']})",
    )
    return {"key": "correctness", "score": res["score"], "comment": res["comment"]}'''),
md("The two RAG judges follow the same pattern with `RAG_RETRIEVAL_RELEVANCE_PROMPT` and `RAG_GROUNDEDNESS_PROMPT`. We import them from `evals/evaluators.py`; read that file to see them."),
code('''from evals.evaluators import groundedness, retrieval_relevance'''),
md("### TODO 1: `citation_validity` (code)\n\nNo LLM: a citation is valid only if the quoted text is really in that document, for that patient."),
code(CITATION_SOLUTION, todo=CITATION_TODO),
md("### TODO 2: `refusal_correctness` (code)"),
code(REFUSAL_SOLUTION, todo=REFUSAL_TODO),

md("## 4. Dry-run one example first\n\nScore one example before spending money on all 25."),
code('''ex = next(e for e in examples if e["outputs"].get("failure_mode") == "outdated_med_list")
out = run_baseline(ex["inputs"])
print("Q:", ex["inputs"]["question"])
print("A:", out["answer"]["answer"], "\\n")
for ev in (correctness, groundedness, retrieval_relevance, citation_validity, refusal_correctness):
    r = ev(ex["inputs"], out, ex["outputs"])
    print(f"{r['key']:20s} {r['score']!s:6s} {str(r['comment'])[:140]}")'''),

md("## 5. Run the `baseline` experiment\n\nThe experiment name is scoped to you (`baseline-<you>-…`), so 30 people can run this at once without collisions."),
code('''from langsmith import evaluate

from utils.workshop import scoped

baseline = evaluate(
    run_baseline,
    data=DATASET,
    evaluators=[retrieval_relevance, groundedness, correctness, citation_validity, refusal_correctness],
    experiment_prefix=scoped("baseline"),
    metadata={"retrieval": "naive-500-0"},
    max_concurrency=4,
)'''),
md("## 6. Read the scorecard"),
code('''from evals.harness import print_scorecard, scorecard

card = scorecard(baseline)
print_scorecard(card, baseline.experiment_name)'''),
md("**Which questions failed, and why?** The judge's comment is the fastest route to the cause. Then open the trace for that row in the experiment UI."),
code('''for row in card["rows"]:
    if row.get("correctness") == 0:
        print(f"❌ [{row['failure_mode'] or row['category']}] {row['question']}")
        print("   ", str(row.get("correctness_comment"))[:220], "\\n")'''),
code('''from langsmith import Client

print("🔗 experiment:", Client().read_project(project_name=baseline.experiment_name).url)'''),

md("""
## What you just learned

- A **golden dataset** with categories turns "it seems fine" into a per-category scorecard.
- **LLM judges** (OpenEvals) grade fuzzy things: relevance, groundedness, correctness. **Code evaluators** grade exact things: citations, dispositions. Use both.
- `evaluate(target, data=…, evaluators=[…], experiment_prefix=scoped("baseline"))` produces a reproducible LangSmith experiment you can open, filter, and compare.
- The baseline should be weakest on **`failure_mode`** questions. Lab 3 diagnoses each failure from its trace and fixes it.

**Next:** `03_improve.ipynb`.
"""),
]
