"""Source for notebooks/optional/improve_retrieval.ipynb (optional / Day 2)."""

from _cells import SETUP, code, md

FILENAME = "optional/improve_retrieval.ipynb"

CELLS = [
md("""
# Optional · Improve retrieval: from eval failures to fixes

**Northfield Health chart assistant** (synthetic data only) · **Optional / Day 2** · **Time box:** 45 min

The day's labs used a pre-tuned retrieval pipeline (`RAG_CONFIG`). This notebook shows **how it got that way**: start from naive defaults, score them, and work the loop that matters in production:

```text
low score → open the trace → name the root cause → change ONE thing → re-run the eval → compare
```

**Goals.** By the end you can:
- pull the lowest-scoring runs from an experiment and diagnose them from their traces
- apply four retrieval fixes: **structure-aware chunking**, **date-aware ranking**, **query rewriting** for clinical abbreviations, and **chunk size/overlap tuning**
- run `improved-v1` and `improved-v2` and **compare experiments side by side**

There are **2 `# TODO` cells** (the two improved configs). Completed versions live in `solutions/optional/improve_retrieval.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),
code('''from langsmith import Client

from evals.harness import compare, print_scorecard, run_experiment, scorecard
from evals.seed_chart_qa import dataset_name, seed_dataset
from utils.retrieval import BASELINE, RetrievalConfig, build_vectorstore, chunk_corpus, format_docs, retrieve
from utils.workshop import scoped

client = Client()
seed_dataset()'''),

md("""
## 1. Find your `baseline` experiment and its worst runs

This experiment uses a create_agent-based RAG app with **naive** retrieval (500-character chunks, no overlap, no date awareness), so the fixes have headroom. If you haven't run it yet, this cell runs it for you.
"""),
code('''def latest_experiment(prefix: str):
    ds = client.read_dataset(dataset_name=dataset_name())
    exps = [p for p in client.list_projects(reference_dataset_id=ds.id) if (p.name or "").startswith(prefix)]
    return max(exps, key=lambda p: p.start_time) if exps else None


base_exp = latest_experiment(scoped("baseline"))
if base_exp is None:
    print("No baseline experiment found: running it now (≈2 min)…")
    base_exp = client.read_project(project_name=run_experiment(BASELINE).experiment_name)
print("baseline experiment:", base_exp.name)'''),
code('''runs = list(client.list_runs(project_id=base_exp.id, is_root=True))
feedback = {}
for fb in client.list_feedback(run_ids=[r.id for r in runs]):
    feedback.setdefault(fb.run_id, {})[fb.key] = (fb.score, fb.comment)

failing = [r for r in runs if (feedback.get(r.id, {}).get("correctness", (1,))[0] or 0) < 1]
print(f"{len(failing)} of {len(runs)} runs scored correctness = 0\\n")
for r in failing:
    score, comment = feedback[r.id]["correctness"]
    print("❌", r.inputs.get("question"))
    print("   judge:", (comment or "")[:200])
    print("   trace:", client.get_run_url(run=r, project_id=base_exp.id), "\\n")'''),
md("""
> 🔎 **Open 2–3 of those traces.** For each, find the `search_chart_notes` tool span and ask:
> 1. Did retrieval return the passage that contains the answer?
> 2. If yes, did the model use it, or prefer an older or competing passage?
> 3. Was the right patient searched?
>
> Most failures fall into a handful of root causes. Each fix below targets one.
"""),

md("""
## 2. Fix 1: structure-aware chunking (keep tables whole)

**Symptom.** *"What was NH-1017's potassium on June 11?"* returns "not found", but the value is in the discharge summary's lab table. **Cause:** 500-character chunks split the table, so the chunk holding the potassium row has no header row: no dates, no "Laboratory Results" title.
"""),
code('''naive = [c for c in chunk_corpus(BASELINE) if c.metadata["doc_id"] == "NH-1017-06-discharge-summary"]
print("NAIVE: the chunk holding the potassium row:\\n")
print(next(c.page_content for c in naive if "Potassium" in c.page_content))'''),
md("`structure_chunks` splits on `##` sections, keeps every Markdown table **atomic**, and prefixes each chunk with `[title | doc type | date | section]`."),
code('''structured = chunk_corpus(RetrievalConfig(chunker="structure", chunk_size=800))
table = next(c for c in structured if c.metadata["doc_id"] == "NH-1017-06-discharge-summary" and "Potassium" in c.page_content)
print("STRUCTURE-AWARE:\\n")
print(table.page_content[:700])'''),

md("""
## 3. Fix 2: date-aware ranking (and pin the patient)

**Symptom.** *"What anticoagulants is NH-1042 on?"* reports **Coagulor**, from a January medication list, but a March progress note switched him to **Thrombelix**. **Cause:** chunks carry no dates, so the model can't tell which source is newer.

**Fix.** (a) Show each source's date and return results newest first. (b) Add a prompt rule: *the newest dated document is the current truth*.

**Also.** *"What Pulmoquin dose is Marian Hale taking?"* has no ID in the question, so the model guesses, sometimes picking Marion Hale's chart. In the app, the clinician has one chart open. We **bind** retrieval to that session's patient ID (`bind_patient=True`) instead of trusting the model's guess.
"""),
code('''q = "What anticoagulants is NH-1042 on, and when was each started?"
print("BASELINE context:\\n", format_docs(retrieve(q, "NH-1042", BASELINE), BASELINE)[:700], "\\n")
dated = RetrievalConfig(chunker="structure", chunk_size=800, date_aware=True)
print("DATE-AWARE context:\\n", format_docs(retrieve(q, "NH-1042", dated), dated)[:700])'''),
code('''from utils.retrieval import DATE_AWARE_RULES

print(DATE_AWARE_RULES)'''),

md("### Run `improved-v1` = Fix 1 + Fix 2\n\n**TODO 1.** Build the config: structure chunking (800 chars), `k=5`, date-aware ranking, and the session-bound patient."),
code('''improved_v1 = RetrievalConfig(
    name="improved-v1", chunker="structure", chunk_size=800, chunk_overlap=0, k=5,
    date_aware=True, bind_patient=True,
)
v1 = run_experiment(improved_v1)''',
todo='''# TODO 1: fill in the fixes (see utils/retrieval.py → RetrievalConfig for field names)
improved_v1 = RetrievalConfig(
    name="improved-v1",
    # chunker=..., chunk_size=..., k=..., date_aware=..., bind_patient=...
)
v1 = run_experiment(improved_v1)'''),
code('''print_scorecard(scorecard(v1), v1.experiment_name)'''),
md("## 4. Fix 3: query rewriting for clinical abbreviations\n\n**Symptom.** *\"Did NH-1023 have episodes of shortness of breath overnight?\"* misses the nursing notes. **Cause:** nurses chart in shorthand (`Pt c/o SOB, SpO2 88% RA → 2L NC`). The phrase \"shortness of breath\" never appears, so the embedding of the plain-English question lands far from the shorthand notes."),
code('''from utils.retrieval import rewrite_query

q = "Did NH-1023 have any episodes of shortness of breath overnight during the September 2025 admission?"
for variant in rewrite_query(q):
    print("•", variant)'''),
code('''for name, cfg in [("no rewrite", improved_v1), ("rewrite", improved_v1.with_(query_rewrite=True))]:
    hits = retrieve(q, "NH-1023", cfg)
    print(f"{name:10s} →", [h.metadata["doc_id"].replace("NH-1023-", "") for h in hits])'''),

md("## 5. Fix 4: chunk size and overlap (don't strand the addendum)\n\n**Symptom.** *\"Does NH-1009 currently have an active allergy to Myrexatin?\"* says **yes**. **Cause:** the allergy was removed in an **addendum** at the end of a long consult note, and its conclusion (\"the challenged agent… label removed\") never names the drug. In small chunks, the conclusion is cut off from the paragraph that does."),
code('''for size, overlap in [(800, 0), (2400, 300)]:
    cfg = RetrievalConfig(chunker="structure", chunk_size=size, chunk_overlap=overlap)
    add = [c for c in chunk_corpus(cfg) if c.metadata["doc_id"] == "NH-1009-03-consult-note" and c.metadata["section"] == "addendum"]
    names_drug = ["Myrexatin" in c.page_content and "removed" in c.page_content for c in add]
    print(f"size={size} overlap={overlap}: addendum in {len(add)} chunk(s); drug name + conclusion together: {any(names_drug)}")'''),

md("### Run `improved-v2` = v1 + Fix 3 + Fix 4\n\n**TODO 2.** Start from `improved_v1` and add query rewriting, `chunk_size=2400`, `chunk_overlap=300`, `k=6`."),
code('''improved_v2 = improved_v1.with_(name="improved-v2", chunk_size=2400, chunk_overlap=300, k=6, query_rewrite=True)
v2 = run_experiment(improved_v2)''',
todo='''# TODO 2: improved_v2 = improved_v1.with_(name="improved-v2", ...)
improved_v2 = ...
v2 = run_experiment(improved_v2)'''),

md("## 6. Compare experiments side by side"),
code('''def card_for(project) -> dict:
    """Rebuild a scorecard from a finished experiment's feedback (works for runs from earlier labs too)."""
    rows = []
    runs = list(client.list_runs(project_id=project.id, is_root=True))
    fbs = {}
    for fb in client.list_feedback(run_ids=[r.id for r in runs]):
        fbs.setdefault(fb.run_id, []).append(fb)
    examples = {e.id: e for e in client.list_examples(dataset_name=dataset_name())}
    for r in runs:
        ex = examples.get(r.reference_example_id)
        if ex is None:
            continue
        rows.append({"example": ex, "evaluation_results": {"results": fbs.get(r.id, [])}})
    return scorecard(rows)


cards = {
    "baseline": card_for(base_exp),
    "improved-v1": scorecard(v1),
    "improved-v2": scorecard(v2),
}
compare(cards)'''),
code('''for name, card in cards.items():
    fm = {r["failure_mode"]: r.get("correctness") for r in card["rows"] if r["category"] == "failure_mode"}
    print(f"{name:12s}", fm)'''),
md("**In the UI:** open the comparison view to see every question across all three experiments, with diffs and traces."),
code('''ds = client.read_dataset(dataset_name=dataset_name())
ids = [str(base_exp.id), str(client.read_project(project_name=v1.experiment_name).id), str(client.read_project(project_name=v2.experiment_name).id)]
print("🔗 compare:", f"{ds.url}/compare?selectedSessions={','.join(ids)}")'''),
md("""
> 💡 **Read the comparison critically.**
> - Which fix moved which category? (The failure-mode row should move most.)
> - Did anything get **worse**? Bigger chunks cost tokens; query rewriting adds an LLM call and latency.
> - Re-run an experiment once: LLM pipelines are noisy, and a one-question difference may be chance. Use `num_repetitions=` in `evaluate` for tighter estimates.
"""),

md("""
## 7. Stretch: let an agent drive retrieval

So far retrieval runs once per tool call, with a fixed recipe. Give the agent a **richer tool** that can filter by document type and recency, and let it search again when the first results look stale. Then run the **same eval**.
"""),
code('''from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from chart_qa_graph import answer_format, system_prompt
from utils.models import get_model

v2_store_cfg = improved_v2


@tool
def search_chart(query: str, patient_id: str, doc_type: str | None = None, newest_first: bool = True,
                 runtime: ToolRuntime = None) -> str:
    """Search ONE patient's chart. Optional doc_type: progress_note, discharge_summary, consult_note,
    nursing_note, medication_list, problem_list. Call again with a different query or doc_type if the
    first results are outdated or incomplete (e.g. check for addenda or newer notes)."""
    ctx = getattr(runtime, "context", None) or {}
    pid = (ctx.get("patient_id") if isinstance(ctx, dict) else None) or patient_id
    hits = retrieve(query, pid, v2_store_cfg.with_(k=10))
    if doc_type:
        hits = [h for h in hits if h.metadata["doc_type"] == doc_type] or hits
    if newest_first:
        hits.sort(key=lambda h: h.metadata["date"], reverse=True)
    return format_docs(hits[:6], v2_store_cfg)


model = get_model()
agentic = create_agent(model, tools=[search_chart], response_format=answer_format(model),
                       system_prompt=system_prompt(improved_v2, "You may search several times. Verify the newest documents before answering."))


def run_agentic(inputs: dict) -> dict:
    result = agentic.invoke({"messages": [{"role": "user", "content": inputs["question"]}]}, context={"patient_id": inputs["patient_id"]})
    return {"answer": result["structured_response"].model_dump(), "contexts": [m.content for m in result["messages"] if m.type == "tool"]}'''),
code('''# Stretch: uncomment to run (≈3 min)
# from langsmith import evaluate
# from evals.evaluators import ALL_EVALUATORS
# agentic_res = evaluate(run_agentic, data=dataset_name(), evaluators=ALL_EVALUATORS,
#                        experiment_prefix=scoped("improved-v2-agentic"), max_concurrency=4)
# cards["v2-agentic"] = scorecard(agentic_res)
# compare(cards)'''),

md("""
## What you just learned

| Root cause (from the trace) | Fix | Where |
|---|---|---|
| Table rows split away from their header | structure-aware chunking, tables atomic | `structure_chunks` |
| Older document contradicts a newer one | dates in context, newest first, plus a prompt rule | `date_aware=True` + `DATE_AWARE_RULES` |
| Model guesses the wrong look-alike patient | bind retrieval to the session's patient | `bind_patient=True` |
| Plain-English query vs shorthand notes | LLM query rewriting with an abbreviation glossary | `query_rewrite=True` |
| Key fact stranded in a long note's addendum | larger chunks with overlap | `chunk_size`, `chunk_overlap` |

The loop: **trace → root cause → one change → experiment → compare**. That loop, not any single trick, is what keeps a clinical assistant improving safely.

`improved-v1` is the pipeline the day's labs use as `RAG_CONFIG` (`chart_qa_graph.py`).
"""),
]
