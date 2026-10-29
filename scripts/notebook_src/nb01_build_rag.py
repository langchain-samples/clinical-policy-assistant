"""Source for notebooks/01_build_rag.ipynb (Milestone 1 · RAG)."""

from _cells import SETUP, code, md

FILENAME = "01_build_rag.ipynb"

CELLS = [
md("""
# 01 · RAG: answer questions from documents

**Northfield Health chart assistant** (synthetic data only) · **Milestone 1 of 4** · **Time box:** 60 min

Today you build **one application** and evolve it through four milestones:

| Lab | Milestone | The chart assistant becomes… |
|---|---|---|
| **01** | **RAG** | **a pipeline that retrieves chart passages and answers with citations** ← you are here |
| 02 | Agent | a tool-using agent that decides what to look up, and where |
| 03 | LangGraph | a controlled, stateful workflow with routing and human approval |
| 04 | Evaluation | measured: answer correctness *and* agent behavior |

Clinicians ask about one patient's record. The assistant answers **only from that record**, with citations to the exact document and section, and it **never makes clinical decisions**.

**Goals.** By the end you can:
- chunk a Markdown chart corpus that carries front-matter metadata, and see why chunking strategy matters
- embed it into a **local, persistent vector store** (Chroma, with an in-memory fallback)
- retrieve with a **patient-ID metadata filter**, so one patient's notes never leak into another's answer (**retrieval**)
- generate a **typed answer with verbatim citations** from only the retrieved text (**grounding**)
- read a LangSmith trace end to end

There are **3 `# TODO` cells**. If you fall behind, `solutions/01_build_rag.ipynb` has them completed.
"""),
md("## 0. Setup: tracing is on from the first cell"),
code(SETUP),

md("""
## 1. The synthetic chart corpus

8 fictional patients (`NH-1001` … `NH-1042`), 6–10 documents each: progress notes, discharge summaries, consult notes, nursing notes, medication lists and problem lists. Every file is Markdown with **front matter**, and that metadata is what lets us filter and cite.
"""),
code('''from collections import Counter

from utils.corpus import load_documents

docs = load_documents()
print(f"{len(docs)} documents")
print(Counter(d.patient_id for d in docs))
print(Counter(d.doc_type for d in docs))'''),
code('''doc = next(d for d in docs if d.doc_id == "NH-1042-04-progress-note")
print(doc.metadata())
print()
print(doc.body[:900])'''),
md("> 💡 Facilitator notes and `data/policies/` are **excluded** from retrieval by `utils/corpus.py`. Only chart notes are searchable."),

md("""
## 2. Chunking: why the strategy matters

Retrieval works on **chunks**, not whole documents. The simplest chunker cuts every 500 characters. Watch what it does to a lab table: the potassium row loses the header that says which column is which date.
"""),
code('''from utils.retrieval import BASELINE, chunk_corpus

naive = chunk_corpus(BASELINE)  # fixed 500-character chunks, no overlap
table = [c for c in naive if c.metadata["doc_id"] == "NH-1017-06-discharge-summary" and "|" in c.page_content]
print(f"naive: {len(naive)} chunks; the lab table is cut into {len(table)} pieces. One of them:\\n")
print(table[1].page_content[:400])'''),
md("""
A **structure-aware** chunker splits on section headings, keeps tables whole, and stamps every chunk with a `[title | type | date | section]` header. We use it, with patient binding and dated sources, for the rest of the day: that's `RAG_CONFIG`.
"""),
code('''from chart_qa_graph import RAG_CONFIG

print(RAG_CONFIG)
chunks = chunk_corpus(RAG_CONFIG)
table = [c for c in chunks if c.metadata["doc_id"] == "NH-1017-06-discharge-summary" and "Potassium" in c.page_content]
print(f"\\nstructure-aware: {len(chunks)} chunks. The table stays in one piece:\\n")
print(table[0].page_content[:700])'''),
md("> 🔎 Tuning chunk size, overlap, ranking and query rewriting against an eval is covered in `notebooks/optional/improve_retrieval.ipynb` (Day 2)."),

md("""
## 3. Embed into a local, persistent vector store

`build_vectorstore` embeds every chunk with `LABS_EMBEDDINGS` and stores it in **Chroma** under `.chroma/`, so re-runs reuse the index. If Chroma can't install or start on your laptop, it falls back to LangChain's `InMemoryVectorStore` automatically.
"""),
code('''from utils.retrieval import build_vectorstore

store = build_vectorstore(RAG_CONFIG)
print(type(store).__name__)'''),

md("""
## 4. Retrieval with a patient-ID filter

Two patients have near-identical names: **Marion Hale** (`NH-1028`) and **Marian Hale** (`NH-1035`). Without a filter, a search for one returns the other's notes.
"""),
code('''question = "What Pulmoquin dose is the patient taking?"

for d in store.similarity_search(question, k=4):
    print("NO FILTER →", d.metadata["patient_id"], d.metadata["doc_id"])'''),
md("**TODO 1.** Restrict the search to `NH-1035`. Chroma takes a metadata filter: `filter={\"patient_id\": ...}`."),
code('''for d in store.similarity_search(question, k=4, filter={"patient_id": "NH-1035"}):
    print("FILTERED  →", d.metadata["patient_id"], d.metadata["doc_id"])''',
todo='''# TODO 1: add a metadata filter so only NH-1035's chunks come back.
for d in store.similarity_search(question, k=4):
    print("FILTERED  →", d.metadata["patient_id"], d.metadata["doc_id"])'''),
md("""
The retrieval step of our pipeline wraps exactly that: `retrieve(question, patient_id, RAG_CONFIG)` filters by patient, returns the top chunks **newest first**, and `format_docs` tags each one with a citation key `[source: doc_id#section]` plus its date.

`@traceable(run_type="retriever")` makes the step show up in the trace with its documents.
"""),
code('''from langsmith import traceable

from utils.retrieval import format_docs, retrieve


@traceable(run_type="retriever", name="retrieve")
def retrieve_step(question: str, patient_id: str):
    return retrieve(question, patient_id, RAG_CONFIG)


hits = retrieve_step("anticoagulant", "NH-1042")
print(format_docs(hits, RAG_CONFIG)[:1500])'''),

md("""
## 5. A typed answer with citations

The answer's **shape is code**, not prompt text: a Pydantic model. The model must fill it in, so every answer comes back with a disposition, citations, and a safety flag you can check programmatically.

**TODO 2.** Add a `quote` field to `Citation`: a short **verbatim** span from the cited passage. Lab 4's `citation_validity` evaluator checks that the quote really is in that document. That is what makes "grounded" testable.
"""),
code('''from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(description="The doc_id from the [source: doc_id#section] tag.")
    section: str = Field(description="The section anchor after '#'.")
    quote: str = Field(description="A short VERBATIM quote (under ~25 words) supporting the claim.")


class ChartAnswer(BaseModel):
    disposition: Literal["answered", "not_in_record", "declined_clinical_decision", "safety_event_flagged"]
    answer: str = Field(description="Concise answer for a clinician, with dates and values as documented.")
    citations: list[Citation]
    safety_event: bool = Field(description="True if a possible patient-safety event is described.")
    safety_event_summary: str | None = None''',
todo='''from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(description="The doc_id from the [source: doc_id#section] tag.")
    section: str = Field(description="The section anchor after '#'.")
    # TODO 2: add `quote: str` with a Field(description=...) asking for a short VERBATIM supporting quote.


class ChartAnswer(BaseModel):
    disposition: Literal["answered", "not_in_record", "declined_clinical_decision", "safety_event_flagged"]
    answer: str = Field(description="Concise answer for a clinician, with dates and values as documented.")
    citations: list[Citation]
    safety_event: bool = Field(description="True if a possible patient-safety event is described.")
    safety_event_summary: str | None = None'''),

md("""
## 6. The system prompt (in Prompt Hub)

The prompt sets the rules: answer only from the given record text, cite every claim, say "not in record" instead of guessing, **decline clinical decisions**, and **flag safety events**. We publish it to Prompt Hub under **your** scoped name, so you can edit it in the UI without touching code.
"""),
code('''from chart_qa_graph import PROMPT_BASENAME, PROMPT_PATH, system_prompt
from utils.prompts import prompt_url, seed_prompt
from utils.workshop import scoped

PROMPT_NAME = scoped(PROMPT_BASENAME)
try:
    seed_prompt(PROMPT_NAME, PROMPT_PATH.read_text().strip())
except Exception as exc:  # already exists with identical content, or Prompt Hub unavailable
    print("(prompt not re-pushed:", type(exc).__name__, ")")
SYSTEM = system_prompt(RAG_CONFIG)  # your Prompt Hub prompt + the "newest dated source wins" rule
print(SYSTEM[:700], "…")
print("\\n📝 edit it here:", prompt_url(PROMPT_NAME))'''),

md("""
## 7. Generate: one grounded model call

The generation step gets the system prompt, the question, and **only** the retrieved passages, and must return a `ChartAnswer`. `with_structured_output(ChartAnswer)` makes the model fill in the schema and gives you back a parsed object.

**TODO 3.** Build the messages and call the structured model.
"""),
code('''from utils.models import get_model

model = get_model()


def generate(question: str, patient_id: str, docs) -> ChartAnswer:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Patient: {patient_id}\\nQuestion: {question}\\n\\n"
                                    f"Answer only from these retrieved chart passages:\\n<context>\\n{format_docs(docs, RAG_CONFIG)}\\n</context>"},
    ]
    return model.with_structured_output(ChartAnswer).invoke(messages)''',
todo='''from utils.models import get_model

model = get_model()


def generate(question: str, patient_id: str, docs) -> ChartAnswer:
    # TODO 3: build two messages:
    #   system → SYSTEM
    #   user   → the patient ID, the question, and format_docs(docs, RAG_CONFIG) inside <context>…</context>
    # then return model.with_structured_output(ChartAnswer).invoke(messages)
    ...'''),
md("**The whole pipeline** is two steps. `@traceable` groups them into one trace."),
code('''@traceable(name="chart_qa_rag")
def rag(question: str, patient_id: str) -> ChartAnswer:
    docs = retrieve_step(question, patient_id)   # 1. retrieval: this patient's most relevant passages
    return generate(question, patient_id, docs)  # 2. grounding: answer only from them, with citations'''),

md("## 8. Ask it questions"),
code('''from utils.trace import run_traced, show

show(run_traced(rag, "What anticoagulants is patient NH-1042 on, and when was each started?", "NH-1042"))'''),
md("""
### 📖 How to read the trace (open the 🔗 link above)

1. **`chart_qa_rag`** (root): the whole question → answer, with total latency and tokens.
2. **`retrieve`** (retriever run): the exact chunks that came back, with their metadata. Were they the right ones, for the right patient?
3. **The model run**: the exact messages it received (system prompt + context) and the structured `ChartAnswer` it returned.

When an answer is wrong, the trace tells you **where**: the right chunk never retrieved? retrieved but ignored? wrong patient?

Check the answer above against the chart: `NH-1042-01-medication-list` (January) still lists Coagulor; `NH-1042-04-progress-note` (March) stops it and starts Thrombelix. Retrieval found both, dated, newest first. Does the answer **lead with what's current** (Thrombelix since 2025-03-14), or does it frame the outdated list as the main fact? If the latter, retrieval wasn't the problem: the notes never label Thrombelix as an "anticoagulant". Lab 2 gives the model a structured record where every medication has a class, a status and dates.
"""),
code('''show(run_traced(rag, "What was the most recent creatinine for NH-1017, and is it newer than the discharge summary?", "NH-1017"))'''),
code('''show(run_traced(rag, "What was NH-1017's potassium on June 11, 2025?", "NH-1017"))  # the answer lives in that lab table'''),

md("## 9. The three guardrails (prompt-level for now)"),
md("**Refusal.** Recommendation and decision requests are declined and pointed to the responsible clinician."),
code('''show(run_traced(rag, "Should I increase NH-1042's Thrombelix dose now that his kidney function changed?", "NH-1042"))'''),
md("**Escalation.** A possible safety event (medication error, fall, near miss) is flagged for patient-safety reporting."),
code('''show(run_traced(rag, "Were there any problems with medication administration during NH-1004's May 2025 stay?", "NH-1004"))'''),
md("**PHI masking.** Identifiers someone pastes into a question (names, MRN-like numbers, DOBs) must be removed **before** the question reaches the model or the trace. `NH-####` is the allowed record key and stays."),
code('''from utils.phi import mask_text

raw = "Pt Corinne Abernathy, DOB 12/05/1959, MRN 00419276, chart NH-1017: current anticoagulant?"
masked, kinds = mask_text(raw)
print("masked:", masked, kinds)
show(run_traced(rag, masked, "NH-1017"))'''),
md("""
> ⚠️ Right now each guardrail depends on the prompt, or on *you remembering* to call `mask_text`. In Lab 3 they become **guaranteed steps in a graph**: masking always runs first, decision requests are refused without calling the model, and a safety report needs human approval.
"""),

md("""
## 10. Where a fixed pipeline breaks (why Lab 2 exists)

The pipeline retrieves **exactly once**, with **the clinician's own words**, from **one source**. It can't search again with different wording, and it can't look anything up in the structured record.
"""),
code('''# The overnight episodes are charted only in nursing shorthand ("c/o SOB", "2L NC").
show(run_traced(rag, "Did NH-1023 have any episodes of shortness of breath overnight during the September 2025 admission?", "NH-1023"))'''),
code('''# The answer is in a late addendum to a long consult note, and in the structured record.
show(run_traced(rag, "Does NH-1009 currently have an active allergy to Myrexatin?", "NH-1009"))'''),
md("If either answer is incomplete or wrong, open its trace: the right passage probably never made it into `retrieve`'s top chunks. Lab 2 lets the model decide what to search for, search again, and use a second tool."),

md("""
## What you just learned

| Step | What we used |
|---|---|
| Corpus + metadata | Markdown front matter → `patient_id`, `doc_type`, `date`, `section` on every chunk |
| Chunking | fixed-size cuts tables apart; structure-aware keeps them whole (`RAG_CONFIG`) |
| Local vector store | Chroma persisted in `.chroma/` (in-memory fallback) |
| **Retrieval** | patient metadata `filter={"patient_id": …}`, newest first, citation keys |
| **Grounding** | answer only from retrieved text; typed `ChartAnswer` with verbatim quotes (`with_structured_output`) |
| Guardrails (prompt-level) | refusal · safety escalation · PHI masking |
| Observability | `@traceable` steps → one LangSmith trace per question |

The same pipeline is packaged as `rag_answer()` in `chart_qa_graph.py`. **Next:** `02_agent.ipynb` turns it into an agent.
"""),
]
