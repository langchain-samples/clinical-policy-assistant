"""Source for notebooks/01_build_rag.ipynb."""

from _cells import SETUP, code, md

FILENAME = "01_build_rag.ipynb"

CELLS = [
md("""
# 01 · Build a RAG chart assistant

**Northfield Health chart assistant** (synthetic data only) · **Agenda slot:** Lab 1 · **Time box:** 75 min

Clinicians ask questions about one patient's record. The assistant answers **only from that record**, with citations to the exact document and section. It **never makes clinical decisions**.

**Goals.** By the end you can:
- load and chunk a Markdown chart corpus that carries front-matter metadata
- embed it into a **local, persistent vector store** (Chroma, with an in-memory fallback)
- retrieve with a **patient-ID metadata filter**, so one patient's notes never leak into another's answer
- build an agent with `create_agent` that returns a **typed answer with citations** (`response_format`)
- read a LangSmith trace end to end

> ⚠️ We deliberately use **naive defaults** here: fixed 500-character chunks, no overlap, no query rewriting, and no date awareness. Some answers will be wrong. That's the point: Labs 2 and 3 measure and fix them.

There are **3 `# TODO` cells**. If you fall behind, `solutions/01_build_rag.ipynb` has them completed.
"""),
md("## 0. Setup: tracing is on from the first cell"),
code(SETUP),

md("""
## 1. The synthetic chart corpus

8 fictional patients (`NH-1001` … `NH-1042`), 6–10 documents each: progress notes, discharge summaries, consult notes, nursing notes, medication lists and problem lists. Every file is Markdown with **front matter**, and that metadata is what lets us filter and cite.
"""),
code('''from collections import Counter

from utils.corpus import load_documents, roster

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
## 2. Chunk it, the naive way

A `RecursiveCharacterTextSplitter` with a **fixed 500-character** chunk size and **no overlap**. Each chunk inherits the document's metadata, plus the section it came from (used for citations).
"""),
code('''from utils.retrieval import BASELINE, chunk_corpus

print(BASELINE)
chunks = chunk_corpus(BASELINE)
print(f"{len(chunks)} chunks; first one:")
print(chunks[0].metadata)
print(chunks[0].page_content)'''),
md("**Look at what fixed-size chunking does to a lab table.** Keep this cell in mind for Lab 3."),
code('''table_chunks = [c for c in chunks if c.metadata["doc_id"] == "NH-1017-06-discharge-summary" and "|" in c.page_content]
for c in table_chunks:
    print(f"--- chunk {c.metadata['chunk']} (section: {c.metadata['section']})")
    print(c.page_content[:400], "\\n")'''),

md("""
## 3. Embed into a local, persistent vector store

`build_vectorstore` embeds every chunk with `LABS_EMBEDDINGS` and stores it in **Chroma** under `.chroma/`, so re-runs reuse the index. If Chroma can't install or start on your laptop, it falls back to LangChain's `InMemoryVectorStore` automatically.
"""),
code('''from utils.retrieval import build_vectorstore

store = build_vectorstore(BASELINE)
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
## 5. Wrap retrieval as a tool

`make_search_tool(BASELINE)` returns `search_chart_notes(query, patient_id)`. It filters by patient and formats each hit with a citation key `[source: doc_id#section]`.
"""),
code('''from chart_qa_graph import make_search_tool

search_chart_notes = make_search_tool(BASELINE)
print(search_chart_notes.invoke({"query": "anticoagulant", "patient_id": "NH-1042"})[:1200])'''),

md("""
## 6. A typed answer with citations

The answer's **shape is code**, not prompt text: a Pydantic model passed as `response_format`. LangChain enforces it and returns a parsed object.

**TODO 2.** Add a `quote` field to `Citation`: a short **verbatim** span from the cited passage. Lab 2's `citation_validity` evaluator checks that the quote really is in that document.
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
## 7. The system prompt (in Prompt Hub)

The prompt sets the rules: ground every claim, cite it, say "not in record" instead of guessing, **decline clinical decisions**, and **flag safety events**. We publish it to Prompt Hub under **your** scoped name, so you can edit it in the UI without touching code.
"""),
code('''from chart_qa_graph import PROMPT_BASENAME, PROMPT_PATH
from utils.prompts import get_prompt, prompt_url, seed_prompt
from utils.workshop import scoped

PROMPT_NAME = scoped(PROMPT_BASENAME)
try:
    seed_prompt(PROMPT_NAME, PROMPT_PATH.read_text().strip())
except Exception as exc:  # already exists with identical content, or Prompt Hub unavailable
    print("(prompt not re-pushed:", type(exc).__name__, ")")
system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())
print(system_prompt[:600], "…")
print("\\n📝 edit it here:", prompt_url(PROMPT_NAME))'''),

md("## 8. Build the agent\n\n**TODO 3.** One `create_agent` call: model + tools + prompt + the response schema."),
code('''from langchain.agents import create_agent

from chart_qa_graph import answer_format
from utils.models import get_model

model = get_model()
agent = create_agent(
    model=model,
    tools=[search_chart_notes],
    system_prompt=system_prompt,
    response_format=answer_format(model, ChartAnswer),  # = ChartAnswer for hosted models (ToolStrategy on Ollama)
)''',
todo='''from langchain.agents import create_agent

from chart_qa_graph import answer_format
from utils.models import get_model

model = get_model()
# TODO 3: agent = create_agent(model=..., tools=[...], system_prompt=..., response_format=answer_format(model, ChartAnswer))
agent = ...'''),
code('''from IPython.display import Image, display

try:
    display(Image(agent.get_graph().draw_mermaid_png()))
except Exception:
    print(agent.get_graph().draw_ascii())'''),

md("## 9. Ask it questions"),
code('''from utils.trace import ask, show

show(ask(agent, "What anticoagulants is patient NH-1042 on, and when was each started?"))'''),
md("""
### 📖 How to read the trace (open the 🔗 link above)

1. **Root run** (`LangGraph`): the whole question → answer, with total latency and tokens.
2. **`model`** spans: the LLM deciding what to do. Expand one to see the exact system prompt and messages it received.
3. **`search_chart_notes`** tool spans: the **query** the model wrote and the **patient_id** it passed. The output is exactly what retrieval returned.
4. The last `model` span produces the structured `ChartAnswer`.

When an answer is wrong, the trace tells you **where**: bad query? wrong patient? the right chunk never retrieved? retrieved but ignored?

Does the answer above match the record? Hint: compare `NH-1042-01-medication-list` (January) with `NH-1042-04-progress-note` (March).
"""),
code('''show(ask(agent, "Summarize the last three cardiology notes for NH-1017."))'''),
code('''show(ask(agent, "What was the most recent creatinine for NH-1017, and is it newer than the discharge summary?"))'''),

md("## 10. The three guardrails"),
md("**Refusal.** Recommendation and decision requests are declined and pointed to the responsible clinician."),
code('''show(ask(agent, "Should I increase NH-1042's Thrombelix dose now that his kidney function changed?"))'''),
md("**Escalation.** A possible safety event (medication error, fall, near miss) is flagged for patient-safety reporting. (Lab 4 adds human approval before filing.)"),
code('''show(ask(agent, "Were there any problems with medication administration during NH-1004's May 2025 stay?"))'''),
md("**PHI masking.** Identifiers someone pastes into a question (names, MRN-like numbers, DOBs) are masked **before** the question reaches the model or the trace. `NH-####` is the allowed record key and stays."),
code('''answer = ask(agent, "Pt Corinne Abernathy, DOB 12/05/1959, MRN 00419276, chart NH-1017: current anticoagulant?", mask=True)
show(answer)  # open the trace: the input shows [NAME], [DOB], [MRN]'''),

md("## 11. Where naive RAG breaks (preview of Lab 3)"),
code('''show(ask(agent, "What was NH-1017's potassium on June 11, 2025?"))  # the answer is in a lab table…'''),
code('''show(ask(agent, "Does NH-1009 currently have an active allergy to Myrexatin?"))  # …and this one hides in an addendum'''),

md("""
## What you just learned

| Step | What we used |
|---|---|
| Corpus + metadata | Markdown front matter → `patient_id`, `doc_type`, `date`, `section` on every chunk |
| Chunking (naive) | fixed 500 chars, no overlap. Tables and addenda get cut apart |
| Local vector store | Chroma persisted in `.chroma/` (in-memory fallback) |
| Patient isolation | metadata `filter={"patient_id": …}` |
| Typed, cited answer | `create_agent(…, response_format=ChartAnswer)` with verbatim quotes |
| Guardrails | refusal · safety escalation · PHI masking before model **and** trace |
| Observability | every run is a LangSmith trace; read it top to bottom |

You probably saw some wrong answers. **Next:** `02_evaluate.ipynb` measures how often, and on which kinds of questions.
"""),
]
