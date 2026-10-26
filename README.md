# Chart Q&A assistant: a LangChain / LangGraph / LangSmith workshop lab

> **⚠️ Synthetic data notice.** Every patient, document, medication, dose, policy and the health system itself
> ("Northfield Health") is **fictional**, invented for teaching. There is no real patient data, no PHI, and
> no real clinical guidance. Medication names (Alvexora, Thrombelix, Nephravant, …) are made up. This lab is
> not a clinical tool.

Clinicians ask questions about a synthetic patient's record. The assistant answers **only from that record**, with **citations to the exact document and section**:

- *"What anticoagulants is patient NH-1042 on, and when was each started?"*
- *"Summarize the last three cardiology notes for NH-1017."*
- *"What was the most recent creatinine, and is it newer than the discharge summary?"*

It **never makes clinical decisions**. It declines requests like *"should I increase the dose?"* and defers to the responsible clinician. Three guardrails run throughout:

| Guardrail | Behavior |
|---|---|
| **Refusal** | recommendations and treatment decisions are declined (`disposition = declined_clinical_decision`) |
| **Escalation** | possible safety events (medication error, fall, near miss) are flagged; in the agent, filing goes through **human approval** |
| **PHI masking** | names, MRN-like numbers and DOBs pasted into a question are masked before they reach the model **or the trace** (`NH-####` is the allowed record key) |

Everything runs **locally on a laptop**: local Chroma vector store, Jupyter notebooks, and `langgraph dev`. No cloud deployment.

## The day

| Notebook | Slot | You learn |
|---|---|---|
| `00_setup_check` | before 9:00 | env, model, embeddings, LangSmith, and a trace that lands (also `scripts/check_setup.py`) |
| `01_build_rag` | Lab 1 · 75 min | chunk → embed (Chroma) → patient-filtered retrieval → `create_agent` + typed, cited `response_format` → read a trace |
| `02_evaluate` | Lab 2 · 60 min | golden dataset in LangSmith, 3 LLM judges (OpenEvals) + 2 code evaluators, a `baseline` experiment |
| `03_improve` | Lab 3 · 45 min | eval failure → trace → fix: structure-aware chunking, date-aware ranking, query rewriting, size/overlap → `improved-v1/v2` comparison |
| `04_agent` | 3:00 demo · 35 min | structured FHIR lookup, PHI middleware, tool-call limit, HITL before filing a safety event; then a deep agent with per-source subagents; trajectory eval |
| `05_observe` | Day 2 (optional) | online-eval run rules and annotation queues (opt-in: `ENABLE_ONLINE_EVALS=1`) |

Attendee notebooks in `notebooks/` contain `# TODO` exercises. Completed versions with the same filenames are in **`solutions/`**.

## Prerequisites

- macOS / Linux / WSL laptop, **Python 3.11** (uv installs it for you) and [`uv`](https://docs.astral.sh/uv/)
- A **LangSmith API key** for a workspace with the LLM Gateway enabled. By default OpenAI models are called **through the LangSmith gateway with that same key**, so no OpenAI key is needed.
- *(Optional)* [Ollama](https://ollama.com) to run generation and embeddings locally

## Quickstart

```bash
git clone <this repo> chart-qa-workshop && cd chart-qa-workshop
uv sync                      # Python 3.11 + pinned dependencies
cp .env.example .env         # set WORKSHOP_USER and LANGSMITH_API_KEY
uv run python scripts/check_setup.py
uv run jupyter lab notebooks/
```

`check_setup.py` prints a ✅/❌ line for each item and exits non-zero if anything fails. Open `notebooks/00_setup_check.ipynb`, then work through 01 → 04.

## `.env` keys

| Key | Required | Default | Notes |
|---|---|---|---|
| `WORKSHOP_USER` | **yes** | `<first-last>` placeholder | e.g. `jane-doe`. Every LangSmith dataset, experiment, project, prompt, rule and queue you create is suffixed with it, so 30 people can share one workspace. Notebooks **fail at the first cell** if it's unset. |
| `LANGSMITH_API_KEY` | **yes** | | Tracing, datasets, experiments, and the default model route (gateway) |
| `LANGSMITH_TRACING` | | `true` | |
| `LANGSMITH_PROJECT` | | `chart-qa-${WORKSHOP_USER}` | Always scoped to you |
| `LANGSMITH_WORKSPACE_ID` | | | Pin a workspace if your key sees several |
| `LABS_MODEL` | | `openai:gpt-5.4-mini` | Generation. `provider:model`, e.g. `ollama:qwen2.5:7b` |
| `LABS_EMBEDDINGS` | | `openai:text-embedding-3-small` | Embeddings, e.g. `ollama:nomic-embed-text` |
| `LABS_JUDGE_MODEL` | | `openai:gpt-5.4-mini` | LLM-as-judge evaluators. Kept **hosted** even when generation is local |
| `LABS_OPENAI_BASE_URL` | | LangSmith gateway | Set to `""` to call OpenAI directly with `OPENAI_API_KEY` |
| `OPENAI_API_KEY` | only if direct | | Only when `LABS_OPENAI_BASE_URL=""` |
| `OLLAMA_BASE_URL` | | `http://localhost:11434` | |
| `LABS_VECTORSTORE` | | `chroma` | Set to `memory` to force the in-memory store |
| `ENABLE_ONLINE_EVALS` | | unset | `1` lets notebook 05 create run rules / queues |

### Local models with Ollama

```bash
brew install ollama          # or see ollama.com for Linux/Windows
ollama serve &               # leave running
ollama pull qwen2.5:7b       # ~4.7 GB, generation
ollama pull nomic-embed-text # ~270 MB, embeddings
```

```dotenv
LABS_MODEL="ollama:qwen2.5:7b"
LABS_EMBEDDINGS="ollama:nomic-embed-text"
# keep the judge hosted:
LABS_JUDGE_MODEL="openai:gpt-5.4-mini"
```

Changing `LABS_EMBEDDINGS` creates a separate Chroma collection automatically; nothing needs deleting. The agent uses `ToolStrategy` for structured output on Ollama, because native JSON mode there suppresses tool calls.

> ⚠️ **Local models run the whole pipeline, but the guardrails are weaker.** In testing with `qwen2.5:7b`, notebooks 00, 01 and 03 ran without errors. However, the model sometimes returned free text instead of the typed answer, answered a dose-change question instead of declining, and missed the planted medication error. Use the hosted model for anything you evaluate or demo.

A single exported variable can override `.env` for one run, without editing it. This works for `WORKSHOP_USER` and the `LABS_*` switches (credentials always come from `.env`):

```bash
LABS_MODEL=ollama:qwen2.5:7b LABS_EMBEDDINGS=ollama:nomic-embed-text uv run python scripts/check_setup.py
```

## LangGraph Studio

```bash
uv run langgraph dev    # http://127.0.0.1:2024
```

Studio: <https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>

| Graph | What it is |
|---|---|
| `chart_qa_rag` | the RAG app (improved retrieval) |
| `chart_qa_agent_hitl` | notes search + structured lookup + `file_safety_event` behind human approval, PHI middleware, tool-call limit |
| `chart_qa_deep_agent` | coordinator delegating to `notes-researcher` and `structured-lookup` subagents |

## Repository layout

```
chart_qa_graph.py       typed ChartAnswer, tools, and the three graphs
utils/                  models (LABS_* switches), workshop.scoped(), corpus loader, retrieval + fixes, PHI detectors, tracing
evals/                  golden dataset (JSON), seeders, evaluators, experiment harness
data/charts/NH-xxxx/    synthetic chart notes (Markdown + front matter)
data/fhir/              synthetic FHIR-style structured records
data/policies/          fictional refusal / escalation / identifier policies (not retrieved)
data/FAILURE_MODES.md   facilitator-only answer key for Lab 3 (not retrieved)
data/checks.py          offline consistency + synthetic-only checks
notebooks/  solutions/  attendee notebooks with TODOs / completed versions
scripts/                check_setup.py, run_experiments.py, build_notebooks.py (+ notebook_src/)
```

Notebooks are generated from `scripts/notebook_src/*.py`. Edit there, then run `uv run python scripts/build_notebooks.py`.

## Common issues

| Symptom | Fix |
|---|---|
| `RuntimeError: WORKSHOP_USER is not set` | Set it in `.env`, then **restart the kernel** (`.env` is read at the first cell) |
| `401 … Incorrect API key provided: lsv2_…` on model calls | The gateway didn't accept your key. Check that the key belongs to the gateway-enabled workspace (`LANGSMITH_WORKSPACE_ID`), or switch to direct OpenAI (`LABS_OPENAI_BASE_URL=""` + `OPENAI_API_KEY`) |
| `.env` changes ignored | Notebooks load `.env` with `override=True`. Restart the kernel after editing |
| Chroma fails to install or start | It falls back to `InMemoryVectorStore` automatically (warning printed). Or set `LABS_VECTORSTORE=memory` |
| Stale answers after editing a chart doc | Delete `.chroma/` to re-index |
| `evaluate()` grades old references | `seed_dataset(refresh=True)` after editing `evals/dataset_chart_qa.json` |
| Ollama: `connection refused` | `ollama serve` isn't running, or `OLLAMA_BASE_URL` is wrong |
| Scores differ by a question or two between runs | LLM pipelines are noisy. Compare averages over `num_repetitions=2+` |
| Mermaid graph image doesn't render | Offline renderer. The notebooks print the node list instead |
