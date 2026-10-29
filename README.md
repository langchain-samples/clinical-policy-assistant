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
| **Escalation** | possible safety events (medication error, fall, near miss) are flagged; in the workflow, filing goes through **human approval** |
| **PHI masking** | names, MRN-like numbers and DOBs pasted into a question are masked before they reach the model **or the trace** (`NH-####` is the allowed record key) |

Everything runs **locally on a laptop**: local Chroma vector store, Jupyter notebooks, and `langgraph dev`. No cloud deployment.

## The day: one application, four milestones

You build the chart assistant once and evolve it through four hands-on milestones:

| Notebook | Milestone | You build | You learn |
|---|---|---|---|
| `00_setup_check` | before 9:00 | — | env, model, embeddings, LangSmith, a trace that lands (also `scripts/check_setup.py`) |
| `01_build_rag` | **1 · RAG** (60 min) | retrieve → generate pipeline with a typed, cited answer | chunking, Chroma, patient-filtered **retrieval**, **grounding** with verbatim citations, reading a trace |
| `02_agent` | **2 · Agent → deep agent** (60 min) | `create_agent` with notes search + a structured-record tool; then a deep agent with per-source subagents | **tool selection**, the **reasoning loop**, tool-call limits, memory; deep-agent **design considerations**; managed **Context Hub** (versioned `AGENTS.md` + skills) and **Sandboxes** (isolated `execute`) |
| `03_langgraph` | **3 · LangGraph** (75 min) | a `StateGraph`: mask → classify → refuse / agent → human review → file | typed state + reducers, nodes, edges, **conditional routing**, `interrupt()` + resume, **durable execution**, the **framework vs application** line, Studio |
| `04_evaluate` | **4 · Evaluation** (60 min) | experiments for the pipeline, the agent and the workflow | **answer** + **behavior** evals (tool selection, trajectory, route), **regression gates**, quality over time, **observability & governance** (trace redaction, deployment posture) |

**Optional / Day 2** (`notebooks/optional/`):

| Notebook | Topic |
|---|---|
| `improve_retrieval` | tune the RAG pipeline against evals: structure-aware chunking, date-aware ranking, query rewriting, size/overlap |
| `deep_agent` | a deep agent: coordinator + per-source subagents, memory, guardrails, evals |
| `observe` | online-eval run rules and annotation queues (opt-in: `ENABLE_ONLINE_EVALS=1`) |

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

> ⚠️ **Local models run the labs, but behave less reliably.** In testing with `qwen2.5:7b`, notebooks 00, 01 and 03 ran without errors. However, the RAG pipeline answered a dose-change question instead of declining, and in Lab 3 the router sent the medication-error question to `refuse` instead of `answer` → `human_review`. (The deterministic graph steps still held: masking ran first and nothing was filed without approval.) Use the hosted model for anything you evaluate or demo.

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
| `chart_qa_rag` | Milestone 1: the retrieve → generate pipeline. Input: `{"question": …, "patient_id": "NH-…"}` |
| `chart_qa_agent` | Milestone 2: the agent (notes search + structured lookup, tool-call limit). Input: a chat message |
| `chart_qa_workflow` | Milestone 3: the controlled workflow with the human-review interrupt. Input: `{"question": …, "patient_id": "NH-…"}` |
| `chart_qa_deep_agent` | Optional: coordinator delegating to `notes-researcher` and `structured-lookup` subagents |

## Repository layout

```
chart_qa_graph.py       typed ChartAnswer, tools, and the app at each milestone (pipeline, agent, workflow, deep agent)
agent_context/          deep agent's team conventions (AGENTS.md) + skills; published to Context Hub in Lab 2
utils/                  models (LABS_* switches), workshop.scoped(), corpus loader, retrieval, PHI detectors, tracing, managed (Context Hub / Sandboxes)
evals/                  golden dataset (JSON), seeder, evaluators, experiment harness, regression thresholds
data/charts/NH-xxxx/    synthetic chart notes (Markdown + front matter)
data/fhir/              synthetic FHIR-style structured records
data/policies/          fictional refusal / escalation / identifier policies (not retrieved)
data/checks.py          offline consistency + synthetic-only checks
notebooks/  solutions/  attendee notebooks with TODOs / completed versions
scripts/                check_setup.py, run_experiments.py, eval_gate.py (CI regression gate), build_notebooks.py (+ notebook_src/)
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
| Lab 2 Part B: "Context Hub unavailable" or "Sandboxes unavailable" | Those LangSmith features aren't enabled for your workspace. The cells still run: the agent reads `agent_context/` from disk, and reports raw values instead of running code |
