# Medical-Information QA Agent — a LangChain / LangGraph / LangSmith lab

A hands-on lab that builds a **medical-information QA assistant** and walks it through the full
agent-engineering loop: **build → evaluate → orchestrate → deep agent → observe.**

The agent answers drug questions with **cited** answers grounded in prescribing information (PI),
**refuses** off-label and personal-medical-advice requests, **escalates** suspected adverse events, and
**never exposes patient identifiers**. Everything runs on **synthetic, fictional data** — the drugs
(Alvexora, Brevamistol, Cardioryn, Dermasolix, Elmurestat), their labels, and every inquiry are invented
for teaching. No real drugs, prescribing information, patient data, PII, secrets, or clinical advice.

## What you'll build (5 notebooks)

| Notebook | You learn |
|---|---|
| `01_build.ipynb` | Create grounding tools, wire an agent with `create_agent`, and enforce a typed answer with `response_format`. |
| `02_evaluate.ipynb` | Seed an eval dataset and score the agent with LangSmith `evaluate()` — LLM-as-judge and code evaluators. |
| `03_orchestrate.ipynb` | Add middleware — human-in-the-loop approval, PII redaction, tool-call limits — and memory/threads. |
| `04_deep_agent.ipynb` | Scale to a **deep agent**: a coordinator that delegates per drug, with the three healthcare guardrails — PHI masking, refusal/adverse-event policy, and human-in-the-loop — and short-term vs long-term (non-PHI) memory. |
| `05_observe.ipynb` | Production observability: tracing, online-eval run rules, and routing low scores to annotation queues. |

## Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/)
- An **OpenAI API key** (required)
- A **LangSmith API key** (optional — needed for evaluation, tracing, and the observability notebook)

## Quickstart

```bash
git clone https://github.com/langchain-samples/medical-information-qa.git
cd medical-information-qa
uv sync

cp .env.example .env      # then add your OPENAI_API_KEY (LangSmith key optional)

uv run jupyter lab notebooks/
```

Open `notebooks/01_build.ipynb` and run top to bottom, then continue through `05_observe.ipynb`.

### What needs a LangSmith key?

- **Runs without LangSmith:** `01_build`, `03_orchestrate` — the system prompt falls back to the
  committed `data/prompt.md`, and trace links are simply skipped.
- **Needs `LANGSMITH_API_KEY`:** `02_evaluate`, `04_deep_agent` (eval sections), `05_observe` — these
  seed datasets, run experiments, and create observability rules in **your own** workspace.

The eval dataset is committed as JSON (`evals/dataset_medical_information_qa.json`) and seeded
idempotently, so re-running never duplicates. Notebook 05's live online-eval rules are opt-in behind
`ENABLE_ONLINE_EVALS=1`.

> **Tip:** `04_deep_agent` includes a commented cell at the top that clears this lab's prior LangSmith
> experiments — uncomment it if repeated runs clutter your dataset's *Experiments* view, then re-run to
> repopulate with a single fresh experiment.

## Run the graphs in LangGraph Studio

The agents are registered in `langgraph.json`, so you can open them in LangGraph Studio:

```bash
uv run langgraph dev
```

This serves three graphs: `medical_information_qa_agent`, `medical_information_qa_agent_hitl`, and
`medical_information_qa_deep_agent`.

## Choosing the model

The model is set in one place — `utils/models.py` (default `openai:gpt-5.4-mini`). Override it without
touching code via `LABS_MODEL` in your `.env` (e.g. `LABS_MODEL="openai:gpt-4o-mini"`) if your key can't
call the default.

## Repository layout

```
medical-information-qa/
├── medical_information_qa_graph.py   # the deployable agent: tools, typed answer, and the 3 graphs
├── utils/                            # model selection, prompt loading, tracing, online-eval rules
├── evals/                            # dataset seeder + committed eval dataset (JSON)
├── data/                             # synthetic fixtures: PI labels, drug profiles, inquiries, policies
├── notebooks/                        # the 5 lab notebooks (+ images)
└── langgraph.json                    # graph registry for LangGraph Studio
```

---

*Synthetic data notice: all drugs, prescribing information, inquiries, and policies are fictional and
generated for teaching. Any resemblance to real medicines, people, or events is coincidental. This lab
is not medical advice and is not a pharmacovigilance system.*
