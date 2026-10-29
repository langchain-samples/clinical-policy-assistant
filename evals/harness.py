"""Eval targets, experiment runners, and scorecards shared by Lab 4, the optional retrieval notebook, and
`scripts/run_experiments.py`.

Milestone systems (Lab 4) all return the same output shape, so one evaluator list grades every system:
    {"answer": <ChartAnswer dict>, "contexts": [...], "tool_calls": [...]?, "messages": [...]?, "path": [...]?}
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from chart_qa_graph import RAG_CONFIG, agent_tool_calls, answer_from_result, build_agent, build_rag_agent, build_workflow, rag_answer
from evals.evaluators import ALL_EVALUATORS, ANSWER_EVALUATORS, BEHAVIOR_EVALUATORS
from evals.seed_chart_qa import dataset_name, seed_dataset
from utils.retrieval import RetrievalConfig, build_vectorstore, config_summary
from utils.workshop import scoped

METRICS = ["correctness", "groundedness", "retrieval_relevance", "citation_validity", "refusal_correctness",
           "tool_selection", "trajectory_accuracy", "route_correctness", "tool_call_count"]


# --------------------------------------------------------------------------------------------------
# Milestone systems (Lab 4): the RAG pipeline, the agent, and the LangGraph workflow
# --------------------------------------------------------------------------------------------------

def _openai_messages(messages) -> list[dict]:
    from langchain_core.messages import convert_to_openai_messages

    return convert_to_openai_messages(messages)


def make_rag_target(model=None):
    build_vectorstore(RAG_CONFIG)

    def rag(inputs: dict) -> dict:
        out = rag_answer(inputs["question"], inputs["patient_id"], model=model)
        return {"answer": out["answer"].model_dump(), "contexts": out["contexts"]}

    return rag


def make_agent_target(model=None):
    build_vectorstore(RAG_CONFIG)
    agent = build_agent(model=model)

    def agent_target(inputs: dict) -> dict:
        result = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]},
                              context={"patient_id": inputs["patient_id"]})
        return {
            "answer": answer_from_result(result),
            "contexts": [str(m.content) for m in result["messages"] if getattr(m, "type", "") == "tool"],
            "tool_calls": agent_tool_calls(result["messages"]),
            "messages": _openai_messages(result["messages"]),
        }

    agent_target.__name__ = "agent"
    return agent_target


def make_workflow_target(model=None, *, review: str = "approve"):
    """Runs the Lab 3 graph. The human-review interrupt is answered automatically (`review`), and the path
    records that the review happened before anything was filed."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    build_vectorstore(RAG_CONFIG)
    graph = build_workflow(model=model, checkpointer=InMemorySaver())

    def workflow(inputs: dict) -> dict:
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = graph.invoke({"question": inputs["question"], "patient_id": inputs["patient_id"]}, cfg)
        while state.get("__interrupt__"):
            state = graph.invoke(Command(resume={"type": review, "message": "auto-review during evaluation"}), cfg)
        # Refused / emergency routes never reach the agent: their trajectory is just the request and the reply.
        messages = state.get("trajectory") or [{"role": "user", "content": inputs["question"]},
                                               {"role": "assistant", "content": state["answer"].get("answer", "")}]
        return {"answer": state["answer"], "contexts": state.get("contexts") or [],
                "tool_calls": state.get("tool_calls") or [], "messages": messages, "path": state.get("path") or []}

    return workflow


SYSTEMS = {"rag": make_rag_target, "agent": make_agent_target, "workflow": make_workflow_target}


def run_system(system: str, *, model=None, evaluators=None, max_concurrency: int = 5, num_repetitions: int = 1):
    """Run one milestone system over the golden dataset as a scoped experiment, e.g. `agent-jane-doe-…`."""
    from langsmith import evaluate

    seed_dataset()
    return evaluate(
        SYSTEMS[system](model=model),
        data=dataset_name(),
        evaluators=evaluators or (ANSWER_EVALUATORS + BEHAVIOR_EVALUATORS),
        experiment_prefix=scoped(system),
        metadata={"system": system, "retrieval_config": config_summary(RAG_CONFIG)},
        max_concurrency=max_concurrency,
        num_repetitions=num_repetitions,
    )


# --------------------------------------------------------------------------------------------------
# Retrieval-config experiments (optional notebook: improve_retrieval)
# --------------------------------------------------------------------------------------------------


def make_target(config: RetrievalConfig, model=None):
    """A target for `evaluate()`: one agent invoke per example, the session patient passed as context."""
    agent = build_rag_agent(config, model=model)
    build_vectorstore(config)  # index up front, not inside the first concurrent example

    def target(inputs: dict) -> dict:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": inputs["question"]}]},
            context={"patient_id": inputs["patient_id"]},
        )
        answer = result.get("structured_response")
        contexts = [m.content for m in result["messages"] if getattr(m, "type", "") == "tool"]
        if answer is None:
            payload = {"disposition": "answered", "answer": str(result["messages"][-1].content),
                       "citations": [], "safety_event": False, "safety_event_summary": None}
        else:
            payload = answer.model_dump()
        return {"answer": payload, "contexts": contexts}

    target.__name__ = f"chart_qa_{config.name.replace('-', '_')}"
    return target


def run_experiment(config: RetrievalConfig, *, name: str | None = None, max_concurrency: int = 4, model=None,
                   evaluators=None, num_repetitions: int = 1):
    """Run the golden dataset through `config` as a scoped LangSmith experiment, e.g. `baseline-jane-doe-…`."""
    from langsmith import evaluate

    seed_dataset()
    return evaluate(
        make_target(config, model=model),
        data=dataset_name(),
        evaluators=evaluators or ALL_EVALUATORS,
        experiment_prefix=scoped(name or config.name),
        metadata={"retrieval_config": config_summary(config)},
        max_concurrency=max_concurrency,
        num_repetitions=num_repetitions,
    )


def scorecard(results) -> dict:
    """{'overall': {metric: mean}, 'by_category': {category: {metric: mean}}, 'rows': [...]}"""
    overall, by_cat, rows = defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    for item in results:
        ex = item["example"]
        cat = (ex.outputs or {}).get("category", "?")
        row = {"question": ex.inputs["question"], "category": cat, "failure_mode": (ex.outputs or {}).get("failure_mode")}
        for r in item["evaluation_results"]["results"]:
            if r.score is None:
                continue
            val = float(r.score)
            overall[r.key].append(val)
            by_cat[cat][r.key].append(val)
            row[r.key] = val
            row[f"{r.key}_comment"] = r.comment
        rows.append(row)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    order = lambda keys: [k for k in METRICS if k in keys] + sorted(k for k in keys if k not in METRICS)  # noqa: E731
    return {
        "overall": {k: mean(overall[k]) for k in order(overall)},
        "by_category": {c: {k: mean(v[k]) for k in order(v)} for c, v in sorted(by_cat.items())},
        "rows": rows,
    }


def print_scorecard(card: dict, title: str = "") -> None:
    if title:
        print(f"\n📊 {title}")
    print("  " + "  ".join(f"{m:>20s}" for m in card["overall"]))
    print("  " + "  ".join(f"{v:>20.2f}" for v in card["overall"].values()))
    print("\n  by category:")
    for cat, scores in card["by_category"].items():
        print(f"    {cat:22s} " + "  ".join(f"{k}={v:.2f}" for k, v in scores.items()))


def gate(card: dict, thresholds: dict[str, float]) -> list[str]:
    """Aggregate regression gate: the metrics that fall below their minimum (empty list = pass).
    Metrics that don't apply to the system (missing from its scorecard) are skipped."""
    return [f"{m}={card['overall'][m]:.2f} < {floor:.2f}" for m, floor in thresholds.items()
            if m in card["overall"] and card["overall"][m] is not None and card["overall"][m] < floor]


def regressions(baseline: dict, candidate: dict, metric: str = "correctness") -> list[dict]:
    """Per-example regression check: questions that PASSED in `baseline` and FAIL in `candidate`.
    An average can improve while individual cases break; this catches them."""
    def by_q(card):
        out = {}
        for r in card["rows"]:
            if r.get(metric) is not None:
                out.setdefault(r["question"], []).append(r[metric])
        return {q: sum(v) / len(v) for q, v in out.items()}

    before, after = by_q(baseline), by_q(candidate)
    return [{"question": q, "before": before[q], "after": after[q]}
            for q in before if q in after and before[q] >= 0.5 > after[q]]


def _cell(value) -> str:
    return f"{value:>16.2f}" if isinstance(value, (int, float)) else f"{'—':>16s}"


def compare(cards: dict[str, dict], metrics=None) -> None:
    """Side-by-side table: rows = metrics (+ failure-mode correctness), columns = experiments.
    '—' means the metric doesn't apply to that system (e.g. tool selection for the RAG pipeline)."""
    names = list(cards)
    metrics = metrics or [m for m in METRICS if any(m in cards[n]["overall"] for n in names)]
    print(f"{'metric':28s}" + "".join(f"{n:>16s}" for n in names))
    for m in metrics:
        print(f"{m:28s}" + "".join(_cell(cards[n]["overall"].get(m)) for n in names))
    for cat in ["failure_mode"]:
        print(f"{cat + ' · correctness':28s}" + "".join(
            _cell(cards[n]["by_category"].get(cat, {}).get("correctness")) for n in names))
