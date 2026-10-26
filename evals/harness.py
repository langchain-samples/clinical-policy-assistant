"""Eval targets, experiment runner, and scorecards shared by notebooks 02/03 and `scripts/run_experiments.py`."""

from __future__ import annotations

from collections import defaultdict

from chart_qa_graph import build_rag_agent
from evals.evaluators import ALL_EVALUATORS
from evals.seed_chart_qa import dataset_name, seed_dataset
from utils.retrieval import RetrievalConfig, build_vectorstore, config_summary
from utils.workshop import scoped

METRICS = ["correctness", "groundedness", "retrieval_relevance", "citation_validity", "refusal_correctness"]


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


def compare(cards: dict[str, dict], metrics=None) -> None:
    """Side-by-side table: rows = metrics (+ failure-mode correctness), columns = experiments."""
    metrics = metrics or METRICS
    names = list(cards)
    print(f"{'metric':28s}" + "".join(f"{n:>16s}" for n in names))
    for m in metrics:
        print(f"{m:28s}" + "".join(f"{(cards[n]['overall'].get(m) or 0):>16.2f}" for n in names))
    for cat in ["failure_mode"]:
        print(f"{cat + ' · correctness':28s}" + "".join(
            f"{(cards[n]['by_category'].get(cat, {}).get('correctness') or 0):>16.2f}" for n in names))
