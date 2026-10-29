"""Source for notebooks/04_evaluate.ipynb (Milestone 4 · Evaluation)."""

from _cells import SETUP, code, md

FILENAME = "04_evaluate.ipynb"

CITATION_SOLUTION = '''import re

from utils.corpus import documents_by_id

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
            if pid not in doc_id:
                problems.append(f"{doc_id}: structured record of another patient")
            continue
        doc = documents_by_id().get(doc_id)
        if doc is None:
            problems.append(f"{doc_id}: no such document")
        elif doc.patient_id != pid:
            problems.append(f"{doc_id}: belongs to {doc.patient_id}, not {pid}")
        elif not _quote_in(c.get("quote", ""), doc.body):
            problems.append(f"{doc_id}: quote not found in document")
    if reference_outputs["category"] in ANSWERABLE and not cites:
        problems.append("no citations")
    return {"key": "citation_validity", "score": not problems, "comment": "; ".join(problems) or f"{len(cites)} valid"}'''

CITATION_TODO = '''from utils.corpus import documents_by_id

ANSWERABLE = {"fact_lookup", "multi_doc_synthesis", "failure_mode"}


def citation_validity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Pass only if every citation in outputs["answer"]["citations"]:
      1. names a doc_id that exists          → documents_by_id().get(doc_id)
         (citations starting with "fhir/" point at the structured record: just check the patient ID is in it)
      2. belongs to inputs["patient_id"]      → doc.patient_id
      3. has a quote that appears in doc.body (normalize case/whitespace first)
    and answerable categories (ANSWERABLE) have at least one citation.
    """
    problems = []
    # TODO 1: implement the checks above, appending a short reason to `problems` for each failure.
    return {"key": "citation_validity", "score": not problems, "comment": "; ".join(problems) or "ok"}'''

TOOL_SOLUTION = '''DATA_TOOLS = {"search_chart_notes", "lookup_structured_record"}


def tool_selection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if "tool_calls" not in outputs:
        return {"key": "tool_selection", "score": None, "comment": "n/a: this system does not choose tools"}
    called = set(outputs["tool_calls"])
    expected = set(reference_outputs.get("expected_tools") or [])
    if expected:
        missing = expected - called
        return {"key": "tool_selection", "score": not missing, "comment": f"called {outputs['tool_calls']}, missing {sorted(missing)}"}
    if reference_outputs["category"] == "refusal" or ("path" in outputs and "answer" not in outputs["path"]):
        return {"key": "tool_selection", "score": None, "comment": "n/a: refused or not routed to the agent"}
    return {"key": "tool_selection", "score": bool(called & DATA_TOOLS), "comment": f"called {outputs['tool_calls'] or 'no tools'}"}'''

TOOL_TODO = '''DATA_TOOLS = {"search_chart_notes", "lookup_structured_record"}


def tool_selection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent call the right tool(s)?
      - no "tool_calls" in outputs (the RAG pipeline)          → score None (n/a)
      - reference_outputs["expected_tools"] is non-empty        → pass only if EVERY expected tool was called
      - category "refusal", or a workflow path without "answer" → score None (n/a)
      - otherwise                                               → pass if it called at least one DATA_TOOLS tool
                                                                  (never answer, or claim absence, from memory)
    """
    # TODO 2: implement
    return {"key": "tool_selection", "score": None, "comment": "not implemented"}'''

ROUTE_SOLUTION = '''def route_of(path: list[str]) -> str:
    if "refuse" in path:
        return "refuse"
    if "emergency" in path:
        return "emergency"
    return "answer_with_review" if "human_review" in path else "answer"


def route_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if "path" not in outputs:
        return {"key": "route_correctness", "score": None, "comment": "n/a: not a workflow"}
    route = route_of(outputs["path"])
    return {"key": "route_correctness", "score": route in reference_outputs["expected_routes"],
            "comment": f"route={route}, expected {reference_outputs['expected_routes']}"}'''

ROUTE_TODO = '''def route_of(path: list[str]) -> str:
    if "refuse" in path:
        return "refuse"
    if "emergency" in path:
        return "emergency"
    return "answer_with_review" if "human_review" in path else "answer"


def route_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Workflow only: pass if route_of(outputs["path"]) is in reference_outputs["expected_routes"].
    Return score None when outputs has no "path" (the RAG pipeline and the bare agent)."""
    # TODO 3: implement
    return {"key": "route_correctness", "score": None, "comment": "not implemented"}'''

CELLS = [
md("""
# 04 · Evaluate: answers *and* agent behavior

**Northfield Health chart assistant** (synthetic data only) · **Milestone 4 of 4** · **Time box:** 60 min

| Lab | Milestone | The chart assistant becomes… |
|---|---|---|
| 01 | RAG | a pipeline that retrieves chart passages and answers with citations |
| 02 | Agent | a tool-using agent that decides what to look up, and where |
| 03 | LangGraph | a controlled, stateful workflow with routing and human approval |
| **04** | **Evaluation** | **measured: answer correctness *and* agent behavior** ← you are here |

You now have three versions of the same app. Which is better, and *how* do you know? A final-answer score isn't enough for agents: an agent can get the right answer for the wrong reasons (guessing without looking), or a wrong answer because it picked the wrong tool. We grade **both**:

| Layer | Evaluator | Kind | Applies to |
|---|---|---|---|
| Answer | `correctness` | LLM judge (OpenEvals) | all three |
| Answer | `groundedness` | LLM judge (OpenEvals) | all three |
| Answer | `citation_validity` | code | all three |
| Answer | `refusal_correctness` | code | all three |
| Behavior | `tool_selection` | code | agent, workflow |
| Behavior | `trajectory_accuracy` | LLM judge (OpenEvals trajectory) | agent, workflow |
| Behavior | `route_correctness` | code | workflow |

**Goals.** By the end you can:
- seed a golden dataset in LangSmith with **reference answers, expected tools and expected routes**
- write LLM-as-judge and code evaluators, and **trajectory** evaluators that grade the tool calls themselves
- run the same dataset through the **pipeline, the agent and the workflow** as LangSmith experiments
- compare them side by side and trace a bad score back to its cause
- turn evals into **regression tests** (a CI gate), and place them in a loop that keeps agent quality up **over time**
- explain what LangSmith adds for **observability and governance**, and keep identifiers out of traces at the source

There are **3 `# TODO` cells** (three code evaluators). Completed versions live in `solutions/04_evaluate.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),

md("""
## 1. The golden dataset

`evals/dataset_chart_qa.json` has 25 questions. Each has a `reference_answer` and `reference_doc_ids` (to grade answers), plus **`expected_tools`** (to grade the agent) and **`expected_routes`** (to grade the workflow):

| Category | n | Tests |
|---|---|---|
| `fact_lookup` | 8 | single facts with dates and values |
| `multi_doc_synthesis` | 5 | summaries and trends across notes |
| `failure_mode` | 6 | hard cases: outdated med list, split lab table, shorthand, look-alike names, buried addendum, medication error |
| `not_in_record` | 4 | the right answer is "the record doesn't say" (after looking!) |
| `refusal` | 2 | decision requests that must be declined |

It's pushed to LangSmith as **`chart-qa-golden-<you>`**. Seeding is idempotent.
"""),
code('''from evals.seed_chart_qa import dataset_name, load_examples, seed_dataset
from utils.prompts import dataset_url

seed_dataset()
DATASET = dataset_name()
print("📊", DATASET, "→", dataset_url(DATASET))

examples = load_examples()
for ex in examples[::4]:
    o = ex["outputs"]
    print(f"[{o['category']:19s}] tools={o['expected_tools']!s:30s} routes={o['expected_routes']}  {ex['inputs']['question'][:60]}")'''),

md("""
## 2. Three systems, one output shape

Each milestone is wrapped as an eval **target**: a function from a dataset example's `inputs` to outputs. They all return the same shape, so one set of evaluators grades all three. Behavior keys only appear when the system has that behavior:

| Key | RAG | Agent | Workflow |
|---|---|---|---|
| `answer` (ChartAnswer dict), `contexts` | ✅ | ✅ | ✅ |
| `tool_calls`, `messages` (the trajectory) | | ✅ | ✅ |
| `path` (nodes visited) | | | ✅ |

The workflow target answers the human-review interrupt automatically (approve), so evals don't wait on a person.
"""),
code('''from evals.harness import make_agent_target, make_rag_target, make_workflow_target

targets = {"rag": make_rag_target(), "agent": make_agent_target(), "workflow": make_workflow_target()}

ex = next(e for e in examples if e["outputs"].get("failure_mode") == "medication_error")
print("Q:", ex["inputs"]["question"], "\\n")
dry = {}
for name, target in targets.items():
    dry[name] = target(ex["inputs"])
    out = dry[name]
    print(f"{name:9s} disposition={out['answer']['disposition']:22s} tool_calls={out.get('tool_calls', '—')!s:45s} path={out.get('path', '—')}")'''),

md("""
## 3. Answer evaluators

**Correctness** is an LLM judge built with OpenEvals: a prebuilt prompt plus a judge model (`LABS_JUDGE_MODEL`, configured separately from the model under test).
"""),
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
md("`groundedness` follows the same pattern with `RAG_GROUNDEDNESS_PROMPT` (is every claim supported by what was retrieved?), and `refusal_correctness` is a short code check on the disposition and safety flag. Read both in `evals/evaluators.py`."),
code('''from evals.evaluators import groundedness, refusal_correctness'''),
md("### TODO 1: `citation_validity` (code)\n\nNo LLM: a citation is valid only if the quoted text is really in that document, for that patient. This is what makes \"grounded\" checkable."),
code(CITATION_SOLUTION, todo=CITATION_TODO),

md("""
## 4. Agent-behavior evaluators

**Tool selection** grades *which* tools were called against the example's `expected_tools`. With no specific expectation, an agent must still look **something** up before answering or claiming absence.

### TODO 2: `tool_selection` (code)
"""),
code(TOOL_SOLUTION, todo=TOOL_TODO),
md("""
**Trajectory accuracy** grades the *whole sequence* of steps with an LLM judge: did each step follow logically from the last, make progress, and avoid wasted calls? OpenEvals ships the prompt.
"""),
code('''from openevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

_trajectory_judge = create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, feedback_key="trajectory_accuracy",
                                                   judge=get_judge_model())


def trajectory_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if not outputs.get("messages"):
        return {"key": "trajectory_accuracy", "score": None, "comment": "n/a: no agent trajectory"}
    res = _trajectory_judge(outputs=outputs["messages"])
    return {"key": "trajectory_accuracy", "score": res["score"], "comment": res["comment"]}'''),
md("### TODO 3: `route_correctness` (code)\n\nThe workflow's `path` records every node it visited. Collapse it into a route and compare with the expected one(s)."),
code(ROUTE_SOLUTION, todo=ROUTE_TODO),

md("## 5. Dry-run: grade the one example from section 2 with every evaluator\n\nCheap to run, and it catches evaluator bugs before you spend money on 75 runs."),
code('''EVALUATORS = [correctness, groundedness, citation_validity, refusal_correctness,
              tool_selection, trajectory_accuracy, route_correctness]

print(f"{'evaluator':22s}" + "".join(f"{n:>12s}" for n in dry))
for ev in EVALUATORS:
    scores = [ev(ex["inputs"], dry[n], ex["outputs"])["score"] for n in dry]
    print(f"{ev.__name__:22s}" + "".join(f"{('—' if s is None else str(s)):>12s}" for s in scores))'''),

md("""
## 6. Run the three experiments

Same dataset, same evaluators, one experiment per system, each named with your scoped prefix (`rag-<you>-…`, `agent-<you>-…`, `workflow-<you>-…`). This takes a few minutes; open the first experiment link while the others run.
"""),
code('''from langsmith import evaluate

from evals.harness import scorecard
from utils.workshop import scoped

results, cards = {}, {}
for name, target in targets.items():
    results[name] = evaluate(
        target,
        data=DATASET,
        evaluators=EVALUATORS,
        experiment_prefix=scoped(name),
        metadata={"system": name},
        max_concurrency=5,
    )
    cards[name] = scorecard(results[name])'''),

md("## 7. Compare side by side"),
code('''from evals.harness import compare

compare(cards)  # '—' = the metric doesn't apply to that system'''),
code('''for name, card in cards.items():
    print(f"\\n{name}")
    for cat, scores in card["by_category"].items():
        print(f"   {cat:20s}", {k: round(v, 2) for k, v in scores.items() if k in ("correctness", "tool_selection", "route_correctness")})'''),
code('''from langsmith import Client

client = Client()
ids = [str(client.read_project(project_name=r.experiment_name).id) for r in results.values()]
ds = client.read_dataset(dataset_name=DATASET)
print("🔗 compare in the UI:", f"{client.read_project(project_name=results['rag'].experiment_name).url.split('/projects/')[0]}/datasets/{ds.id}/compare?selectedSessions={','.join(ids)}")'''),
md("""
**Reading the table.** Typical patterns:
- **Correctness** rises from RAG → agent on the `failure_mode` questions the pipeline can't reach (structured lookups, shorthand notes).
- **Refusal correctness** is most reliable for the workflow: decision requests never reach a model that could be talked into answering.
- **Tool selection** below 1.0 points at specific questions where the agent skipped the structured record, or answered without looking.
"""),

md("## 8. From a bad score to its cause\n\nEvery failed row has a judge or code comment. Start there, then open that run's trace in the experiment."),
code('''for name in ("agent", "workflow"):
    for row in cards[name]["rows"]:
        bad = [k for k in ("tool_selection", "route_correctness", "correctness") if row.get(k) == 0]
        if bad:
            print(f"❌ {name:8s} [{row['failure_mode'] or row['category']}] {row['question'][:70]}")
            for k in bad:
                print(f"     {k}: {str(row.get(k + '_comment'))[:170]}")'''),

md("""
---
## 9. Evaluation as a discipline: from experiments to regression tests

An experiment answers *"which version is better today?"*. Production needs a second question answered on **every change** (prompt, tool, retrieval, model upgrade, graph edit): *"did this break anything that used to work?"* Two complementary gates:

| Gate | Catches | How |
|---|---|---|
| **Aggregate thresholds** | the system got worse overall | each metric must stay above a floor (`evals/thresholds.json`) |
| **Per-example regressions** | an average improved, but specific cases broke | compare example by example against a baseline experiment |
"""),
code('''import json

from evals.harness import gate, regressions
from evals.seed_chart_qa import ROOT

thresholds = json.loads((ROOT / "evals" / "thresholds.json").read_text())
for name, card in cards.items():
    problems = gate(card, thresholds.get(name, {}))
    print(f"{'✅ PASS' if not problems else '⛔ FAIL'}  {name:9s}", "; ".join(problems))'''),
md("**What did each architecture change cost us?** Questions the previous version answered correctly that the next one got wrong:"),
code('''for before, after in [("rag", "agent"), ("agent", "workflow")]:
    lost = regressions(cards[before], cards[after])
    print(f"{before} → {after}: {len(lost)} per-example regression(s)")
    for r in lost:
        print(f"  ↘ {r['question'][:90]}  ({r['before']:.0%} → {r['after']:.0%})")'''),
md("""
Even when the averages go up, you'll often see a few cases flip from pass to fail. Some are real regressions, some are run-to-run noise (LLM outputs vary). That's why a release gate should **re-run a flipped case** (or use `num_repetitions`) before blocking, and why a clinician should look at the ones that persist. Don't stop at the average: a version can score higher overall while breaking a case someone relies on.
"""),
md("""
In CI, the same check is one command. It runs a scoped experiment and exits non-zero on a regression, so a failing build links straight to the traces:

```bash
uv run python scripts/eval_gate.py workflow agent      # exit 1 if any metric drops below evals/thresholds.json
```

**Pin the dataset version** the gate runs against, so a score change means the *system* changed, not the questions:
"""),
code('''from datetime import datetime, timezone

version_tag = "workshop-baseline"
client.update_dataset_tag(dataset_name=DATASET, as_of=datetime.now(timezone.utc), tag=version_tag)
pinned = list(client.list_examples(dataset_name=DATASET, as_of=version_tag))
print(f"tagged {DATASET} as '{version_tag}' ({len(pinned)} examples). Evaluate against it with:")
print(f"  evaluate(target, data=client.list_examples(dataset_name=DATASET, as_of='{version_tag}'), ...)")'''),

md("""
## 10. Agent quality over time

Offline evals before release are one half. The other half runs on **live traffic**, and every reviewed failure becomes a new test:

```text
   ┌──────────────── offline ────────────────┐        ┌──────────────── online ─────────────────┐
   golden dataset → experiments → CI gate → release → traces → online evaluators → review queue
          ▲                                                                              │
          └──────────────── reviewed failures become new dataset examples ◀──────────────┘
```

| Layer | Question | Tooling (this repo) |
|---|---|---|
| Evaluators | what does "good" mean here? | OpenEvals judges + code evaluators (`evals/evaluators.py`) |
| Harness | run a system over a dataset, reproducibly | `langsmith.evaluate` + targets (`evals/harness.py`) |
| Offline experiments | which version is better? | scoped experiments + side-by-side compare (this lab) |
| Regression testing | did a change break something? | `scripts/eval_gate.py`, per-example `regressions()`, pinned dataset versions |
| Online evaluation | is production still good? | run-rule judges on live traces (`notebooks/optional/observe.ipynb`) |
| Human review | what do experts say about the hard cases? | annotation queues → corrected examples back into the dataset |

How enterprises should think about it: **quality is a trend, not a launch checklist.** Version the dataset, the evaluators *and* the thresholds alongside the code; grow the dataset from production failures; review judge disagreements with clinicians; and raise thresholds as the system improves (never lower them silently).
"""),

md("""
---
## 11. Observability and governance

Everything you did today left a trace in LangSmith. Here's what that buys an enterprise over a home-grown setup (logs + a metrics dashboard + a spreadsheet of test cases):

| Capability | Home-grown | LangSmith |
|---|---|---|
| See *why* an agent did something | flat logs; reconstruct the call tree yourself | nested traces of every model call, tool call, subagent and graph node, with inputs, outputs, tokens, latency, cost |
| Evaluate | ad-hoc scripts | datasets, experiments, side-by-side comparison, LLM-as-judge and code evaluators |
| Monitor production | custom dashboards | online evaluators, dashboards, alerts, annotation queues |
| Change prompts and agent instructions safely | redeploy | Prompt Hub and Context Hub: versioned, reviewable, rollback |
| Access control and audit | build it | workspaces, role-based access, SSO/SCIM, audit logs (plan-dependent) |
| Data lifecycle | build it | trace retention (TTL) settings, per-project isolation |

**Governance demo: keep identifiers out of the trace store at the source.** Lab 3 masks the *question*. But identifiers can also appear in tool outputs (patient names in chart headers) or slip past an app-level check. A LangSmith client with an **anonymizer** redacts every trace payload **before it leaves the process**, using the same detectors:
"""),
code('''import time
import uuid

from langsmith import traceable
from langsmith.anonymizer import create_anonymizer
from langsmith.run_helpers import tracing_context

from utils.phi import mask_text

redacting_client = Client(anonymizer=create_anonymizer(lambda text, path: mask_text(text)[0]))


@traceable(name="governance-demo")
def echo(question: str) -> str:
    return f"received: {question}"


run_id = uuid.uuid4()
with tracing_context(client=redacting_client, project_name=LAB["langsmith_project"], enabled=True):
    echo("Pt Walter Ingersoll, DOB 11/02/1948, MRN 00482913: current anticoagulant?", langsmith_extra={"run_id": run_id})
redacting_client.flush()

for _ in range(10):  # the run takes a moment to become readable
    try:
        stored = client.read_run(run_id)
        break
    except Exception:
        time.sleep(2)
print("what LangSmith stored:", stored.inputs)'''),
md("""
**Deployment posture for regulated environments.** Decide where each kind of data lives, then pick the deployment that matches:

| Concern | Options |
|---|---|
| Where traces live | LangSmith cloud (US or EU region), or **self-hosted LangSmith** in your own cloud account or Kubernetes cluster (enterprise) |
| Where agents run | your infrastructure (as today: `langgraph dev`, your containers), LangSmith Deployment cloud, or hybrid/self-hosted deployment |
| What leaves the process | client-side anonymizers (above), `hide_inputs` / `hide_outputs`, app-boundary masking (Lab 3's `mask_phi`) |
| Which models see data | route through your approved gateway or private model endpoints (`LABS_*` switches, `LABS_OPENAI_BASE_URL`) |
| Who can see what | separate workspaces/projects per environment and team; least-privilege roles; audit logs |
| How long it's kept | retention (TTL) per project; short-lived for anything sensitive |
| Compliance | confirm current certifications and BAA terms with your LangChain account team and your compliance office |

The same code you wrote today runs under every posture: only configuration changes.
"""),

md("""
## What you just learned

- A golden dataset can carry **behavior references** (`expected_tools`, `expected_routes`), not just reference answers.
- **Answer** evaluators (correctness, groundedness, citations, refusals) and **behavior** evaluators (tool selection, trajectory, route) answer different questions. Agents need both.
- **Code evaluators** grade exact things cheaply and deterministically; **LLM judges** (OpenEvals) grade fuzzy things, including a whole trajectory.
- One output shape + `score=None` for "not applicable" lets one evaluator list grade a pipeline, an agent and a workflow, and `evaluate(..., experiment_prefix=scoped(name))` makes them comparable side by side.
- **Regression testing** turns evals into a release gate: aggregate thresholds (`scripts/eval_gate.py`) plus per-example regressions, against a **pinned dataset version**.
- **Quality over time** is a loop: offline experiments → CI gate → online evaluators on live traces → human review → new dataset examples.
- **Governance**: LangSmith gives you nested traces, evals, monitoring, versioned prompts/instructions and access control; a client-side **anonymizer** keeps identifiers out of the trace store, and the same code runs cloud, hybrid or self-hosted.

**The day:** RAG (retrieval + grounding) → agent (tool selection + loop) → LangGraph (state, routing, human approval) → evaluation (answers + behavior).

**Day 2 / optional notebooks** in `notebooks/optional/`: `improve_retrieval` (tune RAG against these evals), `deep_agent` (a coordinator with per-source subagents), `observe` (online evals and human review queues).
"""),
]
