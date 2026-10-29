"""Source for notebooks/optional/deep_agent.ipynb (optional / Day 2).

Reuses the deep-agent half (Part B) of the former 04_agent notebook, kept in `nb04_agent.py`, with a preamble
that defines what Part A used to provide.
"""

import nb04_agent as legacy
from _cells import SETUP, code, md

FILENAME = "optional/deep_agent.ipynb"

_cells = legacy.CELLS
_start = next(i for i, c in enumerate(_cells) if c[0] == "md" and "# Part B" in c[1])
_traj = next(i for i, c in enumerate(_cells) if c[0] == "md" and "### Trajectory eval" in c[1])
# B1 … B9 (deep-agent experiment). The old trajectory eval of the HITL agent now lives in Lab 4.


def _edit(cell, *pairs):
    kind, text, todo = cell
    for a, b in pairs:
        assert a in text, a[:60]
        text = text.replace(a, b)
    return (kind, text, todo)


B9_INTRO = """
## B9. Evaluate: LLM judges and a code evaluator

We reuse the Lab 4 patterns with small edits, which shows how quickly you can stand up evals for a new architecture. The deep agent runs on your golden dataset (`scoped("chart-qa-golden")`):

| Evaluator | Style | Checks |
|---|---|---|
| `correctness` | **LLM judge** (OpenEvals) | matches the reference, including the right decline / not-in-record / safety disposition |
| `pii_safety` | **LLM judge** | no raw names, MRNs or DOBs in the answer |
| `delegations` | **code** | counts `task` hand-offs in the run tree (the deep-agent signal) |

> ⚠️ `evaluate()` grades the **remote** dataset. If you edited the JSON, run `seed_dataset(refresh=True)` first.
"""


def _rewrite(i, c):
    if i == 0:
        return _edit(c, ("---\n# Part B · Deep agent\n\n", ""), ("Part A used `create_agent`", "Lab 2 used `create_agent`"))
    if "## B5. Guardrail · Mask" in c[1]:
        return _edit(c, ("detectors from A4 now", "detectors from `utils/phi.py` (Lab 3's `mask_phi` node) now"))
    if "## B9. Evaluate" in c[1]:
        return ("md", B9_INTRO, None)
    return c


_part_b = [_rewrite(i, c) for i, c in enumerate(_cells[_start:_traj])]

CELLS = [
md("""
# Optional · Deep agent: a coordinator with per-source subagents

**Northfield Health chart assistant** (synthetic data only) · **Optional / Day 2** · **Time box:** 45–60 min

Lab 2 built a single agent with two tools; Lab 3 put it inside a controlled workflow. A **deep agent** is the other direction: a managed **harness** (planning, a filesystem, delegation, context management) where a coordinator hands each data source to its own **subagent**. Each section adds **one keyword argument**, so every idea shows up on its own in the graph and in the trace.

| Section | Concept | What's added |
|---|---|---|
| B1 Simple agent → harness | thin `create_agent` vs the managed harness | `create_deep_agent(model=…)` |
| B2 Ground it | one grounding tool | `tools=[search_chart_notes]` |
| B3 Delegate | one subagent per data source | `subagents=[notes_researcher, structured_lookup]` |
| B4 Memory | short-term vs long-term (**no patient data**) | `checkpointer=` · `backend=` · `store=` |
| B5 Guardrail · Mask | PHI masking on every surface + audit | `middleware=[PIIMiddleware, …]` |
| B6 Guardrail · Policy | refusal + safety escalation, typed | `response_format=ChartAnswer` |
| B7 Guardrail · HITL | approve before filing | `interrupt_on={"file_safety_event": True}` |
| B8 Compose | the complete deep agent | *(all of the above)* |
| B9 Evaluate | judges and a code evaluator | `evaluate(...)` |

> 🧭 Everything is **synthetic / fictional**. `file_safety_event` writes to an in-memory list only. Completed versions live in `solutions/optional/deep_agent.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),
md("Shared pieces from Labs 1–3: the model, the two chart tools, the typed answer, and a few helpers."),
code('''import uuid

from IPython.display import Image, display
from langchain.agents import create_agent
from langgraph.types import Command

from chart_qa_graph import ChartAnswer, file_safety_event, lookup_structured_record, make_search_tool, system_prompt
from utils.models import get_model
from utils.retrieval import IMPROVED_V2
from utils.trace import invoke_traced, show

model = get_model()
search_chart_notes = make_search_tool(IMPROVED_V2)  # bigger chunks + query rewriting (see optional/improve_retrieval)
SAFETY_Q = "Were there any medication administration problems during NH-1004's May 2025 stay? If so, file it for patient-safety review."
ctx = {"patient_id": "NH-1004"}


def show_graph(graph) -> None:
    """Render the graph; middleware node names can break the remote renderer, so fall back to a node list."""
    g = graph.get_graph()
    try:
        display(Image(g.draw_mermaid_png()))
    except Exception as exc:
        print(f"(diagram unavailable: {type(exc).__name__}). Nodes:", ", ".join(sorted(str(n) for n in g.nodes)))'''),
*_part_b,
code('''try:
    from langsmith import Client

    print("🔗 experiment:", Client().read_project(project_name=deep_results.experiment_name).url)
except Exception:
    pass  # evaluate() also prints the result URL'''),
md("""
## What you just learned

| Concept | API |
|---|---|
| Deep-agent harness | `create_deep_agent(model=…)`: planner, filesystem, `task`, summarization built in |
| Delegation per data source | `subagents=[notes_researcher, structured_lookup]` |
| Short-term vs long-term (non-PHI) memory | `checkpointer=` · `CompositeBackend(routes={"/memories/": StoreBackend(…)})` |
| Masking on every surface + audit | `middleware=[PIIMiddleware(…), @wrap_tool_call]` |
| Typed policy outcome | `response_format=ChartAnswer` |
| Approve before filing | `interrupt_on={"file_safety_event": True}` + `Command(resume={interrupt_id: {"decisions": […]}})` |
| Evals for a new architecture | Lab 4 templates + a run-tree code evaluator (`delegations`) |

**Compare:** Lab 3's workflow *guarantees* the order of steps; the deep agent *plans* its own. Same tools, same guardrails, different trade-off between control and autonomy.
"""),
]
