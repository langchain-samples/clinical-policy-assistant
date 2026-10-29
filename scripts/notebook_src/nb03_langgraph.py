"""Source for notebooks/03_langgraph.ipynb (Milestone 3 · LangGraph workflow)."""

from _cells import SETUP, code, md

FILENAME = "03_langgraph.ipynb"

STATE_SOLUTION = '''import operator
from typing import Annotated

from typing_extensions import TypedDict


class ChartState(TypedDict, total=False):
    question: str                            # the clinician's request (masked in place by mask_phi)
    patient_id: str | None                   # the chart that is open
    masked: list[str]                        # identifier kinds removed from the question
    intent: str                              # chart_question | clinical_decision | emergency
    answer: dict                             # ChartAnswer as a dict
    tool_calls: list[str]                    # tools the agent called inside the `answer` node
    review: dict | None                      # the human reviewer's decision
    path: Annotated[list[str], operator.add]  # every node appends its name; the reducer concatenates


class WorkflowInput(TypedDict):  # what a caller (or Studio) must provide
    question: str
    patient_id: str | None'''

STATE_TODO = '''import operator
from typing import Annotated

from typing_extensions import TypedDict


class ChartState(TypedDict, total=False):
    question: str                            # the clinician's request (masked in place by mask_phi)
    patient_id: str | None                   # the chart that is open
    masked: list[str]                        # identifier kinds removed from the question
    intent: str                              # chart_question | clinical_decision | emergency
    answer: dict                             # ChartAnswer as a dict
    tool_calls: list[str]                    # tools the agent called inside the `answer` node
    review: dict | None                      # the human reviewer's decision
    # TODO 1: add `path`, a list[str] that ACCUMULATES across nodes instead of being overwritten.
    #         Hint: Annotated[list[str], operator.add]


class WorkflowInput(TypedDict):  # what a caller (or Studio) must provide
    question: str
    patient_id: str | None'''

ROUTER_SOLUTION = '''from typing import Literal


def route_intent(state: ChartState) -> Literal["refuse", "emergency", "answer"]:
    if state["intent"] == "clinical_decision":
        return "refuse"
    if state["intent"] == "emergency":
        return "emergency"
    return "answer"'''

ROUTER_TODO = '''from typing import Literal


def route_intent(state: ChartState) -> Literal["refuse", "emergency", "answer"]:
    # TODO 2: return the NAME of the next node:
    #   "refuse" for clinical_decision, "emergency" for emergency, otherwise "answer".
    ...'''

REVIEW_SOLUTION = '''from langgraph.types import interrupt


def human_review(state: ChartState) -> dict:
    cites = state["answer"].get("citations") or []
    decision = interrupt({                       # ⏸️ the run stops here and is saved by the checkpointer
        "action": "file_safety_event",
        "patient_id": state.get("patient_id"),
        "summary": state["answer"].get("safety_event_summary") or state["answer"]["answer"],
        "source_doc_id": cites[0]["doc_id"] if cites else "unknown",
    })
    return {"review": decision, "path": ["human_review"]}  # ▶️ resumes here with the reviewer's decision'''

REVIEW_TODO = '''from langgraph.types import interrupt


def human_review(state: ChartState) -> dict:
    cites = state["answer"].get("citations") or []
    draft = {
        "action": "file_safety_event",
        "patient_id": state.get("patient_id"),
        "summary": state["answer"].get("safety_event_summary") or state["answer"]["answer"],
        "source_doc_id": cites[0]["doc_id"] if cites else "unknown",
    }
    # TODO 3: pause for a human: `decision = interrupt(draft)`. When the run is resumed with
    #         Command(resume={...}), interrupt() returns that value. Then return
    #         {"review": decision, "path": ["human_review"]}.
    ...'''

CELLS = [
md("""
# 03 · LangGraph: a controlled, stateful workflow

**Northfield Health chart assistant** (synthetic data only) · **Milestone 3 of 4** · **Time box:** 75 min

| Lab | Milestone | The chart assistant becomes… |
|---|---|---|
| 01 | RAG | a pipeline that retrieves chart passages and answers with citations |
| 02 | Agent | a tool-using agent that decides what to look up, and where |
| **03** | **LangGraph** | **a controlled, stateful workflow with routing and human approval** ← you are here |
| 04 | Evaluation | measured: answer correctness *and* agent behavior |

Lab 2's agent decides everything. That's great for *finding* information, and wrong for things that must **always** or **never** happen:

| Requirement | Agent alone | Workflow |
|---|---|---|
| Identifiers are masked before any model sees them | hope the prompt works | a `mask_phi` node that **always runs first** |
| Decision requests ("increase the dose?") are refused | the model usually declines | a router sends them to `refuse`; the agent is **never called** |
| A safety report is filed only after human approval | middleware on one tool | an explicit `human_review` node that **pauses the graph** |

You'll build this graph, with the Lab 2 agent as **one node** inside it:

```text
START → mask_phi → classify ─┬─(clinical_decision)→ refuse ──────────────────────────→ END
                             ├─(emergency)────────→ emergency ───────────────────────→ END
                             └─(chart_question)───→ answer ─┬─(no safety event)──────→ END
                                                  (Lab 2 agent)└─(safety event)→ human_review ─┬─(approve)→ file_event → END
                                                                                ⏸️ interrupt    └─(reject)─────────────→ END
```

**Goals.** By the end you can:
- define typed **state** and a **reducer**
- write **nodes** (functions: state in → partial update out) and connect them with **edges**
- add **conditional routing** with a router function
- pause a run with **`interrupt()`**, inspect the saved state, and **resume** it with a human decision (approve / reject)
- explain how a **checkpointer** makes a graph stateful across calls, and resume a run after a **failure** (durable execution)
- place each building block (control loop, state, memory, durability, HITL) on the **framework vs application** line, and open the graph in **Studio**

There are **3 `# TODO` cells**. Completed versions live in `solutions/03_langgraph.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),

md("""
## 1. State: what flows through the graph

Every node reads the **state** and returns a **partial update**. By default an update *overwrites* a key. A **reducer** changes that: `Annotated[list[str], operator.add]` *appends*, so each node can add its name to `path` and we get a record of the route the run took.

**TODO 1.** Add the `path` key with its reducer.
"""),
code(STATE_SOLUTION, todo=STATE_TODO),

md("""
## 2. Nodes: plain functions

Each node is a function `state → dict`. Nothing magic: you can call one directly to test it.
"""),
code('''from utils.phi import mask_text


def mask_phi(state: ChartState) -> dict:
    question, kinds = mask_text(state["question"])
    return {"question": question, "masked": kinds, "path": ["mask_phi"]}


mask_phi({"question": "Pt Walter Ingersoll, DOB 11/02/1948 (chart NH-1042): current anticoagulant?"})'''),
md("""
**`classify`** uses the model as a **router**: structured output with exactly three allowed intents. It's one small, cheap, well-defined decision, unlike handing the model the whole task.
"""),
code('''from chart_qa_graph import CLASSIFY_PROMPT, Intent
from utils.models import get_model

model = get_model()
router = model.with_structured_output(Intent)


def classify(state: ChartState) -> dict:
    verdict = router.invoke([{"role": "system", "content": CLASSIFY_PROMPT}, {"role": "user", "content": state["question"]}])
    return {"intent": verdict.intent, "path": ["classify"]}


for q in ["Should I increase NH-1042's Thrombelix dose?", "What is NH-1023's code status?", "NH-1023 is unresponsive, help!"]:
    print(f"{classify({'question': q})['intent']:18s} ← {q}")'''),
md("**`refuse`** and **`emergency`** are deterministic: no model call at all, so they can't be talked out of refusing."),
code('''from chart_qa_graph import EMERGENCY_TEXT, REFUSAL_TEXT, ChartAnswer


def _declined(text: str) -> dict:
    return ChartAnswer(disposition="declined_clinical_decision", answer=text, citations=[], safety_event=False).model_dump()


def refuse(state: ChartState) -> dict:
    return {"answer": _declined(REFUSAL_TEXT), "path": ["refuse"]}


def emergency(state: ChartState) -> dict:
    return {"answer": _declined(EMERGENCY_TEXT), "path": ["emergency"]}'''),
md("**`answer`** wraps the **Lab 2 agent** (`build_agent()` is the packaged version of what you built): the agent becomes one step of a larger, controlled process."),
code('''from chart_qa_graph import agent_tool_calls, answer_from_result, build_agent

agent = build_agent(model=model)


def answer(state: ChartState) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": state["question"]}]},
                          context={"patient_id": state.get("patient_id")})
    return {"answer": answer_from_result(result), "tool_calls": agent_tool_calls(result["messages"]), "path": ["answer"]}'''),

md("""
## 3. Conditional routing

A **router** is a function that reads the state and returns the **name of the next node**. LangGraph calls it after `classify` and follows the edge it picks.

**TODO 2.** Write `route_intent`.
"""),
code(ROUTER_SOLUTION, todo=ROUTER_TODO),

md("""
## 4. Wire the graph (first version)

`add_node` registers each function, `add_edge` adds fixed transitions, and `add_conditional_edges(source, router, [possible targets])` adds the branch. `compile()` turns the builder into a runnable graph, the same kind of object `create_agent` returned in Lab 2.
"""),
code('''from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph


def show_graph(graph) -> None:
    g = graph.get_graph()
    try:
        display(Image(g.draw_mermaid_png()))
    except Exception as exc:  # offline renderer: print the structure instead
        print(f"(diagram unavailable: {type(exc).__name__})")
        for e in g.edges:
            print(f"  {e.source} -> {e.target}" + ("  [conditional]" if e.conditional else ""))


builder = StateGraph(ChartState, input_schema=WorkflowInput)
for fn in (mask_phi, classify, refuse, emergency, answer):
    builder.add_node(fn.__name__, fn)
builder.add_edge(START, "mask_phi")
builder.add_edge("mask_phi", "classify")
builder.add_conditional_edges("classify", route_intent, ["refuse", "emergency", "answer"])
builder.add_edge("refuse", END)
builder.add_edge("emergency", END)
builder.add_edge("answer", END)
v1 = builder.compile()
show_graph(v1)'''),
code('''from utils.trace import invoke_traced, show

for q, pid in [("Should I increase NH-1042's Thrombelix dose now that his kidney function changed?", "NH-1042"),
               ("Pt Corinne Abernathy, DOB 12/05/1959: what is the current anticoagulant?", "NH-1017")]:
    out = invoke_traced(v1, {"question": q, "patient_id": pid})
    print("path  :", out["path"], "| masked:", out.get("masked"))
    show(out["answer"])
    print()'''),
md("""
> 💡 **Open the first trace.** The decision request went `mask_phi → classify → refuse`: **no agent, no tools, no chance** to answer it. The second trace shows the identifiers already masked when `classify` and the agent ran.
"""),

md("""
## 5. Human approval before filing a safety event

Filing a patient-safety report is a **side effect**, so a person must approve it. Three more pieces:

1. `route_safety`: after `answer`, go to `human_review` if the answer flags a safety event, otherwise end.
2. `human_review`: calls **`interrupt(payload)`**. The graph **stops**, saves its state, and returns the payload to the caller. When resumed, `interrupt()` returns the human's decision.
3. `file_event`: runs **only** on approval. It calls `file_safety_event` directly (no model involved).

**TODO 3.** Write `human_review`.
"""),
code(REVIEW_SOLUTION, todo=REVIEW_TODO),
code('''from chart_qa_graph import SAFETY_EVENT_LOG, file_safety_event


def route_safety(state: ChartState) -> Literal["human_review", "__end__"]:
    return "human_review" if state["answer"].get("safety_event") else END


def route_review(state: ChartState) -> Literal["file_event", "__end__"]:
    return "file_event" if (state.get("review") or {}).get("type") == "approve" else END


def file_event(state: ChartState) -> dict:
    cites = state["answer"].get("citations") or []
    receipt = file_safety_event.invoke({
        "patient_id": state.get("patient_id") or "unknown",
        "event_type": "possible patient-safety event",
        "summary": state["answer"].get("safety_event_summary") or state["answer"]["answer"],
        "source_doc_id": cites[0]["doc_id"] if cites else "unknown",
    })
    return {"answer": {**state["answer"], "answer": state["answer"]["answer"] + f"\\n\\n✅ {receipt}"}, "path": ["file_event"]}'''),
md("""
**Rebuild with a checkpointer.** An interrupt needs somewhere to save the paused run: the **checkpointer** stores the state after every step, keyed by `thread_id`. That's what makes the graph *stateful* across calls.
"""),
code('''from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(ChartState, input_schema=WorkflowInput)
for fn in (mask_phi, classify, refuse, emergency, answer, human_review, file_event):
    builder.add_node(fn.__name__, fn)
builder.add_edge(START, "mask_phi")
builder.add_edge("mask_phi", "classify")
builder.add_conditional_edges("classify", route_intent, ["refuse", "emergency", "answer"])
builder.add_edge("refuse", END)
builder.add_edge("emergency", END)
builder.add_conditional_edges("answer", route_safety, ["human_review", END])
builder.add_conditional_edges("human_review", route_review, ["file_event", END])
builder.add_edge("file_event", END)
workflow = builder.compile(checkpointer=InMemorySaver())
show_graph(workflow)'''),

md("""
## 6. Run it: pause, inspect, resume

NH-1004's chart has a nursing note describing a medication error.
"""),
code('''import uuid

SAFETY_Q = "Were there any problems with medication administration during NH-1004's May 2025 stay?"
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

out = invoke_traced(workflow, {"question": SAFETY_Q, "patient_id": "NH-1004"}, config=thread)
print("path so far:", out["path"])
print("⏸️  paused with:", out["__interrupt__"][0].value if out.get("__interrupt__") else "(no interrupt: no safety event flagged)")'''),
md("**The paused run is saved.** `get_state` reads it back: the values so far and which node runs next."),
code('''snapshot = workflow.get_state(thread)
print("next node :", snapshot.next)
print("intent    :", snapshot.values.get("intent"))
print("tool calls:", snapshot.values.get("tool_calls"))
print("draft     :", snapshot.values["answer"]["safety_event_summary"])'''),
md("**▶️ Approve.** `Command(resume=…)` continues the *same* thread; `interrupt()` returns the decision inside `human_review`."),
code('''from langgraph.types import Command

out = invoke_traced(workflow, Command(resume={"type": "approve"}), config=thread)
print("path:", out["path"])
show(out["answer"])
print("\\nsafety log:", SAFETY_EVENT_LOG[-1:])'''),
md("**⛔ Reject.** Same question, new thread: this time the reviewer rejects, so `file_event` never runs."),
code('''reject_thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
before = len(SAFETY_EVENT_LOG)
out = workflow.invoke({"question": SAFETY_Q, "patient_id": "NH-1004"}, reject_thread)
if out.get("__interrupt__"):
    out = workflow.invoke(Command(resume={"type": "reject", "message": "Already reported by the charge nurse."}), reject_thread)
print("path:", out["path"], "| new log entries:", len(SAFETY_EVENT_LOG) - before)'''),

md("""
## 7. State history: every step is a checkpoint

The checkpointer kept a snapshot after **every** node. That's what powers resuming, debugging ("what did the state look like before `answer`?"), and time travel in Studio.
"""),
code('''for snap in workflow.get_state_history(thread):
    print(f"step {snap.metadata.get('step'):>2} | next={snap.next!s:22s} | path={snap.values.get('path')}")'''),
md("**Streaming** shows the same steps live, one node update at a time:"),
code('''for update in workflow.stream({"question": "What is NH-1023's documented code status?", "patient_id": "NH-1023"},
                              {"configurable": {"thread_id": str(uuid.uuid4())}}, stream_mode="updates"):
    for node, value in update.items():
        print(f"▶ {node:12s}", {k: v for k, v in (value or {}).items() if k in ("intent", "masked", "tool_calls")})'''),

md("""
## 8. Durable execution: survive a failure mid-run

Real side effects fail: the reporting system times out, a pod restarts. Because every step is checkpointed, a failed run **resumes from its last checkpoint** instead of starting over, so the expensive (and non-deterministic) agent step does **not** run again, and the human doesn't have to re-approve.

We simulate it: `file_event` crashes the first time it's called, and a counter shows how often the agent node runs.
"""),
code('''calls = {"agent": 0, "file_event": 0}


def counted_answer(state: ChartState) -> dict:
    calls["agent"] += 1
    return answer(state)


def flaky_file_event(state: ChartState) -> dict:
    calls["file_event"] += 1
    if calls["file_event"] == 1:
        raise ConnectionError("safety-reporting system timed out (simulated)")
    return file_event(state)


def build_workflow_with(nodes: dict, checkpointer):
    b = StateGraph(ChartState, input_schema=WorkflowInput)
    for name, fn in nodes.items():
        b.add_node(name, fn)
    b.add_edge(START, "mask_phi")
    b.add_edge("mask_phi", "classify")
    b.add_conditional_edges("classify", route_intent, ["refuse", "emergency", "answer"])
    b.add_edge("refuse", END)
    b.add_edge("emergency", END)
    b.add_conditional_edges("answer", route_safety, ["human_review", END])
    b.add_conditional_edges("human_review", route_review, ["file_event", END])
    b.add_edge("file_event", END)
    return b.compile(checkpointer=checkpointer)


durable = build_workflow_with(
    {"mask_phi": mask_phi, "classify": classify, "refuse": refuse, "emergency": emergency,
     "answer": counted_answer, "human_review": human_review, "file_event": flaky_file_event},
    InMemorySaver(),
)
d_thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
out = durable.invoke({"question": SAFETY_Q, "patient_id": "NH-1004"}, d_thread)
print("paused at:", durable.get_state(d_thread).next, "| agent runs:", calls["agent"])'''),
code('''try:
    durable.invoke(Command(resume={"type": "approve"}), d_thread)   # approve → file_event crashes
except ConnectionError as exc:
    print("💥 run failed:", exc)

snap = durable.get_state(d_thread)
print("saved state → next:", snap.next, "| review:", snap.values.get("review"), "| agent runs:", calls["agent"])'''),
code('''out = durable.invoke(None, d_thread)  # input=None → retry from the last checkpoint on the same thread
print("path:", out["path"])
print(f"agent runs: {calls['agent']} (not re-run) · file_event attempts: {calls['file_event']}")'''),
md("""
> 💡 `InMemorySaver` is fine for a notebook; it dies with the process. In production the same code takes a durable checkpointer (`langgraph-checkpoint-postgres` / `-sqlite`), or runs on **LangSmith Deployment**, which provides managed persistence, so a run paused for a reviewer on Friday resumes on Monday after any number of restarts.

## 9. Packaged, and in Studio

`build_workflow()` in `chart_qa_graph.py` is the graph you just built. It's registered in `langgraph.json` as **`chart_qa_workflow`** (the server provides the checkpointer):

```bash
uv run langgraph dev     # serves at http://127.0.0.1:2024
```

In **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024**, pick `chart_qa_workflow` and submit:

| Input | Watch |
|---|---|
| `{"question": "Is NH-1004 safe to discharge home today?", "patient_id": "NH-1004"}` | `classify → refuse`; the agent node never lights up |
| `{"question": "Were there any problems with medication administration during NH-1004's May 2025 stay?", "patient_id": "NH-1004"}` | pauses at `human_review`; resume with `{"type": "approve"}` or `{"type": "reject"}` |

> 💡 Lab 2's `create_agent` is itself a LangGraph graph (model ⇄ tools). Now you've built one by hand and put that agent **inside** it as a node.
"""),

md("""
## 10. The architecture view: framework vs application

Step back from the code. Across Labs 2 and 3 you've used five architectural building blocks. For each one, the framework provides the **mechanism** and your application provides the **policy**:

| Building block | Where you saw it | Framework provides (LangChain / LangGraph) | You own (application) |
|---|---|---|---|
| **Control loop** | Lab 2's model ⇄ tools loop; Lab 3's explicit graph | the agent loop, graph runtime, routing, parallel tool calls, limits | which decisions the model makes vs which are hard-coded; tool design and docstrings |
| **State** | `ChartState`, reducers | typed state, reducers, per-step snapshots | what goes in the state, and what must never (raw identifiers) |
| **Memory** | thread memory (Lab 2); deep-agent `AGENTS.md` | checkpointers (short-term, per thread); stores and backends (long-term) | what is remembered, for how long, and the rule that patient data never enters long-term memory |
| **Durable execution** | section 8 | checkpoint after every step, resume after failure, `get_state_history` | idempotent side effects; the production checkpointer (Postgres, or LangSmith Deployment) |
| **Human in the loop** | `interrupt()` → `Command(resume=…)` | pause anywhere, persist, resume with a decision | *which* actions need approval, *who* approves, and what the reviewer sees |

**Where to draw the line:** keep the framework doing what's generic (loops, persistence, interrupts, streaming, tracing) and keep **clinical policy in your code and data**: the router's categories, the refusal text, the approval rule, the masking detectors, the tool contracts. That split is what lets you upgrade the framework without re-validating your policy, and re-validate your policy (Lab 4) without touching the framework.

| Pattern | Use when | In this workshop |
|---|---|---|
| Fixed pipeline | the steps never change | Lab 1 RAG |
| Agent loop | the model must choose tools and how many | Lab 2 Part A |
| Deep agent | long, multi-source tasks, planning, execution, team playbooks | Lab 2 Part B |
| Workflow graph | some steps must *always* or *never* happen | Lab 3 (with the agent as one node) |
"""),

md("""
## What you just learned

| Concept | API |
|---|---|
| Typed state + reducer | `class ChartState(TypedDict)` · `path: Annotated[list[str], operator.add]` |
| Nodes | `builder.add_node("name", fn)`; each `fn(state) -> partial update` |
| Edges | `add_edge(START, "mask_phi")`, `add_edge("refuse", END)` |
| **Conditional routing** | `add_conditional_edges("classify", route_intent, ["refuse", "emergency", "answer"])` |
| Agent as a node | `answer` calls the Lab 2 agent: autonomy *inside* a controlled process |
| Human-in-the-loop | `interrupt(payload)` → `Command(resume={"type": "approve"})` |
| **Stateful** | `compile(checkpointer=InMemorySaver())` + `thread_id`; `get_state`, `get_state_history`, `stream` |
| **Durable execution** | a failed run resumes from its last checkpoint: `graph.invoke(None, thread)` |
| Architecture | framework = loops, state, persistence, interrupts; application = policy, tools, data rules |

**Next:** `04_evaluate.ipynb` measures all three versions of the app (pipeline, agent, workflow): are the answers correct, did the agent pick the right tools, and did the graph take the right route?
"""),
]
