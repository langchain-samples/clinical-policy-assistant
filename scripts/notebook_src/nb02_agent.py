"""Source for notebooks/02_agent.ipynb (Milestone 2 · Agent)."""

from _cells import SETUP, code, md

FILENAME = "02_agent.ipynb"

LOOKUP_SOLUTION = '''import json

from langchain.tools import ToolRuntime, tool

from utils.corpus import load_fhir


@tool
def lookup_structured_record(
    patient_id: str,
    resource_type: Literal["MedicationStatement", "Observation", "Condition"],
    contains: str | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """Look up the patient's STRUCTURED record (simplified FHIR-style JSON): MedicationStatement
    (meds with status, dose, start/end dates), Observation (labs and vitals with dates), or Condition.
    Optional `contains` filters by a case-insensitive substring, e.g. 'creatinine', 'Thrombelix',
    'anticoagulant'. Cite results as [source: fhir/<patient_id>.json#<resource id>]."""
    pid = (runtime.context or {}).get("patient_id") if runtime and isinstance(runtime.context, dict) else None
    record = load_fhir(pid or patient_id)
    rows = record.get(resource_type, [])
    if contains:
        rows = [r for r in rows if contains.lower() in json.dumps(r).lower()]
    if not rows:
        return f"No {resource_type} entries for {pid or patient_id}" + (f" matching {contains!r}." if contains else ".")
    return "\\n".join(f"[source: fhir/{pid or patient_id}.json#{r['id']}] {json.dumps(r)}" for r in rows)'''

LOOKUP_TODO = '''import json

from langchain.tools import ToolRuntime, tool

from utils.corpus import load_fhir


@tool
def lookup_structured_record(
    patient_id: str,
    resource_type: Literal["MedicationStatement", "Observation", "Condition"],
    contains: str | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """TODO 1: replace this line with a docstring the MODEL will read to decide when to use this tool.
    Say what it returns (structured meds with status/dose/start-end dates, dated labs/vitals, conditions),
    what `contains` does, and how to cite results: [source: fhir/<patient_id>.json#<resource id>]."""
    pid = (runtime.context or {}).get("patient_id") if runtime and isinstance(runtime.context, dict) else None
    record = load_fhir(pid or patient_id)
    rows = record.get(resource_type, [])
    if contains:
        rows = [r for r in rows if contains.lower() in json.dumps(r).lower()]
    if not rows:
        return f"No {resource_type} entries for {pid or patient_id}" + (f" matching {contains!r}." if contains else ".")
    return "\\n".join(f"[source: fhir/{pid or patient_id}.json#{r['id']}] {json.dumps(r)}" for r in rows)'''

DEEP_RULES = '''DEEP_RULES = """## Deep agent
Your team conventions (AGENTS.md) and skills are loaded below. Delegate record lookups to your subagents.
Use the filesystem under /work/ for scratch files. Any arithmetic (differences, percentages, days between
dates) MUST follow the `lab-trend` skill: read it, then run the calculation with `execute`."""'''

MANAGED_SOLUTION = '''from deepagents.backends import CompositeBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemPermission

from utils.managed import context_backend

managed_agent = create_deep_agent(
    model=model,
    system_prompt=system_prompt(RAG_CONFIG, DEEP_RULES),
    subagents=subagents,
    backend=CompositeBackend(
        default=StateBackend(),                                   # scratch files live in the run's state
        routes={"/context/": context_backend(published=bool(url))},  # /context/* is read from Context Hub
    ),
    memory=["/context/AGENTS.md"],                                # team conventions, loaded into the prompt
    skills=["/context/skills/"],                                  # skills: read on demand (progressive disclosure)
    permissions=[FilesystemPermission(operations=["write"], paths=["/context/**"], mode="deny")],  # read-only
    response_format=ChartAnswer,
)'''

MANAGED_TODO = '''from deepagents.backends import CompositeBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemPermission

from utils.managed import context_backend

managed_agent = create_deep_agent(
    model=model,
    system_prompt=system_prompt(RAG_CONFIG, DEEP_RULES),
    subagents=subagents,
    backend=CompositeBackend(
        default=StateBackend(),                                   # scratch files live in the run's state
        routes={"/context/": context_backend(published=bool(url))},  # /context/* is read from Context Hub
    ),
    # TODO 4: load the team conventions and skills from Context Hub, and make them read-only:
    #   memory=["/context/AGENTS.md"],
    #   skills=["/context/skills/"],
    #   permissions=[FilesystemPermission(operations=["write"], paths=["/context/**"], mode="deny")],
    response_format=ChartAnswer,
)'''

PART_B = [
md("""
---
# Part B · Deep agents

## B1. When a tool loop isn't enough

Part A's agent is a **thin loop**: model + two tools. That's the right default for focused questions. Some requests are different: *"Prepare a pre-visit brief: current meds, labs since discharge with the exact change, open plan items, any safety events."* That's long, multi-source, needs exact arithmetic, and should follow the team's house style.

A **deep agent** (`create_deep_agent`) is a managed **harness** around the same loop. Before reaching for it, weigh the considerations:

| Consideration | The question to ask | What the harness gives you |
|---|---|---|
| **Planning** | Is the task multi-step, where losing track is the failure mode? | `write_todos`: an explicit, visible plan |
| **Context management** | Will tool outputs flood the context window? | a **virtual filesystem** to offload results; automatic **summarization** near the limit |
| **Decomposition** | Which work can run in an isolated context? Where are the boundaries? | **subagents** via `task`: one per data source, each with its own tools and context |
| **Instructions and know-how** | Where do house rules live, who changes them, how fast? | **memory** (`AGENTS.md`) and **skills** loaded from a backend |
| **Execution** | Does it need to compute, not estimate? Where does that code run? | an `execute` tool, backed by a **sandbox** |
| **Permissions** | What may it read and write? | `permissions=[FilesystemPermission(…)]` |
| **Data in memory** | What persists between runs? | pluggable backends; in a clinical setting, **never** patient data |
| **Cost, latency, predictability** | Is the extra autonomy worth more model calls and less determinism? | traces show every step and its cost; Lab 4's trajectory evals catch regressions |

> 💡 Rule of thumb: start with `create_agent` (Part A) or a graph (Lab 3); move to a deep agent when planning, context isolation or execution become the bottleneck.
"""),
code('''from deepagents import create_deep_agent

harness = create_deep_agent(model=model)  # no domain tools yet: just the harness
thin = sorted(agent.nodes["tools"].bound.tools_by_name)
deep = sorted(harness.nodes["tools"].bound.tools_by_name)
print("Part A agent tools  :", thin)
print("deep harness built-in:", deep)'''),
md("""
| Built-in tool | Purpose |
|---|---|
| `write_todos` | planning: a checklist the agent updates as it works |
| `ls` · `read_file` · `write_file` · `edit_file` · `glob` · `grep` | a filesystem to offload intermediate results instead of keeping them in context |
| `task` | delegate to a subagent that runs in its own isolated context |
| `execute` | run shell commands; it does nothing until you give the agent a **sandbox** backend (B4) |
"""),

md("""
## B2. Decompose by data source: subagents

The coordinator gets **no record tools**. It delegates: `notes-researcher` owns notes search, `structured-lookup` owns the structured record. Each subagent's raw tool output stays in **its own** context; the coordinator only sees the condensed findings. That's what keeps a long task from drowning in JSON.
"""),
code('''from chart_qa_graph import deep_subagents

subagents = deep_subagents(RAG_CONFIG)
for s in subagents:
    print(f"• {s['name']:18s} tools={[t.name for t in s['tools']]}  — {s['description']}")'''),
code(DEEP_RULES),
code('''deep_agent = create_deep_agent(
    model=model,
    system_prompt=system_prompt(RAG_CONFIG, "Delegate record lookups to your subagents."),
    subagents=subagents,
    response_format=ChartAnswer,
)

from utils.trace import print_deep_steps

BRIEF_Q = "Prepare a pre-visit brief for NH-1017: current medications, latest labs versus discharge, and the open plan items."
result = invoke_traced(deep_agent, {"messages": [{"role": "user", "content": BRIEF_Q}]}, context={"patient_id": "NH-1017"})
print_deep_steps(result)'''),
md("""
> 💡 **Open the trace.** The coordinator plans (`write_todos`), then each `task` call expands into a full subagent run with its own model calls and tool calls. Compare the coordinator's token count with a subagent's: the raw search results never reach the coordinator.
"""),

md("""
## B3. Managed: versioned instructions and skills in LangSmith Context Hub

The agent's **house rules** (`AGENTS.md`) and **skills** (step-by-step playbooks it reads on demand) shouldn't be buried in Python. Context Hub stores them as a **versioned agent repo** in LangSmith:

| Without Context Hub | With Context Hub |
|---|---|
| rules hard-coded in the system prompt; changing them is a code deploy | edit and commit in LangSmith; the next run picks it up, no deploy |
| who changed what, and when? `git blame`, if it's even in git | every change is a commit with history; roll back to any version |
| each team copies prompts around | one repo per agent, shareable across apps and teams |

Our source of truth is `agent_context/` in this repo: `AGENTS.md` plus two skills (`pre-visit-brief`, `lab-trend`). Publish it to **your** scoped repo:
"""),
code('''from utils.managed import hub_repo, local_agent_context, publish_agent_context

for path in local_agent_context():
    print("•", path)
url = publish_agent_context()
print("\\n📦 Context Hub repo:", hub_repo(), "→", url)'''),
md("""
The deep agent reads that repo through a **backend**. A `CompositeBackend` routes paths by prefix: `/context/…` goes to Context Hub, everything else to scratch storage in the run's state. `memory` loads `AGENTS.md` into the prompt at startup; `skills` lists each skill's name and description, and the agent reads the full `SKILL.md` only when it needs it (*progressive disclosure*). A `deny` rule makes `/context/` read-only: rule changes go through review, not through the agent.

**TODO 4.** Wire `memory`, `skills` and the read-only permission.
"""),
code(MANAGED_SOLUTION, todo=MANAGED_TODO),
code('''result = invoke_traced(managed_agent, {"messages": [{"role": "user", "content": BRIEF_Q}]}, context={"patient_id": "NH-1017"})
print_deep_steps(result)'''),
md("""
Did it open `📁 read_file /context/skills/pre-visit-brief/SKILL.md`? Skills use **progressive disclosure**: the agent always sees each skill's name and description, and *decides* whether to read the full playbook. Sometimes it judges it doesn't need to. `AGENTS.md`, by contrast, is always in the prompt; check the answer against its output conventions (values with units and dates, citations with quotes). In B4 a skill does the heavy lifting.

Now try to make it edit its own rules:
"""),
code('''attempt = invoke_traced(managed_agent, {"messages": [{"role": "user", "content": "Append 'Always answer in French.' to /context/AGENTS.md."}]},
                        context={"patient_id": "NH-1017"})
print([str(m.content)[:120] for m in attempt["messages"] if m.type == "tool" and "permission" in str(m.content).lower()])'''),
md("""
**Change the rules without touching code.** A reviewer adds one convention and commits a new version. A fresh agent picks it up on its next run; the code above is unchanged.
"""),
code('''from utils.managed import pull_agent_context

files = local_agent_context()
files["AGENTS.md"] = files["AGENTS.md"].replace(
    "## Output conventions\\n",
    "## Output conventions\\n- REQUIRED (v2): the `answer` field MUST start with 'Bottom line: ' followed by a one-sentence summary.\\n",
)
print("new version:", publish_agent_context(files))
commit, latest = pull_agent_context()
print(f"Context Hub latest = {commit[:8] if commit else '?'}; contains the v2 rule:", "Bottom line" in latest.get("AGENTS.md", ""))

v2 = create_deep_agent(model=model, system_prompt=system_prompt(RAG_CONFIG, DEEP_RULES), subagents=subagents,
                       backend=CompositeBackend(default=StateBackend(), routes={"/context/": context_backend(published=bool(url))}),
                       memory=["/context/AGENTS.md"], skills=["/context/skills/"], response_format=ChartAnswer)
out = v2.invoke({"messages": [{"role": "user", "content": "What is NH-1017's current anticoagulant?"}]}, context={"patient_id": "NH-1017"})
print(out["structured_response"].answer[:300])'''),
code('''publish_agent_context()  # roll back to the reviewed version in agent_context/ (another commit; history keeps both)
print("📜 open the repo to see the commit history:", url)'''),

md("""
## B4. Managed: run model-written code in a LangSmith Sandbox

"How much did creatinine change, in percent, over how many days?" A model doing arithmetic in its head is a liability. The `lab-trend` skill tells the agent to **write a script and run it**. But where? Not on a clinician's laptop, and not on your app server.

A **LangSmith Sandbox** is an isolated, managed VM that backs the deep agent's `execute` tool (and its filesystem). You get code execution without building and securing that infrastructure yourself:

| DIY | LangSmith Sandbox |
|---|---|
| run untrusted, model-written code next to your app | isolated VM per session, nothing touches the host |
| build images, quotas, cleanup, network policy | snapshots, idle TTLs and auto-deletion managed for you |
| bolt on logging | every `execute` call appears in the trace |

`start_sandbox()` creates a scoped sandbox (or returns `None` if Sandboxes aren't enabled for your workspace; the cell still runs, without `execute`).
"""),
code('''from utils.managed import start_sandbox, stop_sandbox

sandbox = start_sandbox()
print("sandbox:", sandbox.id if sandbox else "unavailable: the agent will report raw values instead of computing")

compute_agent = create_deep_agent(
    model=model,
    system_prompt=system_prompt(RAG_CONFIG, DEEP_RULES),
    subagents=subagents,
    backend=CompositeBackend(default=sandbox or StateBackend(),  # ← the sandbox backs files AND execute
                             routes={"/context/": context_backend(published=bool(url))}),
    memory=["/context/AGENTS.md"],
    skills=["/context/skills/"],
    permissions=[FilesystemPermission(operations=["write"], paths=["/context/**"], mode="deny")],
    response_format=ChartAnswer,
)
TREND_Q = "How did NH-1017's creatinine and potassium change from discharge to the latest value? Give the exact percent change and the days between draws."
result = invoke_traced(compute_agent, {"messages": [{"role": "user", "content": TREND_Q}]}, context={"patient_id": "NH-1017"})
print_deep_steps(result)'''),
code('''stop_sandbox(sandbox)  # sandboxes also delete themselves after 30 idle minutes'''),
md("""
> 💡 **Open the trace.** The plan, the delegation to `structured-lookup`, the skill read from Context Hub, the files written into the sandbox, the `execute` call and its exact output are all in one trace. That's the audit trail a reviewer needs before trusting a number.

## B5. What you get from the harness and the managed services

| Need | Hand-rolled `create_agent` | Deep agent harness | + LangSmith managed |
|---|---|---|---|
| Plan long tasks | prompt engineering | `write_todos` built in | |
| Keep context small | manual truncation | filesystem offload, summarization, subagents | |
| Team rules and playbooks | hard-coded prompt | `memory` + `skills` from any backend | **Context Hub**: versioned, reviewable, no redeploy |
| Exact computation | trust the model's arithmetic | `execute` tool | **Sandboxes**: isolated, managed execution |
| Guardrails | your code | `permissions`, `interrupt_on`, middleware | |
| See what happened | your logging | | **LangSmith tracing** of every step |

Optional / Day 2: `notebooks/optional/deep_agent.ipynb` goes further (long-term memory backends, PHI masking on every surface, HITL, and evals for a deep agent).
"""),
]

CELLS = [
md("""
# 02 · Agent: tool selection and the reasoning loop

**Northfield Health chart assistant** (synthetic data only) · **Milestone 2 of 4** · **Time box:** 60 min

| Lab | Milestone | The chart assistant becomes… |
|---|---|---|
| 01 | RAG | a pipeline that retrieves chart passages and answers with citations |
| **02** | **Agent** | **a tool-using agent that decides what to look up, and where** ← you are here |
| 03 | LangGraph | a controlled, stateful workflow with routing and human approval |
| 04 | Evaluation | measured: answer correctness *and* agent behavior |

Lab 1's pipeline always retrieves **once**, with the clinician's words, from **one** source. An **agent** gives the model tools and a loop: it decides **which tool** to call, reads the result, and decides whether to call another one or answer.

```text
question → 🧠 model ──(tool call)──▶ 🔧 tool ──(result)──┐
              ▲                                         │
              └─────────────────────────────────────────┘   … until the model answers
```

**Goals.** By the end you can:
- turn retrieval into a **tool**, and write a second tool over the structured record
- explain how the model **selects tools** (name + arguments + docstring) and watch it do so
- build an agent with `create_agent` and replay its **reasoning loop** step by step, live
- bound the loop with a **tool-call limit**, and give the agent **memory** for follow-up questions
- **Part B:** extend it into a **deep agent**, weigh the design considerations, and plug in two **LangSmith-managed** capabilities: versioned instructions and skills (**Context Hub**) and isolated code execution (**Sandboxes**)

There are **4 `# TODO` cells**. Completed versions live in `solutions/02_agent.ipynb`.
"""),
md("## 0. Setup"),
code(SETUP),

md("""
## 1. Recap: Lab 1's pipeline, packaged

`rag_answer()` in `chart_qa_graph.py` is exactly what you built in Lab 1 (retrieve → generate). Here's a question it struggles with:
"""),
code('''from chart_qa_graph import RAG_CONFIG, rag_answer
from utils.trace import run_traced, show

ABBREV_Q = "Did NH-1023 have any episodes of shortness of breath overnight during the September 2025 admission?"
show(run_traced(rag_answer, ABBREV_Q, "NH-1023")["answer"])'''),

md("""
## 2. Tools: functions the model can call

A tool is a Python function plus a **description**. The model never sees your code. It only sees each tool's **name, arguments and docstring**, and picks tools from that. Our first tool wraps Lab 1's retrieval.
"""),
code('''from chart_qa_graph import make_search_tool

search_chart_notes = make_search_tool(RAG_CONFIG)
print("name:", search_chart_notes.name)
print("args:", search_chart_notes.args)
print("description:", search_chart_notes.description)'''),
code('''print(search_chart_notes.invoke({"query": "c/o SOB overnight", "patient_id": "NH-1023"})[:900])'''),

md("""
## 3. A second tool: the structured record

Each patient also has a **structured** record (FHIR-style JSON): medications with status and start/end dates, dated labs and vitals, conditions. For "current" and "most recent" questions it's more reliable than narrative notes.
"""),
code('''from utils.corpus import load_fhir

rec = load_fhir("NH-1042")
print(list(rec))
print(rec["MedicationStatement"][0])'''),
md("""
**TODO 1.** The function body is done. Write its **docstring**: it *is* the tool's instructions to the model. A vague docstring ("gets data") means the agent won't know when to pick this tool over notes search.

> `runtime: ToolRuntime` is injected by LangChain and hidden from the model. We use it to read the chart that's open (`context={"patient_id": …}`), so the agent can't look up the wrong patient.
"""),
code('''from typing import Literal

''' + LOOKUP_SOLUTION, todo='''from typing import Literal

''' + LOOKUP_TODO),
code('''print("args:", lookup_structured_record.args)
print(lookup_structured_record.invoke({"patient_id": "NH-1042", "resource_type": "MedicationStatement", "contains": "anticoagulant"}))'''),

md("""
## 4. Build the agent

`create_agent` wires model + tools + prompt into a loop. We reuse Lab 1's system prompt and typed `ChartAnswer`, plus short **tool rules** that say when each tool fits.

**TODO 2.** One `create_agent` call.
"""),
code('''from chart_qa_graph import TOOL_RULES, answer_format, system_prompt

print(TOOL_RULES)'''),
code('''from langchain.agents import create_agent

from chart_qa_graph import ChartAnswer
from utils.models import get_model

model = get_model()
tools = [search_chart_notes, lookup_structured_record]
prompt = system_prompt(RAG_CONFIG, TOOL_RULES)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt,
    response_format=answer_format(model, ChartAnswer),  # = ChartAnswer for hosted models (ToolStrategy on Ollama)
)''',
todo='''from langchain.agents import create_agent

from chart_qa_graph import ChartAnswer
from utils.models import get_model

model = get_model()
tools = [search_chart_notes, lookup_structured_record]
prompt = system_prompt(RAG_CONFIG, TOOL_RULES)

# TODO 2: agent = create_agent(model=..., tools=..., system_prompt=..., response_format=answer_format(model, ChartAnswer))
agent = ...'''),
md("**The agent's shape** is a small graph: the `model` node and the `tools` node, with an edge back. That loop *is* the agent."),
code('''from IPython.display import Image, display


def show_graph(graph) -> None:
    g = graph.get_graph()
    try:
        display(Image(g.draw_mermaid_png()))
    except Exception as exc:  # offline renderer: print the structure instead
        print(f"(diagram unavailable: {type(exc).__name__})")
        for e in g.edges:
            print(f"  {e.source} -> {e.target}" + ("  [conditional]" if e.conditional else ""))


show_graph(agent)'''),

md("""
## 5. Tool selection: watch the model choose

`print_steps` replays a run as its loop: each model decision, the tool it called with which arguments, and what came back. Same agent, three kinds of question:
"""),
code('''from utils.trace import invoke_traced, print_steps


def run(question: str, patient_id: str) -> dict:
    return invoke_traced(agent, {"messages": [{"role": "user", "content": question}]}, context={"patient_id": patient_id})


print_steps(run("What anticoagulants is patient NH-1042 on, and when was each started?", "NH-1042"))  # structured'''),
code('''print_steps(run("Summarize the last three cardiology notes for NH-1017.", "NH-1017"))  # narrative'''),
code('''print_steps(run("What was the most recent creatinine for NH-1017, and what did the last cardiology note plan for it?", "NH-1017"))  # both'''),
md("""
> 💡 **Open one of these traces.** Each `model` span shows the tool schemas the model was offered and the tool call it chose; each tool span shows the arguments and output. Tool selection is just the model reading **names + docstrings + the question**, which is why TODO 1 matters.
"""),

md("""
## 6. The reasoning loop, live

`stream(..., stream_mode="updates")` yields one update per step as it happens: the model deciding, the tools running, the model deciding again. Here's the question the pipeline struggled with:
"""),
code('''for update in agent.stream({"messages": [{"role": "user", "content": ABBREV_Q}]},
                           context={"patient_id": "NH-1023"}, stream_mode="updates"):
    for node, value in update.items():
        last = (value or {}).get("messages", [None])[-1] if isinstance(value, dict) else None
        if node == "model" and last is not None and getattr(last, "tool_calls", None):
            print("🧠 model →", [f"{tc['name']}({tc['args'].get('query') or tc['args'].get('resource_type')})" for tc in last.tool_calls])
        elif node == "tools" and last is not None:
            print("🔧 tools →", " ".join(str(last.content).split())[:140], "…")
        elif node == "model":
            print("🧠 model → final answer")'''),
code('''result = run(ABBREV_Q, "NH-1023")
show(result["structured_response"])'''),
md("Compare with section 1. The agent can **rephrase and search again** (for example in nursing shorthand), and it can reach for the structured record. A fixed pipeline can do neither."),

md("""
## 7. Bound the loop: a tool-call limit

A loop needs a stop condition beyond "the model decides it's done". **Middleware** wraps the agent; `ToolCallLimitMiddleware` ends the run after N tool calls, so a confused agent can't loop (and spend) forever.

**TODO 3.** Add the middleware with `run_limit=6`. Then see what a limit of 1 does.
"""),
code('''from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt,
    response_format=answer_format(model, ChartAnswer),
    middleware=[ToolCallLimitMiddleware(run_limit=6, exit_behavior="end")],
)''',
todo='''from langchain.agents.middleware import ToolCallLimitMiddleware

# TODO 3: rebuild `agent` like TODO 2, plus middleware=[ToolCallLimitMiddleware(run_limit=6, exit_behavior="end")]
agent = ...'''),
code('''tight = create_agent(model=model, tools=tools, system_prompt=prompt,
                     middleware=[ToolCallLimitMiddleware(run_limit=1, exit_behavior="end")])
out = invoke_traced(tight, {"messages": [{"role": "user", "content": "Summarize every nursing note for NH-1023, one by one."}]},
                    context={"patient_id": "NH-1023"})
requested = [tc["name"] for m in out["messages"] if m.type == "ai" for tc in (m.tool_calls or [])]
blocked = [m for m in out["messages"] if m.type == "tool" and "limit" in str(m.content).lower()]
print(f"tool calls requested: {len(requested)} | blocked by the limit: {len(blocked)} | executed: {len(requested) - len(blocked)}")
print("final message:", str(out["messages"][-1].content)[:300])'''),
md("> 💡 Models often request several tool calls **in parallel** in one step. The limit counts every call, so a single over-eager step can hit it, and the run ends instead of fanning out."),

md("""
## 8. Memory: follow-up questions

A **checkpointer** saves the conversation under a `thread_id`, so a follow-up can say "it" instead of repeating the question. The graph's shape doesn't change: memory is runtime state, not structure.
"""),
code('''import uuid

from chart_qa_graph import chart_checkpointer
from utils.trace import ask

mem_agent = create_agent(model=model, tools=tools, system_prompt=prompt, response_format=answer_format(model, ChartAnswer),
                         middleware=[ToolCallLimitMiddleware(run_limit=6, exit_behavior="end")],
                         checkpointer=chart_checkpointer())  # an InMemorySaver allowed to restore our ChartAnswer type
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

print(ask(mem_agent, "What is NH-1017's current anticoagulant?", patient_id="NH-1017", config=thread).answer)'''),
code('''show(ask(mem_agent, "What dose is documented for it, and since when?", patient_id="NH-1017", config=thread))  # same thread → remembers'''),
code('''fresh = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(ask(mem_agent, "What dose is documented for it, and since when?", patient_id="NH-1017", config=fresh).answer)

print("\\nmessages in memory → original thread:", len(mem_agent.get_state(thread).values["messages"]),
      "| new thread:", len(mem_agent.get_state(fresh).values["messages"]))'''),
md("On the new thread the agent has **no history**, so \"it\" is ambiguous: it may look something up and guess, or ask what you mean. Either way, it isn't *remembering*. Compare the two message counts."),

md("""
## 9. Packaged, and in Studio

`build_agent()` in `chart_qa_graph.py` is what you just built (two tools, typed answer, tool-call limit). It's registered in `langgraph.json` as **`chart_qa_agent`**:

```bash
uv run langgraph dev     # serves at http://127.0.0.1:2024
```

Open **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024**, pick `chart_qa_agent`, and ask *"What was the most recent creatinine for NH-1017?"*. Studio draws the model ⇄ tools loop and lets you step through each tool call.
"""),

*PART_B,

md("""
## What you just learned

| Concept | API |
|---|---|
| Tools | `@tool` functions; the model sees **name + args + docstring** only |
| Injected context | `runtime: ToolRuntime` → `runtime.context["patient_id"]` (hidden from the model) |
| Agent | `create_agent(model, tools, system_prompt, response_format=ChartAnswer)` |
| **Tool selection** | the model picks `lookup_structured_record` vs `search_chart_notes` per question |
| **Reasoning loop** | model → tool → result → model … ; replay with `print_steps`, watch with `stream(stream_mode="updates")` |
| Loop bound | `ToolCallLimitMiddleware(run_limit=6, exit_behavior="end")` |
| Memory | `checkpointer=…` + `config={"configurable": {"thread_id": …}}` |
| **Deep agent harness** | `create_deep_agent`: planning, a filesystem, subagents, summarization built in |
| Decomposition | `subagents=[…]`: one per data source, each in an isolated context |
| **Managed: Context Hub** | conventions + skills in a versioned LangSmith repo: `memory=[…/AGENTS.md]`, `skills=[…]`, read-only via `permissions` |
| **Managed: Sandboxes** | `backend=LangSmithSandbox(…)` → `execute` runs model-written code in an isolated VM |

**When to use which:** a `create_agent` loop for focused questions; a deep agent when the task is long, multi-source, needs exact computation, or should follow team know-how that changes faster than your code. Labs 3–4 keep building on the simpler agent.

The agent decides **everything**: which tools, how many, and whether to refuse or flag a safety event. **Next:** `03_langgraph.ipynb` takes some of those decisions back with an explicit, controlled workflow.
"""),
]
