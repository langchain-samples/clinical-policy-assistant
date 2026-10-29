"""Northfield Health chart Q&A assistant: one application, evolved across the four workshop milestones.

SYNTHETIC / FICTIONAL: every patient, document and medication is invented for teaching. The assistant
finds, summarizes and cites what is in the record. It never makes clinical decisions.

| Milestone | Builder | Graph (`langgraph.json`) |
|---|---|---|
| Lab 1 · RAG | `rag_retrieve` → `rag_generate` (`build_rag_graph`) | `chart_qa_rag` |
| Lab 2 · Agent | `build_agent`: `create_agent` + notes search + structured lookup + tool-call limit | `chart_qa_agent` |
| Lab 3 · LangGraph | `build_workflow`: mask → classify → refuse / answer → safety review (interrupt) → file | `chart_qa_workflow` |
| Optional | `build_deep_agent`: coordinator + per-source subagents | `chart_qa_deep_agent` |

Safety: tools are pure-Python lookups over committed synthetic files. `file_safety_event` only appends to
an in-memory list; it does not contact any real reporting system.
"""

from __future__ import annotations

import itertools
import json
import operator
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.corpus import DATA_DIR, load_fhir
from utils.models import get_model
from utils.prompts import get_prompt
from utils.retrieval import BASELINE, DATE_AWARE_RULES, IMPROVED_V1, IMPROVED_V2, RetrievalConfig, format_docs, retrieve

# The retrieval pipeline used all day: structure-aware chunks (tables stay whole), dated sources, newest first,
# and the session's patient always wins. (`optional/improve_retrieval.ipynb` explains how it was tuned.)
RAG_CONFIG = IMPROVED_V1.with_(name="rag")

PROMPT_PATH = DATA_DIR / "prompt.md"
PROMPT_BASENAME = "chart-qa-system"  # stored in Prompt Hub as scoped(PROMPT_BASENAME)


# --------------------------------------------------------------------------------------------------
# Typed answer (passed as response_format; the schema is code, not prompt text)
# --------------------------------------------------------------------------------------------------

class Citation(BaseModel):
    doc_id: str = Field(description="The doc_id from the [source: doc_id#section] tag, e.g. 'NH-1042-04-progress-note'.")
    section: str = Field(description="The section anchor after '#', e.g. 'assessment-and-plan'.")
    quote: str = Field(description="A short VERBATIM quote (under ~25 words) from that passage that supports the claim.")


class ChartAnswer(BaseModel):
    disposition: Literal["answered", "not_in_record", "declined_clinical_decision", "safety_event_flagged"] = Field(
        description="answered = found and cited; not_in_record = the record does not contain it; "
        "declined_clinical_decision = a recommendation/decision was requested; "
        "safety_event_flagged = the answer is about a possible patient-safety event."
    )
    answer: str = Field(description="Concise answer for a clinician, with dates and values as documented.")
    citations: list[Citation] = Field(description="One entry per source used. Empty only if nothing was found.")
    safety_event: bool = Field(description="True if the question or any retrieved document describes a possible "
                               "patient-safety event (medication error, wrong dose, fall, near miss).")
    safety_event_summary: str | None = Field(default=None, description="1-2 factual sentences: what, when, source. Null if none.")


@dataclass
class ChartContext:
    """Per-request context. In the app, `patient_id` is the chart the clinician has open."""

    patient_id: str | None = None


def _context_patient(runtime: ToolRuntime | None) -> str | None:
    ctx = getattr(runtime, "context", None) if runtime is not None else None
    if ctx is None:
        return None
    if isinstance(ctx, dict):
        return ctx.get("patient_id")
    return getattr(ctx, "patient_id", None)


def answer_format(model, schema=None):
    """How to enforce the answer schema (default `ChartAnswer`). Hosted OpenAI models use native structured
    output. Local Ollama models get `ToolStrategy`: native JSON mode there suppresses tool calls entirely."""
    schema = schema or ChartAnswer
    if type(model).__name__ == "ChatOllama":
        from langchain.agents.structured_output import ToolStrategy

        return ToolStrategy(schema)
    return schema


def system_prompt(config: RetrievalConfig = BASELINE, extra: str = "") -> str:
    """Base prompt (Prompt Hub, falling back to `data/prompt.md`) plus any rules the config's fixes need."""
    from utils.workshop import scoped

    try:
        name = scoped(PROMPT_BASENAME)
    except RuntimeError:
        name = PROMPT_BASENAME
    prompt = get_prompt(name, fallback=PROMPT_PATH.read_text().strip())
    if config.date_aware:
        prompt += "\n" + DATE_AWARE_RULES
    return prompt + (("\n\n" + extra.strip()) if extra else "")


# --------------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------------

def make_search_tool(config: RetrievalConfig = BASELINE):
    """Build the `search_chart_notes` tool for one retrieval configuration."""

    @tool("search_chart_notes")
    def search_chart_notes(query: str, patient_id: str | None = None, runtime: ToolRuntime = None) -> str:
        """Search the patient's chart notes (progress notes, discharge summaries, consults, nursing notes,
        medication and problem lists). Pass the NH-#### patient ID to restrict results to that patient.
        Returns passages tagged [source: doc_id#section] to cite."""
        pid = patient_id
        bound = _context_patient(runtime)
        if config.bind_patient and bound:
            pid = bound  # the session's chart wins over whatever ID the model guessed
        return format_docs(retrieve(query, pid, config), config)

    return search_chart_notes


_RESOURCE = Literal["MedicationStatement", "Observation", "Condition"]


@tool
def lookup_structured_record(
    patient_id: str, resource_type: _RESOURCE, contains: str | None = None, runtime: ToolRuntime = None
) -> str:
    """Look up the patient's STRUCTURED record (simplified FHIR-style JSON): MedicationStatement
    (meds with status, dose, start/end dates), Observation (labs and vitals with dates), or Condition.
    Optional `contains` filters by a case-insensitive substring of the display name, e.g. 'creatinine',
    'Thrombelix', 'anticoagulant'. Cite results as [source: fhir/<patient_id>.json#<resource id>]."""
    pid = _context_patient(runtime) or patient_id
    try:
        record = load_fhir(pid)
    except KeyError as exc:
        return str(exc)
    rows = record.get(resource_type, [])
    if contains:
        needle = contains.lower()
        rows = [r for r in rows if needle in json.dumps(r).lower()]
    if not rows:
        return f"No {resource_type} entries for {pid}" + (f" matching {contains!r}." if contains else ".")
    return "\n".join(f"[source: fhir/{pid}.json#{r.get('id')}] {json.dumps(r)}" for r in rows)


SAFETY_EVENT_LOG: list[dict] = []
_event_ids = itertools.count(1)


@tool
def file_safety_event(patient_id: str, event_type: str, summary: str, source_doc_id: str) -> str:
    """File a patient-safety event report (medication error, fall, near miss). REQUIRES HUMAN APPROVAL:
    the run pauses until a reviewer approves, edits, or rejects. Synthetic: writes to an in-memory log only."""
    event = {
        "event_id": f"SE-{next(_event_ids):04d}",
        "patient_id": patient_id,
        "event_type": event_type,
        "summary": summary,
        "source_doc_id": source_doc_id,
    }
    SAFETY_EVENT_LOG.append(event)
    return f"Filed synthetic safety event {event['event_id']} for {patient_id} ({event_type})."


# --------------------------------------------------------------------------------------------------
# Lab 1 · RAG: retrieve → generate (no agent; retrieval always runs exactly once, with the raw question)
# --------------------------------------------------------------------------------------------------

def rag_retrieve(question: str, patient_id: str | None, config: RetrievalConfig = RAG_CONFIG) -> list[Document]:
    """Step 1: the top-k chunks for this question, from THIS patient's chart only."""
    return retrieve(question, patient_id, config)


def rag_messages(question: str, patient_id: str | None, context: str, config: RetrievalConfig = RAG_CONFIG) -> list:
    """The generation prompt: system rules + the question + the retrieved passages (`format_docs` output)."""
    return [
        {"role": "system", "content": system_prompt(config)},
        {"role": "user", "content": f"Patient: {patient_id or 'unknown'}\nQuestion: {question}\n\n"
                                    f"Answer only from these retrieved chart passages:\n<context>\n{context}\n</context>"},
    ]


def rag_generate(question: str, patient_id: str | None, docs: list[Document], *, model=None,
                 config: RetrievalConfig = RAG_CONFIG) -> ChartAnswer:
    """Step 2: one model call that must return a typed, cited `ChartAnswer`."""
    model = model or get_model()
    messages = rag_messages(question, patient_id, format_docs(docs, config), config)
    return model.with_structured_output(ChartAnswer).invoke(messages)


@traceable(name="chart_qa_rag", run_type="chain")
def rag_answer(question: str, patient_id: str | None, *, model=None, config: RetrievalConfig = RAG_CONFIG) -> dict:
    """The whole Lab 1 pipeline. Returns {"answer": ChartAnswer, "contexts": [passage, ...]}."""
    docs = rag_retrieve(question, patient_id, config)
    answer = rag_generate(question, patient_id, docs, model=model, config=config)
    return {"answer": answer, "contexts": [d.page_content for d in docs]}


class RagState(TypedDict, total=False):
    question: str
    patient_id: str | None
    contexts: list[str]
    answer: dict


def build_rag_graph(config: RetrievalConfig = RAG_CONFIG, *, model=None):
    """The same two steps as a two-node LangGraph graph, so Studio can show them (Lab 3 explains graphs)."""
    model = model or get_model()

    def retrieve_node(state: RagState) -> dict:
        docs = rag_retrieve(state["question"], state.get("patient_id"), config)
        return {"contexts": [format_docs([d], config) for d in docs]}

    def generate_node(state: RagState) -> dict:
        context = "\n\n---\n\n".join(state.get("contexts", []))
        msgs = rag_messages(state["question"], state.get("patient_id"), context, config)
        return {"answer": model.with_structured_output(ChartAnswer).invoke(msgs).model_dump()}

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(name="chart_qa_rag")


# --------------------------------------------------------------------------------------------------
# Lab 2 · Agent: the model chooses tools and loops until it can answer
# --------------------------------------------------------------------------------------------------

TOOL_RULES = """## Tools (agent version)
- `lookup_structured_record`: current medications (status, dose, start/end dates), dated labs and vitals,
  and conditions. Prefer it for "current", "most recent" and dated-value questions.
- `search_chart_notes`: narrative notes (what was documented, by whom, why). Search again with different
  wording (clinical shorthand, synonyms, "addendum") if the first results don't answer the question.
- Use both when a question needs both, and reconcile them: the newest dated source wins.
- Never answer from memory: look before you say something is or is not in the record."""


def agent_tools(config: RetrievalConfig = RAG_CONFIG) -> list:
    return [make_search_tool(config), lookup_structured_record]


def build_agent(config: RetrievalConfig = RAG_CONFIG, *, model=None, checkpointer=None, tool_call_limit: int = 6,
                name: str = "chart_qa_agent"):
    """Lab 2: notes search + structured lookup, typed answer, and a cap on runaway tool loops."""
    from langchain.agents.middleware import ToolCallLimitMiddleware

    model = model or get_model()
    return create_agent(
        model,
        tools=agent_tools(config),
        system_prompt=system_prompt(config, TOOL_RULES),
        response_format=answer_format(model),
        middleware=[ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="end")],
        checkpointer=checkpointer,
        name=name,
    )


def chart_checkpointer():
    """An in-memory checkpointer that is explicitly allowed to restore our typed answer (`ChartAnswer`) from a
    saved thread. A bare `InMemorySaver()` also works today, but warns, and future LangGraph versions block
    unregistered types by default."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    allowed = [(__name__, "ChartAnswer"), (__name__, "Citation")]
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def agent_tool_calls(messages) -> list[str]:
    """Tool names in the order the agent called them."""
    return [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]


def answer_from_result(result: dict) -> dict:
    """The agent's typed answer as a dict, or a best-effort fallback if it stopped early (e.g. tool limit)."""
    if result.get("structured_response") is not None:
        return result["structured_response"].model_dump()
    return {"disposition": "answered", "answer": str(result["messages"][-1].content), "citations": [],
            "safety_event": False, "safety_event_summary": None}


# --------------------------------------------------------------------------------------------------
# Lab 3 · LangGraph: a controlled, stateful workflow around the agent
# --------------------------------------------------------------------------------------------------

class Intent(BaseModel):
    intent: Literal["chart_question", "clinical_decision", "emergency"] = Field(
        description="chart_question = asks what the record says (including past errors or events); "
        "clinical_decision = asks for a recommendation or decision (start/stop/change a medication or dose, "
        "discharge readiness, which treatment is better, whether to order something); "
        "emergency = a patient in acute danger right now (unresponsive, code, crashing).")
    reason: str = Field(description="One short sentence.")


CLASSIFY_PROMPT = ("You route requests sent to a chart assistant that only finds and summarizes what a patient's "
                   "record says. Classify the request.")

REFUSAL_TEXT = ("I can't make that decision: it requires clinical judgment, and a clinician responsible for this "
                "patient must decide. I can summarize what the record documents if you ask a factual question.")
EMERGENCY_TEXT = ("This assistant summarizes chart records and is not for emergencies. Activate your unit's "
                  "rapid-response process and contact the responsible clinician now.")


class ChartState(TypedDict, total=False):
    question: str                            # the clinician's request (masked in place by mask_phi)
    patient_id: str | None                   # the chart that is open
    masked: list[str]                        # identifier kinds removed from the question
    intent: str                              # chart_question | clinical_decision | emergency
    answer: dict                             # ChartAnswer as a dict
    tool_calls: list[str]                    # tools the agent called inside `answer`
    contexts: list[str]                      # what those tools returned
    trajectory: list[dict]                   # the agent's full message trajectory (OpenAI format), for evals
    review: dict | None                      # the human reviewer's decision
    path: Annotated[list[str], operator.add]  # every node appends its name (reducer = list concatenation)


class WorkflowInput(TypedDict):
    question: str
    patient_id: str | None


def _declined(text: str) -> dict:
    return ChartAnswer(disposition="declined_clinical_decision", answer=text, citations=[], safety_event=False).model_dump()


def build_workflow(*, agent=None, model=None, checkpointer=None, name: str = "chart_qa_workflow"):
    """Lab 3: the agent becomes ONE node in an explicit graph. Masking always runs first, decision requests are
    refused without calling the agent, and a safety report is filed only after a human approves it."""
    from utils.phi import mask_text

    model = model or get_model()
    agent = agent or build_agent(model=model)
    router = model.with_structured_output(Intent)

    def mask_phi(state: ChartState) -> dict:
        question, kinds = mask_text(state["question"])
        return {"question": question, "masked": kinds, "path": ["mask_phi"]}

    def classify(state: ChartState) -> dict:
        verdict = router.invoke([{"role": "system", "content": CLASSIFY_PROMPT},
                                 {"role": "user", "content": state["question"]}])
        return {"intent": verdict.intent, "path": ["classify"]}

    def route_intent(state: ChartState) -> Literal["refuse", "emergency", "answer"]:
        return {"clinical_decision": "refuse", "emergency": "emergency"}.get(state["intent"], "answer")

    def refuse(state: ChartState) -> dict:
        return {"answer": _declined(REFUSAL_TEXT), "path": ["refuse"]}

    def emergency(state: ChartState) -> dict:
        return {"answer": _declined(EMERGENCY_TEXT), "path": ["emergency"]}

    def answer(state: ChartState) -> dict:
        from langchain_core.messages import convert_to_openai_messages

        result = agent.invoke({"messages": [{"role": "user", "content": state["question"]}]},
                              context={"patient_id": state.get("patient_id")})
        return {
            "answer": answer_from_result(result),
            "tool_calls": agent_tool_calls(result["messages"]),
            "contexts": [str(m.content) for m in result["messages"] if getattr(m, "type", "") == "tool"],
            "trajectory": convert_to_openai_messages(result["messages"]),
            "path": ["answer"],
        }

    def route_safety(state: ChartState) -> Literal["human_review", "__end__"]:
        return "human_review" if state["answer"].get("safety_event") else END

    def human_review(state: ChartState) -> dict:
        cites = state["answer"].get("citations") or []
        decision = interrupt({
            "action": "file_safety_event",
            "patient_id": state.get("patient_id"),
            "summary": state["answer"].get("safety_event_summary") or state["answer"].get("answer"),
            "source_doc_id": cites[0]["doc_id"] if cites else "unknown",
            "instructions": 'Resume with {"type": "approve"} or {"type": "reject", "message": "why"}.',
        })
        return {"review": decision, "path": ["human_review"]}

    def route_review(state: ChartState) -> Literal["file_event", "__end__"]:
        return "file_event" if (state.get("review") or {}).get("type") == "approve" else END

    def file_event(state: ChartState) -> dict:
        cites = state["answer"].get("citations") or []
        receipt = file_safety_event.invoke({
            "patient_id": state.get("patient_id") or "unknown",
            "event_type": "possible patient-safety event",
            "summary": state["answer"].get("safety_event_summary") or state["answer"].get("answer"),
            "source_doc_id": cites[0]["doc_id"] if cites else "unknown",
        })
        updated = {**state["answer"], "answer": f"{state['answer']['answer']}\n\n✅ {receipt}"}
        return {"answer": updated, "path": ["file_event"]}

    g = StateGraph(ChartState, input_schema=WorkflowInput)
    for fn in (mask_phi, classify, refuse, emergency, answer, human_review, file_event):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "mask_phi")
    g.add_edge("mask_phi", "classify")
    g.add_conditional_edges("classify", route_intent, ["refuse", "emergency", "answer"])
    g.add_edge("refuse", END)
    g.add_edge("emergency", END)
    g.add_conditional_edges("answer", route_safety, ["human_review", END])
    g.add_conditional_edges("human_review", route_review, ["file_event", END])
    g.add_edge("file_event", END)
    return g.compile(checkpointer=checkpointer, name=name)


# --------------------------------------------------------------------------------------------------
# Earlier variants, kept for the optional notebooks
# --------------------------------------------------------------------------------------------------

def build_rag_agent(config: RetrievalConfig = BASELINE, *, model=None, prompt: str | None = None, name: str = "chart_qa_rag"):
    """Lab 1/3: one search tool + typed answer. `config` selects the retrieval pipeline."""
    model = model or get_model()
    return create_agent(
        model,
        tools=[make_search_tool(config)],
        system_prompt=prompt or system_prompt(config),
        response_format=answer_format(model),
        name=name,
    )


AGENT_RULES = """## Tools and escalation (agent version)
- Use `lookup_structured_record` for current medications, dated lab values and conditions; use
  `search_chart_notes` for narrative context. Prefer the structured record for "current" questions and
  cite it as [source: fhir/<id>.json#<resource id>].
- If you find a possible patient-safety event, call `file_safety_event` ONCE with a factual summary and
  the source doc_id. A human reviewer must approve it before it is filed. If the reviewer rejects it,
  do not retry; say it was not filed."""


def build_hitl_agent(
    config: RetrievalConfig = IMPROVED_V2, *, model=None, checkpointer=None, tool_call_limit: int = 8,
    name: str = "chart_qa_agent_hitl",
):
    """Lab 4: notes search + structured lookup + safety filing behind human approval, with PHI masking
    and a tool-call limit."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware

    from utils.phi import pii_middleware

    model = model or get_model()
    return create_agent(
        model,
        tools=[make_search_tool(config), lookup_structured_record, file_safety_event],
        system_prompt=system_prompt(config, AGENT_RULES),
        response_format=answer_format(model),
        middleware=[
            *pii_middleware(),
            ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="end"),
            HumanInTheLoopMiddleware(interrupt_on={"file_safety_event": True}),
        ],
        checkpointer=checkpointer,
        name=name,
    )


COORDINATOR_RULES = """## Coordinator (deep agent)
You have no record tools of your own except `file_safety_event`. Delegate with the `task` tool:
- `notes-researcher`: narrative chart notes (progress, discharge, consult, nursing notes, med/problem lists).
- `structured-lookup`: the structured record (current meds with dates, dated labs/vitals, conditions).
Always include the NH-#### patient ID and the exact question in each task description. For questions
that need both sources, delegate to both, then reconcile (newest dated source wins) and answer.
If any subagent reports 'SAFETY EVENT: yes', set safety_event=true with a factual summary, even if the
question did not ask about safety. Do not use the filesystem tools. Never make clinical decisions."""


def deep_subagents(config: RetrievalConfig = IMPROVED_V2) -> list[dict]:
    return [
        {
            "name": "notes-researcher",
            "description": "Searches one patient's narrative chart notes and returns cited findings with dates.",
            "system_prompt": (
                "You search ONE patient's chart notes with search_chart_notes (always pass the NH-#### ID). For summaries, search several times (the topic, the date, 'medication administration', 'event or incident') so every section of the relevant notes is covered. "
                "Return concise findings with dates, each with its [source: doc_id#section] tag and a short "
                "verbatim quote. End with a line 'SAFETY EVENT: yes - <what, when, doc_id>' or 'SAFETY EVENT: no'; a possible patient-safety event is any medication error, wrong dose, fall or near miss in the notes. Do not make clinical decisions."
            ),
            "tools": [make_search_tool(config)],
        },
        {
            "name": "structured-lookup",
            "description": "Queries one patient's structured record (MedicationStatement, Observation, Condition).",
            "system_prompt": (
                "You query ONE patient's structured record with lookup_structured_record. Return EVERY matching "
                "entry with its status, value, unit and date (all draws of a lab, not just the latest, so the caller "
                "can compare), each with its [source: fhir/...#id] tag."
            ),
            "tools": [lookup_structured_record],
        },
    ]


def build_deep_agent(config: RetrievalConfig = IMPROVED_V2, *, model=None, checkpointer=None, name: str = "chart_qa_deep_agent"):
    """Lab 4 (second half): coordinator + one subagent per data source, HITL on safety filing."""
    from deepagents import create_deep_agent

    from utils.phi import pii_middleware

    model = model or get_model()
    return create_deep_agent(
        model=model,
        tools=[file_safety_event],
        system_prompt=system_prompt(config, COORDINATOR_RULES),
        subagents=deep_subagents(config),
        middleware=pii_middleware(),
        interrupt_on={"file_safety_event": True},
        response_format=answer_format(model),
        checkpointer=checkpointer,
        name=name,
    )


# Module-level graphs for `langgraph dev` (persistence, which interrupts need, is provided by the server).
rag_graph = build_rag_graph()
agent_graph = build_agent()
workflow_graph = build_workflow(agent=agent_graph)
deep_graph = build_deep_agent(IMPROVED_V2)
