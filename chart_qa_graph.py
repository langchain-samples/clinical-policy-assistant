"""Northfield Health chart Q&A assistant: tools, typed answer, and the three graphs for `langgraph dev`.

SYNTHETIC / FICTIONAL: every patient, document and medication is invented for teaching. The assistant
finds, summarizes and cites what is in the record. It never makes clinical decisions.

Graphs (registered in `langgraph.json`):
- `rag_graph`   (`chart_qa_rag`): `create_agent` + notes search + typed `ChartAnswer` (improved retrieval)
- `hitl_graph`  (`chart_qa_agent_hitl`): + structured lookup over FHIR-style JSON, PHI masking, a tool-call
  limit, and human approval before a safety event is filed
- `deep_graph`  (`chart_qa_deep_agent`): coordinator delegating to per-source subagents

Safety: tools are pure-Python lookups over committed synthetic files. `file_safety_event` only appends to
an in-memory list; it does not contact any real reporting system.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from utils.corpus import DATA_DIR, load_fhir
from utils.models import get_model
from utils.prompts import get_prompt
from utils.retrieval import BASELINE, DATE_AWARE_RULES, IMPROVED_V2, RetrievalConfig, format_docs, retrieve

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
# Graph builders
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
                "You query ONE patient's structured record with lookup_structured_record. Return the relevant "
                "entries with status, values, units and dates, each with its [source: fhir/...#id] tag."
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


# Module-level graphs for `langgraph dev` (persistence is provided by the server).
rag_graph = build_rag_agent(IMPROVED_V2)
hitl_graph = build_hitl_agent(IMPROVED_V2)
deep_graph = build_deep_agent(IMPROVED_V2)
