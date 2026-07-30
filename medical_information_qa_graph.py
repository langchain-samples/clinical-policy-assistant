"""Deployable medical-information QA agent over synthetic prescribing-information fixtures.

This graph is built over fictional fixtures in `verticals/healthcare/medical-information-qa/data/` so the
labs can teach cited medical-information Q&A without real drugs, real prescribing information, PII, secrets,
or clinical advice.

In `langchain>=1.0`, `create_agent` returns a compiled LangGraph graph. The module-level graphs below are
registered in `langgraph.json` for LangGraph Studio and reused by notebooks/evals.

Safety: all tools are pure-Python lookups/searches over committed JSON/Markdown fixtures (no SQL, eval,
network, shelling out, or real pharmacovigilance action). The system prompt is pulled with an offline-safe
fallback via `utils.prompts`; API keys are read only by the model provider at runtime.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.models import get_model
from utils.prompts import get_prompt

_DATA_DIR = Path(__file__).resolve().parent / "data"
KB_DIR = _DATA_DIR / "kb"
DRUG_PROFILES_PATH = _DATA_DIR / "drug_profiles.json"
INQUIRIES_PATH = _DATA_DIR / "inquiries.json"
PROMPT_PATH = _DATA_DIR / "prompt.md"

PROMPT_NAME = "medical-information-qa-01-build"

DRUG_PROFILES: dict = json.loads(DRUG_PROFILES_PATH.read_text())
INQUIRIES: dict = json.loads(INQUIRIES_PATH.read_text())
POLICY_FILES = {
    "response": "medical_information_response_policy.md",
    "refusal": "refusal_policy.md",
    "adverse_event": "adverse_event_pii_policy.md",
    "pii": "adverse_event_pii_policy.md",
}


def _talk_to_openai_directly() -> None:
    """LangGraph Platform may inject an OpenAI gateway URL. This workspace's gateway is not assumed to
    serve the configured lab model, so remove that injected value and use the normal provider path."""
    if "gateway.smith.langchain.com" in os.environ.get("OPENAI_BASE_URL", ""):
        os.environ.pop("OPENAI_BASE_URL", None)


_talk_to_openai_directly()


def _records(obj: dict) -> list[dict]:
    return [rec for key, rec in obj.items() if not key.startswith("_")]


def _fmt_list(values: list[str] | None) -> str:
    return ", ".join(values or []) if values else "none"


def _section_anchor(line: str) -> str:
    """Convert a markdown heading to a stable, simple section anchor used in citations."""
    cleaned = re.sub(r"[^a-z0-9\s-]", "", line.lower().lstrip("# ").strip())
    return re.sub(r"\s+", "-", cleaned)


@tool
def get_drug_profile(query: str) -> str:
    """Get one synthetic drug profile by drug ID, brand, or generic name, e.g. `ALV`, `Alvexora`, or
    `namilovir`. Returns approved indications, contraindication tags, AE trigger terms, and source files."""
    q = query.strip().lower()
    if not q:
        return "No drug query provided."

    rec = None
    for drug_id, profile in DRUG_PROFILES.items():
        if drug_id.startswith("_"):
            continue
        haystack = " ".join(
            [drug_id, profile["drug_id"], profile["brand"], profile["generic"], profile["route"]]
            + profile.get("approved_indications", [])
            + profile.get("contraindication_tags", [])
        ).lower()
        if q == drug_id.lower() or q in haystack:
            rec = profile
            break
    if rec is None:
        known = ", ".join(k for k in DRUG_PROFILES if not k.startswith("_"))
        return f"No synthetic drug profile found for {query!r}. Known drug IDs: {known}."

    return (
        f"{rec['drug_id']} — {rec['brand']} ({rec['generic']}), {rec['route']}\n"
        f"  approved_indications: {_fmt_list(rec.get('approved_indications'))}\n"
        f"  contraindication_tags: {_fmt_list(rec.get('contraindication_tags'))}\n"
        f"  ae_trigger_terms: {_fmt_list(rec.get('ae_trigger_terms'))}\n"
        f"  source_filenames: {_fmt_list(rec.get('source_filenames'))}\n"
        f"  [source: drug_profiles.json]"
    )


@tool
def get_inquiry(inquiry_id: str) -> str:
    """Get a synthetic medical-information inquiry record by inquiry ID, e.g. `MIQ-009`. Use before
    answering a record-specific eval/demo prompt."""
    q = inquiry_id.strip().upper()
    rec = INQUIRIES.get(q)
    if rec is None:
        known = ", ".join(k for k in INQUIRIES if not k.startswith("_"))
        return f"No inquiry found for {inquiry_id!r}. Known inquiry IDs: {known}."

    lines = [
        f"{rec['inquiry_id']} — drug_id {rec['drug_id']} · category {rec['category']}",
        f"  expected_disposition: {rec['expected_disposition']}",
        f"  requires_refusal: {rec['requires_refusal']} · requires_adverse_event_escalation: {rec['requires_adverse_event_escalation']} · requires_pii_redaction: {rec['requires_pii_redaction']}",
        f"  query: {rec['query']}",
        f"  required_citations: {_fmt_list(rec.get('required_citations'))}",
        f"  expected_facts: {_fmt_list(rec.get('expected_facts'))}",
        f"  forbidden_claims: {_fmt_list(rec.get('forbidden_claims'))}",
        "  [source: inquiries.json]",
    ]
    return "\n".join(lines)


@tool
def search_prescribing_info(query: str) -> str:
    """Search synthetic prescribing information (PI) and medical-information policy markdown files. Returns
    cited snippets by Markdown filename and section anchor."""
    q = query.strip().lower()
    tokens = [tok for tok in re.split(r"[\s,;:/()]+", q) if len(tok) > 2]
    hits: list[tuple[int, Path, str, str]] = []
    for path in sorted(KB_DIR.glob("*.md")):
        # PI search includes labels and policy; check_safety_policy narrows to policy when needed.
        text = path.read_text()
        lower = text.lower()
        score = (5 if q and q in lower else 0) + sum(1 for tok in tokens if tok in lower)
        if not score:
            continue
        sections = re.split(r"(?m)^## ", text)
        best_heading = "document"
        best_body = sections[0]
        best_score = 0
        for section in sections:
            heading = section.splitlines()[0].strip() if section.splitlines() else "document"
            section_score = (5 if q and q in section.lower() else 0) + sum(1 for tok in tokens if tok in section.lower())
            if section_score > best_score:
                best_score = section_score
                best_heading = heading
                best_body = section
        anchor = _section_anchor(best_heading)
        excerpt = best_body.strip()[:1500]
        hits.append((score + best_score, path, anchor, excerpt))
    hits.sort(key=lambda item: (-item[0], item[1].name, item[2]))
    if not hits:
        return "No relevant synthetic prescribing-information passages found."
    return "\n\n---\n\n".join(
        f"[source: {path.name}#{anchor}]\n{excerpt}" for _, path, anchor, excerpt in hits[:5]
    )


@tool
def check_safety_policy(topic: str) -> str:
    """Retrieve synthetic medical-information policy snippets for response grounding, off-label/personal-advice
    refusal, adverse-event escalation, or PII redaction. Topics: response, refusal, adverse_event, pii."""
    q = topic.strip().lower().replace("-", "_")
    filenames: list[str]
    if q in POLICY_FILES:
        filenames = [POLICY_FILES[q]]
    elif "off" in q or "refus" in q or "personal" in q or "compar" in q:
        filenames = ["refusal_policy.md"]
    elif "adverse" in q or q == "ae" or "event" in q or "pii" in q or "privacy" in q or "redact" in q:
        filenames = ["adverse_event_pii_policy.md"]
    elif "ground" in q or "citation" in q or "response" in q:
        filenames = ["medical_information_response_policy.md"]
    else:
        filenames = ["medical_information_response_policy.md", "refusal_policy.md", "adverse_event_pii_policy.md"]

    blocks = []
    for filename in dict.fromkeys(filenames):
        path = KB_DIR / filename
        text = path.read_text()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        tokens = [tok for tok in re.split(r"[\s,;:/()]+", q) if len(tok) > 2]
        if filename == "adverse_event_pii_policy.md":
            # AE/PII handling usually needs both escalation and redaction language, so return enough policy
            # context rather than a single matching heading paragraph.
            excerpt = "\n\n".join(paragraphs[:6])
        else:
            excerpt = next((p for p in paragraphs if any(tok in p.lower() for tok in tokens)), "\n\n".join(paragraphs[:3]))
        blocks.append(f"[source: {filename}]\n{excerpt[:1800]}")
    return "\n\n---\n\n".join(blocks)


def medical_information_qa_tools() -> list:
    """The agent's four grounding tools: drug profile lookup, inquiry lookup, PI search, and policy lookup."""
    return [get_drug_profile, get_inquiry, search_prescribing_info, check_safety_policy]


class MedicalInformationAnswer(BaseModel):
    """The medical-information answer shape. Passed as `response_format=` so LangChain enforces it and
    returns a parsed object at `result["structured_response"]` — the schema lives here as code, NOT as a
    prompt template. (Mirrored visibly in the 01_build lab, where structured output is taught.)"""

    disposition: Literal[
        "answer", "answer_with_limitations", "refuse_off_label", "refuse_personal_advice", "escalate_adverse_event"
    ] = Field(description="The final response disposition after applying grounding and safety policy.")
    answer: str = Field(description="Concise cited answer, refusal, or escalation summary with no raw PII.")
    safety_flags: list[Literal["off_label", "personal_medical_advice", "adverse_event", "pii_redacted", "unsupported_claim"]] = Field(
        description="Safety flags detected in the request or needed for the answer."
    )
    escalate_adverse_event: bool = Field(description="True when the inquiry should route to medical review as a suspected adverse event.")
    pii_redacted: bool = Field(description="True when direct identifiers were present and must not be repeated.")
    redaction_notes: list[str] = Field(description="Types of identifiers redacted, never the raw identifier values.")
    citations: list[str] = Field(description="Fixture IDs / KB filenames used: inquiry IDs, drug IDs, structured JSON, kb *.md#section.")


def build_agent(*, use_checkpointer: bool = False, interrupt_on: dict | None = None, name: str = "medical_information_qa_agent"):
    """Build the medical-information QA agent.

    Args:
        use_checkpointer: attach an in-memory checkpointer for local memory/threads + HITL.
        interrupt_on: optional HumanInTheLoopMiddleware config mapping tool name -> approval policy.
        name: the graph's internal name.
    """
    system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())

    kwargs = {}
    if interrupt_on:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        kwargs["middleware"] = [HumanInTheLoopMiddleware(interrupt_on=interrupt_on)]
    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver

        kwargs["checkpointer"] = MemorySaver()

    return create_agent(
        get_model(),
        tools=medical_information_qa_tools(),
        system_prompt=system_prompt,
        response_format=MedicalInformationAnswer,  # enforce typed answer shape (schema is code, not prompt text)
        name=name,
        **kwargs,
    )


DRUG_RESEARCHER_BASE_INSTRUCTIONS = (
    "You are a synthetic prescribing-information label researcher. Research ONLY the single fictional drug "
    "named in your task instructions. Use search_prescribing_info for cited PI snippets and check_safety_policy "
    "for refusal, adverse-event, or PII-redaction rules. Return concise cited findings, required safety flags, "
    "and any raw placeholder-PII categories to redact; do not provide the final cross-drug answer. Cite exact "
    "kb markdown filename#section anchors. Never repeat raw placeholder PII."
)

DEEP_AGENT_COORDINATOR_INSTRUCTIONS = (
    "You are the coordinator for complex, multi-drug medical-information inquiries. You have no direct "
    "prescribing-information tools yourself; use the built-in task delegation tool to hand off to the "
    "label-researcher sub-agent ONCE PER DRUG mentioned or needed, naming the specific drug to research in "
    "each task. Ask each researcher for cited PI facts, limitations, policy flags, and adverse-event/PII-"
    "redaction needs for only that drug. Then synthesize one grounded MedicalInformationAnswer. Never invent "
    "figures, never compare drugs beyond what the cited synthetic labels support, refuse off-label or "
    "personal-medical-advice requests per policy, escalate suspected adverse events, and never repeat raw "
    "placeholder PII; summarize only the identifier categories in redaction_notes."
)


def _label_researcher_subagent() -> dict:
    """One generic label-researcher sub-agent, reused for every drug.

    The coordinator holds no leaf tools, so it delegates via the built-in ``task`` tool — once per drug in
    the inquiry, naming the drug to research in each hand-off. The sub-agent carries two tools: cited PI
    search and safety-policy lookup. One definition, many invocations (visible as ``task`` calls in the trace).
    """
    return {
        "name": "label-researcher",
        "description": (
            "Research ONE named fictional drug's synthetic prescribing information and safety policy. Delegate "
            "to this sub-agent once per drug in the inquiry, naming the specific drug to research each time."
        ),
        "system_prompt": DRUG_RESEARCHER_BASE_INSTRUCTIONS,
        "tools": [search_prescribing_info, check_safety_policy],
    }


def build_deep_agent(*, name: str = "medical_information_qa_deep_agent"):
    """Build the capstone deep-agent coordinator for complex multi-drug medical-information inquiries.

    The coordinator intentionally holds no leaf tools. It gets only deepagents' built-in harness tools
    (notably ``task``) plus ONE reusable label-researcher sub-agent it delegates to per drug, then returns
    the same typed ``MedicalInformationAnswer`` schema as the single-agent graph.
    """
    from deepagents import create_deep_agent

    base_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())
    coordinator_prompt = f"{base_prompt}\n\n{DEEP_AGENT_COORDINATOR_INSTRUCTIONS}"
    return create_deep_agent(
        model=get_model(),
        system_prompt=coordinator_prompt,
        subagents=[_label_researcher_subagent()],
        response_format=MedicalInformationAnswer,
        name=name,
    )


# Module-level compiled graphs for langgraph.json / LangGraph Studio.
# 1) medical_information_qa_agent       — plain ReAct-style cited med-info QA agent and eval target.
# 2) medical_information_qa_agent_hitl  — same agent, with approval pause before safety-policy lookup for HITL demos.
# 3) medical_information_qa_deep_agent  — capstone coordinator that delegates per drug and synthesizes one answer.
graph = build_agent()
graph_hitl = build_agent(interrupt_on={"check_safety_policy": True}, name="medical_information_qa_agent_hitl")
graph_deep = build_deep_agent()
