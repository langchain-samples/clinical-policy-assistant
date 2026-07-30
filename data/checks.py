#!/usr/bin/env python3
"""Per-use-case consistency plugin for the Medical-Information QA Assistant.

lab-check is domain-agnostic; THIS file holds medical-information-specific consistency the generic checker
can't know: (1) the medical_information_qa_graph.py tool contract still works, (2) every structured fixture
is synthetic/provenanced and internally referential, and (3) eval reference answers are grounded in
data/inquiries.json, data/drug_profiles.json, and data/kb/*.md with no raw placeholder PII leakage.

Contract: exit 0 = consistent, non-zero = violation. Offline: imports the deployable graph module but makes
no network call (tools are local text/JSON lookups; model clients are constructed only).

Run standalone too:  uv run python verticals/healthcare/medical-information-qa/data/checks.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# content/root: data -> medical-information-qa -> healthcare -> verticals -> content
CONTENT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
KB = DATA / "kb"
DRUGS_PATH = DATA / "drug_profiles.json"
INQUIRIES_PATH = DATA / "inquiries.json"
DATASET = CONTENT / "evals" / "dataset_medical_information_qa.json"
LANGGRAPH_CONFIG = CONTENT / "langgraph.json"

errors: list[str] = []
oks: list[str] = []

ALLOWED_TOOLS = {"get_drug_profile", "get_inquiry", "search_prescribing_info", "check_safety_policy"}
DISPOSITIONS = {"answer", "answer_with_limitations", "refuse_off_label", "refuse_personal_advice", "escalate_adverse_event"}
POLICY_FILES = {"medical_information_response_policy.md", "refusal_policy.md", "adverse_event_pii_policy.md"}
PLACEHOLDER_PII_MARKERS = ["[FAKE_NAME:", "[FAKE_EMAIL:", "[FAKE_PHONE:", "[FAKE_ADDRESS:", "[FAKE_ID:"]


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _records(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if not k.startswith("_")}


def _text_blob(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text_blob(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_text_blob(v) for v in value.values())
    return str(value)


def _anchor_exists(path: Path, anchor: str) -> bool:
    if not anchor:
        return True
    text = path.read_text()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        cleaned = re.sub(r"[^a-z0-9\s-]", "", line.lower().lstrip("# ").strip())
        candidate = re.sub(r"\s+", "-", cleaned)
        if candidate == anchor:
            return True
    return False


def check_synthetic_markers() -> None:
    """Every corpus doc carries a synthetic banner and every structured JSON has a synthetic _note."""
    md_files = sorted(KB.glob("*.md"))
    if len(md_files) != 8:
        errors.append(f"expected 8 KB docs, found {len(md_files)}")
    for md in md_files:
        text = md.read_text().upper()
        if "SYNTHETIC" not in text or "FICTIONAL" not in text:
            errors.append(f"{md.name} missing SYNTHETIC / FICTIONAL banner")
        else:
            oks.append(f"synthetic banner: {md.name}")
    for path in (DRUGS_PATH, INQUIRIES_PATH):
        obj = _load(path)
        note = obj.get("_note", "") if isinstance(obj, dict) else ""
        if "SYNTHETIC" not in note.upper() or "FICTIONAL" not in note.upper():
            errors.append(f"{path.name} missing synthetic _note provenance")
        else:
            oks.append(f"structured provenance: {path.name}")


def check_graph_contract() -> None:
    """medical_information_qa_graph.py imports and each public structured/search tool returns fixture facts."""
    sys.path.insert(0, str(CONTENT))
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-offline-construct-only")  # construct, never call
    if "gateway.smith.langchain.com" in os.environ.get("OPENAI_BASE_URL", ""):
        os.environ.pop("OPENAI_BASE_URL", None)
    try:
        import medical_information_qa_graph as g  # noqa: E402 — deployable module
    except Exception as e:  # pragma: no cover
        errors.append(f"medical_information_qa_graph.py failed to import: {e}")
        return

    for attr in [
        "graph",
        "graph_hitl",
        "graph_deep",
        "search_prescribing_info",
        "get_drug_profile",
        "get_inquiry",
        "check_safety_policy",
        "MedicalInformationAnswer",
    ]:
        if not hasattr(g, attr):
            errors.append(f"medical_information_qa_graph.py missing export {attr}")
    # The deep agent uses ONE reusable label-researcher sub-agent (the coordinator delegates per drug via
    # the built-in `task` tool), carrying exactly the PI-search + safety-policy leaf tools.
    if hasattr(g, "_label_researcher_subagent"):
        subagent = g._label_researcher_subagent()
        expected_tools = {"search_prescribing_info", "check_safety_policy"}
        tool_names = {getattr(t, "name", getattr(t, "__name__", "")) for t in subagent.get("tools", [])}
        if not tool_names.issubset(ALLOWED_TOOLS):
            errors.append(f"deep sub-agent has tools outside the allowed set: {sorted(tool_names)}")
        if tool_names != expected_tools:
            errors.append(f"deep label-researcher sub-agent has unexpected tool set {sorted(tool_names)}")
        if "label-researcher" not in subagent.get("name", ""):
            errors.append("graph_deep sub-agent is not named label-researcher")
    else:
        errors.append("medical_information_qa_graph.py missing _label_researcher_subagent helper")

    try:
        config = _load(LANGGRAPH_CONFIG)
        registered = config.get("graphs", {}).get("medical_information_qa_deep_agent")
        if registered != "./medical_information_qa_graph.py:graph_deep":
            errors.append("langgraph.json missing medical_information_qa_deep_agent -> graph_deep registration")
        else:
            oks.append("graph contract: graph_deep registered in langgraph.json")
    except Exception as e:  # pragma: no cover
        errors.append(f"langgraph.json registration check failed: {e}")

    drugs = _records(_load(DRUGS_PATH))
    inquiries = _records(_load(INQUIRIES_PATH))

    for drug_id, rec in drugs.items():
        out = g.get_drug_profile.invoke({"query": drug_id})
        if "No synthetic drug profile" in out or rec["brand"] not in out or rec["generic"] not in out:
            errors.append(f"get_drug_profile({drug_id!r}) did not return the drug profile")
        else:
            oks.append(f"graph contract: get_drug_profile({drug_id}) OK")

    for inquiry_id, rec in inquiries.items():
        out = g.get_inquiry.invoke({"inquiry_id": inquiry_id})
        if "No inquiry" in out or rec["expected_disposition"] not in out or rec["drug_id"] not in out:
            errors.append(f"get_inquiry({inquiry_id!r}) did not return the inquiry record")
        else:
            oks.append(f"graph contract: get_inquiry({inquiry_id}) OK")

    out = g.search_prescribing_info.invoke({"query": "Alvexora 20 mg orally once daily with food"})
    if "alvexora_pi.md#" not in out or "20 mg" not in out:
        errors.append("search_prescribing_info did not return expected Alvexora dosage PI citation")
    else:
        oks.append("graph contract: search_prescribing_info returns cited PI snippets")

    out = g.check_safety_policy.invoke({"topic": "adverse event pii redaction"})
    if "adverse_event_pii_policy.md" not in out or "medical-review queue" not in out:
        errors.append("check_safety_policy did not return expected AE/PII policy citation")
    else:
        oks.append("graph contract: check_safety_policy returns AE/PII policy")


def check_referential_integrity() -> None:
    """Inquiry drug IDs resolve; required citation files/anchors resolve; fixture records have valid safety shape."""
    drugs = _records(_load(DRUGS_PATH))
    inquiries = _records(_load(INQUIRIES_PATH))
    kb_files = {p.name for p in KB.glob("*.md")}

    if len(drugs) != 5:
        errors.append(f"expected 5 drug profiles, found {len(drugs)}")
    if len(inquiries) != 18:
        errors.append(f"expected 18 inquiry records, found {len(inquiries)}")

    for drug_id, rec in drugs.items():
        for src in rec.get("source_filenames", []):
            if src not in kb_files:
                errors.append(f"drug {drug_id} source file {src!r} not found in data/kb")

    for inquiry_id, rec in inquiries.items():
        if rec.get("drug_id") not in drugs:
            errors.append(f"{inquiry_id} drug_id {rec.get('drug_id')!r} not found in drug_profiles.json")
        if rec.get("expected_disposition") not in DISPOSITIONS:
            errors.append(f"{inquiry_id} invalid expected_disposition {rec.get('expected_disposition')!r}")
        for field in ("requires_refusal", "requires_adverse_event_escalation", "requires_pii_redaction", "contains_placeholder_pii"):
            if not isinstance(rec.get(field), bool):
                errors.append(f"{inquiry_id}.{field} must be boolean")
        if rec.get("requires_pii_redaction") and not rec.get("pii_tokens_to_redact"):
            errors.append(f"{inquiry_id} requires PII redaction but has no pii_tokens_to_redact")
        for citation in rec.get("required_citations", []):
            fname, _, anchor = citation.partition("#")
            if fname not in kb_files:
                errors.append(f"{inquiry_id} required citation {fname!r} not found in data/kb")
            elif anchor and not _anchor_exists(KB / fname, anchor):
                errors.append(f"{inquiry_id} required citation anchor {citation!r} not found")
        blob = _text_blob(rec).lower()
        for fact in rec.get("expected_facts", []):
            tokens = [tok.lower().strip(".,;:()") for tok in re.split(r"[\s/-]+", fact) if len(tok) >= 4]
            if tokens and not any(tok in blob or any(tok in (KB / c.partition('#')[0]).read_text().lower() for c in rec.get("required_citations", []) if (KB / c.partition('#')[0]).exists()) for tok in tokens):
                errors.append(f"{inquiry_id} expected fact {fact!r} is not grounded in inquiry or cited KB")

    if not any("not found" in e or "invalid expected" in e or "must be boolean" in e for e in errors):
        oks.append(f"referential integrity: {len(drugs)} drugs, {len(inquiries)} inquiries, {len(kb_files)} KB docs resolve")


def check_eval_grounding() -> None:
    """The 12 eval examples agree with inquiries.json and cite only existing KB docs/fixture IDs."""
    drugs = _records(_load(DRUGS_PATH))
    inquiries = _records(_load(INQUIRIES_PATH))
    kb_files = {p.name for p in KB.glob("*.md")}
    examples = _load(DATASET)

    if len(examples) != 12:
        errors.append(f"eval dataset should contain 12 examples, found {len(examples)}")

    checked = 0
    for i, ex in enumerate(examples):
        meta = ex.get("metadata", {})
        outputs = ex.get("outputs", {})
        inquiry_id = meta.get("inquiry_id")
        if inquiry_id not in inquiries:
            errors.append(f"eval example {i}: metadata.inquiry_id {inquiry_id!r} not in inquiries.json")
            continue
        inquiry = inquiries[inquiry_id]
        drug_id = inquiry["drug_id"]
        reference_answer = outputs.get("reference_answer", "")

        if outputs.get("disposition") != inquiry.get("expected_disposition"):
            errors.append(f"eval example {i} ({inquiry_id}): outputs.disposition drifted from inquiries.json")
        if outputs.get("expected_facts") != inquiry.get("expected_facts"):
            errors.append(f"eval example {i} ({inquiry_id}): expected_facts drifted from inquiries.json")
        if outputs.get("required_citations") != inquiry.get("required_citations"):
            errors.append(f"eval example {i} ({inquiry_id}): required_citations drifted from inquiries.json")
        if outputs.get("forbidden_claims") != inquiry.get("forbidden_claims"):
            errors.append(f"eval example {i} ({inquiry_id}): forbidden_claims drifted from inquiries.json")
        if outputs.get("refusal_expected") != inquiry.get("requires_refusal"):
            errors.append(f"eval example {i} ({inquiry_id}): refusal_expected drifted from inquiries.json")
        if outputs.get("adverse_event_escalation_expected") != inquiry.get("requires_adverse_event_escalation"):
            errors.append(f"eval example {i} ({inquiry_id}): AE escalation expectation drifted from inquiries.json")
        if outputs.get("pii_redaction_expected") != inquiry.get("requires_pii_redaction"):
            errors.append(f"eval example {i} ({inquiry_id}): PII redaction expectation drifted from inquiries.json")
        if not set(outputs.get("expected_tools", [])).issubset(ALLOWED_TOOLS):
            errors.append(f"eval example {i} ({inquiry_id}): unexpected tool in expected_tools")

        for required in [inquiry_id, drug_id, "inquiries.json", "drug_profiles.json"]:
            if required not in reference_answer:
                errors.append(f"eval example {i} ({inquiry_id}): reference_answer missing {required!r}")
        for citation in inquiry.get("required_citations", []):
            fname = citation.partition("#")[0]
            if fname not in kb_files:
                errors.append(f"eval example {i} ({inquiry_id}): cites {fname!r}, not present in data/kb")
            if citation not in reference_answer:
                errors.append(f"eval example {i} ({inquiry_id}): reference_answer missing required citation {citation!r}")
        for fact in inquiry.get("expected_facts", []):
            content_tokens = [tok for tok in re.split(r"[\s/-]+", fact.lower()) if len(tok) >= 5 and tok not in {"after", "while", "should", "route", "provide", "approved"}]
            if content_tokens and not any(tok.strip(".,;:()") in reference_answer.lower() for tok in content_tokens):
                errors.append(f"eval example {i} ({inquiry_id}): reference_answer missing fact token from {fact!r}")
        for forbidden in inquiry.get("forbidden_claims", []):
            if inquiry.get("requires_pii_redaction") and forbidden in inquiry.get("pii_tokens_to_redact", []):
                continue
            # Forbidden clinical claims may be mentioned only as negated/policy boundaries in reference answers.
            if forbidden in reference_answer and not any(prefix in reference_answer for prefix in ["not approved", "can't", "do not", "does not support", "without"]):
                errors.append(f"eval example {i} ({inquiry_id}): reference_answer appears to assert forbidden claim {forbidden!r}")
        for pii in inquiry.get("pii_tokens_to_redact", []):
            if pii in reference_answer:
                errors.append(f"eval example {i} ({inquiry_id}): raw placeholder PII leaked in reference_answer")
        checked += 1

    if checked == len(examples) and not any(e.startswith("eval example") for e in errors):
        oks.append(f"eval grounding: {len(examples)} examples match inquiry references, citations, and safety flags")


def check_no_real_pii_or_secrets() -> None:
    """Allow obvious fake placeholders; block accidental real-looking direct identifiers or secrets in fixtures."""
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"ls__[A-Za-z0-9]{20,}|lsv2_[A-Za-z0-9_]{20,}"),
        re.compile(r"tvly-[A-Za-z0-9]{16,}"),
    ]
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_pattern = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b")

    for path in [DRUGS_PATH, INQUIRIES_PATH, DATASET] + sorted(KB.glob("*.md")) + [DATA / "prompt.md"]:
        text = path.read_text()
        for pat in secret_patterns:
            if pat.search(text):
                errors.append(f"possible secret-shaped value in {path.name}")
        for email in email_pattern.findall(text):
            if not email.endswith(".invalid"):
                errors.append(f"non-placeholder email in {path.name}: value redacted")
        for phone in phone_pattern.findall(text):
            if phone != "000-000-0000":
                errors.append(f"non-placeholder phone number in {path.name}: value redacted")
        if "Patient Alpha" in text or "Patient Bravo" in text or "123 Placeholder Lane" in text:
            if not any(marker in text for marker in PLACEHOLDER_PII_MARKERS) and path == DATASET:
                errors.append(f"placeholder PII appeared outside tagged input context in {path.name}")
    if not any("secret" in e or "email" in e or "phone" in e or "placeholder PII" in e for e in errors):
        oks.append("privacy/secrets: only obvious fake placeholders and no secret-shaped values")


def main() -> int:
    check_synthetic_markers()
    check_graph_contract()
    check_referential_integrity()
    check_eval_grounding()
    check_no_real_pii_or_secrets()
    for m in oks:
        print(f"  ✅ {m}")
    for m in errors:
        print(f"  ⛔ {m}")
    print(f"checks.py: {len(oks)} ok · {len(errors)} error")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
