#!/usr/bin/env python
"""Offline consistency checks for the synthetic chart corpus, golden dataset, and graph contract.

No network calls. Exit 0 = consistent, 1 = violations.

    uv run python data/checks.py

Checks:
1. every chart doc has valid front matter + a synthetic notice, IDs match filenames, patients are in range
2. FHIR JSON parses, has a synthetic _note, and every source_doc_id exists for that patient
3. golden dataset: 25 examples, category counts, reference_doc_ids exist and belong to the patient
4. facilitator-only FAILURE_MODES.md and policies are excluded from retrieval
5. synthetic-only: no real drug names (deny-list), no real-looking identifiers (SSN/phone/email/other MRNs),
   no real health-system names, only fictional formulary medications
6. graph contract: chart_qa_graph exports + langgraph.json registration
"""

from __future__ import annotations

import codecs
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
errors: list[str] = []
oks: list[str] = []

PATIENTS = {"NH-1001", "NH-1004", "NH-1009", "NH-1017", "NH-1023", "NH-1028", "NH-1035", "NH-1042"}
DOC_TYPES = {"progress_note", "discharge_summary", "consult_note", "nursing_note", "medication_list", "problem_list"}
EXPECTED_CATEGORIES = {"fact_lookup": 8, "multi_doc_synthesis": 5, "failure_mode": 6, "not_in_record": 4, "refusal": 2}

# Common real medication names (brand and generic). Not exhaustive; a tripwire for accidental realism.
REAL_DRUGS = """
acetaminophen paracetamol ibuprofen naproxen aspirin heparin enoxaparin warfarin apixaban rivaroxaban dabigatran
edoxaban clopidogrel ticagrelor prasugrel insulin metformin glipizide sitagliptin empagliflozin dapagliflozin
semaglutide liraglutide furosemide bumetanide torsemide hydrochlorothiazide spironolactone lisinopril losartan
valsartan amlodipine metoprolol carvedilol atenolol diltiazem verapamil amiodarone digoxin atorvastatin
rosuvastatin simvastatin albuterol salbutamol tiotropium fluticasone budesonide montelukast prednisone
prednisolone methylprednisolone dexamethasone hydrocortisone ceftriaxone cefazolin cephalexin azithromycin
amoxicillin penicillin piperacillin vancomycin doxycycline levofloxacin ciprofloxacin nitrofurantoin
ondansetron morphine oxycodone hydromorphone fentanyl tramadol gabapentin pregabalin sertraline fluoxetine
citalopram escitalopram trazodone zolpidem melatonin lorazepam diazepam haloperidol quetiapine olanzapine
levothyroxine omeprazole pantoprazole famotidine lasix eliquis xarelto coumadin plavix lipitor zocor
ventolin advair symbicort ozempic humira keytruda tylenol advil motrin zofran ambien
""".split()
# Real health-system names, ROT13-encoded so the deny-list itself never contains them in plain text.
REAL_SYSTEMS = [codecs.decode(s, "rot13") for s in ("znlb", "pyrirynaq pyvavp", "xnvfre", "wbuaf ubcxvaf", "znff trareny", "vagrezbhagnva")]
IDENTIFIER_PATTERNS = {
    "SSN-like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone-like": re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "long-number (MRN-like)": re.compile(r"(?<![\w.-])\d{7,}(?![\w.-])"),
    "non-NH patient id": re.compile(r"\bNH-(?!10(?:01|04|09|17|23|28|35|42)\b)\d{4}\b"),
}
# Files where example identifiers are intentionally shown (masking demos/tests), excluded from the identifier scan.
IDENTIFIER_SCAN_SKIP = {"phi.py", "phi_handling_policy.md", "checks.py"}


def check_corpus() -> None:
    from utils.corpus import CHARTS_DIR, load_documents

    docs = load_documents()
    per_patient = Counter(d.patient_id for d in docs)
    if set(per_patient) != PATIENTS:
        errors.append(f"patients in corpus {sorted(per_patient)} != expected {sorted(PATIENTS)}")
    for pid, n in per_patient.items():
        if not 6 <= n <= 12:
            errors.append(f"{pid} has {n} documents (expected 6-12)")
    for d in docs:
        if d.path.stem != d.doc_id:
            errors.append(f"{d.path.name}: doc_id {d.doc_id!r} does not match filename")
        if d.path.parent.name != d.patient_id:
            errors.append(f"{d.path.name}: patient_id {d.patient_id} does not match folder")
        if d.doc_type not in DOC_TYPES:
            errors.append(f"{d.doc_id}: unknown doc_type {d.doc_type!r}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.date):
            errors.append(f"{d.doc_id}: bad date {d.date!r}")
        if d.meta.get("synthetic") is not True or "SYNTHETIC" not in str(d.meta.get("notice", "")).upper():
            errors.append(f"{d.doc_id}: missing synthetic flag/notice in front matter")
    if not any(e.startswith(("patients", "NH-", "unknown")) for e in errors):
        oks.append(f"corpus: {len(docs)} docs across {len(per_patient)} patients, front matter valid")
    names = {p.name for p in CHARTS_DIR.rglob("*.md")}
    if "FAILURE_MODES.md" in names:
        errors.append("FAILURE_MODES.md must not live under data/charts (it would be retrievable)")


def check_fhir() -> None:
    from utils.corpus import documents_by_id

    by_id = documents_by_id()
    for pid in sorted(PATIENTS):
        path = DATA / "fhir" / f"{pid}.json"
        try:
            rec = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
            continue
        if "SYNTHETIC" not in rec.get("_note", "").upper():
            errors.append(f"{path.name}: missing synthetic _note")
        if rec.get("patient", {}).get("id") != pid:
            errors.append(f"{path.name}: patient.id mismatch")
        for rtype in ("MedicationStatement", "Observation", "Condition"):
            for r in rec.get(rtype, []):
                src = r.get("source_doc_id")
                if src and (src not in by_id or by_id[src].patient_id != pid):
                    errors.append(f"{path.name}: {rtype}/{r.get('id')} source_doc_id {src!r} not in {pid}'s chart")
    oks.append("fhir: 8 structured records parse with synthetic notes")


def check_dataset() -> None:
    from utils.corpus import documents_by_id

    raw = json.loads((ROOT / "evals" / "dataset_chart_qa.json").read_text())
    if "SYNTHETIC" not in raw.get("_note", "").upper():
        errors.append("dataset_chart_qa.json missing synthetic _note")
    ex = raw["examples"]
    cats = Counter(e["category"] for e in ex)
    if dict(cats) != EXPECTED_CATEGORIES:
        errors.append(f"dataset categories {dict(cats)} != {EXPECTED_CATEGORIES}")
    by_id = documents_by_id()
    for e in ex:
        for k in ("question", "patient_id", "reference_answer", "reference_doc_ids", "category"):
            if k not in e:
                errors.append(f"dataset example missing {k}: {e.get('question', '?')[:50]}")
        for ref in e.get("reference_doc_ids", []):
            if ref not in by_id:
                errors.append(f"dataset ref {ref!r} does not exist")
            elif by_id[ref].patient_id != e["patient_id"]:
                errors.append(f"dataset ref {ref!r} belongs to another patient")
    fms = {e.get("failure_mode") for e in ex if e["category"] == "failure_mode"}
    if len(fms) != 6:
        errors.append(f"expected 6 distinct failure modes, found {fms}")
    oks.append(f"dataset: {len(ex)} examples, {dict(cats)}")


def check_exclusions() -> None:
    from utils.corpus import EXCLUDED_FROM_RETRIEVAL, load_documents

    if "FAILURE_MODES.md" not in EXCLUDED_FROM_RETRIEVAL:
        errors.append("FAILURE_MODES.md is not in EXCLUDED_FROM_RETRIEVAL")
    if any("policies" in str(d.path) or d.path.name == "FAILURE_MODES.md" for d in load_documents()):
        errors.append("facilitator/policy docs leaked into retrieval")
    oks.append("retrieval excludes FAILURE_MODES.md and policies")


def _scan_files() -> list[Path]:
    exts = {".md", ".json", ".py", ".toml", ".example"}
    skip_dirs = {".venv", ".git", ".chroma", ".scratch", "__pycache__", ".ipynb_checkpoints", ".langgraph_api"}
    out = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip_dirs) and (p.suffix in exts or p.suffix == ".ipynb"):
            if p.name.startswith("coding agent brief") or p.name == "PLAN.md":  # local-only, git-ignored
                continue
            out.append(p)
    return out


def check_synthetic_only() -> None:
    drug_re = re.compile(r"\b(" + "|".join(REAL_DRUGS) + r")\b", re.IGNORECASE)
    formulary = {k for k in json.loads((DATA / "formulary.json").read_text()) if not k.startswith("_")}
    for p in _scan_files():
        text = p.read_text(errors="ignore")
        rel = p.relative_to(ROOT)
        if p.name != "checks.py":
            for m in set(drug_re.findall(text)):
                errors.append(f"{rel}: real drug name {m!r}")
            low = text.lower()
            for sysname in REAL_SYSTEMS:
                if sysname in low:
                    errors.append(f"{rel}: real health-system name {sysname!r}")
        if p.name not in IDENTIFIER_SCAN_SKIP and "notebook_src" not in rel.parts and p.suffix != ".ipynb" and "evals" not in rel.parts:
            for label, pat in IDENTIFIER_PATTERNS.items():
                for m in pat.findall(text):
                    errors.append(f"{rel}: {label} {m!r}")
    # every capitalized drug-looking token in med lists must be in the fictional formulary
    for p in (DATA / "charts").rglob("*medication-list.md"):
        for brand in re.findall(r"^\|\s*([A-Z][a-z]+)\s*\(", p.read_text(), flags=re.M):
            if brand.lower() not in formulary:
                errors.append(f"{p.name}: medication {brand!r} not in fictional formulary")
    if not any("real" in e or "-like" in e or "email" in e or "formulary" in e for e in errors):
        oks.append("synthetic-only: no real drug names, health-system names, or real-looking identifiers")


def check_graph_contract() -> None:
    os.environ.setdefault("WORKSHOP_USER", "checks")
    os.environ.setdefault("OPENAI_API_KEY", "sk-offline-construct-only")
    os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_TRACING"] = "false"
    try:
        import chart_qa_graph as g
    except Exception as exc:  # noqa: BLE001
        errors.append(f"chart_qa_graph failed to import: {exc}")
        return
    for attr in ("rag_graph", "hitl_graph", "deep_graph", "ChartAnswer", "lookup_structured_record", "file_safety_event"):
        if not hasattr(g, attr):
            errors.append(f"chart_qa_graph missing {attr}")
    cfg = json.loads((ROOT / "langgraph.json").read_text())["graphs"]
    want = {"chart_qa_rag": "./chart_qa_graph.py:rag_graph", "chart_qa_agent_hitl": "./chart_qa_graph.py:hitl_graph",
            "chart_qa_deep_agent": "./chart_qa_graph.py:deep_graph"}
    if cfg != want:
        errors.append(f"langgraph.json graphs {cfg} != {want}")
    out = g.lookup_structured_record.invoke({"patient_id": "NH-1042", "resource_type": "MedicationStatement", "contains": "Thrombelix"})
    if "2025-03-14" not in out:
        errors.append("lookup_structured_record did not return Thrombelix start date for NH-1042")
    oks.append("graph contract: exports, langgraph.json, structured lookup")


def main() -> int:
    for fn in (check_corpus, check_fhir, check_dataset, check_exclusions, check_synthetic_only, check_graph_contract):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fn.__name__} crashed: {type(exc).__name__}: {exc}")
    for m in oks:
        print(f"  ✅ {m}")
    for m in errors:
        print(f"  ⛔ {m}")
    print(f"checks.py: {len(oks)} ok · {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
