"""Load the synthetic Northfield Health chart corpus (Markdown with front matter) and the FHIR-style JSON.

Everything here is SYNTHETIC / FICTIONAL teaching data.

Only `data/charts/<patient>/*.md` is retrievable. The facilitator-only `data/FAILURE_MODES.md` and the
policy documents in `data/policies/` are deliberately **excluded** from retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
FHIR_DIR = DATA_DIR / "fhir"
POLICIES_DIR = DATA_DIR / "policies"
EXCLUDED_FROM_RETRIEVAL = {"FAILURE_MODES.md"}

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def slugify(heading: str) -> str:
    """Section heading → stable anchor used in citations, e.g. 'Assessment and Plan' → 'assessment-and-plan'."""
    cleaned = re.sub(r"[^a-z0-9\s-]", "", heading.lower().strip().lstrip("#").strip())
    return re.sub(r"[\s-]+", "-", cleaned).strip("-")


@dataclass
class Section:
    anchor: str
    heading: str
    text: str


@dataclass
class ChartDoc:
    doc_id: str
    patient_id: str
    doc_type: str
    author_role: str
    date: str
    department: str
    title: str
    body: str
    path: Path
    meta: dict = field(default_factory=dict)

    @property
    def sections(self) -> list[Section]:
        return split_sections(self.body)

    def metadata(self) -> dict:
        """Flat, filterable metadata attached to every chunk (Chroma needs scalar values)."""
        return {
            "doc_id": self.doc_id,
            "patient_id": self.patient_id,
            "doc_type": self.doc_type,
            "author_role": self.author_role,
            "date": self.date,
            "department": self.department,
            "title": self.title,
        }


def split_sections(body: str) -> list[Section]:
    """Split a Markdown body on `## ` headings. Text before the first `##` becomes section 'header'."""
    parts = re.split(r"(?m)^## ", body)
    out: list[Section] = []
    head = parts[0].strip()
    if head:
        out.append(Section("header", "header", head))
    for part in parts[1:]:
        heading, _, text = part.partition("\n")
        out.append(Section(slugify(heading), heading.strip(), f"## {heading.strip()}\n{text.strip()}"))
    return out


def parse_doc(path: Path) -> ChartDoc:
    raw = path.read_text()
    m = _FRONT_MATTER.match(raw)
    if not m:
        raise ValueError(f"{path} has no front matter")
    meta = yaml.safe_load(m.group(1))
    body = raw[m.end():].strip()
    title_match = re.search(r"(?m)^# (.+)$", body)
    return ChartDoc(
        doc_id=str(meta["doc_id"]),
        patient_id=str(meta["patient_id"]),
        doc_type=str(meta["doc_type"]),
        author_role=str(meta.get("author_role", "")),
        date=str(meta["date"]),
        department=str(meta.get("department", "")),
        title=title_match.group(1).strip() if title_match else str(meta["doc_id"]),
        body=body,
        path=path,
        meta=meta,
    )


@lru_cache(maxsize=1)
def load_documents() -> tuple[ChartDoc, ...]:
    """All retrievable chart documents, sorted by patient then date."""
    docs = [
        parse_doc(p)
        for p in sorted(CHARTS_DIR.glob("*/*.md"))
        if p.name not in EXCLUDED_FROM_RETRIEVAL
    ]
    return tuple(sorted(docs, key=lambda d: (d.patient_id, d.date, d.doc_id)))


def documents_by_id() -> dict[str, ChartDoc]:
    return {d.doc_id: d for d in load_documents()}


def patient_ids() -> list[str]:
    return sorted({d.patient_id for d in load_documents()})


@lru_cache(maxsize=1)
def roster() -> dict:
    data = json.loads((DATA_DIR / "patients.json").read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=None)
def load_fhir(patient_id: str) -> dict:
    path = FHIR_DIR / f"{patient_id}.json"
    if not path.exists():
        raise KeyError(f"No structured record for {patient_id!r}")
    return json.loads(path.read_text())


def load_policy(name: str) -> str:
    return (POLICIES_DIR / name).read_text()
