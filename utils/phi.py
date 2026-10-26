"""Identifier detectors for PHI masking (synthetic teaching data).

Policy (`data/policies/phi_handling_policy.md`):
- `NH-####` is the allowed pseudonymous record key and is NOT masked.
- Names, other MRN-like numbers, dates of birth, phone numbers and emails ARE masked before they reach
  the model, any tool, or the trace.

Each `detect_*` function has the `PIIMiddleware` detector signature: `str -> list[PIIMatch]`, so the same
code powers both `mask_text()` (used at the app entry point, before tracing) and the agent middleware.
"""

from __future__ import annotations

import re

from utils.corpus import roster

_HONORIFIC_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+)?")
_NAMED_PATIENT = re.compile(r"\b(?:patient|pt)\s+(?:named\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b")
_MRN = re.compile(r"\b(?:MRN|medical record (?:number|no\.?)|record #)\s*[:#]?\s*[A-Z]{0,3}-?\d{5,12}\b", re.IGNORECASE)
_BARE_LONG_NUMBER = re.compile(r"(?<![\w-])\d{7,12}(?![\w-])")
_DOB = re.compile(
    r"\b(?:DOB|D\.O\.B\.?|date of birth|born(?: on)?)\s*[:\-]?\s*"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\.? \d{1,2},? \d{4})",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _matches(pattern: re.Pattern, text: str, kind: str, group: int = 0) -> list[dict]:
    return [
        {"type": kind, "value": m.group(group), "start": m.start(group), "end": m.end(group)}
        for m in pattern.finditer(text)
    ]


def _roster_name_pattern() -> re.Pattern:
    names = set()
    for rec in roster().values():
        first, last = rec["name"].split(" ", 1)
        names.add(re.escape(rec["name"]))
        names.add(re.escape(last))  # surnames alone (e.g. "the Hale chart") are identifying too
        names.add(re.escape(first) + r"\s+" + re.escape(last[0]) + r"\.?")
    return re.compile(r"\b(?:" + "|".join(sorted(names, key=len, reverse=True)) + r")\b")


def detect_name(text: str) -> list[dict]:
    found = _matches(_roster_name_pattern(), text, "name")
    found += _matches(_HONORIFIC_NAME, text, "name")
    found += _matches(_NAMED_PATIENT, text, "name", group=1)
    return _dedupe(found)


def detect_mrn(text: str) -> list[dict]:
    return _dedupe(_matches(_MRN, text, "mrn") + _matches(_BARE_LONG_NUMBER, text, "mrn"))


def detect_dob(text: str) -> list[dict]:
    return _matches(_DOB, text, "dob")


def detect_phone(text: str) -> list[dict]:
    return _matches(_PHONE, text, "phone")


def detect_email(text: str) -> list[dict]:
    return _matches(_EMAIL, text, "email")


DETECTORS = {
    "dob": detect_dob,
    "mrn": detect_mrn,
    "email": detect_email,
    "phone": detect_phone,
    "name": detect_name,
}


def _dedupe(matches: list[dict]) -> list[dict]:
    """Drop matches contained in an earlier, longer match."""
    out: list[dict] = []
    for m in sorted(matches, key=lambda m: (m["start"], -(m["end"] - m["start"]))):
        if out and m["start"] < out[-1]["end"]:
            continue
        out.append(m)
    return out


def find_identifiers(text: str) -> list[dict]:
    allm: list[dict] = []
    for fn in DETECTORS.values():
        allm += fn(text)
    return _dedupe(allm)


def mask_text(text: str) -> tuple[str, list[str]]:
    """Replace identifiers with `[NAME]`, `[MRN]`, `[DOB]`, `[PHONE]`, `[EMAIL]`. Returns (masked, types)."""
    matches = find_identifiers(text)
    out, last = [], 0
    for m in matches:
        out.append(text[last : m["start"]])
        out.append(f"[{m['type'].upper()}]")
        last = m["end"]
    out.append(text[last:])
    return "".join(out), sorted({m["type"] for m in matches})


def pii_middleware(**kwargs) -> list:
    """`PIIMiddleware` instances for every identifier type (redact strategy). kwargs pass through, e.g.
    `apply_to_tool_results=True`."""
    from langchain.agents.middleware import PIIMiddleware

    return [PIIMiddleware(kind, detector=fn, strategy="redact", **kwargs) for kind, fn in DETECTORS.items()]
