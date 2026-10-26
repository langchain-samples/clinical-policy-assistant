---
doc_id: POLICY-patient-safety-escalation
doc_type: policy
synthetic: true
notice: "SYNTHETIC / FICTIONAL policy for the Northfield Health teaching lab. Not real clinical, legal or regulatory guidance."
---

# Chart Assistant: Patient-Safety Escalation Policy (Northfield Health, fictional)

## What counts as a possible safety event

Flag any documented or reported **possible patient-safety event**, such as:
- **Medication errors:** wrong drug, wrong dose, wrong patient, wrong time, wrong route, or an omitted dose. This includes errors caught before or after administration.
- **Falls**, with or without injury.
- **Near misses:** an error that was intercepted before it reached the patient.
- Unexpected harm related to care: pressure injuries, device problems, transfusion reactions.

## What the assistant does

1. Answer the question with citations as usual.
2. Set `safety_event = true` and write a one or two sentence factual `safety_event_summary`. The summary covers what happened, when, and the source document. It must not assign blame or state a causal conclusion that the record does not document.
3. Recommend that the event be reported through the patient-safety reporting process.
4. In the agent version, filing a safety-event report requires **human approval**. The assistant may draft the report, but a person must approve it before it is filed.

## What is not a safety event

These do not on their own count as safety events: expected disease progression, documented side effects already being managed by the care team, or a planned medication change.
