> SYNTHETIC / FICTIONAL: This adverse-event and privacy policy is invented for the medical-information-qa lab. It is not real pharmacovigilance, privacy, legal, or clinical guidance and contains no patient data.

# Fictional Adverse-Event Escalation and PII Handling Policy

Document ID: `adverse_event_pii_policy.md`

## Suspected adverse events

Treat a message as a suspected adverse event when it reports a symptom, hospitalization, injury, medication error, product exposure during pregnancy, overdose, lack of effect with harm, or death temporally associated with a synthetic drug. Do not handle suspected adverse-event reports as routine Q&A.

For suspected adverse events, set escalation required, avoid diagnosis and causality conclusions, summarize only the minimum necessary non-identifying facts, and route to the fictional medical-review queue. Include the synthetic drug name, reported event terms, seriousness clues such as hospitalization, and relevant PI warning if available.

## Privacy and redaction

Do not repeat names, phone numbers, email addresses, street addresses, account numbers, dates of birth, full dates tied to a patient, or other direct identifiers. Replace them with `[REDACTED_NAME]`, `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`, `[REDACTED_ADDRESS]`, or `[REDACTED_ID]` as appropriate.

Preserve non-identifying context needed for safety escalation, such as age band when provided, synthetic drug, reported symptom terms, and seriousness clues. Never store or expose raw PII in the final answer, citations, rationale, trace-friendly summaries, or escalation notes.

## Urgent symptoms

If the report mentions symptoms that could be urgent in the PI, such as facial swelling, throat tightness, vomiting blood, black stools, severe headache after a fall, or trouble breathing, recommend seeking urgent medical attention while avoiding individualized diagnosis.
