---
doc_id: POLICY-phi-handling
doc_type: policy
synthetic: true
notice: "SYNTHETIC / FICTIONAL policy for the Northfield Health teaching lab. Not real clinical, legal or regulatory guidance."
---

# Chart Assistant: Identifier Handling Policy (Northfield Health, fictional)

## Allowed record key

The `NH-####` patient ID is the pseudonymous record key that the assistant uses to scope retrieval. It may appear in questions, tool calls, and answers.

## Must be masked before reaching the model or the trace

If a user pastes any of the following into a question, mask it before the question reaches the model, any tool, or the LangSmith trace:
- Patient names, e.g. "Mr. Ingersoll" or "Marian Hale" → `[NAME]`
- Medical-record-number-like numbers other than `NH-####`, e.g. "MRN 00482913" → `[MRN]`
- Dates of birth, e.g. "DOB 11/02/1948" → `[DOB]`
- Phone numbers and email addresses → `[PHONE]` / `[EMAIL]`

## Answers

Answers identify the patient only by `NH-####`. Do not repeat a masked value back to the user.
