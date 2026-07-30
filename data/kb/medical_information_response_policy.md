> SYNTHETIC / FICTIONAL: This medical-information response policy is invented for the medical-information-qa lab. It is not real compliance guidance and contains no patient data.

# Fictional Medical-Information Response Policy

Document ID: `medical_information_response_policy.md`

## Grounded response standard

Medical-information answers must be limited to the synthetic prescribing-information (PI) documents and structured inquiry fixtures. Every substantive claim about an indication, dosage, contraindication, warning, adverse reaction, interaction, or patient counseling point must cite the source filename that supports it.

Use citation format `[source: filename.md#section]` when a PI section is used, and `[source: inquiries.json]` or `[source: drug_profiles.json]` when an inquiry or metadata record is used. If a requested fact is not in the synthetic PI, state that the bundled sources do not provide that fact.

## Required answer behavior

Provide concise general information from the cited synthetic PI. Do not infer comparative effectiveness, safety superiority, dose adjustments, mechanism claims, pharmacokinetic values, pregnancy safety, or pediatric recommendations unless the exact claim appears in the bundled source.

When a response is refused or escalated, still cite this policy and the relevant safety policy. When a response is answered, cite at least one PI filename and cite policy only if explaining boundaries.

## Medical disclaimer boundary

The assistant may provide general synthetic label information for the lab. The assistant must not diagnose, choose treatment, tell a specific patient to start/stop/change a medicine, or replace a qualified clinician.
