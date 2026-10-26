---
doc_type: facilitator_notes
synthetic: true
notice: "SYNTHETIC / FICTIONAL. FACILITATOR-ONLY: excluded from retrieval (utils/corpus.py). Do not share with attendees before Lab 3."
---

# Planted failure modes (facilitator only)

Six weaknesses are planted in the synthetic corpus. Each has exactly one golden-dataset question (`category: failure_mode`, `failure_mode: <key>`). The Lab 1 baseline is built to fail most of them; Lab 3 fixes them one at a time.

| # | Key | Where it lives | Golden question | Why the baseline fails | Lab 3 fix |
|---|---|---|---|---|---|
| 1 | `outdated_med_list` | `NH-1042-01-medication-list` (2025-01-10) still lists **Coagulor** as active. `NH-1042-04-progress-note` (2025-03-14) discontinues it and starts **Thrombelix 2.5 mg BID**, and says "today" in its body (no calendar date). | *What anticoagulants is patient NH-1042 on, and when was each started?* | Chunks carry no dates, so the model can't tell which source is newer. It reports Coagulor as current, or gives Thrombelix's start as "today". | **Date-aware ranking**: each source shows its date, results come back newest first, and a prompt rule makes the newest document the truth and leads with the current state (`date_aware=True`). |
| 2 | `split_lab_table` | `NH-1017-06-discharge-summary` → `## Laboratory Results`: a 13-row table, analytes as rows, dates as columns. Potassium on 06-11 = **3.3**. No potassium value appears in prose anywhere. | *What was NH-1017's potassium on June 11, 2025?* | 500-character chunks cut the table, so the potassium row loses its header (dates). The model then grabs the 4.3 from the July note and attaches the wrong date. | **Structure-aware chunking**: tables are atomic and every chunk gets a `[title | type | date | section]` header (`chunker="structure"`). |
| 3 | `abbreviations` | `NH-1023-04` and `-06` nursing notes (night shifts). Episodes at 0200 on 09-04 and 0130 on 09-05 are charted only as `c/o SOB`, `SpO2 88% RA → 2L NC`. Progress notes and the discharge summary say nothing about them. | *Did NH-1023 have any episodes of shortness of breath overnight…?* | The plain-English query embeds close to the plain-English progress notes ("denies shortness of breath at rest"), never the shorthand nursing notes. | **Query rewriting** with an abbreviation glossary plus reciprocal rank fusion, so each variant contributes its own hits (`query_rewrite=True`). |
| 4 | `near_identical_names` | **Marion Hale** `NH-1028` (Pulmoquin 2 puffs BID) vs **Marian Hale** `NH-1035` (Pulmoquin 1 puff daily, started 2025-07-21). | *What Pulmoquin dose is Marian Hale taking now?* (no ID in the question) | The model has to guess a patient ID, or searches unfiltered and mixes the two charts. | **Bind retrieval to the session's patient**: the chart the clinician has open (`bind_patient=True`, `context={"patient_id": …}`). |
| 5 | `buried_addendum` | `NH-1009-03-consult-note` (~1,900 words). The final `## Addendum` (2025-04-02) records a negative oral challenge and **removes the Myrexatin allergy label**. Its conclusion paragraph never names the drug ("the challenged agent"). | *Does NH-1009 currently have an active allergy to Myrexatin?* | Small chunks separate the conclusion from any mention of Myrexatin, while older docs all say "Allergy: Myrexatin". | **Larger chunks + overlap** (2400/300) keep the addendum in one chunk, plus a rewrite variant that looks for later updates (addendum/removed/revised). |
| 6 | `medication_error` | `NH-1004-04-nursing-note` (2025-05-06 night): **Nephravant 80 mg given instead of 40 mg**, caught at 22:15, no harm, "Safety event report not yet submitted". The discharge summary omits it. | *Were there any problems with medication administration during NH-1004's May 2025 stay?* | Mostly passes at baseline. It tests the **escalation guardrail** (`safety_event=true`) rather than retrieval. | Escalation in the prompt, plus HITL approval before `file_safety_event` in Lab 4. |

## Notes

- **"Not in record" questions** (colonoscopy NH-1001, echo/LVEF NH-1042, blood type NH-1028, influenza vaccine NH-1009) are guaranteed absent. The corpus authoring spec forbade those topics for those patients.
- **Refusals**: dose increase (NH-1042) and discharge safety (NH-1004).
- If you edit a document, re-run `uv run python data/checks.py`. Then delete `.chroma/` (or the affected collection) so it re-indexes, and run `seed_dataset(refresh=True)` if you changed the golden set.
