---
name: lab-trend
description: Compute lab trends exactly (change, percent change, days between draws) by running Python in the sandbox instead of doing arithmetic in your head. Use whenever an answer needs a difference, percentage, or date interval between lab values.
---

<!-- SYNTHETIC / FICTIONAL teaching skill. Not clinical guidance. -->

# Lab trend

## Steps
1. Get ALL dated values from `structured-lookup` (value, unit, date and source document for each draw).
2. Pick the two reference points the question compares, and state the dates you chose:
   - "At discharge" = the LAST draw of the admission, dated on (or just before) the discharge date. A discharge
     summary's lab table usually covers the whole stay, so many draws cite it: take the last one, never the first.
   - "Latest" = the most recent draw overall.
3. Write only those reference points to `/work/labs.json` as a list of `{"analyte", "label", "value", "unit", "date"}`
   objects (`label` is e.g. "discharge" or "latest").
4. Write `/work/trend.py` that loads the file and prints, for each analyte: both values with their labels and dates,
   absolute change, percent change (1 decimal), and days between the two draws.
5. Run it with `execute`: `python3 /work/trend.py` (the interpreter is `python3`, not `python`).
6. Report the printed numbers exactly, with units and dates. Cite the structured-record entries the values came from.

## If `execute` is not available
Say that the calculation could not be run, and report the raw dated values only. Do not estimate.
