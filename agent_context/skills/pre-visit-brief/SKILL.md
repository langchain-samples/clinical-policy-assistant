---
name: pre-visit-brief
description: Compile a cited pre-visit brief for one patient (current medications, latest labs versus the last discharge, open plan items, safety events). Use when a clinician asks to prepare for a visit or wants a chart summary across sources.
---

<!-- SYNTHETIC / FICTIONAL teaching skill. Not clinical guidance. -->

# Pre-visit brief

## When to use
A clinician asks to "prepare", "brief me", "summarize the chart before a visit", or asks a question that spans medications, labs and recent notes.

## Steps
1. `write_todos` with one item per section below.
2. Delegate to `structured-lookup`: active and recently stopped medications (with dates); the latest value AND the discharge value for each lab that has both.
3. Delegate to `notes-researcher`: the most recent note's assessment and plan; open follow-up items; any documented safety event.
4. If a lab changed between discharge and the latest value, follow the `lab-trend` skill to compute the change.
5. Assemble the brief in this order: Medications · Labs (latest vs discharge, with the computed change) · Plan and follow-up · Safety events.

## Rules
- Facts only, each with a citation. No recommendations.
- If a section has nothing documented, write "Not documented in the record".
