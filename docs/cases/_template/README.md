---
case: <slug>
title: "<one-line case title>"
hydroplot_version: "<version or commit>"
showcases:
  - <analysis_feature>
data_source: "<dataset, URL, and access date>"
runtime_minutes: 0
created: YYYY-MM-DD
---

# <Title>

## Scenario

Describe the PNW hydrology question, the reach or gauge, and why it matters.

## What This Proves About HydroPlot

- `<feature>`: describe the reusable HydroPlot analysis behavior being validated.

## How To Run

```powershell
python scripts/run_case_study.py docs/cases/<case>/case.yml
```

## Outputs

- `outputs/<file>.csv`: describe table.
- `outputs/<file>.png`: describe figure.

## Validation

List the published report, agency method, or expected range used for PASS/FLAG checks.
