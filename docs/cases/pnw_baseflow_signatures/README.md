# PNW Baseflow And Hydrologic Signatures

## Scenario

This case validates HydroPlot's baseflow and hydrologic signature workflow on a Pacific Northwest daily streamflow record. It focuses on method agreement, plausible BFI range, flow-duration behavior, and flashiness.

## What This Proves About HydroPlot

- `lyne_hollick_filter`: bounded multi-pass recursive filter.
- `eckhardt_filter`: independent two-parameter filter.
- `compute_hydrologic_signatures`: compact basin-behavior summary.
- `validate_range`: explicit PASS/FLAG result instead of informal notebook checks.

## How To Run

```powershell
python -m scripts docs/cases/pnw_baseflow_signatures/case.yml
```

## Outputs

- `outputs/baseflow_components.csv`
- `outputs/signatures.csv`
- `outputs/validation_summary.csv`

## Validation

BFI and flashiness are screened against method-comparison ranges. These checks are not a substitute for a basin-specific groundwater study.
