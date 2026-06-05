# Spokane Groundwater Reach Screening

## Scenario

This case validates HydroPlot's reach-scale groundwater screening on a Spokane River reach where upstream/downstream discharge differences are important for dry-season interpretation.

## What This Proves About HydroPlot

- `build_reach_chain`: selected gauges can be represented in upstream-to-downstream order.
- `derive_adjacent_reaches`: a river continuum can be assessed reach-by-reach.
- `summarize_reach_gain_loss`: paired upstream/downstream gain-loss calculation.
- Low-flow median gain/loss: groundwater contribution during dry windows.
- `classify_thermal_sensitivity`: screening context for shade, low flow, and losing/gaining reach conditions.

## How To Run

```powershell
python -m scripts docs/cases/spokane_groundwater_reach/case.yml
```

## Outputs

- `outputs/reach_gain_loss.csv`
- `outputs/validation_summary.csv`

## Validation

This is a screening workflow. It does not claim to replace seepage runs, groundwater modeling, TTools, Shade, Heat Source, QUAL2K, or QUAL2Kw.
