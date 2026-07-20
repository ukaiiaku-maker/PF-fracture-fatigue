# v10.2.15 Stage 3 four-option monotonic campaign

This release starts directly from the completed reduced-model parameterization
and mechanical convergence work. It runs the 40-case controlled 2-D monotonic
matrix:

- options: `ceramic_primary`, `weakT_primary`, `dbtt_primary`, `peak_primary`;
- temperatures: 300 through 1200 K in 100 K increments;
- orientation: 45 degrees;
- one unbranched front;
- deterministic cleavage renewal with fixed 5 micrometre geometry events;
- 500 micrometre target projected crack extension;
- active-only mechanically measured signed shielding kernel;
- no wake shielding and no mobile-content shielding.

## Files

- `arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv` is the
  exact v9.11.1 registry supplied for transfer.
- `arrhenius_fracture/parameter_registry_v9111.py` selects one exact option row,
  verifies the candidate fingerprint, and writes a persistent one-row manifest
  for the existing MPZ engine.
- `arrhenius_fracture/sharp_front_v10_2_15.py` combines that row with the
  v10.2.14 active-only signed FEM kernel family.
- `scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh` constructs and
  executes the 40-case matrix.
- `scripts/classify_v10_2_15_stage3_case.py` distinguishes target completion,
  incomplete post-initiation propagation, and right censoring.
- `scripts/summarize_v10_2_15_stage3.py` writes campaign-level CSV and JSON.

## Full run

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_15_stage3
conda activate arrhenius-sharp-front-v10

MODE=full \
SIGNED_KERNEL_FAMILY_JSON=/absolute/path/to/v10_2_14_active_only_family.json \
OUTROOT=runs/v10_2_15_stage3_four_option_monotonic_500um_theta45_1x_v1 \
MAX_JOBS=2 \
bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh
```

Use `MODE=plan` to print the matrix without launching it and `MODE=smoke` for
four 700 K, 20 micrometre checks. The full mode fails closed unless the default
four options and ten temperatures produce exactly 40 unique cases.

## Output classification

A solver exit code of zero is not automatically called a complete R-curve.
Each case receives one of:

- `COMPLETE`: projected extension reached the target within tolerance;
- `INCOMPLETE`: first passage occurred but the target extension was not reached;
- `CENSORED`: no first passage was recorded within the run horizon;
- `RUN_FAILED`: the simulation or post-run classifier failed.

The aggregate files are `stage3_campaign_summary.csv` and
`stage3_campaign_summary.json` in the campaign output root.
