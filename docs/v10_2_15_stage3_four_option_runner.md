# v10.2.15 Stage 3 four-option monotonic campaign

This campaign uses the final accepted 2-D implementation directly and changes
only the selected material parameter row and the option-specific MPZ length/bin
count. It runs the 40-case controlled monotonic matrix:

- options: `ceramic_primary`, `weakT_primary`, `dbtt_primary`, `peak_primary`;
- temperatures: 300 through 1200 K in 100 K increments;
- orientation: 45 degrees;
- one unbranched front;
- deterministic cleavage renewal with fixed 5 micrometre geometry events;
- 500 micrometre target projected crack extension;
- no wake shielding and no mobile-content shielding, as requested for Stage 3.

## Physics path

The executed physics entry is
`arrhenius_fracture.sharp_front_v10_1_7_5`. Its source, anisotropic emission,
transport, shielding, mechanics, and geometry modules are unchanged from the
pre-Stage-3 branch.

The final 2-D source lifecycle is preserved:

- `source_sites_per_system` supplies the promoted reference continuum budget
  `S0` for each channel;
- the available activity is depleted by Arrhenius emission;
- the mobile-plus-retained field ahead of the tip generates the local Taylor
  back stress;
- emitted state evolves through the existing Peierls--Taylor transport,
  storage, recovery, and moving-frame operators;
- source activity is refreshed only as crack advance exposes new geometry over
  the promoted refresh length;
- there is no stationary temporal source recycling.

No signed-atlas engine, source-capacity packing model, kernel-grid adapter, or
state-resolved engine substitution is used by the campaign launcher.

## Files

- `arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv` is the
  exact v9.11.1 registry supplied for transfer.
- `arrhenius_fracture/parameter_registry_v9111.py` selects one exact option row,
  verifies the candidate fingerprint, and writes the one-row manifest consumed
  by the existing 2-D model.
- `arrhenius_fracture/sharp_front_v10_2_15.py` is a parameter-selection wrapper
  that calls `sharp_front_v10_1_7_5` without replacing its physics classes.
- `scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh` constructs and
  executes the 40-case matrix.
- `scripts/classify_v10_2_15_stage3_case.py` distinguishes target completion,
  incomplete post-initiation propagation, and right censoring.
- `scripts/summarize_v10_2_15_stage3.py` writes campaign-level CSV and JSON.

## Full overnight run

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_15_stage3
conda activate arrhenius-sharp-front-v10

OUTROOT=runs/v10_2_15_stage3_existing_2d_parameter_overlay_500um_theta45_1x_v1

nohup env \
OUTROOT="$OUTROOT" \
MAX_JOBS=2 \
STEPS=300000 \
TARGET_EXT_UM=500 \
THETA=45 \
SKIP_FINISHED=1 \
RUNTIME_PREFLIGHT=0 \
bash scripts/run_v10_2_15_stage3_overnight.sh \
> "${OUTROOT}.nohup.log" 2>&1 &

echo $! | tee "${OUTROOT}.pid"
```

Use `MODE=plan` to print the matrix without launching it and `MODE=smoke` for
four 700 K, 20 micrometre checks. Full mode requires the exact four options and
ten temperatures, producing 40 unique cases.

## Output classification

A solver exit code of zero is not automatically called a complete R-curve.
Each case receives one of:

- `COMPLETE`: projected extension reached the target within tolerance;
- `INCOMPLETE`: first passage occurred but the target extension was not reached;
- `CENSORED`: no first passage was recorded within the run horizon;
- `RUN_FAILED`: the simulation or post-run classifier failed.

The aggregate files are `stage3_campaign_summary.csv` and
`stage3_campaign_summary.json` in the campaign output root.
