# v10.1.7.1 final production temperature sweep

This branch contains no constitutive change relative to v10.1.7. It adds a
resume-safe final production campaign and publication-facing summaries for the
three promoted classes:

- ceramic
- weakT
- DBTT

The default sweep is 300--1100 K in 100 K increments, 500 micrometres of crack
extension, theta = 45 degrees, branching disabled, active wake shielding
disabled, campaign back-stress scale = 1, and campaign refresh scale = 1.

The default mode is `full`, giving 27 calculations. Set
`MODES="full plasticity_off"` to add the matched no-plasticity decomposition,
for 54 calculations total.

Run:

```bash
OUTROOT=runs/v10_1_7_1_final_production_three_class_500um_v1 \
CLASSES="ceramic weakT DBTT" \
TEMPS="300 400 500 600 700 800 900 1000 1100" \
MODES="full" \
TARGET_EXT_UM=500 \
STEPS=12000 \
PRINT_EVERY=200 \
bash scripts/run_v10_1_7_1_final_temperature_sweep.sh
```

The runner validates completed cases before skipping them. Incomplete or
invalid directories are rerun unless the underlying solver fails.

Primary outputs:

- `final_production_manifest.tsv`
- `final_production_case_summary.csv`
- `final_production_event_curves.json`
- `final_production_assessment.json`
- `production_K_init_vs_temperature.png`
- `production_K_late_vs_temperature.png`
- `production_R_rise_vs_temperature.png`
- `production_R_curves_ceramic.png`
- `production_R_curves_weakT.png`
- `production_R_curves_DBTT.png`

When `plasticity_off` is included, the analyzer also writes the matched
ablation summary and plot.
