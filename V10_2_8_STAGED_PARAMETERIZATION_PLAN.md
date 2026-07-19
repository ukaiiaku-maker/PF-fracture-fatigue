# v10.2.8 staged parameterization

## Sequence

The campaign is deliberately split into four separately versioned and inspectable
stages. A stage never overwrites its predecessor.

1. **Analytical screen:** 300--1200 K in 100 K increments. Uses the production
   EXP-floor barriers, serialized local tip geometry, state-resolved signed drive
   family, mechanically derived source normalization, and the signed source-bin
   shielding operator. It integrates no-feedback cleavage and first-emission
   hazards and is used only for ranking.
2. **Exact 1-D first passage:** DBTT at 300, 700, 900, and 1200 K; FCC-like weakT
   at 300, 700, and 1200 K. Uses the exact v10.2.6 state-resolved signed engine and
   stops after one production checkpoint.
3. **Exact 1-D R-curves:** promoted candidates run to 50 um. Full curves use the
   same four DBTT or three weakT temperatures. Plasticity-off, shielding-off, and
   backstress-off ablations are run at the low/high endpoints.
4. **2-D validation:** top four candidates per class run to 100 um. DBTT uses four
   full temperatures plus endpoint ablations; weakT uses three full temperatures
   plus endpoint ablations. Only candidates preserving class and mechanism in 2-D
   proceed to 500 um production and fatigue.

The ceramic-like manifest is frozen. It is a first-passage and R-curve regression
control, not a search target.

## Default campaign sizes

- Analytical: 4096 DBTT + 2048 weakT; promote 256 per class.
- First passage: promote 48 per class.
- 1-D R-curve: promote 4 per class.
- 2-D: validate all promoted candidates.

The candidate source population is sampled only inside the common mechanically
derived source-capacity interval in the authorized kernel family.

## Required inputs

Use actual reviewed files, not example paths:

- authorized v10.2.6 state-resolved signed shielding-kernel family;
- authorized v10.2.7 signed drive family;
- complete v10.2.6 engine configuration from an equilibrated trace;
- current FCC-like weakT manifest;
- frozen ceramic-like manifest.

Both mechanical families must contain
`production_parameterization_allowed=true`.

## Stage 1: analytical

```bash
STAGE=analytical \
KERNEL_FAMILY="$KERNEL_FAMILY" \
DRIVE_FAMILY="$DRIVE_FAMILY" \
ENGINE_TEMPLATE="$ENGINE_TEMPLATE" \
WEAKT_ANCHOR="$WEAKT_ANCHOR" \
OUTROOT=runs/v10_2_8_analytical_screen_v1 \
WORKERS=4 \
bash scripts/run_v10_2_8_staged_parameterization.sh
```

Primary output:

```text
runs/v10_2_8_analytical_screen_v1/promoted_to_first_passage.csv
```

## Stage 2: exact first passage

```bash
STAGE=first-passage \
KERNEL_FAMILY="$KERNEL_FAMILY" \
DRIVE_FAMILY="$DRIVE_FAMILY" \
ENGINE_TEMPLATE="$ENGINE_TEMPLATE" \
CERAMIC_REFERENCE="$CERAMIC_REFERENCE" \
CANDIDATES=runs/v10_2_8_analytical_screen_v1/promoted_to_first_passage.csv \
OUTROOT=runs/v10_2_8_first_passage_v1 \
WORKERS=4 \
bash scripts/run_v10_2_8_staged_parameterization.sh
```

Primary output:

```text
runs/v10_2_8_first_passage_v1/promoted_to_rcurve.csv
```

## Stage 3: exact 1-D R-curves

```bash
STAGE=rcurve \
KERNEL_FAMILY="$KERNEL_FAMILY" \
DRIVE_FAMILY="$DRIVE_FAMILY" \
ENGINE_TEMPLATE="$ENGINE_TEMPLATE" \
CERAMIC_REFERENCE="$CERAMIC_REFERENCE" \
CANDIDATES=runs/v10_2_8_first_passage_v1/promoted_to_rcurve.csv \
OUTROOT=runs/v10_2_8_rcurve_v1 \
WORKERS=2 \
TARGET_EXT_UM=50 \
TWO_D_EXT_UM=100 \
bash scripts/run_v10_2_8_staged_parameterization.sh
```

Primary outputs:

```text
runs/v10_2_8_rcurve_v1/promoted_to_2d.csv
runs/v10_2_8_rcurve_v1/promoted_manifests/
runs/v10_2_8_rcurve_v1/two_d_validation_cases.csv
```

## Stage 4: 2-D validation plan

The R-curve stage already writes the case matrix. It can be regenerated without
rerunning the reduced calculations:

```bash
STAGE=2d-plan \
CANDIDATES=runs/v10_2_8_rcurve_v1/promoted_to_2d.csv \
OUTROOT=runs/v10_2_8_2d_validation_plan_v1 \
TWO_D_EXT_UM=100 \
bash scripts/run_v10_2_8_staged_parameterization.sh
```

Every 2-D run must use `arrhenius_fracture.sharp_front_v10_2_6` with the same
signed kernel family and exact promoted manifest. No shielding cap, source-count
rescaling, or fitted attenuation may be introduced during validation.
