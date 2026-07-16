# v10.0.2 resolved-mesh progression gates

## Entry condition

The resolved 700 K, 10 um three-class gate used:

- `TIP_H_FINE=5e-7 m`, with measured `hbar_tip=7.014e-7 m`;
- `L_pz=1.0e-6 m`, so the nominal process zone was resolved;
- `BULK_PLASTICITY_MODE=tip_only`;
- production `root_signed` directional J;
- one active front and a 5 um physical renewal length.

Relative to the preceding mesh, the first-event toughness changed by only:

- ceramic: 0.02%;
- weakT: 0.13%;
- DBTT: 0.11%.

The second event changed by approximately 0.8--1.4%. This is sufficient to
proceed to longer growth on the resolved mesh.

## Stage A: 700 K, 50 um, wake shielding on

```bash
OUTROOT_BASE=runs/v10_0_2_three_class_700K_50um_wake_on_v1 \
bash scripts/run_v10_0_2_50um_700K.sh
```

This runs ceramic, weakT and DBTT sequentially, with ten 5 um physical renewal
events requested per class.

Acceptance conditions:

1. all three cases reach 50 um committed extension;
2. `KJ` remains finite and positive after loading begins;
3. each case reports exactly one physical renewal per accepted FEM/J state;
4. the output contains `v10_0_1_driver_modes.json` with `tip_only` and
   `root_signed`;
5. active-to-wake state is conserved at every renewal;
6. no unrequested branch is born (`max_fronts=1`);
7. material separation persists beyond the first two events.

The result should be audited before starting Stage B.

## Stage B: causal wake-shielding ablation

After Stage A passes:

```bash
OUTROOT_BASE_PREFIX=runs/v10_0_2_wake_ablation_700K_50um_v1 \
TARGET_EXT_UM=50 \
bash scripts/run_v10_0_2_wake_ablation_700K.sh
```

The two runs have identical state evolution and geometry controls; the only
intended difference is whether the persistent wake contribution is included in
`K_shield`. The active process-zone shielding remains enabled in both cases.

Interpretation:

- a meaningful reduction in the R-curve with wake shielding off identifies the
  residual dislocation wake as a causal toughness mechanism;
- no meaningful change means that the stored wake exists but is mechanically
  negligible under the current signed kernel;
- a sign reversal or toughness reduction with wake shielding on indicates that
  the wake field is anti-shielding for the selected orientation and must not be
  replaced by an unsigned magnitude.

## Stage C: separate physical renewal length from geometry substep

Do **not** run a nominal `DA_PHYS_M=5, 2, 1 um` convergence sweep in v10.0.2.
The current engine uses `f.da` simultaneously as:

1. the physical length represented by one completed cleavage renewal; and
2. the geometry increment applied after that renewal.

Changing it therefore changes crack kinetics and source/wake translation as
well as numerical resolution. A later branch must introduce, for example:

- `renewal_length_m`: calibrated physical length per cleavage first passage;
- `geometry_substep_m`: numerical sharp-front update length;
- a transactional sequence that completes one physical renewal through one or
  more same-load geometry substeps, with FEM/J recomputation and no additional
  cleavage-clock consumption until the full renewal length is committed.

Only after that separation is tested should a 5/2/1/0.5 um geometry-substep
convergence study be interpreted as numerical convergence.

## Later gates

After Stages A--C:

1. geometry-substep convergence at fixed 5 um physical renewal length;
2. 300/700/1100 K three-class transfer and 50 um growth;
3. anisotropic deflection and multifront branching;
4. fatigue using the same validated MPZ/wake state.
