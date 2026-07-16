# PF-fracture-fatigue: v10 unified sharp-front MPZ

This repository is the clean sharp-front successor to the mixed legacy
`PF-Fatigue_code` archive.  The production fracture path does **not** evolve an
AT1 or AT2 phase field.  A binary broken-material indicator is retained only
for stiffness removal and crack-path representation.

The active architecture is:

```
anisotropic FEM + directional J
    -> absolute Arrhenius cleavage and emission hazards
    -> unified finite-source active/wake moving process zone
    -> one first-passage renewal
    -> one mesh-independent sharp-front increment
    -> same-load FEM/J recomputation and optional multifront branching
```

## Integrated v10.0 baseline

The first public baseline implements development items 1--5 as one unit:

1. **Sanitized production stack.** Only the dependency-closed sharp-front FEM,
   mesh, crystal, J-integral, plasticity, fatigue, and postprocessing modules
   are included. Peridynamics, S-N initiation variants, mixed legacy patch
   scripts, and AT2 scientific entry points are excluded.
2. **Portable promoted manifests.** The exact promoted ceramic, weakT, and DBTT
   CSV rows are vendored under `arrhenius_fracture/data/materials`.
3. **Unified monotonic/fatigue MPZ state.** Both loading modes use the same
   finite source inventory, independent-shape EXP-floor emission/Peierls/Taylor
   kinetics, encounter/Taylor retention, recovery, transport, blunting, signed
   active shielding, and persistent signed wake.
4. **One-renewal geometry transactions.** A front accepts at most one physical
   `da` increment per FEM/J state. Excess cleavage action remains in `B` for the
   post-advance equilibrium state; it is never converted into several geometry
   jumps under one field solve.
5. **Matched-stress constitutive audit.** The preflight verifies absolute
   cleavage, emission, retention, source exhaustion, and shielding before any
   two-dimensional interpretation.

## Capabilities retained from the sharp-front archive

- cubic anisotropic plane-strain elasticity and arbitrary crystal rotation;
- crystallographic cleavage-plane competition;
- signed directional and cluster/local J decomposition;
- multifront branching, branch lineage, starvation/stagnation controls;
- multi-tip graded remeshing and field-state transfer;
- monotonic loading and cycle-block fatigue;
- full cyclic mechanics options in the two-dimensional sharp-front driver;
- stochastic/first-passage-compatible front clocks and deterministic mean mode.

The v10 front engine conservatively splits state at branch birth so no source,
mobile, retained, slip, or wake inventory is duplicated.

## Installation

```bash
conda activate arrhenius-fem-czm
python -m pip install -e . --no-deps
python -m pytest -q
```

## Required preflight

```bash
OUT=runs/v10_preflight bash scripts/run_v10_preflight.sh
```

This writes `matched_stress_v10.csv/json` and runs a ceramic first-passage
K-ramp. At 700 K the constitutive audit should show the qualitative ordering:

- ceramic: negligible emission during the cleavage waiting time;
- weakT: source inventory exhausted rapidly, finite retained population;
- DBTT: still faster emission and a materially different retention response.

## Initial two-dimensional gate

```bash
OUTROOT=runs/v10_three_class_700K_10um_v1 \
TARGET_EXT_UM=10 \
bash scripts/run_v10_three_class_700K_gate.sh
```

This gate intentionally uses one front to isolate material transfer. It does
not remove branching from the solver.

## Branching gate

```bash
CLASS=DBTT T_K=700 TARGET_EXT_UM=20 \
bash scripts/run_v10_branching_gate.sh
```

## Fatigue gate

```bash
CLASS=weakT T_K=700 KMAX=17 CYCLES_MAX=1e6 \
bash scripts/run_v10_fatigue_gate.sh
```

The standalone fatigue controller delegates to the same `UnifiedMPZState` used
by monotonic fracture. The two-dimensional `sharp_front_v10` driver also
retains the archive's full cyclic-mechanics and multifront fatigue path.

## Physical conventions

- Cleavage and emission use the promoted independent EXP-floor surfaces.
- Peierls and Taylor barriers have independent enthalpy, entropy, alpha, and n.
- Peierls motion creates geometric obstacle encounters; Taylor completion
  releases retained dislocations. There is no independent legacy trap barrier.
- Wake slip is conserved and audited but does not create bridging or
  transformation toughening.
- Only signed mobile/retained line fields contribute to active/wake shielding.
- No empirical scalar backstress, `N_sat`, or cohesive work is active in v10.
- Crack advance remains event driven. It is not an AT2 evolution and not a
  continuous expected-velocity approximation.

## Validation status

Completed locally before repository publication:

- Python compilation;
- nine integrated unit tests;
- matched-stress three-class constitutive audit;
- ceramic one-dimensional first-passage run;
- anisotropic two-dimensional FEM/J smoke;
- standalone fatigue smoke using the unified state.

Long-growth R-curves, branch-state convergence, crack-quantum convergence, and
publication fatigue calculations are not yet validated. They should begin only
after the 10 um three-class gate passes.

## Provenance

The geometry/FEM/multifront implementation was sanitized from the user-supplied
sharp-front hazard archive. The promoted material rows and unified MPZ
constitutive structure were ported from the Arrhenius FEM-CZM-MPZ development
line. The new repository deliberately separates these active components from
legacy AT2, peridynamics, and S-N code.
