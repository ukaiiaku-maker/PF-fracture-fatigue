# PF-fracture-fatigue: v10 unified sharp-front MPZ

This repository is the clean sharp-front successor to the mixed legacy
`PF-Fatigue_code` archive. The production fracture path is a discrete,
event-driven sharp interface. A binary broken-material indicator is retained
only for stiffness removal and crack-path representation.

The active architecture is:

```text
anisotropic FEM + directional J
    -> absolute Arrhenius cleavage and emission hazards
    -> unified finite-source active/wake moving process zone
    -> one first-passage renewal
    -> one mesh-independent sharp-front increment
    -> same-load FEM/J recomputation and optional multifront branching
```

## v10.0.1 correction

The original v10.0 two-dimensional gate was not dependency closed. Because the
older `arrhenius-fem-czm` editable distribution used the same
`arrhenius_fracture` namespace, a local environment could silently obtain
`crack_backend` and `coalescence` from the older checkout. That mixed execution
also advanced a legacy full-field plasticity model that was not parameterized
from the promoted ceramic/weakT/DBTT manifest.

v10.0.1 corrects this by:

- vendoring a sharp-wake-only crack backend with no cohesive/CZM imports;
- vendoring geometry-only crack-path coalescence;
- testing that every production module resolves inside the current checkout;
- using the unified active/wake MPZ as the plastic state in the initial 2-D gates;
- leaving the surrounding FEM elastic until a manifest-coupled full-field model
  is implemented and audited;
- retaining signed directional J as the production anisotropy/branching
  convention;
- recording `v10_0_1_driver_modes.json` in every 2-D output directory.

A clean anisotropic regression gives a monotonically increasing signed-J drive
before the first event. The one-front `abs_forward` diagnostic gives the same
values, demonstrating that directional-J sign clipping was not the failure.

## Integrated baseline

1. **Sanitized production stack.** The repository contains the sharp-front FEM,
   mesh, crystal, J-integral, unified MPZ, fatigue, and postprocessing modules.
   Peridynamics, S-N initiation variants, legacy variational-fracture entry
   points, and CZM geometry are excluded.
2. **Portable promoted manifests.** The promoted ceramic, weakT, and DBTT CSV
   rows are vendored under `arrhenius_fracture/data/materials`.
3. **Unified monotonic/fatigue MPZ state.** The active model uses finite source
   inventories, independent EXP-floor emission/Peierls/Taylor kinetics,
   encounter/Taylor retention, recovery, transport, blunting, signed active
   shielding, and a persistent signed wake.
4. **One-renewal geometry transactions.** A front accepts at most one physical
   `da` increment per FEM/J state. Excess cleavage action remains in `B` for the
   next equilibrium state.
5. **Matched-stress constitutive audit.** The required preflight verifies
   cleavage, emission, retention, source exhaustion, and shielding before a 2-D
   result is interpreted.

## Capabilities retained

- cubic anisotropic plane-strain elasticity and arbitrary crystal rotation;
- crystallographic cleavage-plane competition;
- signed directional and cluster/local J decomposition;
- multifront branching, lineage, starvation, stagnation, and coalescence;
- multi-tip graded remeshing and state transfer;
- monotonic first-passage loading and cycle-block fatigue;
- conservative branch splitting without duplication of source, mobile,
  retained, slip, or wake inventories.

The legacy full-field bulk/cyclic plasticity source remains available for future
porting, but v10.0.1 deliberately blocks it until its barriers and carrier state
are mapped to the promoted material manifest.

## Installation

A separate environment is strongly recommended because the older project uses
the same Python package namespace:

```bash
conda create -n arrhenius-sharp-front-v10 python=3.12 pip numpy scipy matplotlib pytest -y
conda activate arrhenius-sharp-front-v10
python -m pip install -e . --no-deps
python -m pytest -q
```

`pip` is listed explicitly because some minimal Conda Python environments do not
install it automatically.

Verify import provenance:

```bash
python - <<'PY'
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture import crack_backend, coalescence, sharp_front
for module in (arrhenius_fracture, crack_backend, coalescence, sharp_front):
    print(module.__name__, Path(module.__file__).resolve())
PY
```

Every printed path must lie inside this repository.

## Required preflight

```bash
OUT=runs/v10_0_1_preflight bash scripts/run_v10_preflight.sh
```

The preflight anchors itself to the repository root and puts the current checkout
first on `PYTHONPATH`, so it is not dependent on the shell's working directory or
on an older editable installation.

At 700 K the constitutive audit should show:

- ceramic: negligible emission during the cleavage waiting time;
- weakT: rapid source exhaustion and a finite retained population;
- DBTT: still faster emission and a materially different retention response.

## Initial two-dimensional gate

```bash
OUTROOT=runs/v10_0_1_three_class_700K_10um_v1 \
TARGET_EXT_UM=10 \
bash scripts/run_v10_three_class_700K_gate.sh
```

This gate intentionally uses one active front and
`BULK_PLASTICITY_MODE=tip_only` to isolate transfer of the promoted material
physics. It does not remove anisotropy or branching from the solver.

## Branching gate

Run only after the three-class transfer gate passes:

```bash
CLASS=DBTT T_K=700 TARGET_EXT_UM=20 \
bash scripts/run_v10_branching_gate.sh
```

The branching gate uses `root_signed` directional J and the full multifront
inventory.

## Fatigue gate

```bash
CLASS=weakT T_K=700 KMAX=17 CYCLES_MAX=1e6 \
bash scripts/run_v10_fatigue_gate.sh
```

The standalone fatigue controller delegates to the same `UnifiedMPZState` used
by monotonic fracture and does not impose a Paris law.

## Physical conventions

- Cleavage and emission use the promoted independent EXP-floor surfaces.
- Peierls and Taylor barriers have independent enthalpy, entropy, alpha, and n.
- Peierls motion creates obstacle encounters; Taylor completion releases
  retained dislocations.
- Wake slip is conserved and audited but does not create bridging or
  transformation toughening.
- Only signed mobile/retained line fields contribute to active/wake shielding.
- No empirical scalar backstress, `N_sat`, cohesive work, or smeared
  variational-fracture criterion is active.
- Crack advance remains an event-driven first-passage process.

## Validation status

Validated for v10.0.1:

- clean editable installation and import-provenance gate;
- compilation and integrated unit tests;
- matched-stress three-class constitutive audit;
- ceramic one-dimensional first-passage run;
- dependency-closed anisotropic 2-D signed-J driving regression;
- sharp-wake backend and branch-state conservation;
- standalone fatigue smoke using the unified state.

Long-growth R-curves, mature branch-state convergence, crack-quantum
convergence, manifest-coupled full-field plasticity, and publication fatigue
calculations remain future gates.
