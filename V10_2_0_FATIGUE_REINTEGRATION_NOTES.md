# v10.2.0 fatigue reintegration

## Scope

This branch restores fatigue-cycle execution through the current v10 moving
process-zone and anisotropic-emission engine. It does not introduce a Paris law
or direct crack-growth-rate relation.

The inherited sharp-front driver already contains:

- tensile K-waveform quadrature;
- adaptive cycle-block selection;
- shared-cycle selection for multiple fronts;
- spatial fatigue/PZ field deposition;
- optional cyclic FEM/plasticity phases;
- branch-clock fatigue integration;
- cycle, hazard, and PZ diagnostics.

The stale coupling point was the front update: the driver called the old scalar
`FatigueCycleHazardController.cycle_step_front` even when the selected front was
a v10 moving-MPZ engine.

## v10.2 change

`fatigue_reintegration_v1020.py` temporarily dispatches two controller calls:

1. `integrate_one_cycle` predicts cleavage and emission using the current v10
   state, finite campaign source budget, channel factors, and Taylor back stress.
2. `cycle_step_front` calls the engine-native `cycle_step_waveform`, so the
   committed block evolves the moving mobile/retained fields, transport,
   recovery, source refresh, shielding, stochastic threshold, and variable
   event reward.

Legacy scalar fronts retain the old controller path unchanged.

## Stochastic defaults

The v10.2 entry point defaults to:

- exponential unit-mean cleavage thresholds;
- threshold-scaled, mean-preserving event lengths;
- the stochastic avalanche geometry backend;
- validated common transport with anisotropic emission.

Deterministic regression remains available by exporting:

```bash
CLEAVAGE_HAZARD_MODE=deterministic
CLEAVAGE_EVENT_LENGTH_MODE=fixed
```

## Initial validation stage

The first smoke test uses the scalar K-waveform mechanics surrogate
(`--no-cyclic-mechanics`) while retaining the full moving-MPZ front state. This
isolates cycle integration, adaptive blocks, stochastic renewals, source
budgets, transport, and geometry transactions.

Full cyclic FEM/plasticity is still available in the inherited driver, but it
must be validated separately because v10 currently runs `bulk_plasticity=tip_only`.
Turning on cyclic mechanics before that audit would repeatedly solve the body
without a mapped promoted bulk-plasticity law.

## Run

```bash
bash scripts/run_v10_2_0_fatigue_smoke.sh
```

The runner requires native moving-MPZ fatigue audit records, positive accepted
cycle blocks, stochastic defaults, and at least one positive variable-length
geometry event by default.
