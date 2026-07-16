# v10.0 validation gates

## Gate A: package and constitutive integration

```bash
python -m pytest -q
OUT=runs/v10_preflight bash scripts/run_v10_preflight.sh
```

Pass conditions:

- all unit tests pass;
- all three manifests resolve to their promoted candidate IDs;
- emission ordering at matched initiation stress is ceramic < weakT < DBTT;
- `one_renewal_transaction` is true and `n_fire` never exceeds one;
- active-to-wake state and branch split conserve inventory;
- fatigue reports `cycle_limiter=unified_hazard_state`.

## Gate B: 10 um three-class straight-front transfer

Run `scripts/run_v10_three_class_700K_gate.sh`.

Required checks:

- each class reaches the committed target;
- every geometry solve consumes at most one renewal;
- source, active, wake and discarded inventories balance;
- K shielding is separated into active and wake terms;
- no class silently falls back to the legacy scalar `N_em` model;
- initiation values remain ordered ceramic < weakT < DBTT.

## Gate C: wake ablation

Repeat Gate B with `--no-wake-shielding`. State must remain conserved while
only the wake contribution to K shielding is removed.

## Gate D: crack-quantum convergence

Repeat with da = 5, 2, 1 and 0.5 um, recalibrating the renewal-length mapping
rather than changing da alone. Compare K initiation, R-curve slope, event count,
source refresh, and wake shielding.

## Gate E: anisotropic deflection and multifront branching

Use `scripts/run_v10_branching_gate.sh`. Verify front-specific directional J,
conservative branch-state split, independent clocks, multi-tip refinement, and
no duplicated wake/source state.

## Gate F: fatigue

Use `scripts/run_v10_fatigue_gate.sh`, then the full two-dimensional cyclic
mechanics route. Confirm the same manifest and MPZ state are used in monotonic
and cyclic loading.
