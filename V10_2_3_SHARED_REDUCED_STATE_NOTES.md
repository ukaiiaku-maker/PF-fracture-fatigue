# v10.2.3 shared reduced-state equivalence

## Purpose

Replace the legacy scalar reduced DBTT front by a no-FEM execution path that
uses the same cap-free constitutive state and update operators as the v10.2.2
2-D solver.

The reduced calculation is not permitted to introduce its own shielding law,
source law, transport approximation, recovery law, or blunting state.  It uses:

- `MaterialManifest`;
- the spatial `UnifiedMPZState` fields for both reduced slip-trace channels;
- the campaign finite source budget and crack-advance-only source refresh;
- the anisotropic emission stress before the barrier;
- the promoted validated common Peierls--Taylor transport operator by default;
- the production near-tip Taylor back stress;
- retained-population recovery and mobile escape;
- accumulated-slip blunting;
- moving-frame active-to-wake translation;
- raw signed elastic dislocation shielding with no constitutive cap.

Only the mechanics closure is reduced.  The monotonic calibration mode prescribes
a scalar K ramp and supplied channel factors.  The replay mode consumes the
actual K, temperature, timestep, and channel-factor sequence recorded by a 2-D
run.

## Preserved fallback parameterizations

Three candidates are stored as immutable one-row material manifests:

| Candidate | Preserved role |
|---|---|
| `DBTT_A0002333` | large 2-D DBTT rise |
| `DBTT_A0003837` | strongest 2-D shielding sensitivity |
| `DBTT_A0002277` | non-cap-limited 2-D shielding state |

The legacy `max_K_shield_MPa_sqrt_m` field remains in these files because the
portable manifest schema requires it.  v10.2.3 treats the value only as historical
provenance; it does not enter cleavage, emission, fatigue control, or state
evolution.

## Mechanical closure

### Monotonic reduced mode

The applied opening scale is

```text
sigma_open = K / sqrt(2*pi*r_eff)
```

and the channel emission stresses are

```text
sigma_emit,s = max(f_s*sigma_open - sigma_back,s, 0).
```

The default 45-degree factors are the mean tensor-probe values from the initial
2-D transfer scouts.  They are not material parameters.  A later response atlas
may replace the constant factors by an interpolated geometry/state-dependent
mechanical operator.

### Exact outer-step replay

`sharp_front_v10_2_3` records every production anisotropic engine call:

- `K`;
- temperature;
- requested timestep;
- two channel factors;
- fired state and continuous micro-advance;
- scalar population and shielding diagnostics;
- final spatial mobile, retained, accumulated-slip, source, and wake arrays.

`compare_v10_2_3_shared_state_replay.py` reconstructs the production MPZ and tip
configuration, calls the same coupled engine step sequence, and compares the
complete final arrays.  No parameter fitting occurs in this gate.

## Required validation order

1. Run the 2-D/replay gate on `DBTT_A0002277` at 600 K, where the previous capped
   scout remained below the historical shielding reference.
2. Require matching fired flags, raw/effective shielding identity, and final
   spatial-state errors below the specified numerical tolerance.
3. Repeat for `DBTT_A0002333` and `DBTT_A0003837` after the first replay mismatch,
   if any, is resolved.
4. Only then run a new reduced endpoint campaign.
5. Optimize parameters only after the reduced endpoint ordering and shielding
   ablations reproduce the 2-D behavior.

## Explicit non-features

v10.2.3 does not add:

- a shielding cap;
- a fitted saturation factor;
- unsigned pair annihilation;
- temporal source recycling;
- an empirical bulk-plasticity field;
- a Paris law;
- a neural or black-box reduced model.

A physical saturation mechanism may be added later only after the uncapped shared
state demonstrates a reproducible deficiency and the added mechanism has a
sign-resolved or conservation-based formulation.
