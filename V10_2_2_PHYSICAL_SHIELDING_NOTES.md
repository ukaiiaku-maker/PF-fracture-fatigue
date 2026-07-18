# v10.2.2 uncapped population-limited shielding

## Problem

The protected v10.1 campaign wrapper evaluated the signed elastic shielding field
from the active dislocation population and then imposed

```text
Kshield_effective = clip(Kshield_raw, -Kcap, +Kcap)
```

where `Kcap` came from `max_K_shield_MPa_sqrt_m` in the promoted zero-dimensional
manifest. That value was a calibration bound, not a constitutive law. It therefore
must not remain in a spatial moving-population model as an artificial saturation.

## v10.2.2 law

The effective shielding is now the raw signed superposition:

```text
Kshield_effective = Kshield_raw
```

The legacy manifest value is retained only in audit output as a reference. It is
not used by the tip stress, cleavage hazard, emission kinetics, source evolution,
or fatigue controller.

No replacement fitted cap is introduced.

## Physical mechanisms that can limit the population

The current model already contains the following population-level negative
feedbacks:

1. finite crack-tip source capacity;
2. local Taylor back stress reducing the emission driving stress;
3. Peierls--Taylor transport and escape from the active process zone;
4. retained-population recovery;
5. moving-frame transfer of active population into the wake as the crack advances.

These mechanisms, rather than a direct bound on `Kshield`, must determine whether
the dislocation population and shielding approach a steady state.

The tip stress remains naturally unilateral:

```text
Ktip = max(Kapplied - Kshield, 0)
sigma_tip = Ktip / sqrt(2*pi*r_eff)
```

Thus shielding may reduce the local opening drive to zero without clipping the
dislocation field itself.

## Numerical resolution

The v10.2.2 runner tightens `target-dN-emit` and `target-dN-store` from 0.25 to
0.05. This limits the amount by which the population can change in one tau-leap
block, so the uncapped shielding feedback is resolved between accepted blocks.
It is a convergence control and has no direct constitutive effect.

## Required acceptance checks

A run is accepted only if:

- the fixed-DeltaK audit remains exact;
- `constitutive_K_shield_clip_applied` is false;
- `legacy_manifest_cap_used_in_kinetics` is false;
- raw and effective shielding agree within relative `1e-12`;
- at least one shielding sample is recorded.

The audit reports how often the raw field would have exceeded the old cap and the
maximum raw-to-legacy-cap ratio.

## If the uncapped population runs away

Do not restore the hard cap. First determine which missing population mechanism is
responsible. Candidate physical extensions are sign-resolved annihilation,
thermally activated climb/cross-slip recovery, grain-boundary transmission, and a
self-consistent dislocation-free-zone or pile-up equilibrium. Each requires an
explicit state variable or rate law and a convergence/ablation test.

## Run

```bash
MODE=smoke bash scripts/run_v10_2_2_uncapped_shielding_sweep.sh
```

The smoke defaults compare `DeltaK = 24` and `27 MPa*sqrt(m)` for seed 1720.

```bash
MODE=pilot bash scripts/run_v10_2_2_uncapped_shielding_sweep.sh
```

The pilot defaults use `DeltaK = 21, 24, 27 MPa*sqrt(m)` and seeds 1720--1722.
