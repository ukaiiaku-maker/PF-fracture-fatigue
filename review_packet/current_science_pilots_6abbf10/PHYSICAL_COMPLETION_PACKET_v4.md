# Physical fracture completion packet v4

This packet supersedes the cavity and FEM/CZM interpretations in
`ACTIVE_GROWTH_COMPLETION_v3`; it does not replace or alter any accepted PF
trajectory. No manuscript text is included.

## PF physical-clock result

The accepted hot 1200 K path contains 203 geometry events and reaches
1004.925763 µm projected extension (1008.559512 µm path length). Native
stage-aware queries reproduce the requested brackets without interpolation:

| target (µm) | measured pre-event (µm) | committed upper (µm) | pre-event K_J (MPa√m) | model time (s) |
|---:|---:|---:|---:|---:|
| 100 | 99.482803541 | 102.254572373 | 101.678159551 | 1699.838488455 |
| 500 | 496.620883822 | 505.455206670 | 118.929516803 | 1699.838573900 |
| 1000 | 998.191926254 | 1004.925763476 | 110.872783699 | 1699.838679147 |

The first-to-last lower brackets span 898.709123 µm in 0.000190692 s of
accepted model clock. This is not a measured crack velocity. The prescribed
opening continues to increase; the controller collapses to an adaptive
fraction of `1e-8`, concentrating the seeded stochastic event sequence into a
very short model-time interval. `B` is a resettable within-event progress
clock, not cumulative hazard history.

The hot and cold developed paths have byte-identical stochastic geometry-event
files (203 events). Their temperature-dependent time, opening, reaction, and
K/J histories differ, but their fixed seeded event-length/mark sequence gives
the same final geometry. This identity must precede any physical
interpretation of the common endpoint.

Evidence:

- `PHYSICAL_HISTORY_HOT_COLD_v4.csv` — time, loading, reaction, native staged
  K/J, coordinates, source drive/backstress/shielding, analytical tip radius,
  active/wake populations, and energy ledgers.
- `LENGTH_QUERY_BRACKETS_HOT_COLD_v4.csv` — non-interpolated native brackets.
- `HOT_COLD_SAME_GEOMETRY_EVENTS_v4.csv` — event-aligned hot/cold comparison.
- `ACCEPTED_STEP_DIAGNOSTICS_HOT_COLD_v4.csv` — accepted `dt`, adaptive
  fraction, event flags, resettable `B`, and stochastic action/threshold.
- `physical_time_growth_hot_cold_v4.png`, `hot_growth_time_zoom_v4.png`, and
  `hot_accepted_step_diagnostics_v4.png`.

## Cavity result

The crack tip is at 592.599300405 µm and the cavity center is at 600 µm. Their
7.400699595 µm separation is not the solid ligament. The actual first
polygon-ray intersection gives a 7.358229034 µm solid ligament.

Every supported local contour crosses the cavity, so the reported local
isolated-tip J remains invalid. The current source nevertheless contains the
original, distinct complete-ligament topology operator. A polygon-versus-circle
certificate bug had falsely rejected a terminal point lying exactly on the
faceted cavity boundary; pass/fail now uses the actual polygon while retaining
the equivalent-circle result only as a diagnostic.

The read-only matched calculation gives 44.3878234 and 40.4943938 J/m² at
levels 3 and 4, respectively: 8.771391% difference against the existing 10%
limit. Both exact topology certificates pass, the accepted fingerprint is
unchanged, and neither physical time nor transport advances. Candidate-specific
dispatch uses this result only for the ray that reaches the cavity; all other
candidates keep their original observation. It is not labeled local J.

At the retained load, the original hazard law gives an effective rate of
`5.39093e-58 s^-1` and crossing time `2.23393e56 s`. Thus the supported result
is a qualified finite-ligament mechanical operation and no connection event in
a practical bounded window—not a general coalescence law.

Evidence: `CAVITY_LIGAMENT_QUALIFICATION_v4.json` and the hashed raw operator
record under `integration_evidence/current_science_pilots_6abbf10/cavity_ligament/`.

## FEM/CZM companion and execution gate

The historical checkout remains clean at `293491157063...`; its `a85b57...`
guard is unchanged. A separate companion worktree/branch,
`codex/current-pf-companion` at `fa24f7b`, binds the same native full-order
adaptive-CZM builder to the available immutable `d41b08...` family and
`1447653...` mechanical configuration. It explicitly does not claim to
recreate historical `e239710...`. Its focused suite passes 10 tests; the main
repository passes 787 tests.

The new one-step trajectory has not been launched because no durable,
independently mounted writable backup destination is currently authorized.
`/private/tmp` is on a separate internal device with ample space but is
ephemeral, so it is not accepted as the second copy. The prepared backup is
about 190 MiB; the gate script requires at least 512 MiB, refuses a same-device
or nonempty destination, copies source/configuration/manifests/checkpoints and
irreplaceable outputs, and verifies a SHA-256 manifest. No data are deleted.

Once a durable destination parent is supplied, run:

```bash
/Volumes/Data/working-papers/Fracture/shared-fracture-fatigue-options-v1/scripts/create_physical_completion_backup.sh \
  /durable/independent/parent/fracture-completion-20261010
```

Only after that succeeds, run the bounded full-order companion:

```bash
env \
  OUT=/Volumes/Data/working-papers/Fracture/shared-fracture-fatigue-options-v1/integration_evidence/current_science_pilots_6abbf10/fem_czm_current_companion_one_step \
  /Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion/run_current_pf_companion_one_step.sh
```

This is a one-accepted-step full-order calculation. It may legitimately end
with no fracture event; actual solver steps and fields, not a forced event,
are the qualification target. No push was attempted because no explicit
remote/destination was provided.
