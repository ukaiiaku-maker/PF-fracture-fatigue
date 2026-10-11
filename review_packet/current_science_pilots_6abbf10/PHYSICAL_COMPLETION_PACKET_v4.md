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

## FEM/CZM companion execution

The historical checkout remains clean at `293491157063...`; its `a85b57...`
guard is unchanged. A separate companion worktree/branch,
`codex/current-pf-companion` at `fa24f7b`, binds the same native full-order
adaptive-CZM builder to the available immutable `d41b08...` family and
`1447653...` mechanical configuration. It explicitly does not claim to
recreate historical `e239710...`. Its focused suite passes 10 tests; the main
repository passes 787 tests.

A durable independent backup was completed first on `/dev/disk3s5`. Its source
archive is pinned to main commit `d1d15bf` and companion commit `fa24f7b`; every
copied file passed its SHA-256 manifest. No data were deleted.

The gated full-order companion then completed one accepted step at 1000 K on a
2,020-node graded mesh. At 0.2 µm prescribed opening it gives reaction
2650.777674 N, native J = 0.0881706431 J/m², and native
K_J = 0.198053581 MPa√m. The crack remains at 0.500 mm with zero advance and an
empty cohesive advance log. The cleavage progress over the accepted step is
`8.42047e-101`, so this is a finite supported no-event result, not a forced
event or evidence of crack resistance at a developed load.

The run saved the solver step, crack/front coordinates, native traction and
J/K diagnostics, cohesive-element state, MPZ state, and full 3,900-point mobile
and retained bulk-density arrays. The local full-field constitutive state was
updated twice; spatial bulk-density transport is explicitly absent. The
authoritative theta-zero normalization records the requested 0° orientation;
some inherited detailed diagnostic summaries retain a legacy 45° label and
must not override that contract.

The complete 2.8 MiB output was copied after the run to the same independent
backup and all 39 files again passed SHA-256 verification. Compact evidence is
in `FEM_CZM_CURRENT_COMPANION_ONE_STEP_v4.md` and `.json`; raw output remains
under `integration_evidence/current_science_pilots_6abbf10/`. No push was
attempted because no explicit remote/destination was provided.

The reconciled capability/claim map is `EXECUTION_MAP.md`. `NEXT_BATCH_PLAN.md`
records why a broader temperature/rate/orientation command set is not yet
scientifically releasable and the exact measurements required before creating
that campaign launcher.
