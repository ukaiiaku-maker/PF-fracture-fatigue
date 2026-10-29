# QSTE340TM native signed-bulk refit: first executed acquisition

Date: 28 October 2026

## Decision

The native full 2-D model now supports an opt-in, signed accounting of bulk
GND content swept into the crack tip.  The first paired calculations establish
that this pathway can preferentially increase blunting at high load ratio and
therefore moves the modeled R dependence in the experimentally required
direction.  Uniform bulk hardening and signed long-range shielding alone did
not produce the effect at the tested anchor.

This is a positive mechanism result and an executed first acquisition, not a
completed four-R refit.  The first curvature candidate was rejected because it
generated a renewal cascade at high R.

## Source and model identity

- Branch: `codex/qste340tm-restored-native-refit`
- Base HEAD: `fe3762d44d17681206260b570b122cc0cd16b5d4`
- Fit parent: `resolved_increment_1d/released_sloping_barrier_search/fit.json`
- Parent SHA-256: `3f874b33183874a022ce3fbd1e201905b93777e40caeb13f7a696107ac29024c`
- Temperature: 298.15 K
- Frequency assumption: 10 Hz
- Event length: 5 micrometres
- Loading: fixed native local Kmax controller; actual FEM/J K retained
- Mechanics: full 2-D FEM, phase-resolved cyclic plastic update, moving MPZ,
  existing EXP_floor emission/Peierls/Taylor kinetics, and existing cooperative
  cleavage completion law
- No experimental crack history was used to advance the crack.

## Implemented transaction

The bulk constitutive state remains separated into:

1. scalar SSD density for Taylor hardening;
2. signed zero-plastic-spin Nye/GND content derived from gradients of the
   saved plastic-strain field for long-range shielding; and
3. a new signed crack-swept tip transaction, resolved by slip system and lower
   or upper crack face.

For each accepted crack increment, the signed GND density is exactly
area-integrated over the material strip crossed by the tip.  The signed count
is projected on the outward normal of each face.  Only the positive
opening-effective component contributes to blunting, using the already fitted
`c_blunt * b * count` conversion.  Signed and absolute ledgers remain in the
diagnostics.  Local absorbed state convects with the existing blunting length.

The option is disabled by default.  It is explicitly a crack-swept absorption
calculation.  The bulk FEM still does not solve a time-continuous signed mobile
dislocation flux into the stationary tip, so this result must not be described
as such.

## Paired native results: saved sloping-floor parent

| R | Delta K (MPa sqrt(m)) | arm | event cycles | developed wait (cycles) | da/dN (m/cycle) | nearest experiment (m/cycle) |
|---:|---:|---|---|---:|---:|---:|
| 0.1 | 40 | tip only | 76, 131, 185 | 54.5 | 9.174e-8 | 5.823e-7 |
| 0.1 | 40 | signed bulk + absorption | 76, 140, 198 | 61.0 | 8.197e-8 | 5.823e-7 |
| 0.7 | 18 | tip only | 43, 66, 89 | 23.0 | 2.174e-7 | 4.179e-8 |
| 0.7 | 18 | signed bulk + absorption | 43, 95, 170 | 63.5 | 7.874e-8 | 4.179e-8 |

Bulk absorption changes the low-R rate by a factor of 0.893, but the high-R
rate by 0.362.  Equivalently, it increases the developed waiting time by 1.12x
at low R and 2.76x at high R.  This is the selective response needed to reduce
the excessive R separation.

At low R the three absorbed opening counts were 16.53, 0.187, and 0.341.  At
high R they were 166.34, 83.65, and 42.12.  The high-R native tip radius rose in
steps to approximately 3.9 micrometres; the low-R radius remained near
1.1--1.2 micrometres.  This explains the different post-event delays.

## Ablation at the high-R anchor

Using the preserved main-result material at R=0.7 and Delta K=18:

- tip only: events 44, 67, 90 cycles;
- bulk hardening plus signed GND shielding, without absorption: events 44, 67,
  90 cycles; and
- bulk hardening, signed shielding, and signed tip absorption: events 44 and
  110 cycles, with no third event by 200 cycles.

The shielding-only signed-GND contribution after the third event was only
-0.0030 MPa sqrt(m).  Bulk cycling changed the native tip radius from 1.039 to
1.095 micrometres without changing event timing.  The material difference
therefore comes from the explicit opening-effective tip ledger, not the old
uniform hardening law or the small long-range GND K term.

## Rejected acquisition candidate

`native_bulk_shape_01.json` retained the same plasticity vector but changed the
common opening barrier to floor=10.1 kBT, amplitude=13.0267 kBT,
sigma_b=15 GPa, and exponent=3.  It accelerated the low-R developed intervals
to 29 and 28 cycles, but the high-R events occurred at 31, 32, and 33 cycles.
That is a renewal cascade, not a developed fatigue rate.  The candidate is
retained as a bounded negative and is not a fit or finalist.

## Qualification

- 50 focused tests pass, covering signed reversible transport, moving-frame
  state, high-cycle serialization, restored native binding, signed face
  projection, and the new blunting ledger.
- `git diff --check` passes.
- The new capability is opt-in and does not alter accepted historical runs.
- Every reported rate uses inter-event waiting; no startup secant is mixed into
  the developed-rate table.

## Required continuation

The next fit should retain this native transaction and the sloping-floor
parent, enforce cascade rejection, and acquire complete materials over all four
R histories.  The parameters to search jointly are the permitted opening
floor/amplitude/stress scale/curvature and the existing emission, transport,
recovery, and source/blunting parameters.  The signed absorption coefficient
must not become a free empirical R multiplier.  Geometry/capture-width and mesh
sensitivity of the swept count must be qualified before treating its magnitude
as final.

The phase-field DRX repositories are useful for signed-conservation and
residual-Burgers conventions, but their grain-boundary disconnection physics is
not a drop-in crack-tip transport law.  The fracture model's native signed
transport remains the production owner.
