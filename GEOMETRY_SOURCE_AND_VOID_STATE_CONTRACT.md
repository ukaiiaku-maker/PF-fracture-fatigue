# Geometry, source, and void state contract

## Ownership

- `r_eff` / `tip_radius_m` is the reduced kinetic/source-linked effective tip
  radius. In the persistent donor, its native `_source_geometry` routine owns
  source arc length, active front width, source area and multiplicity, including
  `M = rho_site * c_arc * r_tip * w_eff`.
- The full spatial sharp-front crack topology owns front position, path length,
  wake/remap and remesh generations. Its local kinetic radius is analytical; it
  is not claimed to be a measured rounded FEM shape.
- V5 cavity objects own `R_void`, area, boundary conditions, vacancy debit and
  cavity geometry generation. `R_void` is never added to `r_eff` and neither is
  relabeled as a 3-D spherical radius.
- Numerical core width, sampling length, source-area radius and physical cavity
  radius are distinct fields even when a chosen donor assigns equal values.

`live_geometry_source_void_contract()` reads the native owners; it does not
recompute a parallel source law. Its contribution ledger permits exactly one
resolved mechanics strategy, one unresolved source-resistance owner, one
kinetic geometry owner, and one optional cavity owner. Seam-cut redistribution
and persistent-radius attenuation are not combined.

## State and transfer

Signed mobile, retained, accumulated, returned and escaped populations remain
separate. Crack advance translates the moving window, resharpens locally under
the donor law, and retains wake state. Full-order cavity promotion/remesh carries
vacancy fields, ledgers, site phase, clocks, topology and keyed random ownership
through an atomic checkpoint. Conservation is checked as
`field inventory + annihilation + cavity debit = initial + plastic production + thermal exchange`.

The paired V5 adapter builds cavity-on/off arms from one common pre-divergence
state and common nonvoid random ownership. Both arms see the same accepted load
history. The toggle changes cavity physics only; it does not grant branch birth
or replace source strategy. A transport-bound failure retains the last accepted
pair and reports unused requested exposure rather than clipping concentrations.

## Restart

Reduced replay persists moving-MPZ arrays, source capacity, action, advances and
schedule position. V5 pair schema `shared-v5-physical-pair-checkpoint/2` persists
both arm checkpoints, physical time, interval count, load-factor history,
sequence phase/current factor, common-start fingerprint and configuration. A
restart with a different load history is rejected.
