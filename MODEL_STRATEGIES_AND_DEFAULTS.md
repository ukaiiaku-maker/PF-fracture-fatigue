# Model strategies and defaults

The authoritative machine-readable registry is
`arrhenius_fracture/restoration_registry_r1.py`.

| Identity | Default | Mechanics and geometry | Source/cavity scope |
|---|---:|---|---|
| `RESTORED_REDUCED_REFERENCE` | yes | Qualified v10.2.3 response operator and production signed moving-MPZ state; `FrontEngine.r_eff` owns kinetic geometry | Persistent-site geometry/source law; no invented reduced cavity |
| `FULL_SPATIAL_REFERENCE` | no | Anisotropic PF/sharp-front spatial mechanics, topology/remesh, analytical local kinetic radius | Independent persistent, finite-inventory, current-source, bulk, branch and full-order V5 choices where mapped |
| `SEAM_DIAGNOSTIC_V4_10` | no | Linear-elastic offset seam/cut in a circular body | Opt-in diagnostic only; no V5 cavity; never imported as a reduced/full default |

Source location and lifecycle remain independent axes. Tip-only, bulk-only and
tip-plus-bulk do not imply persistent, finite, or current-source lifecycle.
Cavity permission does not imply branch permission. The public presets
`pf_cavity_v5_single_front`, `pf_cavity_v5_disabled_control`, and
`pf_cavity_v5_cyclic_single_front` use the same native V5 paired adapter; the
cyclic identity requires at least two distinct positive `--load-factors` and
fails closed otherwise.

Supported structural histories remain distinct: monotonic displacement,
explicit cyclic, fixed local range, and hold/dwell. Native single-front
persistent paths map the signed reversible strategies; the full-order V5 path
currently supports positive accepted openings only. Remote-force capabilities
remain native-driver-specific rather than being inferred from a Williams-K
diagnostic.

The full FEM/CZM comparator is the separate clean repository
`Arrhenius_FEM_CZM_MPZ` at
`293491157063484bb10df6adc479f847a6a08ba6`; it is not substituted for PF arrays
or imported into the reduced identity.
