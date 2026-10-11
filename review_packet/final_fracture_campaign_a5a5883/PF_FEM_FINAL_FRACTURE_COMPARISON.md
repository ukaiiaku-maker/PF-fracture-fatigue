# Final PF–FEM fracture comparison

**Decision:** `ACCEPTED_WITH_DISCLOSED_BACKEND_DIFFERENCES`

The identical-mesh elastic reference is bitwise coincident at all three imposed openings. This closes the common geometry, boundary-condition, unit, stiffness, and base-operator checks before fracture or plastic accommodation is introduced.

| Scope | PF first-event K (MPa√m) | FEM/CZM first-event K (MPa√m) | Relative difference | PF opening (µm) | FEM opening (µm) |
|---|---:|---:|---:|---:|---:|
| Tip-only | 53.878 | 54.212 | 0.62% | 31.723 | 18.720 |
| Full-field | 51.110 | 54.407 | 6.06% | 26.610 | 12.492 |

Within PF, activating the exact selected-row bulk law changes first-event K by -5.14%. Within FEM/CZM it changes first-event K by 0.36%.

The long FEM/CZM full-field trajectory is classified `ACCEPTED_PHYSICS_AUDIT_EXPORT_FAILURE`. It completed four physical events and 20.827 µm of projected extension; only the post-run in-memory refinement-pointer audit failed. The persisted compact-mesh record verifies the 330 µm physical refinement radius, and a fresh one-step run completes normally with the recovery installed.

The cross-code K comparison is accepted. Total opening is reported but is not treated as a parity metric because the production PF and FEM/CZM meshes and J supports differ. The common identical-mesh control, rather than the production opening, is the appropriate base-mechanics parity test.

## Excluded prior evidence

- `pf_peak1000_theta0_seed8666_full_field_matched` — runtime builder ignored selected direct PT entropy/shape fields.
- `fem_czm_peak1000_theta0_seed8666_first_passage` — claimed exact overlay did not replace active v9.11 explicit mobile/retained controller.
