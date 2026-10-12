# Shared options and compatibility

The public contract is `arrhenius_fracture.shared_options.SharedOptions` with schema `shared-fracture-fatigue-options/1`. Each run should save both the option JSON and the resolved runtime receipt produced by `arrhenius_fracture.shared_runtime.resolve_runtime`. The receipt names the concrete provider modules and identities; an option name by itself is not provenance.

## Independent axes

| Axis | Values | Meaning |
|---|---|---|
| mechanics | `spatial_pf`, `full_fem_czm` | Original spatial PF family or separately pinned full anisotropic FEM/adaptive-CZM repository. |
| loading | `monotonic`, `explicit_cyclic`, `fixed_delta_k`, `hold_dwell` | Loading ensemble; this does not select source geography. |
| source location | `tip_surface`, `bulk_uniform`, `tip_plus_bulk` | Where source support exists. |
| source lifecycle | `persistent_reusable`, `finite_inventory`, `current_source` | Whether sites persist, deplete/refresh, or are owned by the current-front provider. |
| plasticity | `tip_process_only`, `full_field_fixed_rho`, `full_field_evolving_rho`, `tip_plus_full_field` | Constitutive scope. |
| crack path | `single_front`, `branching_multifront` | Topology owner. |
| cavity | `disabled`, `full_order_v5` | Full vacancy-transport/cavity/topology process, not an effective damage toggle. |
| kinetics | `native_signed`, `reversible_signed`, `operator_consistent_v3_6` | Signed-state evolution/observation strategy. |
| integration | `adaptive_transactional`, `explicit_event`, `coupled_full_half` | Time/event integration strategy. |
| output | `compact_smoke`, `selected_fields`, `publication_restart` | Retention profile; it may not change mechanics. |

Source location and lifecycle are deliberately separate. For example, `tip_surface + persistent_reusable`, `tip_surface + finite_inventory`, and `bulk_uniform + finite_inventory` resolve to different receipts.

## Validated presets

The module exports these named presets:

- `pf_monotonic_tip_persistent`
- `pf_cyclic_tip_persistent`
- `pf_monotonic_bulk`
- `pf_branching_current`
- `pf_branching_void_v5`
- `pf_operator_consistent_cyclic`
- `fem_czm_theta0`

Resolve a preset with:

```bash
python -m scripts.resolve_shared_options --preset pf_branching_void_v5 --output receipt.json
```

## Early compatibility checks

- Bulk source geography requires a full-field plasticity mode.
- Full-field-only plasticity requires `bulk_uniform`; combined tip and bulk uses `tip_plus_bulk`.
- Full-order V5 cavities require spatial PF, current-source ownership, and branching/multifront topology.
- V3.6 operator-consistent projection is currently a single-front, tip-slip-patch spatial-PF strategy.
- The pinned full FEM/CZM pair does not claim the PF V5 cavity adapter or the PF V3.6 fixed-support projection.
- Unsupported or misspelled options fail before solver execution. There is no silent fallback or import-order switch.

## Runtime is not a physics selector

Elapsed time, an agent-session duration, output profile, checkpoint cadence, or
scheduler time slice must never select a different source lifecycle, transport
operator, state resolution, mechanics provider, barrier, completion law, or
event model. An interruption leaves the requested calculation incomplete and
restartable under the same physics. It is not a no-event physical result and it
does not authorize a finite-source, split-update, reduced, or historical
substitute. Numerical residual, conservation, admissibility, event ordering,
accuracy, storage-reserve, and genuine scheduler safeguards remain mandatory.

The option layer wraps existing versioned providers. It does not rename a reduced or prescribed-field diagnostic as a full spatial calculation.
