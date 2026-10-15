# FEM observer and PF K-versus-length lineage

## Result

The K-versus-length analysis remains attached to current native simulations.
No fracture trajectory was rerun.  Historical raw inputs were replayed through
the chronological schema, and a current 203-event native history was queried
and plotted with explicit stage, coordinate, quantity, and validity semantics.

## FEM J/G caller and representation lineage

| Historical source | Obligation retained in the current chain | Disposition |
|---|---|---|
| `94302bdb` / `kj_audit_v10056.py` | physical outer radius, clearance, contributing elements, plateau, load scaling | Preserved; current runtime metadata and support tests enforce these separately from the integral implementation. |
| `2b9122b1` / `fixed_grip_elastic_audit_v10058.py` | represented finite killed-notch fixed-grip derivative; full versus tension-filtered energy | Preserved as a representation-specific audit, not a sharp-crack replacement. |
| `5e1c0c4` / `production_j_parity_v10059.py` | actual production mesh, boundary, energy density, exclusion and segment policy | Preserved and mapped to current call sites; identical `j_integral.py` alone is not treated as runtime parity. |
| `879450ac` / v10.0.5.10 support modules | refinement support independent of inner tip spacing | Preserved; the qualified current series uses physical path-envelope support. |
| `23463cac` / `production_j_same_mesh_energy_v100511.py` | same-production-mesh derivative and full-energy J plateau | Exactly reproduced previously: G = 28.789526830110503 J/m² and median J = 28.468695078083666 J/m². It was not rerun. |
| `2934911` / Phase C | one-time legacy-radius conversion and runtime support metadata | Preserved in the current FEM source line. |
| PR 50 / v10.0.5.17 diagnostic | tension-only J, full-energy J, fixed-displacement drop | Historical advisory diagnostic only; the referenced commit is unavailable locally and no production-parity claim is inferred. |

The current reconciled FEM result remains unchanged: projected killed-band G
is a different finite removal operation, while matched material-coordinate G
agrees with native full-energy J within 0.8618%.  Signed J remains authoritative
for current directional driving; historical absolute-value display conversion
was not restored.

## PF K(a) analysis lineage

| Layer | Current producer/consumer | Verified disposition |
|---|---|---|
| v10.1.6 paired mechanism contrast | historical analyzer plus current pairing policy | The enabled-minus-opening-only change remains a descriptive first-order contrast. Geometry, load, stage, observer, and support match flags are required; it is not an exact shielding decomposition or constitutive correction. |
| v10.2.26/27 event intervals | retained long CSV and legacy plot helpers | Event-associated K is available only inside a recorded event interval. Interpolation across gaps and final-event carry-forward were removed. Pre-coordinate inference remains historical-only and is not used by the current native query. |
| schema `/2` chronology | `fracture_observation_history.py` | Repeated lengths, reversals, stages, front/event identity, projected extension, absolute length, path arclength, signed J and distinct K fields are retained without sorting or deduplication. |
| native event association | `query_k_at_crack_length.py --event-associated` | Returns the measured pre-event mechanics and linked committed geometry as different roles. It never labels the activation K as an equilibrium value at the target or committed coordinate. |
| plotting | `plot_fracture_observation_history.py` | Requires one named K quantity. No fallback among K_J, modal K, probe K, applied K, or effective K is allowed. |

## Same-raw-input replay

- The unchanged legacy CSV hash is
  `3e67545d96d9dfe8c47d83e3e59813699ccf8135e83f849c1bbbec4f8dd028ea`.
  All 2,085 accepted rows were retained: 201 event rows and 1,884 non-event
  rows, including 1,883 repeated committed-length rows. Stage and contour
  validity remain unknown. K_J is finite on every row; it is never relabeled
  K_I.
- The 201-segment geometry-aware replay reaches
  1002.9028827375072 µm projected extension and
  1005.0000000000018 µm path length. These endpoints are specific to that
  retained trajectory.
- The later current-native history has 1,906 chronological records and 203
  physical events. Its endpoints are independently
  1004.9257634758525 µm projected extension and
  1008.5595119085254 µm path length. K_J and signed/direct J are finite on
  1,703 mechanically evaluated rows; modal K, probe K, applied/effective K,
  and independent G are absent rather than zero.
- At 100, 500, and 1,000 µm, projected and path-coordinate queries preserve
  exact repeated matches or event brackets as the data dictate. No
  interpolation is performed.

## Restart and plot evidence

The 64-row uninterrupted and fresh-process-resumed histories are identical
after excluding only the output pathname. The 48-row prefix is retained
exactly, sequence identifiers remain unique, and no restart-boundary record is
duplicated or dropped.

The retained current plot selects exactly the 203 `pre_event` K_J records
against native path arclength. It contains no fallback to another K field, no
smoothing, no grid, and no title. The machine-readable receipt records every
input hash, query status, endpoint, quantity count, restart comparison, and
FEM historical blob mapping.

The plot's adjacent `current_native_203_event_KJ_vs_path.provenance.json`
resolves its result root to exactly one `LAW_TO_RESULT_MAP_V3.json` record.
The trajectory producer is historical commit
`d77abff138f9a7f23bbabba7b3ba582cde35ce68`, using the producer-bound
persistent-site adaptive transactional operator that predates the current
mean-net-rate-over-density-order repair.  It is therefore explicitly
`HISTORICAL_RETAINED_NOT_CURRENT_LAW_PREDICTION`.  Analysis commit
`0943748268d68624b03565d886cebf2b8502625d` only reads, queries, and renders
the existing trajectory; it is not substituted for the producer identity.
The sidecar and qualification receipt bind the law-map hash and eleven raw
artifact hashes.  No trajectory was rerun to add this provenance.

## Scope and unavailable evidence

This qualifies monotonic native K_J history and analysis transport. It does
not create phase-resolved cyclic reload observations, modal K, mixed-mode
evidence, or independent G where the source did not record them. The exact PR
50 commit and the four Library markdown files named in the addendum were not
present in the attachment directory or current workspace; their supplied
summaries are retained as references, while available local Git objects and
raw evidence were inspected directly.

Production execution, manuscript modification, remote writes, and historical
data deletion remain disabled.
