# K at crack length contract

Schema `fracture-observation-history/2` is chronological and
provenance-preserving. `K_J`, interaction-integral `K_I`/`K_II`, effective K,
applied K, and probe K are distinct fields. A finite scalar is not evidence of
contour validity: validity is `valid`, `invalid`, or `unknown`, with a reason.

Each observation carries sequence, time/cycle, loading segment and direction,
front, topology generation, event/aggregate count, stage, pre/committed
projected extension, initial length, optional path arclength, observer/contour
identity, and source row. Absolute projected length is
`initial_length + projected_extension`; it is never silently confused with
extension or arclength.

`J_signed_J_per_m2` is the native signed direct-J quantity; its definition is
recorded in `J_definition`. `G_J_per_m2` is reserved for an independently
computed energy-release rate and remains null when no such calculation exists.
Neither field is reconstructed from K. Multiple events accepted during one
native step share that step identifier, carry distinct chronological sequence
numbers, and record `event_multiplicity`.

## Query rules

1. Select front, loading segment, stage, coordinate and validity before length.
2. Preserve every repeated length and return all matches in chronology.
3. Return `discrete_jump_bracket` within a geometry jump; never interpolate an
   uncomputed equilibrium inside that jump.
4. Return `between_observations_no_interpolation` between independent samples.
   Do not mix a front, owner, stage, segment, reversal or observer.
5. Return `invalid_or_unqualified_at_length` when a selected valid-only query
   excludes a known record at the target; this is distinct from unachieved
   `out_of_range`.
6. Never extrapolate beyond achieved growth.

The legacy steps adapter retains every input row. Because the historical CSV
does not record observation stage or contour-validity status, both remain
unknown. `KJ_Pa_sqrtm` maps only to `K_J_Pa_sqrt_m`; `KJ_probe_Pa_sqrtm` maps
only to probe K. `n_fire` is retained as an aggregate event count and its
`da_block_m` defines the event interval without fabricating sub-event times.

```bash
python -m scripts.query_k_at_crack_length steps_0300K.csv \
  --legacy-steps --target-um 1000 --coordinate extension
```

## Corrected reanalysis

The unchanged long-PF CSV SHA-256 is
`3e67545d96d9dfe8c47d83e3e59813699ccf8135e83f849c1bbbec4f8dd028ea`.
All 2,085 rows were adapted: 201 event and 1,884 non-event rows, including
1,883 repeated committed extensions. All rows retain native `K_J`; none is
reported as `K_I`. Queries at 100, 500 and 1,000 µm remain discrete-jump
brackets, but their stage and validity are honestly unknown.

The protected explicit-cycle case retains five accepted outer records, probe
K, event cycles and Kmin/Kmax target envelopes. Its CSV does not contain
within-cycle unload/reload samples, so a phase-resolved reload K(a) curve is
unavailable from that file. The limitation is recorded rather than filled with
synthetic points. Compact source tables and plots are under
`integration_evidence/corrected_k_history/`.

## New online event history

The qualified three-step full-spatial run under
`integration_evidence/option_completion_v2/event_online_qualified/` emits the
schema `/2` history while the native accepted state is live. It contains one
`reequilibrated` loading record and paired `pre_event`/`committed` records for
two accepted events. The pre-event rows retain the native direct J and K_J;
the committed rows retain the exact projected geometry jump and deliberately
leave K/J null. This solver does not perform a fixed-load solve immediately
after committing geometry, so the next native load step is the next separately
recorded equilibration. Intermediate lengths inside either jump return
`discrete_jump_bracket` and are never interpolated.

## Current developed-growth calculation

The current qualified full-spatial calculation under
`integration_evidence/targeted_completion/current_single_front_1000um/`
reaches 1002.9028827375072 µm projected extension in 2,085 accepted rows and
201 events at 300 K. Its regenerated schema `/2` history contains 2,286
chronological observations: 1,884 reequilibrated, 201 pre-event, and 201
committed rows. Signed direct J is finite on all 2,085 mechanically evaluated
rows; independently computed G is absent. The native steps table has no path
arclength, so that coordinate remains unavailable rather than being inferred
from projected advance.

Exact repeated-length queries are demonstrated near 5 µm. Targets at 100,
600, and 1,000 µm fall inside accepted geometry jumps and correctly return
`discrete_jump_bracket`. This monotonic calculation supports a developed
K-versus-projected-extension history; it is not presented as a phase-resolved
cyclic reload R-curve.
