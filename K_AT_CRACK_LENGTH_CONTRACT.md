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
