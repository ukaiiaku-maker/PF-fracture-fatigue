# K at crack length contract

Schema `fracture-observation-history/1` is chronological. Every observation records sequence, time/cycle, loading segment, front, event, load direction, observation stage, pre-event length, committed length, K components/effective K, J, G, contour identity, validity, and source provenance.

## Query rules

1. Filter by front, loading segment, observation stage and validity before selecting length.
2. Preserve repeated lengths. Loading, unloading, reloading and different stages at the same length are separate records.
3. An exact query returns every matching chronological record (`exact_repeated` when more than one exists).
4. If the target lies inside one accepted geometry increment, return `discrete_jump_bracket` and that event's pre/committed bounds and K. Do not interpolate within the jump.
5. Between independent observations, return lower/upper records with `between_observations_no_interpolation`. Do not blend stages or loading segments.
6. Invalid contours/observers are excluded by default but remain queryable for audit.
7. Requests beyond the achieved range return `out_of_range`; the last K is never silently extended.

The legacy adapter treats `KJ_Pa_sqrtm` on an accepted v10.2.27 event row as the event resistance associated with `(crack_extension_m - da_block_m, crack_extension_m]`. It does not relabel that value as a post-event re-equilibrated contour observation.

```bash
python -m scripts.query_k_at_crack_length steps_0300K.csv \
  --legacy-steps --target-um 1000 --output K_at_1000um.json
```

The plotting adapter retains repeated rows and groups curves by loading segment and observation stage.

## Qualification

The committed-mechanics 300 K full-PF run reached 1002.903 µm extension. Queries at 100, 500 and 1000 µm each returned `discrete_jump_bracket`; none was interpolated. The upper-bracket K_I values were respectively 32.0555, 43.5671 and 39.5423 MPa sqrt(m). The same adapter re-read 12 protected 1000 µm histories at all three targets (36 queries total), again preserving the jump bracket without interpolation. Exact rows, event identifiers, bounds and J values are in `integration_evidence/new_full_pf_1000um_summary.json` and `integration_evidence/protected_1000um_k_at_length.json`.
