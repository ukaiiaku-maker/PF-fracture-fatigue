# PF Canonical Full-Trajectory Data Audit

## Decision

Complete chronological accepted-step PF model-native driving histories were
recovered and hash-verified for all **288** canonical cases. The long
table contains **432,710** accepted rows. No stochastic trajectory,
FEM/CZM run, or production-state mutation was performed.

## Source and join semantics

- Every `steps_*.csv` SHA-256 equals the corresponding published
  `steps_sha256` value.
- Row order is the raw accepted chronological order; the data were not sorted
  by crack extension.
- **419,669** rows participate in repeated-extension coordinates;
  these loading/reloading states are intentionally preserved.
- Native KJ is reported as **PF model-native KJ**, never as applied K or a
  conventional R-curve.
- Initial and reload-separated onset flags are exact joins on case,
  `pre_event_step`, and event-transaction index.
- Physical-avalanche membership is assigned only to event rows using the
  certified transaction-index ranges.
- The last accepted row of every target-reaching case is explicitly marked as
  a target-right-censored endpoint.
- Absolute projected crack length is retained because `a_tip_m` is explicit in
  every raw history; `a0 = a_tip - Delta a` was verified constant within each
  case.
- Physical time comes from the consolidated accepted-state observer. It is not
  reconstructed from `Kdot*time`.

## State availability

Accepted-state scalar histories include radius, front width, multiplicity,
mobile/retained populations, backstress, signed shielding, cleavage and
emission rates/actions, resolved signed shears, and channel-resolved emission
and transport values. The default-off observer did **not** archive complete
opening/channel tensor matrices. Such matrices must therefore be reported as
unavailable unless an exact deterministic frozen-state probe can reconstruct
the requested archived state; they must not be inferred from later states.

## Fingerprint

- `pf_canonical_full_step_trajectories.parquet`: `1e8b502c5e783a20cd777afb05e83ac858f77eb99e934643333691cb294edbdb`
