# Canonical PF fracture campaign: storage-safety checkpoint

Date: 2026-08-27 (America/Los_Angeles)

This is an incomplete checkpoint, not the final campaign report and not a
campaign-result promotion.

## Completed work

- The canonical single-crack theta matrix is complete: 96/96 cases, covering
  four final V2 classes, 12 temperatures, and theta = 15 and 30 degrees.
- The canonical theta-45 strain-rate matrix is partially complete: 42/144
  cases. Two cases interrupted by the safety stop are explicitly
  `FAILED_OR_CENSORED` with return code -2 and must be rerun; the remaining
  100 rate cases have not started.
- All 138 completed PF cases close exactly between steps events and sparse
  event-boundary state profiles: 30,062/30,062.
- Observer state/profile conservation closes to a maximum absolute numerical
  residual of approximately 1.46e-11 line counts. All archived event probes
  are reliable, and all event radii and front widths are finite.
- The three legacy-compatible observer outputs were proven to contain exact
  duplicate records and consolidated fail-closed into one verified artifact
  per completed case. No profile or unique metadata field was discarded.
- Deterministic PF production-discrete sharp-wake mechanics/source maps were
  generated through 1000 micrometres at theta = 15, 30, and 45 degrees.
  Their load-scaling error is at most 7.8e-15. Bounds and leave-one-event
  interpolation uncertainty are explicit; extrapolation is prohibited.
- All 240 matched V2 1-D conditions were evaluated. Sixty-nine reached the
  target; 171 correctly stopped at the qualified source-drive radius bound.
  The bounded cases are retained as reduced-model failures, not extrapolated.
- The current 138 PF cases have matched 1-D rows. Of those, 88 are bounded in
  the reduced model and therefore cannot support target-to-target comparison.

## Safe storage work

- The earlier 8.423 GB extracted historical rate tree remains recoverable in
  the verified 634.1 MB `legacy_pf_fracture_pre_v2_theta0_rate_20260826.tar.zst`.
- The 3.854 GB unique historical bulk-PT tree was preserved in
  `legacy_pf_bulk_PT_positiveJ_fresh48_20260827.tar.zst` (300,585,835 bytes,
  SHA-256 `c5d88e266118f96c523d836ba6328089c43c6592c8e96cec82e4ec08365f44ad`).
  A full test extraction reproduced all 2,060 member sizes and SHA-256 hashes
  before the explicit extracted copy was removed.
- `1_backupdatatouse.zip` is retained.

## Genuine hard stop

The shared `/Volumes/Data` filesystem had only approximately 15 GiB free. An
out-of-scope protected fatigue job under
`Stateful_kitagawa_dadN/runs/sn_v9_four_class_m1/weakT/physical/1380/B000`
was producing approximately 93 MB immutable checkpoint-history members every
three minutes, about 2 GB/hour, and remained near block 310k of a 1M-block
ceiling. The monotonic-PF mission explicitly forbids modifying or stopping
active fatigue. Continuing the canonical rate matrix would therefore risk
exhausting the volume and corrupting both campaigns.

Only the canonical PF scheduler was interrupted. No fatigue process, source
worktree, registry, or completed trajectory was modified.

## Resume gate

Do not resume until the protected fatigue owner has stopped or compacted that
checkpoint stream and at least 30 GiB of durable free space is confirmed.
Resume from the frozen PF launcher worktree at commit
`998665899d15f818203d1742528462f21b99f7ed`; its incremental completion gate
will reuse the 42 completed rate cases and rerun the two interrupted cases.
After 144/144 rate cases pass, consolidate new observers, regenerate the V2
analysis, run the single branching capability demonstration, execute final
tests, and only then issue the final report and retention decision.
