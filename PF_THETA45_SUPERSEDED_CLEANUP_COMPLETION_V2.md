# Theta45 superseded-directory cleanup completion

The revised V2 campaign lock was committed and pushed to GitHub at
`fa4b98b2aa864eb3a2a044de6f946c889a12469d` before cleanup.

On 2026-08-27, the following two interrupted theta45/rate0p01x directories
were permanently removed:

- `canonical_strain_rate__ceramiclike__T1050K__theta45__rate0p01x__seed3009675`
- `canonical_strain_rate__ceramiclike__T1100K__theta45__rate0p01x__seed3010684`

Both were recorded in the pushed V2 cancellation manifest, had
`FAILED_OR_CENSORED` status with return code -2, had no complete result, were
inside the authorized canonical run root, and had no active process owner.
The paths are not recoverable from the live run tree. No completed
theta45 extreme-rate supplemental directory was removed.

The 52 superseded extreme-rate conditions that never started had no directory
to remove. The pre-cleanup paused-stage audit remains immutable evidence of
the original 42 complete, two interrupted, and 100 pending theta45 cases.
