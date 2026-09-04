# V5 limitations and deferred work

> **STATUS: SUPERSEDED_INITIAL_FINALIZATION_ATTEMPT.** Results below are
> retained for provenance only. No PASS claim in this document is authoritative.

- Single void only; duplicate site-to-cavity creation fails closed.
- Two-dimensional plane strain only.
- Diagnostic parameters are not in the canonical four-class registry.
- No material calibration or experimental validation.
- No multiple-void interaction, coalescence campaign, or three-dimensional void.
- No fatigue calibration or fatigue-growth campaign.
- Exact V12 support certification excludes part of the coarse/long-ligament
  geometry matrix; those cases are preserved outside the demonstrated envelope.
- Full transition-specific partition invariance and stage-specific restart
  continuation are incomplete.
- Controlled variant rows share the reference production path; separate
  geometry-specific complete trajectories remain required.
- The current self-comparison is construction determinism only. Base-to-V5
  disabled neutrality is `NOT_RUN`.

These open items are systematic finalization blockers. They are not hidden by
relaxing tolerances, and V5 is not designated a release candidate.
