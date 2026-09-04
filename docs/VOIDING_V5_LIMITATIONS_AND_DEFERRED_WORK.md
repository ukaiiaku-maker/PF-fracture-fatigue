# V5 limitations and deferred work

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
- Disabled neutrality covers production monotonic sentinels; an exact bounded
  unload/reload base-to-V5 comparison remains open.

These open items are systematic finalization blockers. They are not hidden by
relaxing tolerances, and V5 is not designated a release candidate.
