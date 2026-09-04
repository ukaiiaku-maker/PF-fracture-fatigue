# V5 stochastic and timestep robustness

> **STATUS: SUPERSEDED_INITIAL_FINALIZATION_ATTEMPT.** Results below are
> retained for provenance only. No PASS claim in this document is authoritative.

The campaign records 35 partition rows (seven transition labels at partitions
1, 2, 4, 8, and 16) and 32 fixed natural seeds. Thresholds, integrated birth
hazard, hit count, RNG hash, state hash, outcome, and solver status are retained.

No-birth and incomplete trajectories are physical classifications, not software
failures. The reference values are intentionally uncalibrated. The current
bounded seed window predominantly reaches the stable-subgrid classification;
this distribution must not be interpreted as a material probability.

Only the integrated multi-hit birth partition path has full event-time/RNG
invariance coverage in the current runner. Equivalent complete transition-
specific partition histories for promotion, ligament, and downstream topology
remain deferred and prevent release-candidate status.
