# V13 transition-regime final record

Complete: 32/32 bounded followups terminated; no active or pending workers.

Decision: `MIXED_FIRST_BRANCH_INCIDENCE_DEMONSTRATED_IN_TESTED_MODEL` at 15° and the canonical loading rate, over the prespecified 75 µm first-branch window.

Permanent boundary: `BRANCHING_KINETICS_MODEL_UNCALIBRATED`.

| Group | Observed branches / 8 | Completed nonbranching observations | Early-gate censors |
|---|---:|---:|---:|
| Peak, 300 K | 8 | 0 | 0 |
| Peak, 1000 K | 8 | 0 | 0 |
| Weak-T, 300 K | 4 | 4 | 0 |
| Weak-T, 1000 K | 6 | 2 | 0 |

The selection seed 3621 is also the screen/production control. Excluding it, the 28 additional cases contain 24 branches and four completed nonbranching observations. Counts are descriptive; they are not calibrated material probabilities. Pair energies remain far from veto, so this is a kinetic transition, not an energy-boundary qualification. No recursive branch spacing is validated.

## Files on the Data drive

- [Final scientific report](final_transition_record/V13_TRANSITION_REGIME_FINAL_REPORT.md)
- [Incidence and censored first-branch curves](final_transition_record/incidence_and_first_branch_survival.png)
- [All 32 final topologies](final_transition_record/final_32_case_topologies.png)
- [Complete case table](final_transition_record/case_table.json)
- [Compact review archive](V13_TRANSITION_REVIEW.zip)
- [Accepted-ensemble provenance addendum](ACCEPTED_ENSEMBLE_AND_PROVENANCE_ADDENDUM.md)
- Full native results, atomic checkpoints and portable final fields: `transition_ensemble/<case>/`.
- Complete branch-disabled screening contexts and diagnostics: `parents/<case>/opportunities/`.

## Explicit provenance roles

- Accepted immutable θ40 result-record commit: `f09ec8468d4d95df73385742611f3de44455f865`.
- All 32 followup physical execution commits: `35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`.
- Final report-producer code commit: `95c98fc5b75addeba31a84f563d40a683c32f496`.
- Result-record commit: the Git commit adding this index and `final_transition_record/`, distinct from both execution and report production.
- `transition_source_increment.bundle` preserves the producer and preceding transition-source commits, requiring the accepted `f09ec84` base. Git bundle verification passed.

Review archive: 986,922 bytes, 25 manifested members plus the manifest.

SHA-256: `91b1ea8ab75fb0114b49be0f3985beac061c9f5eb509d3c8124a8049df929ab8`.

ZIP integrity and every member hash passed. The final report verified 257 source/result inputs and all 3,620 immutable accepted-ensemble artifacts. Four first-seed screen/production controls match physical state values; raw process hashes and differing run-local mechanics-call serials are retained. Within-run freshness gates were not changed.

Twenty preflight/production-fixture tests and ten reporting/gate tests passed (30 unique focused checks). The 14 non-mechanics source/report/gate checks were repeated after queue completion and passed. Relevant compileall checks and `git diff --check` passed. No full repository suite or new mechanical qualification campaign was run for this bounded task.

No θ40 reruns, loading-rate sweep, extra material groups, second branches, material tuning, emission changes, cleavage-barrier changes, or family rebuilds were performed.
