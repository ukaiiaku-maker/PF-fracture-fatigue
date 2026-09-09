# V13 four-material publication closure

PUBLICATION_RECORD_COMPLETE_MERGE_BLOCKED_FULL_SUITE_FAILURES

**BRANCHING_KINETICS_MODEL_UNCALIBRATED**

The physical simulation program is complete: 64 theta=15°, rate-1x cases, 57 first branches, seven completed nonbranching observations through 75 µm, zero early-gate censors. No new trajectory, restart, continuation, reseed, screen or physics change was performed for publication. The earlier storage hold is superseded by this verified package; its original record remains unchanged.

[Final scientific report and five figures](final_four_material_record/V13_FOUR_MATERIAL_FIRST_BRANCH_REPORT.md) · [Compact review archive](V13_FOUR_MATERIAL_REVIEW.zip) · [Closure verification](PUBLICATION_CLOSURE_VERIFICATION.json) · [Complete test inventory](final_four_material_record/full_suite_review.json)

## Scientific comparison

| Group | Branches / 8 | Median reach (µm) | Restricted mean, 0–75 µm (µm) |
|---|---:|---:|---:|
| Peak_300K | 8 | 10.95 | 13.25 |
| Peak_1000K | 8 | 9.66 | 12.12 |
| weakT_300K | 4 | 62.79 | 58.03 |
| weakT_1000K | 6 | 43.47 | 48.49 |
| DBTT_300K | 8 | 10.95 | 13.09 |
| DBTT_1000K | 7 | 33.81 | 40.77 |
| ceramic_300K | 8 | 10.95 | 15.83 |
| ceramic_1000K | 8 | 10.95 | 15.83 |

Peak, DBTT/300 K and ceramic-like are high-incidence under this tested condition. Weak-T shows the clearest mixed-incidence transition; DBTT/1000 K is delayed and slightly reduced. Temperature contrasts are parameterization-specific. Counts remain descriptive model outcomes, not calibrated physical probabilities. Recursive branch spacing and generalized branch-angle prediction remain outside the dataset.

## Native birth-state ranges

| Group | Primary / companion saturated births | Pending births | χ_B range | Native pair margin range (J/m) |
|---|---:|---:|---:|---:|
| Peak_300K | 8 / 7 | 1 | 0–0.85802 | 0.0422245654–0.106564554 |
| Peak_1000K | 8 / 6 | 3 | 0–0.512987 | 0.158738263–0.355093406 |
| weakT_300K | 4 / 3 | 0 | 0.13462–0.9787 | 0.0423365308–0.110781614 |
| weakT_1000K | 6 / 3 | 1 | 0–0.85485 | 0.0532840065–0.107082369 |
| DBTT_300K | 8 / 6 | 2 | 0–0.852154 | 0.0403447811–0.105216703 |
| DBTT_1000K | 7 / 7 | 1 | 0–0.831091 | 0.196261595–0.799196477 |
| ceramic_300K | 8 / 8 | 1 | 0–0.927174 | 0.0243901829–0.0451516017 |
| ceramic_1000K | 8 / 8 | 1 | 0–0.935839 | 0.0243780221–0.0454960521 |

All 57 primary channels are near the retained 1/tau_c asymptote at birth; ten births use an inherited completed-pending companion. Ranges above are rounded for display only. Exact native values, all opportunities (including unevaluated/infinite statuses), paired seeds and distributions are in the compact record. No new mechanics was used to fill missing native values.

## One-time suite and merge decision

Exactly one invocation: **1,099 passed, 5 skipped, 31 failed** (1,135 cases). All 131 V13-named tests passed. Failures comprise 24 missing historical V5/V6 product fixtures, six assertion/contract failures, and one status test blocked from running ps by the sandbox. The complete identities, tracebacks and skip reasons are retained; none has been waived or relabeled as passing. No second invocation was performed.

**Merge gate: NOT_CLEARED_FULL_SUITE_FAILURES.** The suite and final provenance have been reviewed, but failures remain unresolved. No automatic merge and no corrective physics/source changes are authorized by this record.

## Provenance

Frozen physics: `35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`. Accepted transition record: `b2e0e2d85e048553ad2593d895a98974356213e2`. Report-producer and one-time test source: `98f9b5d579e9d593c088bcfc1da9f7ccc8529093`. Execution commits remain separately recorded for every case. The incremental source bundle verifies against the accepted transition base.

Review archive: 2,969,823 bytes; 32 members (31 SHA-256-manifested members plus manifest). Integrity and every member hash PASS. Archive SHA-256: `35234f3e78afcbd6679b523aacf402ea025ad9b6ec48f982140729ddcb83e519`.

All five final figures were decoded and visually reviewed. All 192 held-out preflight seals and 12913 report-input hashes reproduce, as do the 17,636 frozen transition and 3,620 theta40 artifacts. No accepted data changed. Durable free space at closure: 9.17 GiB.

The compact archive contains the scientific report, figures, tables, full test records and source bundle. This index and closure-verification JSON are external verification sidecars, avoiding a circular archive self-hash.
