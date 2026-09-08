# Complete source-resolution campaign publication

Physical implementation I: `63d53cdbad9125ef0d5180551974ac0e3a3ef721`.
Report-only recovery R1: `8d494165a987c282046b4694cccbc6dc8315a7dd`.
Path-join recovery R2: `e79da0661fe365a5b61cb38563e332ad5f727a1b`.

The one full physical A/B campaign, run **34205485025**, executed all 35
registered shards. Its original FAILURE is retained: the causal-neutrality
report lacked `rg` after all raw captures, and central assembly rejected its
incomplete execution record. Report-only recovery **34231419781** passed.
Assembly recovery **34255392088** exposed a path-subscript error before
reconstruction and remains FAILURE. Recovery **34256623174** passed complete
static/source/lifecycle ontology, exact A/B comparison, packaging, uploads,
and clean-worktree checks. Neither recovery repeated physical histories.

R2 changes exactly one SHA-bound orchestration expression in memory:
`args.shards[s['id']]` becomes `(args.shards/s['id'])`. Every scientific
function, predicate and tolerance remains unchanged. Both recovery helpers,
their six tests, source hashes and original failures are retained separately.
The physical I worktree was never modified by these report repairs.

## Final scientific evidence, identical in A and B

| Area | Result |
| --- | --- |
| Qualified fine production source / coarse fail-closed negative | PASS / PASS |
| Actual downstream first passage and sole child front | PASS |
| Ordinary child continuation | PASS |
| Source separation and causality | 6/7 gates; owned-radius causal-use FAIL |
| Static mechanics | 96 solves; 434/537 family predicates; 0/33 families; 8/12 derivatives |
| Physical transition partitions | 45/45 PASS; all 45 actual transitions occur |
| Common continued-terminal restarts | 11/11 exact; one common complete terminal |
| Controlled histories | 4/12 PASS; all 12 genuinely executed and classified |
| Lifecycle/legacy rollback | 39/39 PASS, including all 28 lifecycle stages |
| Natural ensemble | 32 seeds, 160 peers; 100/160 partition comparisons PASS; 160/160 midpoint restarts exact |
| Same-head disabled dispatch | 4/4 PASS |
| Historical future causal neutrality | 0/4 full gates; all four observed prefixes exact |
| Stagewise topology/front/length/inventory conservation | PASS; 282 actual-state rows independently ontology-valid |

Scientific qualification is **BLOCKED**, not a release candidate. Distinct
ownership `r_tip = 2.1499385833898107e-6 m != R_void` (approximately `5.5e-5 m`) is preserved;
that does not cure the separately failed owned-radius causal-use test.

The qualified source has global minimum quality `0.05370466567156866`,
normalized traction `0.036938352594854076`, log-rate error
`0.011087190514885492`, barrier error/kBT `0.0036957313563794766`, and
crossing-time relative error `0.011025954140189698`. Frozen limits are not
relaxed. The actual source event waits `4593376243926.258 s`; ordinary child
continuation waits `7.445685040821792e16 s`. These controlled demonstrations
are not evidence of downstream continuation inside the unchanged 16-us
natural-history window.

Connection, downstream nucleation and continued-front gates remain separate.
Four failed controlled cases retain `UNQUALIFIED_CAVITY_SOURCE_TENSOR`:
delayed_downstream, downstream_zero_drive, local_remesh_refinement and
negative_offset. A case label is not proof of zero raw drive; the preserved
source-quality failure must not be relabeled a physical zero-drive outcome.
The remaining controlled failures and all static/coarse failures remain in
their original source-bound rows. Historical 683/791 static evidence and raw
full-state neutrality FAIL at c4fbd2f are unchanged.

## Lossless publication and exact-head audit

`final_campaign_publication/` contains **79 bounded archive parts**, totaling
6,564,257,228 bytes, plus its publication and integrity manifests. The archive
retains **2,277 original files / 10,930,987,846 bytes**, verified locally by a
complete streaming hash audit after GitHub ZIP SHA-256 and CRC verification.
A/B compare exactly: 1,136 manifest entries per side, or 1,137 files including
each side's own manifest. Packaging is not another execution. The outer
assembly-repair receipt identifies R2 separately from physical implementation I.

Publication uses normal descendant commits and size-bounded non-force pushes.
The temporary recovery workflow is removed in the final evidence commit to
restore the non-artifact tree exactly to I; it remains recoverable in history.
The existing publication verifier must prove I/E ancestry, exact non-artifact
tree equality and artifact-root-only net differences.

The immutable capture-time ledger correctly retains `mission_terminal=false`
and then-pending publication requirements. Do not rewrite captured evidence
after CI. The final exact publication SHA, audit run, focused-test count,
actual repository-CI conclusion and terminal mission status belong in the
final PR #63 ledger after those checks terminate; PR #62 receives a linked
summary. At I, the complete focused registry passed **390/390**. Repository
tests had **1,137 PASS, seven inherited FAIL, one SKIP**; this is not green
repository-wide CI. Publication-head checks are still required.

No force-push, rebase, squash, merge, branch deletion or RC creation is part of
this publication. All earlier accepted checkpoint history is retained.
