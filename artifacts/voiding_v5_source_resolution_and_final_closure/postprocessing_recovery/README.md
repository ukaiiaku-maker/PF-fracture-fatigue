# Retained-capture postprocessing recovery

Physical implementation: `63d53cdbad9125ef0d5180551974ac0e3a3ef721`.
Original single complete A/B campaign: `34205485025`.
Failed causal-neutrality job: `101995141502`.

Both independent A/B attempts completed their four historical and four current
worker captures, then failed in the consumer search because the hosted runner
did not contain `rg`. The original phase exit code is 1 and its
`execution_completed` remains false in the retained failed attempt. This is not
reclassified as an originally successful job.

The recovery installs ripgrep and performs no physical worker invocation. It:

1. Verifies the complete original inventory and exact missing-tool failure.
2. Reconstructs every terminal capture with its exact historical/current code.
3. Executes the report-only suffix of the original frozen scientific script,
   bound to SHA-256 `6a798ecd825fc495e4b722692491efb87ebfb3cf5ff991e2b7594c9c97c74a0e`.
4. Retains every original byte under `retained_failed_attempt/`, alongside a
   separately named repaired-postprocessing envelope and source-bound receipt.
5. Independently repeats that postprocessing from the retained original and
   requires exact recursive identity, then compares the separate original A/B
   captures' completed outputs exactly.

The repaired envelope means the retained physical capture plus report completion
is complete; it does not change the original return code or assert a new solve.
It identifies the physical SHA and recovery SHA separately. No scientific
predicate, tolerance, state field, clock, threshold, RNG, source, or event is
changed. The original future-neutrality failures remain scientific failures.

The recovery workflow is temporary orchestration. After the original campaign
has executed every independent shard and reached a terminal result, its
`assemble` operation uses the frozen physical implementation's complete
assembler with the explicit path-join repair below, and its unchanged packager
with the repaired neutrality envelope. Other failed
physical shards are not accepted by this bridge and require separate inspection.
All original failure evidence remains in the final lossless archive.

The later evidence-only publication restores the non-artifact tree exactly to
the physical implementation by removing this newly introduced temporary workflow
in a normal descendant commit. The recovery workflow remains recoverable in Git
history; no branch, accepted checkpoint, or original evidence is deleted. The
existing publication verifier must still prove exact non-artifact-tree equality
between physical implementation and final evidence head. No full physical
campaign is repeated.

Status at this implementation checkpoint: local report-only tests passed;
clean Linux recovery and final complete reassembly are pending. This document
does not certify a scientific PASS or a terminal mission result.

## Separately retained central path-join failure

Report-only recovery run `34231419781` passed, including independent terminal
reconstruction and exact A/B completion. The original physical campaign
`34205485025` then reached terminal FAILURE: 34/35 physical shards succeeded;
only the known missing-`rg` report failed. Central job `102158191169` correctly
rejected that original incomplete envelope.

Complete-assembly recovery `34255392088`, job `102159780607`, then exposed a
mechanical path error before any static reconstruction: `args.shards[s['id']]`
attempted to subscript a `PosixPath`. It is retained as a failed assembly.
The separately versioned helper now performs exactly one SHA-bound in-memory
replacement to `(args.shards/s['id'])`. The frozen script hash is
`7280e6e03f50915485aa87fb6700edfb99dcdd187760be48a8367cf025711601`.
Every scientific function, classifier, validator, and tolerance is unchanged;
the physical I worktree is not edited. The archive records the original and
executed assembler hashes, helper hash, explicit replacement, recovery SHA,
and failed run identity. No physical worker is called. Three focused tests
cover the exact bounded source change, changed-source rejection, and all 35
shard path resolutions through the actual orchestration main function.
