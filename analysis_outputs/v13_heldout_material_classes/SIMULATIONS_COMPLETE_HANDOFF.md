# Held-out simulations complete; publication storage hold

BRANCHING_KINETICS_MODEL_UNCALIBRATED

All 32 preregistered DBTT/ceramic-like cases are complete. The 13 previously unlaunched cases ran exactly once with no more than two PF workers. The 19 previously completed cases and accepted Peak/weak-T results were not rerun or changed.

| Held-out group | Branches / 8 | Completed 75 µm negatives | Early gates |
|---|---:|---:|---:|
| DBTT 300 K | 8 | 0 | 0 |
| DBTT 1000 K | 7 | 1 | 0 |
| Ceramic-like 300 K | 8 | 0 | 0 |
| Ceramic-like 1000 K | 8 | 0 | 0 |

The completed negative is DBTT 1000 K seed 3628. Together with the frozen Peak/weak-T block, there are 57 branches and seven completed nonbranching observations across 64 cases, with no early-gate censors. These are model outcomes, not calibrated physical probabilities.

The queue exited normally and is empty. All 32 terminal checkpoint manifests, checkpoint payloads, final portable fields and branch ledgers were verified. The 17,636 accepted transition files and 3,620 theta40 files passed their existing SHA-256 seals. See SIMULATIONS_COMPLETE_STORAGE_HOLD.json for eight-group descriptive metrics and all 32 terminal seals.

Data-drive free space declined from approximately 6.6 GiB at continuation to 0.28 GiB at final verification, faster than this queue's output growth. No files were deleted. The full repository suite has NOT been started; its one-time claim remains absent. No final publication archive or merge is claimed.

## Remaining work after durable storage is restored

Do not launch any further simulation, queue, reseed, replay, or restart.

1. Reverify the terminal seals and frozen source/accepted records.
2. Run scripts/run_v13_final_repository_suite.py exactly once, using the qualified conda Python and existing single-thread environment.
3. Preserve the report-producer HEAD in heldout_source_increment.bundle, based on b2e0e2d85e048553ad2593d895a98974356213e2; verify the bundle before analysis.
4. Run scripts/report_v13_heldout_materials.py to generate the complete eight-group tables, figures, native margins, paired contrasts, provenance and compact review archive.
5. Verify figures and archive member hashes; record any full-suite failures honestly. Do not automatically merge.

Current report-source commit: 2d3956f4145c6a92db73c049da67554792a2d25f.
Continuation launcher commit: fdf4380 (full hash in ensemble/continuation_claim.json).
Frozen physics source: 35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46.
No physics or canonical parameter changes were made.
