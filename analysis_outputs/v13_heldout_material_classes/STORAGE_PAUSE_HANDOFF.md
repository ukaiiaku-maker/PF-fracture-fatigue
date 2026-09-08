# Held-out V13 queue — incomplete, storage paused

The original scheduler has exited after draining its workers. Nineteen cases completed normally, each with a first branch and the existing daughter-growth stop. Thirteen cases have never been launched. There are no active workers. This is a resource pause, not a scientific censor or a model-domain termination.

The Data drive has approximately 0.97 GiB free, below the inherited 1 GiB launch guard. No data were deleted, no case was restarted, and no physics source was modified. More durable free space is needed before continuing.

`HELDOUT_QUEUE_PAUSE.json` records the nineteen completed cases, terminal hashes, and exact thirteen unlaunched case IDs. It verifies every completed checkpoint manifest, checkpoint payload and final portable field hash. The raw scheduler pause is preserved in `ensemble/queue_pause.json`.

The unlaunched block begins with `theta15_rate1x_ceramic_1000K_seed3625`, followed by all four held-out groups for seeds 3626–3628. Resume **only these unlaunched paths** with a fresh scheduler ownership claim. The first-launch `--queue` command deliberately refuses an existing destination and must not be forced to overwrite it. Never rerun the nineteen completed cases.

Frozen physics: `35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`.
Accepted transition result: `b2e0e2d85e048553ad2593d895a98974356213e2`.
The 17,636 transition-record files and 3,620 earlier theta40 files verify unchanged.

The assessment's restricted means, medians, 26 primary-saturated births and five pending-companion births were independently reproduced. Seven launch-gate checks and five new reporting checks passed; relevant compileall and diff checks passed.

Reporting tools are prepared but no final eight-group result or review archive has been generated. **The full repository suite has not been invoked.** After the remaining thirteen cases terminate, stop the simulation program, invoke `scripts/run_v13_final_repository_suite.py` exactly once, then produce and verify the combined record with `scripts/report_v13_heldout_materials.py`. The reporter requires an incremental source Git bundle from the accepted result commit to its committed producer source. No merge is authorized by this pause record.
