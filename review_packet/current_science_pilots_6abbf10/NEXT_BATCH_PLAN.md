# Next bounded study gate

No further temperature/rate/orientation campaign is released by this packet.
The completed work establishes one seeded cold/hot geometry sequence and one
low-load full-order FEM/CZM step; it does not establish replicate uncertainty,
rate sensitivity, or the storage cost of a multi-orientation field campaign.
Issuing production commands before those quantities are measured would violate
the controlling accuracy/storage condition.

## Required pre-launch measurements

1. Run one additional seed only for each of the two accommodation-active PF
   conditions, stopping at the same short accepted-event endpoint. Estimate
   first-event and early-growth seed variability before selecting a temperature
   grid.
2. Measure bytes per accepted event and checkpoint from those bounded runs.
   Require an independently verified backup with projected campaign size plus
   25% margin before any developed trajectory.
3. For loading-rate work, predeclare whether rate is varied through `dU`, `dt`,
   or both. Do not conflate controller resolution with physical loading rate.
4. For orientation work, require the authoritative theta field to agree in run
   arguments, normalized summary, and native diagnostics; inherited legacy
   labels are not sufficient.

## Command-release rule

After those four items are recorded, add a source-bound campaign launcher that
accepts only explicit temperature, `dU`, `dt`, theta, seed, endpoint, and a new
output directory; validates branch/head and free space; preserves
`threshold_scaled` events and all current signed/reversible physics; and refuses
overwrites. Its exact commands and output roots must then be committed with the
predeclared campaign matrix.

Until that launcher and matrix exist, the only exact executable continuation
retained here is the already completed bounded FEM companion command in its
source branch, `run_current_pf_companion_one_step.sh`. Re-running it is neither
needed nor authorized as a substitute for the missing accuracy/rate/orientation
design. The cavity operator similarly requires no forced continuation: its
present hazard predicts no connection event in a practical bounded window.
