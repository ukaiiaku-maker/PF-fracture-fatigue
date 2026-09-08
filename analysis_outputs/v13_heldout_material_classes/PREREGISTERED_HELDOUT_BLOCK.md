# Final V13 held-out material-class evaluation

Accepted transition record: `b2e0e2d85e048553ad2593d895a98974356213e2`.
Frozen physical execution source: `35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`.
Accepted θ40 record: `f09ec8468d4d95df73385742611f3de44455f865`.

The accepted mechanism is frozen. No physics source is modified. The new launcher delegates to the identical transition worker, changing only the registered material groups and output destinations. Its launcher/execution commit is recorded separately from the frozen physics commit. All 251 pinned source/helper/registry files are checked before each launch.

## Accepted scientific disposition

```text
continuous-emission fracture/plasticity parent: PRESERVED
inherited-companion versus renewed-primary race: PHYSICALLY EXECUTED
mixed first-branch incidence: DEMONSTRATED AT THETA=15 DEG, RATE=1X
material-sensitive first-branch behavior: DEMONSTRATED WITHIN TESTED MODEL
temperature sensitivity: SUGGESTIVE, NOT ESTABLISHED
first-branch onset distribution: DEMONSTRATED
recursive branch spacing: NOT TESTED
branch-angle prediction: NOT GENERALIZED
experimental calibration: NOT PERFORMED
predictive branching physics: NOT VALIDATED
```

Permanent boundary: `BRANCHING_KINETICS_MODEL_UNCALIBRATED`.

## Fixed evaluation

32 new bounded cases at θ=15°, rate=1x, seeds 3621–3628:

- DBTT (`v913_zeroD_sobol_0202500`), 300 and 1000 K.
- Ceramic-like (`oneD_v2_focused_ceramic_like_0018`), 300 and 1000 K.

Common random numbers across material–temperature groups at this fixed orientation. At most two physical PF workers. No Peak/weak-T rerun or orientation/rate rescreen. No changes to material rows, emission, cleavage, correlation time, directions, energy gate or event bookkeeping. The exact same existing qualified 15° family is reused.

Each case stops after first committed branch plus 20 µm daughter growth, at 75 µm maximum forward reach without a branch, or at an existing legitimate gate. The original discrete-event endpoint is preserved; a nonbranching endpoint may overshoot 75 µm and is administratively censored at 75 µm. A branch already committed inside the window may grow daughters beyond 75 µm. Early gates are not completed nonbranching observations. No automatic reruns or diagnostic campaigns are authorized.

The combined report will reuse all 32 accepted Peak/weak-T cases and show all eight groups: incidence, product-limit curves, restricted mean branch-free reach to 75 µm, paired-seed contrasts, first-event/extension distributions, saturation and pending-companion status, recorded exact pair-energy margins and final accepted topologies. Unevaluated native early-exit fields remain unavailable; no new mechanics is inferred or rerun to fill them.

DBTT and ceramic-like are held out from condition selection, not from all historical research. Results will not be used to retune the condition or model. After these 32 cases the simulation program stops, and the full repository suite is run once. A merge will not be declared qualified if the suite or record verification leaves unresolved failures.

`accepted_record_freeze.json` seals 17,636 accepted transition files. The existing θ40 seal additionally verifies 3,620 files. Native new outputs will live in `ensemble/<case>/` on the Data drive. No canonical result tables are populated before the held-out cases finish.
