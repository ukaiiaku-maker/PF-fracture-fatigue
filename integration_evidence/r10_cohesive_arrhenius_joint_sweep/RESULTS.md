# Executed result disposition

## Completed evidence

- Analytical acquisition: 32,768 initial rows, 32,575 raw-domain valid; 8,192 bound-triggered thermal-hardening rows, all raw-domain valid. The first box contained no DBTT/Peak proxy. The declared one-coordinate hardening expansion produced 2,301 rising-response proxies and no Peak-T proxy.
- All-secant analytical target audit: zero rows in either bank passed the frozen requirement that every adjacent secant lie in 3.4–4.6 over at least two rate decades.
- Native barrier-only comparisons: two exact complete materials, each with seven 300–1200 K fracture points and five 300 K fatigue amplitudes. Every fatigue point reached 21 accepted events, 71.154 µm committed extension, and an exact 20–60 µm developed window containing 12 events.
- Kinetic-CZM qualifications: the same two material hashes exercised nonlinear interface assembly/tangent, intact load–unload, accepted-event rupture, re-equilibration, broken-contact unloading without healing, disk restart, and saved bulk/interface fields. These are small implementation qualifications, not full scientific 2-D transfers with retained dislocation fields.
- State-aware acquisition: two nonoverlapping blocks totaling 25 complete materials spanning the opening, mixed, and plasticity strata were executed at five loads (125 load conditions), 12 cycles per load, and 32 phases. Fifteen retained their frozen parent coordinates and ten used authorized, raw-domain-valid joint dislocation-coordinate releases. The second block adds the available Weak-T proxy to the mechanism-balanced information design. None met the prespecified information or moderate-slope gate; the largest adjacent-slope displacement in the first block was 0.09894.
- Advancing kinetic CZM: `R10D_COH_10980` completed 15 accepted events and 73.527 µm at each of three loads in both the full signed-state and frozen-dislocation arms. The reported rates use the exact 20–60 µm event window and the connected finite-compliance landscape, not the barrier-only adapter.

## Native outcomes

`R10D_COH_00167` is an information row, not a target success. Its seven-point initiation response falls from 4.704 MPa√m at 300 K to the hot numerical-accessibility floor. Its developed fatigue rates span 4.031 decades, but adjacent slopes are 25.516, 26.915, 28.063, and 28.761; endpoint slope is 27.232. This is recurrent developed fatigue, not startup averaging, and it fails the moderate-slope criterion.

`R10D_HARD_01138` is the selected hardening-bound row. Native initiation rises monotonically through 5.610, 8.178, 10.721, 13.269, 16.476, 19.362, and 22.349 MPa√m from 300 to 1200 K. The direction is reproduced, but the cause is the explicitly searched intrinsic cohesive hardening; it is not classified as a plasticity-created DBTT. Its developed fatigue rates span 3.962 decades with adjacent slopes 36.754, 37.906, 38.731, and 38.835; endpoint slope is 38.002. It also fails the moderate-slope criterion.

No 3×3 portfolio cell is qualified. The rising-fracture/opening-directed cell is retained as a boundary: complete native fracture and fatigue exist, but the fatigue slope is far outside the target and the thermal rise is intrinsic. Peak-T, Weak-T, plasticity-controlled, and mixed qualified cells remain `NOT_YET_FOUND` in the tested domain.

For `R10D_COH_10980`, the 64-phase developed rates are `1.76616e-6`, `4.80428e-6`, and `5.62448e-6 m/cycle` at ΔK `18.88078`, `18.92601`, and `18.94862 MPa√m`. The adjacent slopes are 418.292 and 132.008 (endpoint 322.978), so this is a steep opening-controlled negative. Full-state/opening-reference rate ratios are exactly 1.0; the maximum mobile population is `1.472e-35` and maximum shielding is `3.070e-88 Pa√m`. The middle developed rate differs by `4.039e-5` between 64 and 128 phases. All energy-ledger residuals are zero, the broken patch owns the surface free energy, and the activation barrier carries no second surface charge.

The paired same-material kinetic-CZM thermal ramps resolve an event at every 300–1200 K point with a declared `0.05 MPa√m` load bracket and no physical or numerical censor. Full-state initiation is `[20.431, 18.478, 18.000, 17.750, 17.950, 17.450, 16.800] MPa√m`; the opening references are `[20.431, 18.478, 18.000, 17.650, 17.250, 16.800, 16.300] MPa√m`. The hot state raises the response by at most `0.700 MPa√m`, including a local 900 K rebound, but does not reverse the overall Weak-T trend into DBTT or Peak-T. Every terminal event remains on a metastable connected landscape. Physical times, initialization, terminal action, dislocation state, and root brackets are in `kinetic_czm_execution/R10D_COH_10980/thermal_curve.csv` and the underlying run records.

The earlier 8-, 16-, and 32-phase paths are retained as numerical history but excluded from qualified rates: their phase aliasing was exposed by refinement. The completed 64/128 comparison resolves that numerical issue. The independent historical `R10A_GLOBAL_002081` record also remains a local-slope failure (adjacent 1.703 and 6.242; endpoint 3.972); its endpoint alone is not promoted.

Four rows in the declared 72-material native queue (`R10D_COH_00185`, `04088`, `25753`, and `28647`) overlap the executed state-aware blocks. Their five-load local-slope pattern fails the every-adjacent-secant rule and their state response is opening-insensitive, so they are closed by that discriminating calculation without an unnecessary native ladder. Two prior native materials are complete and 66 native rows remain queued. Exact remaining IDs, accepted checkpoint paths, active-writer status, and the resume command are generated in `queue_status.json`.

## Identity scope

- `EXP_FLOOR_REFERENCE`: preserved byte-for-byte at the reconciled base and covered by unchanged binding regressions; not rerun as a new result.
- `COHESIVE_BARRIER_ONLY`: implemented and scientifically exercised in the two complete 1-D material panels above.
- `COHESIVE_KINETIC_CZM`: implemented in the small assembly qualifications and in the advancing finite-compliance reduced tip-cell comparison above. The latter retains the R10 signed dislocation state, accepted-event geometry advance, stochastic marks, residual waveform, and state/reference pairing. It is not claimed as a full moving-tip 2-D FEM transfer.
- `COHESIVE_PEAK_CUTOFF_CONTROL`: diagnostic peak cutoff only; it traverses `1-2/e` of the cohesive work and is not a matched-work control.
- `COHESIVE_RATE_INDEPENDENT_CONTROL`: matched-work envelope control with an explicit residual-work tolerance; no Arrhenius result is attributed to it.

The detailed source tables, figures, and runtime fingerprints are under `results/`. Restartable native checkpoints and spatial fields are under `runs/r10_cohesive_arrhenius_joint_sweep/`.

## Verification executed

- The final focused suite covers cohesive kernels/branches/energy controls, retained-material binding, Paris extraction, developed event growth, and the R10 global search: 30 passed in the pinned Python 3.12 environment.
- All changed modules and execution/analysis scripts pass byte-code compilation. The corrected small 2-D assembly transaction was also executed independently in a temporary directory and qualified with zero restart displacement error. Its assembled-bulk Schur stiffness reproduces the analytic connected separation to `1.58e-30 m` at the selected intact state. Its event is explicitly labeled a declared transaction fixture, not kinetic evidence.
- The full `continuation` command was rerun after completion. It resumed all six 64-phase and both 128-phase kinetic arms without replay, reused all 14 thermal arms, regenerated analyses, and refreshed the dynamic queue ledger.
- Git whitespace validation passed. Allocation counts remain distinct from executions in `continuation_ledger.json`.
