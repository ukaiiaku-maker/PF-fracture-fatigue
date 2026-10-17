# Executed result disposition

## Completed evidence

- Analytical acquisition: 32,768 initial rows, 32,575 raw-domain valid; 8,192 bound-triggered thermal-hardening rows, all raw-domain valid. The first box contained no DBTT/Peak proxy. The declared one-coordinate hardening expansion produced 2,301 rising-response proxies and no Peak-T proxy.
- All-secant analytical target audit: zero rows in either bank passed the frozen requirement that every adjacent secant lie in 3.4–4.6 over at least two rate decades.
- Native barrier-only comparisons: two exact complete materials, each with seven 300–1200 K fracture points and five 300 K fatigue amplitudes. Every fatigue point reached 21 accepted events, 71.154 µm committed extension, and an exact 20–60 µm developed window containing 12 events.
- Kinetic-CZM qualifications: the same two material hashes exercised nonlinear interface assembly/tangent, intact load–unload, accepted-event rupture, re-equilibration, broken-contact unloading without healing, disk restart, and saved bulk/interface fields. These are small implementation qualifications, not full scientific 2-D transfers with retained dislocation fields.

## Native outcomes

`R10D_COH_00167` is an information row, not a target success. Its seven-point initiation response falls from 4.704 MPa√m at 300 K to the hot numerical-accessibility floor. Its developed fatigue rates span 4.031 decades, but adjacent slopes are 25.516, 26.915, 28.063, and 28.761; endpoint slope is 27.232. This is recurrent developed fatigue, not startup averaging, and it fails the moderate-slope criterion.

`R10D_HARD_01138` is the selected hardening-bound row. Native initiation rises monotonically through 5.610, 8.178, 10.721, 13.269, 16.476, 19.362, and 22.349 MPa√m from 300 to 1200 K. The direction is reproduced, but the cause is the explicitly searched intrinsic cohesive hardening; it is not classified as a plasticity-created DBTT. Its developed fatigue rates span 3.962 decades with adjacent slopes 36.754, 37.906, 38.731, and 38.835; endpoint slope is 38.002. It also fails the moderate-slope criterion.

No 3×3 portfolio cell is qualified. The rising-fracture/opening-directed cell is retained as a boundary: complete native fracture and fatigue exist, but the fatigue slope is far outside the target and the thermal rise is intrinsic. Peak-T, Weak-T, plasticity-controlled, and mixed qualified cells remain `NOT_YET_FOUND` in the tested domain.

## Identity scope

- `EXP_FLOOR_REFERENCE`: preserved byte-for-byte at the reconciled base and covered by unchanged binding regressions; not rerun as a new result.
- `COHESIVE_BARRIER_ONLY`: implemented and scientifically exercised in the two complete 1-D material panels above.
- `COHESIVE_KINETIC_CZM`: implemented and exercised in two small production-assembly qualifications. A full moving-tip/dislocation-coupled 2-D scientific transfer is not claimed.
- `COHESIVE_RATE_INDEPENDENT_CONTROL`: implemented and exercised in kernel/mechanics tests only; no scientific result is attributed to it.

The detailed source tables, figures, and runtime fingerprints are under `results/`. Restartable native checkpoints and spatial fields are under `runs/r10_cohesive_arrhenius_joint_sweep/`.

## Verification executed

- Cohesive kernel, retained-material binding, reduced-reference compatibility, and sharp-front capability tests: 15 passed.
- Existing Paris-slope regression tests: 6 passed.
- Existing R10 global-search and developed-event-growth regressions: 13 passed.
- Total: 34 passed. All new Python modules and scripts also passed byte-code compilation, and the final staged patch passed Git whitespace validation.
