# Current-CZM HCF/VHCF continuation

## Decision

The corrected current-CZM calculation produced **no Paris pass**. Across 16 independently evolved athermal/mixed smooth/bilinear cases, the native phase-resolved engine accepted **12,040,000 cycles (12,040 s)** and observed no cohesive-cell advances. No startup secant, first-passage prediction, or analytical value is reported as a developed rate.

The result is nevertheless an HCF boundary result rather than another short prefix: the executed horizons span 10^4 to 10^6 cycles, every case passed the exact-versus-accelerated one-cycle overlap, and the trajectories record the changing EXP_floor state. Cases reaching the high-cycle stationary tail are classified as supported periodic-orbit shakedown. Cases that merely reached their declared horizon are finite-horizon physical runouts; they are not claims of infinite-life arrest.

## What changed

- Recurrence scoring now admits repeated bounded bursts across an actual 20–60 um developed window, while preserving unsupported-history and target-cascade censors.
- Opening stress is recomputed after every material-cell remap.
- A mixed deterministic unit-action crossing is an acquisition screen only; it cannot commit separation or carry phase action into a successor cell.
- The new HCF adapter uses the current athermal/mixed cohesive owner with the existing EXP_floor full-state engine. The legacy Arrhenius/cooperative opening clock is not called.

## Coverage and interpretation

- The amplitude/timescale map contains 144 points over four previously unexecuted plasticity backgrounds and all four shape/mode arms. 3 unsupported mixed crossover points were excluded rather than assigned rates.
- The executed block contains 4 supported periodic-orbit shakedown cases and 12 finite-horizon runouts.
- First-event times, inter-event waits, growth windows, and developed rates are null because there were no physical events. This is a measured outcome, not missing reporting.
- Same-material monotonic fracture classifications are carried in `same_material_fracture_context.csv`; fatigue runout does not overwrite those fracture results.

## Acquisition consequence

The next block should interpolate inside the already mapped bonded interval toward finite opening/plastic timescale coordinates. State-evolving arms may be continued from their exact checkpoints. Stationary-tail arms should not be repeated at the same load. A higher load is justified only by the mapped timescale/branch coordinate, not by runtime convenience.
