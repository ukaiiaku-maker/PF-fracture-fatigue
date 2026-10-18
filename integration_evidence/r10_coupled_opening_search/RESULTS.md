# R10 coupled opening fatigue/fracture continuation

The completed V2 descriptor bank was retained unchanged. A state-informed reduction was evaluated for all 24 paired smooth/bilinear and athermal/mixed arms (144 cyclic cases), followed by monotonic same-material fracture ramps for 12 arms and a 32-case current full-state 1-D block. The legacy cooperative opening clock was not used.

The inactive bilinear convention is now explicit: endpoint accessibility is `K_cut=25.90033965`, occupied bonded-branch loss is `27.65295455`, and the eight-phase discrete-wave loss is `28.63378215 MPa sqrt(m)`. None is treated as committed separation, and no production correction was fitted.

The reduced block produced no developed recurrent rate. The native block likewise produced **0 developed rates and 0 local slopes**: 25 supported 24-cycle non-advances, 5 target-truncated same-cycle cascades, and 2 unresolved mixed-crossover cases. Thus the strict joint Pareto set remains empty; cascade extensions and startup passages were not relabeled as Paris rates.

The monotonic ramp is distinct from the cyclic screen. Bilinear `01617` is weak-T with committed roots `[28,28,27,27,26,27,28]`; bilinear `07211` is also weak-T at `[27,27,27,30,31,31,31]`. Strong-plasticity `03832` is cold-censored through 200 MPa sqrt(m) and commits only at hot temperatures (`188,171,156` at 900/1050/1200 K), so it is recorded as hot commitment with a cold Kmax censor, not manufactured into a completed DBTT class. Unsupported mixed ramps remain unresolved.

`continuing_acquisition.csv` ranks the 16 unexecuted paired arms using mechanism diversity plus the observed current-state boundary. It does not enlarge or regenerate the descriptor bank. No new 2-D comparison is authorized by this block because no complete material produced a developed recurrent-growth interval.
