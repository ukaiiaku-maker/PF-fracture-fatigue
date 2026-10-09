# Option-to-execution table

Resolution, construction, and execution are distinct states. `resolve_runtime`
performs an isolated import/source check and sets `execution_verified=false`.
`run_shared_fracture_options.py` constructs the selected native entry in the
executing process, records observed files and hashes, and only sets
`execution_verified=true` after the native entry returns successfully.

| Option | Native consumer | Applied state/callable | Role | End-to-end evidence |
|---|---|---|---|---|
| `mechanics=spatial_pf` | `fem.assemble_mechanics` | full spatial element assembly | physical | shared PF execution smoke; mechanics-delta tests |
| `mechanics=full_fem_czm` | pinned theta0 parity module | external `main` in `arrhenius-fem-czm` | physical | source/environment receipt; 4 native adapter tests |
| `loading=monotonic` | `sharp_front.run_2d` | displacement-step controller | physical | shared PF execution smoke |
| `loading=explicit_cyclic` | `fatigue_v1.FatigueWaveform` | phase-resolved waveform | physical | consumer-composition tests; native cyclic suites retained |
| `loading=fixed_delta_k` | `install_fixed_deltaK_waveform` | prescribed local range controller | physical | fixed-range consumer test; single-front native launcher retained |
| `loading=hold_dwell` | global scheduler | `load_history` hold/ramp consumer | physical | constructor mapping; shared top-level executor pending |
| source location | sharp-front support and/or `accepted_bulk_load_step` | tip, material-measure bulk, or composed support | physical | no-drop composition tests; donor bulk tests |
| source lifecycle | persistent installer, finite entry, or current-source binding | owning source ledger/state equation | physical | independent-axis tests; some composed shared executors pending |
| plasticity | tip process state or accepted bulk step | selected state update | physical | element/full-field donor tests; consumer test |
| crack path | sharp-front or V11/V12 ownership graph | single- or multifront topology | physical | branch provider tests; native restart families named |
| cavity | V5 production state builder | full-order cavity state | physical | cavity no longer implies branch birth or current-source law |
| kinetics | native, reversible installer, or V3.6 projection | signed transport/projection | physical/numerical | reversible parity suites retained; V3.6 composed executor pending |
| integration | adaptive transaction, global event scheduler, or coupled waveform update | accepted-state integrator | numerical | callable/hash attestation; component suites |
| output | native snapshot/plot flags | retention only | output | public smoke used compact output with spatial state active |
| material/kernel manifest | content reader/hash | attested input bytes | physical input | construction hash tests; execution rejects unmapped manifests |
| random seed | `CLEAVAGE_HAZARD_SEED` set before imports | native hazard RNG stream | stochastic | strict type tests and executed seed 1733 receipt |

## Current execution boundary

The shared API executes the established single-front monotonic, explicit
cyclic, fixed-range, V11 branching, V12 current-source branching, and exact
theta0 FEM entry mappings. A mapping is not claimed for bulk/cavity/hold/V3.6
compositions that lack a qualified top-level adapter. Their native modules and
legacy launchers remain present, their independent roles resolve, and `run`
fails before execution with a precise pending message. This is an open
integration boundary, not a claim that the physical capability was removed.

