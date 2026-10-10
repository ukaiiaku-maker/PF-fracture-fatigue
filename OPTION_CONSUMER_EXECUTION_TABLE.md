# Option-to-execution table

Resolution, construction, and execution are distinct states. `resolve_runtime`
performs an isolated import/source check and sets `execution_verified=false`.
`run_shared_fracture_options.py` launches a fresh interpreter for every selected
native entry, records observed files and hashes, and only sets
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
| source lifecycle | persistent installer, `sharp_front_v10_1` finite-site engine, or current-source binding | owning source ledger/state equation | physical | public persistent and finite-site spatial executions; current-source V12 exact-kernel blocker recorded |
| plasticity | tip process state or accepted bulk step | selected state update | physical | element/full-field donor tests; consumer test |
| crack path | sharp-front or V11/V12 ownership graph | single- or multifront topology | physical | branch provider tests; native restart families named |
| cavity | V5 production state builder | full-order cavity state | physical | cavity no longer implies branch birth or current-source law |
| kinetics | native, reversible installer, or V3.6 projection | signed transport/projection | physical/numerical | live reversible method receipt; bounded public V3.6 full-FEM projection execution |
| integration | adaptive transaction, global event scheduler, or coupled waveform update | accepted-state integrator | numerical | callable/hash attestation; component suites |
| output | native snapshot/plot flags plus independent checkpoint policy | retention only | output | compact public run; publication profile three-generation checkpoint and fresh-worker continuation |
| material/kernel manifest | content reader/hash | attested input bytes | physical input | construction hash tests; execution rejects unmapped manifests |
| random seed | `CLEAVAGE_HAZARD_SEED` set before imports | native hazard RNG stream | stochastic | strict type tests and executed seed 1733 receipt |

## Current execution boundary

The shared API executes established single-front monotonic, explicit cyclic,
fixed-range, finite-site, operator-consistent V3.6, V11 branching, V12
current-source branching, and exact theta0 FEM entry mappings. The finite-site
and V3.6 adapters have bounded public execution evidence. V12 restore now reads
schema-2 additive fields and dispatches V11 versus V12 checkpoint arguments,
but the retained theta=40 V11 state cannot yet continue because its exact
`adb775...` signed-kernel family is absent from all retained caches. Bulk,
cavity, hold, and fixed-range branching shared compositions remain open; their
native modules and launchers remain present and are not replaced by a reduced
solver.
