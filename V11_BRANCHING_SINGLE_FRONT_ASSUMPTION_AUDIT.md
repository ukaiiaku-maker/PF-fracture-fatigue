# V11 Single-Front Assumption Audit

This audit describes the production v10.2.30 path at source HEAD
`9e884fb0b0845da621d2612bdf1042e481b8df49`. It changes no physics. Older
multifront code in `sharp_front.py` is useful historical evidence, but the
v10.2.30 release contract explicitly selects one active nonbranching front and
does not make that older inventory a restart-complete v11 network.

## Findings

| File / class or function | Current single-front representation | Branching consequence and proposed migration | Domain / required tests |
|---|---|---|---|
| `arrhenius_fracture/sharp_front.py`, `FrontEngine.reset`, `step` | One scalar `B`, `N_em`, `a_adv`, `n_adv`, `W_emit`, `t`, and `K_prev`; one barrier pair and one `FrontConfig`. | Each active tip needs an owned engine state. Keep the engine untouched and place exactly one engine/state reference behind a stable tip ID first; later make competing candidate clocks explicit. | Physics/numerics; one-tip trace equality, independent-tip state, no ledger duplication. |
| `sharp_front.py`, `run_2d` and its `fronts` list | Geometry is locally represented by dictionaries containing one tip, direction, path, engine, status and lineage, but leading-tip scalars (`a_tip`, `a_killed`) and shared histories remain authoritative in several paths. | Replace dictionary conventions with a validated crack-network container. Adapt the current one-front path without changing ordering or calls. Do not promote the legacy `clone_split` rule as the v11 physical branching law. | Geometry/physics; deterministic ordering, topology, leading-tip compatibility. |
| `sharp_front.py`, `FrontEngine.clone_split` | Splits scalar renewal/plastic ledgers by a supplied fraction and deep-copies the engine. | This is not sufficient for persistent-site fields, stochastic identities, energy, or first-passage competition. Retain only as legacy behavior; branch birth must follow a documented state-ownership/energy transaction. | Physics/restart; conservation and explicit rejection of accidental clone duplication. |
| `arrhenius_fracture/stochastic_hazard_tip.py`, `StochasticHazardDiagnosticTipEngine` | One RNG, persistent threshold, integrated action, event index and threshold history per engine. | Assign deterministic stream identity per `(network seed, branch ID, candidate ID)`. Candidate clocks persist when they lose. A one-tip compatibility wrapper must not instantiate or draw from additional RNGs. | Stochastic/restart; fixed-seed trace equality, losing-action preservation, tied crossings. |
| `arrhenius_fracture/stochastic_avalanche_tip.py`, `_set_current_event_length`, module queue | One current threshold-correlated event-length latent and one module-global pending-geometry FIFO. | Move pending proposals under tip/candidate ownership for multi-tip use. In compatibility mode retain the exact existing queue and draw timing. Branch creation cannot redraw after veto. | Physics/geometry; event-length equality, queue isolation, veto/no-redraw. |
| `arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py`, `HazardEnergyGatedPersistentSiteCyclicTipEngine` | One `_energy_gate_pending`, one provisional flag, one event index, one checkpoint advance ledger and one event history. | Generalize later to a network transaction with explicit energy reservations. Initially expose this engine as the sole tip payload and do not alter its commit/restore sequence. | Energy/restart; full rollback, insufficient-energy veto, no double spend. |
| same, `prepare_energy_gated_event`, `commit_energy_gated_event`, `restore_geometry_veto` | Snapshots the single engine's RNG/threshold/action/history counters and restores one proposal. | A branch trial must snapshot network generation, all affected tips, candidate clocks, RNG states and global energy ledger atomically. Geometry is created only after energy acceptance. | Energy/geometry; transactional fault injection and exact state hash restoration. |
| `arrhenius_fracture/persistent_site_source_v10221.py` and cyclic/state-resolved engines | Persistent sites, signed mobile/retained populations and source diagnostics are attributes of one moving engine/MPZ state. | Each tip requires local ownership; overlapping process zones need a deterministic ownership or interaction operator rather than duplicated sites. | Physics; ownership conservation, overlap, translation/rotation. |
| `arrhenius_fracture/physical_shielding_v1022.py` and signed kernel providers | Shielding is evaluated for one current tip/geometry coordinate through a configured provider; production policy uses signed retained shielding, zero mobile and wake shielding. | Add a network shielding interface returning a signed value/state per active tip. Existing kernel validity for rotated/nearby branches is unproven; do not sum scalar values by assumption. | Physics; translation, rotation, parent/sibling interaction, sign tests. |
| `arrhenius_fracture/persistent_site_high_cycle_state_v10230.py`, `geometry_signature`, serialization | Captures one engine active vector and one geometry signature. | Define deterministic concatenation by branch ID and include topology/segment hashes. One-tip adapter must produce the unchanged v10 active vector. | Restart/numerics; ordering invariance and one-tip byte equality. |
| `arrhenius_fracture/persistent_site_high_cycle_checkpoint_v10230.py`, `write_checkpoint`, `restore_checkpoint` | One active-state NPZ plus JSON with one stochastic block, ledger set, geometry signature and high-cycle cache. | Introduce a versioned network checkpoint only after the compatibility layer. Validate unique IDs, parents, tips, lengths, actions, RNG state, energy totals and schema. Never silently load an incomplete network as one tip. | Restart; corruption, topology mismatch, uninterrupted/restarted equality. |
| `arrhenius_fracture/fatigue_v1.py`, `FatigueCycleHazardController` | Delegates cycle blocks and cleavage-clock commit to one `front`; a global block cap selects one trajectory. | Multi-tip fatigue needs simultaneous physical-time candidate integration and a global earliest crossing, not sequential per-tip blocks. One-tip delegation remains byte-for-byte unchanged. | Fatigue/numerics; analytical competition, partition and restart equivalence. |
| `arrhenius_fracture/persistent_site_high_cycle_engine_v10230*.py` | Cache keys, Poincare/DMD states, checkpoint hooks and event restart assume one engine and one geometry signature. | Network acceleration requires a state layout per stable tip and invalidation on topology generation. It is outside the initial compatibility increment. | Fatigue/restart; cache invalidation and multi-tip continuation. |
| `sharp_front.py` output section | Primary CSV/path names, histories (`a_tip`, `B`, `N_em`, etc.), snapshots and summary scalars select one leading/front state; legacy per-front files are partial. | Preserve all existing columns in disabled mode. Later add normalized branch/segment/tip tables keyed by stable IDs, without changing primary-column meaning. | Output/analysis; schema compatibility and deterministic serialization. |
| `scripts/analyze_v10_2_30_energy_gated_qualification.py` and fatigue analyzers | Read one event list, one path/projected extension and one fatigue time/cycle series. | Add network-aware analyzers separately; disabled-mode input and results must hash identically. Distinguish total physical length from primary projected extension. | Analysis; golden disabled outputs and branched accounting fixtures. |
| `scripts/run_v10_2_30_*` | One option, seed namespace, target extension, output root and one checkpoint directory; generated launcher binds a single repository root. | Add an explicit branching configuration later, default disabled. Record candidate set/network schema in provenance. Keep existing command generation untouched in compatibility mode. | Launcher/provenance; default-off and root-binding tests. |
| termination in drivers/launchers | Target is one projected crack extension or one scalar checkpoint-advance total. | Define primary projected extension, maximum-tip extension and total network length separately; initial mode maps all three to the one path. | Numerics/output; exact one-tip stopping step and branched termination rules. |

## Highest-risk assumptions

1. The production stochastic queue and energy proposal are singleton/global in
   places; naive iteration over tips would create ordering-dependent draws.
2. Fatigue block selection assumes one earliest first passage. Sequential tip
   evaluation would bias competition and consumed cycles.
3. The checkpoint's one active vector and one stochastic object cannot detect
   missing branches, so extending it permissively would weaken fail-closed
   restart.
4. Signed shielding kernels are calibrated for prescribed single-path geometry;
   rotation, nearby sibling interaction and process-zone overlap are not proven.
5. Energy-gated commits account for one event. Two independently accepted tip
   commits could spend the same stored elastic energy.
6. Output and stopping logic conflate leading projected extension, path length,
   and scalar cumulative advance.

## Migration boundary

The safe first code increment is a pure, validated crack-network data container
holding one stable branch/tip and a compatibility projection of the existing
path. It must not own an extra RNG, call the hazard engine, alter event order,
replace existing checkpoint bytes, or enter the v10.2.30 production driver.
Only after its one-tip invariants and disabled regression pass should adapters
move state ownership behind the container.
