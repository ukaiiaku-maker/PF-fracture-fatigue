# V13 initial qualification — continuous-emission parent preserved

Permanent boundary: **BRANCHING_KINETICS_MODEL_UNCALIBRATED**.

The requested restoration, behavioral parity, historical impact, canonical
parent trace, and initial conditional-mark implementation are complete. This
is initial software/mathematical qualification, not validation of predictive
branch probabilities or authorization for a new PF trajectory or atlas.

## Source and evidence

- Branch: `codex/v13-rehydration-and-conditional-mark`.
- Restoration and canonical-parent producer: `c359ee015c606a0378a2bad548faa58f407bef10`.
- Conditional-mark producer: `23e55e0872af251e46be6b0078a786b72fa3f03f`.
- Earlier preflight `81dd9a1` and historical V12 raw outputs remain unchanged.
- Records: [initial qualification directory](analysis_outputs/v13_initial_qualification/).
- No material row, cleavage/emission barrier, production multihit law, family,
  historical checkpoint, or historical trajectory was changed. No new FEM
  equilibrium solve, PF trajectory, seed screen, or physical ensemble ran.

## 1–2. Exact runtime rehydration and behavior

`current_source_runtime_bindings.py` selects a closed registry using serialized
source model, signed-state model and transport-mode IDs. It binds methods only;
it does not run a constructor, replace arrays, consume RNG or initialize state.
Unknown/inconsistent IDs, missing required state, method/data collisions and
unregistered instance callables fail closed before binding.

The production source ID remains
`persistent_areal_sites_backstress_limited_no_inventory`. The signed-state ID
remains `v10.2.5_shared_signed_burgers_population`.

| MPZ method | Intended implementation |
|---|---|
| `_emit` | `persistent_site_source_v10221._persistent_emit` |
| `advance` | `persistent_site_source_v10221._persistent_advance` |
| `diagnostics` | `persistent_site_source_v10221._persistent_diagnostics` |
| `_persistent_base_diagnostics` | `signed_burgers_shared_v1025._signed_diagnostics` |
| `evolve` | signed scalar or channel-resolved implementation selected by serialized ID |
| `active_K_shielding` | `signed_burgers_shared_v1025._active_K` |
| `wake_K_shielding` | `signed_burgers_shared_v1025._wake_K` |
| `shielding_K` | `signed_burgers_shared_v1025._total_K` |

There are no dynamically installed engine-instance methods in the qualified
initialized reference. Python callables are not added to checkpoint payloads.

The A reference runs the actual production constructor, then receives the
identical frozen physical state while retaining its constructor-installed
methods. It does not use the rehydration function. B is capture plus corrected
restore. The qualification compares every serialized physical field recursively,
all interval outputs and installed callable identities—not just scalar rows.

All eight material/temperature states pass all three interval checks (24 total):
zero duration, negligible emission, and a loaded interval of `1e-10 s`. Seven
loaded intervals emit; DBTT/1000 K is physically blocked in this particular
frozen state and is explicitly labelled `loaded_emission_blocked`, not active.
The record includes populations, signed ledgers, available sites, backstress,
shielding, radius, multiplicity, source counters, actions, thresholds, endpoint
and RNG. Every source checkpoint SHA-256 is checked before and after.

## 3. Historical impact is material

Only isolated process intervals were evaluated; neither result was written
back to an accepted checkpoint.

| Frozen source | Old fallback emitted increment | Correct persistent increment | Different serialized fields |
|---|---:|---:|---:|
| Peak/300 K | 0 | 4.128343432448399e-6 | 81 |
| DBTT/1000 K | 278.466714094464 | 0 | 69 |

Both use the same `1e-10 s` frozen interval and owner-cached loading. Thus
metadata survival did **not** imply intended emission-law execution after
restart. The old atlas remains topology/software capability evidence; its
post-restart process histories must not be cited as quantitative demonstrations
of the intended persistent-source law. This audit does not repair those
histories retrospectively or authorize their rerun.

## 4. Immutable canonical parent

The cached branch-disabled weak-T/700 K/theta40 control, step 301, invokes the
actual `production_step_loop_v11.advance_accepted_step`, production process
hook, one physical renewal and existing exact single-arm transaction. The
reference is the `run_2d(maximum_fronts=1)` path, not the legacy V12 pair selector.
Mechanics cache hashes, topology and energy are checked; the cached/logged
opening difference is one float64 ULP (maximum allowed: four ULPs), with no
interpolation or rescaling. Stress is assembled from cached displacement, not
obtained by a new equilibrium solve.

- Winner: `cleave:cleavage:(010):694b159e5c3c21b95300`.
- Primary event ordinal: 4.
- Raw first passage: `2410.8000040824813 s`.
- Accepted single-arm endpoint: `2410.800004111201 s`.
- Accepted opening: `5.740000009788535e-5 m`.
- Raw-to-accepted difference: approximately 28.72 ns, the canonical outer
  interval endpoint—not the legacy `tau_c` correlation-ready delay.

Initialized and restored canonical records agree exactly. Continuous emission,
directional actions/thresholds, winning event, endpoint, process state and RNG
remain the parent mechanism. No discrete emission-versus-cleavage race was added.

## 5. Default-off conditional mark

`accepted_step_overlay_v13.advance_marked_accepted_step` calls the canonical
interval exactly once and evaluates a mark only after accepted single-arm
cleavage. It is not enabled in the campaign CLI. Disabled and precleavage paths
do not evaluate channels, branch RNG or pair mechanics.

For each companion the unchanged raw Arrhenius arrival rate and existing
multi-hit order define a gamma waiting-time law. Branch-only junction and
continuous overlap barriers multiply that raw rate. For multiple companions,
the first completed subgrid embryo wins. The single probability is the product
of companion survival probabilities; companion probabilities integrate each
waiting-time density against all other survivals. Their sum is one, and the
marked intensities sum to the original cleavage intensity. The baseline
effective multi-hit rate and its `1/tau_c` asymptote are untouched.

Exposure is a subgrid mark coordinate: it advances neither global time nor
emission. Overlap decreases continuously as `exp(-distance/process_zone_length)`;
there is no owner-handoff-distance branch-permission switch. Quenched disorder
and cooperative assistance default to zero. No branch parameter is calibrated.

The branch RNG is stateless SHA-256 counter based, keyed by model, branch seed,
front lineage, parent event ID/ordinal, companion ID and opportunity ordinal.
Sibling and generation keys are distinct; baseline RNG is not used.

The marked pair adapter calls the existing `execute_topology_trial` with its
existing geometry/energy callbacks and tolerances. A private two-member
reservation adapts the legacy transaction API; it is **not** a companion
cleavage event in authoritative clocks. On acceptance, the exact canonical
competition, RNG, process state and physical event count are retained. The
primary endpoint must remain unchanged. Pair topology release/cost replace the
single topology ledger increments; emission ledgers remain the baseline values.
No companion or energy rejection returns the identical accepted single object.

The exact transaction adapter is tested using small analytic-energy fixtures
and real topology operations. A positive two-arm PF mechanics solve was not
performed; future production wiring must supply the same qualified exact pair
provider and owner-transition callbacks. Initial adapter tests are not a new
multifront production qualification.

## 6. Initial validation and limitations

The actual cached parent runs through the new accepted-step adapter with V13
disabled and enabled. Full parent/process/clock/RNG fingerprints match. The
enabled canonical sentinel uses an empty admissible companion inventory and
therefore proves precleavage and no-companion parity; pair acceptance/rejection
is separately covered by exact transaction unit fixtures.

Frozen Monte Carlo covers four archived parameterizations at 300 and 1000 K:
20,000 mark-only draws for each of 9 illustrative settings, 72 configurations
and **1,440,000 draws**. All analytic probabilities lie within their exact
99.9% binomial intervals. The mark-only settings are exposure
`1e-12/1e-9/1e-6 s`, with `(junction, overlap)` barriers
`(0,0)/(0.1,0.1)/(0.3,0.1) eV`, fixed branch seed 130001, and zero junction
separation. Process-zone length is read from each source state.

These use the complete rehydrated historical state and its saved owner opening
as a **frozen scalar stimulus**. They do not reconstruct candidate-specific
post-cleavage tensor mechanics or elect a new physical parent. Inventory
eligibility is not exact pair-geometry acceptance. Full checkpoint/process/
stress hashes and material-manifest hashes are recorded. The resulting
probabilities must not be ranked as validated material-dependent branching
predictions. The figure is a sampler-validation plot, not a physical atlas:
markers are sampled frequencies and lines merely connect the three tested
analytic settings, not a densely evaluated probability curve.

The focused tests cover runtime parity, analytic laws, independent RNG,
stale/duplicate identities, continuous overlap, exact primary preservation,
pair acceptance/rejection, energy-ledger accounting and checkpoint transactions.
The focused set passes 57 tests. The full suite is not green: 998 pass, 5 skip,
and 31 fail (24 historical-product tests and seven legacy tests). Failure IDs,
compilation and whitespace checks
are in `verification.json`; failures are not suppressed or relabelled passes.

No further physical calculation was launched. Predictive branching kinetics
remain uncalibrated, and the historical process-state limitation remains in force.
