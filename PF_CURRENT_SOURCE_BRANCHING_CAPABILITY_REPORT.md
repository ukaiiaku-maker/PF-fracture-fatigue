# PF current-source branching capability demonstration

`CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS`

## Decision

**NO_CURRENT_SOURCE_BRANCH_BIRTH_IN_BOUNDED_TEST**

This bounded result demonstrates current-source multi-front PF morphology and atomic topology software capability. It is not branching-parameter calibration, is not part of the canonical single-crack toughness matrix, and does not validate branching physics. Model-native KJ is not reported as calibrated toughness or an R-curve.

## Exact contract and lineage

- Current final material: `oneD_v2_focused_weak_T_0016`.
- 700 K, θ=40°, canonical rate1x, `tip_only`, current canonical sharp-wake backend.
- Historical positive-run seed recovered from its manifest: `3621`.
- Qualified physical PF source commit: `9e884fb0b0845da621d2612bdf1042e481b8df49`.
- Atomic topology overlay audit source: `2b5e535`.
- Historical positive V11 source commit (lineage evidence only; not executed): `9cc5795d8461ea28d24227b19e17cd233485ab72`.
- Historical source refs containing that commit: branch `codex/v11-branching` and tag `v11.0.0-hazard-branching-production`.
- Historical positive contract recovered directly from its final case status and audit: `v913_paper_weakT01_0129902_persistent_sites`, 700 K, θ=30°, seed `3621`, `dU=2e-7 m`, `dt=8.4 s`, `da_phys=5 µm`, `tip_only`, signed active shielding with zero mobile shielding, sharp wake, and a 300 µm launcher target. Its first two-arm transaction was step 295 at `(540.980762, -10.980762) µm`. The restart audit retained an older 1000 µm argument, but the final case-status command and completion record establish the bounded 300 µm launch semantics. Those V11 network/branch outputs are lineage evidence, not current-publisher data.
- Matched fresh cases: `max_fronts=1` control and `max_fronts=2` enabled, with identical parent RNG, loading, event-length, mesh, and material contracts.

## Historical-to-current source delta audit

The historical V11 executable and historical material row were not invoked. The current adapter selects the exact current transfer row and enters the current signed-dislocation PF production stack through `sharp_front_v10_2_27.py`; the audited atomic multi-front overlay is then applied without adding a probability, clone/split rule, or state interpolation.

- Signed mobile/retained transport, Peierls/Taylor state evolution, signed shielding, backstress, and source multiplicity are owned by the current physical engine and its MPZ state, selected through `sharp_front_current_source_branching.py` and the pinned current registry/selection files.
- Directional signed-J evaluation is owned by `live_topology_kernel_v11.py`; positive-part kinetic use and directional rates/first-passage clocks are owned by `directional_competition_v11.py`, `production_step_loop_v11.py`, and `multi_tip_step_loop_v11.py`.
- Shared unresolved-cluster ownership and independent-tip handoff are owned by `branch_cluster_v11.py`, `process_state_ownership_v11.py`, `resolved_tip_state_v11.py`, and `branch_cluster_guard_v11.py`.
- Atomic causal wake mutation and whole-topology energy acceptance are owned by `causal_sharp_wake_v11.py` and `topology_transaction_v11.py`.
- Active-front inventory, parent retirement, intersection/coalescence rules, and maximum-forward-reach accounting are owned by `crack_network_v11.py`, `production_counts_v11.py`, and `network_metrics_v11.py`.

This separation is material: current signed transport/state and current material provenance come from the physical production lineage, while the topology layer comes from the audited atomic overlay. The old V11 run is used only to recover the prescribed seed and historical comparison facts.

## Result

The control reached 300.918 µm. The θ=40 enabled case reached 302.236 µm and produced a sustained daughter, but its final independent local-contour flags were unreliable. It therefore failed the explicit probe-reliability gate.

The prescribed θ=45 fallback produced two non-stub daughters and reached 49.497 µm. It then stopped fail-closed with `active_tip_resolution_marker_inconsistency`: active-tip hbar was already finer than the 1.5 µm target, no justified refinement marks remained, and the proposed trial lacked causal stiffness visibility. Continuing would require a prohibited topology/visibility-gate relaxation. Its birth-time local probes were also marked unreliable.

Thus neither current-source orientation produced a **qualified** daughter under every required gate. Raw branch births did occur, so the exact required terminal label is interpreted as “no qualified current-source branch birth in the bounded test,” not as an assertion that no topology transaction occurred. No branch threshold, topology gate, material parameter, or wake rule was tuned.

## Qualification

All success gates are recorded fail-closed in `pf_branching_topology_audit.json`. Pre-birth front, directional-drive, and accepted-action histories are neutral for the θ=40 matched pair after excluding identity-only hashes and timing telemetry. The θ=40 mechanics map covers 0–320 µm projected extension with a 20 µm target margin; all 455 recorded tensor-probe rows are reliable, and the maximum load-scaling relative error is `7.980489955135959e-15`. Its signed family covers the required physical path through 410.001 µm. The pinned θ=45 family covers 1575 µm. Both families are candidate-independent, hash-frozen, load-scaled, and non-extrapolating for their attempted paths.

## Interpretation boundary

The current implementation demonstrated atomic multi-front birth and sustained daughter growth, but it did **not** pass the complete bounded capability gate. It does not establish branching probability, calibrated branch resistance, or predictive morphology. The historical positive V11 result remains lineage evidence only.
