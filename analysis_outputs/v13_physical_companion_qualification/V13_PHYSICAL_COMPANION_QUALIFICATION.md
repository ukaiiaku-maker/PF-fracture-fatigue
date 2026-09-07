# V13 physical companion-state qualification

Permanent boundary: **BRANCHING_KINETICS_MODEL_UNCALIBRATED**.

Initial V13 sampler/restoration qualification remains accepted. No sampler-only Monte Carlo or historical atlas continuation was performed.

The initial record is `55955722db40ef65c43d2f1c08357ad8f77b82fa`. Complete initial history and the physical-source increment are preserved as verified Git bundles on the Data drive.

## Physical result

All 8 clean-parent checks completed. The largest committed-branch probability across 5096 analytic case/parameter evaluations is **4.4611289e-08**. No physical short ensemble was launched.

The stop decision follows from the analytic surface, independently of the eight reference random marks. Exact two-arm mechanical acceptance does not imply appreciable companion-embryo probability.

## Clean parents and actual companion mechanics

| Case | First accepted endpoint (s) | Opening (µm) | Companion check | Exact pair admissible |
|---|---:|---:|---|---|
| Peak_300K | 3368.4000000000233 | 80.20000000000027 | COMPANION_CHECK_COMPLETE | True |
| Peak_1000K | 3124.8000000000206 | 74.40000000000013 | COMPANION_CHECK_COMPLETE | True |
| DBTT_300K | 3360.000000000023 | 80.00000000000027 | COMPANION_CHECK_COMPLETE | True |
| DBTT_1000K | 3301.2000000000226 | 78.60000000000024 | COMPANION_CHECK_COMPLETE | True |
| weakT_300K | 2671.2000000000157 | 63.599999999999866 | COMPANION_CHECK_COMPLETE | True |
| weakT_1000K | 2234.400000000011 | 53.19999999999988 | COMPANION_CHECK_COMPLETE | True |
| ceramic_300K | 2167.2000000000103 | 51.5999999999999 | COMPANION_CHECK_COMPLETE | True |
| ceramic_1000K | 1923.6000000000079 | 45.799999999999955 | COMPANION_CHECK_COMPLETE | True |

Parents start from fresh initialization at theta40 with the pinned four material rows and seed 3621. Each stops at its first accepted canonical single-arm cleavage. Full pre-cleavage and accepted single-arm checkpoints, event context, callable inventory, raw first-passage time, winner/ordinal, complete process state and RNG hashes are saved beneath `physical_parents/`.

The original parent loop is unchanged apart from default-off capture hooks and a demonstrated diagnostic-name repair. Startup failures are retained in `parents/` and `clean_parents/`; neither produced an accepted physical interval. A new terminal-label publication error occurred after some valid parent captures; those raw worker reports are retained and do not require rerunning their saved parent states.

Companion mechanics uses the exact same imposed opening and primary endpoint. The candidate-ray tensor is probed from the actual post-primary FEM stress field at the original junction. The companion is not yet a physical crack tip: its kinetic drive is the exact discrete single-to-pair marginal energy per companion length. This is model-native discrete mechanics, **not remote K or continuum-qualified G**. The existing signed-process strength law and raw cleavage barrier are evaluated on an isolated copy of the complete accepted process state. No additional emission, time, renewal, threshold or baseline RNG update occurs.

The unchanged exact whole-pair geometry and energy transaction is evaluated separately. Primary energy cost is held exactly at the canonical accepted value; companion cost uses the existing hazard-energy formula. Branch-only junction/overlap barriers modify mark kinetics, not the established whole-topology energy rule. A single preregistered reference mark checks the overlay; no repeated-seed search is used.

Sensitivity surfaces store embryo probability, pair-admissibility indicator, and committed-branch probability separately. Invalid observations remain unevaluated—no artificial rate or probability is supplied. Process-state attribution removes shielding/blunting only algebraically for diagnostics, never by changing an accepted physical state.

At a newly born junction, separation is zero, so junction and overlap barriers are identifiable only through their sum. These surfaces do not calibrate the three new branch-only quantities or establish branch spacing.

## Ensemble decision

**NOT_PASSED_NO_ENSEMBLE**

- no_positive_exact_pair_through_reference_overlay
- no_common_robust_nondegenerate_material_and_temperature_sensitive_grid_region
- defensible_process_state_rate_sensitivity_not_demonstrated

A failed grid-robustness gate is not a universal no-branching theorem. A short physical ensemble is not run unless all required physical checks and a defensible nondegenerate region pass. No long field atlas is authorized.

## Candidate-level evidence

| Case | Candidate observation | Raw arrival (s⁻¹) | Raw barrier (eV) | Pair margin (J/m) | Embryo probability range | Committed probability range |
|---|---|---:|---:|---:|---|---|
| Peak_300K | EXACT_PAIR_EVALUATED | 746.70276 | 0.54328893 | 0.0081187501 | 4.00519e-59–6.93504e-11 | 4.00519e-59–6.93504e-11 |
| Peak_1000K | EXACT_PAIR_EVALUATED | 6404.4408 | 1.6257697 | 0.0068750051 | 3.71272e-35–4.35719e-08 | 3.71272e-35–4.35719e-08 |
| DBTT_300K | EXACT_PAIR_EVALUATED | 517.02687 | 0.55279141 | 0.0080776793 | 1.32959e-59–2.30261e-11 | 1.32959e-59–2.30261e-11 |
| DBTT_1000K | EXACT_PAIR_EVALUATED | 6455.047 | 1.6250915 | 0.0076787559 | 3.80142e-35–4.46113e-08 | 3.80142e-35–4.46113e-08 |
| weakT_300K | EXACT_PAIR_EVALUATED | 1149.5306 | 0.53213529 | 0.0050932287 | 1.46131e-58–2.52951e-10 | 1.46131e-58–2.52951e-10 |
| weakT_1000K | EXACT_PAIR_EVALUATED | 1961.0058 | 1.7277588 | 0.0034522946 | 1.06582e-36–1.25501e-09 | 1.06582e-36–1.25501e-09 |
| ceramic_300K | EXACT_PAIR_EVALUATED | 1217.062 | 0.5306595 | 0.0033387237 | 1.73427e-58–3.00186e-10 | 1.73427e-58–3.00186e-10 |
| ceramic_1000K | EXACT_PAIR_EVALUATED | 1304.5521 | 1.7628829 | 0.0025156356 | 3.13785e-37–3.69665e-10 | 3.13785e-37–3.69665e-10 |

The full JSON records retain candidate identities, tensor probes, process-state fingerprints, costs, exact fallback identities, and every preregistered grid point. A range spans the entire sensitivity grid and is not a fitted or calibrated uncertainty interval.

## Interpretation and limits

The opportunity range is beta_B = 10⁻⁶ to 1 with tau_c = 1 µs. All eight raw multihit orders are three. Even with zero added junction/overlap barriers and the longest tested opportunity, raw arrival × tau_c is only approximately 0.000517–0.006455. Three arrivals within that opportunity are therefore very unlikely. Nonnegative branch-only barriers can only reduce this probability. No exposure range or barrier was changed after seeing the result.

All eight accepted parent radii equal the 1 µm reference radius, and the frozen shielding/blunting attribution does not demonstrate the preregistered 5% raw-rate effect. This is a statement about these first-cleavage snapshots, not a proof that the continuous-emission process has no effect on the preceding parent history or on later growth. Material/temperature differences in small raw probabilities exist, but do not establish the required nondegenerate physical branching regime.

The deterministic pair adapter accepted real PF FEM states in all eight cases. All eight preregistered stochastic reference marks retained the exact single-arm result; a positive stochastic pair commitment and subsequent owner-transition trajectory have not been demonstrated here. No mark seed search or forced physical branch was performed.

Focused capture, companion-contract, topology, clock and checkpoint checks: 35 passed. Compileall and git diff --check passed. The accepted initial sampler qualification was not repeated; no new full-suite/Monte Carlo claim is made.
