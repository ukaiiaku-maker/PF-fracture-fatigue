#!/usr/bin/env python3
"""Actual post-primary tensor probes and exact pair mechanics; no trajectory.

The companion is a virtual addition at the original junction, not an existing
crack tip. Its discrete kinetic drive is the same marginal energy construction
used by production for candidates without a valid local contour, now relative
to the accepted primary. It is not remote K or continuum-qualified G.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.conditional_branch_mark_v13 import (
    BOUNDARY, ParentCleavageEvent, BranchMarkParameters, CompanionChannel,
    BranchMark, conditional_probabilities, apply_conditional_branch_mark,
)
from arrhenius_fracture.marked_topology_trial_v13 import trial_conditional_pair
from arrhenius_fracture.sharp_front_v11_branching import _request, _realized_trial_network, _capture_shared_engine
from arrhenius_fracture.topology_transaction_v11 import apply_causal_sharp_wake_trial_geometry, extend_network_arm
from arrhenius_fracture.live_topology_runtime_v11 import LiveTopologyRuntime
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.anisotropic_emission_v10174 import probe_tensor_ahead, build_front_drive, require_admissible_tensor_drive
from arrhenius_fracture.hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2, _infer_boundary_opening
from scripts.run_pf_current_source_multifront_field_atlas_v12 import CASES, atomic_json, sha256, campaign_environment
from scripts.v13_frozen_support import initialized_engine
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
from scripts.qualify_v13_process_restore import safe_json, selected_state


def evaluate_parent(case, parents_root, output_root, plan):
    source = parents_root/case
    record_path = source/"parent/parent_record.json"
    if not record_path.exists():
        return {"case": case, "status": "NO_CLEAN_PARENT", "source_terminal": json.loads((source/"terminal.json").read_text())}
    record = json.loads(record_path.read_text())
    path = source/"parent/event_context.pkl"
    if sha256(path) != record["event_context_sha256"] or not record["fresh_initialization"]:
        raise RuntimeError("parent hash or clean-history contract failed")
    payload = pickle.loads(path.read_bytes())
    single = payload["accepted_single_checkpoint"]
    baseline = single.state
    pre = payload["solved_pre_event_state"]
    args = SimpleNamespace(**payload["args"])
    cfg = payload["configuration"]
    launch = json.loads((source/"launch.json").read_text())
    family = Path(launch["family_validation"]["family_validation"]["family"])
    os.environ.update(campaign_environment(family))
    out = output_root/case
    out.mkdir(parents=True, exist_ok=True)
    candidates = baseline.competition.candidates
    primary_id = record["winning_candidate_id"]
    companion_candidates = [c for c in candidates if c.candidate_id != primary_id]
    before_hash = fp(baseline)
    T = float(args.temperatures[0])
    sigma = assemble_mechanics(baseline.mesh, baseline.displacement, baseline.ep_gp, baseline.rho_gp,
        baseline.damage, baseline.elasticity_D, baseline.material, cohesive_network=baseline.cohesive_network)[2]
    stress_hash = fp(sigma)
    tip_id = payload["pre_cleavage_checkpoint"].state.crack_network.active_tip_ids[0]
    junction = pre.crack_network.branch(tip_id).tip
    accepted_primary_endpoint = baseline.crack_network.branch(tip_id).tip
    da = math.dist(junction, accepted_primary_endpoint)
    rows, surfaces = [], []
    with initialized_engine(single.shared_process_state) as engine:
        process_hash = fp(_capture_shared_engine(engine))
        parent = ParentCleavageEvent((case, tip_id), tip_id, primary_id, record["event_id"], record["event_ordinal"],
            single.physical_time_s, single.accepted_load, record["accepted_single_checkpoint_sha256"],
            process_hash, fp(baseline.competition), fp(engine._hazard_rng))
        for companion in companion_candidates:
            pair_proposal = SimpleNamespace(action_type="two_arm", member_candidate_ids=tuple(sorted((primary_id, companion.candidate_id))))
            ledger = dict(baseline.energy_ledgers)
            for key in ("topology_release_J_per_m", "hazard_dissipation_J_per_m"):
                ledger[key] = pre.energy_ledgers.get(key, 0.0)
            pair_start = replace(pre, tip_process_state=baseline.tip_process_state, energy_ledgers=ledger)
            network, cluster, pairs = _realized_trial_network(pair_start, pair_proposal, candidates, da, None)
            initial_arms = tuple(arm for _, arm in pairs)
            # Use the bit-exact primary endpoint, not a normalized ray rebuild.
            initial_arms = tuple(replace(a, end_xy_m=accepted_primary_endpoint) if a.candidate_id == primary_id else a for a in initial_arms)
            def geometry(state, arms):
                realized = network
                for arm in arms:
                    realized = extend_network_arm(realized, arm)
                return apply_causal_sharp_wake_trial_geometry(replace(state, crack_network=realized,
                    junction_process_state={"cluster": cluster, "crack_representation": "sharp_wake_causal_v11"}), arms)
            # Actual fixed-opening PF internal FEM: geometry only, no process
            # integration, no random draws and no accepted-state mutation.
            provisional_pair = geometry(pair_start.isolated_copy(), initial_arms)
            request = _request(provisional_pair, candidates, args=args, cfg=cfg,
                runtime_step=payload["context"].step, cluster=cluster)
            runtime = replace(single.provider_runtime, cache_root=str((out/"pair_mechanics_cache").resolve()))
            runtime, live = runtime.evaluate_trial(request)
            pair_energy = float(live["base_equilibrium"]["recoverable_potential_energy_J_per_m"])
            pair_displacement = np.asarray(live["base_equilibrium"]["displacement"])
            top, bottom = _infer_boundary_opening(provisional_pair.boundary, provisional_pair.displacement)
            inferred = top - bottom
            # The provider infers fixed loading from the supplied accepted
            # displacement; compare it to the captured exact parent opening.
            if not math.isclose(float(inferred), single.accepted_load, rel_tol=1e-12, abs_tol=1e-18):
                raise RuntimeError("pair trial changed the parent imposed opening")
            G_comp = (float(baseline.stored_energy_J_per_m) - pair_energy)/da
            K_comp = math.sqrt(float(baseline.material.Eprime)*max(G_comp, 0.0))
            probe = probe_tensor_ahead(baseline.mesh, sigma, baseline.damage, np.asarray(junction),
                np.asarray(companion.direction_xy), engine.anisotropic_cfg)
            tensor_drive = build_front_drive(baseline.mesh, sigma, baseline.damage, np.asarray(junction), engine.anisotropic_cfg)
            rejection = None
            try:
                require_admissible_tensor_drive(tensor_drive)
            except RuntimeError as exc:
                rejection = str(exc)
            tensor = np.asarray(probe.get("tensor", np.full((2, 2), np.nan)))
            if not probe["reliable"]:
                rejection = "unreliable_candidate_specific_post_primary_tensor"
            normal = np.asarray(companion.normal_xy)
            sigma_nn = float(normal @ tensor @ normal)
            if rejection is None and sigma_nn <= 0:
                rejection = "nonpositive_candidate_normal_traction"
            if rejection is None and G_comp <= 0:
                rejection = "nonpositive_post_primary_marginal_energy"
            raw, barrier, effective, stress = None, None, None, None
            process_attribution = None
            if rejection is None:
                preview = copy.deepcopy(engine)
                stress = preview.sigma_tip(K_comp/math.sqrt(companion.gamma_rel))
                effective, raw, barrier = preview.lambda_cleave(stress, T)
                r_eff = float(preview.r_eff())
                no_shield = min(K_comp/math.sqrt(companion.gamma_rel*2*math.pi*r_eff), preview.f.sigma_cap)
                no_blunting_or_shield = min(K_comp/math.sqrt(companion.gamma_rel*2*math.pi*preview.f.r0), preview.f.sigma_cap)
                process_attribution = {
                    "scope": "analytical_native_strength_law_counterfactuals_not_physical_states",
                    "raw_actual_per_s": raw,
                    "raw_without_signed_shielding_per_s": preview.lambda_cleave(no_shield, T)[1],
                    "raw_without_shielding_or_blunting_per_s": preview.lambda_cleave(no_blunting_or_shield, T)[1],
                    "r_eff_m": r_eff, "r0_m": preview.f.r0,
                    "signed_K_shield_Pa_sqrt_m": float(preview.K_shield()),
                }
            if rejection is not None:
                # Do not fabricate rates or energy costs for inadmissible
                # observations. Probabilities remain unevaluated, not zero.
                row = {"case": case, "candidate_id": companion.candidate_id, "status": "COMPANION_OBSERVATION_INADMISSIBLE",
                    "reason": rejection, "tensor_probe": probe, "tensor_drive": tensor_drive,
                    "sigma_nn_Pa": sigma_nn, "G_companion_discrete_J_per_m2": G_comp,
                    "K_companion_discrete_Pa_sqrt_m": K_comp, "pair_energy_J_per_m": pair_energy,
                    "probabilities_evaluated": False, "single_fallback_sha256": before_hash}
                rows.append(row)
                continue
            companion_cost = hazard_resistance_J_per_m2(barrier_J=barrier, cooperative_hits=engine.f.m_hits,
                burgers_vector_m=engine.b, gamma_relative=companion.gamma_rel)*da
            primary_cost = next(t.result.hazard_dissipation_J_per_m for t in payload["canonical_result"].trials if t.selected)
            arms = tuple(replace(a, hazard_dissipation_J_per_m=primary_cost if a.candidate_id == primary_id else companion_cost) for a in initial_arms)
            def equilibrium(state):
                return replace(state, displacement=pair_displacement, stored_energy_J_per_m=pair_energy)
            mark = BranchMark(companion.candidate_id, None, ())
            trial = trial_conditional_pair(pre_event_state=pair_start, baseline_single_state=baseline,
                parent=parent, mark=mark, arms=arms, apply_trial_geometry=geometry,
                equilibrate_fixed_load=equilibrium, network_geometry_already_realized=True)
            channel = CompanionChannel(companion.candidate_id, raw, engine.f.m_hits, barrier, T,
                0.0, engine.mpz.length_m, process_hash)
            reference = plan["single_overlay_reference"]
            reference_params = BranchMarkParameters(True, reference["beta_B"]*engine.f.tau_c,
                reference["Q_junction_eV"]*1.602176634e-19,
                reference["Q_overlap_eV"]*1.602176634e-19, reference["branch_seed"])
            overlay = apply_conditional_branch_mark(parent=parent, baseline_single_state=baseline,
                parameters=reference_params, channels=lambda: (channel,), exact_pair_trial=lambda *_: trial)
            accepted_hash = fp(trial.state)
            # Deterministic preselected proposal checks the exact pair bridge;
            # it is not an extra sampler run or a stochastic physical outcome.
            invariant_names = ("competition", "rng_state", "tip_process_state", "event_counters")
            if not all(getattr(trial.state, n) is getattr(baseline, n) for n in invariant_names):
                raise RuntimeError("pair bridge changed a canonical parent invariant")
            if not trial.accepted and trial.state is not baseline:
                raise RuntimeError("rejected companion did not retain exact single object")
            row = {"case": case, "candidate_id": companion.candidate_id, "status": "EXACT_PAIR_EVALUATED",
                "junction_xy_m": junction, "primary_endpoint_xy_m": accepted_primary_endpoint,
                "tensor_probe": probe, "tensor_drive": tensor_drive, "sigma_nn_Pa": sigma_nn,
                "post_primary_stress_sha256": stress_hash, "post_primary_process_sha256": process_hash,
                "native_discrete_G_companion_J_per_m2": G_comp, "K_companion_discrete_Pa_sqrt_m": K_comp,
                "raw_arrival_per_s": raw, "raw_barrier_J": barrier, "multihit_order": engine.f.m_hits,
                "effective_parent_law_rate_diagnostic_per_s": effective, "effective_opening_Pa": stress,
                "exact_pair_admissible": trial.accepted, "pair_rejection_reason": trial.rejection_reason,
                "pair_energy_release_J_per_m": trial.energy_release_J_per_m,
                "primary_cost_J_per_m": primary_cost, "companion_cost_J_per_m": companion_cost,
                "pair_total_cost_J_per_m": trial.hazard_dissipation_J_per_m,
                "pair_energy_margin_J_per_m": trial.energy_margin_J_per_m,
                "fixed_parent_endpoint_s": single.physical_time_s, "fixed_opening_m": single.accepted_load,
                "geometry_admissible": True, "process_renewals_by_overlay": 0,
                "global_time_increment_s": 0.0, "new_baseline_rng_draws": 0,
                "baseline_invariants_exact": True, "single_fallback_sha256": before_hash,
                "trial_outcome_sha256": accepted_hash, "process_state": selected_state(copy.deepcopy(engine)),
                "process_state_attribution": process_attribution,
                "single_reference_overlay_disposition": overlay.disposition,
                "single_reference_branch_rng_identities": overlay.mark.branch_rng_identities,
                "single_reference_parameters": reference,
                "single_reference_fallback_object_exact": overlay.state is baseline if overlay.disposition != "PAIR_ACCEPTED" else None,
                "pair_provider": runtime.routing.active_mechanics_provider, "pair_topology_fingerprint": live["topology_fingerprint"]}
            rows.append(row)
            for log_beta in plan["beta_B_log10_grid"]:
                for qj in plan["Q_junction_eV_grid"]:
                    for qo in plan["Q_overlap_eV_grid"]:
                        beta = 10.0**log_beta
                        p = BranchMarkParameters(True, beta*engine.f.tau_c, qj*1.602176634e-19, qo*1.602176634e-19)
                        probability = conditional_probabilities([channel], p).companions[0][1]
                        surfaces.append({"case": case, "companion_id": companion.candidate_id,
                            "beta_B": beta, "tau_B_s": p.exposure_time_s, "tau_c_s": engine.f.tau_c,
                            "Q_junction_eV": qj, "Q_overlap_eV": qo,
                            "P_embryo": probability, "exact_pair_admissible": trial.accepted,
                            "P_committed": probability if trial.accepted else 0.0})
            target = out/"exact_pair_result.pkl"
            target.write_bytes(pickle.dumps({"trial": trial, "parent": parent, "arms": arms, "cluster": cluster}, protocol=5))
        assert fp(_capture_shared_engine(engine)) == process_hash
    assert fp(baseline) == before_hash
    assert sha256(path) == record["event_context_sha256"]
    result = {"case": case, "status": "COMPANION_CHECK_COMPLETE", "clean_parent_record": record,
        "companions": rows, "sensitivity_surface": surfaces, "boundary": BOUNDARY,
        "companion_drive_semantics": "exact_native_single_to_pair_marginal_energy_not_remote_K_not_continuum_G",
        "mechanics_parent_call_repeated": False, "physical_time_advanced": False,
        "ensemble_gate": "NOT_YET_ESTABLISHED"}
    atomic_json(out/"companion_qualification.json", safe_json(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if any(not (args.parents_root/case/"terminal.json").exists() for case in CASES):
        raise RuntimeError("parent queue is still active; do not add a third heavy worker")
    plan = json.loads(args.plan.read_text())
    results = []
    for case in CASES:
        try:
            result = evaluate_parent(case, args.parents_root, args.output_root, plan)
        except (RuntimeError, ValueError) as exc:
            import traceback
            result = {"case": case, "status": "COMPANION_CHECK_STOP_REQUIRES_CLASSIFICATION",
                "exception_type": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc(),
                "automatic_retry": False, "probabilities_evaluated": False}
            atomic_json(args.output_root/case/"companion_stop.json", result)
        results.append(result)
        print(case, results[-1]["status"], flush=True)
    atomic_json(args.output_root/"summary.json", safe_json({"boundary": BOUNDARY, "cases": results,
        "producer_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "plan_sha256": sha256(args.plan), "sampler_only_draws": 0, "physical_ensemble_launched": False}))


if __name__ == "__main__":
    main()
