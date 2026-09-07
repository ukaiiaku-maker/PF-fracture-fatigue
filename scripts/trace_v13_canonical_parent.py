#!/usr/bin/env python3
"""Replay one branch-disabled canonical interval with checked cached mechanics.

No linear solve is permitted. Stress is assembled from cached displacement;
the existing exact single-arm transaction uses the archived accepted cache.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess

import numpy as np

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.directional_competition_v11 import preview_production_cleavage_rate, competition_state_to_dict
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from arrhenius_fracture.production_step_loop_v11 import AcceptedStepContext, advance_accepted_step
from arrhenius_fracture.sharp_front_v11_branching import (
    _hash, _capture_shared_engine, _realized_trial_network, evolve_process_engine_v12_hook,
)
from arrhenius_fracture.tip_directional_observation_v11 import TipDirectionalObservation, apply_post_interval_event_renewal
from arrhenius_fracture.topology_transaction_v11 import (
    execute_topology_trial, apply_causal_sharp_wake_trial_geometry, extend_network_arm,
)
from arrhenius_fracture.anisotropic_emission_v10174 import OBSERVER
from scripts.v13_frozen_support import initialized_engine, recapture, restore_complete_current_source_engine, field_differences, rng_hash
from scripts.qualify_v13_process_restore import sha, safe_json, selected_state


CONTROL = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_control_max1_seed3621")
CHECKPOINT = CONTROL / "checkpoint/transitions/step0000301_mesh_adaptation_g0017.json"


def cache_for(fingerprint, energy, opening):
    for manifest_path in sorted((CONTROL / "live_kernel_cache").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest["topology_fingerprint"] != fingerprint:
            continue
        path = manifest_path.parent / "provider_state.pkl"
        if sha(path) != manifest["state_sha256"]:
            raise RuntimeError("canonical mechanics cache hash mismatch")
        result = pickle.loads(path.read_bytes())
        eq = result["base_equilibrium"]
        opening_residual = abs(eq["applied_displacement"] - opening)
        if math.isclose(eq["recoverable_potential_energy_J_per_m"], energy, rel_tol=1e-12) and opening_residual <= 4 * np.spacing(opening):
            return result, {"path": str(path), "sha256": sha(path), "manifest_sha256": sha(manifest_path),
                "cached_vs_logged_opening_roundoff_m": opening_residual,
                "opening_tolerance": "four_float64_ULPs_no_rescaling"}
    raise RuntimeError("required canonical mechanics cache absent; no solve authorized")


def fixture():
    rows = [json.loads(line) for line in (CONTROL / "branch_action_trials.jsonl").read_text().splitlines()]
    trials = [row for row in rows if row["step"] == 301]
    if len(trials) != 1 or not trials[0]["accepted"] or trials[0]["action_type"] != "one_arm":
        raise RuntimeError("canonical reference is not an isolated accepted single-arm event")
    row = trials[0]
    checkpoint = restore_branch_checkpoint(CHECKPOINT)
    if len(checkpoint.state.crack_network.active_tip_ids) != 1:
        raise RuntimeError("canonical reference must be single-front")
    base, base_hash = cache_for(row["topology_fingerprint_before"], row["pretrial_potential_energy_J_per_m"], row["applied_displacement_m"])
    post, post_hash = cache_for(row["topology_fingerprint_after"], row["posttrial_potential_energy_J_per_m"], row["applied_displacement_m"])
    return checkpoint, row, base, post, {"base_cache": base_hash, "single_arm_cache": post_hash}


def replay(engine, data):
    checkpoint, row, base, post, hashes = data
    solved = replace(checkpoint.state,
        displacement=np.asarray(base["base_equilibrium"]["displacement"]),
        stored_energy_J_per_m=float(base["base_equilibrium"]["recoverable_potential_energy_J_per_m"]),
    )
    duration = row["physical_time_s"] - checkpoint.physical_time_s
    context = AcceptedStepContext(301, checkpoint.physical_time_s, duration, row["accepted_state_id"])
    candidates = solved.competition.candidates
    local = {r["candidate_id"]: r for r in base["tips"][0]["directional"]}
    sigma = assemble_mechanics(solved.mesh, solved.displacement, solved.ep_gp, solved.rho_gp,
        solved.damage, solved.elasticity_D, solved.material, cohesive_network=solved.cohesive_network)[2]
    T = 700.0
    state_id = _hash((context.accepted_state_id, solved.crack_network, solved.displacement, solved.ep_gp, solved.rho_gp))
    stress_id = _hash((state_id, sigma))
    rates = tuple(preview_production_cleavage_rate(engine, c,
        signed_J_J_per_m2=float(local[c.candidate_id]["J_local_signed_J_per_m2"]),
        Eprime_Pa=float(solved.material.Eprime), temperature_K=T) for c in candidates)
    rate_map = {r.candidate_id: r for r in rates}
    controlling = max(rates, key=lambda r: (r.K_directional_Pa_sqrt_m, r.candidate_id))
    front = solved.crack_network.active_tip_ids[0]
    observation = TipDirectionalObservation(
        tip_id=front, parent_branch_id=None, candidate_id=controlling.candidate_id,
        tip_xy_m=solved.crack_network.branch(front).tip, branch_arclength_m=0.0, projected_reach_m=0.0,
        signed_local_J_J_per_m2=controlling.signed_J_J_per_m2,
        kinetic_J_J_per_m2=controlling.signed_J_J_per_m2, marginal_J_J_per_m2=None,
        directional_K_Pa_sqrt_m=controlling.K_directional_Pa_sqrt_m,
        directional_rate_per_s=controlling.lambda_per_s, local_contour_valid=True,
        local_contour_invalid_reason=None, accepted_state_id=state_id,
        stress_field_state_id=stress_id, pre_event_topology_fingerprint=_hash(solved.crack_network),
    )
    process = {}
    def trial(current, proposal):
        if proposal.action_type != "one_arm" or list(proposal.member_candidate_ids) != row["candidate_ids"]:
            raise RuntimeError("canonical replay requested an uncached action; no solve authorized")
        network, cluster, pairs = _realized_trial_network(current, proposal, candidates, 5e-6, None)
        arms = []
        for candidate, arm in pairs:
            drive = rate_map[candidate.candidate_id]
            barrier = engine.lambda_cleave(engine.sigma_tip(drive.K_directional_Pa_sqrt_m / math.sqrt(candidate.gamma_rel)), T)[2]
            cost = hazard_resistance_J_per_m2(barrier_J=barrier, cooperative_hits=engine.f.m_hits,
                burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel)
            arms.append(replace(arm, hazard_dissipation_J_per_m=cost * arm.event_reward_m))
        def geometry(state, arm_values):
            realized = network
            for arm in arm_values:
                realized = extend_network_arm(realized, arm)
            return apply_causal_sharp_wake_trial_geometry(replace(state, crack_network=realized), arm_values)
        def equilibrium(state):
            return replace(state, displacement=np.asarray(post["base_equilibrium"]["displacement"]),
                stored_energy_J_per_m=float(post["base_equilibrium"]["recoverable_potential_energy_J_per_m"]))
        return execute_topology_trial(current, proposal, arms, apply_trial_geometry=geometry,
            equilibrate_fixed_load=equilibrium, network_geometry_already_realized=True)
    def update(state, ctx, proposal):
        updated, info = evolve_process_engine_v12_hook(engine, observation, solved, sigma,
            accepted_state_id=state_id, stress_field_state_id=stress_id,
            pre_event_topology_fingerprint=_hash(solved.crack_network), duration_s=ctx.duration_s,
            temperature_K=T, pre_progress=max(h.residual_action for h in checkpoint.state.competition.hazard_states),
            permitted_physical_hazard_action=0.075)
        process["at_endpoint_before_renewal"] = _capture_shared_engine(updated)
        renewal = apply_post_interval_event_renewal(updated.mpz, info,
            event_selected=proposal is not None, event_distance_m=5e-6 if proposal is not None else 0.0)
        process.update(engine=updated, info=info, renewal=renewal)
        return state
    rng_before = rng_hash(engine)
    observer_before = copy.deepcopy(vars(OBSERVER))
    try:
        result = advance_accepted_step(checkpoint.state, context, correlation_interval_s=engine.f.tau_c,
            solve_accepted=lambda state, ctx: solved, evaluate_directional_rates=lambda state, ctx: rates,
            trial_action=trial, update_shared_state_once=update)
    finally:
        vars(OBSERVER).clear(); vars(OBSERVER).update(observer_before)
    chosen = next(x for x in result.trials if x.selected)
    if list(chosen.proposal.member_candidate_ids) != row["candidate_ids"]:
        raise RuntimeError("canonical winner changed")
    record = {
        "canonical_source_function": "arrhenius_fracture.production_step_loop_v11.advance_accepted_step",
        "canonical_driver": "arrhenius_fracture.sharp_front_v11_branching.run_2d(maximum_fronts=1)",
        "checkpoint_path": str(CHECKPOINT), "checkpoint_manifest_sha256": sha(CHECKPOINT),
        "checkpoint_state_sha256": sha(CHECKPOINT.with_name(CHECKPOINT.name + ".state.pkl")),
        "starting_accepted_state_id": context.accepted_state_id,
        "winning_candidate_id": chosen.proposal.member_candidate_ids[0],
        "primary_event_id": chosen.proposal.member_event_ids[0],
        "primary_event_ordinal": chosen.proposal.member_event_ordinals[0],
        "raw_first_passage_time_s": chosen.proposal.completion_times_s[0],
        "accepted_single_arm_endpoint_s": context.physical_time_s + context.duration_s,
        "accepted_opening_m": row["applied_displacement_m"],
        "physical_duration_s": context.duration_s,
        "legacy_v12_correlation_delay_applied": False,
        "endpoint_semantics": "canonical_accepted_outer_interval_endpoint_not_V12_correlation_ready_time",
        "process_state_at_endpoint": selected_state(process["engine"]),
        "process_rng_before": rng_before, "process_rng_after": rng_hash(process["engine"]),
        "directional_clocks_before": competition_state_to_dict(checkpoint.state.competition),
        "directional_clocks_after": competition_state_to_dict(result.state.competition),
        "process_interval_output": process["info"], "renewal": process["renewal"],
        "cache_provenance": hashes, "new_fem_solves": 0,
        "stress_assembly_from_cached_displacement": True,
    }
    return record, result, process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data = fixture()
    with initialized_engine(data[0].shared_process_state) as a:
        b = restore_complete_current_source_engine(recapture(a))
        ra, sa, pa = replay(a, data)
        rb, sb, pb = replay(b, data)
        differences = field_differences(ra, rb) + field_differences(_capture_shared_engine(pa["engine"]), _capture_shared_engine(pb["engine"]))
        if differences:
            raise RuntimeError(f"canonical initialized/restore parity failed: {differences}")
    output = {
        "schema": "v13.immutable-canonical-parent/1", "status": "PASS",
        "boundary": "BRANCHING_KINETICS_MODEL_UNCALIBRATED",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initialized_vs_restored_exact": True, "parent": ra,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "canonical_parent.json").write_text(json.dumps(safe_json(output), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(ra["winning_candidate_id"], ra["raw_first_passage_time_s"], ra["accepted_single_arm_endpoint_s"])


if __name__ == "__main__":
    main()
