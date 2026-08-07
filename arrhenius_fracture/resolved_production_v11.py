"""Resolved multi-tip continuation for the v11 production FEM adapter."""
from __future__ import annotations

from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .branch_checkpoint_v11 import ProductionBranchCheckpoint, write_branch_checkpoint
from .branch_cluster_guard_v11 import evaluate_unresolved_cluster_guard
from .branch_cluster_v11 import create_unresolved_branch_cluster
from .branch_policy_v11 import branch_birth_policy
from .branch_snapshot_v11 import write_topology_snapshot
from .branch_output_v11 import BRANCH_EVENT_FIELDS, TRIAL_FIELDS
from .directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalRate, preview_production_cleavage_rate,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .live_topology_kernel_v11 import PROVIDER_ID
from .multi_tip_step_loop_v11 import advance_multi_tip_step
from .network_metrics_v11 import crack_growth_metrics
from .adaptive_multitip_mesh_v11 import adapt_accepted_state_for_trials, mesh_fingerprint
from .production_step_loop_v11 import AcceptedStepContext, DirectionalStepRefinementRequired
from .resolved_tip_state_v11 import resolve_unresolved_cluster, tip_lineage_seed
from .process_state_ownership_v11 import ProcessStateOwner, ProcessStateOwnerRegistry
from .topology_transaction_v11 import (
    TopologyArm, TopologyTrialResult, apply_causal_sharp_wake_trial_geometry,
    clip_arm_at_first_intersection, execute_topology_trial, extend_network_arm,
    mark_coalesced,
)


def _fresh_tip_payload(_branch_id: str):
    return (
        {"historical_state_imported": False, "B": 0.0, "N_em": 0.0, "W_emit": 0.0, "time_s": 0.0},
        {"historical_state_imported": False, "source_inventory": 0.0},
    )


def continue_resolved_production(
    *, args, cfg, state, shared_engine, runtime, cluster, candidates,
    writer, out: Path, cache_root: Path, physical_time: float,
    accepted_load: float, start_step: int, engine_factory,
    resume_bundle=None, resume_competitions=None, resume_clusters=(),
) -> int:
    """Resolve the first cluster and continue the exact network to target."""
    from . import sharp_front as base
    from .fem import assemble_mechanics, solve_dirichlet
    from .plasticity import update_plasticity
    from .sharp_front_v11_branching import (
        _capture_shared_engine, _hash, _mesh_identity, _request,
        _restore_shared_engine, _stored_energy,
    )
    if resume_bundle is None:
        resolution = resolve_unresolved_cluster(
            state.crack_network, cluster, candidates=candidates,
            global_hazard_seed=state.competition.global_hazard_seed,
            fresh_tip_factory=_fresh_tip_payload,
        )
        cluster = resolution.cluster
        clusters = {cluster.cluster_id: cluster}
        reservoirs = {resolution.reservoir.reservoir_id: resolution.reservoir}
        competitions = {tip: item.competition for tip, item in resolution.tips.items()}
        engines = {tip: engine_factory() for tip in competitions}
        ownership = ProcessStateOwnerRegistry(
            {tip: tip for tip in competitions},
            {
                **{tip: ProcessStateOwner(tip, "resolved_tip_engine", process_engine_id=tip) for tip in competitions},
                resolution.reservoir.reservoir_id: ProcessStateOwner(
                    resolution.reservoir.reservoir_id, "junction_reservoir",
                    cluster_id=cluster.cluster_id,
                    junction_reservoir_id=resolution.reservoir.reservoir_id,
                ),
            },
        )
        state = replace(
            state, crack_network=resolution.network,
            competition=competitions[sorted(competitions)[0]],
            tip_process_state={"mode": "independent_resolved_tips", "tips": resolution.tips},
            junction_process_state={
                "crack_representation": "sharp_wake_causal_v11",
                "reservoirs": reservoirs, "clusters": clusters,
            },
        )
    else:
        if resume_bundle.get("schema") != "v11.multi-tip-engine-bundle/2":
            raise ValueError("unsupported resolved multi-tip restart bundle")
        competitions = dict(resume_competitions or {})
        clusters = {item.cluster_id: item for item in resume_clusters}
        reservoirs = dict(resume_bundle.get("junction_reservoirs", {}))
        ownership = ProcessStateOwnerRegistry.from_dict(resume_bundle["ownership_registry"])
        engines = {
            owner: _restore_shared_engine(engine_factory(), payload)
            for owner, payload in resume_bundle["engines"].items()
        }
    owner_by_tip = dict(ownership.owner_by_tip)
    mat = state.material
    D = state.elasticity_D
    da_phys = float(args.da_phys if args.da_phys is not None else max(5.0 * base.eng_r_pz_hint(args), 2.0e-6))
    latest_sigma = None
    latest_live = None
    latest_rates = {}
    branch_birth_count = int(state.event_counters.get("branch_birth_count", 1))
    coalescence_count = int(state.event_counters.get("coalescence_count", 0))
    termination = None
    checkpoint = None
    adaptation_required = True
    growth = crack_growth_metrics(state.crack_network, initial_crack_length_m=cfg.geometry.a0)
    last_snapshot_extension = growth.max_root_to_tip_path_extension_m
    if resume_bundle is None:
        write_topology_snapshot(
            out, state, step=start_step - 1, reason="cluster_resolved",
            physical_extension_m=last_snapshot_extension,
            branch_birth_count=branch_birth_count, latest_action=None,
            growth_metrics=growth.to_dict_um(), coalescence_count=coalescence_count,
        )

    for step in range(start_step, int(args.steps) + 1):
        if adaptation_required:
            adaptation_start = time.perf_counter()
            candidate_inventory = {tip: tuple(candidates) for tip in state.crack_network.active_tip_ids}
            prior_generation = int(state.event_counters.get("mesh_generation", 0))
            prior_operation = int(state.event_counters.get("refinement_operation_index", 0))
            state, adaptation = adapt_accepted_state_for_trials(
                state, candidate_inventory, da_phys_m=da_phys,
                tip_h_fine_m=float(getattr(args, "tip_h_fine", 0.0) or 1.0e-6),
                contour_radius_m=float(getattr(args, "rJ", None) or args.L_pz),
                crack_band_radius_m=0.5e-6, accepted_load_m=accepted_load,
                starting_generation=prior_generation,
                starting_operation_index=prior_operation,
            )
            if adaptation.lineages:
                counters = dict(state.event_counters)
                counters["mesh_generation"] = adaptation.lineages[-1].mesh_generation
                counters["refinement_operation_index"] = adaptation.lineages[-1].refinement_operation_index
                junction = dict(state.junction_process_state)
                junction["mesh_refinement"] = {
                    "latest": adaptation.lineages[-1].to_dict(),
                    "physical_topology_fingerprint": _hash(state.crack_network),
                    "mechanical_discretization_fingerprint": mesh_fingerprint(state.mesh),
                }
                state = replace(state, event_counters=counters, junction_process_state=junction)
                record = {
                    "step": step, "physical_time_s": physical_time,
                    "root_to_tip_extension_um": crack_growth_metrics(
                        state.crack_network, initial_crack_length_m=cfg.geometry.a0,
                    ).max_root_to_tip_path_extension_m * 1e6,
                    "node_count": state.mesh.nn, "element_count": state.mesh.ne,
                    "active_tip_count": len(state.crack_network.active_tip_ids),
                    "levels_added": len(adaptation.lineages),
                    "elements_marked": sum(len(item.refined_parent_element_ids) for item in adaptation.lineages),
                    "elements_added_by_conformity": sum(item.elements_added_by_conformity for item in adaptation.lineages),
                    "minimum_active_tip_hbar_m": min(adaptation.active_tip_hbar_m.values()),
                    "maximum_active_tip_hbar_m": max(adaptation.active_tip_hbar_m.values()),
                    "parent_energy_J_per_m": adaptation.parent_energy_J_per_m,
                    "prolonged_energy_J_per_m": adaptation.prolonged_energy_J_per_m,
                    "refined_equilibrium_energy_J_per_m": adaptation.refined_equilibrium_energy_J_per_m,
                    "parent_vs_prolonged_relative_error": adaptation.parent_vs_prolonged_relative_error,
                    "refined_equilibrium_relative_correction": adaptation.refined_equilibrium_relative_correction,
                    "reaction_prolongated_N_per_m": adaptation.reaction_prolongated_N_per_m,
                    "reaction_refined_equilibrium_N_per_m": adaptation.reaction_refined_equilibrium_N_per_m,
                    "adaptation_wall_time_s": time.perf_counter() - adaptation_start,
                    "provider_solve_count": runtime.live_fem_solve_count,
                    "cumulative_mark_operations": adaptation.refinement_marking_diagnostics["cumulative_mark_operations"],
                    "unique_initial_parent_elements_affected": adaptation.refinement_marking_diagnostics["unique_initial_parent_elements_affected"],
                    "physical_marked_area_m2_by_level": [
                        item["physical_marked_area_m2"]
                        for item in adaptation.refinement_marking_diagnostics["levels"]
                    ],
                }
                with (out / "mesh_adaptations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                with (out / "refinement_marking_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "step": step,
                        "root_to_tip_extension_um": record["root_to_tip_extension_um"],
                        **adaptation.refinement_marking_diagnostics,
                    }, sort_keys=True, allow_nan=False) + "\n")
                bundle = {
                    "schema": "v11.multi-tip-engine-bundle/2",
                    "ownership_registry": ownership.to_dict(),
                    "engines": {key: _capture_shared_engine(value) for key, value in engines.items()},
                    "junction_reservoirs": reservoirs,
                }
                growth_now = crack_growth_metrics(state.crack_network, initial_crack_length_m=cfg.geometry.a0)
                checkpoint = ProductionBranchCheckpoint(
                    state=state, shared_process_state=bundle, physical_time_s=physical_time,
                    accepted_load=accepted_load, mesh_identity=_mesh_identity(state.mesh),
                    boundary_condition_state={"opening_m": accepted_load}, provider_runtime=runtime,
                    provider_cache_identity=str(cache_root.resolve()),
                    topology_fingerprint=_hash((state.crack_network, mesh_fingerprint(state.mesh))),
                    front_competitions=competitions, branch_clusters=tuple(clusters.values()),
                    projected_extension_m=growth_now.max_forward_projected_extension_m,
                    physical_extension_m=growth_now.max_root_to_tip_path_extension_m,
                    handoff_guard_diagnostics={}, termination_reason=None,
                )
                write_branch_checkpoint(
                    checkpoint, out / "checkpoint" / "transitions" /
                    f"step{step:07d}_mesh_adaptation_g{counters['mesh_generation']:04d}.json",
                )
                write_branch_checkpoint(checkpoint, out / "checkpoint" / "latest.json")
                write_topology_snapshot(
                    out, state, step=step, reason=f"mesh_adaptation_g{counters['mesh_generation']:04d}",
                    physical_extension_m=growth_now.max_root_to_tip_path_extension_m,
                    branch_birth_count=branch_birth_count,
                    latest_action=state.event_counters.get("latest_successful_action"),
                    growth_metrics=growth_now.to_dict_um(), coalescence_count=coalescence_count,
                )
            adaptation_required = False
        fraction = 1.0
        context = AcceptedStepContext(step, physical_time, float(args.dt), _hash((step, state.crack_network, competitions)))

        def solve_accepted(current, local_context):
            nonlocal latest_sigma
            trial_load = accepted_load + float(args.dU) * fraction
            u = current.displacement.copy(); ep = current.ep_gp.copy(); rho = current.rho_gp.copy()
            sigma = None
            for _ in range(int(args.n_stagger)):
                Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(current.mesh, u, ep, rho, current.damage, D, mat, cohesive_network=current.cohesive_network)
                u, _ = solve_dirichlet(Kmat, Rint, u, current.boundary, 0.5 * trial_load, -0.5 * trial_load)
                Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(current.mesh, u, ep, rho, current.damage, D, mat, cohesive_network=current.cohesive_network)
                ep, rho, _ = update_plasticity(ep, rho, sigma, mat, float(args.temperatures[0]), local_context.duration_s, base.PlasticityModel(cfg.plasticity_barrier, mat), cfg.dislocations)
            Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(current.mesh, u, ep, rho, current.damage, D, mat, cohesive_network=current.cohesive_network)
            u, _ = solve_dirichlet(Kmat, Rint, u, current.boundary, 0.5 * trial_load, -0.5 * trial_load)
            Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(current.mesh, u, ep, rho, current.damage, D, mat, cohesive_network=current.cohesive_network)
            latest_sigma = np.asarray(sigma).copy()
            return replace(current, displacement=u, ep_gp=ep, rho_gp=rho, stored_energy_J_per_m=_stored_energy(current.mesh, u, ep, sigma, D))

        def evaluate_rates(current, _context):
            nonlocal latest_live, latest_rates, runtime
            request = _request(current, candidates, args=args, cfg=cfg, runtime_step=step, cluster=None)
            request = replace(request, cluster_frame={
                "mode": "multi_tip_with_junction_reservoirs",
                "unresolved_cluster_ids": sorted(key for key, value in clusters.items() if value.unresolved),
                "junction_reservoir_ids": sorted(reservoirs),
            }, candidates_by_tip={tip: tuple(candidates) for tip in current.crack_network.active_tip_ids})
            from .kernel_resolver_v11 import resolve_live_topology_request
            latest_live, _ = resolve_live_topology_request(request, cache_root=runtime.cache_root, accepted=True)
            result = {}
            for tip_id in current.crack_network.active_tip_ids:
                tip = current.crack_network.branch(tip_id).tip
                live_tip = next(item for item in latest_live["tips"] if np.allclose(item["tip_xy_m"], tip))
                owner = owner_by_tip[tip_id]
                engine = engines[owner]
                by_id = {item["candidate_id"]: item for item in live_tip["directional"]}
                rows = []
                for candidate in candidates:
                    local = by_id[candidate.candidate_id]
                    local_signed = float(local["J_local_signed_J_per_m2"])
                    marginal = None
                    if bool(local["local_J_valid"]):
                        kinetic = max(local_signed, 0.0)
                    else:
                        start = current.crack_network.branch(tip_id).tip
                        raw = TopologyArm(
                            candidate.candidate_id, tip_id, start,
                            (start[0] + da_phys * candidate.direction_xy[0],
                             start[1] + da_phys * candidate.direction_xy[1]),
                            da_phys, 0.0,
                        )
                        arm, target = clip_arm_at_first_intersection(current.crack_network, raw)
                        if arm.event_reward_m <= 0.0:
                            marginal = 0.0
                        else:
                            realized = extend_network_arm(current.crack_network, arm)
                            if target is not None:
                                realized = mark_coalesced(realized, arm.branch_id, target)
                            ephemeral = replace(current.isolated_copy(), crack_network=realized)
                            ephemeral = apply_causal_sharp_wake_trial_geometry(ephemeral, (arm,))
                            marginal_request = _request(
                                ephemeral, candidates, args=args, cfg=cfg,
                                runtime_step=step, cluster=None,
                            )
                            marginal_request = replace(
                                marginal_request,
                                cluster_frame={"mode": "candidate_marginal_kinetic_drive"},
                                candidates_by_tip={
                                    active_tip: tuple(candidates)
                                    for active_tip in ephemeral.crack_network.active_tip_ids
                                },
                            )
                            runtime, marginal_live = runtime.evaluate_trial(marginal_request)
                            trial_energy = float(
                                marginal_live["base_equilibrium"]["recoverable_potential_energy_J_per_m"]
                            )
                            marginal = (
                                float(latest_live["base_equilibrium"]["recoverable_potential_energy_J_per_m"])
                                - trial_energy
                            ) / float(arm.event_reward_m)
                        kinetic = max(float(marginal), 0.0)
                    rate = preview_production_cleavage_rate(
                        engine, candidate, signed_J_J_per_m2=kinetic,
                        Eprime_Pa=float(mat.Eprime), temperature_K=float(args.temperatures[0]),
                    )
                    rows.append(replace(
                        rate,
                        J_local_signed_J_per_m2=local_signed,
                        local_J_valid=bool(local["local_J_valid"]),
                        G_marginal_J_per_m2=marginal,
                        J_kin_used_J_per_m2=kinetic,
                        local_J_invalid_reason=local.get("local_J_invalid_reason"),
                    ))
                result[tip_id] = tuple(rows)
            latest_rates = result
            return result

        trial_data = {}
        def trial_action(current, tip_id, proposal):
            policy = branch_birth_policy(proposal, committed_branch_birth_count=branch_birth_count)
            if not policy.permitted:
                return TopologyTrialResult(False, current, proposal.action_id, 0.0, 0.0, 0.0, policy.veto_reason)
            if proposal.action_type == "two_arm" and not ownership.recursive_branch_eligible(tip_id):
                return TopologyTrialResult(
                    False, current, proposal.action_id, 0.0, 0.0, 0.0,
                    "parent_process_zone_still_unresolved",
                )
            network = current.crack_network
            trial_cluster = None
            if proposal.action_type == "two_arm":
                network, trial_cluster = create_unresolved_branch_cluster(
                    network, parent_branch_id=tip_id,
                    candidate_ids=proposal.member_candidate_ids,
                    event_index=branch_birth_count + 1,
                    shared_process_state={
                        "ownership": "shared_unresolved_cluster_engine",
                        "engine_owner_id": owner_by_tip[tip_id],
                        "engine_type": type(engines[owner_by_tip[tip_id]]).__name__,
                        "B": float(engines[owner_by_tip[tip_id]].B),
                        "N_em": float(engines[owner_by_tip[tip_id]].N_em),
                        "W_emit": float(engines[owner_by_tip[tip_id]].W_emit),
                        "time_s": float(engines[owner_by_tip[tip_id]].t),
                        "birth_step": int(context.step),
                        "birth_extension_m": crack_growth_metrics(
                            current.crack_network, initial_crack_length_m=cfg.geometry.a0,
                        ).max_root_to_tip_path_extension_m,
                    },
                    conserved_ledgers={name: float(current.energy_ledgers.get(name, 0.0)) for name in (
                        "retained", "mobile", "escaped", "recovered", "stored_energy", "emission_work", "unconsumed_action",
                    )},
                )
            cmap = {item.candidate_id: item for item in candidates}
            arms = []
            for candidate_id in proposal.member_candidate_ids:
                candidate = cmap[candidate_id]
                branch_id = tip_id if trial_cluster is None else next(
                    item for item in trial_cluster.arm_branch_ids if network.branch(item).local_state.get("candidate_id") == candidate_id
                )
                start = network.branch(branch_id).tip
                raw = TopologyArm(candidate_id, branch_id, start, (start[0] + da_phys * candidate.direction_xy[0], start[1] + da_phys * candidate.direction_xy[1]), da_phys, 0.0)
                clipped, target = clip_arm_at_first_intersection(network, raw)
                drive = {item.candidate_id: item for item in latest_rates[tip_id]}[candidate_id]
                engine = engines[owner_by_tip[tip_id]]
                _, _, barrier = engine.lambda_cleave(engine.sigma_tip(drive.K_directional_Pa_sqrt_m / math.sqrt(candidate.gamma_rel)), float(args.temperatures[0]))
                resistance = hazard_resistance_J_per_m2(barrier_J=barrier, cooperative_hits=float(engine.f.m_hits), burgers_vector_m=float(engine.b), gamma_relative=candidate.gamma_rel)
                arms.append((replace(clipped, hazard_dissipation_J_per_m=resistance * clipped.event_reward_m), target))

            def geometry(trial, realized_arms):
                realized = network
                for arm, target in arms:
                    realized = extend_network_arm(realized, arm)
                    if target is not None:
                        realized = mark_coalesced(realized, arm.branch_id, target)
                trial = replace(trial, crack_network=realized)
                return apply_causal_sharp_wake_trial_geometry(trial, realized_arms)

            def equilibrate(trial):
                request = _request(trial, candidates, args=args, cfg=cfg, runtime_step=step, cluster=trial_cluster)
                request = replace(
                    request,
                    cluster_frame={"mode": "multi_tip_trial", "trial_cluster_id": None if trial_cluster is None else trial_cluster.cluster_id},
                    candidates_by_tip={active_tip: tuple(candidates) for active_tip in trial.crack_network.active_tip_ids},
                )
                runtime_next, live = runtime.evaluate_trial(request)
                trial_data[(tip_id, proposal.action_id)] = (trial_cluster, tuple(arms), request, live, runtime_next)
                return replace(trial, displacement=np.asarray(live["base_equilibrium"]["displacement"]), stored_energy_J_per_m=float(live["base_equilibrium"]["recoverable_potential_energy_J_per_m"]))

            return execute_topology_trial(
                current, proposal, tuple(item[0] for item in arms),
                apply_trial_geometry=geometry, equilibrate_fixed_load=equilibrate,
                network_geometry_already_realized=True,
            )

        def update_process(current, local_context, selected_tip, proposal):
            nonlocal engines
            from .crystal import near_tip_stress_tensor
            selected_owner = owner_by_tip.get(selected_tip) if selected_tip else None
            evolved = {}
            for owner in sorted(set(owner_by_tip.values())):
                engine = copy.deepcopy(engines[owner])
                owner_tips = [tip for tip, value in owner_by_tip.items() if value == owner]
                K = max((rate.K_directional_Pa_sqrt_m for tip in owner_tips for rate in latest_rates[tip]), default=0.0)
                probe_tip = current.crack_network.branch(owner_tips[0]).tip
                near_tip_stress_tensor(
                    latest_sigma, current.mesh, np.asarray(probe_tip),
                    3.0 * max(float(getattr(current.mesh, "hbar_tip", 0.0) or current.mesh.hbar), 1e-12),
                )
                residual = max((hazard.residual_action for tip in owner_tips for hazard in competitions[tip].hazard_states), default=0.0)
                expected = selected_owner == owner
                engine.B = 1.0 if expected else residual
                if hasattr(engine, "hazard_action_current"):
                    engine.hazard_action_current = 1.0 if expected else residual
                if hasattr(engine, "hazard_threshold_action"):
                    # Directional clocks are authoritative.  A kinetically
                    # complete but energy-vetoed event remains pending, so the
                    # scalar compatibility observer must not consume it.
                    engine.hazard_threshold_action = 1.0 if expected else 1.0e300
                info = engine.step(K, float(args.temperatures[0]), local_context.duration_s)
                if bool(info.get("fired")) != expected or int(info.get("n_fire", 0)) > 1:
                    raise DirectionalStepRefinementRequired(max(float(info.get("physical_hazard_action_step", info.get("dB", 0.0))), 1e-300), max(float(getattr(args, "adaptive_event_target", 0.15)) * 0.5, 1e-6))
                evolved[owner] = engine
            engines = evolved
            counters = dict(current.event_counters)
            counters["accepted_steps"] = step
            counters["shared_state_updates"] = counters.get("shared_state_updates", 0) + len(engines)
            return replace(current, event_counters=counters)

        while True:
            context = replace(context, duration_s=float(args.dt) * fraction)
            try:
                result = advance_multi_tip_step(
                    state, competitions, context,
                    correlation_interval_s=float(shared_engine.f.tau_c),
                    solve_accepted=solve_accepted, evaluate_rates=evaluate_rates,
                    trial_action=trial_action, update_process_states=update_process,
                    maximum_directional_action_increment=float(getattr(args, "adaptive_event_target", 0.15)),
                )
                break
            except DirectionalStepRefinementRequired as error:
                shrink = 0.7 * error.target_increment / max(error.predicted_increment, 1e-300)
                next_fraction = max(float(getattr(args, "adaptive_min_frac", 1e-8)), fraction * min(0.5, shrink))
                if next_fraction >= fraction or next_fraction <= float(getattr(args, "adaptive_min_frac", 1e-8)):
                    raise RuntimeError("v11 resolved multi-tip stepping reached its minimum fraction") from error
                fraction = next_fraction
        state = result.state
        competitions = dict(result.competitions)
        accepted_load += float(args.dU) * fraction
        physical_time += context.duration_s
        with (out / "directional_rates.jsonl").open("a", encoding="utf-8") as stream:
            for tip_id in sorted(result.rates_by_tip):
                hazard_by_candidate = {
                    item.candidate_id: item for item in competitions[tip_id].hazard_states
                }
                for rate in result.rates_by_tip[tip_id]:
                    hazard = hazard_by_candidate[rate.candidate_id]
                    stream.write(json.dumps({
                        "step": step, "physical_time_s": physical_time,
                        "accepted_state_id": context.accepted_state_id,
                        "tip_id": tip_id, "candidate_id": rate.candidate_id,
                        "J_local_signed_J_per_m2": rate.J_local_signed_J_per_m2,
                        "local_J_valid": rate.local_J_valid,
                        "local_J_invalid_reason": rate.local_J_invalid_reason,
                        "G_marginal_J_per_m2": rate.G_marginal_J_per_m2,
                        "J_kin_used_J_per_m2": rate.J_kin_used_J_per_m2,
                        "lambda_directional_per_s": rate.lambda_per_s,
                        "accumulated_integrated_hazard_H": hazard.action,
                        "current_threshold_H_star": hazard.current_threshold_action,
                        "directional_event_ordinal": hazard.completed_event_count + 1,
                        "pending_event_ids": [item.event_id for item in hazard.pending_events],
                    }, sort_keys=True, allow_nan=False) + "\n")
        for item in result.trials:
            proposal_item = item.diagnostic.proposal
            result_item = item.diagnostic.result
            rate_map = {rate.candidate_id: rate for rate in result.rates_by_tip[item.tip_id]}
            record = {field: None for field in TRIAL_FIELDS}
            arm_count = len(proposal_item.member_candidate_ids)
            before_equilibrium = (latest_live or {}).get("base_equilibrium", {})
            trial_payload = trial_data.get((item.tip_id, proposal_item.action_id))
            after_live = None if trial_payload is None else trial_payload[3]
            after_equilibrium = {} if after_live is None else after_live.get("base_equilibrium", {})
            reaction_before = before_equilibrium.get("reaction_force")
            reaction_after = after_equilibrium.get("reaction_force")

            def apparent_compliance(reaction):
                if reaction is None or not math.isfinite(float(reaction)) or abs(float(reaction)) <= 1.0e-300:
                    return None
                return float(accepted_load) / abs(float(reaction))

            record.update({
                "step": step, "physical_time_s": physical_time,
                "accepted_state_id": context.accepted_state_id,
                "trial_id": f"{item.tip_id}:{proposal_item.action_id}",
                "action_type": proposal_item.action_type,
                "participating_front_ids": [item.tip_id],
                "candidate_ids": list(proposal_item.member_candidate_ids),
                "pending_event_ids": [f"{item.tip_id}:{event}" for event in proposal_item.member_event_ids],
                "completion_times_s": list(proposal_item.completion_times_s),
                "correlation_time_difference_s": max(proposal_item.completion_times_s) - min(proposal_item.completion_times_s),
                "signed_directional_J_J_per_m2": [rate_map[candidate].signed_J_J_per_m2 for candidate in proposal_item.member_candidate_ids],
                "positive_directional_J_J_per_m2": [rate_map[candidate].positive_J_J_per_m2 for candidate in proposal_item.member_candidate_ids],
                "directional_K_Pa_sqrt_m": [rate_map[candidate].K_directional_Pa_sqrt_m for candidate in proposal_item.member_candidate_ids],
                "J_local_signed_J_per_m2": [rate_map[candidate].J_local_signed_J_per_m2 for candidate in proposal_item.member_candidate_ids],
                "local_J_valid": [rate_map[candidate].local_J_valid for candidate in proposal_item.member_candidate_ids],
                "local_J_invalid_reason": [rate_map[candidate].local_J_invalid_reason for candidate in proposal_item.member_candidate_ids],
                "G_marginal_J_per_m2": [rate_map[candidate].G_marginal_J_per_m2 for candidate in proposal_item.member_candidate_ids],
                "J_kin_used_J_per_m2": [rate_map[candidate].J_kin_used_J_per_m2 for candidate in proposal_item.member_candidate_ids],
                "lambda_directional_per_s": [rate_map[candidate].lambda_per_s for candidate in proposal_item.member_candidate_ids],
                "applied_displacement_m": float(accepted_load),
                "reaction_force_before_N_per_m": reaction_before,
                "reaction_force_after_N_per_m": reaction_after,
                "apparent_compliance_before_m2_per_N": apparent_compliance(reaction_before),
                "apparent_compliance_after_m2_per_N": apparent_compliance(reaction_after),
                "topology_fingerprint_before": (latest_live or {}).get("topology_fingerprint"),
                "topology_fingerprint_after": None if after_live is None else after_live.get("topology_fingerprint"),
                "proposed_arm_lengths_m": [da_phys] * arm_count,
                "realized_arm_lengths_m": [da_phys] * arm_count if result_item.accepted else [0.0] * arm_count,
                "pretrial_potential_energy_J_per_m": result_item.state.stored_energy_J_per_m + result_item.energy_release_J_per_m,
                "posttrial_potential_energy_J_per_m": result_item.state.stored_energy_J_per_m,
                "released_energy_J_per_m": result_item.energy_release_J_per_m,
                "hazard_derived_cost_per_arm_J_per_m": [result_item.hazard_dissipation_J_per_m / arm_count] * arm_count,
                "total_dissipative_cost_J_per_m": result_item.hazard_dissipation_J_per_m,
                "net_energy_margin_J_per_m": result_item.energy_margin_J_per_m,
                "relative_energy_residual": result_item.energy_margin_J_per_m / max(abs(result_item.state.stored_energy_J_per_m), 1e-300),
                "geometry_status": "realized" if result_item.accepted else "rolled_back",
                "equilibrium_status": "converged" if result_item.accepted else "trial_vetoed",
                "provider_identity": PROVIDER_ID, "accepted": item.diagnostic.selected,
                "veto_reason": None if item.diagnostic.selected else result_item.rejection_reason,
                "reservation_result": "accepted" if item.diagnostic.selected else "released",
                "consumption_result": "consumed" if item.diagnostic.selected else "preserved",
                "pretrial_state_hash": context.accepted_state_id,
                "postrollback_state_hash": None if result_item.accepted else context.accepted_state_id,
                "trial_copy_bytes": result_item.trial_copy_bytes,
                "trial_copy_wall_time_s": result_item.trial_copy_wall_time_s,
            })
            writer.append_trial(record)
        snapshot_reason = None
        latest_mechanics = {}
        if result.selected_proposal is not None:
            adaptation_required = True
            tip = result.selected_tip_id
            key = (tip, result.selected_proposal.action_id)
            trial_cluster, arms, request, live, runtime = trial_data[key]
            runtime = runtime.accept_trial(request, live)
            counters = dict(state.event_counters)
            counters["latest_successful_action"] = result.selected_proposal.action_id
            if trial_cluster is not None:
                branch_birth_count += 1
                counters["branch_birth_count"] = branch_birth_count
                parent_owner = owner_by_tip[tip]
                owner_engine = engines.pop(parent_owner)
                clusters[trial_cluster.cluster_id] = trial_cluster
                engines[trial_cluster.cluster_id] = owner_engine
                competitions.pop(tip)
                for child in trial_cluster.arm_branch_ids:
                    competitions[child] = DirectionalCompetitionState.initialize(
                        candidates,
                        global_hazard_seed=tip_lineage_seed(
                            state.competition.global_hazard_seed,
                            trial_cluster.cluster_id, child,
                        ),
                    )
                ownership = ownership.branch(tip, trial_cluster.cluster_id, trial_cluster.arm_branch_ids)
                owner_by_tip = dict(ownership.owner_by_tip)
                snapshot_reason = "branch_birth"
                branches = [state.crack_network.branch(child) for child in trial_cluster.arm_branch_ids]
                branch_record = {field: None for field in BRANCH_EVENT_FIELDS}
                branch_record.update({
                    "event_record_id": f"{tip}:{result.selected_proposal.action_id}",
                    "step": step, "branch_junction": list(trial_cluster.junction_xy_m),
                    "parent_front": tip, "arm_front_ids": list(trial_cluster.arm_branch_ids),
                    "arm_directions": [branch.current_orientation_rad for branch in branches],
                    "plane_identities": list(result.selected_proposal.member_candidate_ids),
                    "event_ids_consumed": [f"{tip}:{event}" for event in result.selected_proposal.member_event_ids],
                    "completion_time_difference_s": max(result.selected_proposal.completion_times_s) - min(result.selected_proposal.completion_times_s),
                    "arm_lengths_m": [branch.physical_path_length_m for branch in branches],
                    "tip_positions_m": [list(branch.tip) for branch in branches],
                    "tip_separation_m": math.dist(branches[0].tip, branches[1].tip),
                    "shared_cluster_id": trial_cluster.cluster_id,
                    "shared_cluster_state_hash": _hash(trial_cluster.shared_process_state),
                    "topology_fingerprint": runtime.routing.topology_fingerprint,
                    "released_energy_J_per_m": next(item.diagnostic.result.energy_release_J_per_m for item in result.trials if item.diagnostic.selected),
                    "total_cost_J_per_m": next(item.diagnostic.result.hazard_dissipation_J_per_m for item in result.trials if item.diagnostic.selected),
                    "energy_margin_J_per_m": next(item.diagnostic.result.energy_margin_J_per_m for item in result.trials if item.diagnostic.selected),
                })
                writer.branch_event(branch_record)
            for arm, target in arms:
                if target is not None:
                    coalescence_count += 1
                    counters["coalescence_count"] = coalescence_count
                    competitions.pop(arm.branch_id, None)
                    ownership, removable_engine = ownership.retire_tip(arm.branch_id)
                    owner_by_tip = dict(ownership.owner_by_tip)
                    if removable_engine is not None:
                        engines.pop(removable_engine, None)
                    snapshot_reason = "coalescence"
            state = replace(state, event_counters=counters)
            selected_trial = next(item for item in result.trials if item.diagnostic.selected)
            latest_mechanics = {
                "energy_summary": {
                    "released_J_per_m": selected_trial.diagnostic.result.energy_release_J_per_m,
                    "cost_J_per_m": selected_trial.diagnostic.result.hazard_dissipation_J_per_m,
                    "margin_J_per_m": selected_trial.diagnostic.result.energy_margin_J_per_m,
                },
                "J_K_summary": {
                    tip: [
                        {"candidate_id": rate.candidate_id, "signed_J_J_per_m2": rate.signed_J_J_per_m2, "K_Pa_sqrt_m": rate.K_directional_Pa_sqrt_m}
                        for rate in result.rates_by_tip[tip]
                    ] for tip in result.rates_by_tip
                },
            }

        # Resolve every local cluster only after the exact live contours identify both tips.
        for cluster_id, pending in tuple(clusters.items()):
            if not pending.unresolved:
                continue
            valid = []
            for branch_id in pending.arm_branch_ids:
                branch = state.crack_network.branch(branch_id)
                match = next((item for item in (latest_live or {}).get("tips", ()) if np.allclose(item["tip_xy_m"], branch.tip)), None)
                valid.append(bool(match and match["directional"] and all(row["local_contour_valid"] for row in match["directional"])))
            guard = evaluate_unresolved_cluster_guard(state.crack_network, pending, process_zone_length_m=float(args.L_pz), local_J_contour_radius_m=float(getattr(args, "rJ", None) or args.L_pz), independently_valid_local_J=tuple(valid))
            writer.cluster({
                "step": step, "cluster_id": cluster_id,
                "parent_tip": pending.parent_branch_id,
                "birth_step": pending.shared_process_state.get("birth_step"),
                "birth_extension_m": pending.shared_process_state.get("birth_extension_m"),
                "arm_ids": list(pending.arm_branch_ids),
                "arm_lengths_m": list(guard.arm_arclengths_from_junction_m),
                "tip_separation_m": guard.tip_separation_m,
                "process_owner_id": (
                    cluster_id if pending.unresolved else f"reservoir:{cluster_id}"
                ),
                "unresolved": pending.unresolved,
                "sufficient_post_junction_length": list(guard.sufficient_post_junction_length),
                "separation_reaches_process_zone": guard.separation_reaches_process_zone,
                "local_contours_overlap": guard.local_contours_overlap,
                "independently_valid_local_J": list(guard.independently_valid_local_J),
                "handoff_required": guard.handoff_required,
                "handoff_step": step if guard.handoff_required else None,
                "junction_reservoir_id": f"reservoir:{cluster_id}" if guard.handoff_required else None,
                "resolved_tip_engine_ids": list(pending.arm_branch_ids) if guard.handoff_required else [],
            })
            if guard.handoff_required:
                resolved = resolve_unresolved_cluster(
                    state.crack_network, pending, candidates=candidates,
                    global_hazard_seed=state.competition.global_hazard_seed,
                    fresh_tip_factory=_fresh_tip_payload,
                    existing_competitions={child: competitions[child] for child in pending.arm_branch_ids},
                )
                resolved = replace(resolved, reservoir=replace(
                    resolved.reservoir,
                    historical_process_state={
                        **resolved.reservoir.historical_process_state,
                        "process_engine_checkpoint": _capture_shared_engine(engines[cluster_id]),
                    },
                ))
                state = replace(state, crack_network=resolved.network)
                clusters[cluster_id] = resolved.cluster
                reservoirs[resolved.reservoir.reservoir_id] = resolved.reservoir
                engines.pop(cluster_id, None)
                for child in resolved.cluster.arm_branch_ids:
                    owner_by_tip[child] = child; engines[child] = engine_factory()
                    competitions[child] = resolved.tips[child].competition
                ownership = ownership.resolve(
                    cluster_id, resolved.cluster.arm_branch_ids, resolved.reservoir.reservoir_id,
                )
                owner_by_tip = dict(ownership.owner_by_tip)
                writer.cluster({
                    "step": step, "cluster_id": cluster_id,
                    "parent_tip": pending.parent_branch_id,
                    "birth_step": pending.shared_process_state.get("birth_step"),
                    "birth_extension_m": pending.shared_process_state.get("birth_extension_m"),
                    "arm_ids": list(pending.arm_branch_ids),
                    "arm_lengths_m": list(guard.arm_arclengths_from_junction_m),
                    "tip_separation_m": guard.tip_separation_m,
                    "process_owner_id": resolved.reservoir.reservoir_id,
                    "unresolved": False,
                    "sufficient_post_junction_length": list(guard.sufficient_post_junction_length),
                    "separation_reaches_process_zone": guard.separation_reaches_process_zone,
                    "local_contours_overlap": guard.local_contours_overlap,
                    "independently_valid_local_J": list(guard.independently_valid_local_J),
                    "handoff_required": True, "handoff_step": step,
                    "junction_reservoir_id": resolved.reservoir.reservoir_id,
                    "resolved_tip_engine_ids": list(resolved.cluster.arm_branch_ids),
                })
                snapshot_reason = "cluster_resolved"

        active = state.crack_network.active_tip_ids
        competitions = {tip: competitions[tip] for tip in active}
        state = replace(state, competition=competitions[sorted(active)[0]], junction_process_state={
            "crack_representation": "sharp_wake_causal_v11",
            "reservoirs": reservoirs, "clusters": clusters,
        })
        selected_result = next((item.diagnostic.result for item in result.trials if item.diagnostic.selected), None)
        writer.energy({
            "step": step, "accepted_state_id": context.accepted_state_id,
            "stored_energy_J_per_m": state.stored_energy_J_per_m,
            "released_energy_J_per_m": 0.0 if selected_result is None else selected_result.energy_release_J_per_m,
            "dissipative_cost_J_per_m": 0.0 if selected_result is None else selected_result.hazard_dissipation_J_per_m,
            "residual_J_per_m": 0.0 if selected_result is None else selected_result.energy_margin_J_per_m,
        })
        for branch in state.crack_network.branches:
            writer.front({"step": step, "front_id": branch.branch_id, "parent_front_id": branch.parent_branch_id, "status": branch.status, "termination_reason": branch.local_state.get("termination_reason"), "tip_x_m": branch.tip[0], "tip_y_m": branch.tip[1], "arclength_m": branch.physical_path_length_m})
        ownership.validate(
            active, engine_ids=engines, cluster_ids=clusters, reservoir_ids=reservoirs,
        )
        bundle = {
            "schema": "v11.multi-tip-engine-bundle/2",
            "ownership_registry": ownership.to_dict(),
            "engines": {key: _capture_shared_engine(value) for key, value in engines.items()},
            "junction_reservoirs": reservoirs,
        }
        growth = crack_growth_metrics(state.crack_network, initial_crack_length_m=cfg.geometry.a0)
        extension = growth.max_root_to_tip_path_extension_m
        checkpoint = ProductionBranchCheckpoint(
            state=state, shared_process_state=bundle, physical_time_s=physical_time,
            accepted_load=accepted_load, mesh_identity=_mesh_identity(state.mesh),
            boundary_condition_state={"opening_m": accepted_load}, provider_runtime=runtime,
            provider_cache_identity=str(cache_root.resolve()), topology_fingerprint=runtime.routing.topology_fingerprint or _hash(state.crack_network),
            front_competitions=competitions, branch_clusters=tuple(clusters.values()),
            projected_extension_m=growth.max_forward_projected_extension_m,
            physical_extension_m=extension, handoff_guard_diagnostics={}, termination_reason=None,
        )
        write_branch_checkpoint(checkpoint, out / "checkpoint" / "latest.json")
        crossed = next((value for value in (25e-6, 50e-6, 75e-6, 100e-6, 250e-6, 500e-6, 750e-6, 1000e-6) if last_snapshot_extension < value <= extension), None)
        if snapshot_reason is not None or crossed is not None:
            reason = snapshot_reason or f"extension_{int(round(crossed * 1e6))}um"
            write_branch_checkpoint(
                checkpoint,
                out / "checkpoint" / "transitions" / f"step{step:07d}_{reason}.json",
            )
            write_topology_snapshot(
                out, state, step=step, reason=reason,
                physical_extension_m=extension, branch_birth_count=branch_birth_count,
                latest_action=state.event_counters.get("latest_successful_action"),
                mechanics=latest_mechanics,
                growth_metrics=growth.to_dict_um(), coalescence_count=coalescence_count,
            )
            last_snapshot_extension = extension
        target = float(getattr(args, "target_crack_extension_um", float("inf"))) * 1e-6
        if extension >= target:
            termination = "target_reached"; break
        if not active:
            termination = "physical_veto_no_branch"; break

    termination = termination or "physical_veto_no_branch"
    checkpoint = replace(checkpoint, termination_reason=termination)
    write_branch_checkpoint(checkpoint, out / "checkpoint" / "latest.json")
    write_topology_snapshot(
        out, state, step=int(state.event_counters.get("accepted_steps", 0)),
        reason=termination, physical_extension_m=checkpoint.physical_extension_m,
        branch_birth_count=branch_birth_count,
        latest_action=state.event_counters.get("latest_successful_action"), final=True,
        growth_metrics=growth.to_dict_um(), coalescence_count=coalescence_count,
    )
    writer.complete(status=termination, final_checkpoint="checkpoint/latest.json", validation={
        "checkpoint": True, "branch_birth_count": branch_birth_count,
        "coalescence_count": coalescence_count, "active_tip_count": len(state.crack_network.active_tip_ids),
        "provider_transition": True, "live_fem_solve_count": runtime.live_fem_solve_count,
        **growth.to_dict_um(),
    })
    return 0


__all__ = ["continue_resolved_production"]
