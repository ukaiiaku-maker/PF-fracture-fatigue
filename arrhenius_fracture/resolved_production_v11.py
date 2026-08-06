"""Resolved multi-tip continuation for the v11 production FEM adapter."""
from __future__ import annotations

from dataclasses import replace
import copy
import math
from pathlib import Path
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
from .production_step_loop_v11 import AcceptedStepContext, DirectionalStepRefinementRequired
from .resolved_tip_state_v11 import resolve_unresolved_cluster
from .topology_transaction_v11 import (
    TopologyArm, TopologyTrialResult, apply_sharp_wake_trial_geometry,
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
        owner_by_tip = {tip: tip for tip in competitions}
        state = replace(
            state, crack_network=resolution.network,
            competition=competitions[sorted(competitions)[0]],
            tip_process_state={"mode": "independent_resolved_tips", "tips": resolution.tips},
            junction_process_state={"reservoirs": reservoirs, "clusters": clusters},
        )
    else:
        if resume_bundle.get("schema") != "v11.multi-tip-engine-bundle/1":
            raise ValueError("unsupported resolved multi-tip restart bundle")
        competitions = dict(resume_competitions or {})
        clusters = {item.cluster_id: item for item in resume_clusters}
        reservoirs = dict(resume_bundle.get("junction_reservoirs", {}))
        owner_by_tip = dict(resume_bundle["owner_by_tip"])
        engines = {
            owner: _restore_shared_engine(engine_factory(), payload)
            for owner, payload in resume_bundle["engines"].items()
        }
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
    last_snapshot_extension = state.crack_network.total_physical_crack_length_m - cfg.geometry.a0
    if resume_bundle is None:
        write_topology_snapshot(
            out, state, step=start_step - 1, reason="cluster_resolved",
            physical_extension_m=last_snapshot_extension,
            branch_birth_count=branch_birth_count, latest_action=None,
        )

    for step in range(start_step, int(args.steps) + 1):
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
            nonlocal latest_live, latest_rates
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
                result[tip_id] = tuple(preview_production_cleavage_rate(
                    engine, candidate,
                    signed_J_J_per_m2=float(by_id[candidate.candidate_id]["signed_J_J_per_m2"]),
                    Eprime_Pa=float(mat.Eprime), temperature_K=float(args.temperatures[0]),
                ) for candidate in candidates)
            latest_rates = result
            return result

        trial_data = {}
        def trial_action(current, tip_id, proposal):
            policy = branch_birth_policy(proposal, committed_branch_birth_count=branch_birth_count)
            if not policy.permitted:
                return TopologyTrialResult(False, current, proposal.action_id, 0.0, 0.0, 0.0, policy.veto_reason)
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
                return apply_sharp_wake_trial_geometry(trial, realized_arms, kill_radius_m=max(float(getattr(current.mesh, "hbar_tip", 0.0) or current.mesh.hbar), 0.5e-6))

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
        for item in result.trials:
            proposal_item = item.diagnostic.proposal
            result_item = item.diagnostic.result
            rate_map = {rate.candidate_id: rate for rate in result.rates_by_tip[item.tip_id]}
            record = {field: None for field in TRIAL_FIELDS}
            arm_count = len(proposal_item.member_candidate_ids)
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
            })
            writer.append_trial(record)
        snapshot_reason = None
        latest_mechanics = {}
        if result.selected_proposal is not None:
            tip = result.selected_tip_id
            key = (tip, result.selected_proposal.action_id)
            trial_cluster, arms, request, live, runtime = trial_data[key]
            runtime = runtime.accept_trial(request, live)
            counters = dict(state.event_counters)
            counters["latest_successful_action"] = result.selected_proposal.action_id
            if trial_cluster is not None:
                branch_birth_count += 1
                counters["branch_birth_count"] = branch_birth_count
                parent_owner = owner_by_tip.pop(tip)
                owner_engine = engines.pop(parent_owner)
                clusters[trial_cluster.cluster_id] = trial_cluster
                engines[trial_cluster.cluster_id] = owner_engine
                competitions.pop(tip)
                for child in trial_cluster.arm_branch_ids:
                    competitions[child] = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=state.competition.global_hazard_seed + branch_birth_count)
                    owner_by_tip[child] = trial_cluster.cluster_id
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
                    competitions.pop(arm.branch_id, None); owner_by_tip.pop(arm.branch_id, None)
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
            if guard.handoff_required:
                resolved = resolve_unresolved_cluster(state.crack_network, pending, candidates=candidates, global_hazard_seed=state.competition.global_hazard_seed, fresh_tip_factory=_fresh_tip_payload)
                state = replace(state, crack_network=resolved.network)
                clusters[cluster_id] = resolved.cluster
                reservoirs[resolved.reservoir.reservoir_id] = resolved.reservoir
                engines.pop(cluster_id, None)
                for child in resolved.cluster.arm_branch_ids:
                    owner_by_tip[child] = child; engines[child] = engine_factory()
                snapshot_reason = "cluster_resolved"

        active = state.crack_network.active_tip_ids
        competitions = {tip: competitions[tip] for tip in active}
        state = replace(state, competition=competitions[sorted(active)[0]], junction_process_state={"reservoirs": reservoirs, "clusters": clusters})
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
        bundle = {"schema": "v11.multi-tip-engine-bundle/1", "owner_by_tip": owner_by_tip, "engines": {key: _capture_shared_engine(value) for key, value in engines.items()}, "junction_reservoirs": reservoirs}
        extension = state.crack_network.total_physical_crack_length_m - cfg.geometry.a0
        checkpoint = ProductionBranchCheckpoint(
            state=state, shared_process_state=bundle, physical_time_s=physical_time,
            accepted_load=accepted_load, mesh_identity=_mesh_identity(state.mesh),
            boundary_condition_state={"opening_m": accepted_load}, provider_runtime=runtime,
            provider_cache_identity=str(cache_root.resolve()), topology_fingerprint=runtime.routing.topology_fingerprint or _hash(state.crack_network),
            front_competitions=competitions, branch_clusters=tuple(clusters.values()),
            projected_extension_m=max(branch.tip[0] for branch in state.crack_network.branches) - cfg.geometry.a0,
            physical_extension_m=extension, handoff_guard_diagnostics={}, termination_reason=None,
        )
        write_branch_checkpoint(checkpoint, out / "checkpoint" / "latest.json")
        crossed = next((value for value in (25e-6, 50e-6, 75e-6, 100e-6) if last_snapshot_extension < value <= extension), None)
        if snapshot_reason is not None or crossed is not None:
            reason = snapshot_reason or f"extension_{int(round(crossed * 1e6))}um"
            write_topology_snapshot(
                out, state, step=step, reason=reason,
                physical_extension_m=extension, branch_birth_count=branch_birth_count,
                latest_action=state.event_counters.get("latest_successful_action"),
                mechanics=latest_mechanics,
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
    )
    writer.complete(status=termination, final_checkpoint="checkpoint/latest.json", validation={
        "checkpoint": True, "branch_birth_count": branch_birth_count,
        "coalescence_count": coalescence_count, "active_tip_count": len(state.crack_network.active_tip_ids),
        "provider_transition": True, "live_fem_solve_count": runtime.live_fem_solve_count,
    })
    return 0


__all__ = ["continue_resolved_production"]
