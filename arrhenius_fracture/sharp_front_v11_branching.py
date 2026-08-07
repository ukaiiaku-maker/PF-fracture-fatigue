"""Bounded monotonic 2-D FEM adapter for v11 mechanistic branching.

This is a separate production loop.  It reuses the established mesh, FEM,
plasticity, material, and v10.2.28 engine construction, but never enters the
legacy probabilistic branching code in :mod:`sharp_front`.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping

import numpy as np

from .branch_checkpoint_v11 import (
    ProductionBranchCheckpoint, restore_branch_checkpoint, write_branch_checkpoint,
)
from .branch_cluster_guard_v11 import evaluate_unresolved_cluster_guard
from .branch_cluster_v11 import BranchClusterState, create_unresolved_branch_cluster
from .branch_output_v11 import (
    BRANCH_EVENT_FIELDS, ENERGY_FIELDS, FRONT_FIELDS, PROVIDER_FIELDS, TRIAL_FIELDS,
    BranchOutputWriter,
)
from .branch_snapshot_v11 import write_topology_snapshot
from .crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from .directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalRate, preview_production_cleavage_rate,
    tungsten_cleavage_candidates,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .live_topology_kernel_registry_v11 import PREBRANCH_PROVIDER_ID
from .live_topology_kernel_v11 import LiveTopologyRequest, PROVIDER_ID
from .live_topology_runtime_v11 import LiveTopologyRuntime
from .adaptive_multitip_mesh_v11 import adapt_accepted_state_for_trials, mesh_fingerprint
from .network_metrics_v11 import crack_growth_metrics
from .production_step_loop_v11 import (
    AcceptedStepContext, DirectionalStepRefinementRequired, advance_accepted_step,
)
from .topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, TopologyTrialResult,
    apply_causal_sharp_wake_trial_geometry, execute_topology_trial, extend_network_arm,
)


MODEL_ID = "v11.mechanistic_branching.production_fem_adapter/1"


def _hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _stored_energy(mesh, displacement, ep_gp, sigma_gp, D) -> float:
    from .fem import elastic_energy_densities
    density, _ = elastic_energy_densities(mesh, displacement, ep_gp, sigma_gp, D)
    return float(np.sum(density * mesh.area_e))


def _mesh_identity(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.nodes).tobytes())
    digest.update(np.ascontiguousarray(mesh.elems).tobytes())
    return digest.hexdigest()


def _serializable_fields(owner) -> dict[str, Any]:
    result = {}
    for name, value in owner.__dict__.items():
        if callable(value) or name == "mpz":
            continue
        try:
            pickle.dumps(value, protocol=5)
        except Exception:
            continue
        result[name] = value
    return result


def _capture_shared_engine(engine) -> dict[str, Any]:
    return {
        "schema": "v11.shared-production-engine-state/1",
        "engine_type": type(engine).__name__,
        "engine_fields": _serializable_fields(engine),
        "mpz_type": type(engine.mpz).__name__,
        "mpz_fields": _serializable_fields(engine.mpz),
    }


def _restore_shared_engine(engine, payload: Mapping[str, Any]):
    if payload.get("schema") != "v11.shared-production-engine-state/1":
        raise ValueError("unsupported shared production-engine state")
    if payload.get("engine_type") != type(engine).__name__:
        raise RuntimeError("restart engine type differs from initialized production engine")
    for name, value in payload.get("engine_fields", {}).items():
        setattr(engine, name, value)
    for name, value in payload.get("mpz_fields", {}).items():
        setattr(engine.mpz, name, value)
    return engine


def _request(
    state: LiveFEMTopologyState, candidates, *, args, cfg, runtime_step: int,
    cluster: BranchClusterState | None,
) -> LiveTopologyRequest:
    candidate_map = _candidate_map(candidates)
    by_tip = {}
    for tip in state.crack_network.active_tip_ids:
        physical_id = state.crack_network.branch(tip).local_state.get("candidate_id")
        by_tip[tip] = (
            (candidate_map[physical_id],) if physical_id in candidate_map
            else tuple(candidates)
        )
    geometry = {
        name: float(getattr(cfg.geometry, name))
        for name in ("Lx", "Ly", "a0", "notch_half_thickness")
    }
    frame = (
        {"mode": "single_front"}
        if cluster is None else {
            "mode": "shared_unresolved_cluster", "cluster_id": cluster.cluster_id,
            "junction_xy_m": list(cluster.junction_xy_m),
        }
    )
    mat = state.material
    return LiveTopologyRequest(
        mesh=state.mesh, boundary=state.boundary,
        displacement=state.displacement, ep_gp=state.ep_gp, rho_gp=state.rho_gp,
        damage=state.damage, elasticity_D=state.elasticity_D, material=mat,
        cohesive_network=state.cohesive_network, crack_network=state.crack_network,
        candidates_by_tip=by_tip,
        mechanical_configuration_fingerprint=str(os.environ.get("MECHANICAL_CONFIG_SHA256", "v10.2.28-installed")),
        specimen_geometry=geometry,
        boundary_condition_identity=f"symmetric_opening:step={runtime_step}",
        elastic_constants={"E_Pa": float(mat.E), "nu": float(mat.nu), "Eprime_Pa": float(mat.Eprime)},
        cluster_frame=frame, mpz_station_coordinates_m=(), wake_station_coordinates_m=(),
        contour_radius_m=float(getattr(args, "rJ", None) or max(args.L_pz, 1.0e-6)),
        exclude_radius_m=max(float(getattr(state.mesh, "hbar_tip", 0.0) or state.mesh.hbar), 1.0e-12),
    )


def _direct_measurement(state, candidates, args) -> dict[str, Any]:
    from .fem import assemble_mechanics
    from .j_integral import compute_J_integral
    Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    segments = [
        (np.asarray(a), np.asarray(b)) for branch in state.crack_network.branches
        for a, b in zip(branch.path, branch.path[1:])
    ]
    branch = state.crack_network.branch(state.crack_network.active_tip_ids[0])
    directional = []
    ell = float(getattr(args, "rJ", None) or max(args.L_pz, 1.0e-6))
    exclude = max(float(getattr(state.mesh, "hbar_tip", 0.0) or state.mesh.hbar), 1.0e-12)
    for candidate in candidates:
        _, _, info = compute_J_integral(
            state.mesh, state.displacement, sigma, psi, state.damage,
            np.asarray(branch.tip), np.asarray(candidate.direction_xy), state.material,
            ell=ell, crack_segments=segments, exclude_radius=exclude,
        )
        signed = float(info.get("J_signed", info.get("J", 0.0)))
        positive = max(signed, 0.0)
        directional.append({
            "candidate_id": candidate.candidate_id,
            "signed_J_J_per_m2": signed,
            "positive_J_J_per_m2": positive,
            "K_directional_Pa_sqrt_m": math.sqrt(float(state.material.Eprime) * positive),
        })
    return {
        # The accepted solver supplies the authoritative Dirichlet reaction;
        # this helper owns only directional configurational measurements.
        "reaction_force": 0.0,
        "recoverable_potential_energy_J_per_m": state.stored_energy_J_per_m,
        "directional": directional,
    }


def _candidate_map(candidates):
    return {candidate.candidate_id: candidate for candidate in candidates}


def _realized_trial_network(state, proposal, candidates, da_phys, cluster):
    cmap = _candidate_map(candidates)
    network = state.crack_network
    trial_cluster = cluster
    if proposal.action_type == "two_arm" and cluster is None:
        network, trial_cluster = create_unresolved_branch_cluster(
            network, parent_branch_id=ROOT_BRANCH_ID,
            candidate_ids=proposal.member_candidate_ids,
            event_index=state.competition.competition_event_index + 1,
            shared_process_state=dict(state.tip_process_state),
            conserved_ledgers={name: float(state.energy_ledgers.get(name, 0.0)) for name in (
                "retained", "mobile", "escaped", "recovered", "stored_energy",
                "emission_work", "unconsumed_action",
            )},
        )
    arms = []
    for candidate_id in proposal.member_candidate_ids:
        candidate = cmap[candidate_id]
        if trial_cluster is None:
            branch_id = ROOT_BRANCH_ID
        else:
            branch_id = next(
                item for item in trial_cluster.arm_branch_ids
                if network.branch(item).local_state.get("candidate_id") == candidate_id
            )
        start = network.branch(branch_id).tip
        end = (
            start[0] + da_phys * candidate.direction_xy[0],
            start[1] + da_phys * candidate.direction_xy[1],
        )
        arms.append((candidate, TopologyArm(
            candidate_id=candidate_id, branch_id=branch_id, start_xy_m=start,
            end_xy_m=end, event_reward_m=da_phys, hazard_dissipation_J_per_m=0.0,
        )))
    return network, trial_cluster, tuple(arms)


def run_2d(args):
    from . import sharp_front as base
    from .fem import assemble_mechanics, plane_strain_D, solve_dirichlet
    from .mesh import make_boundary_data, make_tri_mesh
    from .plasticity import update_plasticity

    if bool(getattr(args, "fatigue_cycles", False)):
        raise SystemExit("v11 mechanistic branching supports monotonic loading only")
    if str(args.crack_backend) != "sharp_wake":
        raise SystemExit("v11 mechanistic branching requires the sharp_wake backend")
    cfg = base.make_emergent_config()
    cfg.mesh.nx = args.nx; cfg.mesh.ny = args.ny
    cfg.mesh.tip_h_fine = getattr(args, "tip_h_fine", 0.0) or 0.0
    cfg.mesh.tip_ratio = getattr(args, "tip_ratio", 1.15)
    cfg.loading.n_steps = args.steps; cfg.loading.dU_top = args.dU; cfg.loading.dt = args.dt
    mat = cfg.material
    mesh = make_tri_mesh(cfg.geometry, cfg.mesh, seed=42)
    boundary = make_boundary_data(mesh, cfg.geometry)
    if getattr(args, "crystal_aniso", False):
        from .crystal import W_C11, W_C12, W_C44, cubic_plane_strain_D
        D = cubic_plane_strain_D(
            float(getattr(args, "crystal_C11", None) or W_C11),
            float(getattr(args, "crystal_C12", None) or W_C12),
            float(getattr(args, "crystal_C44", None) or W_C44),
            float(getattr(args, "crystal_theta_deg", 0.0) or 0.0),
        )
    else:
        D = plane_strain_D(mat)
    engine = base.build_engine(args, mat)
    da_phys = float(args.da_phys if args.da_phys is not None else max(5.0 * base.eng_r_pz_hint(args), 2.0e-6))
    engine.f.da = da_phys
    engine.f.max_advances_per_step = 1
    theta = float(getattr(args, "crystal_theta_deg", 0.0) or 0.0)
    candidates = tungsten_cleavage_candidates(
        theta_deg=theta, include_110=bool(getattr(args, "crystal_include_110", False)),
        gamma_110_rel=float(getattr(args, "gamma_110_rel", 1.3) or 1.3),
    )
    seed = int(os.environ.get("CLEAVAGE_HAZARD_SEED", getattr(args, "hazard_seed", 3621) or 3621))
    competition = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=seed)
    network = replace(
        CrackNetworkState.one_tip(((0.0, 0.0), (float(cfg.geometry.a0), 0.0))),
        branching_enabled=True,
    )
    damage = np.zeros(mesh.nn)
    damage[(mesh.nodes[:, 0] <= cfg.geometry.a0) & (np.abs(mesh.nodes[:, 1]) <= cfg.geometry.notch_half_thickness)] = 1.0
    displacement = np.zeros(mesh.ndof); ep_gp = np.zeros((3, mesh.ne)); rho_gp = np.full(mesh.ne, engine.f.rho0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    writer = BranchOutputWriter(out)
    cache_root = Path(os.environ.get("KERNEL_CACHE_ROOT", out / "live_kernel_cache"))
    runtime = LiveTopologyRuntime(str(cache_root)); cluster = None
    physical_time = 0.0; accepted_load = 0.0; termination = None; start_step = 1
    state = LiveFEMTopologyState(
        mesh=mesh, boundary=boundary, damage=damage, displacement=displacement,
        ep_gp=ep_gp, rho_gp=rho_gp, elasticity_D=D, material=mat,
        cohesive_network=None, crack_network=network, competition=competition,
        tip_process_state={"shared_engine_model": type(engine).__name__},
        junction_process_state={"crack_representation": "sharp_wake_causal_v11"}, energy_ledgers={name: 0.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action", "topology_release_J_per_m",
            "hazard_dissipation_J_per_m",
        )}, rng_state=np.random.default_rng(seed).bit_generator.state,
        event_counters={"topology_actions": 0, "shared_state_updates": 0},
        stored_energy_J_per_m=0.0,
    )
    restart_path = os.environ.get("V11_BRANCH_RESTART_CHECKPOINT", "").strip()
    if not restart_path:
        write_topology_snapshot(
            out, state, step=0, reason="initiation", physical_extension_m=0.0,
            branch_birth_count=0, latest_action=None,
        )
    if restart_path:
        restored = restore_branch_checkpoint(restart_path)
        state = restored.state
        restored_junction = dict(state.junction_process_state)
        restored_junction["crack_representation"] = "sharp_wake_causal_v11"
        state = replace(state, junction_process_state=restored_junction)
        runtime = restored.provider_runtime
        physical_time = restored.physical_time_s
        accepted_load = restored.accepted_load
        cluster = restored.branch_clusters[0] if restored.branch_clusters else None
        mesh = state.mesh; boundary = state.boundary; D = state.elasticity_D; mat = state.material
        start_step = int(state.event_counters.get("accepted_steps", 0)) + 1
        if restored.shared_process_state.get("schema") == "v11.multi-tip-engine-bundle/1":
            from .resolved_production_v11 import continue_resolved_production
            return continue_resolved_production(
                args=args, cfg=cfg, state=state, shared_engine=engine,
                runtime=runtime, cluster=None, candidates=candidates,
                writer=writer, out=out, cache_root=cache_root,
                physical_time=physical_time, accepted_load=accepted_load,
                start_step=start_step, engine_factory=lambda: base.build_engine(args, mat),
                resume_bundle=restored.shared_process_state,
                resume_competitions=restored.front_competitions,
                resume_clusters=restored.branch_clusters,
            )
        engine = _restore_shared_engine(engine, restored.shared_process_state)
        if (
            (
                restored.termination_reason == "branch_cluster_independent_tip_handoff_required"
                or bool(restored.handoff_guard_diagnostics.get("handoff_required"))
            )
            and cluster is not None
        ):
            from .resolved_production_v11 import continue_resolved_production
            return continue_resolved_production(
                args=args, cfg=cfg, state=state, shared_engine=engine,
                runtime=runtime, cluster=cluster, candidates=candidates,
                writer=writer, out=out, cache_root=cache_root,
                physical_time=physical_time, accepted_load=accepted_load,
                start_step=start_step, engine_factory=lambda: base.build_engine(args, mat),
            )
    last_measurement = {}; latest_sigma = None; latest_interval_rates = (); latest_live_result = None
    prebranch_snapshot_written = False
    adaptation_required = True

    for step in range(start_step, int(args.steps) + 1):
        if adaptation_required:
            adaptation_start = time.perf_counter()
            inventory = {tip: tuple(candidates) for tip in state.crack_network.active_tip_ids}
            previous_generation = int(state.event_counters.get("mesh_generation", 0))
            previous_operation = int(state.event_counters.get("refinement_operation_index", 0))
            state, adaptation = adapt_accepted_state_for_trials(
                state, inventory, da_phys_m=da_phys,
                tip_h_fine_m=float(getattr(args, "tip_h_fine", 0.0) or 1.0e-6),
                contour_radius_m=float(getattr(args, "rJ", None) or args.L_pz),
                crack_band_radius_m=0.5e-6, accepted_load_m=accepted_load,
                starting_generation=previous_generation,
                starting_operation_index=previous_operation,
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
                mesh = state.mesh; boundary = state.boundary
                growth_now = crack_growth_metrics(
                    state.crack_network, initial_crack_length_m=cfg.geometry.a0,
                )
                record = {
                    "step": step, "physical_time_s": physical_time,
                    "root_to_tip_extension_um": (
                        growth_now.max_root_to_tip_path_extension_m * 1.0e6
                    ),
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
                adaptation_checkpoint = ProductionBranchCheckpoint(
                    state=state, shared_process_state=_capture_shared_engine(engine),
                    physical_time_s=physical_time, accepted_load=accepted_load,
                    mesh_identity=_mesh_identity(state.mesh),
                    boundary_condition_state={"opening_m": accepted_load}, provider_runtime=runtime,
                    provider_cache_identity=str(cache_root.resolve()),
                    topology_fingerprint=_hash((state.crack_network, mesh_fingerprint(state.mesh))),
                    front_competitions={tip: state.competition for tip in state.crack_network.active_tip_ids},
                    branch_clusters=(() if cluster is None else (cluster,)),
                    projected_extension_m=growth_now.max_forward_projected_extension_m,
                    physical_extension_m=growth_now.max_root_to_tip_path_extension_m,
                    handoff_guard_diagnostics={}, termination_reason=None,
                )
                path = out / "checkpoint" / "transitions" / f"step{step:07d}_mesh_adaptation_g{counters['mesh_generation']:04d}.json"
                write_branch_checkpoint(adaptation_checkpoint, path)
                write_branch_checkpoint(adaptation_checkpoint, out / "checkpoint" / "latest.json")
            adaptation_required = False
        trial_fraction = 1.0
        context = AcceptedStepContext(step, physical_time, float(args.dt), _hash((step, state.crack_network, state.competition, mesh_fingerprint(state.mesh))))

        def solve_accepted(current, _context):
            nonlocal last_measurement, latest_sigma
            trial_load = accepted_load + float(args.dU) * trial_fraction
            u = current.displacement.copy(); ep = current.ep_gp.copy(); rho = current.rho_gp.copy()
            sigma = psi = None; reaction = 0.0
            for _ in range(int(args.n_stagger)):
                Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(
                    current.mesh, u, ep, rho, current.damage, D, mat,
                    cohesive_network=current.cohesive_network,
                )
                u, reaction = solve_dirichlet(
                    Kmat, Rint, u, current.boundary, 0.5 * trial_load, -0.5 * trial_load
                )
                Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(
                    current.mesh, u, ep, rho, current.damage, D, mat,
                    cohesive_network=current.cohesive_network,
                )
                plast = base.PlasticityModel(cfg.plasticity_barrier, mat)
                ep, rho, _ = update_plasticity(ep, rho, sigma, mat, float(args.temperatures[0]), _context.duration_s, plast, cfg.dislocations)
            # Transition parity compares equilibrated accepted states.  Plastic
            # evolution changes the internal force, so close the final accepted
            # state once more without another constitutive update.
            Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(
                current.mesh, u, ep, rho, current.damage, D, mat,
                cohesive_network=current.cohesive_network,
            )
            u, reaction = solve_dirichlet(
                Kmat, Rint, u, current.boundary, 0.5 * trial_load, -0.5 * trial_load
            )
            Kmat, Rint, sigma, seq, s1, psi = assemble_mechanics(
                current.mesh, u, ep, rho, current.damage, D, mat,
                cohesive_network=current.cohesive_network,
            )
            energy = _stored_energy(current.mesh, u, ep, sigma, D)
            latest_sigma = np.asarray(sigma).copy()
            solved = replace(current, displacement=u, ep_gp=ep, rho_gp=rho, stored_energy_J_per_m=energy)
            last_measurement = _direct_measurement(solved, candidates, args)
            last_measurement["reaction_force"] = float(reaction)
            return solved

        def rates(current, _context):
            nonlocal latest_interval_rates, latest_live_result
            source = last_measurement["directional"]
            if runtime.routing.active_mechanics_provider == PROVIDER_ID:
                request = _request(current, candidates, args=args, cfg=cfg, runtime_step=step, cluster=cluster)
                live, _ = __import__("arrhenius_fracture.kernel_resolver_v11", fromlist=["resolve_live_topology_request"]).resolve_live_topology_request(
                    request, cache_root=runtime.cache_root, accepted=True
                )
                latest_live_result = live
                source = live["tips"][0]["directional"] if len(live["tips"]) == 1 else [
                    item for tip in live["tips"] for item in tip["directional"]
                    if item["candidate_id"] in {c.candidate_id for c in candidates}
                ]
            by_id = {}
            for item in source:
                by_id.setdefault(item["candidate_id"], item)
            latest_interval_rates = tuple(preview_production_cleavage_rate(
                engine, candidate, signed_J_J_per_m2=float(by_id[candidate.candidate_id]["signed_J_J_per_m2"]),
                Eprime_Pa=float(mat.Eprime), temperature_K=float(args.temperatures[0]),
            ) for candidate in candidates)
            return latest_interval_rates

        trial_requests = {}; trial_live_results = {}; trial_clusters = {}; trial_drives = {}

        def trial_action(current, proposal):
            nonlocal runtime, prebranch_snapshot_written, latest_live_result
            if proposal.action_type == "two_arm" and not prebranch_snapshot_written:
                write_topology_snapshot(
                    out, current, step=step, reason="before_first_branch",
                    physical_extension_m=current.crack_network.total_physical_crack_length_m - cfg.geometry.a0,
                    branch_birth_count=int(current.event_counters.get("branch_birth_count", 0)),
                    latest_action=current.event_counters.get("latest_successful_action"),
                )
                prebranch_snapshot_written = True
            if runtime.routing.active_mechanics_provider == PREBRANCH_PROVIDER_ID:
                request0 = _request(current, candidates, args=args, cfg=cfg, runtime_step=step, cluster=None)
                runtime, live0 = runtime.transition(
                    step=step, state_hash=context.accepted_state_id,
                    legacy_result=last_measurement, request=request0,
                    protected_state=(current.competition, pickle.dumps(engine, protocol=5)),
                )
                latest_live_result = live0
                writer.provider_transition({
                    "step": step, "from_provider": PREBRANCH_PROVIDER_ID, "to_provider": PROVIDER_ID,
                    "state_hash": context.accepted_state_id, "topology_fingerprint": live0["topology_fingerprint"],
                    "parity_passed": True, "residuals": runtime.routing.transition_parity_results,
                })
            network0, trial_cluster, pairs = _realized_trial_network(current, proposal, candidates, da_phys, cluster)
            rate_by_id = {item.candidate_id: item for item in rates(current, context)}
            if any(rate_by_id[cid].signed_J_J_per_m2 <= 0.0 for cid in proposal.member_candidate_ids):
                return TopologyTrialResult(False, current, proposal.action_id, 0.0, 0.0, 0.0, "nonpositive_signed_directional_J")
            arms = []
            for candidate, arm in pairs:
                drive = rate_by_id[candidate.candidate_id]
                _, _, barrier = engine.lambda_cleave(engine.sigma_tip(drive.K_directional_Pa_sqrt_m / math.sqrt(candidate.gamma_rel)), float(args.temperatures[0]))
                resistance = hazard_resistance_J_per_m2(
                    barrier_J=barrier, cooperative_hits=float(engine.f.m_hits),
                    burgers_vector_m=float(engine.b), gamma_relative=candidate.gamma_rel,
                )
                arms.append(replace(arm, hazard_dissipation_J_per_m=resistance * arm.event_reward_m))

            def geometry(trial_state, trial_arms):
                realized = network0
                for item in trial_arms:
                    realized = extend_network_arm(realized, item)
                trial_state = replace(
                    trial_state, crack_network=realized,
                    junction_process_state={
                        "crack_representation": "sharp_wake_causal_v11",
                        **({"cluster": trial_cluster} if trial_cluster else {}),
                    },
                )
                return apply_causal_sharp_wake_trial_geometry(trial_state, trial_arms)

            def equilibrate(trial_state):
                nonlocal runtime
                request = _request(trial_state, candidates, args=args, cfg=cfg, runtime_step=step, cluster=trial_cluster)
                runtime, live = runtime.evaluate_trial(request)
                trial_requests[proposal.action_id] = request; trial_clusters[proposal.action_id] = trial_cluster
                trial_live_results[proposal.action_id] = live
                trial_drives[proposal.action_id] = rate_by_id
                return replace(
                    trial_state,
                    displacement=np.asarray(live["base_equilibrium"]["displacement"]),
                    stored_energy_J_per_m=float(live["base_equilibrium"]["recoverable_potential_energy_J_per_m"]),
                )

            return execute_topology_trial(
                current, proposal, tuple(arms), apply_trial_geometry=geometry,
                equilibrate_fixed_load=equilibrate, network_geometry_already_realized=True,
            )

        shared_info = {}
        def update_shared(selected_state, _context, proposal):
            nonlocal runtime, cluster, shared_info, engine
            from .crystal import near_tip_stress_tensor
            rate_map = (
                trial_drives[proposal.action_id] if proposal is not None
                else {item.candidate_id: item for item in latest_interval_rates}
            )
            K = max((item.K_directional_Pa_sqrt_m for item in rate_map.values()), default=0.0)
            # Populate the installed v10.2.28 tensor-resolved emission observer
            # from this accepted 2-D FEM state before evolving the one shared MPZ.
            probe_tip = selected_state.crack_network.branch(
                selected_state.crack_network.active_tip_ids[0]
            ).tip
            near_tip_stress_tensor(
                latest_sigma, selected_state.mesh, np.asarray(probe_tip),
                3.0 * max(float(getattr(selected_state.mesh, "hbar_tip", 0.0) or selected_state.mesh.hbar), 1.0e-12),
            )
            engine_trial = copy.deepcopy(engine)
            pre_progress = max(
                (hazard.residual_action for hazard in state.competition.hazard_states),
                default=0.0,
            )
            engine_trial.B = pre_progress
            if hasattr(engine_trial, "hazard_action_current"):
                engine_trial.hazard_action_current = pre_progress
            if hasattr(engine_trial, "hazard_threshold_action"):
                engine_trial.hazard_threshold_action = 1.0
            info = engine_trial.step(K, float(args.temperatures[0]), _context.duration_s)
            expected = proposal is not None
            if bool(info.get("fired")) != expected or int(info.get("n_fire", 0)) > 1:
                raise DirectionalStepRefinementRequired(
                    max(float(info.get("physical_hazard_action_step", info.get("dB", 0.0))), 1.0e-300),
                    max(float(getattr(args, "adaptive_event_target", 0.15) or 0.15) * 0.5, 1.0e-6),
                )
            engine = engine_trial
            counters = dict(selected_state.event_counters)
            counters["shared_state_updates"] = counters.get("shared_state_updates", 0) + 1
            counters["accepted_steps"] = step
            ledgers = dict(selected_state.energy_ledgers)
            ledgers["retained"] = float(info.get("N_em", 0.0)); ledgers["emission_work"] = float(info.get("W_emit", engine.W_emit))
            shared_info = info
            if proposal is not None:
                runtime = runtime.accept_trial(
                    trial_requests[proposal.action_id],
                    trial_live_results[proposal.action_id],
                )
                cluster = trial_clusters[proposal.action_id]
            return replace(
                selected_state, event_counters=counters, energy_ledgers=ledgers,
                tip_process_state={
                    "shared_engine_model": type(engine).__name__, "B": float(engine.B),
                    "N_em": float(engine.N_em), "W_emit": float(engine.W_emit), "time_s": float(engine.t),
                },
            )

        while True:
            context = replace(context, duration_s=float(args.dt) * trial_fraction)
            try:
                result = advance_accepted_step(
                    state, context, correlation_interval_s=float(engine.f.tau_c),
                    solve_accepted=solve_accepted, evaluate_directional_rates=rates,
                    trial_action=trial_action,
                    update_shared_state_once=update_shared,
                    maximum_directional_action_increment=float(getattr(args, "adaptive_event_target", 0.15) or 0.15),
                )
                break
            except DirectionalStepRefinementRequired as exc:
                shrink = 0.7 * exc.target_increment / max(exc.predicted_increment, 1.0e-300)
                next_fraction = max(float(getattr(args, "adaptive_min_frac", 1.0e-8)), trial_fraction * min(0.5, shrink))
                if next_fraction >= trial_fraction or next_fraction <= float(getattr(args, "adaptive_min_frac", 1.0e-8)):
                    raise RuntimeError("v11 directional adaptive stepping reached its minimum fraction") from exc
                trial_fraction = next_fraction
        state = result.state
        accepted_load += float(args.dU) * trial_fraction
        physical_time += context.duration_s
        selected = next((item for item in result.trials if item.selected), None)
        if selected is not None:
            adaptation_required = True
        for item in result.trials:
            rate_map = trial_drives.get(item.proposal.action_id, {r.candidate_id: r for r in result.rates})
            rec = {field: None for field in TRIAL_FIELDS}
            post_energy = item.result.state.stored_energy_J_per_m
            pre_energy = post_energy + item.result.energy_release_J_per_m
            arm_count = len(item.proposal.member_candidate_ids)
            participating = []
            request_for_trial = trial_requests.get(item.proposal.action_id)
            if request_for_trial is not None:
                participating = list(request_for_trial.crack_network.active_tip_ids)
            before_equilibrium = (latest_live_result or {}).get("base_equilibrium", {})
            after_live = trial_live_results.get(item.proposal.action_id)
            after_equilibrium = {} if after_live is None else after_live.get("base_equilibrium", {})
            reaction_before = before_equilibrium.get("reaction_force")
            reaction_after = after_equilibrium.get("reaction_force")

            def apparent_compliance(reaction):
                if reaction is None or not math.isfinite(float(reaction)) or abs(float(reaction)) <= 1.0e-300:
                    return None
                return float(accepted_load) / abs(float(reaction))

            rec.update({
                "step": step, "physical_time_s": physical_time, "accepted_state_id": context.accepted_state_id,
                "trial_id": item.proposal.action_id, "action_type": item.proposal.action_type,
                "participating_front_ids": participating,
                "candidate_ids": list(item.proposal.member_candidate_ids), "pending_event_ids": list(item.proposal.member_event_ids),
                "completion_times_s": list(item.proposal.completion_times_s),
                "correlation_time_difference_s": max(item.proposal.completion_times_s) - min(item.proposal.completion_times_s),
                "signed_directional_J_J_per_m2": [rate_map[c].signed_J_J_per_m2 for c in item.proposal.member_candidate_ids],
                "positive_directional_J_J_per_m2": [rate_map[c].positive_J_J_per_m2 for c in item.proposal.member_candidate_ids],
                "directional_K_Pa_sqrt_m": [rate_map[c].K_directional_Pa_sqrt_m for c in item.proposal.member_candidate_ids],
                "applied_displacement_m": float(accepted_load),
                "reaction_force_before_N_per_m": reaction_before,
                "reaction_force_after_N_per_m": reaction_after,
                "apparent_compliance_before_m2_per_N": apparent_compliance(reaction_before),
                "apparent_compliance_after_m2_per_N": apparent_compliance(reaction_after),
                "topology_fingerprint_before": (latest_live_result or {}).get("topology_fingerprint"),
                "topology_fingerprint_after": None if after_live is None else after_live.get("topology_fingerprint"),
                "proposed_arm_lengths_m": [da_phys] * arm_count,
                "realized_arm_lengths_m": [da_phys] * arm_count if item.result.accepted else [0.0] * arm_count,
                "pretrial_potential_energy_J_per_m": pre_energy,
                "posttrial_potential_energy_J_per_m": post_energy,
                "released_energy_J_per_m": item.result.energy_release_J_per_m,
                "hazard_derived_cost_per_arm_J_per_m": [
                    item.result.hazard_dissipation_J_per_m / arm_count
                ] * arm_count,
                "total_dissipative_cost_J_per_m": item.result.hazard_dissipation_J_per_m,
                "net_energy_margin_J_per_m": item.result.energy_margin_J_per_m,
                "relative_energy_residual": item.result.energy_margin_J_per_m / max(
                    abs(pre_energy), abs(post_energy), item.result.hazard_dissipation_J_per_m, 1.0e-300
                ),
                "geometry_status": "realized" if item.result.accepted else "rolled_back",
                "equilibrium_status": "converged" if item.result.accepted else "trial_vetoed",
                "provider_identity": runtime.routing.active_mechanics_provider,
                "accepted": item.selected, "veto_reason": None if item.selected else item.result.rejection_reason,
                "reservation_result": "accepted" if item.selected else "released",
                "consumption_result": "consumed" if item.selected else "preserved",
                "pretrial_state_hash": context.accepted_state_id,
                "postrollback_state_hash": (
                    None if item.result.accepted else context.accepted_state_id
                ),
                "trial_copy_bytes": item.result.trial_copy_bytes,
                "trial_copy_wall_time_s": item.result.trial_copy_wall_time_s,
            })
            writer.append_trial(rec)
        if selected is not None and selected.proposal.action_type == "two_arm" and cluster is not None:
            arm_branches = tuple(state.crack_network.branch(item) for item in cluster.arm_branch_ids)
            event = {field: None for field in BRANCH_EVENT_FIELDS}
            event.update({
                "event_record_id": selected.proposal.action_id, "step": step,
                "branch_junction": list(cluster.junction_xy_m), "parent_front": cluster.parent_branch_id,
                "arm_front_ids": list(cluster.arm_branch_ids),
                "arm_directions": [branch.current_orientation_rad for branch in arm_branches],
                "plane_identities": list(selected.proposal.member_candidate_ids),
                "event_ids_consumed": list(selected.proposal.member_event_ids),
                "completion_time_difference_s": max(selected.proposal.completion_times_s) - min(selected.proposal.completion_times_s),
                "arm_lengths_m": [sum(math.dist(a, b) for a, b in zip(branch.path, branch.path[1:])) for branch in arm_branches],
                "tip_positions_m": [list(branch.tip) for branch in arm_branches],
                "tip_separation_m": math.dist(arm_branches[0].tip, arm_branches[1].tip),
                "shared_cluster_id": cluster.cluster_id,
                "shared_cluster_state_hash": _hash((engine.B, engine.N_em, engine.W_emit)),
                "topology_fingerprint": runtime.routing.topology_fingerprint,
                "released_energy_J_per_m": selected.result.energy_release_J_per_m,
                "total_cost_J_per_m": selected.result.hazard_dissipation_J_per_m,
                "energy_margin_J_per_m": selected.result.energy_margin_J_per_m,
            })
            writer.branch_event(event)
            counters = dict(state.event_counters)
            counters["branch_birth_count"] = counters.get("branch_birth_count", 0) + 1
            counters["latest_successful_action"] = selected.proposal.action_id
            state = replace(state, event_counters=counters)
            write_topology_snapshot(
                out, state, step=step, reason="branch_birth",
                physical_extension_m=state.crack_network.total_physical_crack_length_m - cfg.geometry.a0,
                branch_birth_count=counters["branch_birth_count"],
                latest_action=selected.proposal.action_id,
                mechanics={
                    "energy_summary": {
                        "released_J_per_m": selected.result.energy_release_J_per_m,
                        "cost_J_per_m": selected.result.hazard_dissipation_J_per_m,
                        "margin_J_per_m": selected.result.energy_margin_J_per_m,
                    }
                },
            )
        writer.energy({
            "step": step, "accepted_state_id": context.accepted_state_id,
            "stored_energy_J_per_m": state.stored_energy_J_per_m,
            "released_energy_J_per_m": selected.result.energy_release_J_per_m if selected else 0.0,
            "dissipative_cost_J_per_m": selected.result.hazard_dissipation_J_per_m if selected else 0.0,
            "residual_J_per_m": selected.result.energy_margin_J_per_m if selected else 0.0,
        })
        for branch in state.crack_network.branches:
            writer.front({
                "step": step, "front_id": branch.branch_id, "parent_front_id": branch.parent_branch_id,
                "status": branch.status, "termination_reason": branch.local_state.get("termination_reason"),
                "tip_x_m": branch.tip[0], "tip_y_m": branch.tip[1],
                "arclength_m": sum(math.dist(a, b) for a, b in zip(branch.path, branch.path[1:])),
            })
        guard = None
        if cluster is not None:
            accepted_live = (
                trial_live_results.get(selected.proposal.action_id)
                if selected is not None else latest_live_result
            )
            live_tips = tuple((accepted_live or {}).get("tips", ()))
            independently_valid = tuple(
                any(
                    np.allclose(item.get("tip_xy_m", ()), state.crack_network.branch(branch_id).tip)
                    and bool(item.get("directional"))
                    and all(bool(direction.get("local_contour_valid", False)) for direction in item["directional"])
                    for item in live_tips
                )
                for branch_id in cluster.arm_branch_ids
            )
            guard = evaluate_unresolved_cluster_guard(
                state.crack_network, cluster, process_zone_length_m=float(args.L_pz),
                local_J_contour_radius_m=float(getattr(args, "rJ", None) or args.L_pz),
                independently_valid_local_J=independently_valid,
            )
            writer.cluster({
                "step": step, "cluster_id": cluster.cluster_id,
                "state_hash": _hash((cluster, state.tip_process_state)),
                "unresolved": cluster.unresolved,
                "tip_separation_m": guard.tip_separation_m,
                "handoff_required": guard.handoff_required,
            })
        checkpoint = ProductionBranchCheckpoint(
            state=state, shared_process_state=_capture_shared_engine(engine),
            physical_time_s=physical_time, accepted_load=accepted_load,
            mesh_identity=_mesh_identity(state.mesh), boundary_condition_state={"opening_m": accepted_load},
            provider_runtime=runtime, provider_cache_identity=str(cache_root.resolve()),
            topology_fingerprint=runtime.routing.topology_fingerprint or _hash(state.crack_network),
            front_competitions={tip: state.competition for tip in state.crack_network.active_tip_ids},
            branch_clusters=(() if cluster is None else (cluster,)),
            projected_extension_m=max(branch.tip[0] for branch in state.crack_network.branches) - cfg.geometry.a0,
            physical_extension_m=state.crack_network.total_physical_crack_length_m - cfg.geometry.a0,
            handoff_guard_diagnostics={} if guard is None else guard.to_dict(),
            termination_reason=None,
        )
        checkpoint_path = out / "checkpoint" / "latest.json"
        write_branch_checkpoint(checkpoint, checkpoint_path)
        if guard is not None and guard.handoff_required:
            from .resolved_production_v11 import continue_resolved_production
            return continue_resolved_production(
                args=args, cfg=cfg, state=state, shared_engine=engine,
                runtime=runtime, cluster=cluster, candidates=candidates,
                writer=writer, out=out, cache_root=cache_root,
                physical_time=physical_time, accepted_load=accepted_load,
                start_step=step + 1, engine_factory=lambda: base.build_engine(args, mat),
            )
        extension = state.crack_network.total_physical_crack_length_m - cfg.geometry.a0
        target = float(getattr(args, "target_crack_extension_um", float("inf"))) * 1e-6
        if extension >= target:
            termination = "target_reached"; break

    termination = termination or "physical_veto_no_branch"
    checkpoint = replace(checkpoint, termination_reason=termination)
    write_branch_checkpoint(checkpoint, out / "checkpoint" / "latest.json")
    final_growth = crack_growth_metrics(
        state.crack_network, initial_crack_length_m=cfg.geometry.a0,
    )
    write_topology_snapshot(
        out, state, step=int(state.event_counters.get("accepted_steps", 0)),
        reason=termination, physical_extension_m=checkpoint.physical_extension_m,
        branch_birth_count=int(state.event_counters.get("branch_birth_count", 0)),
        latest_action=state.event_counters.get("latest_successful_action"),
        growth_metrics=final_growth.to_dict_um(), final=True,
    )
    writer.complete(status=termination, final_checkpoint="checkpoint/latest.json", validation={
        "checkpoint": True, "shared_state_updates": state.event_counters.get("shared_state_updates", 0),
        "topology_actions": state.event_counters.get("topology_actions", 0),
        "provider_transition": runtime.routing.transition_step is not None,
        "live_fem_solve_count": runtime.live_fem_solve_count,
        "accepted_provider_state_count": runtime.accepted_provider_state_count,
    })
    return 0


def _remove_flag_with_value(args, name):
    result = []; index = 0
    while index < len(args):
        token = args[index]
        if token == name:
            index += 2; continue
        if token.startswith(name + "="):
            index += 1; continue
        result.append(token); index += 1
    return result


def main(argv=None, *, audit_already_written=False):
    from . import sharp_front as base
    from . import sharp_front_v10_2_28_audited as entry
    from . import sharp_front_v10_1_7_3 as avalanche_entry
    args = list(sys.argv[1:] if argv is None else argv)
    restart = None
    for index, token in enumerate(args):
        if token.startswith("--v11-restart-checkpoint="):
            restart = token.split("=", 1)[1]
        elif token == "--v11-restart-checkpoint" and index + 1 < len(args):
            restart = args[index + 1]
    for flag in ("--mechanistic-branching", "--audit-only"):
        args = [item for item in args if item != flag]
    args = _remove_flag_with_value(args, "--maximum-fronts")
    args = _remove_flag_with_value(args, "--hazard-seed")
    args = _remove_flag_with_value(args, "--v11-restart-checkpoint")
    args.extend(["--max-fronts", "1"])
    original = base.run_2d
    original_geometry_diagnostics = avalanche_entry._write_geometry_diagnostics
    old_restart = os.environ.get("V11_BRANCH_RESTART_CHECKPOINT")
    if restart:
        os.environ["V11_BRANCH_RESTART_CHECKPOINT"] = restart
    base.run_2d = run_2d
    # The v10 wrapper's post-run report is specific to its stochastic-avalanche
    # geometry backend, which this audited adapter intentionally never builds.
    avalanche_entry._write_geometry_diagnostics = lambda _args: None
    try:
        return entry.main(args)
    finally:
        base.run_2d = original
        avalanche_entry._write_geometry_diagnostics = original_geometry_diagnostics
        if old_restart is None:
            os.environ.pop("V11_BRANCH_RESTART_CHECKPOINT", None)
        else:
            os.environ["V11_BRANCH_RESTART_CHECKPOINT"] = old_restart


if __name__ == "__main__": main()


__all__ = ["MODEL_ID", "main", "run_2d"]
