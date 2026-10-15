#!/usr/bin/env python3
"""Execute the compact R8 current-backend/full-spatial qualification."""
from __future__ import annotations

from dataclasses import replace
import argparse
import csv
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_off_axis_prescribed_void_benchmark_v2 import initial_state

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.current_coupled_execution_r4 import complete_state
from arrhenius_fracture.current_front_process_adapter_r8 import CurrentFrontProcessAdapter
from arrhenius_fracture.front_constitutive_strategy_r8 import CURRENT_STRATEGY
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_event_aware_coupled_interval, advance_global_physical_interval,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_ledgers, capture_stochastic_state, serialize_active_state,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import complete_row_bundle, material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, _refine_state_around_graph, build_production_void_state,
    crack_tip_tensor, restore_sharp_front_engine,
)


SCHEMA = "r8.current-backend-full-spatial-integration/1"
K_EVENT = 51.1004675e6
T_K = 900.0
EVENT_EXPOSURE_S = 1.0e-6
POST_EVENT_EXPOSURE_S = 0.25e-6
TURN_DEG = 6.974492530416398
ACTIVE_BUNDLE = material_bundle("DBTT")
ACTIVE_BINDING = {"mode": "PINNED_CLASS_LABEL", "candidate_id": ACTIVE_BUNDLE.fracture_material_row_id}


def finite(value):
    if isinstance(value, Mapping):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_record(state, initial=None):
    owner = state.tip_process_state["owner_by_front"][ROOT_BRANCH_ID]
    engine = restore_sharp_front_engine(
        state.material, _bundle_from_state(state),
        state.tip_process_state["by_branch"][owner])
    mpz = engine.mpz
    cavities = () if state.void_state is None else state.void_state.cavities
    path = state.crack_network.branch(ROOT_BRANCH_ID).path
    path_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    plastic_measure = float(np.sum(
        np.asarray(state.mesh.area_e)*np.linalg.norm(np.asarray(state.ep_gp), axis=0)))
    density_measure = float(np.sum(
        np.asarray(state.mesh.area_e)*np.asarray(state.rho_gp)))
    initial_plastic_measure = (None if initial is None else float(np.sum(
        np.asarray(initial.mesh.area_e)
        *np.linalg.norm(np.asarray(initial.ep_gp), axis=0))))
    initial_density_measure = (None if initial is None else float(np.sum(
        np.asarray(initial.mesh.area_e)*np.asarray(initial.rho_gp))))
    return {
        "accepted_fingerprint": complete_accepted_state_fingerprint(state),
        "geometry_generation": state.crack_network.geometry_generation,
        "tip_position_xy_m": list(path[-1]),
        "resolved_crack_path_length_m": path_length,
        "projected_crack_extension_m": path[-1][0]-path[0][0],
        "current_engine": {
            "strategy": getattr(engine, "_front_constitutive_strategy", None),
            "time_s": float(engine.t), "n_adv": int(engine.n_adv),
            "a_adv_m": float(engine.a_adv),
            "micro_advance_total_m": float(engine.micro_advance_total_m),
            "effective_radius_m": float(engine.r_eff()),
            "mobile": float(mpz.mobile_count),
            "retained": float(mpz.retained_count),
            "gross_source": float(mpz.cumulative_gross_source_activity),
            "gross_return": float(mpz.cumulative_gross_return_activity),
            "permanent_slip": float(mpz.local_slip_count()),
            "returned_mobile": float(np.sum(mpz.cumulative_returned_mobile)),
            "physical_returned_mobile": float(np.sum(
                mpz.cumulative_physical_returned_mobile)),
            "hazard_action": float(engine.hazard_action_current),
            "hazard_threshold": float(engine.hazard_threshold_action),
            "hazard_event_index": int(engine.hazard_event_index),
            "signed_tau_Pa": np.asarray(
                mpz._anisotropic_tau_signed_Pa, float).tolist(),
            "drive_factors": np.asarray(
                mpz._anisotropic_drive_factors, float).tolist(),
        },
        "bulk": {
            "maximum_abs_plastic_strain": float(np.max(np.abs(state.ep_gp))),
            "minimum_density_m2": float(np.min(state.rho_gp)),
            "maximum_density_m2": float(np.max(state.rho_gp)),
            "area_integrated_plastic_measure_m2": plastic_measure,
            "area_integrated_density": density_measure,
            "remap_compatible_plastic_measure_increment_m2": (
                None if initial is None else plastic_measure-initial_plastic_measure),
            "remap_compatible_density_measure_increment": (
                None if initial is None else density_measure-initial_density_measure),
        },
        "void": {
            "cavity_count": len(cavities),
            "radii_m": [float(item.radius_m) for item in cavities],
            "phases": [item.phase.value for item in cavities],
            "length_ledgers": ({} if state.void_state is None else
                               dict(state.void_state.length_ledgers)),
        },
        "vacancy_ledgers": dict(state.vacancy_transport_state_v2.ledgers),
        "directional_clocks": [{
            "candidate_id": item.candidate_id,
            "action": float(item.action),
            "threshold": float(item.current_threshold_action),
            "completed": int(item.completed_event_count),
            "pending_event_ids": [event.event_id for event in item.pending_events],
        } for item in state.competition.hazard_states],
    }


def _observation(actual=True, reference_tensor=None):
    def provider(state):
        if actual:
            tensor, element_ids = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
            identity = {"mode": "ACTIVE_ACTUAL_CURRENT_FEM", "element_ids": list(element_ids)}
        else:
            tensor = np.asarray(reference_tensor, float)
            identity = {
                "mode": "PASSIVE_DISABLE_CAVITY_TO_TIP_MECHANICAL_FEEDBACK",
                "retained_owners": ["bulk", "void", "transport", "current_tip"],
            }
        return tensor, K_EVENT, identity, None
    return provider


def _run_segment(state, duration, provider):
    return advance_event_aware_coupled_interval(
        state, duration, temperature_K=T_K,
        directional_observation_provider=provider,
    )


def _event_rows(audit):
    result = []
    scheduler = audit.get("event_scheduler") or audit.get("scheduler")
    if scheduler:
        result.extend(scheduler.get("events", ()))
    remainder = audit.get("remainder")
    if remainder:
        result.extend(_event_rows(remainder))
    return result


def matched_arms(destination: Path):
    no_initial = initial_state(
        "no_void", load_scale=75.0,
        front_constitutive_strategy=CURRENT_STRATEGY,
        material_bundle_override=ACTIVE_BUNDLE)
    active_initial = initial_state(
        "above", load_scale=75.0,
        front_constitutive_strategy=CURRENT_STRATEGY,
        material_bundle_override=ACTIVE_BUNDLE)
    passive_initial = copy.deepcopy(active_initial)
    reference_tensor, _ = crack_tip_tensor(no_initial, branch_id=ROOT_BRANCH_ID)
    definitions = {
        "no_void": (no_initial, _observation(actual=True)),
        "passive": (passive_initial, _observation(
            actual=False, reference_tensor=reference_tensor)),
        "active": (active_initial, _observation(actual=True)),
    }
    results = {}
    boundary_states = {}
    for name, (initial, provider) in definitions.items():
        boundary, event_audit = _run_segment(initial, EVENT_EXPOSURE_S, provider)
        final, continuation_audit = _run_segment(
            boundary, POST_EVENT_EXPOSURE_S, provider)
        boundary_states[name] = boundary
        results[name] = {
            "initial": state_record(initial),
            "event_boundary": state_record(boundary, initial),
            "final": state_record(final, initial),
            "event_audit": finite(event_audit),
            "continuation_audit": finite(continuation_audit),
            "selected_events": [row["selected_event"]
                                for row in _event_rows(event_audit)
                                if row["selected_event"] is not None],
            "common_external_schedule": [
                {"K_Pa_sqrt_m": K_EVENT, "temperature_K": T_K,
                 "duration_s": EVENT_EXPOSURE_S},
                {"K_Pa_sqrt_m": K_EVENT, "temperature_K": T_K,
                 "duration_s": POST_EVENT_EXPOSURE_S},
            ],
        }
        (destination / f"matched_{name}.json").write_text(
            json.dumps(finite(results[name]), indent=2, sort_keys=True)+"\n")

    checkpoint_root = destination / "active_interrupted_checkpoint"
    checkpoint_root.mkdir()
    checkpoint_path = checkpoint_root / "accepted_state.json"
    write_checkpoint(boundary_states["active"], checkpoint_path, compression="gzip")
    restarted = restore_checkpoint(checkpoint_path)
    uninterrupted, uninterrupted_audit = _run_segment(
        boundary_states["active"], POST_EVENT_EXPOSURE_S,
        definitions["active"][1])
    restarted_final, restarted_audit = _run_segment(
        restarted, POST_EVENT_EXPOSURE_S, definitions["active"][1])
    files = [{"path": str(path.relative_to(destination)), "sha256": sha256(path)}
             for path in sorted(checkpoint_root.rglob("*")) if path.is_file()]
    restart = {
        "files": files,
        "owning_state_reconstructed_from_files_only": True,
        "uninterrupted_fingerprint": complete_accepted_state_fingerprint(uninterrupted),
        "restarted_fingerprint": complete_accepted_state_fingerprint(restarted_final),
        "exact": complete_accepted_state_fingerprint(uninterrupted)
                 == complete_accepted_state_fingerprint(restarted_final),
        "uninterrupted_audit": finite(uninterrupted_audit),
        "restarted_audit": finite(restarted_audit),
    }
    return results, restart


def _loaded_turning_state():
    state, _ = build_production_void_state(
        bundle=ACTIVE_BUNDLE, enabled=False,
        cleavage_theta_deg=TURN_DEG,
        front_constitutive_strategy=CURRENT_STRATEGY)
    state = _refine_state_around_graph(state, 3)
    displacement = np.asarray(state.displacement).copy()
    displacement[2*np.asarray(state.boundary.top_nodes, int)+1] = 15.0e-6
    displacement[2*np.asarray(state.boundary.bot_nodes, int)+1] = -15.0e-6
    displacement[2*int(state.boundary.left_bot)+1] = -15.0e-6
    return equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=displacement))


def geometry_correspondence():
    before = _loaded_turning_state()
    tensor_before, _ = crack_tip_tensor(before, branch_id=ROOT_BRANCH_ID)
    owner = before.tip_process_state["owner_by_front"][ROOT_BRANCH_ID]
    payload = before.tip_process_state["by_branch"][owner]
    direct = CurrentFrontProcessAdapter(
        before.material, _bundle_from_state(before)).advance_front_process(
            payload, {"K_effective_Pa_sqrt_m": K_EVENT,
                      "stress_tensor_Pa": tensor_before.tolist(),
                      "tangent_xy": [1.0, 0.0]}, T_K, EVENT_EXPOSURE_S)
    spatial, audit = advance_global_physical_interval(
        before, before, EVENT_EXPOSURE_S,
        {"kind": "hold", "temperature_K": T_K,
         "stress_tensor_Pa": tensor_before,
         "K_effective_by_front": {ROOT_BRANCH_ID: K_EVENT},
         "maximum_events": 4})
    accepted_payload = spatial.tip_process_state["by_branch"][owner]
    direct_engine = restore_sharp_front_engine(
        before.material, _bundle_from_state(before), direct.advanced_front_state)
    spatial_engine = restore_sharp_front_engine(
        spatial.material, _bundle_from_state(spatial), accepted_payload)
    direct_active = serialize_active_state(direct_engine)
    spatial_active = serialize_active_state(spatial_engine)
    active_scale = np.maximum.reduce((
        np.abs(direct_active.vector), np.abs(spatial_active.vector),
        np.full_like(direct_active.vector, 1.0),
    ))
    active_relative_error = float(np.max(
        np.abs(direct_active.vector-spatial_active.vector)/active_scale))
    direct_ledgers = capture_ledgers(direct_engine)
    spatial_ledgers = capture_ledgers(spatial_engine)
    ledger_relative_error = max((
        abs(direct_ledgers.get(key, 0.0)-spatial_ledgers.get(key, 0.0))
        /max(abs(direct_ledgers.get(key, 0.0)),
             abs(spatial_ledgers.get(key, 0.0)), 1.0)
        for key in set(direct_ledgers) | set(spatial_ledgers)
    ), default=0.0)
    stochastic_equal = (
        capture_stochastic_state(direct_engine)
        == capture_stochastic_state(spatial_engine))
    same_contract = bool(
        direct_active.geometry_signature == spatial_active.geometry_signature
        and active_relative_error <= 1.0e-12
        and ledger_relative_error <= 1.0e-12
        and stochastic_equal)
    path = spatial.crack_network.branch(ROOT_BRANCH_ID).path
    tangent = np.asarray(path[-1])-np.asarray(path[-2])
    tangent /= np.linalg.norm(tangent)
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    tensor_after, _ = crack_tip_tensor(spatial, branch_id=ROOT_BRANCH_ID)
    spatial_next = CurrentFrontProcessAdapter(
        spatial.material, _bundle_from_state(spatial)).advance_front_process(
            accepted_payload,
            {"K_effective_Pa_sqrt_m": K_EVENT,
             "stress_tensor_Pa": tensor_after.tolist(),
             "tangent_xy": tangent.tolist()}, T_K, POST_EVENT_EXPOSURE_S)
    autonomous_next = CurrentFrontProcessAdapter(
        spatial.material, _bundle_from_state(spatial)).advance_front_process(
            accepted_payload,
            {"K_effective_Pa_sqrt_m": K_EVENT,
             "stress_tensor_Pa": tensor_before.tolist(),
             "tangent_xy": [1.0, 0.0]}, T_K, POST_EVENT_EXPOSURE_S)
    spatial_next_engine = restore_sharp_front_engine(
        spatial.material, _bundle_from_state(spatial),
        spatial_next.advanced_front_state)
    autonomous_engine = restore_sharp_front_engine(
        spatial.material, _bundle_from_state(spatial),
        autonomous_next.advanced_front_state)
    spatial_population = (spatial_next_engine.mpz.mobile_count
                          +spatial_next_engine.mpz.retained_count)
    autonomous_population = (autonomous_engine.mpz.mobile_count
                             +autonomous_engine.mpz.retained_count)
    event_row = audit["events"][0]
    return {
        "common_pre_event_fingerprint": complete_accepted_state_fingerprint(before),
        "bitwise_native_numerical_state_after_event": (
            complete_state(direct_engine) == complete_state(spatial_engine)),
        "same_native_contract_after_event_within_tolerance": same_contract,
        "native_active_state_maximum_relative_error": active_relative_error,
        "native_ledger_maximum_relative_error": ledger_relative_error,
        "native_stochastic_state_equal": stochastic_equal,
        "correspondence_tolerance": 1.0e-12,
        "direct_native_geometry_events": finite(direct.geometry_events),
        "spatial_selected_event": event_row["selected_event"],
        "operation_sequence": [
            "signed_FEM_drive", "native_source_transport_and_return",
            "native_continuous_micro_translation", "native_mark_and_reward",
            "outer_geometry_admission", "field_remap", "fixed_load_reequilibrium",
            "next_signed_source_evaluation",
        ],
        "accepted_position_xy_m": list(path[-1]),
        "accepted_tangent_xy": tangent.tolist(),
        "accepted_tangent_deg": angle,
        "target_tangent_deg": TURN_DEG,
        "tangent_error_deg": angle-TURN_DEG,
        "material_frame": finite(accepted_payload["material_frame"]),
        "projected_extension_m": path[-1][0]-path[-2][0],
        "path_extension_m": math.dist(path[-1], path[-2]),
        "native_micro_advance_m": direct.audit["continuous_micro_advance_m"],
        "effective_radius_m": float(spatial_engine.r_eff()),
        "resolved_geometry_generation": spatial.crack_network.geometry_generation,
        "next_actual_signed_tau_Pa": spatial_next.audit["signed_drive"]["tau_signed_Pa"],
        "next_autonomous_signed_tau_Pa": autonomous_next.audit["signed_drive"]["tau_signed_Pa"],
        "autonomous_population_relative_error": (
            (autonomous_population-spatial_population)
            / max(abs(spatial_population), 1.0)),
        "autonomous_radius_relative_error": (
            (autonomous_engine.r_eff()-spatial_next_engine.r_eff())
            / max(abs(spatial_next_engine.r_eff()), 1.0e-30)),
        "material_fixed_wake_origin_preserved": (
            accepted_payload["material_frame"]["_spatial_material_wake_origin_xy_m"]
            != accepted_payload["material_frame"]["_spatial_tip_position_xy_m"]),
    }


def main():
    global ACTIVE_BUNDLE, ACTIVE_BINDING
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--offspring-registry", type=Path)
    parser.add_argument("--candidate-id")
    args = parser.parse_args()
    if bool(args.offspring_registry) != bool(args.candidate_id):
        parser.error("--offspring-registry and --candidate-id must be supplied together")
    if args.offspring_registry:
        source = args.offspring_registry.resolve()
        with source.open(newline="") as stream:
            matches = [row for row in csv.DictReader(stream)
                       if row["candidate_id"] == args.candidate_id]
        if len(matches) != 1:
            raise KeyError(f"expected one complete row {args.candidate_id!r}")
        record = matches[0]
        if record["family"] != "A_ORIGINAL_SINGLE_EXP_FLOOR":
            raise ValueError("prospective spatial binding accepts strict-family rows only")
        complete = json.loads(record["complete_parameter_vector"])
        row_hash = hashlib.sha256(json.dumps(
            complete, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        if row_hash != record["complete_row_sha256"]:
            raise ValueError("complete candidate row hash mismatch")
        ACTIVE_BUNDLE = complete_row_bundle(complete)
        ACTIVE_BINDING = {
            "mode": "EXACT_COMPLETE_ROW", "candidate_id": args.candidate_id,
            "family": record["family"], "complete_row_sha256": row_hash,
            "source": {"path": str(source), "sha256": sha256(source)},
            "no_refit": True,
        }
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    arms, restart = matched_arms(destination)
    geometry = geometry_correspondence()
    (destination / "geometry_correspondence.json").write_text(
        json.dumps(finite(geometry), indent=2, sort_keys=True)+"\n")
    result = {
        "schema": SCHEMA,
        "declared_strategy": CURRENT_STRATEGY,
        "material_family": ACTIVE_BUNDLE.material_class,
        "material_binding": ACTIVE_BINDING,
        "event_law": "native current persistent/reversible deterministic mark/reward",
        "no_refit_or_multiplier": True,
        "matched_arms": arms,
        "files_only_restart": restart,
        "geometry_correspondence": geometry,
        "passed": bool(
            all(row["selected_events"] == ["DIRECTIONAL_CLEAVAGE"]
                for row in arms.values())
            and restart["exact"]
            and geometry["same_native_contract_after_event_within_tolerance"]
            and abs(geometry["tangent_error_deg"]) < 1.0e-10
            and geometry["material_fixed_wake_origin_preserved"]),
    }
    (destination / "r8_current_backend_spatial.json").write_text(
        json.dumps(finite(result), indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "passed": result["passed"],
        "arms": {name: row["selected_events"] for name, row in arms.items()},
        "restart_exact": restart["exact"],
        "turning_tangent_deg": geometry["accepted_tangent_deg"],
        "common_state_equal_within_tolerance": geometry[
            "same_native_contract_after_event_within_tolerance"],
        "output": str(destination),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
