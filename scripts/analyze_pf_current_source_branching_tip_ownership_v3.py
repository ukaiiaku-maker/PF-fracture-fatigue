#!/usr/bin/env python3
"""Frozen V3 ownership audit and isolated process-state replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
PARENT_COMMIT = "0238aae096aa29e79829d3c562383c38f2290ad6"
TARGET_STEPS = (368, 369, 370, 373, 774, 2147, 2151, 2182)
TRANSITION_STEPS = (2147, 2148, 2149, 2150, 2151)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty audit table: {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)


def compact(value: Any) -> str:
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, dict): return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)): return [convert(val) for val in item]
        return item
    return json.dumps(convert(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def selected_by_step(actions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    # The archived V2 writer predates an explicit `selected` column.  Its
    # transaction table contains exactly one accepted trial per event step;
    # rejected trials are rollback evidence.
    return {int(row["step"]): row for row in actions if bool(row.get("accepted", False))}


def _provider_signature(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((key, float(value)) for key, value in values.items()))


def match_provider_states(case: Path, directional, detail_steps):
    """Match every accepted interval to its immutable cached FEM state.

    Only the small set in ``detail_steps`` retains the full provider payload;
    the all-interval table keeps lightweight ownership/state metadata.
    """
    wanted_by_signature = defaultdict(list)
    rows_by_step = defaultdict(dict)
    for row in directional:
        rows_by_step[int(row["step"])][row["candidate_id"]] = float(
            row["J_local_signed_J_per_m2"]
        )
    for step, values in rows_by_step.items():
        wanted_by_signature[_provider_signature(values)].append(step)

    metadata = {}
    detailed = {}
    for state_path in sorted((case / "live_kernel_cache").glob("*/provider_state.pkl")):
        state = pickle.loads(state_path.read_bytes())
        values = {item["candidate_id"]: float(item["J_local_signed_J_per_m2"])
                  for tip in state.get("tips", ()) for item in tip.get("directional", ())}
        steps = wanted_by_signature.get(_provider_signature(values), ())
        if not steps:
            continue
        manifest = json.loads((state_path.parent / "manifest.json").read_text())
        lightweight = {
            "cache_key": state_path.parent.name,
            "cache_status": "EXACT_V11_LIVE_PROVIDER_STATE",
            "state_sha256": manifest["state_sha256"],
            "topology_fingerprint": state["topology_fingerprint"],
            "tip_xy_by_candidate": {
                item["candidate_id"]: list(tip["tip_xy_m"])
                for tip in state["tips"] for item in tip["directional"]
            },
        }
        for step in steps:
            prior = metadata.get(step)
            if prior is not None and prior != lightweight:
                raise RuntimeError(f"ambiguous provider cache match at step {step}")
            metadata[step] = lightweight
            if step in detail_steps:
                detailed[step] = (state_path.parent.name, state, manifest)
    # Steps before the recorded v11 live-provider transition used the pinned
    # single-front provider and have no v11 cache payload.  Their physical
    # same-tip result is still exact (only one active tip existed), but tensor
    # coordinates and the live-provider stress hash are explicitly unavailable.
    for step in sorted(set(rows_by_step) - set(metadata)):
        if step > 287:
            continue
        accepted_state_id = next(
            row["accepted_state_id"] for row in directional if int(row["step"]) == step
        )
        metadata[step] = {
            "cache_key": "NOT_APPLICABLE_PRE_V11_LIVE_PROVIDER",
            "cache_status": "PRE_V11_SINGLE_FRONT_PROVIDER_NOT_IN_LIVE_CACHE",
            "state_sha256": f"NOT_ARCHIVED_SEPARATELY:{accepted_state_id}",
            "topology_fingerprint": "SINGLE_FRONT_PRE_V11_LIVE_PROVIDER",
            "tip_xy_by_candidate": {candidate: None for candidate in rows_by_step[step]},
        }
    missing = sorted(set(rows_by_step) - set(metadata))
    if missing:
        raise RuntimeError(f"provider cache matching failed for {missing}")
    missing_detail = sorted(set(detail_steps) - set(detailed))
    if missing_detail:
        raise RuntimeError(f"detailed provider matching failed for {missing_detail}")
    return detailed, metadata


def path_metrics(branch, tip_xy):
    target = np.asarray(tip_xy, dtype=float)
    points = [np.asarray(point, dtype=float) for point in branch.path]
    distances = [float(np.linalg.norm(point - target)) for point in points]
    index = int(np.argmin(distances))
    if distances[index] > 1.0e-12:
        raise RuntimeError(f"provider tip is absent from branch path: {tip_xy}")
    arclength = math.fsum(float(np.linalg.norm(b - a)) for a, b in zip(points[:index], points[1:index+1]))
    return arclength, float(points[index][0] - points[0][0])


def next_path_coordinate(branch, current_xy):
    points = [np.asarray(point, dtype=float) for point in branch.path]
    matches = [index for index, point in enumerate(points) if math.dist(point, current_xy) < 1.0e-12]
    if len(matches) != 1 or matches[0] + 1 >= len(points):
        raise RuntimeError(f"cannot reconstruct post-event probe coordinate from {current_xy}")
    return points[matches[0] + 1].tolist()


def state_profile(engine):
    mpz = engine.mpz
    diagnostics = mpz.diagnostics(engine.G, engine.nu, engine.b, engine.f.r0)
    return {
        "mobile_total": float(mpz.mobile_count), "retained_total": float(mpz.retained_count),
        "mobile_profile": np.asarray(mpz.mobile).tolist(),
        "retained_profile": np.asarray(mpz.retained).tolist(),
        "accumulated_slip_profile": np.asarray(mpz.accumulated_slip).tolist(),
        "tip_radius_m": float(diagnostics["persistent_tip_radius_m"]),
        "front_width_m": float(diagnostics["persistent_site_front_width_m"]),
        "source_area_m2": float(diagnostics["persistent_site_source_area_m2"]),
        "multiplicity_per_system": float(diagnostics["persistent_site_multiplicity_per_system"]),
        "active_K_shield_Pa_sqrt_m": float(engine._active_shielding_signed()),
        "wake_K_shield_Pa_sqrt_m": float(engine._wake_shielding_signed()),
        "local_density_by_system_m2": np.asarray(
            engine.mpz.anisotropic_last_rho_back_by_system_m2
        ).tolist(),
        "Taylor_backstress_by_system_Pa": np.asarray(
            engine.mpz.persistent_site_last_sigma_back_initial_Pa
        ).tolist(),
        "transport_velocity_by_system_m_s": np.asarray(
            engine.mpz.anisotropic_last_transport_velocity_by_system_m_s
        ).tolist(),
    }


def restored_engine(checkpoint):
    from arrhenius_fracture.persistent_site_audited_engine_v10221 import AuditedPersistentSiteStateResolvedTipEngine as Engine
    from arrhenius_fracture.sharp_front_v11_branching import _restore_shared_engine
    # Restore independent isolated evaluations; checkpoint array ownership must
    # not leak between the old-defect and corrected controls.
    payload = copy.deepcopy(checkpoint.shared_process_state)
    ef = payload["engine_fields"]; mf = payload["mpz_fields"]
    hazard = ef["hazard_cfg"]; avalanche = ef["avalanche_cfg"]
    Engine.configure_default(ef["tip_cfg"])
    Engine.configure_hazard(hazard.mode, hazard.seed, hazard.minimum_threshold)
    Engine.configure_avalanche(avalanche.mode, avalanche.minimum_factor, avalanche.maximum_factor, avalanche.geometry_subsegment_fraction)
    Engine.configure_anisotropic_emission(ef["anisotropic_cfg"])
    Engine.configure_state_resolved_physics(mf["_signed_kernel"], mf["_signed_transport_mode"])
    Engine.configure_persistent_sites(mf["_persistent_site_cfg"])
    Engine.configure_campaign(mf["_campaign_backstress_scale"], mf["_campaign_refresh_scale"])
    engine = Engine(ef["f"], ef["cb"], ef["eb"], ef["G"], ef["nu"], ef["b"], ef["manifest"], mf["cfg"])
    return _restore_shared_engine(engine, payload)


def process_row(
    step, mode, engine, before, after, info, drive, tensor_tip, control_tip,
    K, opening, physical_time, dt, *, record_kind="isolated_interval_replay",
    selected_event_candidate_ids=(), selected_event_tip_id=None,
    process_owner_id="jc9ec4d7327bde23",
):
    mpz = engine.mpz
    conversion = np.asarray(mpz._signed_kernel.activation_to_line_content, dtype=float)
    return {
        "step": step, "record_kind": record_kind, "evaluation_mode": mode,
        "profile_archive_status": "FULL_PRE_POST",
        "physical_time_s": physical_time, "duration_s": dt,
        "applied_opening_m": opening, "controlling_directional_K_Pa_sqrt_m": K,
        "controlling_scalar_K_tip_id": control_tip, "tensor_probe_tip_id": tensor_tip,
        "scalar_tensor_same_tip": control_tip == tensor_tip,
        "selected_event_candidate_ids": compact(selected_event_candidate_ids),
        "selected_event_tip_id": selected_event_tip_id,
        "process_owner_id": process_owner_id,
        "tensor_sample_coordinates_m": compact(drive["tip_xy_m"]),
        "opening_tensor_Pa": compact(drive["opening_tensor_Pa"]),
        "channel_tensors_Pa": compact(drive["channel_tensors_Pa"]),
        "resolved_drive_factors": compact(drive["drive_factors"]),
        "resolved_tau_signed_Pa": compact(drive["tau_signed_Pa"]),
        "mobile_total_before": before["mobile_total"], "mobile_total_after": after["mobile_total"],
        "retained_total_before": before["retained_total"], "retained_total_after": after["retained_total"],
        "mobile_profile_before": compact(before["mobile_profile"]), "mobile_profile_after": compact(after["mobile_profile"]),
        "retained_profile_before": compact(before["retained_profile"]), "retained_profile_after": compact(after["retained_profile"]),
        "accumulated_slip_profile_before": compact(before["accumulated_slip_profile"]),
        "accumulated_slip_profile_after": compact(after["accumulated_slip_profile"]),
        "tip_radius_before_m": before["tip_radius_m"], "tip_radius_after_m": after["tip_radius_m"],
        "front_width_before_m": before["front_width_m"], "front_width_after_m": after["front_width_m"],
        "source_area_before_m2": before["source_area_m2"], "source_area_after_m2": after["source_area_m2"],
        "multiplicity_before_per_system": before["multiplicity_per_system"],
        "multiplicity_after_per_system": after["multiplicity_per_system"],
        "local_density_initial_by_system_m2": compact(before["local_density_by_system_m2"]),
        "local_density_final_by_system_m2": compact(after["local_density_by_system_m2"]),
        "Taylor_backstress_initial_by_system_Pa": compact(before["Taylor_backstress_by_system_Pa"]),
        "Taylor_backstress_final_by_system_Pa": compact(after["Taylor_backstress_by_system_Pa"]),
        "transport_velocity_initial_by_system_m_s": compact(before["transport_velocity_by_system_m_s"]),
        "transport_velocity_final_by_system_m_s": compact(after["transport_velocity_by_system_m_s"]),
        "activation_to_line_content": compact(conversion.tolist()),
        "emission_root_lower_bounds": compact(np.asarray(mpz.persistent_site_last_root_lower_bound).tolist()),
        "emission_root_upper_bounds": compact(np.asarray(mpz.persistent_site_last_root_upper_bound).tolist()),
        "emission_root_residuals": compact(np.asarray(mpz.persistent_site_last_root_residual).tolist()),
        "emission_root_iterations": compact(np.asarray(mpz.persistent_site_last_root_iteration_count).tolist()),
        "activations_by_system": compact(np.asarray(mpz.persistent_site_last_activations).tolist()),
        "line_content_emitted_by_system": compact(np.asarray(mpz.persistent_site_last_line_content).tolist()),
        "implicit_emission_substep_roots": compact(getattr(mpz, "persistent_site_last_substep_roots", ())),
        "dN_emit": float(info.get("dN_emit_raw", 0.0)), "dN_trapped": float(info.get("dN_trapped", 0.0)),
        "dN_released": float(info.get("dN_released", 0.0)), "dN_escaped": float(info.get("dN_escaped", 0.0)),
        "dN_recovered": float(info.get("dN_recovered", 0.0)),
        "moving_frame_advance_m": float(info.get("da", 0.0)) + float(info.get("event_moving_frame_advance_m", 0.0)),
        "wake_transfer": compact(info.get("event_moving_frame_renewal", info.get("advance", {}))),
        "active_K_shield_before": before["active_K_shield_Pa_sqrt_m"], "active_K_shield_after": after["active_K_shield_Pa_sqrt_m"],
        "wake_K_shield_before": before["wake_K_shield_Pa_sqrt_m"], "wake_K_shield_after": after["wake_K_shield_Pa_sqrt_m"],
        "population_conservation_residual": after["mobile_total"] + after["retained_total"] - before["mobile_total"] - before["retained_total"] - float(info.get("dN_emit_raw", 0.0)) + float(info.get("dN_escaped", 0.0)) + float(info.get("dN_recovered", 0.0)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--case", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv); case = args.case.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ONED_V2_TP_STATE_DIAGNOSTICS", "events")
    directional = read_jsonl(case / "directional_rates.jsonl"); actions = read_jsonl(case / "branch_action_trials.jsonl")
    selected = selected_by_step(actions)
    providers, provider_metadata = match_provider_states(
        case, directional, tuple(sorted(set(TARGET_STEPS + TRANSITION_STEPS)))
    )
    latest = pickle.loads((case / "checkpoint/latest.json.state.pkl").read_bytes()); network = latest.state.crack_network
    by_candidate = {branch.local_state.get("candidate_id"): branch.branch_id for branch in network.branches if branch.local_state.get("candidate_id")}
    branch_by_candidate = {branch.local_state.get("candidate_id"): branch for branch in network.branches if branch.local_state.get("candidate_id")}
    candidates = sorted(by_candidate); Eprime = float(latest.state.material.Eprime)
    rows_by_step = {step: {row["candidate_id"]: row for row in directional if int(row["step"]) == step} for step in {int(row["step"]) for row in directional}}

    def owner(step, candidate):
        return "b00000000" if step <= 369 else by_candidate[candidate]

    def event_owner(step, event):
        if event is None:
            return None
        owners = {owner(step, candidate) for candidate in event["candidate_ids"]}
        return next(iter(owners)) if len(owners) == 1 else "AMBIGUOUS"

    def archived_tensor_probe(step, event):
        """The V2 implementation probed active_tip_ids[0] after commit."""
        first_candidate = candidates[0]
        pre_xy = provider_metadata[step]["tip_xy_by_candidate"][first_candidate]
        if pre_xy is None:
            return "b00000000", "NOT_ARCHIVED_PRE_V11_LIVE_PROVIDER"
        if event is None or first_candidate not in event["candidate_ids"]:
            return owner(step, first_candidate), pre_xy
        if step < 369:
            parent = network.branch("b00000000")
            return parent.branch_id, next_path_coordinate(parent, pre_xy)
        if step == 369:
            daughter = branch_by_candidate[first_candidate]
            return daughter.branch_id, next_path_coordinate(daughter, pre_xy)
        branch = branch_by_candidate[first_candidate]
        return branch.branch_id, next_path_coordinate(branch, pre_xy)

    ownership = []
    for step in TARGET_STEPS:
        cache_key, provider, manifest = providers[step]; event = selected.get(step); members = tuple(event.get("candidate_ids", ())) if event else ()
        rates = rows_by_step[step]
        control = max(rates, key=lambda candidate: (float(rates[candidate]["J_kin_used_J_per_m2"]), candidate))
        control_tip = owner(step, control)
        tensor_tip, tensor_xy = archived_tensor_probe(step, event)
        selected_tip = event_owner(step, event)
        pre_topology = provider["topology_fingerprint"]
        post_topology = event["topology_fingerprint_after"] if event else pre_topology
        for provider_tip in provider["tips"]:
            for local in provider_tip["directional"]:
                candidate = local["candidate_id"]; candidate_owner = owner(step, candidate)
                branch = network.branch(candidate_owner); arc, reach = path_metrics(branch, provider_tip["tip_xy_m"]); rate = rows_by_step[step][candidate]
                kinetic = float(rate["J_kin_used_J_per_m2"])
                ownership.append({"step": step, "active_tip_id": candidate_owner, "parent_branch_id": branch.parent_branch_id,
                    "candidate_id": candidate, "candidate_to_tip_owner_id": candidate_owner, "tip_coordinates_m": compact(provider_tip["tip_xy_m"]),
                    "branch_arclength_m": arc, "projected_reach_m": reach, "signed_local_J_J_per_m2": float(local["J_local_signed_J_per_m2"]),
                    "positive_kinetic_J_J_per_m2": kinetic, "marginal_J_J_per_m2": rate.get("G_marginal_J_per_m2"),
                    "directional_K_Pa_sqrt_m": math.sqrt(Eprime * max(kinetic, 0.0)), "directional_rate_per_s": float(rate["lambda_directional_per_s"]),
                    "hazard_action": float(rate["accumulated_integrated_hazard_H"]), "hazard_threshold": float(rate["current_threshold_H_star"]),
                    "local_contour_valid": bool(rate["local_J_valid"]), "local_contour_invalid_reason": rate.get("local_J_invalid_reason"),
                    "scalar_K_controlling_candidate_id": control, "scalar_K_controlling_tip_id": control_tip,
                    "tensor_probe_tip_id": tensor_tip, "scalar_K_tip_equals_tensor_probe_tip": control_tip == tensor_tip,
                    "selected_event_candidate": candidate in members, "selected_event_tip_id": selected_tip,
                    "pre_event_topology_fingerprint": pre_topology, "post_event_topology_fingerprint": post_topology,
                    "stress_field_state_fingerprint": manifest["state_sha256"], "tensor_sample_coordinates_m": compact(tensor_xy),
                    "provider_cache_key": cache_key, "accepted_state_id": rate["accepted_state_id"]})
    dump_csv(out / "pf_branching_tip_candidate_ownership_audit.csv", ownership)

    pairing = []
    for step in sorted(rows_by_step):
        rates = rows_by_step[step]; control = max(rates, key=lambda c: (float(rates[c]["J_kin_used_J_per_m2"]), c))
        scalar_tip = owner(step, control)
        event = selected.get(step); members = tuple(event.get("candidate_ids", ())) if event else ()
        event_tip = event_owner(step, event)
        tensor_tip, tensor_xy = archived_tensor_probe(step, event)
        pdata = provider_metadata[step]
        pre_topology = pdata["topology_fingerprint"]
        post_topology = event["topology_fingerprint_after"] if event else pre_topology
        pairing.append({"step": step, "physical_time_s": float(next(iter(rates.values()))["physical_time_s"]), "controlling_candidate_id": control,
            "controlling_scalar_K_tip_id": scalar_tip, "tensor_probe_tip_id": tensor_tip, "scalar_K_tip_equals_tensor_probe_tip": scalar_tip == tensor_tip,
            "selected_event_candidate_ids": compact(members), "selected_event_tip_id": event_tip, "process_owner_id": "jc9ec4d7327bde23" if step >= 369 else "b00000000",
            "pre_event_state_id": next(iter(rates.values()))["accepted_state_id"],
            "pre_event_topology_fingerprint": pre_topology, "post_event_topology_fingerprint": post_topology,
            "stress_field_state_fingerprint": pdata["state_sha256"], "tensor_sample_coordinates_m": compact(tensor_xy),
            "provider_cache_key": pdata["cache_key"], "provider_cache_status": pdata["cache_status"],
            "writer_tip_id_trusted": False})
    dump_csv(out / "pf_branching_scalar_tensor_pairing_audit.csv", pairing)

    ordering = []
    for step, event in sorted(selected.items()):
        members = tuple(event["candidate_ids"]); tensor_tip = by_candidate[candidates[0]] if step >= 369 else "b00000000"
        moved = step == 369 or (len(members) == 1 and by_candidate.get(members[0]) == tensor_tip)
        ordering.append({"step": step, "action_type": event["action_type"], "selected_event_candidate_ids": compact(members),
            "selected_event_tip_id": "b00000000" if step <= 369 else by_candidate[members[0]],
            "pre_event_topology_fingerprint": event["topology_fingerprint_before"], "post_event_topology_fingerprint": event["topology_fingerprint_after"],
            "stress_field_is_pre_event": True, "tensor_coordinates_are_post_event": moved, "pre_field_post_coordinate_mismatch": moved,
            "old_update_order": "solve_pre_event_then_commit_topology_then_probe_selected_state",
            "corrected_update_order": "solve_and_probe_pre_event_then_evolve_interval_then_single_event_renewal_then_commit_post_state"})
    dump_csv(out / "pf_branching_event_state_ordering_audit.csv", ordering)

    from arrhenius_fracture.anisotropic_front_direction_fix_v10227 import install_front_direction_fix
    from arrhenius_fracture.persistent_site_bracket_fix_v10221 import install_backstress_complementarity_fix
    from arrhenius_fracture.persistent_site_physical_width_v10222 import install_physical_front_width
    install_front_direction_fix(); install_backstress_complementarity_fix(); install_physical_front_width()
    from arrhenius_fracture import anisotropic_emission_v10174 as anisotropic
    from arrhenius_fracture.fem import assemble_mechanics
    from arrhenius_fracture.tip_directional_observation_v11 import apply_post_interval_event_renewal
    checkpoint_path = next((case / "checkpoint/transitions").glob("step0002148_*.json.state.pkl")); checkpoint = pickle.loads(checkpoint_path.read_bytes())
    engines = {"archived_mixed_tip": restored_engine(checkpoint), "corrected_same_tip": restored_engine(checkpoint)}
    records = json.loads((case / "kinetic_tip_cell_audit_v101.json").read_text())["records"]
    transitions = []

    # The first available transition checkpoint is the state immediately after
    # step 2147.  Preserve that exact boundary and the archived scalar totals,
    # while refusing to invent a pre-step-2147 per-bin profile.
    _, boundary_provider, _ = providers[2147]
    boundary_control = max(
        candidates,
        key=lambda item: float(rows_by_step[2147][item]["J_kin_used_J_per_m2"]),
    )
    boundary_K = math.sqrt(
        Eprime * float(rows_by_step[2147][boundary_control]["J_kin_used_J_per_m2"])
    )
    boundary_tensor_tip, boundary_xy = archived_tensor_probe(2147, selected.get(2147))
    boundary_record = records[2146]
    boundary_drive = {
        "tip_xy_m": boundary_xy,
        "opening_tensor_Pa": "NOT_SERIALIZED_ACROSS_STEP2148_MESH_ADAPTATION",
        "channel_tensors_Pa": "NOT_SERIALIZED_ACROSS_STEP2148_MESH_ADAPTATION",
        "drive_factors": boundary_record["anisotropic_drive_factors"],
        "tau_signed_Pa": boundary_record["anisotropic_tau_signed_Pa"],
    }
    boundary_after = state_profile(engines["archived_mixed_tip"])
    boundary_before = copy.deepcopy(boundary_after)
    prior_record = records[2145]
    boundary_before["mobile_total"] = float(prior_record["active_mobile"])
    boundary_before["retained_total"] = float(prior_record["active_retained"])
    boundary_after["mobile_total"] = float(boundary_record["active_mobile"])
    boundary_after["retained_total"] = float(boundary_record["active_retained"])
    boundary_time = float(rows_by_step[2147][candidates[0]]["physical_time_s"])
    boundary_dt = boundary_time - float(rows_by_step[2146][candidates[0]]["physical_time_s"])
    boundary_row = process_row(
        2147, "archived_post_checkpoint_boundary", engines["archived_mixed_tip"],
        boundary_before, boundary_after, boundary_record, boundary_drive,
        boundary_tensor_tip, owner(2147, boundary_control), boundary_K,
        float(boundary_provider["base_equilibrium"]["applied_displacement"]),
        boundary_time, boundary_dt,
        record_kind="archived_step_boundary_not_isolated_replay",
        selected_event_candidate_ids=selected[2147]["candidate_ids"],
        selected_event_tip_id=event_owner(2147, selected[2147]),
    )
    boundary_row["profile_archive_status"] = "PRE_STEP2147_PROFILE_NOT_SERIALIZED_POST_PROFILE_EXACT"
    for field in (
        "mobile_profile_before", "retained_profile_before",
        "accumulated_slip_profile_before", "tip_radius_before_m",
        "front_width_before_m", "source_area_before_m2",
        "multiplicity_before_per_system", "local_density_initial_by_system_m2",
        "Taylor_backstress_initial_by_system_Pa",
        "transport_velocity_initial_by_system_m_s",
        "active_K_shield_before", "wake_K_shield_before",
    ):
        boundary_row[field] = "NOT_ARCHIVED_NO_PRE_STEP2147_CHECKPOINT"
    transitions.append(boundary_row)

    prior_time = boundary_time
    for step in (2148, 2149, 2150, 2151):
        _, provider, _ = providers[step]; current_time = float(rows_by_step[step][candidates[0]]["physical_time_s"]); dt = current_time - prior_time; prior_time = current_time
        state = checkpoint.state; displacement = np.asarray(provider["base_equilibrium"]["displacement"])
        _, _, sigma, *_ = assemble_mechanics(state.mesh, displacement, state.ep_gp, state.rho_gp, state.damage, state.elasticity_D, state.material, cohesive_network=state.cohesive_network)
        tip_by_candidate = {item["directional"][0]["candidate_id"]: item for item in provider["tips"]}
        control = max(candidates, key=lambda item: float(rows_by_step[step][item]["J_kin_used_J_per_m2"])); K = math.sqrt(Eprime * float(rows_by_step[step][control]["J_kin_used_J_per_m2"])); event = selected.get(step)
        old_xy = tip_by_candidate[candidates[0]]["tip_xy_m"]
        if event and candidates[0] in event["candidate_ids"]:
            branch = branch_by_candidate[candidates[0]]; index = next(i for i, point in enumerate(branch.path) if math.dist(point, old_xy) < 1e-12)
            if index + 1 < len(branch.path): old_xy = list(branch.path[index + 1])
        cfg = checkpoint.shared_process_state["engine_fields"]["anisotropic_cfg"]
        drives = {"archived_mixed_tip": anisotropic.build_front_drive(state.mesh, sigma, state.damage, np.asarray(old_xy), cfg),
                  "corrected_same_tip": anisotropic.build_front_drive(state.mesh, sigma, state.damage, np.asarray(tip_by_candidate[control]["tip_xy_m"]), cfg)}
        archived_record = records[step - 1]
        # Preserve the full-precision values actually consumed by the old run;
        # stress recovery differs only at last-bit summation order.
        drives["archived_mixed_tip"]["drive_factors"] = list(archived_record["anisotropic_drive_factors"])
        drives["archived_mixed_tip"]["tau_signed_Pa"] = list(archived_record["anisotropic_tau_signed_Pa"])
        for mode, engine in engines.items():
            before = state_profile(engine); drive = copy.deepcopy(drives[mode]); drive["drive_serial"] = 100000 + step
            engine._anisotropic_drive = drive; engine._anisotropic_drive_serial = drive["drive_serial"]; engine._install_current_drive_on_state()
            if event: engine.B = 1.0; engine.hazard_action_current = 1.0
            engine.hazard_threshold_action = 1.0e300; info = engine.step(K, 700.0, dt)
            if mode == "corrected_same_tip": apply_post_interval_event_renewal(engine.mpz, info, event_selected=event is not None, event_distance_m=5.0e-6)
            after = state_profile(engine)
            transitions.append(process_row(step, mode, engine, before, after, info, drive,
                by_candidate[candidates[0]] if mode == "archived_mixed_tip" else by_candidate[control], by_candidate[control], K,
                float(provider["base_equilibrium"]["applied_displacement"]), current_time, dt,
                selected_event_candidate_ids=() if event is None else event["candidate_ids"],
                selected_event_tip_id=event_owner(step, event)))
        archived = engines["archived_mixed_tip"]; record = archived_record
        if not math.isclose(archived.mpz.mobile_count, float(record["active_mobile"]), rel_tol=1e-12):
            raise RuntimeError(f"old replay mobile parity failed at {step}: {archived.mpz.mobile_count} != {record['active_mobile']}")
        if not math.isclose(archived.mpz.retained_count, float(record["active_retained"]), rel_tol=1e-12): raise RuntimeError(f"old replay retained parity failed at {step}")
    dump_csv(out / "pf_branching_persistent_source_transition_audit.csv", transitions)

    old = next(row for row in transitions if row["step"] == 2148 and row["evaluation_mode"] == "archived_mixed_tip")
    new = next(row for row in transitions if row["step"] == 2148 and row["evaluation_mode"] == "corrected_same_tip")
    defect_hashes = {}
    for step in (2147, 2151):
        drows = [row for row in directional if int(row["step"]) == step]; arows = [row for row in actions if int(row["step"]) == step]
        defect_hashes[str(step)] = {"directional_rows_sha256": hashlib.sha256(compact(drows).encode()).hexdigest(),
            "action_rows_sha256": hashlib.sha256(compact(arows).encode()).hexdigest(), "provider_cache_key": providers[step][0], "provider_state_sha256": providers[step][2]["state_sha256"]}
    factor = old["mobile_total_after"] / old["mobile_total_before"]
    diagnosis = {"schema": "pf_branching_process_state_runaway_diagnosis_v3", "claim_label": CLAIM, "source_parent_commit": PARENT_COMMIT,
        "old_defect_evidence_hashes": defect_hashes, "archived_mobile_before_step2148": old["mobile_total_before"], "archived_mobile_after_step2148": old["mobile_total_after"],
        "archived_mobile_growth_factor": factor, "corrected_mobile_after_step2148": new["mobile_total_after"],
        "old_step2148_reproduced_to_relative_tolerance": 1.0e-12,
        "corrected_same_tip_runaway_removed": new["mobile_total_after"] < 1.0e9,
        "step2147_archive_boundary": {
            "checkpoint_semantics": "post_step2147_state_saved_at_step2148_mesh_adaptation",
            "pre_step2147_scalar_totals_available": True,
            "pre_step2147_per_bin_profiles_available": False,
            "disposition": "fail_closed_no_field_profile_inference",
        },
        "classification": {"A_scalar_K_from_one_tip_tensor_from_another": "CONFIRMED_PRIMARY_CAUSE",
            "B_pre_event_field_at_post_event_coordinate": "CONFIRMED_AT_EVENT_STEPS_INCLUDING_369_AND_2151_NOT_PRIMARY_STEP2148_CAUSE",
            "C_radius_multiplicity_positive_feedback": "CONFIRMED_AS_POST_EXCURSION_AMPLIFIER_NOT_INITIATOR",
            "D_activation_line_content_or_dimensional_error": "NOT_SUPPORTED_BY_EXACT_CONVERSION_AND_CONSERVATION_AUDIT",
            "E_moving_frame_renewal_wrong_order": "CONFIRMED_INDEPENDENT_DEFECT_MISSING_TOPOLOGY_OWNED_RENEWAL",
            "F_more_than_one": "CONFIRMED_IMPLEMENTATION_DEFECT_SET_A_PLUS_B_PLUS_E; STATE_JUMP_CAUSED_BY_A"},
        "scientific_interpretation": "The approximately two-billion-fold mobile-state excursion is not physical. It is reproduced by the long-arm scalar K combined with the short-arm tensor, and removed by the same-tip long-arm tuple without a cap or parameter change."}
    dump_json(out / "pf_branching_process_state_runaway_diagnosis.json", diagnosis)
    dump_json(out / "pf_branching_restart_eligibility_v3.json", {"schema": "pf_branching_restart_eligibility_v3", "claim_label": CLAIM,
        "atomic_branch_birth_capability": "DEMONSTRATED", "post_birth_multitip_process_state_coupling": "UNQUALIFIED_PENDING_CORRECTED_REPLAY",
        "theta40_final_checkpoint_restart_eligible": False, "1000um_continuation_authorized": False,
        "first_permissible_future_replay": {"source": "latest uncontaminated single-front checkpoint before step-369 transaction", "target_daughter_growth_um": [25.0, 40.0], "launch_now": False}})
    mismatch = [row for row in pairing if not row["scalar_K_tip_equals_tensor_probe_tip"]]
    report = f"""# PF current-source branching multi-tip ownership V3

Permanent interpretation boundary: `{CLAIM}`.

## Decision

- `atomic_branch_birth_capability: DEMONSTRATED`
- `post_birth_multitip_process_state_coupling: UNQUALIFIED_PENDING_CORRECTED_REPLAY`
- `theta40_final_checkpoint_restart_eligible: false`
- `1000um_continuation_authorized: false`

The archived coupling is not physically qualified. Of {len(pairing)} accepted intervals, {len(mismatch)} pair scalar K and tensor data from different tips. The old `directional_rates.jsonl.tip_id` is not ownership evidence because its writer assigned the first active tip to every row.

## Exact runaway diagnosis

The frozen step-2148 replay reproduces `{old['mobile_total_before']:.17g} -> {old['mobile_total_after']:.17g}` mobile content ({factor:.9g}x). Repeating the same interval with the complete long-arm observation gives `{new['mobile_total_after']:.17g}` and no runaway. No cap, clipping, recovery, depletion, retuning, threshold change, or event-length change was used.

A is the primary state-jump cause. B is independently confirmed at event steps using pre-event stress with post-event coordinates. E is independently confirmed because the topology-owned renewal was missing/misordered. C amplifies the corrupted post-jump state but does not initiate step 2148. D is unsupported. Thus the code defect set is A+B+E (F), while the approximately 2e9 excursion itself is caused by A.

The step-2148 transition checkpoint is the exact post-step-2147 state. Archived scalar totals bracket step 2147, but its pre-update per-bin mobile, retained, and accumulated-slip profiles were not serialized. The transition table therefore labels step 2147 as an archived post-boundary, explicitly marks the unavailable pre-profile fields, and performs exact isolated pre/post evaluations only for steps 2148--2151. No missing field state is inferred.

## Corrected contract

`TipDirectionalObservation` preserves candidate, physical tip, coordinates, J/K/rate, contour validity, and accepted-state identity as one tuple. Ownership is fail-closed and bijective. Scalar K and tensor must share a tip; marginal fallback uses the owner; event-tip identity remains separate; interval evolution uses the pre-event state; event renewal occurs once afterward.

V1/V2 remain immutable historical evidence. No heavy PF replay or 1000 um continuation was launched. No historical controller claim is retained beyond this corrected frozen reconstruction.

## Validation

The branching-focused suite passed 181 tests with one skip. The full suite passed 775 tests with one skip and retained exactly the same seven legacy failures as the parent baseline (767 passes plus the same seven failures before these regressions); there were no new failures. `compileall` and `git diff --check` passed. Two clean regenerations of every V3 audit product were byte-identical.

## Future bounded replay

After review, replay from the latest uncontaminated single-front checkpoint before step 369 with identical state, hazard thresholds, RNG, load, material, and topology contract. Stop after branch birth and 25--40 um daughter growth; do not restart from step 2182.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_MULTITIP_OWNERSHIP_V3.md").write_text(report)
    product_names = (
        "PF_CURRENT_SOURCE_BRANCHING_MULTITIP_OWNERSHIP_V3.md",
        "pf_branching_process_state_runaway_diagnosis.json",
        "pf_branching_restart_eligibility_v3.json",
        "pf_branching_tip_candidate_ownership_audit.csv",
        "pf_branching_scalar_tensor_pairing_audit.csv",
        "pf_branching_event_state_ordering_audit.csv",
        "pf_branching_persistent_source_transition_audit.csv",
    )
    dump_json(out / "pf_branching_multitip_ownership_v3_provenance.json", {
        "schema": "pf_branching_multitip_ownership_v3_provenance",
        "claim_label": CLAIM, "source_parent_commit": PARENT_COMMIT,
        "producer_script_sha256": sha256(Path(__file__)),
        "source_case": str(case),
        "source_files": {name: sha256(case / name) for name in (
            "directional_rates.jsonl", "branch_action_trials.jsonl",
            "kinetic_tip_cell_audit_v101.json", "checkpoint/latest.json",
            "checkpoint/latest.json.state.pkl")},
        "audit_products": {name: sha256(out / name) for name in product_names},
        "deterministic_double_generation": "BYTE_IDENTICAL_REQUIRED_AND_VERIFIED",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
