#!/usr/bin/env python3
"""Checkpoint every physical event in the active reflected off-axis campaign."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    directional_sharp_front_rates, sharp_front_load_provider,
)
from scripts.run_off_axis_prescribed_void_benchmark_v2 import (
    KINETIC_DELTA_A, RADIUS, finite, initial_state,
)
from scripts.screen_informative_off_axis_operating_point import (
    LOAD_SCALE, MATERIAL, TEMPERATURE_K, centers,
)

OUT = ROOT/"artifacts/informative_off_axis_active_trajectories_iterative"
CENTERS = centers(2.2, 1.4)
MAX_EVENT_HORIZON_S = 1.0e-3


def dump(path: Path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def _available_rates(state, branch_id):
    from arrhenius_fracture.voiding_production_v5 import (
        _first_ray_cavity_intersection, matched_ligament_energy_observation,
    )
    start = state.crack_network.branch(branch_id).tip
    finite_contact = False
    if state.void_state is not None and state.void_state.cavities:
        finite_contact = any(
            point is not None and math.dist(start, point) <=
            KINETIC_DELTA_A*(1.0+1.0e-12)
            for point in (
                _first_ray_cavity_intersection(state, start, candidate.direction_xy)
                for candidate in state.competition.candidates
            )
        )
    provider = sharp_front_load_provider(
        state, branch_id=branch_id, candidates=state.competition.candidates,
        evaluate_overlap_marginals=not finite_contact,
        marginal_delta_a_m=(KINETIC_DELTA_A,), marginal_mesh_levels=(4, 5),
        # Every ordinary candidate is a discrete physical da trial even when a
        # different candidate can reach the cavity.  The cavity-intersecting
        # row is replaced below by the matched solid-ligament observation.
        # Disabling finite-increment mode globally here required two delta-a
        # values and made all non-contact competitors unavailable before any
        # mechanical trial was attempted.
        finite_increment_mode=True,
        virtual_mechanical_trial_transport="deferred_read_only",
        mechanical_refinement_policy="physical_candidate_support",
    )
    rows = provider["directional"]
    # A cavity can intersect the support tube of a long collinear certification
    # arc even when the actual finite candidate remains wholly in solid.  Retry
    # only such unavailable current candidates with the established full-graph
    # refinement policy; this changes representation, never direction or rate.
    def reaches_cavity(candidate):
        if state.void_state is None or not state.void_state.cavities:
            return False
        point = _first_ray_cavity_intersection(state, start, candidate.direction_xy)
        return (point is not None and
                math.dist(start, point) <= KINETIC_DELTA_A*(1.0+1.0e-12))

    retry_needed = any(
        row["directional_drive_status"] != "AVAILABLE"
        and not reaches_cavity(candidate)
        for candidate, row in zip(state.competition.candidates, rows)
    )
    if retry_needed:
        repaired = []
        for candidate, original in zip(state.competition.candidates, rows):
            if original["directional_drive_status"] == "AVAILABLE" or reaches_cavity(candidate):
                repaired.append(original); continue
            retry = sharp_front_load_provider(
                state, branch_id=branch_id, candidates=(candidate,),
                evaluate_overlap_marginals=False,
                marginal_delta_a_m=(KINETIC_DELTA_A,), marginal_mesh_levels=(2, 3),
                finite_increment_mode=True,
                virtual_mechanical_trial_transport="deferred_read_only",
                mechanical_refinement_policy="legacy_accumulated_generations",
            )["directional"][0]
            # The provider request fingerprint includes the requested candidate
            # inventory. Bind the repaired observable back to the unchanged
            # accepted-state/stress/topology identity carried by the full
            # current competition, while retaining its request fingerprint as
            # representation provenance.
            retry = {**retry,
                "accepted_state_id": original["accepted_state_id"],
                "stress_field_state_id": original["stress_field_state_id"],
                "topology_fingerprint": original["topology_fingerprint"],
                "representation_observation_topology_fingerprint": retry[
                    "topology_fingerprint"],
                "representation_retry": {
                "reason": original.get("directional_drive_unavailable_reason"),
                "policy": "legacy_accumulated_generations",
                "mesh_levels": [2, 3],
                "physical_candidate_unchanged": True,
            }}
            repaired.append(retry)
        rows = repaired
    if finite_contact:
        for index, (candidate, row) in enumerate(zip(state.competition.candidates, rows)):
            point = _first_ray_cavity_intersection(state, start, candidate.direction_xy)
            if point is None or math.dist(start, point) > KINETIC_DELTA_A*(1.0+1.0e-12):
                continue
            observation = matched_ligament_energy_observation(
                state, branch_id=branch_id, candidate=candidate,
            )
            if observation["status"] != "AVAILABLE":
                continue
            G = max(float(observation["G_ligament_J_per_m2"]), 0.0)
            K = math.sqrt(float(state.material.Eprime)*G)
            rows[index] = {
                **row,
                "directional_drive_status": "AVAILABLE",
                "directional_drive_unavailable_reason": None,
                "G_marginal_J_per_m2": G, "G_marginal": G,
                "G_kinetic_used_J_per_m2": G, "G_kinetic_used": G,
                "positive_J_J_per_m2": G,
                "drive_source": "MATCHED_CONNECTED_LIGAMENT_TOPOLOGY_ENERGY",
                "K_energy_equivalent_Pa_sqrt_m": K,
                "K_energy_equivalent": K,
                "K_directional_Pa_sqrt_m": K,
                "ligament_energy_observation": observation,
            }
    rates = directional_sharp_front_rates(
        state, rows, branch_id=branch_id, temperature_K=TEMPERATURE_K)
    observation_by_candidate = {
        row["candidate_id"]: row.get("ligament_energy_observation") for row in rows
    }
    rates = [{**rate, "ligament_energy_observation": observation_by_candidate.get(
        rate["candidate_id"])} for rate in rates]
    available = [row for row in rates
                 if row["directional_drive_status"] == "AVAILABLE"
                 and math.isfinite(float(row["crossing_time_s"]))]
    return rows, rates, min((float(row["crossing_time_s"])
                             for row in available), default=math.inf)


def _phase(state):
    cavities = () if state.void_state is None else state.void_state.cavities
    return None if not cavities else cavities[0].phase.value


def _outcome(case, state, event_count, limit):
    if case == "no_void":
        return "NO_VOID_REFERENCE_EVENT_LIMIT" if event_count >= limit else None
    phase = _phase(state)
    if phase in {"CONNECTED_VOID", "DOWNSTREAM_FRONT_ACTIVE"}:
        return phase
    tip = np.asarray(state.crack_network.branches[-1].tip, dtype=float)
    center = np.asarray(CENTERS[case], dtype=float)
    if tip[0] > center[0]+RADIUS:
        return "BYPASS_WITHOUT_CONNECTION"
    return None


def run_case(case: str, event_limit: int):
    case_dir = OUT/case
    case_dir.mkdir(parents=True, exist_ok=True)
    progress_path = case_dir/"progress.json"
    completed = []
    checkpoints = sorted(case_dir.glob("step_*_restart.json"))
    if checkpoints:
        state = restore_checkpoint(checkpoints[-1])
        if progress_path.exists():
            completed = [row for row in json.loads(progress_path.read_text())["steps"]
                         if row.get("event_committed")]
    else:
        state = initial_state(
            case, material_family=MATERIAL, temperature_K=TEMPERATURE_K,
            load_scale=LOAD_SCALE, cavity_centers=CENTERS)
        write_checkpoint(state, case_dir/"step_000_restart.json", compression="gzip")
    outcome = _outcome(case, state, len(completed), event_limit)
    while outcome is None and len(completed) < event_limit:
        active = tuple(state.crack_network.active_tip_ids)
        if len(active) != 1:
            outcome = "NO_SINGLE_ACTIVE_SHARP_FRONT"
            break
        branch_id = active[0]
        rows, rates, crossing = _available_rates(state, branch_id)
        if not math.isfinite(crossing) or crossing > MAX_EVENT_HORIZON_S:
            outcome = ("CANONICAL_DRIVE_FAIL_CLOSED"
                       if not math.isfinite(crossing) else
                       "QUALIFIED_DRIVE_HAZARD_LIMITED_WITHIN_PHYSICAL_WINDOW")
            terminal_diagnostic = {
                "first_crossing_time_s": crossing,
                "canonical_rates": rates,
                "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
            }
            dump(progress_path, {"case": case, "steps": completed,
                                 "terminal_diagnostic": terminal_diagnostic,
                                 "outcome": outcome})
            break
        bound_fingerprint = complete_accepted_state_fingerprint(state)

        def state_bound_rows(mechanical, active_tip_id):
            if (active_tip_id == branch_id and
                    complete_accepted_state_fingerprint(mechanical) == bound_fingerprint):
                return rows
            refreshed, _, _ = _available_rates(mechanical, active_tip_id)
            return refreshed

        duration = crossing*(1.0+1.0e-9)
        advanced, audit = advance_global_physical_interval(
            state, state, duration,
            {"kind": "hold", "temperature_K": TEMPERATURE_K,
             "directional_load_rows_for_state": state_bound_rows,
             "maximum_events": 1},
        )
        selected = [event["selected_event"] for event in audit["events"]
                    if event["selected_event"]]
        if not selected:
            outcome = "LOCALIZED_CROSSING_NO_EVENT_FAIL_CLOSED"
            break
        step_number = len(completed)+1
        checkpoint = case_dir/f"step_{step_number:03d}_restart.json"
        manifest = write_checkpoint(advanced, checkpoint, compression="gzip")
        restored = restore_checkpoint(checkpoint)
        direct_fp = complete_accepted_state_fingerprint(advanced)
        restored_fp = complete_accepted_state_fingerprint(restored)
        completed.append({
            "step": step_number, "event_committed": True,
            "first_crossing_time_s": crossing,
            "accepted_duration_s": duration,
            "selected_events": selected,
            "canonical_rates": rates,
            "scheduler_audit": audit,
            "terminal_crack_paths_m": [branch.path for branch in advanced.crack_network.branches],
            "cavity_phase": _phase(advanced),
            "accepted_state_fingerprint": direct_fp,
            "restart_manifest": checkpoint.name,
            "restart_state_file": manifest["state_file"],
            "direct_restored_equal": direct_fp == restored_fp,
        })
        state = advanced
        outcome = _outcome(case, state, len(completed), event_limit)
        dump(progress_path, {"case": case, "steps": completed, "outcome": outcome})
        print("ACCEPTED_EVENT", case, step_number, selected[-1], flush=True)
    if outcome is None:
        outcome = "EVENT_LIMIT_WITHOUT_INTERACTION"
    result = {
        "case": case, "classification": "EXPLORATORY_DOMAIN_SENSITIVE",
        "material_family": MATERIAL, "temperature_K": TEMPERATURE_K,
        "load_scale": LOAD_SCALE, "maximum_event_horizon_s": MAX_EVENT_HORIZON_S,
        "outcome": outcome, "steps": completed,
        "terminal_crack_paths_m": [branch.path for branch in state.crack_network.branches],
        "terminal_cavity_phase": _phase(state),
        "latest_valid_restart": sorted(
            case_dir.glob("step_*_restart.json"))[-1].name,
    }
    dump(case_dir/"result.json", result)
    print("TRAJECTORY_COMPLETE", case, outcome, flush=True)
    return result


def make_plot(results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    cases = ("no_void", "above", "below")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, case in zip(axes, cases):
        for path in results[case]["terminal_crack_paths_m"]:
            xy = np.asarray(path, dtype=float)
            ax.plot(xy[:, 0]*1e6, xy[:, 1]*1e6, "o-", lw=1.7, ms=3)
        if case != "no_void":
            center = CENTERS[case]
            ax.add_patch(Circle((center[0]*1e6, center[1]*1e6),
                                RADIUS*1e6, fill=False, color="tab:red"))
        ax.set_title(case+"\n"+results[case]["outcome"])
        ax.set_xlabel("x (µm)"); ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y (µm)")
    fig.tight_layout(); fig.savefig(OUT/"crack_paths.png", dpi=180); plt.close(fig)


def finalize_saved_results():
    results = {
        case: json.loads((OUT/case/"result.json").read_text())
        for case in ("no_void", "above", "below")
    }
    make_plot(results)
    dump(OUT/"summary.json", {
        "classification": "EXPLORATORY_DOMAIN_SENSITIVE_INTERACTION_TRAJECTORIES",
        "domain_sensitivity_status": "COMPLETED_SENSITIVITY_NOT_CONVERGED",
        "computational_stop": None,
        "canonical_observation_profile": {
            "baseline": "../informative_off_axis_canonical_profile/baseline.json",
            "optimized": "../informative_off_axis_canonical_profile/optimized.json",
        },
        "interaction_interpretation": {
            "above": (
                "ACCEPTED_UNFORCED_TURN_TOWARD_CAVITY_THEN_QUALIFIED_"
                "LIGAMENT_HAZARD_LIMIT_WITHIN_1MS"
            ),
            "below": "QUALIFIED_STRAIGHT_FRONT_HAZARD_LIMIT_WITHIN_1MS",
            "no_void": "QUALIFIED_REFERENCE_HAZARD_LIMIT_WITHIN_1MS",
            "capture_or_bypass_claimed": False,
        },
        "cases": {key: {"outcome": value["outcome"],
                         "latest_valid_restart": f"{key}/{value['latest_valid_restart']}",
                         "terminal_crack_paths_m": value["terminal_crack_paths_m"]}
                  for key, value in results.items()},
        "plot": "crack_paths.png",
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "no_void": run_case("no_void", 4),
        "above": run_case("above", 8),
        "below": run_case("below", 8),
    }
    finalize_saved_results()


if __name__ == "__main__":
    main()
