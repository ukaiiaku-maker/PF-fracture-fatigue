#!/usr/bin/env python3
"""Execute REFERENCE_CONTINUATION_CAPTURE_VS_ACTION_10S_V1."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step, vacancy_source_diagnostics,
)
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from scripts.run_active_two_mechanism_natural_void_study import (
    TEMPERATURE_K, canonical_drive_check, finite, site_source_exposure,
)


SOURCE = ROOT / "artifacts/informative_selected_natural_void_trajectory"
OUT = ROOT / "artifacts/informative_natural_capture_vs_action_10s_qualified_replay"
SOURCE_CHECKPOINT = SOURCE / "step_004_restart.json"
PROTOCOL_ID = "REFERENCE_CONTINUATION_CAPTURE_VS_ACTION_10S_V1_QUALIFIED_REPLAY"
MILESTONES_S = (1.0, 3.0, 6.0, 10.0)
MAX_INTERNAL_DT_S = 0.25
MIN_INTERNAL_DT_S = 0.015625
MAX_ONE_HIT_FRACTION_INCREMENT = 0.05
MAX_THRESHOLD_FRACTION_INCREMENT = 0.02


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True) + "\n")


def site_metrics(state):
    transport = state.vacancy_transport_state_v2
    direct = state.void_state.dislocation_interaction_state
    direct_by_id = {site.site_id: site for site in direct.sites}
    rows = {}
    for site in transport.site_states:
        quantum = site.required_embryo_inventory_area_m2 / site.required_hit_count
        owner = direct_by_id[site.site_id]
        rows[site.site_id] = {
            "reserved_inventory_area_m2": site.reserved_inventory_area_m2,
            "one_hit_inventory_area_m2": quantum,
            "one_hit_inventory_fraction": site.reserved_inventory_area_m2 / quantum,
            "birth_action": site.birth_action,
            "next_birth_threshold": site.next_birth_threshold(),
            "birth_threshold_fraction": site.birth_action / site.next_birth_threshold(),
            "successful_hit_count": site.hit_count,
            "direct_interaction_action": owner.interaction_action,
            "next_direct_threshold": owner.next_threshold(),
            "direct_threshold_fraction": owner.interaction_action / owner.next_threshold(),
        }
    return rows


def _accepted_trial(previous, dt, _fracture_root_replays=0):
    mechanical, plastic = accepted_bulk_load_step(
        previous, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0, duration_s=dt, temperature_K=TEMPERATURE_K,
        maximum_stagger_iterations=80,
    )
    exposure, source = site_source_exposure(previous, mechanical, dt)
    tensor, K, canonical, load_rows = canonical_drive_check(mechanical)
    fracture_crossing = min((
        float(row["crossing_time_s"])
        for row in canonical["fracture_engine_rows"]
        if row["directional_drive_status"] == "AVAILABLE"
    ), default=math.inf)
    if fracture_crossing < dt-max(1.0e-12, 1.0e-8*dt):
        if _fracture_root_replays >= 12 or fracture_crossing <= 0.0:
            raise RuntimeError("fracture event root replay did not converge")
        return _accepted_trial(previous, fracture_crossing, _fracture_root_replays+1)
    epdot = equivalent_plastic_strain_rate_s(previous.ep_gp, mechanical.ep_gp, dt)
    direct = [accepted_interaction_opportunity(
        mechanical, site, temperature_K=TEMPERATURE_K,
        accepted_equivalent_plastic_strain_rate_s=epdot,
    ) for site in mechanical.void_state.dislocation_interaction_state.sites]
    advanced, scheduler = advance_global_physical_interval(
        previous, mechanical, dt,
        {"kind": "hold", "temperature_K": TEMPERATURE_K,
         "stress_tensor_Pa": tensor,
         "K_effective_by_front": {ROOT_BRANCH_ID: K},
         "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
         "maximum_events": 16},
    )
    return advanced, plastic, exposure, source, canonical, K, epdot, direct, scheduler


def _fractions(before, after):
    increments = {}
    for site_id, row in after.items():
        old = before[site_id]
        increments[site_id] = {
            "one_hit_fraction_increment": (
                row["reserved_inventory_area_m2"] - old["reserved_inventory_area_m2"]
            ) / row["one_hit_inventory_area_m2"],
            "threshold_fraction_increment": (
                row["birth_action"] - old["birth_action"]
            ) / row["next_birth_threshold"],
        }
    return increments


def make_plot(history):
    import matplotlib.pyplot as plt
    times = np.asarray([row["end_additional_time_s"] for row in history], float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for site_id in ("site-1", "site-2"):
        axes[0].plot(times, [row["site_metrics"][site_id][
            "one_hit_inventory_fraction"] for row in history], label=site_id)
        axes[1].plot(times, [row["site_metrics"][site_id][
            "birth_threshold_fraction"] for row in history], label=site_id)
    axes[0].axhline(1.0, color="black", ls="--", lw=1)
    axes[0].set_title("Inventory remains below one hit")
    axes[0].set_ylabel("one-hit inventory fraction")
    axes[1].set_title("Successful action remains limiting")
    axes[1].set_ylabel("next-threshold action fraction")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("additional accepted time (s)")
        ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "inventory_vs_action_10s.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "progress.json"
    checkpoints = sorted(OUT.glob("milestone_*_restart.json"))
    if checkpoints:
        state = restore_checkpoint(checkpoints[-1])
        progress = json.loads(progress_path.read_text())
        history = progress["history"]
        additional_time = float(progress["accepted_additional_time_s"])
    else:
        state = restore_checkpoint(SOURCE_CHECKPOINT)
        history = []
        additional_time = 0.0
        manifest = write_checkpoint(state, OUT / "milestone_000_restart.json", compression="gzip")
        dump(progress_path, {
            "protocol_id": PROTOCOL_ID, "source_checkpoint": str(SOURCE_CHECKPOINT),
            "accepted_additional_time_s": 0.0, "history": history,
            "initial_restart_state_file": manifest["state_file"],
        })
    initial_metrics = site_metrics(state) if not history else history[0]["initial_site_metrics"]
    initial_ledgers = (dict(state.vacancy_transport_state_v2.ledgers) if not history
                       else history[0]["initial_transport_ledgers"])
    event_stop = None
    dt_next = MAX_INTERNAL_DT_S
    milestone_index = sum(additional_time >= value-1e-12 for value in MILESTONES_S)
    while additional_time < MILESTONES_S[-1]-1e-12 and event_stop is None:
        target = MILESTONES_S[milestone_index]
        dt = min(dt_next, target-additional_time)
        before_metrics = site_metrics(state)
        before_fp = complete_accepted_state_fingerprint(state)
        while True:
            trial = _accepted_trial(state, dt)
            advanced = trial[0]
            after_metrics = site_metrics(advanced)
            increments = _fractions(before_metrics, after_metrics)
            limiter = max(
                max(abs(row["one_hit_fraction_increment"])
                    / MAX_ONE_HIT_FRACTION_INCREMENT for row in increments.values()),
                max(abs(row["threshold_fraction_increment"])
                    / MAX_THRESHOLD_FRACTION_INCREMENT for row in increments.values()),
            )
            if limiter <= 1.0 or dt <= MIN_INTERNAL_DT_S*(1.0+1e-12):
                break
            dt = max(MIN_INTERNAL_DT_S, 0.5*dt)
        (advanced, plastic, exposure, source, canonical, K, epdot,
         direct, scheduler) = trial
        transport = advanced.vacancy_transport_state_v2
        selected = [event["selected_event"] for event in scheduler["events"]
                    if event["selected_event"] is not None]
        additional_time += float(scheduler["advanced_duration_s"])
        history.append(finite({
            "substep": len(history)+1,
            "initial_site_metrics": initial_metrics,
            "initial_transport_ledgers": initial_ledgers,
            "start_additional_time_s": additional_time-float(scheduler["advanced_duration_s"]),
            "dt_s": float(scheduler["advanced_duration_s"]),
            "end_additional_time_s": additional_time,
            "adaptive_limiter_ratio": limiter,
            "site_metrics": after_metrics,
            "site_fraction_increments": increments,
            "transport_ledgers": dict(transport.ledgers),
            "transport_solver_audit": dict(transport.last_solver_audit),
            "site_source_exposure_fraction_area": exposure,
            "integrated_plastic_source_fraction_area": source[
                "integrated_source_fraction_area_per_s"]*dt,
            "maximum_accepted_plastic_increment": float(np.max(epdot)*dt),
            "plastic_work_J_per_m": plastic["accepted_endpoint_path_plastic_work_J_per_m"],
            "canonical_crack_drive": canonical,
            "K_effective_Pa_sqrt_m": K,
            "direct_opportunity": direct,
            "selected_events": selected,
            "all_owners_previewed": scheduler["all_owners_previewed"],
            "input_state_fingerprint": before_fp,
            "output_state_fingerprint": complete_accepted_state_fingerprint(advanced),
            "site_phases": {site.site_id: site.phase.value
                            for site in advanced.void_state.sites},
            "cavity_count": len(advanced.void_state.cavities),
        }))
        state = advanced
        if selected:
            event_stop = {"selected_events": selected,
                          "accepted_additional_time_s": additional_time}
        reached_milestone = additional_time >= target-1e-10
        if reached_milestone or event_stop is not None:
            label = int(round(additional_time*1000.0))
            checkpoint = OUT / f"milestone_{label:05d}ms_restart.json"
            manifest = write_checkpoint(state, checkpoint, compression="gzip")
            restored = restore_checkpoint(checkpoint)
            history[-1]["restart_manifest"] = checkpoint.name
            history[-1]["restart_state_file"] = manifest["state_file"]
            history[-1]["direct_restored_equal"] = (
                complete_accepted_state_fingerprint(state)
                == complete_accepted_state_fingerprint(restored)
            )
            if reached_milestone:
                milestone_index += 1
            print("NATURAL_10S_MILESTONE", additional_time, selected, flush=True)
        dump(progress_path, {
            "protocol_id": PROTOCOL_ID, "source_checkpoint": str(SOURCE_CHECKPOINT),
            "accepted_additional_time_s": additional_time,
            "history": history, "event_stop": event_stop,
        })
        dt_next = min(MAX_INTERNAL_DT_S, 2.0*dt if limiter < 0.25 else dt)
    final_metrics = site_metrics(state)
    ledgers = dict(state.vacancy_transport_state_v2.ledgers)
    ledger_increments = {key: float(value)-float(initial_ledgers.get(key, 0.0))
                         for key, value in ledgers.items()}
    outcome = (
        "FIRST_ACTUAL_EVENT" if event_stop else
        "INVENTORY_LIMITED" if all(row["one_hit_inventory_fraction"] < 1.0
                                   for row in final_metrics.values()) else
        "HAZARD_LIMITED"
    )
    crack_drive_status_counts = {}
    for row in history:
        status = row["canonical_crack_drive"]["interpretation"]
        crack_drive_status_counts[status] = crack_drive_status_counts.get(status, 0) + 1
    result = {
        "schema": "informative.natural-capture-vs-action-10s/1",
        "protocol_id": PROTOCOL_ID,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "parameters_changed": False,
        "material_family": "DBTT", "vacancy_energy_variant": "combined",
        "temperature_K": TEMPERATURE_K, "load_factor": 1.0,
        "requested_additional_time_s": MILESTONES_S[-1],
        "accepted_additional_time_s": additional_time,
        "internal_maximum_step_s": MAX_INTERNAL_DT_S,
        "adaptive_inventory_fraction_limit": MAX_ONE_HIT_FRACTION_INCREMENT,
        "adaptive_action_fraction_limit": MAX_THRESHOLD_FRACTION_INCREMENT,
        "event_stop": event_stop, "outcome": outcome,
        "crack_drive_status_counts": crack_drive_status_counts,
        "global_event_ordering_qualification": (
            "COMPLETE" if set(crack_drive_status_counts) == {"CANONICAL_DRIVE_AVAILABLE"}
            else "CRACK_DRIVE_NUMERICALLY_INCOMPLETE"
        ),
        "initial_site_metrics": initial_metrics, "final_site_metrics": final_metrics,
        "incremental_transport_ledgers": ledger_increments,
        "history": history,
        "terminal_site_phases": {site.site_id: site.phase.value
                                 for site in state.void_state.sites},
        "terminal_cavity_count": len(state.void_state.cavities),
        "latest_valid_restart": sorted(OUT.glob("milestone_*_restart.json"))[-1].name,
    }
    dump(OUT / "result.json", result)
    make_plot(history)


if __name__ == "__main__":
    main()
