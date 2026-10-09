#!/usr/bin/env python3
"""Continue the accepted upper/lower off-axis histories to a common 2 s origin."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_global_physical_interval, preview_downstream_first_passage,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_v5 import VoidPhase
from scripts.run_active_reflected_off_axis_trajectories_iterative import (
    _available_rates, _phase,
)
from scripts.run_off_axis_prescribed_void_benchmark_v2 import finite
from scripts.screen_informative_off_axis_operating_point import TEMPERATURE_K


SOURCE = ROOT / "artifacts/informative_off_axis_active_trajectories_iterative"
OUT = ROOT / "artifacts/informative_off_axis_2s_continuation"
TOTAL_HORIZON_S = 2.0
STARTS = {"above": "step_006_restart.json", "below": "step_005_restart.json"}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True) + "\n")


def accepted_time_s(state) -> float:
    return float(state.junction_process_state.get(
        "production_time_s", state.vacancy_transport_state_v2.transport_time_s,
    ))


def _event_rows(audit):
    return [row for row in audit["events"] if row["selected_event"] is not None]


def _checkpoint(case_dir: Path, state, interval_index: int):
    path = case_dir / f"interval_{interval_index:03d}_restart.json"
    manifest = write_checkpoint(state, path, compression="gzip")
    restored = restore_checkpoint(path)
    direct = complete_accepted_state_fingerprint(state)
    return path, manifest, direct == complete_accepted_state_fingerprint(restored)


def run_case(case: str):
    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    progress_path = case_dir / "progress.json"
    existing = sorted(case_dir.glob("interval_*_restart.json"))
    if existing:
        state = restore_checkpoint(existing[-1])
        intervals = json.loads(progress_path.read_text())["intervals"]
        for interval in intervals:
            for row in interval.get("current_competitor_availability", ()):
                if row.get("event_type") == "NO_DOWNSTREAM_EVENT":
                    row["status"] = "NO_EVENT_WITHIN_HORIZON"
    else:
        state = restore_checkpoint(SOURCE / case / STARTS[case])
        intervals = []
        _checkpoint(case_dir, state, 0)
    original_start_time = float(
        json.loads(progress_path.read_text()).get("original_start_time_s")
        if progress_path.exists() else accepted_time_s(state)
    )
    numerical_stop = None
    while accepted_time_s(state) < TOTAL_HORIZON_S - 1.0e-10:
        now = accepted_time_s(state)
        remaining = TOTAL_HORIZON_S - now
        phase = _phase(state)
        rates = None
        availability = None
        topology_preview = None
        active = tuple(state.crack_network.active_tip_ids)
        try:
            if phase == VoidPhase.RESOLVED_VOID.value:
                if len(active) != 1:
                    raise RuntimeError("resolved void has no unique active sharp front")
                branch_id = active[0]
                rows, rates, crossing = _available_rates(state, branch_id)
                availability = [{
                    "candidate_id": row["candidate_id"],
                    "status": row["directional_drive_status"],
                    "reason": row.get("directional_drive_unavailable_reason"),
                    "drive_source": row.get("drive_source"),
                    "G_kinetic_used_J_per_m2": row.get("G_kinetic_used_J_per_m2"),
                    "crossing_time_s": rate.get("crossing_time_s"),
                } for row, rate in zip(rows, rates)]
                unavailable = [row for row in availability if row["status"] != "AVAILABLE"]
                if unavailable:
                    numerical_stop = {
                        "status": "CURRENT_DIRECTIONAL_COMPETITION_INCOMPLETE",
                        "accepted_time_s": now, "availability": availability,
                    }
                    break
                bound_fingerprint = complete_accepted_state_fingerprint(state)

                def state_bound_rows(mechanical, active_tip_id):
                    if (active_tip_id == branch_id and
                            complete_accepted_state_fingerprint(mechanical) == bound_fingerprint):
                        return rows
                    if active_tip_id is None or _phase(mechanical) != VoidPhase.RESOLVED_VOID.value:
                        return None
                    refreshed, _, _ = _available_rates(mechanical, active_tip_id)
                    return refreshed

                history = {
                    "kind": "hold", "temperature_K": TEMPERATURE_K,
                    "directional_load_rows_for_state": state_bound_rows,
                    "maximum_events": 2,
                }
            elif phase in {VoidPhase.CONNECTED_VOID.value,
                           VoidPhase.DOWNSTREAM_FRONT_ACTIVE.value}:
                continuation = phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE.value
                topology_preview = preview_downstream_first_passage(
                    state, continuation=continuation,
                    temperature_K=TEMPERATURE_K, maximum_duration_s=remaining,
                )
                crossing = float(topology_preview.crossing_time_s)
                availability = [{
                    "event_type": topology_preview.event_type,
                    "owner_kind": topology_preview.owner_kind,
                    "status": (
                        "AVAILABLE" if topology_preview.eligible else
                        "NO_EVENT_WITHIN_HORIZON"
                        if topology_preview.event_type == "NO_DOWNSTREAM_EVENT" else
                        "UNAVAILABLE"
                    ),
                    "crossing_time_s": crossing,
                    "candidate_ids": topology_preview.candidate_ids,
                }]
                history = {
                    "kind": "hold", "temperature_K": TEMPERATURE_K,
                    "maximum_events": 2,
                }
            else:
                numerical_stop = {
                    "status": "UNSUPPORTED_CURRENT_TOPOLOGY_PHASE",
                    "accepted_time_s": now, "phase": phase,
                }
                break
            # Stop exactly on the owned root.  Overshooting by even a tiny
            # epsilon can make the same scheduler call enter the next topology
            # regime before the connection/event checkpoint is published.
            duration = (remaining if not math.isfinite(crossing) or crossing > remaining
                        else min(remaining, crossing))
            before_fp = complete_accepted_state_fingerprint(state)
            advanced, audit = advance_global_physical_interval(
                state, state, duration, history,
            )
            if float(audit["advanced_duration_s"]) <= 0.0:
                raise RuntimeError("global scheduler made no physical-time progress")
            checkpoint, manifest, restored_equal = _checkpoint(
                case_dir, advanced, len(intervals) + 1,
            )
            intervals.append({
                "interval": len(intervals) + 1,
                "start_time_s": now,
                "requested_duration_s": duration,
                "end_time_s": accepted_time_s(advanced),
                "phase_before": phase,
                "phase_after": _phase(advanced),
                "current_competitor_availability": availability,
                "canonical_rates": rates,
                "scheduler_audit": audit,
                "selected_events": [row["selected_event"] for row in _event_rows(audit)],
                "input_state_fingerprint": before_fp,
                "output_state_fingerprint": complete_accepted_state_fingerprint(advanced),
                "restart_manifest": checkpoint.name,
                "restart_state_file": manifest["state_file"],
                "direct_restored_equal": restored_equal,
                "terminal_crack_paths_m": [branch.path for branch in advanced.crack_network.branches],
            })
            state = advanced
            dump(progress_path, {
                "case": case, "total_horizon_s": TOTAL_HORIZON_S,
                "original_start_time_s": original_start_time,
                "intervals": intervals, "numerical_stop": numerical_stop,
            })
            print("OFF_AXIS_2S_INTERVAL", case, len(intervals),
                  accepted_time_s(state), intervals[-1]["selected_events"], flush=True)
        except Exception as exc:
            numerical_stop = {
                "status": "CURRENT_STATE_CONTINUATION_EXCEPTION",
                "accepted_time_s": now,
                "type": type(exc).__name__, "message": str(exc),
                "availability": availability,
            }
            dump(progress_path, {
                "case": case, "total_horizon_s": TOTAL_HORIZON_S,
                "original_start_time_s": original_start_time,
                "intervals": intervals, "numerical_stop": numerical_stop,
            })
            break
    final_time = accepted_time_s(state)
    phase = _phase(state)
    outcome = (
        "CONNECTED_DOWNSTREAM_SOURCE_NUMERICALLY_UNAVAILABLE"
        if numerical_stop is not None and phase == VoidPhase.CONNECTED_VOID.value else
        "CURRENT_DIRECTIONAL_COMPETITION_INCOMPLETE" if numerical_stop is not None else
        "DOWNSTREAM_CHILD_ACTIVE_BY_2S"
        if phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE.value else
        "CONNECTED_NO_DOWNSTREAM_BIRTH_BY_2S"
        if phase == VoidPhase.CONNECTED_VOID.value else
        "NO_CONNECTION_BY_2S"
    )
    result = {
        "schema": "informative.off-axis-2s-continuation/1",
        "classification": "EXPLORATORY_DOMAIN_SENSITIVE_STOCHASTIC_TRAJECTORY",
        "case": case, "source_checkpoint": str(SOURCE / case / STARTS[case]),
        "total_physical_observation_horizon_s": TOTAL_HORIZON_S,
        "accepted_start_time_s": original_start_time,
        "accepted_final_time_s": final_time,
        "outcome": outcome, "numerical_stop": numerical_stop,
        "terminal_phase": phase,
        "terminal_crack_paths_m": [branch.path for branch in state.crack_network.branches],
        "intervals": intervals,
        "latest_valid_restart": sorted(case_dir.glob("interval_*_restart.json"))[-1].name,
    }
    dump(case_dir / "result.json", result)
    return result


def make_plot(results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scripts.run_active_reflected_off_axis_trajectories_iterative import CENTERS, RADIUS
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
    for ax, case in zip(axes, ("above", "below")):
        row = results[case]
        for path in row["terminal_crack_paths_m"]:
            xy = np.asarray(path, float)
            ax.plot(xy[:, 0]*1e6, xy[:, 1]*1e6, "o-", lw=1.7, ms=3)
        center = CENTERS[case]
        ax.add_patch(Circle((center[0]*1e6, center[1]*1e6), RADIUS*1e6,
                            fill=False, color="tab:red"))
        short = (
            "Turn toward void—connected at 1.748 s"
            if row["terminal_phase"] == VoidPhase.CONNECTED_VOID.value else
            "One more straight step—competition incomplete"
            if row["numerical_stop"] else "2 s continuation"
        )
        ax.set_title(case.capitalize() + "—" + short)
        ax.set_xlabel("x (µm)"); ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y (µm)")
    fig.tight_layout(); fig.savefig(OUT / "crack_paths_2s.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {case: run_case(case) for case in ("above", "below")}
    make_plot(results)
    dump(OUT / "summary.json", {
        "schema": "informative.off-axis-2s-continuation-summary/1",
        "total_physical_observation_horizon_s": TOTAL_HORIZON_S,
        "parameters_changed": False,
        "stochastic_histories_preserved": True,
        "cases": {case: {
            "outcome": row["outcome"], "accepted_final_time_s": row["accepted_final_time_s"],
            "terminal_phase": row["terminal_phase"],
            "latest_valid_restart": f"{case}/{row['latest_valid_restart']}",
        } for case, row in results.items()},
        "plot": "crack_paths_2s.png",
    })


if __name__ == "__main__":
    main()
