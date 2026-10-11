"""Autonomous event-bearing fatigue on the restored persistent/signed engine."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .fatigue_v1 import (
    FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform,
    ScaledExpFloorBarrier,
)
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .reduced_persistent_reference_r2 import advance_signed, build_restored_engine
from .reduced_shared_state_v1023 import SharedReducedConfig


MODEL_ID = "r2.autonomous-reduced-persistent-fatigue/1"
EVIDENCE_ROOT = (
    Path(__file__).resolve().parents[1] / "integration_evidence"
    / "option_completion_v2" / "event_online_qualified"
)
MATERIAL = EVIDENCE_ROOT / "selected_material_manifest_v10_2_22.csv"
SELECTION = EVIDENCE_ROOT / "v10_2_22_parameter_selection.json"


def _config() -> SharedReducedConfig:
    return SharedReducedConfig(
        mpz_length_um=50.0, mpz_n_bins=80, wake_length_um=50.0,
        max_internal_steps=100000,
    )


def _controller():
    barrier = ScaledExpFloorBarrier()
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=96), barrier, barrier, barrier,
    )


def _cycle_row(cycle: int, result: dict[str, Any], engine: Any) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "time_s": float(engine.t),
        "B": float(result["B"]),
        "dB": float(result["dB"]),
        "fired": bool(result["fired"]),
        "dN_emit": float(result["dN_emit"]),
        "micro_advance_step_m": float(result["kinetic_micro_advance_step_m"]),
        "micro_advance_total_m": float(result["kinetic_micro_advance_total_m"]),
        "checkpoint_committed_total_m": float(engine.checkpoint_advance_total_m),
        "r_eff_m": float(engine.r_eff()),
        "gross_source_slip": float(engine.mpz.cumulative_gross_source_activity),
        "gross_return_slip": float(engine.mpz.cumulative_gross_return_activity),
        "net_source_slip": float(engine.mpz.local_slip_count()),
    }


def _restore_geometry(engine: Any, checkpoint: Path) -> None:
    payload = json.loads((checkpoint / "high_cycle_live_checkpoint.json").read_text())
    geometry = payload["geometry_signature"]
    engine.n_adv = int(geometry[0])
    engine.a_adv = float(geometry[1])
    engine.micro_advance_total_m = float(geometry[2])
    engine.checkpoint_advance_total_m = float(geometry[3])
    engine.mpz.advance_total_m = float(geometry[4])


def run_autonomous_reduced_fatigue(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    engine = build_restored_engine(
        _config(), material_path=MATERIAL, selection_path=SELECTION,
    )
    controller = _controller()
    waveform = FatigueWaveform(Kmax=20.0e6, R=0.1, frequency_Hz=10.0)
    history = []
    for cycle in range(1, 5):
        result = engine.cycle_step_waveform(
            controller, waveform, 1800.0, force_cycles=1.0,
        )
        history.append(_cycle_row(cycle, result, engine))
        if result["fired"]:
            raise RuntimeError("unexpected event before restart checkpoint")

    checkpoint = root / "cycle4_checkpoint"
    old = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint)
    try:
        write_checkpoint(
            engine, temperature_K=1800.0, reason="r2_autonomous_cycle4",
            metadata={"model_id": MODEL_ID, "completed_cycles": 4},
        )
    finally:
        if old is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = old
    checkpoint_state = serialize_active_state(engine)
    uninterrupted = copy.deepcopy(engine)

    restarted = build_restored_engine(
        _config(), material_path=MATERIAL, selection_path=SELECTION,
    )
    _restore_geometry(restarted, checkpoint)
    restore_checkpoint(restarted, checkpoint)
    restored_state = serialize_active_state(restarted)
    checkpoint_exact = bool(
        np.array_equal(checkpoint_state.vector, restored_state.vector)
        and checkpoint_state.fields == restored_state.fields
        and checkpoint_state.geometry_signature == restored_state.geometry_signature
    )
    if not checkpoint_exact:
        raise RuntimeError("autonomous fatigue restart changed cycle-4 state")

    def finish(target):
        rows = []
        for cycle in range(5, 20):
            outcome = target.cycle_step_waveform(
                controller, waveform, 1800.0, force_cycles=1.0,
            )
            rows.append(_cycle_row(cycle, outcome, target))
            if outcome["fired"]:
                return rows
        raise RuntimeError("autonomous fatigue did not reach its native event")

    direct_tail = finish(uninterrupted)
    restarted_tail = finish(restarted)
    direct_terminal = serialize_active_state(uninterrupted)
    restart_terminal = serialize_active_state(restarted)
    continuation_exact = bool(
        np.array_equal(direct_terminal.vector, restart_terminal.vector)
        and direct_terminal.fields == restart_terminal.fields
        and direct_terminal.geometry_signature == restart_terminal.geometry_signature
    )
    if not continuation_exact:
        raise RuntimeError("autonomous fatigue continuation changed after restart")
    history.extend(restarted_tail)

    before_return_radius = float(restarted.r_eff())
    reverse = advance_signed(
        restarted, K_Pa_sqrt_m=-10.0e6, temperature_K=1800.0, dt_s=1.0e-3,
    )
    after_return_radius = float(restarted.r_eff())
    reload_result = advance_signed(
        restarted, K_Pa_sqrt_m=20.0e6, temperature_K=1800.0, dt_s=1.0e-3,
    )
    event_rows = [row for row in history if row["fired"]]
    payload = {
        "schema": MODEL_ID,
        "classification": "AUTONOMOUS_REDUCED_HIGH_TEMPERATURE_EXECUTION",
        "loading_owner": "FatigueWaveform and native persistent engine",
        "future_spatial_schedule_consumed": False,
        "material_path": str(MATERIAL),
        "selection_path": str(SELECTION),
        "temperature_K": 1800.0,
        "temperature_scope": "bounded constitutive execution; no new calibration claim",
        "waveform": {"Kmax_MPa_sqrt_m": 20.0, "R": 0.1, "frequency_Hz": 10.0},
        "history": history,
        "a_N": [
            {"cycle": row["cycle"], "a_m": row["checkpoint_committed_total_m"]}
            for row in history
        ],
        "event_count": len(event_rows),
        "event_cycle": event_rows[0]["cycle"],
        "event_advance_m": event_rows[0]["checkpoint_committed_total_m"],
        "event_policy": "native cooperative cleavage first passage; fixed 5 um checkpoint",
        "checkpoint_exact": checkpoint_exact,
        "restart_continuation_exact": continuation_exact,
        "direct_tail": direct_tail,
        "post_event_return_reload": {
            "negative_applied_load": True,
            "negative_local_drive": float(reverse.get("reversible_mobile_velocity_m_s", 0.0)) < 0.0,
            "physical_returned": float(reverse.get("dN_physical_returned", 0.0)),
            "radius_before_return_m": before_return_radius,
            "radius_after_return_m": after_return_radius,
            "reload_dN_emit": float(reload_result.get("dN_emit", 0.0)),
            "reload_r_eff_m": float(restarted.r_eff()),
        },
        "spatial_comparison": {
            "level": 3,
            "reused_same_material_spatial_root": str(EVIDENCE_ROOT),
            "reused_spatial_temperature_K": 900.0,
            "same_external_loading_and_temperature": False,
            "status": "SAME_LOAD_SPATIAL_COMPARISON_REMAINS_OPEN",
        },
    }
    (root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = ["MODEL_ID", "run_autonomous_reduced_fatigue"]
