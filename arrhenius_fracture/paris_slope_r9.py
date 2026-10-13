"""Phase-exact stochastic fatigue trajectories for the R9 Paris-slope study."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .archived_fatigue_material_r9 import (
    ArchivedMaterialBinding, build_archived_current_engine,
)
from .developed_fatigue_r3 import _restore_geometry
from .fatigue_v1 import FatigueWaveform
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .reduced_persistent_reference_r2 import (
    ReducedPersistentReversibleTipEngine, advance_signed,
)
from .stochastic_avalanche_tip import (
    clear_pending_geometry_events, pending_geometry_event_scope,
    pop_pending_geometry_event,
)


MODEL_ID = "r9.archived-v3.1-current-backend-phase-exact-fatigue/1"


@dataclass
class PhaseCursor:
    completed_cycles: int = 0
    phase_index: int = 0
    remaining_phase_s: float = 0.0


def configure_stochastic_engine(
    binding: ArchivedMaterialBinding, *, kernel_family_path: str | Path,
    seed: int,
):
    engine_type = ReducedPersistentReversibleTipEngine
    old_hazard = copy.deepcopy(engine_type._hazard_config_default)
    old_avalanche = copy.deepcopy(engine_type._avalanche_config_default)
    try:
        engine_type.configure_hazard(mode="exponential", seed=int(seed))
        engine_type.configure_avalanche(
            mode="threshold_scaled", minimum_factor=0.5,
            maximum_factor=4.0, geometry_subsegment_fraction=0.1,
        )
        engine = build_archived_current_engine(
            binding, kernel_family_path=kernel_family_path,
        )
        engine._hazard_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 1])
        )
        engine.hazard_threshold_action = engine._draw_threshold()
        engine.hazard_action_current = 0.0
        engine.hazard_event_index = 0
        engine.hazard_threshold_history = []
        return engine
    finally:
        engine_type._hazard_config_default = old_hazard
        engine_type._avalanche_config_default = old_avalanche


def phase_values(waveform: FatigueWaveform, phase_count: int) -> np.ndarray:
    phase = (np.arange(int(phase_count), dtype=float) + 0.5) * (
        2.0 * np.pi / int(phase_count)
    )
    return np.asarray(waveform.K_phase(phase), float)


def mean_cycle_inputs(engine: Any, waveform: FatigueWaveform, temperature_K: float,
                      phase_count: int) -> dict[str, float]:
    values = phase_values(waveform, phase_count)
    stress = np.asarray([engine.sigma_tip(float(value)) for value in values])
    emission = np.asarray(engine.manifest.emission.rate(stress, temperature_K), float)
    weights = np.maximum(emission, 0.0)
    average_stress = (float(np.sum(weights * stress) / np.sum(weights))
                      if np.sum(weights) > 0.0 else float(np.mean(stress)))
    rates = np.asarray([
        engine.lambda_cleave(float(value), temperature_K)[0] for value in stress
    ])
    return {
        "hazard_action_per_cycle": float(np.mean(rates) * waveform.period_s),
        "mean_cleavage_rate_s-1": float(np.mean(rates)),
        "emission_weighted_stress_Pa": average_stress,
        "sigma_min_Pa": float(np.min(stress)),
        "sigma_max_Pa": float(np.max(stress)),
    }


def _mean_block(engine: Any, waveform: FatigueWaveform, temperature_K: float,
                phase_count: int, cycles: int) -> dict[str, Any]:
    inputs = mean_cycle_inputs(engine, waveform, temperature_K, phase_count)
    engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = float(waveform.Kmax)
    engine.mpz._reversible_tip_radius_m = float(engine.r_eff())
    result = engine._integrate_coupled(
        waveform.Kmax, temperature_K, float(cycles) * waveform.period_s,
        stress_override=inputs["emission_weighted_stress_Pa"],
        lambda_override=inputs["mean_cleavage_rate_s-1"],
    )
    if result["fired"]:
        raise RuntimeError("qualified mean block crossed an event; safety bound failed")
    return {**inputs, "cycles": int(cycles), "result": result}


def _exact_phase_call(engine: Any, cursor: PhaseCursor, waveform: FatigueWaveform,
                      temperature_K: float, phase_count: int) -> dict[str, Any]:
    duration = waveform.period_s / int(phase_count)
    if cursor.remaining_phase_s <= 0.0:
        cursor.remaining_phase_s = duration
    K = float(phase_values(waveform, phase_count)[cursor.phase_index])
    before_remaining = float(cursor.remaining_phase_s)
    before_cycle = float(cursor.completed_cycles) + cursor.phase_index / phase_count
    result = advance_signed(
        engine, K_Pa_sqrt_m=K, temperature_K=temperature_K,
        dt_s=before_remaining,
    )
    consumed = float(result["kinetic_dt_consumed_s"])
    unused = float(result["kinetic_dt_unused_s"])
    if consumed <= 0.0 or unused < 0.0:
        raise RuntimeError("phase-exact current backend made invalid progress")
    event = pop_pending_geometry_event() if result["fired"] else None
    if bool(result["fired"]) != bool(event is not None):
        raise RuntimeError("native event and outer descriptor disagree")
    cursor.remaining_phase_s = max(before_remaining - consumed, 0.0)
    event_cycle = before_cycle + consumed / waveform.period_s
    tolerance = max(64.0 * np.spacing(duration), 1.0e-15)
    if cursor.remaining_phase_s <= tolerance:
        cursor.remaining_phase_s = 0.0
        cursor.phase_index += 1
        if cursor.phase_index == phase_count:
            cursor.phase_index = 0
            cursor.completed_cycles += 1
    return {
        "event": event, "result": result, "K_Pa_sqrt_m": K,
        "event_cycles": event_cycle, "consumed_s": consumed,
        "unused_s": unused,
    }


def qualify_mean_cycle_overlap(engine: Any, waveform: FatigueWaveform,
                               temperature_K: float, phase_count: int) -> dict[str, Any]:
    explicit = copy.deepcopy(engine)
    averaged = copy.deepcopy(engine)
    for value in (explicit, averaged):
        value.hazard_threshold_action = 1.0e100
        value.hazard_action_current = 0.0
        value.B = 0.0
    cursor = PhaseCursor()
    clear_pending_geometry_events()
    with pending_geometry_event_scope():
        while cursor.completed_cycles < 1:
            row = _exact_phase_call(
                explicit, cursor, waveform, temperature_K, phase_count,
            )
            if row["event"] is not None:
                raise RuntimeError("overlap fixture unexpectedly crossed its huge mark")
    with pending_geometry_event_scope():
        mean = _mean_block(averaged, waveform, temperature_K, phase_count, 1)
    left = serialize_active_state(explicit).vector
    right = serialize_active_state(averaged).vector
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    state_error = float(np.max(np.abs(left-right) / scale))
    hazard_explicit = float(explicit.hazard_action_current)
    hazard_mean = float(averaged.hazard_action_current)
    hazard_error = abs(hazard_explicit-hazard_mean) / max(
        abs(hazard_explicit), abs(hazard_mean), 1.0e-300,
    )
    micro_error = abs(
        float(explicit.micro_advance_total_m)-float(averaged.micro_advance_total_m)
    ) / max(abs(float(explicit.micro_advance_total_m)), 1.0e-30)
    passed = state_error <= 2.0e-3 and hazard_error <= 2.0e-3 and micro_error <= 2.0e-3
    return {
        "passed": passed, "phase_count": int(phase_count),
        "state_relative_max_error": state_error,
        "hazard_relative_error": hazard_error,
        "micro_translation_relative_error": micro_error,
        "explicit_hazard_action": hazard_explicit,
        "mean_hazard_action": hazard_mean,
        "mean_inputs": {key: value for key, value in mean.items() if key != "result"},
    }


def _checkpoint(engine: Any, waveform: FatigueWaveform, temperature_K: float,
                directory: Path, cursor: PhaseCursor, reason: str) -> None:
    previous = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(directory)
    try:
        write_checkpoint(
            engine, waveform=waveform, temperature_K=temperature_K,
            reason=reason, metadata={"model_id": MODEL_ID, "cursor": asdict(cursor)},
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = previous


def _write_progress(directory: Path, payload: dict[str, Any]) -> None:
    path = directory/"trajectory_progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def run_trajectory(
    binding: ArchivedMaterialBinding, *, kernel_family_path: str | Path,
    Kmax_MPa_sqrt_m: float, seed: int, checkpoint_directory: str | Path,
    target_events: int = 20, target_extension_m: float = 100.0e-6,
    maximum_events: int = 60, maximum_cycles: float = 1.0e12,
    phase_count: int = 64, temperature_K: float = 300.0,
    R: float = 0.1, frequency_Hz: float = 1000.0,
) -> dict[str, Any]:
    directory = Path(checkpoint_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    waveform = FatigueWaveform(
        Kmax=float(Kmax_MPa_sqrt_m) * 1.0e6, R=float(R),
        frequency_Hz=float(frequency_Hz),
    )
    engine = configure_stochastic_engine(
        binding, kernel_family_path=kernel_family_path, seed=seed,
    )
    progress_path = directory/"trajectory_progress.json"
    native_checkpoint = directory/"high_cycle_live_checkpoint.json"
    if progress_path.is_file() and native_checkpoint.is_file():
        saved = json.loads(progress_path.read_text())
        identity = saved["identity"]
        expected = {
            "candidate_id": binding.optimized_row["candidate_id"],
            "Kmax_MPa_sqrt_m": float(Kmax_MPa_sqrt_m), "seed": int(seed),
            "phase_count": int(phase_count), "temperature_K": float(temperature_K),
            "R": float(R), "frequency_Hz": float(frequency_Hz),
        }
        if identity != expected:
            raise RuntimeError("R9 restart identity differs from requested trajectory")
        _restore_geometry(engine, directory)
        restore_checkpoint(engine, directory)
        cursor = PhaseCursor(**saved["cursor"])
        overlap = dict(saved["overlap"])
        events = list(saved["events"])
        history = list(saved["history"])
        accelerations = list(saved["acceleration"])
    else:
        cursor = PhaseCursor()
        overlap = qualify_mean_cycle_overlap(
            engine, waveform, temperature_K, phase_count,
        )
        clear_pending_geometry_events()
        events: list[dict[str, Any]] = []
        history = [{"cycles": 0.0, "extension_m": 0.0, "kind": "initial"}]
        accelerations = []
        _checkpoint(engine, waveform, temperature_K, directory, cursor, "initial")
    identity = {
        "candidate_id": binding.optimized_row["candidate_id"],
        "Kmax_MPa_sqrt_m": float(Kmax_MPa_sqrt_m), "seed": int(seed),
        "phase_count": int(phase_count), "temperature_K": float(temperature_K),
        "R": float(R), "frequency_Hz": float(frequency_Hz),
    }

    def save_progress() -> None:
        _write_progress(directory, {
            "schema": "r9.trajectory-progress/1", "identity": identity,
            "cursor": asdict(cursor), "overlap": overlap, "events": events,
            "history": history, "acceleration": accelerations,
            "geometry": {
                "n_adv": int(engine.n_adv),
                "committed_extension_m": float(engine.checkpoint_advance_total_m),
                "continuous_mpz_progress_m": float(engine.micro_advance_total_m),
            },
        })
    save_progress()
    def progress_cycles() -> float:
        base = float(cursor.completed_cycles) + cursor.phase_index/phase_count
        if cursor.remaining_phase_s > 0.0:
            duration = waveform.period_s/phase_count
            base += (1.0-cursor.remaining_phase_s/duration)/phase_count
        return base

    while ((len(events) < int(target_events)
            or float(engine.checkpoint_advance_total_m) < float(target_extension_m))
           and len(events) < int(maximum_events)):
        current_cycles = progress_cycles()
        if current_cycles >= float(maximum_cycles):
            break
        if (overlap["passed"] and cursor.phase_index == 0
                and cursor.remaining_phase_s == 0.0):
            inputs = mean_cycle_inputs(engine, waveform, temperature_K, phase_count)
            action = max(inputs["hazard_action_per_cycle"], 0.0)
            remaining = max(float(engine.hazard_threshold_action)
                            - float(engine.hazard_action_current), 0.0)
            safe = 0 if action <= 0.0 else int(max(math.floor(0.9*remaining/action)-1, 0))
            safe = min(safe, 100000, int(maximum_cycles-cursor.completed_cycles))
            if safe >= 2:
                block = _mean_block(
                    engine, waveform, temperature_K, phase_count, safe,
                )
                cursor.completed_cycles += safe
                accelerations.append({
                    "start_cycle": cursor.completed_cycles-safe,
                    "cycles": safe, "safety_fraction": 0.9,
                    "hazard_action_per_cycle": action,
                    "overlap_qualified": True,
                })
                history.append({
                    "cycles": float(cursor.completed_cycles),
                    "extension_m": float(engine.checkpoint_advance_total_m),
                    "continuous_mpz_progress_m": float(engine.micro_advance_total_m),
                    "kind": "qualified_mean_block",
                })
                _checkpoint(engine, waveform, temperature_K, directory, cursor, "mean_block")
                save_progress()
                continue
        row = _exact_phase_call(
            engine, cursor, waveform, temperature_K, phase_count,
        )
        if row["event"] is None:
            continue
        descriptor = dict(row["event"])
        diagnostic = engine.mpz.diagnostics()
        event = {
            "event_index": len(events)+1,
            "cycles": float(row["event_cycles"]),
            "phase": float(row["event_cycles"] % 1.0),
            "K_Pa_sqrt_m": float(row["K_Pa_sqrt_m"]),
            "threshold_action": float(descriptor["threshold_action"]),
            "event_advance_m": float(descriptor["event_advance_m"]),
            "extension_m": float(engine.checkpoint_advance_total_m),
            "continuous_mpz_progress_m": float(engine.micro_advance_total_m),
            "tip_radius_m": float(engine.r_eff()),
            "K_shield_Pa_sqrt_m": float(engine.K_shield()),
            "mobile_count": float(engine.mpz.mobile_count),
            "retained_count": float(engine.mpz.retained_count),
            "gross_source_activity": float(engine.mpz.cumulative_gross_source_activity),
            "gross_return_activity": float(engine.mpz.cumulative_gross_return_activity),
            "unused_phase_time_s": float(row["unused_s"]),
            "remaining_waveform_continued": bool(row["unused_s"] > 0.0),
            "persistent_site_multiplicity_per_system": float(
                diagnostic["persistent_site_multiplicity_per_system"]),
        }
        events.append(event)
        history.append({"cycles": event["cycles"], "extension_m": event["extension_m"],
                        "continuous_mpz_progress_m": event["continuous_mpz_progress_m"],
                        "kind": "event", "event_index": event["event_index"]})
        _checkpoint(engine, waveform, temperature_K, directory, cursor, "accepted_event")
        save_progress()
    final_cycles = progress_cycles()
    completed = (len(events) >= int(target_events)
                 and float(engine.checkpoint_advance_total_m) >= float(target_extension_m))
    return {
        "schema": MODEL_ID,
        "status": ("developed_target_completed" if completed
                   else ("maximum_event_budget_censor" if len(events) >= maximum_events
                         else "maximum_cycles_censor")),
        "candidate_id": binding.optimized_row["candidate_id"],
        "Kmax_MPa_sqrt_m": float(Kmax_MPa_sqrt_m),
        "DeltaK_MPa_sqrt_m": (1.0-float(R))*float(Kmax_MPa_sqrt_m),
        "temperature_K": float(temperature_K), "R": float(R),
        "frequency_Hz": float(frequency_Hz), "phase_count": int(phase_count),
        "waveform_start_phase": "KMAX_COSINE_PHASE_ZERO",
        "seed": int(seed), "events": events, "history": history,
        "target_events": int(target_events),
        "target_extension_m": float(target_extension_m),
        "maximum_events": int(maximum_events),
        "acceleration": accelerations, "overlap": overlap,
        "final_cycles": final_cycles,
        "final_extension_m": float(engine.checkpoint_advance_total_m),
        "final_continuous_mpz_progress_m": float(engine.micro_advance_total_m),
        "checkpoint_directory": str(directory),
    }


__all__ = [
    "MODEL_ID", "PhaseCursor", "configure_stochastic_engine",
    "mean_cycle_inputs", "qualify_mean_cycle_overlap", "run_trajectory",
]
