"""Phase-complete multi-advance fatigue on the restored reduced state.

Unlike the legacy mean-cycle entry, this executor never discards the remainder
of a phase when a checkpoint is crossed.  The kinetic integrator advances the
MPZ continuously; the reduced outer geometry ledger is updated once from the
queued descriptor and does not translate the MPZ again.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .fatigue_v1 import FatigueWaveform
from .matched_persistent_execution_r3 import SPATIAL_EVIDENCE
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .reduced_persistent_reference_r2 import advance_signed, build_restored_engine
from .reduced_shared_state_v1023 import SharedReducedConfig
from .stochastic_avalanche_tip import (
    clear_pending_geometry_events, pop_pending_geometry_event,
)


MODEL_ID = "r3.phase-complete-multi-advance-fatigue/1"


def build_developed_engine():
    """The already event-bearing R2 donor configuration, unchanged."""
    return build_restored_engine(
        SharedReducedConfig(
            mpz_length_um=50.0, mpz_n_bins=80, wake_length_um=50.0,
            max_internal_steps=100000,
        ),
        material_path=SPATIAL_EVIDENCE / "selected_material_manifest_v10_2_22.csv",
        selection_path=SPATIAL_EVIDENCE / "v10_2_22_parameter_selection.json",
    )


def _same_active_state(left: Any, right: Any) -> bool:
    return bool(
        np.array_equal(left.vector, right.vector)
        and left.fields == right.fields
        and left.geometry_signature == right.geometry_signature
    )


def _restore_geometry(engine: Any, checkpoint: Path) -> None:
    payload = json.loads((checkpoint / "high_cycle_live_checkpoint.json").read_text())
    geometry = payload["geometry_signature"]
    engine.n_adv = int(geometry[0])
    engine.a_adv = float(geometry[1])
    engine.micro_advance_total_m = float(geometry[2])
    engine.checkpoint_advance_total_m = float(geometry[3])
    engine.mpz.advance_total_m = float(geometry[4])


def advance_explicit_cycle(
    engine: Any,
    waveform: FatigueWaveform,
    *,
    temperature_K: float,
    phase_count: int = 24,
) -> dict[str, Any]:
    if phase_count < 4:
        raise ValueError("explicit waveform requires at least four phases")
    phase = (
        np.arange(int(phase_count), dtype=float) + 0.5
    ) * (2.0 * np.pi / phase_count)
    K_values = waveform.K_phase(phase)
    phase_duration = waveform.period_s / phase_count
    events = []
    phase_rows = []
    for phase_index, K in enumerate(K_values):
        remaining = float(phase_duration)
        calls = 0
        while remaining > max(np.spacing(phase_duration) * 64.0, 1.0e-15):
            calls += 1
            before_translation = float(engine.mpz.advance_total_m)
            result = advance_signed(
                engine, K_Pa_sqrt_m=float(K), temperature_K=temperature_K,
                dt_s=remaining,
            )
            consumed = float(result["kinetic_dt_consumed_s"])
            unused = float(result["kinetic_dt_unused_s"])
            if consumed <= 0.0 or consumed > remaining * (1.0 + 1.0e-12):
                raise RuntimeError("kinetic phase executor made invalid time progress")
            if result["fired"]:
                descriptor = pop_pending_geometry_event()
                if descriptor is None:
                    raise RuntimeError("checkpoint crossing lacked one outer descriptor")
                translated = float(engine.mpz.advance_total_m) - before_translation
                proposed = float(descriptor["event_advance_m"])
                cumulative_error = (
                    float(engine.mpz.advance_total_m)
                    - float(engine.checkpoint_advance_total_m)
                    - float(engine.B) * float(engine.avalanche_event_advance_m)
                )
                if abs(cumulative_error) > 2.0e-14:
                    raise RuntimeError("continuous MPZ/checkpoint ledger does not close")
                events.append({
                    "phase_index": int(phase_index),
                    "phase_rad": float(phase[phase_index]),
                    "K_Pa_sqrt_m": float(K),
                    "consumed_phase_time_s": consumed,
                    "unused_phase_time_s": unused,
                    "event_advance_m": proposed,
                    "threshold_action": float(descriptor["threshold_action"]),
                    "mpz_translation_in_final_subcall_m": translated,
                    "cumulative_mpz_checkpoint_error_m": cumulative_error,
                    "outer_geometry_translation_m": 0.0,
                    "remaining_waveform_continued": unused > 0.0,
                })
            remaining = unused
            if calls > 64:
                raise RuntimeError("too many checkpoint crossings in one phase")
        phase_rows.append({
            "phase_index": int(phase_index), "K_Pa_sqrt_m": float(K),
            "requested_time_s": float(phase_duration), "subcalls": calls,
        })
    return {
        "events": events,
        "phases": phase_rows,
        "cycle_time_consumed_s": float(waveform.period_s),
        "unused_cycle_time_s": 0.0,
    }


def advance_mean_cycle_complete(
    engine: Any,
    waveform: FatigueWaveform,
    *,
    temperature_K: float,
    phase_count: int = 96,
) -> dict[str, Any]:
    """Execute the retained phase-quadrature/mean-rate law without lost time.

    The donor law evaluates rates on phase points, then evolves one coupled
    mean-rate block.  It therefore has no within-cycle phase coordinate to
    restart.  This routine preserves that law and repeatedly invokes the same
    accepted block on its reported unused duration after each renewal.
    """
    if phase_count < 8:
        raise ValueError("mean-cycle quadrature requires at least eight phases")
    phase = (np.arange(phase_count, dtype=float) + 0.5) * (2.0*np.pi/phase_count)
    K_values = waveform.K_phase(phase)
    sig = np.asarray([engine.sigma_tip(float(K)) for K in K_values])
    lam_e = engine.manifest.emission.rate(sig, temperature_K)
    weights = np.maximum(lam_e, 0.0)
    average_stress = (
        float(np.sum(weights*sig)/np.sum(weights))
        if np.sum(weights) > 0.0 else float(np.mean(sig))
    )
    lam_c = np.asarray([
        engine.lambda_cleave(float(value), temperature_K)[0] for value in sig
    ])
    lambda_average = float(np.mean(lam_c))
    remaining = float(waveform.period_s)
    events = []
    calls = 0
    engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = float(waveform.Kmax)
    while remaining > max(64.0*np.spacing(waveform.period_s), 1.0e-15):
        calls += 1
        before_translation = float(engine.mpz.advance_total_m)
        result = engine._integrate_coupled(
            waveform.Kmax, temperature_K, remaining,
            stress_override=average_stress, lambda_override=lambda_average,
        )
        consumed = float(result["dt_consumed"])
        unused = float(result["dt_unused"])
        if consumed <= 0.0 or consumed > remaining*(1.0+1.0e-12):
            raise RuntimeError("mean-cycle executor made invalid time progress")
        if result["fired"]:
            descriptor = pop_pending_geometry_event()
            if descriptor is None:
                raise RuntimeError("mean-cycle renewal lacked outer descriptor")
            translated = float(engine.mpz.advance_total_m)-before_translation
            proposed = float(descriptor["event_advance_m"])
            cumulative_error = (
                float(engine.mpz.advance_total_m)
                - float(engine.checkpoint_advance_total_m)
                - float(engine.B) * float(engine.avalanche_event_advance_m)
            )
            if abs(cumulative_error) > 2.0e-14:
                raise RuntimeError("mean-cycle MPZ/checkpoint ledger does not close")
            events.append({
                "K_provider": "phase-quadrature mean-rate block",
                "Kmax_Pa_sqrt_m": float(waveform.Kmax),
                "phase_quadrature_count": int(phase_count),
                "mean_cleavage_rate_s-1": lambda_average,
                "emission_weighted_stress_Pa": average_stress,
                "consumed_cycle_time_s": consumed,
                "unused_cycle_time_s": unused,
                "event_advance_m": proposed,
                "threshold_action": float(descriptor["threshold_action"]),
                "mpz_translation_in_final_subcall_m": translated,
                "cumulative_mpz_checkpoint_error_m": cumulative_error,
                "outer_geometry_translation_m": 0.0,
                "remaining_waveform_continued": unused > 0.0,
            })
        remaining = unused
        if calls > 64:
            raise RuntimeError("too many mean-cycle renewals in one cycle")
    return {
        "events": events,
        "subcalls": calls,
        "cycle_time_consumed_s": float(waveform.period_s),
        "unused_cycle_time_s": 0.0,
        "mean_cleavage_rate_s-1": lambda_average,
        "emission_weighted_stress_Pa": average_stress,
    }


def _run_until(engine: Any, waveform: FatigueWaveform, *, start_cycle: int,
               target_events: int, phase_count: int, maximum_cycles: int):
    history = []
    events = []
    cycle = int(start_cycle)
    while engine.n_adv < target_events and cycle < maximum_cycles:
        cycle += 1
        result = advance_mean_cycle_complete(
            engine, waveform, temperature_K=1800.0, phase_count=phase_count,
        )
        for event in result["events"]:
            events.append({**event, "cycle": cycle})
        history.append({
            "cycle": cycle,
            "time_s": float(engine.t),
            "checkpoint_count": int(engine.n_adv),
            "a_m": float(engine.checkpoint_advance_total_m),
            "micro_advance_total_m": float(engine.micro_advance_total_m),
            "fractional_checkpoint_progress": float(engine.B),
            "gross_source_slip": float(engine.mpz.cumulative_gross_source_activity),
            "gross_return_slip": float(engine.mpz.cumulative_gross_return_activity),
            "net_source_slip": float(engine.mpz.local_slip_count()),
            "tip_radius_m": float(engine.r_eff()),
            "event_count_this_cycle": len(result["events"]),
            "unused_cycle_time_s": float(result["unused_cycle_time_s"]),
        })
    if engine.n_adv < target_events:
        raise RuntimeError(f"only {engine.n_adv} events by cycle {cycle}")
    # Event ordinals are trajectory-local and deterministic.
    for index, event in enumerate(events, start=1):
        event["event_index_in_segment"] = index
        event.pop("event_index", None)
    return history, events, cycle


def run_developed_fatigue(
    output: str | Path, *, target_events: int = 4, phase_count: int = 24,
    maximum_cycles: int = 80,
) -> dict[str, Any]:
    if target_events < 2:
        raise ValueError("developed path requires at least two advances")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    clear_pending_geometry_events()
    waveform = FatigueWaveform(Kmax=20.0e6, R=0.1, frequency_Hz=10.0)
    split = max(1, target_events // 2)
    engine = build_developed_engine()
    prefix, prefix_events, prefix_cycle = _run_until(
        engine, waveform, start_cycle=0, target_events=split,
        phase_count=phase_count, maximum_cycles=maximum_cycles,
    )
    checkpoint = root / "developed_checkpoint"
    previous = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint)
    try:
        write_checkpoint(
            engine, temperature_K=1800.0, reason="r3_developed_midpoint",
            metadata={"cycle": prefix_cycle, "phase_index": 0, "model": MODEL_ID},
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = previous
    checkpoint_state = serialize_active_state(engine)
    direct = copy.deepcopy(engine)
    restarted = build_developed_engine()
    _restore_geometry(restarted, checkpoint)
    restore_checkpoint(restarted, checkpoint)
    checkpoint_exact = _same_active_state(
        checkpoint_state, serialize_active_state(restarted)
    )
    direct_history, direct_events, terminal_cycle = _run_until(
        direct, waveform, start_cycle=prefix_cycle, target_events=target_events,
        phase_count=phase_count, maximum_cycles=maximum_cycles,
    )
    clear_pending_geometry_events()
    restarted_history, restarted_events, restarted_cycle = _run_until(
        restarted, waveform, start_cycle=prefix_cycle, target_events=target_events,
        phase_count=phase_count, maximum_cycles=maximum_cycles,
    )
    continuation_exact = _same_active_state(
        serialize_active_state(direct), serialize_active_state(restarted)
    )
    payload = {
        "schema": MODEL_ID,
        "event_strategy": "deterministic unit-mark fixed reward",
        "waveform": {
            "Kmax_Pa_sqrt_m": waveform.Kmax, "R": waveform.R,
            "frequency_Hz": waveform.frequency_Hz, "phase_count": phase_count,
        },
        "temperature_K": 1800.0,
        "prefix_history": prefix,
        "prefix_events": prefix_events,
        "restarted_history": restarted_history,
        "restarted_events": restarted_events,
        "direct_tail_events": direct_events,
        "a_N": [
            {"cycle": row["cycle"], "a_m": row["a_m"]}
            for row in [*prefix, *restarted_history]
        ],
        "target_events": target_events,
        "terminal_cycle": restarted_cycle,
        "terminal_extension_m": float(restarted.checkpoint_advance_total_m),
        "all_waveform_time_consumed": all(
            row["unused_cycle_time_s"] == 0.0
            for row in [*prefix, *restarted_history]
        ),
        "checkpoint_exact": checkpoint_exact,
        "restart_continuation_exact": continuation_exact,
        "direct_terminal_cycle": terminal_cycle,
        "no_second_mpz_translation": all(
            row["outer_geometry_translation_m"] == 0.0
            and abs(row["cumulative_mpz_checkpoint_error_m"]) <= 2.0e-14
            for row in [*prefix_events, *restarted_events]
        ),
    }
    (root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


__all__ = [
    "MODEL_ID", "advance_explicit_cycle", "advance_mean_cycle_complete",
    "build_developed_engine", "run_developed_fatigue",
]
