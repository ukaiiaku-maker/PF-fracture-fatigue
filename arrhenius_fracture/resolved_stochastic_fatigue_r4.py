"""Phase-exact stochastic recurrent fatigue on the restored current engine."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .current_coupled_execution_r4 import complete_state
from .developed_fatigue_r3 import (
    _restore_geometry, advance_mean_cycle_complete, build_developed_engine,
)
from .fatigue_v1 import FatigueWaveform
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from .reduced_persistent_reference_r2 import ReducedPersistentReversibleTipEngine
from .stochastic_avalanche_tip import (
    clear_pending_geometry_events, pop_pending_geometry_event,
)


MODEL_ID = "r4.phase-exact-stochastic-recurrent-fatigue/1"


@dataclass
class PhaseCursor:
    cycle: int = 0
    phase_index: int = 0
    remaining_phase_s: float = 0.0


def _build_stochastic_engine():
    engine_type = ReducedPersistentReversibleTipEngine
    old_hazard = copy.deepcopy(engine_type._hazard_config_default)
    old_avalanche = copy.deepcopy(engine_type._avalanche_config_default)
    try:
        engine_type.configure_hazard(mode="exponential", seed=1733)
        engine_type.configure_avalanche(
            mode="threshold_scaled", minimum_factor=0.5,
            maximum_factor=4.0, geometry_subsegment_fraction=0.1,
        )
        return build_developed_engine()
    finally:
        engine_type._hazard_config_default = old_hazard
        engine_type._avalanche_config_default = old_avalanche


def _phase_K(waveform: FatigueWaveform, phase_count: int, index: int) -> float:
    phase = (float(index) + 0.5) * 2.0 * math.pi / phase_count
    return float(waveform.K_phase(np.asarray([phase], dtype=float))[0])


def _advance_one_subcall(engine: Any, cursor: PhaseCursor,
                         waveform: FatigueWaveform, phase_count: int):
    duration = waveform.period_s / phase_count
    if cursor.remaining_phase_s <= 0.0:
        cursor.remaining_phase_s = duration
    K = _phase_K(waveform, phase_count, cursor.phase_index)
    from .reduced_persistent_reference_r2 import advance_signed
    result = advance_signed(
        engine, K_Pa_sqrt_m=K, temperature_K=1800.0,
        dt_s=cursor.remaining_phase_s,
    )
    consumed = float(result["kinetic_dt_consumed_s"])
    unused = float(result["kinetic_dt_unused_s"])
    if consumed <= 0.0 or unused < 0.0:
        raise RuntimeError("resolved fatigue subcall did not make valid progress")
    event = pop_pending_geometry_event() if result["fired"] else None
    if bool(result["fired"]) != bool(event is not None):
        raise RuntimeError("resolved fatigue event/descriptor ownership mismatch")
    row = {
        "cycle": cursor.cycle + 1, "phase_index": cursor.phase_index,
        "phase_midpoint_rad": (
            (cursor.phase_index + 0.5) * 2.0 * math.pi / phase_count
        ),
        "K_Pa_sqrt_m": K, "requested_s": cursor.remaining_phase_s,
        "consumed_s": consumed, "unused_s": unused,
        "fired": bool(result["fired"]), "event": event,
        "threshold_before_completion": float(
            result.get("hazard_threshold_completed_action", 0.0)
        ),
        "next_pending_threshold": float(engine.hazard_threshold_action),
        "event_index": int(engine.hazard_event_index),
        "checkpoint_count": int(engine.n_adv),
        "extension_m": float(engine.checkpoint_advance_total_m),
        "gross_source_slip": float(engine.mpz.cumulative_gross_source_activity),
        "gross_return_slip": float(engine.mpz.cumulative_gross_return_activity),
        "net_source_slip": float(engine.mpz.local_slip_count()),
    }
    cursor.remaining_phase_s = unused
    tolerance = max(64.0 * np.spacing(duration), 1.0e-15)
    if cursor.remaining_phase_s <= tolerance:
        cursor.remaining_phase_s = 0.0
        cursor.phase_index += 1
        if cursor.phase_index == phase_count:
            cursor.phase_index = 0
            cursor.cycle += 1
    return row


def _continue(engine: Any, cursor: PhaseCursor, waveform: FatigueWaveform,
              phase_count: int, target_events: int, maximum_cycles: int,
              target_extension_m: float | None = None):
    rows = []
    events = []
    target_extension = (None if target_extension_m is None
                        else float(target_extension_m))
    def incomplete():
        event_short = engine.n_adv < target_events
        extension_short = (
            target_extension is not None
            and float(engine.checkpoint_advance_total_m) < target_extension
        )
        return event_short or extension_short
    while incomplete() and cursor.cycle < maximum_cycles:
        row = _advance_one_subcall(engine, cursor, waveform, phase_count)
        rows.append(row)
        if row["fired"]:
            events.append(copy.deepcopy(row))
    if incomplete():
        raise RuntimeError(
            f"resolved stochastic path reached only {engine.n_adv} events "
            f"and {engine.checkpoint_advance_total_m:.9g} m by cycle {cursor.cycle}"
        )
    return rows, events


def execute_resolved_stochastic_fatigue_r4(
    output_directory: str | Path, *, target_events: int = 4,
    phase_count: int = 24, maximum_cycles: int = 120,
    target_extension_m: float = 20.0e-6,
) -> dict[str, Any]:
    root = Path(output_directory).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    clear_pending_geometry_events()
    waveform = FatigueWaveform(Kmax=20.0e6, R=0.1, frequency_Hz=10.0)
    engine = _build_stochastic_engine()
    cursor = PhaseCursor()
    prefix_rows, prefix_events = _continue(
        engine, cursor, waveform, phase_count,
        target_events=max(1, target_events // 2), maximum_cycles=maximum_cycles,
    )
    checkpoint_cursor = copy.deepcopy(cursor)
    checkpoint_state = complete_state(engine)
    checkpoint = root / "resolved_checkpoint"
    previous = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint)
    try:
        write_checkpoint(
            engine, waveform=waveform, temperature_K=1800.0,
            reason="r4_resolved_stochastic_midphase_event",
            metadata={"phase_cursor": asdict(checkpoint_cursor)},
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = previous
    direct = copy.deepcopy(engine)
    restarted = _build_stochastic_engine()
    _restore_geometry(restarted, checkpoint)
    restore_checkpoint(restarted, checkpoint)
    restore_exact = checkpoint_state == complete_state(restarted)
    def causal_state(value):
        state = complete_state(value)
        # The checkpoint intentionally omits recomputable last-call diagnostic
        # caches; all owning arrays, ledgers, clocks, RNG and geometry remain.
        state.pop("source_diagnostics", None)
        return state
    restore_owned_state_exact = causal_state(engine) == causal_state(restarted)
    direct_cursor = copy.deepcopy(checkpoint_cursor)
    restarted_cursor = copy.deepcopy(checkpoint_cursor)
    direct_rows, direct_events = _continue(
        direct, direct_cursor, waveform, phase_count,
        target_events=target_events, maximum_cycles=maximum_cycles,
        target_extension_m=target_extension_m,
    )
    clear_pending_geometry_events()
    restarted_rows, restarted_events = _continue(
        restarted, restarted_cursor, waveform, phase_count,
        target_events=target_events, maximum_cycles=maximum_cycles,
        target_extension_m=target_extension_m,
    )
    continuation_exact = (
        causal_state(direct) == causal_state(restarted)
        and asdict(direct_cursor) == asdict(restarted_cursor)
        and direct_events == restarted_events
    )

    # Affordable approximation check from one common virgin stochastic state.
    parent = _build_stochastic_engine()
    explicit = copy.deepcopy(parent)
    averaged = copy.deepcopy(parent)
    explicit_cursor = PhaseCursor()
    explicit_rows = []
    while explicit_cursor.cycle < 1:
        explicit_rows.append(_advance_one_subcall(
            explicit, explicit_cursor, waveform, phase_count
        ))
    clear_pending_geometry_events()
    averaged_result = advance_mean_cycle_complete(
        averaged, waveform, temperature_K=1800.0, phase_count=96,
    )
    approximation = {
        "common_initial_complete_state": complete_state(parent),
        "resolved_phase_count": phase_count,
        "mean_cycle_quadrature_count": 96,
        "resolved_terminal": complete_state(explicit),
        "mean_cycle_terminal": complete_state(averaged),
        "mean_cycle_result": averaged_result,
        "resolved_events": [row for row in explicit_rows if row["fired"]],
        "same_stochastic_mark_at_start": True,
        "complete_states_equal_claimed": False,
    }
    all_rows = [*prefix_rows, *restarted_rows]
    all_events = [*prefix_events, *restarted_events]
    result = {
        "schema": MODEL_ID,
        "strategy": {
            "integration": "PHASE_RESOLVED_NATIVE_STATE_EVOLUTION",
            "hazard_mark": "UNIT_EXPONENTIAL_SEED_1733",
            "geometry_reward": "THRESHOLD_SCALED_BOUNDED_MEAN_PRESERVING",
            "mpz_translation": "CONTINUOUS_EVENT_CONDITIONED_OUTER_COMMIT",
            "temperature_K": 1800.0,
            "waveform": {
                "Kmax_Pa_sqrt_m": waveform.Kmax, "R": waveform.R,
                "frequency_Hz": waveform.frequency_Hz,
                "phase_count": phase_count,
            },
        },
        "prefix_rows": prefix_rows, "prefix_events": prefix_events,
        "restarted_rows": restarted_rows, "restarted_events": restarted_events,
        "direct_tail_events": direct_events,
        "a_N": [
            {"cycle": row["cycle"], "a_m": row["extension_m"]}
            for row in all_rows if row["phase_index"] == phase_count - 1
            or row["fired"]
        ],
        "restart": {
            "checkpoint_cursor": asdict(checkpoint_cursor),
            "checkpoint_owned_state_exact": restore_owned_state_exact,
            "checkpoint_including_recomputable_diagnostics_exact": restore_exact,
            "continuation_complete_state_phase_and_events_exact": continuation_exact,
        },
        "mean_cycle_comparison": approximation,
        "minimum_target_events": target_events,
        "target_extension_m": target_extension_m,
        "terminal_cycle": restarted_cursor.cycle,
        "terminal_extension_m": float(restarted.checkpoint_advance_total_m),
        "events": all_events,
        "multiple_events_same_cycle_observed": len({
            row["cycle"] for row in all_events
        }) < len(all_events),
        "pending_mark_preserved": restore_owned_state_exact,
        "remaining_phase_continued_after_event": any(
            row["unused_s"] > 0.0 for row in all_events
        ),
    }
    result["passed"] = bool(
        len(all_events) >= target_events
        and result["terminal_extension_m"] >= target_extension_m
        and math.isclose(result["terminal_extension_m"], sum(
            float(row["event"]["event_advance_m"]) for row in all_events
        ), rel_tol=0.0, abs_tol=2.0e-14)
        and restore_owned_state_exact and continuation_exact
        and result["remaining_phase_continued_after_event"]
    )
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


__all__ = ["MODEL_ID", "PhaseCursor", "execute_resolved_stochastic_fatigue_r4"]
