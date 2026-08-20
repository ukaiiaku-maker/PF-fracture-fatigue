"""Event-to-event VHCF accelerator for intrinsic reverse-glide v7.

The target use case is 10^12--10^14 physical cycles, not a succession of short
cycle demonstrations.  Exact adaptive v7 cycles resolve the transient.  Once the
cycle-end state is predictable, power-of-two projective blocks grow aggressively
and are accepted only through cycle-level step doubling.  The stochastic
cleavage clock is the unchanged exponential first-passage clock.

If a proposed block would cross the current stochastic threshold, the block is
never committed.  Its stride is halved until the crossing is confined to a
small exact-cycle guard; the event is then localized inside the crossing cycle
with the adaptive v7 phase localizer.  Crack advance uses the unchanged v9.14
event-length law, after which projective readiness is reset because geometry has
changed.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture import fatigue_v914 as base
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_intrinsic_reverse_glide_v7 import IntrinsicReverseGlideState
from v914_v7_adaptive_block_accelerator import AdaptiveBlockControls
from v914_v7_adaptive_projective_accelerator import readiness_prediction
from v914_v7_cycle_map import advance_v7_cycle, cycle_endpoint_summary
from v914_v7_rate_separated_dmd import (
    MODEL_ID as DMD_MODEL_ID,
    evaluate_rate_separated_dmd_block,
)
from v914_v7_event_localizer import (
    LOCALIZER_ID,
    advance_v7_phase_span,
    localize_v7_action_in_cycle,
)


ENGINE_ID = "v9.14_v7_vhcf_event_to_event_rate_separated_dmd_v2"
CHECKPOINT_SCHEMA = "v914_v7_vhcf_event_to_event_checkpoint_v1"


@dataclass(frozen=True)
class VHCFRunControls:
    maximum_physical_cycles: int = 10**14
    maximum_cycle_map_evaluations: int = 4096
    heartbeat_cycle_map_evaluations: int = 12
    event_guard_stride: int = 4
    phase_localization_tolerance: float = 1.0e-13

    def validate(self) -> None:
        if int(self.maximum_physical_cycles) < 1:
            raise ValueError("maximum_physical_cycles must be positive")
        if int(self.maximum_cycle_map_evaluations) < 1:
            raise ValueError("maximum_cycle_map_evaluations must be positive")
        if int(self.heartbeat_cycle_map_evaluations) < 1:
            raise ValueError("heartbeat_cycle_map_evaluations must be positive")
        guard = int(self.event_guard_stride)
        if guard < 1 or guard & (guard - 1):
            raise ValueError("event_guard_stride must be a positive power of two")
        tol = float(self.phase_localization_tolerance)
        if not math.isfinite(tol) or tol <= 0.0:
            raise ValueError("phase_localization_tolerance must be positive and finite")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    complete = copy.deepcopy(payload)
    complete["integrity_sha256"] = _canonical_digest(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as stream:
        json.dump(complete, stream, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    checksum = payload.pop("integrity_sha256", None)
    if checksum != _canonical_digest(payload):
        raise ValueError("v7 VHCF checkpoint integrity mismatch")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("v7 VHCF checkpoint schema mismatch")
    return payload


def _largest_power_of_two_at_most(value: int) -> int:
    n = int(value)
    if n < 1:
        return 0
    return 1 << (n.bit_length() - 1)


def _heartbeat(
    *,
    cycle: int,
    stride: int,
    evaluations: int,
    action: float,
    threshold: float,
    state,
    events: list[dict[str, Any]],
    controls: VHCFRunControls,
    mode: str,
) -> None:
    if evaluations % int(controls.heartbeat_cycle_map_evaluations) != 0:
        return
    print(
        "  v7 VHCF heartbeat: "
        f"cycle={cycle} stride={stride} maps={evaluations} mode={mode} "
        f"clock={action:.6g}/{threshold:.6g} events={len(events)} "
        f"extension_um={1.0e6*float(state.extension_m):.6g}",
        flush=True,
    )


def _event_record(
    state,
    loading,
    *,
    event_index: int,
    cycle_index: int,
    phase: float,
    threshold: float,
    physical_action: float,
    advance_m: float,
    interval_start_cycles: float,
) -> dict[str, Any]:
    cumulative = float(cycle_index) + float(phase)
    record = {
        "event_index": int(event_index),
        "cumulative_cycles": cumulative,
        "cycle_index": int(cycle_index),
        "cycle_phase": float(phase),
        "interval_cycles": cumulative - float(interval_start_cycles),
        "K_MPa_sqrt_m": float(loading.K_at_phase(float(phase))),
        "threshold_action": float(threshold),
        "physical_hazard_action": float(physical_action),
        "committed_advance_m": float(advance_m),
        "cumulative_extension_m": float(state.extension_m),
        "shielding_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
        "tip_radius_m": float(state.tip_radius_m()),
        "mobile_total_m2": float(np.sum(state.mobile_m2)),
        "retained_total_m2": float(np.sum(state.retained_m2)),
    }
    record.update(state.reversibility_diagnostics())
    return record


def _resolve_exact_cycle_with_events(
    state,
    loading,
    cycle_controls: AdaptiveFeedbackControls,
    numerics,
    rng,
    *,
    cycle_index: int,
    action: float,
    threshold: float,
    interval_start_cycles: float,
    events: list[dict[str, Any]],
    phase_tolerance: float,
) -> tuple[Any, float, float, float, bool, float]:
    """Resolve one complete physical cycle and every event inside it.

    Returns ``state, action, threshold, interval_start, event_occurred,
    integrated_cycle_hazard`` at the next cycle boundary or growth target.
    """
    phase = 0.0
    committed = copy.deepcopy(state)
    event_occurred = False
    integrated_cycle_hazard = 0.0

    while phase < 1.0 - 1.0e-15:
        trial, increment = advance_v7_phase_span(
            committed, loading, cycle_controls, phase, 1.0
        )
        if not math.isfinite(increment) or increment < 0.0:
            raise FloatingPointError("invalid exact-cycle VHCF hazard")

        if action + increment < threshold:
            committed = trial
            action += increment
            integrated_cycle_hazard += increment
            phase = 1.0
            break

        required = max(float(threshold) - float(action), 0.0)
        if required <= 0.0:
            required = max(float(threshold), 1.0e-300)
        localized, event_phase, used = localize_v7_action_in_cycle(
            committed,
            loading,
            cycle_controls,
            required,
            start_phase=phase,
            phase_tolerance=phase_tolerance,
        )
        integrated_cycle_hazard += used
        physical_action = action + used
        factor = base._event_length_factor(float(threshold), numerics)
        proposed = float(numerics.base_event_length_m) * float(factor)
        before = float(localized.extension_m)
        localized.translate_tip(proposed)
        advance = float(localized.extension_m) - before
        event = _event_record(
            localized,
            loading,
            event_index=len(events),
            cycle_index=cycle_index,
            phase=event_phase,
            threshold=threshold,
            physical_action=physical_action,
            advance_m=advance,
            interval_start_cycles=interval_start_cycles,
        )
        event["event_length_factor"] = float(factor)
        events.append(event)
        interval_start_cycles = float(cycle_index) + float(event_phase)
        event_occurred = True

        committed = localized
        phase = float(event_phase)
        action = 0.0
        threshold = float(rng.exponential())
        if float(committed.extension_m) >= float(numerics.target_extension_m):
            break

    return (
        committed,
        float(action),
        float(threshold),
        float(interval_start_cycles),
        bool(event_occurred),
        float(integrated_cycle_hazard),
    )


def run_v7_vhcf_event_to_event(
    candidate,
    physics,
    loading,
    *,
    seed: int,
    cycle_controls: AdaptiveFeedbackControls,
    block_controls: AdaptiveBlockControls,
    run_controls: VHCFRunControls,
    numerics=base.FatigueNumerics(),
    checkpoint_path: str | Path | None = None,
    restart_from: str | Path | None = None,
    contract_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance v7 to a VHCF horizon or crack-extension target.

    ``maximum_physical_cycles`` may be 10^12, 10^14, or another integer horizon.
    Runtime is controlled by the number of accepted/rejected block evaluations,
    not by looping over every physical cycle.
    """
    loading.validate()
    physics.validate()
    cycle_controls.validate()
    block_controls.validate()
    run_controls.validate()
    numerics.validate()

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    metadata = dict(contract_metadata or {})
    identity_contract = {
        "engine_id": ENGINE_ID,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": base._canonical_digest(asdict(candidate)),
        "physics_sha256": base._canonical_digest(asdict(physics)),
        "loading": asdict(loading),
        "numerics": asdict(numerics),
        "seed": int(seed),
        "cycle_controls": asdict(cycle_controls),
        "block_controls": asdict(block_controls),
        "localizer_id": LOCALIZER_ID,
        **metadata,
    }

    if restart_from:
        saved = read_checkpoint(restart_from)
        if saved["identity_contract"] != identity_contract:
            raise ValueError("v7 VHCF checkpoint identity contract mismatch")
        baseline = base._state_from_checkpoint(candidate, physics, saved["state"])
        state = IntrinsicReverseGlideState.from_existing_state(
            baseline, saved.get("state_extension")
        )
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng_state"]
        controller = saved["controller"]
        cycle = int(controller["cycle"])
        action = float(controller["action"])
        threshold = float(controller["threshold"])
        interval_start_cycles = float(controller["interval_start_cycles"])
        events = list(saved.get("events", []))
        anchor_history = list(saved.get("anchor_history", []))
        block_history = list(saved.get("block_history", []))
        total_maps = int(controller.get("total_cycle_map_evaluations", 0))
        accepted_maps = int(controller.get("accepted_cycle_map_evaluations", 0))
        sequence = int(saved.get("sequence", 0))
        restart_count = int(saved.get("restart_count", 0)) + 1
    else:
        state = IntrinsicReverseGlideState(candidate, physics)
        rng = np.random.default_rng(int(seed))
        cycle = 0
        action = 0.0
        threshold = float(rng.exponential())
        interval_start_cycles = 0.0
        events: list[dict[str, Any]] = []
        anchor_history: list[dict[str, Any]] = []
        block_history: list[dict[str, Any]] = []
        total_maps = 0
        accepted_maps = 0
        sequence = 0
        restart_count = 0

    # Predictor history is intentionally rebuilt after restart and after every
    # crack event.  This costs only a handful of exact cycles and prevents stale
    # projective secants from crossing a geometry change.
    resolved_states: list[tuple[int, Any, float]] = []
    readiness_streak = 0
    promoted = False
    stride = int(block_controls.initial_block_stride)
    promotion_count = 0
    rejected_blocks = 0
    event_guard_halvings = 0
    fallback_exact_cycles = 0
    accepted_blocks = 0
    maximum_accepted_stride = 0
    status = "running"

    def save(reason: str) -> None:
        nonlocal sequence
        if checkpoint is None:
            return
        sequence += 1
        _atomic_json(
            checkpoint,
            {
                "schema": CHECKPOINT_SCHEMA,
                "identity_contract": identity_contract,
                "sequence": sequence,
                "reason": str(reason),
                "restart_count": restart_count,
                "state": base._state_to_checkpoint(state),
                "state_extension": state.reversible_checkpoint_payload(),
                "rng_state": copy.deepcopy(rng.bit_generator.state),
                "controller": {
                    "cycle": int(cycle),
                    "action": float(action),
                    "threshold": float(threshold),
                    "interval_start_cycles": float(interval_start_cycles),
                    "total_cycle_map_evaluations": int(total_maps),
                    "accepted_cycle_map_evaluations": int(accepted_maps),
                },
                "events": events,
                "anchor_history": anchor_history,
                "block_history": block_history,
            },
        )

    horizon = int(run_controls.maximum_physical_cycles)
    max_maps = int(run_controls.maximum_cycle_map_evaluations)

    while cycle < horizon and float(state.extension_m) < float(numerics.target_extension_m):
        if total_maps >= max_maps:
            status = "work_budget_exhausted"
            save(status)
            break

        if not promoted:
            start_state = copy.deepcopy(state)
            trial_state, trial_hazard, telemetry = advance_v7_cycle(
                start_state, loading, cycle_controls
            )
            total_maps += 1

            if action + float(trial_hazard) >= threshold:
                # Discard the trial endpoint.  Re-resolve the physical cycle with
                # threshold-aware adaptive localization and unchanged event law.
                (
                    state,
                    action,
                    threshold,
                    interval_start_cycles,
                    event_occurred,
                    cycle_hazard,
                ) = _resolve_exact_cycle_with_events(
                    start_state,
                    loading,
                    cycle_controls,
                    numerics,
                    rng,
                    cycle_index=cycle,
                    action=action,
                    threshold=threshold,
                    interval_start_cycles=interval_start_cycles,
                    events=events,
                    phase_tolerance=float(run_controls.phase_localization_tolerance),
                )
                cycle += 1
                accepted_maps += 1
                fallback_exact_cycles += 1
                summary = cycle_endpoint_summary(state, cycle_hazard, cycle)
                summary["resolution"] = "exact_event_cycle"
                summary["hazard_clock_action"] = float(action)
                summary["hazard_threshold_action"] = float(threshold)
                anchor_history.append(summary)
                resolved_states = []
                readiness_streak = 0
                promoted = False
                stride = int(block_controls.initial_block_stride)
                save("event_cycle")
                if float(state.extension_m) >= float(numerics.target_extension_m):
                    status = "growth_target_reached"
                    break
                continue

            state = trial_state
            action += float(trial_hazard)
            cycle += 1
            accepted_maps += 1
            summary = cycle_endpoint_summary(state, trial_hazard, cycle)
            summary["resolution"] = "exact_readiness_cycle"
            summary["hazard_clock_action"] = float(action)
            summary["hazard_threshold_action"] = float(threshold)
            anchor_history.append(summary)
            resolved_states.append((cycle, copy.deepcopy(state), float(trial_hazard)))
            if len(resolved_states) > 3:
                resolved_states = resolved_states[-3:]

            if len(resolved_states) == 3:
                c0, s0, _ = resolved_states[0]
                c1, s1, _ = resolved_states[1]
                c2, s2, _ = resolved_states[2]
                readiness = readiness_prediction(
                    s0,
                    s1,
                    s2,
                    frequency_Hz=float(loading.frequency_Hz),
                    cycle_index=c2,
                )
                passed = bool(
                    c1 == c0 + 1
                    and c2 == c1 + 1
                    and readiness["maximum_relative_error"]
                    <= float(block_controls.readiness_relative_tolerance)
                    and readiness["projection_constraint_correction"]
                    <= float(block_controls.max_projection_constraint_correction)
                )
                readiness_streak = readiness_streak + 1 if passed else 0

            if (
                cycle >= int(block_controls.minimum_exact_cycles)
                and readiness_streak >= int(block_controls.readiness_consecutive_passes)
                and horizon - cycle >= int(block_controls.initial_block_stride)
            ):
                promoted = True
                promotion_count += 1
                stride = int(block_controls.initial_block_stride)
            _heartbeat(
                cycle=cycle,
                stride=stride,
                evaluations=total_maps,
                action=action,
                threshold=threshold,
                state=state,
                events=events,
                controls=run_controls,
                mode="exact",
            )
            if total_maps % max(int(run_controls.heartbeat_cycle_map_evaluations), 1) == 0:
                save("exact_progress")
            continue

        remaining = horizon - cycle
        if remaining < int(block_controls.initial_block_stride):
            promoted = False
            readiness_streak = 0
            continue
        if len(resolved_states) < 2:
            promoted = False
            readiness_streak = 0
            continue

        previous_cycle, previous_state, _ = resolved_states[-2]
        current_cycle, current_state, current_hazard = resolved_states[-1]
        if current_cycle != cycle:
            raise RuntimeError("v7 VHCF accepted state and anchor cycle disagree")

        candidate_stride = min(
            int(stride),
            int(block_controls.maximum_block_stride),
            _largest_power_of_two_at_most(remaining),
        )
        if candidate_stride < int(block_controls.initial_block_stride):
            promoted = False
            readiness_streak = 0
            continue

        trial = evaluate_rate_separated_dmd_block(
            current_state,
            block_stride=candidate_stride,
            loading=loading,
            cycle_controls=cycle_controls,
            state_rtol=float(block_controls.block_state_rtol),
            hazard_rtol=float(block_controls.block_hazard_rtol),
        )
        maps_used = int(trial.get("cycle_map_evaluations", 0))
        total_maps += maps_used
        state_err = float(trial["endpoint_state_error"]["maximum_relative_error"])
        hazard_err = float(trial["block_hazard_relative_error"])
        correction = float(trial["maximum_projection_constraint_correction"])
        numerical_pass = bool(trial.get("numerical_pass", False) and
            state_err <= float(block_controls.block_state_rtol)
            and hazard_err <= float(block_controls.block_hazard_rtol)
            and correction <= float(block_controls.max_projection_constraint_correction)
        )
        block_action = max(float(trial["fine_block_hazard_action"]), 0.0)
        upper_block_action = max(
            float(trial.get("upper_block_hazard_action", block_action)), block_action
        )
        threshold_crossed = action + upper_block_action >= threshold

        block_history.append(
            {
                "start_cycle": int(cycle),
                "end_cycle": int(cycle + candidate_stride),
                "block_stride": int(candidate_stride),
                "numerical_pass": numerical_pass,
                "threshold_crossed": threshold_crossed,
                "state_error": state_err,
                "hazard_error": hazard_err,
                "projection_correction": correction,
                "block_hazard_action": block_action,
                "upper_block_hazard_action": upper_block_action,
                "hazard_clock_action_before": float(action),
                "hazard_threshold_action": float(threshold),
            }
        )

        if not numerical_pass:
            rejected_blocks += 1
            if candidate_stride > int(block_controls.initial_block_stride):
                stride = max(
                    int(block_controls.initial_block_stride), candidate_stride // 2
                )
                continue
            promoted = False
            readiness_streak = 0
            fallback_exact_cycles += 1
            stride = int(block_controls.initial_block_stride)
            continue

        if threshold_crossed:
            # Never commit a projective state across a stochastic event.  Reduce
            # only the event bracket.  This costs O(log N) block trials even when
            # the event lies billions or trillions of cycles away.
            if candidate_stride > int(run_controls.event_guard_stride):
                event_guard_halvings += 1
                stride = max(
                    int(run_controls.event_guard_stride), candidate_stride // 2
                )
                continue

            # The crossing is now confined to a handful of cycles.  Switch to
            # exact cycles; the normal exact branch localizes the crossing phase.
            promoted = False
            readiness_streak = 0
            stride = int(block_controls.initial_block_stride)
            continue

        # Accepted fine path.  The private coarse path is discarded.
        accepted_blocks += 1
        accepted_maps += maps_used
        maximum_accepted_stride = max(maximum_accepted_stride, candidate_stride)
        action += block_action
        state = copy.deepcopy(trial["fine_end_state"])
        mid_cycle = int(cycle + candidate_stride // 2)
        end_cycle = int(cycle + candidate_stride)
        mid_state = copy.deepcopy(trial["fine_mid_state"])
        mid_hazard = float(trial["fine_mid_hazard"])
        end_hazard = float(trial["fine_end_hazard"])
        cycle = end_cycle
        resolved_states = [
            (mid_cycle, mid_state, mid_hazard),
            (end_cycle, copy.deepcopy(state), end_hazard),
        ]
        summary = cycle_endpoint_summary(state, end_hazard, cycle)
        summary["resolution"] = "accepted_adaptive_block_end"
        summary["block_stride"] = int(candidate_stride)
        summary["block_hazard_action"] = block_action
        summary["hazard_clock_action"] = float(action)
        summary["hazard_threshold_action"] = float(threshold)
        anchor_history.append(summary)
        stride = min(
            candidate_stride * 2, int(block_controls.maximum_block_stride)
        )
        save("accepted_block")
        _heartbeat(
            cycle=cycle,
            stride=stride,
            evaluations=total_maps,
            action=action,
            threshold=threshold,
            state=state,
            events=events,
            controls=run_controls,
            mode="block",
        )

    if status == "running":
        if float(state.extension_m) >= float(numerics.target_extension_m):
            status = "growth_target_reached"
        elif cycle >= horizon:
            status = "cycle_horizon_reached"
        else:
            status = "stopped"
    save(status)

    return {
        "schema": "v914_v7_vhcf_event_to_event_result_v1",
        "engine_id": ENGINE_ID,
        "dmd_model_id": DMD_MODEL_ID,
        "localizer_id": LOCALIZER_ID,
        "status": status,
        "maximum_physical_cycles": horizon,
        "completed_physical_cycles": int(cycle),
        "final_time_s": float(state.time_s),
        "cumulative_extension_m": float(state.extension_m),
        "event_count": len(events),
        "events": events,
        "anchor_history": anchor_history,
        "block_history": block_history,
        "hazard_clock_action": float(action),
        "hazard_threshold_action": float(threshold),
        "total_cycle_map_evaluations": int(total_maps),
        "accepted_cycle_map_evaluations": int(accepted_maps),
        "accepted_block_count": int(accepted_blocks),
        "rejected_block_count": int(rejected_blocks),
        "event_guard_halvings": int(event_guard_halvings),
        "fallback_exact_cycles": int(fallback_exact_cycles),
        "promotion_count": int(promotion_count),
        "maximum_accepted_stride": int(maximum_accepted_stride),
        "physical_cycles_per_total_cycle_map_evaluation": (
            float(cycle) / max(float(total_maps), 1.0)
        ),
        "final_reversibility": state.reversibility_diagnostics(),
        "identity_contract": identity_contract,
        "restart_count": int(restart_count),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DMD_MODEL_ID",
    "ENGINE_ID",
    "VHCFRunControls",
    "read_checkpoint",
    "run_v7_vhcf_event_to_event",
]
