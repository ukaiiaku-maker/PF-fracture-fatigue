#!/usr/bin/env python3
"""Complete restartable, chronological QSTE340TM active-state calibration.

This is an additive descendant of the f91c2fd1 restored-runtime fit.  It uses
the production v10.2.30 atomic active-state checkpoint and adds only the outer
experimental-history cursor required to resume a material at an observation
and load-interpolation substep boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import MethodType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import _NativePhaseController
from arrhenius_fracture import persistent_site_high_cycle_checkpoint_v10230 as native_checkpoint
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint,
    write_checkpoint,
)
from arrhenius_fracture.persistent_site_high_cycle_engine_v10230 import (
    integrate_state_coupled_waveform,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    ledger_delta,
)
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.qste340tm_restored_binding import (
    MODEL_ID,
    OpeningCoordinates,
    PlasticCoordinates,
    TEMPERATURE_K,
    build_engine,
    install_opening,
    load_parent_binding,
    material_binding,
    opening_surface,
)
from arrhenius_fracture.reduced_persistent_reference_r2 import runtime_receipt
from scripts.refit_qste340tm_restored_native import (
    ALT_REGISTRY,
    DATA,
    FREQUENCY_HZ,
    OUT as LEGACY_OUT,
    PARENT_CANDIDATE,
    PARENT_REGISTRY,
    R_VALUES,
    _state_row,
    _stress_trace,
    atomic_json,
    build_records,
    fit_opening,
    frozen_stress,
    metrics,
    predict,
    sha256,
)


OUT = DATA / "model_conditional/stateful_exp_floor_capacity_policy/restored_active_completion"
SCHEMA = "qste340tm-restored-active-chronological/2"
HISTORY_MODE = "every_observation_native_state_nearest_whole_cycle"
CYCLE_SCHEDULE_POLICY = "nearest_whole_cycle_half_up_preserve_mapped_cycles_v1"
NATIVE_CYCLE_TOLERANCE = 1.0e-6


def prepare_observation_schedule(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve source order while assigning explicit whole-cycle native targets.

    In particular, R=0.5 has fractionally mapped N values.  The schedule uses
    whole cycles to preserve explicit phase ownership and rounds N once, with
    at most 0.5 cycle displacement.  The original mapped coordinate remains
    available for review.
    This is a time-coordinate approximation, not a fitted or interpolated state.
    """
    scheduled = []
    seen = set()
    previous: dict[float, tuple[float, int]] = {}
    for record in records:
        row = dict(record)
        r_value = float(row["R"])
        order = int(row["source_order"])
        key = (r_value, order)
        if r_value not in R_VALUES or key in seen:
            raise ValueError(f"invalid or duplicate source observation {key}")
        mapped = float(row.get("mapped_cycles", row["cycles"]))
        if not math.isfinite(mapped) or mapped < 0.0:
            raise ValueError(f"invalid mapped cycle coordinate for {key}: {mapped}")
        target = int(math.floor(mapped + 0.5))
        if r_value in previous:
            prior_mapped, prior_target = previous[r_value]
            if mapped <= prior_mapped or target <= prior_target:
                raise ValueError(
                    f"whole-cycle scheduling does not preserve distinct chronological "
                    f"observations at {key}; resolve the measured-time mapping explicitly"
                )
        delta = float(target) - mapped
        if abs(delta) > 0.5 + 1.0e-10:
            raise ValueError(f"cycle rounding exceeds half a cycle for {key}")
        row.update({
            "mapped_cycles": mapped,
            "cycles": float(target),
            "cycle_rounding_delta": delta,
            "cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
        })
        scheduled.append(row)
        seen.add(key)
        previous[r_value] = (mapped, target)
    return scheduled


def assert_native_cycle_target(engine: Any, target_cycle: float,
                               frequency_Hz: float) -> float:
    """Verify the actual native cursor before declaring an observation evolved."""
    native_cycle = float(engine.t) * float(frequency_Hz)
    if not math.isfinite(native_cycle) or not math.isclose(
        native_cycle, float(target_cycle), rel_tol=0.0,
        abs_tol=NATIVE_CYCLE_TOLERANCE,
    ):
        raise RuntimeError(
            f"native observation incomplete: actual cycle {native_cycle:.16g}, "
            f"requested cycle {float(target_cycle):.16g}"
        )
    return native_cycle


def bind_exact_observation_states(records: list[dict[str, Any]],
                                  support_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bind by source identity AND native time; never fill a missing state."""
    state_by_key = {}
    for state in support_rows:
        key = (float(state["R"]), int(state["source_order"]))
        if key in state_by_key:
            raise RuntimeError(f"duplicate evolved native state for {key}")
        state_by_key[key] = state
    rows = []
    for observation in records:
        key = (float(observation["R"]), int(observation["source_order"]))
        if key not in state_by_key:
            raise RuntimeError(
                f"missing evolved native state for observation {key}; "
                "held or interpolated support states cannot enter calibration"
            )
        source = state_by_key.pop(key)
        target = float(observation["cycles"])
        if not source.get("support_is_evolved", False):
            raise RuntimeError(f"observation {key} is not an evolved native state")
        for field in ("cycles", "native_cycles"):
            if field not in source or not math.isclose(
                float(source[field]), target, rel_tol=0.0,
                abs_tol=NATIVE_CYCLE_TOLERANCE,
            ):
                raise RuntimeError(f"native state time mismatch for {key}: {field}")
        state = dict(source)
        state.update({
            "observation_id": observation["series_id"],
            "observation_source_order": observation["source_order"],
            "observation_cycles": target,
            "mapped_cycles": observation["mapped_cycles"],
            "cycle_rounding_delta": observation["cycle_rounding_delta"],
            "cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
            "observation_crack_length_m": observation["crack_length_m"],
            "support_is_evolved": True,
        })
        rows.append(state)
    if state_by_key:
        raise RuntimeError("native states contain observations outside the fitted source history")
    return rows

CANDIDATES = {
    "sustained_active": PlasticCoordinates(1.2, 1.5e9, 1.5, 1.5),
    "transport_dominated": PlasticCoordinates(0.8, 1.0e9, 0.7, 2.5),
    "trapping_dominated": PlasticCoordinates(0.8, 1.2e9, 2.8, 0.8),
}


class ScheduleSliceExpired(BaseException):
    """Escape solver-internal handlers only at a committed restart boundary."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def _set_checkpoint_environment(directory: Path) -> dict[str, str | None]:
    previous = {
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR": os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR"),
        "V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS": os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS"),
    }
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(directory)
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS"] = "15"
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _identity(candidate_id: str, plastic: PlasticCoordinates, phase_count: int,
              load_substeps: int, support_points: int, frequency_Hz: float,
              alignment: str, state_only_acceleration: bool,
              state_opening_seed: OpeningCoordinates | None = None) -> dict[str, Any]:
    history_mode = (
        HISTORY_MODE if support_points <= 0
        else f"diagnostic_{int(support_points)}_supports_per_R_piecewise_linear_observed_load"
    )
    payload = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "plastic": asdict(plastic),
        "phase_count": int(phase_count),
        "load_substeps": int(load_substeps),
        "support_points_per_R": int(support_points),
        "frequency_Hz": float(frequency_Hz),
        "temperature_K": TEMPERATURE_K,
        "alignment": alignment,
        "state_only_acceleration": bool(state_only_acceleration),
        "premeasurement": True,
        "history_mode": history_mode,
        "moving_frame_policy": "fixed_local_window_reported_DeltaK_owns_remote_geometry",
        "opening_during_state_replay": (
            "zero observer; opening is postprocessed on the unchanged state path"
            if state_only_acceleration else "production opening observer"
        ),
        "state_opening_seed": (
            asdict(state_opening_seed) if state_opening_seed is not None else None
        ),
        "load_substep_cycle_partition": "nearest_integer_boundaries_exact_scheduled_observation_endpoint",
        "accepted_cycle_roundoff_policy": "snap_to_integer_within_1e-6_cycle",
        "observation_cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
        "observation_state_binding": "exact_R_source_order_and_verified_native_cycle",
        "calibration_eligible": support_points <= 0 and not state_only_acceleration,
        "runtime_model_id": MODEL_ID,
        "kernel_sha256": sha256(Path(R4_KERNEL_FAMILY)),
        "rate_input_sha256": sha256(DATA / "normalized/qste340tm_constant_amplitude.csv"),
        "history_input_sha256": sha256(DATA / "normalized/qste340tm_crack_histories.csv"),
    }
    return {"payload": payload, "sha256": canonical_hash(payload)}


def write_active_cache_provenance(parent: Any) -> dict[str, Any]:
    old_path = LEGACY_OUT / "candidate_cache/sustained_active_f8f63fdfca0e.json"
    stopped_path = LEGACY_OUT / "candidate_cache/sustained_active_23478e586bb6.json"
    old = _load_json(old_path)
    stopped = _load_json(stopped_path)
    plastic = CANDIDATES["sustained_active"]
    legacy_payload = {
        "plastic": asdict(plastic), "points": 2, "phases": 8,
        "alignment": "index_rank", "premeasurement": True,
    }
    stopped_payload = dict(legacy_payload, candidate_wall_limit_s=360)
    binding = material_binding(parent, plastic)
    engine = build_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    current_receipt = runtime_receipt(engine, stage="active_cache_provenance_current_initial")
    # Object identity is process-local and would otherwise rewrite provenance
    # on every restart; implementation identity is carried by the stable fields.
    current_receipt.get("reversible_binding", {}).pop("object_id", None)
    old_receipt = old["runtime_receipt"]
    common_paths = [
        "engine_class", "state_class", "model_id", "kinetics", "mechanics_provider",
        "methods", "taylor_contract", "source_lifecycle", "source_geometry",
        "source_geography", "persistent_invariants", "reversible_binding",
        "signed_kernel_family", "numerical_options",
    ]
    def normalized(value: Any) -> Any:
        return json.loads(json.dumps(value, sort_keys=True, default=str))

    def comparable(key: str, value: Any) -> Any:
        value = normalized(value)
        if key == "reversible_binding" and isinstance(value, dict):
            value.pop("object_id", None)
        return value

    comparison = {
        key: comparable(key, old_receipt.get(key)) == comparable(key, current_receipt.get(key))
        for key in common_paths
    }
    result = {
        "schema": "qste340tm-active-cache-provenance/1",
        "completed_cache": {
            "path": str(old_path), "sha256": sha256(old_path),
            "recorded_identity": old["identity_sha256"],
            "recomputed_pre_wall_identity": canonical_hash(legacy_payload),
            "status": old["status"], "elapsed_s": old["elapsed_s"],
        },
        "stopped_cache": {
            "path": str(stopped_path), "sha256": sha256(stopped_path),
            "recorded_identity": stopped["identity_sha256"],
            "recomputed_wall_identity": canonical_hash(stopped_payload),
            "status": stopped["status"], "reason": stopped["rejection_reason"],
        },
        "identity_difference": {
            "only_added_field": {"candidate_wall_limit_s": 360},
            "physical_or_numerical_input_changed": False,
            "completed_record_reuse_scope": "historical two-support/eight-phase fixed-local-window comparator",
        },
        "common_runtime_receipt_equal": comparison,
        "all_common_runtime_receipt_fields_equal": all(comparison.values()),
        "current_initial_runtime_receipt": current_receipt,
        "history_forcing": {
            "completed_cache": "two endpoint support loads per R; each long interval held at endpoint load",
            "new_primary": HISTORY_MODE,
            "equivalent_to_new_primary": False,
        },
    }
    atomic_json(OUT / "active_cache_provenance.json", result)
    return result


def _new_progress(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "qste340tm-active-outer-checkpoint/1",
        "identity": identity,
        "status": "running",
        "current_r_index": 0,
        "current_observation_position": 0,
        "current_load_substep": 0,
        "completed_observation_rows": [],
        "completed_process_rows": [],
        "interval_start_ledgers": None,
        "interval_modes": [],
        "slices": [],
    }


def _install_state_only_opening_observer(engine: Any) -> None:
    """Suppress only opening integration during a fixed-window state replay."""
    production = engine.lambda_cleave

    def state_only(self: Any, sigma_Pa: float, temperature_K: float) -> tuple[float, float, float]:
        _effective, _raw, barrier_J = production(sigma_Pa, temperature_K)
        return 0.0, 0.0, float(barrier_J)

    engine._qste_production_lambda_cleave = production
    engine.lambda_cleave = MethodType(state_only, engine)
    engine._qste_state_only_opening_observer = True


def _initialize_engine(parent: Any, plastic: PlasticCoordinates, *,
                       state_only_acceleration: bool = False,
                       state_opening_seed: OpeningCoordinates | None = None) -> Any:
    engine = build_engine(material_binding(parent, plastic), kernel_family_path=R4_KERNEL_FAMILY)
    if state_opening_seed is not None:
        install_opening(engine, opening_surface(parent.manifest.cleavage.surface, state_opening_seed))
    engine.hazard_threshold_action = 1.0e300
    engine.hazard_action_current = 0.0
    engine.B = 0.0
    if state_only_acceleration:
        _install_state_only_opening_observer(engine)
    return engine


def _checkpoint_dir(root: Path, r_value: float) -> Path:
    return root / f"R_{str(r_value).replace('.', 'p')}"


def _restore_fixed_window_checkpoint(engine: Any, directory: Path) -> dict[str, Any]:
    """Restore the outer geometry signature, then the production active state.

    Even with an effectively infinite event threshold, the cooperative clock
    can leave a positive subnormal ``micro_advance_total_m``.  The production
    checkpoint correctly includes it in its exact geometry signature.  This
    fixed-window replay restores that value byte-for-byte but rejects any
    mechanically consequential advance.
    """
    payload = json.loads((directory / "high_cycle_live_checkpoint.json").read_text())
    geometry = payload.get("geometry_signature", [])
    if len(geometry) != 8:
        raise RuntimeError("unexpected native checkpoint geometry signature")
    if max(abs(float(geometry[i])) for i in (1, 2, 3, 4)) > 1.0e-18:
        raise RuntimeError("fixed-window restart contains consequential crack advance")
    engine.n_adv = int(geometry[0])
    engine.a_adv = float(geometry[1])
    engine.micro_advance_total_m = float(geometry[2])
    engine.checkpoint_advance_total_m = float(geometry[3])
    engine.mpz.advance_total_m = float(geometry[4])
    return restore_checkpoint(engine, directory)


def qualify_local_state_only_observer(parent: Any, plastic: PlasticCoordinates,
                                      records: list[dict[str, Any]],
                                      phase_count: int) -> dict[str, Any]:
    """Qualify only the local one-cycle observer substitution."""
    checks: list[dict[str, Any]] = []
    reference_root = OUT / "restarts/sustained_active_23b19aa6590c"
    cases: list[tuple[str, float, Path | None]] = [("initial", R_VALUES[0], None)]
    if plastic == CANDIDATES["sustained_active"] and reference_root.is_dir():
        cases.extend(
            (f"completed_R_{r_value}", r_value, _checkpoint_dir(reference_root, r_value))
            for r_value in R_VALUES
            if (_checkpoint_dir(reference_root, r_value) / "high_cycle_live_checkpoint.json").is_file()
        )
    for label, r_value, checkpoint in cases:
        production = _initialize_engine(parent, plastic, state_only_acceleration=False)
        accelerated = _initialize_engine(parent, plastic, state_only_acceleration=True)
        if checkpoint is not None:
            _restore_fixed_window_checkpoint(production, checkpoint)
            _restore_fixed_window_checkpoint(accelerated, checkpoint)
        block = [row for row in records if row["R"] == r_value]
        row = block[-1] if checkpoint is not None else block[0]
        waveform = FatigueWaveform(
            Kmax=float(row["DeltaK_MPa_sqrt_m"])/(1.0-r_value)*1.0e6,
            R=r_value,
            frequency_Hz=FREQUENCY_HZ,
        )
        prod = one_cycle_map(
            production, _NativePhaseController(phase_count), waveform, TEMPERATURE_K,
        )
        fast = one_cycle_map(
            accelerated, _NativePhaseController(phase_count), waveform, TEMPERATURE_K,
        )
        prod_vector = np.asarray(prod.state_end.vector, dtype=float)
        fast_vector = np.asarray(fast.state_end.vector, dtype=float)
        difference = np.abs(prod_vector-fast_vector)
        scale = np.maximum(np.maximum(np.abs(prod_vector), np.abs(fast_vector)), 1.0)
        check = {
            "case": label,
            "R": r_value,
            "state_vector_length": int(prod_vector.size),
            "maximum_absolute_state_difference": float(np.max(difference, initial=0.0)),
            "maximum_scaled_state_difference": float(np.max(difference/scale, initial=0.0)),
            "ledger_delta_equal": prod.ledger_delta_per_cycle == fast.ledger_delta_per_cycle,
            "plastic_totals_equal": prod.plastic_totals == fast.plastic_totals,
            "transition_signature_equal": (
                prod.transition_signature_end == fast.transition_signature_end
            ),
            "production_hazard_action_per_cycle": prod.hazard_action_per_cycle,
            "state_only_hazard_action_per_cycle": fast.hazard_action_per_cycle,
        }
        check["passed"] = bool(
            check["maximum_absolute_state_difference"] == 0.0
            and check["ledger_delta_equal"]
            and check["plastic_totals_equal"]
            and check["transition_signature_equal"]
            and check["state_only_hazard_action_per_cycle"] == 0.0
        )
        checks.append(check)
    result = {
        "schema": "qste340tm-local-one-cycle-state-only-opening-qualification/1",
        "scope": (
            "local one-cycle identity only; this does not qualify long projective "
            "propagation because hazard validation can alter the accepted DMD path"
        ),
        "candidate": asdict(plastic),
        "phase_count": int(phase_count),
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }
    path = OUT / "qualifications" / (
        f"state_only_{canonical_hash({'plastic': asdict(plastic), 'phase_count': phase_count})[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    if not result["all_passed"]:
        raise RuntimeError("state-only opening observer changed the local active state")
    return result


def run_history_slice(candidate_id: str, plastic: PlasticCoordinates, parent: Any,
                      records: list[dict[str, Any]], *, phase_count: int,
                      load_substeps: int, support_points: int, frequency_Hz: float,
                      slice_seconds: int, alignment: str,
                      state_only_acceleration: bool,
                      state_opening_seed: OpeningCoordinates | None) -> dict[str, Any]:
    records = prepare_observation_schedule(records)
    identity = _identity(
        candidate_id, plastic, phase_count, load_substeps, support_points,
        frequency_Hz, alignment, state_only_acceleration, state_opening_seed,
    )
    root = OUT / "restarts" / f"{candidate_id}_{identity['sha256'][:12]}"
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "outer_progress.json"
    progress = _load_json(progress_path, _new_progress(identity))
    if progress["identity"]["sha256"] != identity["sha256"]:
        raise RuntimeError("active outer checkpoint identity mismatch")
    if progress.get("status") in {"complete", "diagnostic_complete"}:
        return progress
    progress["status"] = "running"
    atomic_json(progress_path, progress)

    started = time.monotonic()
    deadline = started + max(int(slice_seconds), 1)
    old_handler = signal.getsignal(signal.SIGALRM)
    original_native_writer = native_checkpoint.write_checkpoint
    expiry_requested = False

    def expire(_signum: int, _frame: Any) -> None:
        # Never raise asynchronously: the native NPZ/JSON pair and the outer
        # observation/cursor commit both contain multiple writes.
        nonlocal expiry_requested
        expiry_requested = True

    def slice_expired() -> bool:
        return expiry_requested or time.monotonic() >= deadline

    def checkpoint_then_yield(*args: Any, **kwargs: Any) -> Any:
        payload = original_native_writer(*args, **kwargs)
        if payload is not None and slice_expired():
            raise ScheduleSliceExpired()
        return payload

    signal.signal(signal.SIGALRM, expire)
    # Native maybe_write_checkpoint resolves this module global.  The runner's
    # direct imported writer remains unchanged, so both observation snapshots
    # and the complete outer cursor commit finish before the explicit check.
    native_checkpoint.write_checkpoint = checkpoint_then_yield
    signal.alarm(max(int(slice_seconds), 1))
    active_context: dict[str, Any] = {}
    try:
        for r_index in range(int(progress["current_r_index"]), len(R_VALUES)):
            r_value = R_VALUES[r_index]
            full_block = [row for row in records if row["R"] == r_value]
            if support_points <= 0 or support_points >= len(full_block):
                block = full_block
            else:
                selected = np.unique(np.linspace(
                    0, len(full_block)-1, int(support_points),
                ).round().astype(int))
                block = [full_block[int(index)] for index in selected]
            engine = _initialize_engine(
                parent, plastic, state_only_acceleration=state_only_acceleration,
                state_opening_seed=state_opening_seed,
            )
            checkpoint_dir = _checkpoint_dir(root, r_value)
            checkpoint_file = checkpoint_dir / "high_cycle_live_checkpoint.json"
            continuing_r = r_index == int(progress["current_r_index"])
            observation_position = int(progress["current_observation_position"]) if continuing_r else 0
            substep_start = int(progress["current_load_substep"]) if continuing_r else 0
            active_interval_checkpoint = bool(
                continuing_r
                and checkpoint_file.is_file()
                and progress.get("active_observation") is not None
            )
            if observation_position > 0 or substep_start > 0 or active_interval_checkpoint:
                if not checkpoint_file.is_file():
                    raise RuntimeError("outer cursor exists without native active-state checkpoint")
                _restore_fixed_window_checkpoint(engine, checkpoint_dir)
            initial_receipt = runtime_receipt(engine, stage=f"active_chronological_R_{r_value}_restored")
            if slice_expired():
                raise ScheduleSliceExpired()

            for position in range(observation_position, len(block)):
                row = block[position]
                global_index = records.index(row)
                prior_cycle = 0.0 if position == 0 else float(block[position-1]["cycles"])
                target_cycle = float(row["cycles"])
                end_dk = float(row["DeltaK_MPa_sqrt_m"])
                substeps = 1 if position == 0 else int(load_substeps)
                if position != observation_position:
                    substep_start = 0
                if progress.get("interval_start_ledgers") is None or position != observation_position:
                    progress["interval_start_ledgers"] = capture_ledgers(engine)
                    progress["interval_modes"] = []
                for substep in range(substep_start, substeps):
                    interval_cycles = target_cycle-prior_cycle
                    substep_start_cycle = prior_cycle + round(
                        interval_cycles*substep/substeps
                    )
                    substep_target_cycle = prior_cycle + round(
                        interval_cycles*(substep+1)/substeps
                    )
                    if substep == substeps-1:
                        substep_target_cycle = target_cycle
                    midpoint_cycle = 0.5*(substep_start_cycle+substep_target_cycle)
                    midpoint_dk = float(np.interp(
                        midpoint_cycle,
                        np.asarray([float(value["cycles"]) for value in full_block]),
                        np.asarray([float(value["DeltaK_MPa_sqrt_m"]) for value in full_block]),
                    ))
                    accepted_total = float(engine.t) * float(frequency_Hz)
                    nearest_cycle = round(accepted_total)
                    if abs(accepted_total-nearest_cycle) <= 1.0e-6:
                        accepted_total = float(nearest_cycle)
                    remaining = max(substep_target_cycle-accepted_total, 0.0)
                    progress.update({
                        "current_r_index": r_index,
                        "current_observation_position": position,
                        "current_load_substep": substep,
                        "active_observation": {
                            "observation_id": row["series_id"],
                            "source_order": row["source_order"],
                            "target_cycle": target_cycle,
                            "substep_target_cycle": substep_target_cycle,
                            "midpoint_DeltaK_MPa_sqrt_m": midpoint_dk,
                        },
                    })
                    atomic_json(progress_path, progress)
                    active_context = dict(progress["active_observation"])
                    if remaining > 1.0e-9:
                        waveform = FatigueWaveform(
                            Kmax=midpoint_dk/(1.0-r_value)*1.0e6,
                            R=r_value,
                            frequency_Hz=frequency_Hz,
                        )
                        previous_env = _set_checkpoint_environment(checkpoint_dir)
                        try:
                            result = integrate_state_coupled_waveform(
                                engine, _NativePhaseController(phase_count), waveform,
                                TEMPERATURE_K, remaining,
                            )
                        finally:
                            _restore_environment(previous_env)
                        progress["interval_modes"].extend(result.get("coupled_hazard_modes", []))
                        if result.get("fired"):
                            raise RuntimeError("suppressed-threshold chronological replay fired")
                        if result.get("coupled_hazard_work_budget_exhausted"):
                            previous_env = _set_checkpoint_environment(checkpoint_dir)
                            try:
                                write_checkpoint(
                                    engine, waveform=waveform,
                                    temperature_K=TEMPERATURE_K,
                                    reason="qste_native_work_budget_restart",
                                    metadata={
                                        "identity_sha256": identity["sha256"],
                                        "observation_id": row["series_id"],
                                        "current_observation_position": position,
                                        "current_load_substep": substep,
                                    },
                                )
                            finally:
                                _restore_environment(previous_env)
                            raise ScheduleSliceExpired()
                    assert_native_cycle_target(engine, substep_target_cycle, frequency_Hz)
                    progress["current_load_substep"] = substep + 1
                    atomic_json(progress_path, progress)

                native_cycles = assert_native_cycle_target(engine, target_cycle, frequency_Hz)
                before = progress["interval_start_ledgers"]
                process = {
                    "requested_cycles": target_cycle-prior_cycle,
                    "accepted_cycles": float(engine.t)*frequency_Hz-prior_cycle,
                    "accepted_seconds": (float(engine.t)*frequency_Hz-prior_cycle)/frequency_Hz,
                    "load_interpolation": (
                        "initial measured load held through premeasurement exposure" if position == 0
                        else f"{load_substeps}-point midpoint quadrature of piecewise-linear ordered observed DeltaK(N)"
                    ),
                    "modes": progress["interval_modes"],
                    "ledger_delta": ledger_delta(before, capture_ledgers(engine)),
                    "moving_frame": {
                        "status": "not_applied_fixed_local_window",
                        "reason": "accepted kernel extension domain is 0.001175 m; no extrapolation or state reset",
                    },
                }
                state = _state_row(engine, row, process, 0.0)
                state.update({
                    "observation_id": row["series_id"],
                    "support_state_id": f"{candidate_id}:R{r_value}:obs{position}:cycle{target_cycle:.12g}",
                    "support_is_evolved": True,
                    "support_observation_position": position,
                    "native_cycles": native_cycles,
                    "mapped_cycles": row["mapped_cycles"],
                    "cycle_rounding_delta": row["cycle_rounding_delta"],
                    "cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
                    "phase_count": phase_count,
                    "load_substeps": substeps,
                })
                observation_checkpoint = checkpoint_dir / "observations" / f"source_{int(row['source_order']):06d}"
                state["native_checkpoint_path"] = str(observation_checkpoint.resolve())
                # Commit both the live restart and the separate observation
                # checkpoint before advancing the outer observation cursor.
                for destination in (checkpoint_dir, observation_checkpoint):
                    previous_env = _set_checkpoint_environment(destination)
                    try:
                        write_checkpoint(
                            engine,
                            waveform=FatigueWaveform(
                                Kmax=end_dk/(1.0-r_value)*1.0e6,
                                R=r_value, frequency_Hz=frequency_Hz,
                            ),
                            temperature_K=TEMPERATURE_K,
                            reason="qste_observation_interval_complete",
                            metadata={
                                "identity_sha256": identity["sha256"],
                                "observation_id": row["series_id"],
                                "source_order": row["source_order"],
                                "support_state_id": state["support_state_id"],
                                "mapped_cycles": row["mapped_cycles"],
                                "native_cycles": native_cycles,
                                "cycle_rounding_delta": row["cycle_rounding_delta"],
                                "cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
                                "next_observation_position": position+1,
                            },
                        )
                    finally:
                        _restore_environment(previous_env)
                progress["completed_observation_rows"].append(state)
                progress["completed_process_rows"].append({
                    "observation_id": row["series_id"],
                    "support_state_id": state["support_state_id"],
                    "R": r_value, "source_order": row["source_order"],
                    "cycles": target_cycle, "mapped_cycles": row["mapped_cycles"],
                    "cycle_rounding_delta": row["cycle_rounding_delta"], "process": process,
                })
                progress["interval_start_ledgers"] = None
                progress["interval_modes"] = []
                progress["current_observation_position"] = position + 1
                progress["current_load_substep"] = 0
                atomic_json(progress_path, progress)
                if slice_expired():
                    raise ScheduleSliceExpired()

            progress["current_r_index"] = r_index + 1
            progress["current_observation_position"] = 0
            progress["current_load_substep"] = 0
            progress["last_runtime_receipt"] = initial_receipt
            atomic_json(progress_path, progress)
            if r_index + 1 < len(R_VALUES) and slice_expired():
                raise ScheduleSliceExpired()

        progress["status"] = "history_complete"
    except ScheduleSliceExpired:
        progress["status"] = "interrupted_restartable"
        progress["interruption"] = {
            "classification": "RESOURCE_SCHEDULING_INTERVAL",
            "active_context": active_context,
            "native_checkpoint_expected": True,
        }
    finally:
        signal.alarm(0)
        native_checkpoint.write_checkpoint = original_native_writer
        signal.signal(signal.SIGALRM, old_handler)
        progress["slices"].append({
            "finished_unix_s": time.time(),
            "elapsed_s": time.monotonic()-started,
            "status": progress["status"],
            "active_context": active_context,
        })
        atomic_json(progress_path, progress)
    return progress


def finalize_candidate(candidate_id: str, plastic: PlasticCoordinates, parent: Any,
                       records: list[dict[str, Any]], progress: dict[str, Any],
                       *, phase_count: int, load_substeps: int, support_points: int,
                       frequency_Hz: float, alignment: str,
                       state_only_acceleration: bool,
                       state_opening_seed: OpeningCoordinates | None) -> dict[str, Any]:
    records = prepare_observation_schedule(records)
    support_rows = progress["completed_observation_rows"]
    expected_supports = sum(
        min(len([row for row in records if row["R"] == r_value]), support_points)
        if support_points > 0 else len([row for row in records if row["R"] == r_value])
        for r_value in R_VALUES
    )
    if len(support_rows) != expected_supports:
        raise RuntimeError(
            f"chronological support history incomplete: {len(support_rows)}/{expected_supports} supports"
        )
    identity = _identity(
        candidate_id, plastic, phase_count, load_substeps, support_points,
        frequency_Hz, alignment, state_only_acceleration, state_opening_seed,
    )
    if support_points > 0 or state_only_acceleration:
        # Sparse native states are useful diagnostics.  They cannot define a
        # calibrated 760-observation history without qualified state output.
        result = {
            "schema": SCHEMA,
            "identity": identity,
            "candidate_id": candidate_id,
            "plastic_coordinates": asdict(plastic),
            "status": "diagnostic_complete",
            "calibration_eligible": False,
            "diagnostic_reason": (
                "sparse_native_supports" if support_points > 0
                else "state_only_acceleration_not_long_path_qualified"
            ),
            "observation_count": len(records),
            "genuine_evolved_support_count": len(support_rows),
            "support_states": support_rows,
            "process_rows": progress["completed_process_rows"],
            "history_mode": identity["payload"]["history_mode"],
            "state_only_acceleration": bool(state_only_acceleration),
            "slices": progress["slices"],
            "claim_limit": "native support diagnostics only; no all-observation fit or predictions",
        }
        path = OUT / "diagnostic_candidates" / f"{candidate_id}_{identity['sha256'][:12]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
        return result
    rows = bind_exact_observation_states(records, support_rows)
    if not math.isclose(float(frequency_Hz), FREQUENCY_HZ, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "the imported opening fitter currently divides by its fixed FREQUENCY_HZ; "
            "wire frequency through that interface before calibrating another frequency"
        )
    stress = _stress_trace(records, rows, phase_count)
    base_surface = parent.manifest.cleavage.surface
    fits = {mode: fit_opening(mode, base_surface, records, stress)
            for mode in ("absolute", "shape", "slope_priority", "slope_priority_weak_offset")}
    frozen = frozen_stress(records, phase_count)
    final_receipts = {}
    for mode, fit in fits.items():
        prediction = np.asarray(fit["prediction"])
        fit["metrics"] = metrics(records, prediction)
        fit["frozen_prediction"] = predict(
            base_surface, np.asarray(fit["coordinates_vector"]), frozen,
        )[0].tolist()
        fit["frozen_metrics"] = metrics(records, np.asarray(fit["frozen_prediction"]))
        engine = _initialize_engine(parent, plastic, state_only_acceleration=False)
        install_opening(engine, opening_surface(base_surface, OpeningCoordinates(**fit["coordinates"])))
        final_receipts[mode] = runtime_receipt(engine, stage=f"final_fitted_opening_{mode}")
        final_receipts[mode]["fitted_opening_coordinates"] = fit["coordinates"]
        final_receipts[mode]["fitted_surface_sha256"] = canonical_hash(
            repr(engine.manifest.cleavage.surface)
        )
    result = {
        "schema": SCHEMA,
        "identity": _identity(
            candidate_id, plastic, phase_count, load_substeps, support_points,
            frequency_Hz, alignment, state_only_acceleration, state_opening_seed,
        ),
        "candidate_id": candidate_id,
        "plastic_coordinates": asdict(plastic),
        "status": "complete",
        "calibration_eligible": True,
        "observation_count": len(rows),
        "genuine_evolved_support_count": len(support_rows),
        "history_mode": progress["identity"]["payload"]["history_mode"],
        "state_only_acceleration": bool(state_only_acceleration),
        "state_opening_seed": (
            asdict(state_opening_seed) if state_opening_seed is not None else None
        ),
        "states": rows,
        "support_states": support_rows,
        "process_rows": progress["completed_process_rows"],
        "fits": fits,
        "final_effective_runtime_receipts": final_receipts,
        "restart_root": str((OUT / "restarts" / f"{candidate_id}_{progress['identity']['sha256'][:12]}").resolve()),
        "slices": progress["slices"],
        "claim_limit": "fixed local kernel window; remote geometry supplied by reported DeltaK; no native crack translation",
    }
    path = OUT / "completed_candidates" / f"{candidate_id}_{result['identity']['sha256'][:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--load-substeps", type=int, default=2)
    parser.add_argument("--support-points", type=int, default=0,
                        help="<=0 calibrates every observation; positive values produce sparse diagnostics only")
    parser.add_argument("--frequency-hz", type=float, default=FREQUENCY_HZ)
    parser.add_argument("--alignment", choices=("index_rank", "deltaK_crack_length"), default="index_rank")
    parser.add_argument("--slice-seconds", type=int, default=600,
                        help="soft scheduling interval; expiry waits for a committed checkpoint or observation")
    parser.add_argument("--max-slices", type=int, default=0,
                        help="<=0 continues checkpointed scheduling until complete; positive values bound this invocation")
    parser.add_argument(
        "--state-only-opening-diagnostic", action="store_true",
        help="diagnostic only: omit opening from DMD validation; not production-qualified",
    )
    parser.add_argument(
        "--state-opening-seed-result", type=Path,
        help="completed candidate JSON whose fitted opening is installed during state replay",
    )
    parser.add_argument(
        "--state-opening-seed-mode", choices=("absolute", "shape"), default="absolute",
    )
    args = parser.parse_args(argv)
    if args.frequency_hz <= 0.0 or not math.isfinite(args.frequency_hz):
        parser.error("frequency must be finite and positive")
    if (args.support_points <= 0 and not args.state_only_opening_diagnostic
            and not math.isclose(args.frequency_hz, FREQUENCY_HZ, rel_tol=0.0, abs_tol=1.0e-12)):
        parser.error("calibration frequency must match the imported opening fitter until frequency is wired through it")
    OUT.mkdir(parents=True, exist_ok=True)
    parent = load_parent_binding(ALT_REGISTRY, PARENT_REGISTRY, candidate_id=PARENT_CANDIDATE)
    provenance = write_active_cache_provenance(parent)
    if not provenance["all_common_runtime_receipt_fields_equal"]:
        raise RuntimeError("completed active cache differs from current common runtime")
    records = prepare_observation_schedule(build_records(args.alignment, premeasurement=True))
    plastic = CANDIDATES[args.candidate]
    state_only_acceleration = bool(args.state_only_opening_diagnostic)
    state_opening_seed = None
    if args.state_opening_seed_result is not None:
        seed_result = _load_json(args.state_opening_seed_result)
        state_opening_seed = OpeningCoordinates(
            **seed_result["fits"][args.state_opening_seed_mode]["coordinates"]
        )
    if state_only_acceleration:
        qualify_local_state_only_observer(parent, plastic, records, args.phase_count)
    progress = None
    slices_run = 0
    while args.max_slices <= 0 or slices_run < args.max_slices:
        slices_run += 1
        progress = run_history_slice(
            args.candidate, plastic, parent, records,
            phase_count=args.phase_count, load_substeps=args.load_substeps,
            support_points=args.support_points,
            frequency_Hz=args.frequency_hz, slice_seconds=args.slice_seconds,
            alignment=args.alignment,
            state_only_acceleration=state_only_acceleration,
            state_opening_seed=state_opening_seed,
        )
        print(json.dumps({
            "candidate": args.candidate,
            "status": progress["status"],
            "completed_observations": len(progress["completed_observation_rows"]),
            "current_r_index": progress["current_r_index"],
            "current_observation_position": progress["current_observation_position"],
            "current_load_substep": progress["current_load_substep"],
        }, sort_keys=True), flush=True)
        if progress["status"] in {"history_complete", "complete", "diagnostic_complete"}:
            break
    if progress is None:
        raise RuntimeError("active history did not start")
    if progress["status"] == "history_complete":
        result = finalize_candidate(
            args.candidate, plastic, parent, records, progress,
            phase_count=args.phase_count, load_substeps=args.load_substeps,
            support_points=args.support_points,
            frequency_Hz=args.frequency_hz, alignment=args.alignment,
            state_only_acceleration=state_only_acceleration,
            state_opening_seed=state_opening_seed,
        )
        progress["status"] = result["status"]
        atomic_json(
            OUT / "restarts" / f"{args.candidate}_{progress['identity']['sha256'][:12]}" / "outer_progress.json",
            progress,
        )
        summary = {
            "candidate": args.candidate,
            "status": result["status"],
            "calibration_eligible": result["calibration_eligible"],
            "genuine_evolved_support_count": result["genuine_evolved_support_count"],
        }
        if result["calibration_eligible"]:
            summary.update({
                "absolute_rmse": result["fits"]["absolute"]["metrics"]["overall_rmse_log10_rate"],
                "shape_rmse": result["fits"]["shape"]["metrics"]["shape_rmse_log10"],
                "slope_priority_score": result["fits"]["slope_priority"]["slope_priority_evaluation"]["primary_score"],
                "slope_priority_eligible": result["fits"]["slope_priority"]["slope_priority_evaluation"]["eligible"],
            })
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    if progress["status"] in {"complete", "diagnostic_complete"}:
        return 0
    return 75


if __name__ == "__main__":
    raise SystemExit(main())
