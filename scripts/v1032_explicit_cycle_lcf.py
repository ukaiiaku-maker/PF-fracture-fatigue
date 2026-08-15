"""Phase-resolved, no-projection LCF integrator for the v9.14 1-D model.

This diagnostic deliberately imports the immutable v9.14 constitutive model
and changes only cycle integration.  It supports multiple stochastic cleavage
events in one physical cycle and resumes the waveform after every event.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture import fatigue_v914 as base
from arrhenius_fracture.emergent_gnd_state_v913 import EmergentGNDState


SCHEMA = "v10.2.32_explicit_cycle_lcf_checkpoint_v1"
MODE = "explicit_physical_cycles"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    complete = copy.deepcopy(payload)
    complete["integrity_sha256"] = base._canonical_digest(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as stream:
        json.dump(complete, stream, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    checksum = payload.pop("integrity_sha256", None)
    if checksum != base._canonical_digest(payload):
        raise ValueError("explicit-cycle checkpoint integrity mismatch")
    if payload.get("schema") != SCHEMA:
        raise ValueError("explicit-cycle checkpoint schema mismatch")
    return payload


def _diagnostic(state: EmergentGNDState, loading, cycle: int, phase: float,
                action: float, threshold: float, extension: float,
                record_type: str) -> dict[str, Any]:
    K = loading.K_at_phase(phase)
    shield = float(state.K_shield_MPa_sqrt_m())
    K_eff = max(K - shield, 0.0)
    radius = float(state.tip_radius_m())
    sigma = K_eff * 1e6 / math.sqrt(2 * math.pi * max(radius, state.c.b_m, 1e-30))
    barrier = float(state.p.cleavage.barrier_eV(sigma, loading.temperature_K))
    rates = state.local_rates(K, loading.temperature_K)
    backstress = state.backstress_state()[2]
    return {
        "record_type": record_type, "cycle_index": int(cycle), "phase": float(phase),
        "cumulative_cycles": float(cycle + phase), "time_s": float(state.time_s),
        "K_MPa_sqrt_m": float(K), "K_eff_MPa_sqrt_m": float(K_eff),
        "tip_stress_Pa": float(sigma), "effective_barrier_eV": barrier,
        "cleavage_rate_s": float(state.cleavage_rate_s(K, loading.temperature_K)),
        "cumulative_hazard_action": float(action), "threshold_action": float(threshold),
        "crack_extension_m": float(extension), "mobile_total_m2": float(np.sum(state.mobile_m2)),
        "retained_total_m2": float(np.sum(state.retained_m2)),
        "accumulated_slip_total_m2": float(np.sum(state.accumulated_slip_m2)),
        "shielding_MPa_sqrt_m": shield, "backstress_Pa": float(np.mean(backstress)),
        "tip_radius_m": radius, "front_width_m": float(state.source_geometry()["front_width_m"]),
        "emission_rate_peak_s": float(np.max(rates["emission_rate_s"])),
        "peierls_velocity_peak_m_s": float(np.max(np.abs(rates["peierls_velocity_m_s"]))),
        "taylor_completion_peak_s": float(np.max(rates["taylor_completion_s"])),
        "encounter_rate_peak_s": float(np.max(rates["encounter_s"])),
    }


def _localized_phase(state, loading, p0: float, p1: float,
                     required_action: float) -> tuple[Any, float, float]:
    """Transactional bisection within one phase interval."""
    whole = copy.deepcopy(state)
    increment = base._advance_phase(whole, loading, p0, p1)
    if increment < required_action:
        raise RuntimeError("phase interval does not bracket threshold")
    lo, hi = p0, p1
    for _ in range(56):
        mid = (lo + hi) / 2
        trial = copy.deepcopy(state)
        value = base._advance_phase(trial, loading, p0, mid)
        if value >= required_action: hi = mid
        else: lo = mid
        if hi - lo <= 1e-14: break
    committed = copy.deepcopy(state)
    value = base._advance_phase(committed, loading, p0, hi)
    return committed, hi, value


def run_explicit_cycle_fatigue(candidate, physics, loading, *, seed: int,
                               numerics=base.FatigueNumerics(),
                               checkpoint_path: str | Path | None = None,
                               restart_from: str | Path | None = None,
                               maximum_physical_cycles: int = 50,
                               checkpoint_each_phase: bool = True,
                               checkpoint_cycle_interval: int | None = None,
                               state_history_cycle_interval: int | None = None,
                               pause_after_phase_advances: int | None = None) -> dict[str, Any]:
    """Resolve every phase and continue the waveform after every crack event."""
    loading.validate(); physics.validate(); numerics.validate()
    if checkpoint_cycle_interval is not None and checkpoint_cycle_interval < 1:
        raise ValueError("checkpoint_cycle_interval must be positive")
    if state_history_cycle_interval is not None and state_history_cycle_interval < 1:
        raise ValueError("state_history_cycle_interval must be positive")
    contract = {
        "mode": MODE, "candidate_id": candidate.candidate_id,
        "candidate_sha256": base._canonical_digest(asdict(candidate)),
        "physics_sha256": base._canonical_digest(asdict(physics)),
        "loading": asdict(loading), "numerics": asdict(numerics), "seed": int(seed),
        "maximum_physical_cycles": int(maximum_physical_cycles),
        "state_history_cycle_interval": state_history_cycle_interval,
    }
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if restart_from:
        saved = _read(Path(restart_from))
        if saved["contract"] != contract: raise ValueError("explicit-cycle checkpoint contract mismatch")
        state = base._state_from_checkpoint(candidate, physics, saved["state"])
        rng = np.random.default_rng(); rng.bit_generator.state = saved["rng_state"]
        c = saved["controller"]
        cycle, phase = int(c["cycle_index"]), float(c["phase"])
        action, threshold = float(c["action"]), float(c["threshold"])
        extension = float(c["extension"]); interval_start = float(c["interval_start"])
        events = saved["events"]; history = saved["state_history"]
        renewals = int(c["renewals"]); sequence = int(saved["sequence"])
        phase_advances = int(c.get("phase_advances", 0))
        restart_count = int(saved.get("restart_count", 0)) + 1
    else:
        state = EmergentGNDState(candidate, physics); rng = np.random.default_rng(int(seed))
        cycle = 0; phase = 0.0; action = 0.0; threshold = float(rng.exponential())
        extension = 0.0; interval_start = 0.0; events = []; history = []
        renewals = 0; sequence = 0; restart_count = 0; phase_advances = 0

    def save(reason: str) -> None:
        nonlocal sequence
        if checkpoint is None: return
        sequence += 1
        _atomic_json(checkpoint, {
            "schema": SCHEMA, "contract": contract, "sequence": sequence,
            "reason": reason, "restart_count": restart_count,
            "state": base._state_to_checkpoint(state),
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "controller": {"cycle_index": cycle, "phase": phase, "action": action,
                           "threshold": threshold, "extension": extension,
                           "interval_start": interval_start, "renewals": renewals,
                           "phase_advances": phase_advances},
            "events": events, "state_history": history,
        })

    if not history:
        history.append(_diagnostic(state, loading, cycle, phase, action, threshold, extension, "initial"))
    while cycle < maximum_physical_cycles and extension < numerics.target_extension_m:
        step_index = min(int(math.floor(phase * loading.phase_steps + 1e-10)), loading.phase_steps - 1)
        boundary = (step_index + 1) / loading.phase_steps
        trial = copy.deepcopy(state)
        increment = base._advance_phase(trial, loading, phase, boundary)
        if increment < 0 or not math.isfinite(increment):
            raise FloatingPointError("invalid explicit phase hazard")
        if action + increment >= threshold:
            state, event_phase, localized = _localized_phase(
                state, loading, phase, boundary, threshold - action)
            phase = event_phase; cumulative = cycle + phase
            physical_action = action + localized
            factor = base._event_length_factor(threshold, numerics)
            proposed = numerics.base_event_length_m * factor
            before_extension = extension
            state.translate_tip(proposed); extension += proposed
            geometry = state.source_geometry(); rates = state.local_rates(loading.K_at_phase(phase), loading.temperature_K)
            events.append({
                "event_index": len(events), "cumulative_cycles": cumulative,
                "cycle_index": cycle, "cycle_phase": phase,
                "interval_cycles": cumulative - interval_start,
                "K_MPa_sqrt_m": loading.K_at_phase(phase),
                "local_tip_stress_Pa": _diagnostic(state, loading, cycle, phase, 0, threshold, extension, "event")["tip_stress_Pa"],
                "threshold_action": threshold, "physical_hazard_action": physical_action,
                "proposed_advance_m": proposed, "committed_advance_m": extension-before_extension,
                "cumulative_extension_m": extension, "event_length_factor": factor,
                "mobile_total_m2": float(np.sum(state.mobile_m2)),
                "retained_total_m2": float(np.sum(state.retained_m2)),
                "shielding_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
                "backstress_Pa": float(np.mean(state.backstress_state()[2])),
                "tip_radius_m": float(geometry["tip_radius_m"]),
                "front_width_m": float(geometry["front_width_m"]),
                "emission_rate_peak_s": float(np.max(rates["emission_rate_s"])),
                "renewal_index": renewals,
            })
            history.append(_diagnostic(state, loading, cycle, phase, 0.0, threshold, extension, "post_event"))
            interval_start = cumulative; threshold = float(rng.exponential()); renewals += 1; action = 0.0
            save("committed_event")
            # Continue from this exact phase.  Multiple events before the next
            # phase boundary and within one cycle are intentionally allowed.
            continue
        state = trial; action += increment; phase = boundary
        phase_advances += 1
        completed_cycle = phase >= 1.0 - 1e-14
        if completed_cycle:
            cycle += 1; phase = 0.0
        retain_phase_diagnostic = (
            state_history_cycle_interval is None
            or (completed_cycle and cycle % state_history_cycle_interval == 0)
        )
        if retain_phase_diagnostic:
            history.append(_diagnostic(
                state, loading, cycle, phase, action, threshold, extension,
                "cycle_boundary" if completed_cycle else "phase_boundary",
            ))
        if checkpoint_each_phase:
            save("phase_boundary")
        elif (completed_cycle and checkpoint_cycle_interval is not None
              and cycle % checkpoint_cycle_interval == 0):
            save("cycle_boundary")
        if pause_after_phase_advances is not None and phase_advances >= pause_after_phase_advances:
            save("diagnostic_pause"); break

    final_cycles = cycle + phase
    status = ("growth_target_reached" if extension >= numerics.target_extension_m else
              "diagnostic_pause" if pause_after_phase_advances is not None and phase_advances >= pause_after_phase_advances else
              "explicit_cycle_limit")
    result = {
        "schema": "v10.2.32_explicit_cycle_lcf_result_v1", "mode": MODE,
        "candidate_id": candidate.candidate_id, "loading": asdict(loading), "seed": int(seed),
        "status": status, "final_cycles": final_cycles, "final_extension_m": extension,
        "events": events, "state_history": history, "renewals": renewals,
        "current_threshold_action": threshold, "current_hazard_action": action,
        "rng_state": copy.deepcopy(rng.bit_generator.state), "restart_count": restart_count,
        "integration": {"DMD_enabled": False, "multi_cycle_projection": False,
                        "phase_resolved": True, "same_cycle_post_event_continuation": True},
    }
    if len(events) >= 2:
        result["trajectory_da_dN_m_per_cycle"] = extension / max(final_cycles, 1e-300)
    save("terminal")
    return result
