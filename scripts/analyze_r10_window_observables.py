#!/usr/bin/env python3
"""Export like-for-like developed-window and censor observables for R10.

The A0 finite-window reference consumes the accepted full-state event marks
and rewards, but advances those marks on the phase-resolved opening clock.
Thus the stochastic stream is common while event times are not imposed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "integration_evidence/r10_global_mechanism_search"
OUT = BASE / "balanced_information_batch"
REGISTRY = BASE / "native_batch_registry.csv"

STREAMS = (
    ("OPENING_DIRECTED", "PRIMARY", BASE / "native_batch_v2/R10A_GLOBAL_018756"),
    ("MIXED_DIRECTED", "PRIMARY", BASE / "native_batch_v5/R10A_GLOBAL_119268"),
    ("MIXED_DIRECTED", "INDEPENDENT_MARK_STREAM", BASE / "native_batch_v8/R10A_GLOBAL_119268"),
    ("PLASTICITY_DIRECTED", "PRIMARY", BASE / "native_batch_v6/R10A_GLOBAL_064416"),
)
DRIFT_FIELDS = (
    "tip_radius_m", "K_shield_Pa_sqrt_m", "mobile_count", "retained_count",
    "gross_source_activity", "gross_return_activity", "continuous_mpz_progress_m",
)


def registry() -> dict[str, dict]:
    return {r["candidate_id"]: r for r in csv.DictReader(REGISTRY.open(newline=""))}


def opening_phase_action(record: dict, result: dict, registry_path: Path = REGISTRY) -> np.ndarray:
    """Action in each retained midpoint phase of the production waveform."""
    from arrhenius_fracture.retained_material_r10 import load_offspring_binding

    candidate = record["candidate_id"]
    binding = load_offspring_binding(registry_path, candidate)
    phases = int(result["phase_count"])
    phi = 2.0 * math.pi * (np.arange(phases) + 0.5) / phases
    R = float(result["R"])
    h = np.maximum(0.5 * (1.0 + R) + 0.5 * (1.0 - R) * np.cos(phi), 0.0)
    sigma = np.minimum(
        float(result["Kmax_MPa_sqrt_m"]) * 1e6 * h / math.sqrt(2.0 * math.pi * 1e-6),
        30e9,
    )
    raw = binding.manifest.cleavage.rate(sigma, float(result["temperature_K"]))
    # These two fields are part of the sealed executable registry and are the
    # values passed by load_offspring_binding into the native configuration.
    hits = float(record["cleavage_hits"])
    tau = float(record["cleavage_tau_s"])
    cooperative = gammainc(hits, np.maximum(raw, 0.0) * tau) / tau
    return cooperative / (float(result["frequency_Hz"]) * phases)


def advance_mark(cycle: float, threshold: float, phase_actions: np.ndarray) -> float:
    """Advance a mark through the retained piecewise phase clock."""
    n = len(phase_actions)
    phase_position = (cycle % 1.0) * n
    phase_index = int(math.floor(phase_position)) % n
    phase_fraction = phase_position - math.floor(phase_position)
    remaining = float(threshold)
    total = float(cycle)
    # Consume the partial starting phase.
    available = phase_actions[phase_index] * (1.0 - phase_fraction)
    if remaining <= available and phase_actions[phase_index] > 0.0:
        return total + remaining / phase_actions[phase_index] / n
    remaining -= available
    total += (1.0 - phase_fraction) / n
    phase_index = (phase_index + 1) % n
    per_cycle = float(np.sum(phase_actions))
    if per_cycle <= 0.0:
        return math.inf
    whole = max(0, int(math.floor(remaining / per_cycle)) - 1)
    total += whole
    remaining -= whole * per_cycle
    while True:
        action = float(phase_actions[phase_index])
        if remaining <= action and action > 0.0:
            return total + remaining / action / n
        remaining -= action
        total += 1.0 / n
        phase_index = (phase_index + 1) % n


def finite_window_reference(result: dict, record: dict,
                            registry_path: Path = REGISTRY) -> dict | None:
    if not result["events"]:
        return None
    actions = opening_phase_action(record, result, registry_path)
    # The retained waveform declaration is KMAX_COSINE_PHASE_ZERO.
    cycle = 0.0
    reference_events = []
    extension = 0.0
    for event in result["events"]:
        cycle = advance_mark(cycle, float(event["threshold_action"]), actions)
        extension += float(event["event_advance_m"])
        reference_events.append({"cycles": cycle, "phase": cycle % 1.0,
                                 "extension_m": extension})
    lo = next((e for e in reference_events if e["extension_m"] >= 20e-6), None)
    hi = next((e for e in reference_events if e["extension_m"] >= 60e-6), None)
    if lo is None or hi is None or hi["cycles"] <= lo["cycles"]:
        return None
    return {
        "a_low_m": lo["extension_m"], "a_high_m": hi["extension_m"],
        "N_low": lo["cycles"], "N_high": hi["cycles"],
        "phase_low": lo["phase"], "phase_high": hi["phase"],
        "rate_m_per_cycle": (hi["extension_m"] - lo["extension_m"]) /
                            (hi["cycles"] - lo["cycles"]),
        "action_per_cycle": float(np.sum(actions)),
        "reference_kind": "FINITE_WINDOW_PHASE_RESOLVED_A0_COMMON_MARK_REWARD_STREAM",
    }


def full_window(result: dict) -> dict | None:
    lo = next((e for e in result["events"] if e["extension_m"] >= 20e-6), None)
    hi = next((e for e in result["events"] if e["extension_m"] >= 60e-6), None)
    if lo is None or hi is None or hi["cycles"] <= lo["cycles"]:
        return None
    delta_N = float(hi["cycles"] - lo["cycles"])
    out = {
        "a_low_m": lo["extension_m"], "a_high_m": hi["extension_m"],
        "N_low": lo["cycles"], "N_high": hi["cycles"], "DeltaN": delta_N,
        "phase_low": lo["phase"], "phase_high": hi["phase"],
        "events_in_interval": int(hi["event_index"] - lo["event_index"]),
        "completed_loading_reversals": int(math.floor(2 * hi["cycles"]) -
                                            math.floor(2 * lo["cycles"])),
        "rate_m_per_cycle": (hi["extension_m"] - lo["extension_m"]) / delta_N,
        "single_excursion_window": delta_N < 0.5,
    }
    for field in DRIFT_FIELDS:
        out[f"drift_{field}"] = float(hi[field]) - float(lo[field])
    return out


def flatten(prefix: str, value: dict | None, out: dict) -> None:
    if value is not None:
        out.update({f"{prefix}_{k}": v for k, v in value.items()})


def main() -> int:
    records = registry()
    rows = []
    censors = []
    for lane, stream, directory in STREAMS:
        references = json.loads((directory / "opening_reference.json").read_text())
        stationary = {round(float(x["Kmax_MPa_sqrt_m"]), 8): x for x in references["references"]}
        for path in sorted(directory.glob("fatigue_Kmax_*/result.json")):
            result = json.loads(path.read_text())
            key = round(float(result["Kmax_MPa_sqrt_m"]), 8)
            ref = stationary[key]
            row = {
                "lane": lane, "stream": stream, "candidate_id": result["candidate_id"],
                "temperature_K": result["temperature_K"],
                "Kmax_MPa_sqrt_m": result["Kmax_MPa_sqrt_m"],
                "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"],
                "status": result["status"], "event_count": len(result["events"]),
                "completed_cycles": result["final_cycles"],
                "completed_reversals": int(math.floor(2 * float(result["final_cycles"]))),
                "final_extension_m": result["final_extension_m"],
                "final_continuous_advance_m": result["final_continuous_mpz_progress_m"],
                "phase_count": result["phase_count"], "frequency_Hz": result["frequency_Hz"],
                "legacy_analytical_A0_proxy_rate_m_per_cycle": ref["A0_physical_m_per_cycle"],
                "numerical_disposition": ("ACCEPTED_NATIVE_QUALIFICATION" if
                    (result.get("native_qualification") or {}).get("accepted") else "NOT_ACCEPTED"),
                "artifact": str(path.resolve()),
            }
            full = full_window(result)
            finite = finite_window_reference(result, records[result["candidate_id"]])
            if finite:
                # The declared renewal-reward conversion used throughout the
                # search is 5 um/action.  Unlike the legacy acquisition proxy,
                # this clock uses the complete row's hits and correlation time.
                row["stationary_native_A0_rate_m_per_cycle"] = 5e-6 * finite["action_per_cycle"]
            flatten("full", full, row)
            flatten("finite_A0", finite, row)
            if full and finite:
                row["full_over_finite_A0_rate"] = full["rate_m_per_cycle"] / finite["rate_m_per_cycle"]
            rows.append(row)
            if not result["events"]:
                checkpoint = json.loads((path.parent / "checkpoint/high_cycle_live_checkpoint.json").read_text())
                cycles = float(result["final_cycles"])
                censors.append({
                    "candidate_id": result["candidate_id"], "temperature_K": result["temperature_K"],
                    "Kmax_MPa_sqrt_m": result["Kmax_MPa_sqrt_m"], "cycles": cycles,
                    "time_s": cycles / float(result["frequency_Hz"]), "events": 0,
                    "final_extension_m": result["final_extension_m"],
                    "final_continuous_advance_m": result["final_continuous_mpz_progress_m"],
                    "accumulated_action": checkpoint["stochastic"]["hazard_action_current"],
                    "next_action_mark": checkpoint["stochastic"]["hazard_threshold_action"],
                    "terminal_tip_radius_m": checkpoint["diagnostics"]["tip_radius_m"],
                    "terminal_mobile_count": checkpoint["diagnostics"]["mobile_count"],
                    "terminal_retained_count": checkpoint["diagnostics"]["retained_count"],
                    "terminal_K_shield_Pa_sqrt_m": checkpoint["diagnostics"]["active_K_shield_Pa_sqrt_m"],
                    "terminal_gross_source_activity": checkpoint["ledgers"]["mpz.cumulative_gross_source_activity"],
                    "terminal_gross_return_activity": checkpoint["ledgers"]["mpz.cumulative_gross_return_activity"],
                    "homogeneous_poisson_95pct_nominal_5um_rate_upper_m_per_cycle":
                        -math.log(0.05) * 5e-6 / cycles,
                    "bound_scope": "MODEL_CONDITIONAL_HOMOGENEOUS_POISSON_SURROGATE_NOT_MODEL_FREE",
                    "artifact": str(path.resolve()),
                })
    OUT.mkdir(parents=True, exist_ok=True)
    for path, values in ((OUT / "window_observables.csv", rows), (OUT / "zero_event_censors.csv", censors)):
        keys = list(dict.fromkeys(k for row in values for k in row))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); writer.writeheader(); writer.writerows(values)
    payload = {
        "schema": "r10.like-for-like-window-audit/1",
        "endpoint_rule": "first event at or above 20 um through first event at or above 60 um",
        "finite_reference": "same material, initial opening state, retained midpoint waveform, marks, rewards, and endpoints; event times independently advanced on A0 opening clock",
        "stationary_reference": "long-time phase-integrated A0 renewal-reward rate",
        "zero_event_bound_scope": "homogeneous-Poisson nominal-reward surrogate only",
        "row_count": len(rows), "zero_event_count": len(censors),
    }
    (OUT / "window_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2));
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
