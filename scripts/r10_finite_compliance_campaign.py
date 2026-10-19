#!/usr/bin/env python3
"""Run/resume, analyze, plot, and verify the R10 inverse-shape campaign."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from statistics import median
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_opening_reduction_r10 import score_recurrence


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_files(root: Path, name: str) -> list[Path]:
    return sorted(root.rglob(name)) if root.exists() else []


def registry_metadata(root: Path) -> dict[str, dict]:
    metadata = {}
    for path in sorted(root.rglob("*registry.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                identity = row.get("complete_row_sha256")
                if not identity:
                    continue
                metadata[identity] = {key: row.get(key) for key in (
                    "source_material_class", "cohesive_fold_thermal_prediction",
                    "thermal_prediction", "mechanism_prediction",
                    "selected_shape", "selected_mode", "predicted_hcf_K_300K",
                    "predicted_hcf_cyclic_action_per_cycle",
                    "predicted_hcf_local_slope",
                    "predicted_hcf_accessible_phase_fraction")}
    return metadata


def complete_material_rows(root: Path) -> list[dict]:
    """Deduplicate complete serialized materials across projected registries."""
    selected: dict[str, dict] = {}
    keep = (
        "candidate_id", "family", "additive_family", "complete_row_sha256",
        "parent_candidate_id", "parent_complete_row_sha256",
        "plasticity_physics_sha256", "cohesive_parameter_sha256",
        "source_material_candidate_id", "selected_shape", "selected_mode",
        "source_material_class", "mechanism_prediction", "thermal_prediction",
        "cohesive_fold_thermal_prediction", "research_status", "search_design",
        "cohesive_parameter_vector", "parent_complete_parameter_vector",
        "predicted_fold_K_300K", "predicted_hcf_K_300K",
        "predicted_hcf_cyclic_action_per_cycle", "predicted_hcf_local_slope",
        "predicted_hcf_accessible_phase_fraction",
    )
    for path in sorted(root.rglob("*registry.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                identity = row.get("complete_row_sha256")
                if not identity:
                    continue
                record = {key: row.get(key) for key in keep}
                record["registry_source"] = str(path.relative_to(root))
                existing = selected.get(identity)
                # Prefer the richer record when the same immutable material was
                # projected at multiple acquisition stages.
                richness = sum(value not in (None, "") for value in record.values())
                previous = (-1 if existing is None else
                    sum(value not in (None, "") for value in existing.values()))
                if richness > previous:
                    selected[identity] = record
    return sorted(selected.values(), key=lambda row: (
        row.get("candidate_id") or "", row["complete_row_sha256"]))


def retained_11566_controls(repository_root: Path = ROOT) -> dict:
    """Hash the inherited phase-sensitive 11566 controls without rerunning them."""
    base = repository_root/"runs"/"r10_12h_autonomous"
    relative_paths = (
        "hcf_round_32_state_active_11566/f_1.06/result.json",
        "hcf_mechanism_controls_11566/frozen_f_1.06/result.json",
        "hcf_round_32_state_active_11566/f_1.1/result.json",
        "hcf_refinement_11566/f_1.1_phase32_same_seed/result.json",
    )
    rows = []
    for relative in relative_paths:
        path = base/relative
        if not path.is_file():
            rows.append({"source_path": str(path.relative_to(repository_root)),
                         "status": "MISSING_RETAINED_CONTROL"})
            continue
        record = json.loads(path.read_text())
        events = record.get("events") or []
        rows.append({
            "source_path": str(path.relative_to(repository_root)),
            "result_sha256": sha256(path), "status": "RETAINED_EXACT_RESULT",
            **{key: record.get(key) for key in (
                "candidate_id", "complete_row_sha256", "Kmax_MPa_sqrt_m",
                "temperature_K", "R", "frequency_Hz", "phase_count",
                "hazard_seed", "plastic_feedback_reference", "accepted_cycles",
                "accepted_seconds", "physical_event_count", "advance_m",
                "developed_rate_m_per_cycle", "disposition", "resource_stop",
                "numeric_failure")},
            "event_cycles": [event.get("cycle") for event in events],
        })
    return {
        "schema": "r10.retained-11566-phase-control/1",
        "rerun_in_current_campaign": False,
        "interpretation": ("PHASE16_DEVELOPED_RESPONSE_FAILS_SAME_SEED_PHASE32; "
                           "RETAIN_AS_PHASE_SENSITIVITY_CENSOR_NOT_QUALIFIED_RATE"),
        "controls": rows,
    }


def hcf_rows(root: Path) -> list[dict]:
    rows = []
    metadata = registry_metadata(root)
    case_paths = {path.parent: path for path in json_files(root, "result.json")}
    for progress_path in json_files(root, "progress.json"):
        case_paths.setdefault(progress_path.parent, progress_path)
    for case, path in sorted(case_paths.items()):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        standalone_progress = path.name == "progress.json"
        if (not standalone_progress and
                record.get("schema") != "r10.czm-owned-hcf-trajectory/1"):
            continue
        if standalone_progress and not all(record.get(key) is not None for key in (
                "candidate_id", "complete_row_sha256", "accepted_cycles",
                "frequency_Hz", "events")):
            continue
        evidence_state = ("ACCEPTED_PROGRESS_WITHOUT_TERMINAL_RESULT"
                          if standalone_progress else "TERMINAL_RESULT")
        evidence_path = path
        progress_path = case / "progress.json"
        if not standalone_progress and progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {}
            if float(progress.get("accepted_cycles") or 0.) > float(
                    record.get("accepted_cycles") or 0.):
                record = {**record, **progress}
                frequency = float(record.get("frequency_Hz") or 1.)
                record["accepted_seconds"] = float(record["accepted_cycles"])/frequency
                events = record.get("events") or []
                scored = score_recurrence(events,
                    float(record.get("observation_path_m") or 120e-6),
                    bool(record.get("unsupported") or
                         record.get("unsupported_validities")))
                record["physical_event_count"] = len(events)
                record["advance_m"] = (float(events[-1]["extension_m"])
                                       if events else float(record.get("advance_m") or 0.))
                record["developed_rate_m_per_cycle"] = scored["developed_rate_m_per_cycle"]
                record["developed_window"] = scored["developed_window"]
                record["recurrence_outcome"] = scored["outcome"]
                record["disposition"] = "ACCEPTED_RESTART_IN_PROGRESS"
                evidence_state = "ACCEPTED_PROGRESS_NEWER_THAN_RESULT"
                evidence_path = progress_path
        identity = record.get("complete_row_sha256")
        meta = metadata.get(identity, {})
        if standalone_progress:
            frequency = float(record["frequency_Hz"])
            events = record.get("events") or []
            scored = score_recurrence(events,
                float(record.get("observation_path_m") or 120e-6),
                bool(record.get("unsupported") or record.get("unsupported_validities")))
            multiplier_label = case.name.removeprefix("m_").replace("p", ".")
            try:
                multiplier = float(multiplier_label)
            except ValueError:
                multiplier = 1.
            predicted = meta.get("predicted_hcf_K_300K")
            Kmax = None if predicted in (None, "") else float(predicted)*multiplier
            record.update({
                "shape": meta.get("selected_shape"), "mode": meta.get("selected_mode"),
                "Kmax_MPa_sqrt_m": Kmax,
                "DeltaK_MPa_sqrt_m": (None if Kmax is None else
                                        Kmax*(1.-float(record.get("R") or 0.))),
                "accepted_seconds": float(record["accepted_cycles"])/frequency,
                "physical_event_count": len(events),
                "advance_m": (float(events[-1]["extension_m"]) if events else 0.),
                "first_event_cycle": (float(events[0]["cycle"]) if events else None),
                "all_waiting_intervals_cycles": [float(b["cycle"])-float(a["cycle"])
                    for a,b in zip(events,events[1:])],
                "developed_rate_m_per_cycle": scored["developed_rate_m_per_cycle"],
                "developed_window": scored["developed_window"],
                "recurrence_outcome": scored["outcome"],
                "disposition": "ACCEPTED_RESTART_IN_PROGRESS",
                "plastic_feedback_reference": ("EVOLVING_FULL_STATE" if
                    record.get("plastic_feedback", True) else "FROZEN_INITIAL_PLASTIC_STATE"),
            })
        checkpoint_path = case/"checkpoint"/"high_cycle_live_checkpoint.json"
        if not record.get("restart_checkpoint") and checkpoint_path.is_file():
            try:
                record["restart_checkpoint"] = str(checkpoint_path.relative_to(ROOT))
            except ValueError:
                record["restart_checkpoint"] = str(checkpoint_path)
        event_rows = record.get("events") or []
        mobile = [float(event.get("wake_transaction", {}).get(
                    "active_mobile_postcommit") or 0.) for event in event_rows]
        retained = [float(event.get("wake_transaction", {}).get(
                      "active_retained_postcommit") or 0.) for event in event_rows]
        wake_slip = [abs(float(event.get("wake_transaction", {}).get(
                       "wake_slip") or 0.)) for event in event_rows]
        record["maximum_active_mobile_postcommit"] = max(mobile, default=None)
        record["maximum_active_retained_postcommit"] = max(retained, default=None)
        record["maximum_absolute_wake_slip"] = max(wake_slip, default=None)
        rows.append({"path": str(evidence_path.relative_to(root)),
            "evidence_state": evidence_state,
            **meta, **{key: record.get(key) for key in (
            "candidate_id", "complete_row_sha256", "shape", "mode", "temperature_K",
            "R", "frequency_Hz", "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m",
            "hazard_seed",
            "accepted_cycles", "accepted_seconds", "physical_event_count", "advance_m",
            "cpu_seconds", "wall_seconds",
            "first_event_cycle", "all_waiting_intervals_cycles", "inter_event_cycles",
            "developed_rate_m_per_cycle", "developed_window", "disposition",
            "recurrence_outcome", "numeric_failure", "resource_stop",
            "unsupported_validities", "restart_checkpoint", "plastic_feedback_reference",
            "maximum_active_mobile_postcommit",
            "maximum_active_retained_postcommit",
            "maximum_absolute_wake_slip")}})
    return rows


def developed_slope_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        rate = row.get("developed_rate_m_per_cycle")
        amplitude = row.get("DeltaK_MPa_sqrt_m")
        if rate in (None, "", 0, 0.) or amplitude in (None, ""):
            continue
        key = (row.get("candidate_id"), row.get("complete_row_sha256"),
               row.get("temperature_K"), row.get("R"), row.get("frequency_Hz"),
               row.get("plastic_feedback_reference"))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        amplitudes: dict[float, list[dict]] = {}
        for row in group:
            amplitudes.setdefault(float(row["DeltaK_MPa_sqrt_m"]), []).append(row)
        points = []
        for amplitude, replicates in sorted(amplitudes.items()):
            rates = sorted(float(row["developed_rate_m_per_cycle"])
                           for row in replicates)
            points.append((amplitude, median(rates), len(rates)))
        for low, high in zip(points, points[1:]):
            output.append({
                "candidate_id": key[0], "complete_row_sha256": key[1],
                "temperature_K": key[2], "R": key[3], "frequency_Hz": key[4],
                "plastic_feedback_reference": key[5],
                "DeltaK_low_MPa_sqrt_m": low[0],
                "DeltaK_high_MPa_sqrt_m": high[0],
                "rate_low_m_per_cycle": low[1],
                "rate_high_m_per_cycle": high[1],
                "local_log_slope": math.log(high[1]/low[1])/math.log(high[0]/low[0]),
                "low_replicates": low[2], "high_replicates": high[2],
                "connected_rate_width_log10": math.log10(high[1]/low[1]),
            })
    return output


def replicate_rate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        rate = row.get("developed_rate_m_per_cycle")
        amplitude = row.get("DeltaK_MPa_sqrt_m")
        if rate in (None, "", 0, 0.) or amplitude in (None, ""):
            continue
        key = (row.get("candidate_id"), row.get("complete_row_sha256"),
               row.get("temperature_K"), row.get("R"), row.get("frequency_Hz"),
               row.get("plastic_feedback_reference"), float(amplitude))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(group) < 2:
            continue
        rates = sorted(float(row["developed_rate_m_per_cycle"]) for row in group)
        output.append({
            "candidate_id": key[0], "complete_row_sha256": key[1],
            "temperature_K": key[2], "R": key[3], "frequency_Hz": key[4],
            "plastic_feedback_reference": key[5], "DeltaK_MPa_sqrt_m": key[6],
            "independent_streams": len(rates), "observed_rate_min_m_per_cycle": rates[0],
            "observed_rate_median_m_per_cycle": median(rates),
            "observed_rate_max_m_per_cycle": rates[-1],
            "observed_max_over_min": rates[-1]/rates[0],
            "interpretation": "OBSERVED_STREAM_RANGE; NOT_A_CONFIDENCE_INTERVAL",
            "evidence_paths": ";".join(sorted(row["path"] for row in group)),
        })
    return output


def state_effect_rows(rows: list[dict]) -> list[dict]:
    """Construct C=v_full/v_frozen and Delta m only where both exist."""
    output = []
    rates: dict[tuple, dict[str, list[float]]] = {}
    for row in rows:
        rate = row.get("developed_rate_m_per_cycle")
        amplitude = row.get("DeltaK_MPa_sqrt_m")
        feedback = row.get("plastic_feedback_reference")
        if rate in (None, "", 0, 0.) or amplitude in (None, "") or not feedback:
            continue
        key = (row.get("candidate_id"), row.get("complete_row_sha256"),
               row.get("temperature_K"), row.get("R"), row.get("frequency_Hz"),
               float(amplitude))
        rates.setdefault(key, {}).setdefault(feedback, []).append(float(rate))
    for key, modes in rates.items():
        full = modes.get("EVOLVING_FULL_STATE")
        frozen = modes.get("FROZEN_INITIAL_PLASTIC_STATE")
        if not full or not frozen:
            continue
        full_rate, frozen_rate = median(full), median(frozen)
        output.append({"comparison": "RATE_RATIO", "candidate_id": key[0],
            "complete_row_sha256": key[1], "temperature_K": key[2], "R": key[3],
            "frequency_Hz": key[4], "DeltaK_MPa_sqrt_m": key[5],
            "full_rate_m_per_cycle": full_rate,
            "frozen_rate_m_per_cycle": frozen_rate,
            "C_full_over_frozen": full_rate/frozen_rate,
            "full_streams": len(full), "frozen_streams": len(frozen)})
    slopes = developed_slope_rows(rows)
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in slopes:
        key = (row["candidate_id"], row["complete_row_sha256"], row["temperature_K"],
               row["R"], row["frequency_Hz"], row["DeltaK_low_MPa_sqrt_m"],
               row["DeltaK_high_MPa_sqrt_m"])
        grouped.setdefault(key, {})[row["plastic_feedback_reference"]] = row
    for key, modes in grouped.items():
        full = modes.get("EVOLVING_FULL_STATE")
        frozen = modes.get("FROZEN_INITIAL_PLASTIC_STATE")
        if full is None or frozen is None:
            continue
        output.append({"comparison": "SLOPE_DIFFERENCE", "candidate_id": key[0],
            "complete_row_sha256": key[1], "temperature_K": key[2], "R": key[3],
            "frequency_Hz": key[4], "DeltaK_low_MPa_sqrt_m": key[5],
            "DeltaK_high_MPa_sqrt_m": key[6],
            "full_local_log_slope": full["local_log_slope"],
            "frozen_local_log_slope": frozen["local_log_slope"],
            "Delta_m_full_minus_frozen": (full["local_log_slope"]-
                                            frozen["local_log_slope"])})
    return output


def state_prefix_rows(rows: list[dict]) -> list[dict]:
    """Compare matched full/frozen first-event evidence without inventing rates."""
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        feedback = row.get("plastic_feedback_reference")
        amplitude = row.get("DeltaK_MPa_sqrt_m")
        if feedback not in {"EVOLVING_FULL_STATE", "FROZEN_INITIAL_PLASTIC_STATE"}:
            continue
        if amplitude in (None, "") or row.get("hazard_seed") in (None, ""):
            continue
        key = (row.get("candidate_id"), row.get("complete_row_sha256"),
               row.get("temperature_K"), row.get("R"), row.get("frequency_Hz"),
               float(amplitude), row.get("hazard_seed"))
        existing = grouped.setdefault(key, {}).get(feedback)
        if existing is None or float(row.get("accepted_cycles") or 0.) > float(
                existing.get("accepted_cycles") or 0.):
            grouped[key][feedback] = row
    output = []
    for key, modes in sorted(grouped.items(), key=lambda item: str(item[0])):
        full = modes.get("EVOLVING_FULL_STATE")
        frozen = modes.get("FROZEN_INITIAL_PLASTIC_STATE")
        if full is None or frozen is None:
            continue
        full_first = full.get("first_event_cycle")
        frozen_first = frozen.get("first_event_cycle")
        full_exposure = float(full.get("accepted_cycles") or 0.)
        frozen_exposure = float(frozen.get("accepted_cycles") or 0.)
        def intervals(row: dict) -> list[float]:
            values = row.get("all_waiting_intervals_cycles") or []
            if isinstance(values, str):
                try:
                    values = json.loads(values)
                except json.JSONDecodeError:
                    values = []
            return [float(value) for value in values]
        full_intervals = intervals(full)
        frozen_intervals = intervals(frozen)
        paired_intervals = min(len(full_intervals), len(frozen_intervals))
        interval_ratios = [frozen_intervals[index]/full_intervals[index]
                           for index in range(paired_intervals)
                           if full_intervals[index] > 0.]
        frozen_next_lower_bound = None
        if (full_first not in (None, "") and frozen_first not in (None, "") and
                len(full_intervals) > len(frozen_intervals)):
            index = len(frozen_intervals)
            frozen_last = float(frozen_first) + sum(frozen_intervals)
            elapsed = max(0., frozen_exposure-frozen_last)
            if full_intervals[index] > 0.:
                frozen_next_lower_bound = elapsed/full_intervals[index]
        full_next_lower_bound = None
        if (full_first not in (None, "") and frozen_first not in (None, "") and
                len(frozen_intervals) > len(full_intervals)):
            index = len(full_intervals)
            full_last = float(full_first) + sum(full_intervals)
            elapsed = max(0., full_exposure-full_last)
            if frozen_intervals[index] > 0.:
                full_next_lower_bound = elapsed/frozen_intervals[index]
        if full_first not in (None, "") and frozen_first not in (None, ""):
            comparison = "MATCHED_FIRST_EVENT_RATIO"
            ratio = float(frozen_first)/float(full_first)
            lower_bound = None
        elif full_first not in (None, "") and frozen_exposure >= float(full_first):
            comparison = "FROZEN_FIRST_EVENT_DELAY_LOWER_BOUND"
            ratio = None
            lower_bound = frozen_exposure/float(full_first)
        elif frozen_first not in (None, "") and full_exposure >= float(frozen_first):
            comparison = "FULL_FIRST_EVENT_DELAY_LOWER_BOUND"
            ratio = None
            lower_bound = full_exposure/float(frozen_first)
        else:
            comparison = "MATCHED_PREFIX_INSUFFICIENT_EXPOSURE"
            ratio = lower_bound = None
        output.append({
            "comparison": comparison, "candidate_id": key[0],
            "complete_row_sha256": key[1], "temperature_K": key[2], "R": key[3],
            "frequency_Hz": key[4], "DeltaK_MPa_sqrt_m": key[5],
            "hazard_seed": key[6], "full_first_event_cycle": full_first,
            "frozen_first_event_cycle": frozen_first,
            "full_accepted_cycles": full_exposure,
            "frozen_accepted_cycles": frozen_exposure,
            "frozen_over_full_first_event_cycle": ratio,
            "observed_delay_lower_bound": lower_bound,
            "full_inter_event_cycles": full_intervals,
            "frozen_inter_event_cycles": frozen_intervals,
            "matched_inter_event_ratios_frozen_over_full": interval_ratios,
            "frozen_next_interval_ratio_lower_bound": frozen_next_lower_bound,
            "full_next_interval_ratio_lower_bound": full_next_lower_bound,
            "interpretation": "FIRST_EVENT_OR_EXPOSURE_ONLY; NOT_A_DEVELOPED_RATE_EFFECT",
            "full_evidence_path": full.get("path"),
            "frozen_evidence_path": frozen.get("path"),
        })
    return output


def hcf_event_window_rows(root: Path) -> list[dict]:
    """Report fixed spatial windows from the newest accepted case state.

    The canonical 20--60 um window remains the classification observable.  The
    adjacent 60--100 um and aggregate 20--100 um windows are diagnostics for
    within-trajectory stationarity and are never substituted for it.
    """
    metadata = registry_metadata(root)
    cases = {path.parent for path in json_files(root, "result.json")}
    cases.update(path.parent for path in json_files(root, "progress.json"))
    output = []
    for case in sorted(cases):
        result_path, progress_path = case/"result.json", case/"progress.json"
        result = None; evidence_path = None
        if result_path.is_file():
            try:
                candidate = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                candidate = {}
            if candidate.get("schema") == "r10.czm-owned-hcf-trajectory/1":
                result, evidence_path = candidate, result_path
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {}
            if (progress.get("complete_row_sha256") and
                    float(progress.get("accepted_cycles") or 0.) >
                    float((result or {}).get("accepted_cycles") or 0.)):
                result, evidence_path = progress, progress_path
        if result is None:
            continue
        events = result.get("events") or []
        # A fractional-time cascade inside one accepted cycle is a first
        # excursion, not a cyclic growth window, even if it spans 20--60 um.
        accepted_event_cycles = {int(math.floor(float(event["cycle"])+1e-12))
                                 for event in events}
        if len(accepted_event_cycles) < 2:
            continue
        identity = result.get("complete_row_sha256")
        meta = metadata.get(identity, {})
        Kmax = result.get("Kmax_MPa_sqrt_m")
        if Kmax in (None, ""):
            predicted = meta.get("predicted_hcf_K_300K")
            label = case.name.removeprefix("m_").replace("p", ".")
            try:
                multiplier = float(label)
            except ValueError:
                multiplier = 1.
            Kmax = None if predicted in (None, "") else float(predicted)*multiplier
        for lower_um, upper_um, purpose in (
                (20., 60., "CANONICAL_DEVELOPED_WINDOW"),
                (60., 100., "LATE_STATIONARITY_DIAGNOSTIC"),
                (20., 100., "AGGREGATE_STATIONARITY_DIAGNOSTIC")):
            lower_m, upper_m = lower_um*1e-6, upper_um*1e-6
            endpoint_tol = max(1e-15, 1e-12*upper_m)
            low = next((event for event in events
                        if float(event["extension_m"])+endpoint_tol >= lower_m), None)
            high = next((event for event in events
                         if float(event["extension_m"])+endpoint_tol >= upper_m), None)
            if low is None or high is None:
                continue
            duration = float(high["cycle"])-float(low["cycle"])
            if duration <= 0.:
                continue
            output.append({
                "candidate_id": result.get("candidate_id"),
                "complete_row_sha256": identity,
                "temperature_K": result.get("temperature_K"), "R": result.get("R"),
                "frequency_Hz": result.get("frequency_Hz"),
                "Kmax_MPa_sqrt_m": Kmax,
                "DeltaK_MPa_sqrt_m": (None if Kmax is None else
                    float(Kmax)*(1.-float(result.get("R") or 0.))),
                "hazard_seed": result.get("hazard_seed"),
                "plastic_feedback_reference": result.get(
                    "plastic_feedback_reference", "EVOLVING_FULL_STATE" if
                    result.get("plastic_feedback", True) else "FROZEN_INITIAL_PLASTIC_STATE"),
                "window_purpose": purpose, "window_start_um": lower_um,
                "window_end_um": upper_um, "start_cycle": float(low["cycle"]),
                "end_cycle": float(high["cycle"]), "duration_cycles": duration,
                "rate_m_per_cycle": (float(high["extension_m"])-
                    float(low["extension_m"]))/duration,
                "accepted_cycles_at_receipt": result.get("accepted_cycles"),
                "event_count_at_receipt": len(events),
                "evidence_path": str(evidence_path.relative_to(root)),
            })
    return output


def search_specification() -> dict:
    return {
        "schema": "r10.finite-compliance-inverse-shape-search-specification/1",
        "status": "OPT_IN_RESEARCH_NOT_CANONICAL_PROMOTION",
        "shape_families": ["smooth_exponential", "bilinear",
                           "compact_independent"],
        "diagnostic_only_exclusions": ["algebraic_tail_p_1p1"],
        "continuous_material_domain": {
            "Gamma_ref_J_m2": [0.5, 8.0],
            "t_hat_ref_Pa": [8e9, 80e9],
            "zero_traction_barrier_eV_300K": [0.25, 12.0],
            "cohesive_entropy_span_fraction_300_to_1200K": [-0.25, 0.75],
            "attempt_frequency_s": 1e12,
            "compact_rise_fraction": [0.0005, 0.05],
            "compact_normalized_drop_fraction": [0.001, 0.08],
            "compact_tail_level": [0.005, 0.3],
        },
        "physical_numerical_filters": {
            "minimum_Gamma_J_m2_over_temperature_panel": 0.2,
            "maximum_completion_separation_m": 1e-7,
            "minimum_peak_separation_m": 1e-13,
            "finite_positive_work_required": True,
            "actual_finite_environment_branch_required": True,
            "two_sided_complete_cycle_derivative_for_promotion": True,
        },
        "strict_internal_reference": {
            "local_slope_bounds": [3.4, 4.6],
            "minimum_connected_rate_decades": 2.0,
            "not_universal_experimental_definition": True,
        },
        "default_protocol": {"temperature_K": 300.0, "R": 0.1,
                             "frequency_Hz": 1000.0,
                             "patch_advance_m": 1e-6},
        "mechanics_mapping": {
            "plane_strain_modulus_Pa": 400e9/(1.-.22**2),
            "process_zone_m": "Eprime*Gamma(T)/t_hat(T)^2",
            "environment_stiffness_Pa_per_m": "Eprime/(2*process_zone_m)",
            "tip_radius_m": 1e-6,
            "imposed_separation_m":
                "K_MPa_sqrt_m*1e6/(sqrt(2*pi*tip_radius_m)*environment_stiffness)",
            "status": "DERIVED_PER_COMPLETE_COHESIVE_MATERIAL; NOT_A_FIT_PARAMETER",
        },
        "temperature_panel_K": [300, 450, 600, 750, 900, 1050, 1200],
        "immutable_physics": [
            "EXP_floor_dislocation_emission", "Peierls_motion",
            "Taylor_release", "activation_entropy_and_parameter_couplings",
            "signed_history", "geometry_and_energy_ownership",
            "current_high_cycle_state_integration",
        ],
        "bound_provenance": {
            "cohesive_energy_strength_activation_and_compact_shape":
                "DECLARED_CONDITIONAL_RESEARCH_DOMAIN",
            "separation_and_mesh_support": "INHERITED_NUMERICAL_RESEARCH_BOUND",
            "EXP_floor_parameters": "RETAINED_COMPLETE_V2_SOURCE_MATERIALS",
        },
    }


def native_rows(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(root.glob("native*")):
        for path in directory.glob("*/*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record.get("cycle_history"), list):
                continue
            history = record["cycle_history"]
            rows.append({"path": str(path.relative_to(root)), **{key: record.get(key) for key in (
                "candidate_id", "same_complete_material_sha256", "paired_plasticity_sha256",
                "shape", "mode", "temperature_K",
                "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m", "stop_category",
                "unsupported_crossover_encountered")}, "cycles_completed": len(history),
                "event_count": len(record.get("events") or []),
                "extension_m": history[-1].get("extension_m", 0.) if history else 0.})
    return rows


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, sort_keys=True) if isinstance(value, (list, dict))
                                   else value) for key, value in row.items()})


def csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def analytical_frontier(root: Path) -> dict:
    banks = sorted(root.rglob("analytical_bank_32768.csv.gz"))
    if not banks:
        return {"bank": None, "valid": 0, "strict_passes": 0, "best": None}
    bank = next((path for path in banks if "round_4_targeted_tail" in str(path)), banks[-1])
    valid = 0
    strict = 0
    best = None
    with gzip.open(bank, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("domain_pass") != "True":
                continue
            valid += 1
            strict += row.get("strict_internal_reference_pass") == "True"
            score = float(row.get("moderate_connected_width_log10rate") or 0.)
            if best is None or score > best[0]:
                best = (score, row)
    keep = None if best is None else {key: best[1].get(key) for key in (
        "candidate_id", "cohesive_shape", "source_material_candidate_id",
        "mechanism_prediction", "thermal_prediction", "moderate_connected_width_log10K",
        "moderate_connected_width_log10rate", "strict_connected_width_log10rate",
        "hcf_slope_median", "hcf_slope_min", "metastable_load_ratio_300K",
        "completion_cutoff_m", "initial_stiffness_Pa_per_m", "Gamma_ref_J_m2",
        "t_hat_ref_Pa", "activation_area_m2", "zero_traction_barrier_eV_300K",
        "evaluated_cohesive_json", "curve_json")}
    return {"bank": str(bank.relative_to(root)), "bank_sha256": sha256(bank),
            "valid": valid, "strict_passes": strict, "best": keep}


def cyclic_inverse_frontier(root: Path) -> dict:
    tables=sorted(root.rglob("cyclic_inverse_starts.csv"))
    executions=[];best=None;best_current=None
    for path in tables:
        valid=0;strict=0;local_best=None;local_qualified=False
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("domain_pass") != "True":continue
                valid+=1;strict+=row.get("cyclic_strict_internal_reference_pass")=="True"
                local_qualified |= bool(row.get("cyclic_local_slope_perturbation_fraction"))
                score=float(row.get("objective") or math.inf)
                if local_best is None or score<local_best[0]:local_best=(score,row)
                if best is None or score<best[0]:best=(score,row,path)
                if row.get("cyclic_local_slope_perturbation_fraction"):
                    if best_current is None or score<best_current[0]:
                        best_current=(score,row,path)
        executions.append({"table":str(path.relative_to(root)),"valid":valid,
                           "strict_passes":strict,
                           "local_derivative_qualified":local_qualified})
    historical_best=best
    if best_current is not None:
        best=best_current
    keep=None
    if best is not None:
        keep={key:best[1].get(key) for key in (
            "candidate_id","source_material_candidate_id","mechanism_prediction",
            "thermal_prediction","objective","cyclic_target_K_MPa_sqrt_m",
            "cyclic_target_action_per_cycle","cyclic_target_local_slope",
            "cyclic_moderate_connected_width_log10K",
            "cyclic_moderate_connected_width_log10rate",
            "cyclic_strict_connected_width_log10rate",
            "optimized_cohesive_json")}
        keep["table"]=str(best[2].relative_to(root))
    historical_keep=None
    if (historical_best is not None and best is not None and
            (historical_best[2] != best[2] or
             historical_best[1].get("candidate_id") != best[1].get("candidate_id"))):
        historical_keep={"candidate_id":historical_best[1].get("candidate_id"),
            "objective":historical_best[1].get("objective"),
            "cyclic_target_local_slope":historical_best[1].get("cyclic_target_local_slope"),
            "table":str(historical_best[2].relative_to(root)),
            "qualification":"HISTORICAL_SPARSE_SECANT_NOT_CURRENT_FRONTIER"}
    return {"executions":executions,"best":keep,
            "historical_sparse_best":historical_keep,
            "strict_passes":sum(row["strict_passes"] for row in executions)}


def analyze(args) -> int:
    out = args.root / "campaign_delivery"
    hcf = hcf_rows(args.root)
    native = native_rows(args.root)
    materials = complete_material_rows(args.root)
    frontier = analytical_frontier(args.root)
    fractures = []
    for path in sorted(args.root.rglob("ramp_dispositions.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                fractures.append({"path": str(path.relative_to(args.root)), **row})
    two_d = []
    for path in sorted(args.root.rglob("qualification.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("schema") == "r10.opening-alternatives-2d-transfer/1":
            two_d.append({"path": str(path.relative_to(args.root)),
                "candidate_id": record.get("candidate_id"), "scope": record.get("scope"),
                "same_complete_material_sha256": record.get("same_complete_material_sha256"),
                "cases": len(record.get("cases") or []),
                "triggers": [case.get("trigger") for case in record.get("cases") or []]})
    native_ids = {row.get("same_complete_material_sha256") for row in native}
    hcf_ids = {row.get("complete_row_sha256") for row in hcf}
    two_d_ids = {row.get("same_complete_material_sha256") for row in two_d}
    fracture_candidates = {row.get("candidate_id") for row in fractures}
    for material in materials:
        identity = material.get("complete_row_sha256")
        candidate = material.get("candidate_id")
        material.update({
            "has_current_native_1d_evidence": identity in native_ids,
            "has_current_hcf_evidence": identity in hcf_ids,
            "has_same_material_fracture_evidence": candidate in fracture_candidates,
            "has_exact_2d_evidence": identity in two_d_ids,
        })
    csv_write(out / "hcf_results.csv", hcf)
    csv_write(out / "native_developed_slopes.csv", developed_slope_rows(hcf))
    csv_write(out / "native_replicate_rate_bounds.csv", replicate_rate_rows(hcf))
    state_effects = state_effect_rows(hcf)
    csv_write(out / "native_state_effects.csv", state_effects)
    prefix_effects = state_prefix_rows(hcf)
    csv_write(out / "native_state_prefix_comparisons.csv", prefix_effects)
    atomic(out / "native_state_effects_summary.json", {
        "schema": "r10.native-state-effects-summary/1",
        "matched_rate_ratios": sum(row["comparison"] == "RATE_RATIO"
                                   for row in state_effects),
        "matched_slope_differences": sum(row["comparison"] == "SLOPE_DIFFERENCE"
                                         for row in state_effects),
        "matched_prefix_comparisons": len(prefix_effects),
        "empty_interpretation": ("NO_MATCHED_DEVELOPED_FULL_AND_FROZEN_RATES; "
                                 "NO_FICTITIOUS_RATIO_FORMED"),
    })
    csv_write(out / "native_event_windows.csv", hcf_event_window_rows(args.root))
    csv_write(out / "complete_material_vectors.csv", materials)
    csv_write(out / "native_results.csv", native)
    csv_write(out / "fracture_results.csv", fractures)
    atomic(out / "two_d_results.json", two_d)
    atomic(out / "retained_11566_controls.json", retained_11566_controls())
    atomic(out / "search_specification.json", search_specification())
    restarts=[]
    for row in hcf:
        checkpoint=row.get("restart_checkpoint")
        if not checkpoint:continue
        restarts.append({key:row.get(key) for key in (
            "candidate_id","complete_row_sha256","accepted_cycles",
            "accepted_seconds","physical_event_count","advance_m","disposition",
            "resource_stop","restart_checkpoint")})
        checkpoint_path = ROOT/checkpoint
        evidence_path = args.root/row["path"]
        restarts[-1]["checkpoint_exists"]=checkpoint_path.exists()
        restarts[-1]["checkpoint_sha256"]=(sha256(checkpoint_path)
            if checkpoint_path.is_file() else None)
        restarts[-1]["accepted_evidence_sha256"]=(sha256(evidence_path)
            if evidence_path.is_file() else None)
        checkpoint_cycles = None
        if checkpoint_path.is_file():
            try:
                checkpoint_record = json.loads(checkpoint_path.read_text())
                checkpoint_cycles = (checkpoint_record.get("metadata") or {}).get(
                    "accepted_cycles", checkpoint_record.get("cycles_from_engine_time"))
            except (OSError, json.JSONDecodeError):
                pass
        restarts[-1]["checkpoint_accepted_cycles"] = checkpoint_cycles
        receipt_cycles = float(row.get("accepted_cycles") or 0.)
        restarts[-1]["checkpoint_cycle_delta_from_receipt"] = (None
            if checkpoint_cycles is None else float(checkpoint_cycles)-receipt_cycles)
        restarts[-1]["checkpoint_not_ahead_of_receipt"] = (None
            if checkpoint_cycles is None else float(checkpoint_cycles) <= receipt_cycles+1e-6)
        restarts[-1]["result_path"]=row["path"]
    csv_write(out/"restart_inventory.csv",restarts)
    analytical_ids=set()
    for bank in args.root.rglob("analytical_bank_32768.csv.gz"):
        with gzip.open(bank,"rt",newline="") as stream:
            analytical_ids.update(row.get("candidate_id") for row in csv.DictReader(stream))
    summary = {"schema": "r10.finite-compliance-campaign-analysis/1",
        "analytical_frontier": frontier,
        "cyclic_inverse_frontier":cyclic_inverse_frontier(args.root),
        "coverage": {"analytical_allocated": sum(json.loads(path.read_text()).get("allocated_vectors", 0)
                for path in args.root.rglob("analytical_summary.json")),
            "analytical_bank_executions":len(list(args.root.rglob("analytical_bank_32768.csv.gz"))),
            "analytical_unique_candidate_labels":len(analytical_ids),
            "bounded_inverse_start_rows":sum(csv_row_count(path)
                for path in args.root.rglob("inverse_starts.csv")),
            "cyclic_inverse_start_rows":sum(csv_row_count(path)
                for path in args.root.rglob("cyclic_inverse_starts.csv")),
            "matched_shape_isolation_arms":sum(csv_row_count(path)
                for path in args.root.rglob("matched_shape_isolation.csv")),
            "stiffness_sensitivity_rows":sum(csv_row_count(path)
                for path in args.root.rglob("stiffness_sensitivity.csv")),
            "complete_material_vectors": len(materials),
            "state_informed_results": sum(1 for path in args.root.glob("state_informed*/*/*.json")),
            "native_full_state_results": len(native),
            "native_unique_complete_materials": len({row.get("same_complete_material_sha256") for row in native
                                                       if row.get("same_complete_material_sha256")}),
            "hcf_results": len(hcf),
            "hcf_unique_complete_materials": len({row.get("complete_row_sha256") for row in hcf
                                                   if row.get("complete_row_sha256")}),
            "hcf_accepted_cycles": sum(float(row.get("accepted_cycles") or 0.) for row in hcf),
            "hcf_accepted_seconds": sum(float(row.get("accepted_seconds") or 0.) for row in hcf),
            "reported_hcf_cpu_seconds": sum(float(row.get("cpu_seconds") or 0.) for row in hcf),
            "reported_hcf_wall_seconds": sum(float(row.get("wall_seconds") or 0.) for row in hcf),
            "hcf_events": sum(int(row.get("physical_event_count") or 0) for row in hcf),
            "fracture_rows": len(fractures), "two_d_receipts": len(two_d),
            "accepted_restart_records":len(restarts)},
        "hcf_dispositions": {}, "native_stop_categories": {}}
    for row in hcf:
        key = str(row.get("disposition")); summary["hcf_dispositions"][key] = summary["hcf_dispositions"].get(key, 0) + 1
    for row in native:
        key = str(row.get("stop_category")); summary["native_stop_categories"][key] = summary["native_stop_categories"].get(key, 0) + 1
    atomic(out / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2)); return 0


def verify(args) -> int:
    failures = []
    summaries = []
    for path in sorted(args.root.rglob("analytical_summary.json")):
        record = json.loads(path.read_text()); summaries.append(str(path.relative_to(args.root)))
        if record.get("algebraic_tail_registered") is not False:
            failures.append(f"algebraic diagnostic registration flag invalid: {path}")
    for path in sorted(args.root.rglob("*registry.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                shape = str(row.get("selected_shape") or row.get("cohesive_shape") or "")
                if "algebraic" in shape.lower():
                    failures.append(f"diagnostic algebraic tail registered in {path}")
                    break
    hcf = hcf_rows(args.root)
    for row in hcf:
        checkpoint = row.get("restart_checkpoint")
        if checkpoint and not (ROOT / checkpoint).exists():
            failures.append(f"missing restart checkpoint: {checkpoint}")
        elif checkpoint:
            try:
                state = json.loads((ROOT/checkpoint).read_text())
                checkpoint_cycles = (state.get("metadata") or {}).get(
                    "accepted_cycles", state.get("cycles_from_engine_time"))
                receipt_cycles = float(row.get("accepted_cycles") or 0.)
                if (checkpoint_cycles is not None and
                        float(checkpoint_cycles) > receipt_cycles+1e-6):
                    failures.append(
                        f"checkpoint ahead of accepted receipt: {checkpoint}")
            except (OSError, json.JSONDecodeError):
                failures.append(f"unreadable restart checkpoint: {checkpoint}")
        if row.get("numeric_failure"):
            failures.append(f"numeric failure in {row['path']}: {row['numeric_failure']}")
    payload = {"schema": "r10.finite-compliance-campaign-verification/1",
        "passed": not failures, "failures": failures,
        "analytical_summaries_checked": summaries, "hcf_results_checked": len(hcf),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip()}
    atomic(args.root / "campaign_delivery" / "verification.json", payload)
    print(json.dumps(payload, indent=2)); return 0 if not failures else 1


def plot(args) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from arrhenius_fracture.cohesive_arrhenius import (
        BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
        cohesive_shape_response,
    )
    from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
        connected_barrier, cyclic_rate_curve, imposed_from_K, shape_coordinates,
    )
    from scripts.run_r10_finite_compliance_inverse_shape import _thermal
    from scripts.run_r10_opening_alternatives import mechanics

    out=args.root/"campaign_delivery"/"figures";out.mkdir(parents=True,exist_ok=True)
    frontier=analytical_frontier(args.root);best=frontier.get("best")
    if best:
        curve=json.loads(best["curve_json"]);K=np.asarray([p["K_MPa_sqrt_m"] for p in curve])
        rate=np.asarray([p["events_per_cycle_peak_rate"] for p in curve])
        slope=np.asarray([np.nan if p["local_log_slope"] is None else p["local_log_slope"] for p in curve])
        fig,axes=plt.subplots(2,1,figsize=(6.2,6.2),sharex=True)
        axes[0].semilogy(K,rate,"o-",color="#1f4e79",ms=3,lw=1.4)
        axes[0].set_ylabel("Peak opening action (cycle$^{-1}$)")
        axes[1].plot(K,slope,"o-",color="#a23b2a",ms=3,lw=1.4)
        axes[1].axhspan(3.4,4.6,color="#4c956c",alpha=.18)
        axes[1].set_xlabel(r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)")
        axes[1].set_ylabel(r"Local $d\ln v/d\ln K$")
        fig.tight_layout();fig.savefig(out/"analytical_rate_slope.png",dpi=240);fig.savefig(out/"analytical_rate_slope.pdf");plt.close(fig)

        cohesive=json.loads(best["evaluated_cohesive_json"]);thermal=_thermal(cohesive)
        fig,axes=plt.subplots(2,1,figsize=(6.2,6.2),sharex=True)
        colors={SMOOTH_EXPONENTIAL:"#1f77b4",BILINEAR:"#ff7f0e",COMPACT_INDEPENDENT:"#2ca02c"}
        labels={SMOOTH_EXPONENTIAL:"Smooth exponential",BILINEAR:"Bilinear",COMPACT_INDEPENDENT:"Compact research"}
        maximum=0.
        data=[]
        for shape in (SMOOTH_EXPONENTIAL,BILINEAR,COMPACT_INDEPENDENT):
            coords=shape_coordinates(shape,thermal,300.);maximum=max(maximum,coords["completion_cutoff_m"])
            cutoff=float(coords["completion_cutoff_m"]);peak=float(coords["peak_separation_m"])
            d=np.unique(np.r_[0.,peak,np.geomspace(max(cutoff*1e-9,1e-20),cutoff,1400)])
            traction,_,potential=cohesive_shape_response(shape,d,thermal,300.)
            data.append((shape,d,np.asarray(traction),np.asarray(potential)))
        for shape,d,traction,potential in data:
            axes[0].plot(d*1e9,traction/1e9,color=colors[shape],lw=1.5,label=labels[shape])
            axes[1].plot(d*1e9,potential,color=colors[shape],lw=1.5)
        axes[0].set_ylabel("Traction (GPa)");axes[0].legend(frameon=False)
        axes[1].set_xlabel("Separation (nm)");axes[1].set_ylabel(r"Potential (J m$^{-2}$)")
        fig.tight_layout();fig.savefig(out/"potential_traction.png",dpi=240);fig.savefig(out/"potential_traction.pdf");plt.close(fig)

        _,kenv=mechanics(thermal,300.);shape=best["cohesive_shape"]
        sample=[float(curve[i]["K_MPa_sqrt_m"]) for i in (0,len(curve)//3,2*len(curve)//3)]
        prepared=[];largest_root=0.
        for index,Kvalue in enumerate(sample):
            imposed=imposed_from_K(Kvalue,kenv)
            points=connected_barrier(imposed,kenv,thermal,300.,shape)["stationary"]
            largest_root=max(largest_root,max((float(point["delta_m"]) for point in points),default=0.))
            prepared.append((index,Kvalue,imposed,points))
        dmax=max(2.*largest_root,5.*shape_coordinates(shape,thermal,300.)["peak_separation_m"])
        d=np.unique(np.r_[0.,np.geomspace(max(dmax*1e-8,1e-20),dmax,1600)])
        fig,ax=plt.subplots(figsize=(6.2,4.4))
        for index,Kvalue,imposed,points in prepared:
            _,_,potential=cohesive_shape_response(shape,d,thermal,300.)
            landscape=np.asarray(potential)+.5*kenv*(imposed-d)**2
            landscape=landscape-float(np.min(landscape))
            ax.plot(d*1e9,landscape,label=fr"{Kvalue:.1f} MPa$\sqrt{{\mathrm{{m}}}}$",lw=1.4)
            for point in points:
                marker="o" if point["kind"]=="MINIMUM" else "x"
                raw_min=float(np.min(np.asarray(potential)+.5*kenv*(imposed-d)**2))
                ax.plot(float(point["delta_m"])*1e9,float(point["energy_J_m2"])-raw_min,
                        marker,color=f"C{index}",ms=5)
        ax.set_xlabel("Separation (nm)");ax.set_ylabel(r"Loaded potential (J m$^{-2}$)")
        ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"finite_compliance_landscape.png",dpi=240);fig.savefig(out/"finite_compliance_landscape.pdf");plt.close(fig)

    all_hcf=hcf_rows(args.root)
    records=[row for row in all_hcf if row.get("candidate_id")=="R10_INV_21321__SM" and
             float(row.get("temperature_K") or 0.)==300. and float(row.get("R") or 0.)==.1 and
             float(row.get("frequency_Hz") or 0.)==1000. and
             row.get("plastic_feedback_reference")=="EVOLVING_FULL_STATE"]
    if records:
        byK={}
        for row in records:
            K=float(row["Kmax_MPa_sqrt_m"]);byK[K]=row
        fig,ax=plt.subplots(figsize=(6.2,4.4))
        floor=1e-14
        for K,row in sorted(byK.items()):
            rate=row.get("developed_rate_m_per_cycle");disposition=row.get("disposition")
            if rate is not None:
                ax.semilogy(K,float(rate),"o",color="#a23b2a")
            elif disposition=="PHYSICAL_RUNOUT_TO_DECLARED_HORIZON":
                ax.semilogy(K,floor,"v",mfc="none",mec="#1f4e79")
            elif disposition in {"TARGET_TRUNCATED_CASCADE","UNRESOLVED_OPENING_PHYSICS"}:
                ax.semilogy(K,1e-4,"^",mfc="none",mec="#6a3d9a")
            else:
                ax.semilogy(K,max(float(row.get("advance_m") or 0.)/max(float(row.get("accepted_cycles") or 1.),1.),floor),"s",mfc="none",mec="#444444")
        ax.set_xlabel(r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)")
        ax.set_ylabel(r"Developed $da/dN$ (m cycle$^{-1}$)")
        fig.tight_layout();fig.savefig(out/"native_amplitude_outcomes.png",dpi=240);fig.savefig(out/"native_amplitude_outcomes.pdf");plt.close(fig)

    compact_id="R10_INVERSE_OPT_027__CM"
    compact_records=[row for row in all_hcf if row.get("candidate_id")==compact_id and
        float(row.get("temperature_K") or 0.)==300. and float(row.get("R") or 0.)==.1 and
        float(row.get("frequency_Hz") or 0.)==1000. and
        row.get("plastic_feedback_reference")=="EVOLVING_FULL_STATE"]
    registries=sorted(args.root.rglob("projected_registry.csv"))
    compact_registry=None
    for registry in registries:
        with registry.open(newline="") as stream:
            match=next((row for row in csv.DictReader(stream)
                        if row.get("candidate_id")==compact_id),None)
        if match is not None:
            compact_registry=match;break
    if compact_records and compact_registry is not None:
        cohesive=json.loads(compact_registry["cohesive_parameter_vector"])
        thermal=_thermal(cohesive);_,kenv=mechanics(thermal,300.)
        base=float(compact_registry["predicted_hcf_K_300K"])
        K=np.geomspace(.9*base,1.25*base,61)
        analytical=cyclic_rate_curve(K,kenv,thermal,300.,COMPACT_INDEPENDENT,
            float(cohesive["activation_area_m2"]),
            float(cohesive["attempt_frequency_s"]),R=.1,frequency_Hz=1000.,phases=32)
        action=np.asarray([row["opening_action_per_cycle"] for row in analytical["rows"]])
        slope=np.asarray([np.nan if row["local_log_slope"] is None else row["local_log_slope"]
                          for row in analytical["rows"]])
        fig,axes=plt.subplots(2,1,figsize=(6.4,6.4),sharex=True)
        axes[0].semilogy(K,np.maximum(action*1e-6,1e-16),color="#1f4e79",lw=1.5,
                         label="Frozen complete-cycle reference")
        developed_by_K={}
        for row in sorted(compact_records,key=lambda item:float(item["Kmax_MPa_sqrt_m"])):
            kval=float(row["Kmax_MPa_sqrt_m"]);rate=row.get("developed_rate_m_per_cycle")
            if rate not in (None,"",0,0.):
                developed_by_K.setdefault(kval,[]).append(float(rate))
            elif row.get("disposition")=="PHYSICAL_RUNOUT_TO_DECLARED_HORIZON":
                axes[0].semilogy(kval,1e-16,"v",mfc="none",mec="#1f4e79")
            else:
                axes[0].semilogy(kval,1e-15,"s",mfc="none",mec="#555555")
        for index,(kval,rates) in enumerate(sorted(developed_by_K.items())):
            centre=median(rates);low=centre-min(rates);high=max(rates)-centre
            axes[0].errorbar(kval,centre,yerr=np.asarray([[low],[high]]),fmt="o",
                color="#a23b2a",ms=5,capsize=3,
                label="Native developed window" if index==0 else None)
        axes[0].set_ylabel(r"Opening / developed rate (m cycle$^{-1}$)")
        axes[0].legend(frameon=False)
        axes[1].plot(K,slope,color="#1f4e79",lw=1.5,label="Analytical local slope")
        for row in developed_slope_rows(compact_records):
            midpoint=math.sqrt(float(row["DeltaK_low_MPa_sqrt_m"])*
                               float(row["DeltaK_high_MPa_sqrt_m"]))/.9
            axes[1].plot(midpoint,float(row["local_log_slope"]),"o",color="#a23b2a")
        axes[1].axhspan(3.4,4.6,color="#4c956c",alpha=.18)
        axes[1].set_xlabel(r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)")
        axes[1].set_ylabel(r"Local $d\ln(da/dN)/d\ln K$")
        fig.tight_layout();fig.savefig(out/"compact_027_analytical_native.png",dpi=240)
        fig.savefig(out/"compact_027_analytical_native.pdf");plt.close(fig)

    full_path=args.root/"hcf_bounded_inverse_top3"/compact_id/"m_1p00000"/"progress.json"
    frozen_path=args.root/"hcf_bounded_inverse_027_frozen"/compact_id/"m_1p00000"/"progress.json"
    if full_path.is_file() and frozen_path.is_file():
        fig,ax=plt.subplots(figsize=(6.2,4.4))
        for path,label,style in ((full_path,"Evolving full state","-"),
                                 (frozen_path,"Frozen initial plastic state","--")):
            record=json.loads(path.read_text());events=record.get("events") or []
            x=[0.]+[float(event["cycle"]) for event in events]
            y=[0.]+[float(event["extension_m"])*1e6 for event in events]
            ax.step(x,y,where="post",lw=1.5,ls=style,label=label)
        ax.set_xlabel("Accepted cycles");ax.set_ylabel("Cohesive advance (micrometres)")
        ax.legend(frameon=False);fig.tight_layout()
        fig.savefig(out/"compact_027_full_frozen_events.png",dpi=240)
        fig.savefig(out/"compact_027_full_frozen_events.pdf");plt.close(fig)

    ramp=args.root/"fracture_bounded_inverse_top3"/"ramp_dispositions.csv"
    if ramp.is_file():
        with ramp.open(newline="") as stream:ramp_rows=list(csv.DictReader(stream))
        fig,ax=plt.subplots(figsize=(6.4,4.5))
        colors={"R10_INVERSE_OPT_002__CM":"#1f77b4",
                "R10_INVERSE_OPT_008__CM":"#ff7f0e",
                "R10_INVERSE_OPT_027__CM":"#2ca02c"}
        for candidate in colors:
            group=sorted((row for row in ramp_rows if row["candidate_id"]==candidate),
                         key=lambda row:float(row["temperature_K"]))
            resolved=[row for row in group if row.get("first_committed_separation_K_MPa_sqrt_m")]
            ax.plot([float(row["temperature_K"]) for row in resolved],
                    [float(row["first_committed_separation_K_MPa_sqrt_m"]) for row in resolved],
                    "o-",lw=1.3,ms=4,color=colors[candidate],label=candidate.replace("R10_INVERSE_OPT_",""))
            censored=[row for row in group if row.get("status")=="KMAX_CENSOR"]
            if censored:
                ax.plot([float(row["temperature_K"]) for row in censored],[200.]*len(censored),
                        "^",mfc="none",mec=colors[candidate])
            unsupported=[row for row in group if row.get("status")=="UNSUPPORTED_MIXED_CROSSOVER"]
            if unsupported:
                ax.plot([float(row["temperature_K"]) for row in unsupported],[200.]*len(unsupported),
                        "x",color=colors[candidate])
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"Committed separation / censor $K$ (MPa$\sqrt{\mathrm{m}}$)")
        ax.legend(frameon=False);fig.tight_layout()
        fig.savefig(out/"bounded_inverse_same_material_fracture.png",dpi=240)
        fig.savefig(out/"bounded_inverse_same_material_fracture.pdf");plt.close(fig)

    stiffness_table=args.root/"stiffness_sensitivity_bounded_trio"/"stiffness_sensitivity.csv"
    if stiffness_table.is_file():
        with stiffness_table.open(newline="") as stream:
            stiffness_rows=list(csv.DictReader(stream))
        fig,axes=plt.subplots(2,1,figsize=(6.2,6.2),sharex=True)
        colors={"R10_INVERSE_OPT_002__CM":"#1f77b4",
                "R10_INVERSE_OPT_008__CM":"#ff7f0e",
                "R10_INVERSE_OPT_027__CM":"#2ca02c"}
        for candidate,color in colors.items():
            group=sorted((row for row in stiffness_rows
                          if row["candidate_id"]==candidate and row["status"]=="RESOLVED"),
                         key=lambda row:float(row["stiffness_factor"]))
            x=[float(row["stiffness_factor"]) for row in group]
            axes[0].plot(x,[float(row["minimum_local_slope"]) for row in group],
                         "o-",color=color,lw=1.4,ms=4,
                         label=candidate.replace("R10_INVERSE_OPT_",""))
            axes[1].plot(x,[float(row["moderate_connected_width_log10rate"])
                            for row in group],"o--",color=color,lw=1.2,ms=4)
            axes[1].plot(x,[float(row["strict_connected_width_log10rate"])
                            for row in group],"o-",color=color,lw=1.5,ms=4)
        for ax in axes: ax.axvline(1.,color="#333333",lw=1.,ls=":")
        axes[0].axhspan(3.4,4.6,color="#4c956c",alpha=.18)
        axes[0].set_xscale("log",base=2);axes[0].set_ylabel("Minimum local slope")
        axes[0].legend(frameon=False)
        axes[1].axhline(2.,color="#333333",lw=1.,ls=":")
        axes[1].legend(handles=[
            Line2D([0],[0],color="#555555",lw=1.2,ls="--",
                   label="Moderate 2--8 slope width"),
            Line2D([0],[0],color="#555555",lw=1.5,ls="-",
                   label="Strict 3.4--4.6 width")],frameon=False)
        axes[1].set_xlabel(r"Diagnostic stiffness factor $k_e/k_{e,derived}$")
        axes[1].set_ylabel("Connected rate width (decades)")
        fig.tight_layout();fig.savefig(out/"stiffness_sensitivity.png",dpi=240)
        fig.savefig(out/"stiffness_sensitivity.pdf");plt.close(fig)

    print(json.dumps({"figure_directory":str(out),"files":sorted(p.name for p in out.iterdir())},indent=2));return 0


def execute_plan(args) -> int:
    plan = json.loads(args.plan.read_text())
    receipts = []
    for task in plan["tasks"]:
        sentinel = ROOT / task["sentinel"]
        if args.resume and sentinel.exists():
            receipts.append({"id": task["id"], "status": "SKIPPED_COMPLETE",
                             "sentinel": task["sentinel"]}); continue
        log = args.root / "campaign_delivery" / "logs" / f"{task['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run(task["argv"], cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        receipts.append({"id": task["id"], "status": "COMPLETE" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode, "sentinel": task["sentinel"],
            "sentinel_exists": sentinel.exists(), "log": str(log.relative_to(args.root))})
        atomic(args.root / "campaign_delivery" / "plan_receipts.json", receipts)
        if result.returncode and not args.keep_going:
            break
    print(json.dumps(receipts, indent=2)); return 0 if all(r["status"] != "FAILED" for r in receipts) else 1


def status(args) -> int:
    payload = {"analytical_summaries": len(json_files(args.root, "analytical_summary.json")),
        "state_results": sum(1 for _ in args.root.glob("state_informed*/*/*.json")),
        "native_results": len(native_rows(args.root)), "hcf_results": len(hcf_rows(args.root)),
        "hcf_progress_files": len(json_files(args.root, "progress.json")),
        "fracture_results": sum(1 for _ in args.root.rglob("ramp_dispositions.csv")),
        "two_d_receipts": sum(1 for _ in args.root.rglob("qualification.json"))}
    print(json.dumps(payload, indent=2)); return 0


def execution_timing(args) -> int:
    def parse(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("campaign timing requires an explicit UTC offset")
        return parsed.astimezone(timezone.utc)
    start = parse(args.start_utc)
    deadline = parse(args.deadline_utc)
    observed = parse(args.end_utc) if args.end_utc else datetime.now(timezone.utc)
    elapsed = (observed-start).total_seconds()
    payload = {
        "schema": "r10.finite-compliance-campaign-timing/1",
        "campaign_start_utc": start.isoformat().replace("+00:00", "Z"),
        "campaign_deadline_utc": deadline.isoformat().replace("+00:00", "Z"),
        "observation_utc": observed.isoformat().replace("+00:00", "Z"),
        "productive_elapsed_wall_seconds": elapsed,
        "declared_allocation_seconds": (deadline-start).total_seconds(),
        "allocation_elapsed": observed >= deadline,
        "campaign_status": "FINAL" if args.final else "ACTIVE",
        "interpretation": ("Wall time covers continuous development, numerical execution, "
                           "analysis, and validation; accepted simulated exposure and CPU "
                           "receipts are reported separately."),
    }
    atomic(args.root/"campaign_delivery"/"execution_timing.json", payload)
    print(json.dumps(payload, indent=2)); return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
        default=Path("runs/r10_inverse_shape_12h"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("status", status), ("analyze", analyze),
                           ("verify", verify), ("plot", plot)):
        child = sub.add_parser(name); child.set_defaults(function=function)
    child = sub.add_parser("run")
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--resume", action="store_true")
    child.add_argument("--keep-going", action="store_true")
    child.set_defaults(function=execute_plan)
    child = sub.add_parser("timing")
    child.add_argument("--start-utc", required=True)
    child.add_argument("--deadline-utc", required=True)
    child.add_argument("--end-utc")
    child.add_argument("--final", action="store_true")
    child.set_defaults(function=execution_timing)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
