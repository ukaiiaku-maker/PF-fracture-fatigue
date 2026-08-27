#!/usr/bin/env python3
"""Regenerate V2 transaction, avalanche, onset, and campaign summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def nested_sum(value: Any) -> float:
    if isinstance(value, list):
        return sum(nested_sum(item) for item in value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def load_json_or_zstd(path: Path) -> dict[str, Any]:
    if path.suffix == ".zst":
        raw = subprocess.run(
            ["zstd", "-q", "-dc", str(path)], check=True, capture_output=True
        ).stdout
    else:
        raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"observer artifact is not an object: {path}")
    return document


def observer_profiles(case_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = (
        case_root / "canonical_pf_state_observer_v2.json.zst",
        case_root / "v10_2_17_final_signed_stochastic_stack.json.zst",
        case_root / "v10_2_17_final_signed_stochastic_stack.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"required event-boundary observer artifact is absent: {case_root}")
    document = load_json_or_zstd(path)
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"observer records are absent: {path}")
    fired = [(index, row) for index, row in enumerate(records)
             if isinstance(row, dict) and bool(row.get("fired"))]
    return fired, {
        "observer_artifact_path": str(path),
        "observer_artifact_sha256": sha256(path),
        "observer_record_count": len(records),
        "observer_fired_record_count": len(fired),
        "observer_schema": document.get("schema"),
    }


def profile_summary(case_id: str, event_index: int, record_index: int,
                    record: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    active_mobile = nested_sum(record.get("mobile_active_by_system_bin", []))
    active_retained = nested_sum(record.get("retained_active_by_system_bin", []))
    wake_mobile = nested_sum(record.get("mobile_wake_by_system_bin", []))
    wake_retained = nested_sum(record.get("retained_wake_by_system_bin", []))
    developed_mobile = float(record.get("developed_state_mobile_count", math.nan))
    developed_retained = float(record.get("developed_state_retained_count", math.nan))
    rates = record.get("anisotropic_lambda_emit_by_system_s", [])
    selected_index = math.nan
    selected_name: Any = ""
    if isinstance(rates, list) and rates:
        selected_index = max(range(len(rates)), key=lambda index: float(rates[index]))
        names = record.get("anisotropic_channel_names", [])
        if isinstance(names, list) and selected_index < len(names):
            selected_name = names[selected_index]
    return {
        "case_id": case_id,
        "event_transaction_index": event_index,
        "observer_record_index": record_index,
        **artifact,
        "observer_event_index": record.get("hazard_event_index"),
        "observer_time_s": record.get("time_s"),
        "tip_radius_um": float(record.get("persistent_tip_radius_m", math.nan)) * 1e6,
        "front_width_um": float(record.get("persistent_site_front_width_m", math.nan)) * 1e6,
        "mobile_count": developed_mobile,
        "retained_count": developed_retained,
        "retained_fraction": record.get("developed_state_retained_fraction", math.nan),
        "active_mobile_profile_sum": active_mobile,
        "active_retained_profile_sum": active_retained,
        "wake_mobile_profile_sum": wake_mobile,
        "wake_retained_profile_sum": wake_retained,
        "mobile_profile_conservation_error": active_mobile - developed_mobile,
        "retained_profile_conservation_error": active_retained - developed_retained,
        "multiplicity_per_system": record.get("persistent_site_multiplicity_per_system", math.nan),
        "backstress_Pa": record.get("persistent_sigma_back_Pa", math.nan),
        "signed_active_shielding_MPa_sqrt_m": float(record.get("active_K_shield_signed_Pa_sqrt_m", math.nan)) / 1e6,
        "signed_wake_shielding_MPa_sqrt_m": float(record.get("wake_K_shield_signed_Pa_sqrt_m", math.nan)) / 1e6,
        "tensor_probe_reliable": record.get("anisotropic_drive_reliable"),
        "maximum_emission_rate_system_index": selected_index,
        "maximum_emission_rate_system_name": selected_name,
        "opening_tensor_Pa_sha256": canonical_json_sha(record.get("opening_tensor_Pa")),
        "channel_tensors_Pa_sha256": canonical_json_sha(record.get("channel_tensors_Pa")),
        "profile_fingerprint_sha256": canonical_json_sha(record),
    }


def event_transactions(case_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    launch = json.loads((case_root / "canonical_case_launch.json").read_text())
    temperature = int(float(launch["temperature_K"]))
    steps_path = case_root / f"steps_{temperature:04d}K.csv"
    with steps_path.open(newline="") as stream:
        steps = list(csv.DictReader(stream))
    if not steps:
        raise ValueError(f"empty steps file: {steps_path}")
    times: list[float] = []
    elapsed = 0.0
    for step in steps:
        elapsed += number(step, "dt_cur_s", 0.0)
        times.append(elapsed)
    positions = [i for i, step in enumerate(steps) if number(step, "n_fire", 0.0) > 0.0]
    fired_records, observer_artifact = observer_profiles(case_root)
    if len(fired_records) != len(positions):
        raise ValueError(
            f"event/observer closure failed in {case_root}: "
            f"steps={len(positions)} observer={len(fired_records)}"
        )
    profiles = [
        profile_summary(launch["case_id"], index, record_index, record, observer_artifact)
        for index, (record_index, record) in enumerate(fired_records)
    ]
    transactions: list[dict[str, Any]] = []
    avalanche_id = 0
    state_fields = [
        "N_em", "N_em_pre_renewal", "N_em_shed_to_wake", "sigma_back_Pa",
        "mpz_K_shield_Pa_sqrt_m", "mpz_mobile_count", "mpz_retained_count",
        "mpz_available_site_fraction", "mpz_local_slip_count", "mpz_escaped_total",
        "mpz_recovered_total", "mpz_wake_retained_total",
    ]
    for index, pos in enumerate(positions):
        event = steps[pos]
        profile = profiles[index]
        before = steps[max(pos - 1, 0)]
        if index:
            prior = positions[index - 1]
            between = steps[prior + 1: pos + 1]
            if any(number(item, "adaptive_frac", 0.0) >= 1.0 - 1e-12 for item in between):
                avalanche_id += 1
        next_pos = positions[index + 1] if index + 1 < len(positions) else None
        reload_slice = [] if next_pos is None else steps[pos + 1: next_pos + 1]
        fingerprint = hashlib.sha256(json.dumps(
            {key: number(event, key) for key in state_fields}, sort_keys=True,
            separators=(",", ":"), allow_nan=True,
        ).encode()).hexdigest()
        transactions.append({
            "case_id": launch["case_id"], "matrix": launch["matrix"],
            "material_class": launch["material_class"], "candidate_id": launch["candidate_id"],
            "temperature_K": launch["temperature_K"], "theta_deg": launch["theta_deg"],
            "rate_tag": launch["rate_tag"], "loading_rate_factor": launch["loading_rate_factor"],
            "seed": launch["seed"], "event_transaction_index": index,
            "physical_avalanche_index": avalanche_id, "pre_event_step": int(number(event, "step", pos)),
            "pre_event_time_s": times[pos], "pre_event_opening_m": number(event, "Uapp_m"),
            "pre_event_native_J_J_per_m2": number(event, "J_effective_direct_J_per_m2"),
            "pre_event_native_KJ_MPa_sqrt_m": number(event, "KJ_Pa_sqrtm") / 1e6,
            "pre_event_projected_extension_um": number(before, "crack_extension_m") * 1e6,
            "post_event_projected_extension_um": number(event, "crack_extension_m") * 1e6,
            "event_extension_um": number(event, "da_block_m") * 1e6,
            "tip_radius_um": profile["tip_radius_um"],
            "front_width_um": profile["front_width_um"],
            "mobile_count": profile["mobile_count"],
            "retained_count": profile["retained_count"],
            "multiplicity": profile["multiplicity_per_system"],
            "backstress_Pa": profile["backstress_Pa"],
            "signed_shielding_MPa_sqrt_m": profile["signed_active_shielding_MPa_sqrt_m"],
            "tensor_probe_reliable": profile["tensor_probe_reliable"],
            "maximum_emission_rate_system_index": profile["maximum_emission_rate_system_index"],
            "maximum_emission_rate_system_name": profile["maximum_emission_rate_system_name"],
            "observer_profile_fingerprint_sha256": profile["profile_fingerprint_sha256"],
            "process_zone_state_fingerprint": fingerprint,
            "reload_time_to_next_event_s": math.nan if next_pos is None else times[next_pos] - times[pos],
            "reload_opening_to_next_event_m": math.nan if next_pos is None else number(steps[next_pos], "Uapp_m") - number(event, "Uapp_m"),
            "certified_reload_before_next_event": bool(next_pos is not None and any(number(item, "adaptive_frac", 0.0) >= 1.0 - 1e-12 for item in reload_slice)),
            "right_censored_at_target": index == len(positions) - 1,
            "source_steps_sha256": sha256(steps_path),
        })
    achieved = number(steps[-1], "crack_extension_m") * 1e6
    return transactions, {**launch, "achieved_extension_um": achieved, "steps_sha256": sha256(steps_path),
                          "numerical_event_count": len(positions), **observer_artifact}, profiles


def physical_avalanches(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in transactions:
        grouped.setdefault((row["case_id"], row["physical_avalanche_index"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (_, avalanche_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["event_transaction_index"])
        first, last = rows[0], rows[-1]
        output.append({
            **{key: first[key] for key in ("case_id", "matrix", "material_class", "candidate_id", "temperature_K", "theta_deg", "rate_tag", "loading_rate_factor", "seed")},
            "physical_avalanche_index": avalanche_id,
            "first_event_transaction_index": first["event_transaction_index"],
            "last_event_transaction_index": last["event_transaction_index"],
            "event_transaction_count": len(rows),
            "start_extension_um": first["pre_event_projected_extension_um"],
            "end_extension_um": last["post_event_projected_extension_um"],
            "avalanche_extension_um": sum(row["event_extension_um"] for row in rows),
            "onset_opening_m": first["pre_event_opening_m"],
            "onset_native_J_J_per_m2": first["pre_event_native_J_J_per_m2"],
            "onset_native_KJ_MPa_sqrt_m": first["pre_event_native_KJ_MPa_sqrt_m"],
            "target_right_censored": last["right_censored_at_target"],
            "grouping_rule": "CONTIGUOUS_EVENTS_WITHOUT_INTERVENING_FULL_ACCEPTED_LOADING_INTERVAL",
        })
    return output


def analyze(campaign_root: Path, out: Path, *, expected_case_count: int | None = None,
            one_d_results: Path | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    transactions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    skipped_incomplete: list[str] = []
    for launch_path in sorted(campaign_root.glob("*/canonical_case_launch.json")):
        result_path = launch_path.parent / "canonical_case_result.json"
        if not result_path.is_file():
            skipped_incomplete.append(launch_path.parent.name)
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") not in {"COMPLETE", "REUSED_COMPLETE"}:
            skipped_incomplete.append(launch_path.parent.name)
            continue
        tx, case, case_profiles = event_transactions(launch_path.parent)
        transactions.extend(tx)
        cases.append(case)
        profiles.extend(case_profiles)
    if not cases:
        raise ValueError(f"no canonical cases found under {campaign_root}")
    avalanches = physical_avalanches(transactions)
    onset_keys = {(row["case_id"], row["first_event_transaction_index"]) for row in avalanches}
    onset: list[dict[str, Any]] = []
    interior: list[dict[str, Any]] = []
    for row in transactions:
        if (row["case_id"], row["event_transaction_index"]) in onset_keys:
            item = dict(row)
            item["onset_role"] = "INITIAL_ONSET_PRE_EVENT" if row["physical_avalanche_index"] == 0 else "REINITIATION_ONSET_PRE_EVENT"
            onset.append(item)
        else:
            item = dict(row)
            item["semantic_role"] = "INTERIOR_PRE_EVENT_NOT_RESISTANCE_POINT"
            interior.append(item)
    ava_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in avalanches:
        ava_by_case.setdefault(row["case_id"], []).append(row)
    profiles_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in profiles:
        profiles_by_case.setdefault(row["case_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for case in cases:
        group = ava_by_case.get(case["case_id"], [])
        state_group = profiles_by_case.get(case["case_id"], [])
        sizes = [float(row["avalanche_extension_um"]) for row in group]
        summaries.append({
            **{key: case[key] for key in ("case_id", "matrix", "material_class", "candidate_id", "temperature_K", "theta_deg", "rate_tag", "loading_rate_factor", "seed")},
            "target_extension_um": case["target_extension_um"], "achieved_extension_um": case["achieved_extension_um"],
            "target_status": "TARGET_REACHED" if case["achieved_extension_um"] >= float(case["target_extension_um"]) - 1e-6 else "RIGHT_CENSORED",
            "numerical_event_count": case["numerical_event_count"], "physical_avalanche_count": len(group),
            "largest_avalanche_extension_um": max(sizes, default=0.0),
            "largest_avalanche_fraction": max(sizes, default=0.0) / max(case["achieved_extension_um"], 1e-30),
            "initial_onset_native_KJ_MPa_sqrt_m": group[0]["onset_native_KJ_MPa_sqrt_m"] if group else math.nan,
            "final_reinitiation_native_KJ_MPa_sqrt_m": group[-1]["onset_native_KJ_MPa_sqrt_m"] if len(group) > 1 else math.nan,
            "delta_K_reinit_MPa_sqrt_m": (group[-1]["onset_native_KJ_MPa_sqrt_m"] - group[0]["onset_native_KJ_MPa_sqrt_m"]) if len(group) > 1 else math.nan,
            "maximum_tip_radius_um": max((float(row["tip_radius_um"]) for row in state_group), default=math.nan),
            "minimum_front_width_um": min((float(row["front_width_um"]) for row in state_group), default=math.nan),
            "maximum_mobile_count": max((float(row["mobile_count"]) for row in state_group), default=math.nan),
            "maximum_retained_count": max((float(row["retained_count"]) for row in state_group), default=math.nan),
            "maximum_backstress_GPa": max((float(row["backstress_Pa"]) for row in state_group), default=math.nan) * 1e-9,
            "maximum_absolute_shielding_MPa_sqrt_m": max((abs(float(row["signed_active_shielding_MPa_sqrt_m"])) for row in state_group), default=math.nan),
            "steps_sha256": case["steps_sha256"],
        })
    tx_fields = list(transactions[0]) if transactions else []
    ava_fields = list(avalanches[0]) if avalanches else []
    write_csv(out / "pf_canonical_event_transactions_v2.csv", transactions, tx_fields)
    write_csv(out / "pf_canonical_physical_avalanches_v2.csv", avalanches, ava_fields)
    write_csv(out / "pf_canonical_onset_candidates_v2.csv", onset, list(onset[0]) if onset else tx_fields + ["onset_role"])
    write_csv(out / "pf_canonical_in_avalanche_native_drive_v2.csv", interior, list(interior[0]) if interior else tx_fields + ["semantic_role"])
    write_csv(out / "pf_canonical_state_profile_index_v2.csv", profiles, list(profiles[0]))
    write_csv(out / "pf_canonical_fracture_run_manifest.csv", summaries, list(summaries[0]))
    write_csv(out / "pf_canonical_theta_results.csv", [r for r in summaries if r["matrix"] == "CANONICAL_SINGLE_CRACK_THETA"], list(summaries[0]))
    write_csv(out / "pf_canonical_rate_results.csv", [r for r in summaries if r["matrix"] == "CANONICAL_STRAIN_RATE"], list(summaries[0]))
    comparison_count = 0
    one_d_bound_count = 0
    if one_d_results is not None:
        with one_d_results.open(newline="") as stream:
            one_d_rows = {row["case_id"]: row for row in csv.DictReader(stream)}
        comparisons: list[dict[str, Any]] = []
        for pf in summaries:
            reduced = one_d_rows.get(pf["case_id"])
            if reduced is None:
                continue
            comparison_count += 1
            bound = reduced["status"] == "RIGHT_CENSORED_DRIVE_MAP_BOUND"
            one_d_bound_count += int(bound)
            comparisons.append({
                **{key: pf[key] for key in (
                    "case_id", "matrix", "material_class", "candidate_id",
                    "temperature_K", "theta_deg", "rate_tag",
                    "loading_rate_factor", "seed")},
                "pf_target_status": pf["target_status"],
                "oneD_status": reduced["status"],
                "comparison_status": "MATCHED_WITH_ONED_DRIVE_MAP_BOUND" if bound else "MATCHED_TARGET_TO_TARGET",
                "pf_initial_onset_native_KJ_MPa_sqrt_m": pf["initial_onset_native_KJ_MPa_sqrt_m"],
                "oneD_initial_onset_native_KJ_MPa_sqrt_m": reduced["first_event_native_KJ_MPa_sqrt_m"],
                "onset_delta_oneD_minus_PF_MPa_sqrt_m": float(reduced["first_event_native_KJ_MPa_sqrt_m"]) - float(pf["initial_onset_native_KJ_MPa_sqrt_m"]),
                "pf_physical_avalanche_count": pf["physical_avalanche_count"],
                "oneD_physical_avalanche_count": reduced["physical_avalanche_count"],
                "pf_largest_avalanche_fraction": pf["largest_avalanche_fraction"],
                "oneD_largest_avalanche_fraction": reduced["largest_avalanche_fraction"],
                "pf_maximum_tip_radius_um": pf["maximum_tip_radius_um"],
                "oneD_maximum_tip_radius_um": reduced["max_tip_radius_um"],
                "pf_minimum_front_width_um": pf["minimum_front_width_um"],
                "oneD_minimum_front_width_um": reduced["minimum_front_width_um"],
                "pf_maximum_backstress_GPa": pf["maximum_backstress_GPa"],
                "oneD_maximum_backstress_GPa": reduced["max_backstress_GPa"],
                "pf_mobile_state_units": "signed_burgers_line_count",
                "oneD_mobile_state_units": "density_m^-2_not_directly_subtracted",
                "eventwise_K_not_compared_as_R_curve": True,
            })
        write_csv(out / "pf_canonical_1D_comparison_results.csv", comparisons,
                  list(comparisons[0]) if comparisons else ["case_id"])
    hashes = {path.name: sha256(path) for path in sorted(out.glob("*.csv"))}
    decision = {
        "schema": "pf_canonical_campaign_decision_v2", "case_count": len(cases),
        "expected_case_count": expected_case_count,
        "campaign_case_count_complete": expected_case_count is None or len(cases) == expected_case_count,
        "skipped_incomplete_case_ids": skipped_incomplete,
        "all_targets_reached": all(row["target_status"] == "TARGET_REACHED" for row in summaries),
        "event_observer_closure": len(profiles) == len(transactions),
        "observer_profiles_are_event_boundary_only": True,
        "state_profile_count": len(profiles),
        "matched_oneD_comparison_count": comparison_count,
        "matched_oneD_drive_map_bound_count": one_d_bound_count,
        "transactions_are_not_resistance_curve_points": True,
        "native_history_label": "PF MODEL-NATIVE DRIVING TRAJECTORY",
        "onset_label": "RELOAD-SEPARATED EFFECTIVE RESISTANCE CANDIDATES",
        "artifact_hashes": hashes,
    }
    (out / "pf_canonical_campaign_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--oneD-results", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.campaign_root, args.out,
                             expected_case_count=args.expected_case_count,
                             one_d_results=args.oneD_results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
