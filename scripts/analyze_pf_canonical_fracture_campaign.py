#!/usr/bin/env python3
"""Regenerate V2 transaction, avalanche, onset, and campaign summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
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


def event_transactions(case_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            "tip_radius_um": number(event, "r_pz_m", number(event, "tip_radius_m")) * 1e6,
            "front_width_um": number(event, "mpz_length_m", number(event, "front_width_m")) * 1e6,
            "mobile_count": number(event, "mpz_mobile_count"),
            "retained_count": number(event, "mpz_retained_count"),
            "multiplicity": number(event, "N_em_pre_renewal", number(event, "N_em")),
            "backstress_Pa": number(event, "sigma_back_Pa"),
            "signed_shielding_MPa_sqrt_m": number(event, "mpz_K_shield_Pa_sqrt_m") / 1e6,
            "process_zone_state_fingerprint": fingerprint,
            "reload_time_to_next_event_s": math.nan if next_pos is None else times[next_pos] - times[pos],
            "reload_opening_to_next_event_m": math.nan if next_pos is None else number(steps[next_pos], "Uapp_m") - number(event, "Uapp_m"),
            "certified_reload_before_next_event": bool(next_pos is not None and any(number(item, "adaptive_frac", 0.0) >= 1.0 - 1e-12 for item in reload_slice)),
            "right_censored_at_target": index == len(positions) - 1,
            "source_steps_sha256": sha256(steps_path),
        })
    achieved = number(steps[-1], "crack_extension_m") * 1e6
    return transactions, {**launch, "achieved_extension_um": achieved, "steps_sha256": sha256(steps_path),
                          "numerical_event_count": len(positions)}


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


def analyze(campaign_root: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    transactions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for launch_path in sorted(campaign_root.glob("*/canonical_case_launch.json")):
        tx, case = event_transactions(launch_path.parent)
        transactions.extend(tx)
        cases.append(case)
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
    summaries: list[dict[str, Any]] = []
    for case in cases:
        group = ava_by_case.get(case["case_id"], [])
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
            "steps_sha256": case["steps_sha256"],
        })
    tx_fields = list(transactions[0]) if transactions else []
    ava_fields = list(avalanches[0]) if avalanches else []
    write_csv(out / "pf_canonical_event_transactions_v2.csv", transactions, tx_fields)
    write_csv(out / "pf_canonical_physical_avalanches_v2.csv", avalanches, ava_fields)
    write_csv(out / "pf_canonical_onset_candidates_v2.csv", onset, list(onset[0]) if onset else tx_fields + ["onset_role"])
    write_csv(out / "pf_canonical_in_avalanche_native_drive_v2.csv", interior, list(interior[0]) if interior else tx_fields + ["semantic_role"])
    write_csv(out / "pf_canonical_fracture_run_manifest.csv", summaries, list(summaries[0]))
    write_csv(out / "pf_canonical_theta_results.csv", [r for r in summaries if r["matrix"] == "CANONICAL_SINGLE_CRACK_THETA"], list(summaries[0]))
    write_csv(out / "pf_canonical_rate_results.csv", [r for r in summaries if r["matrix"] == "CANONICAL_STRAIN_RATE"], list(summaries[0]))
    hashes = {path.name: sha256(path) for path in sorted(out.glob("*.csv"))}
    decision = {
        "schema": "pf_canonical_campaign_decision_v1", "case_count": len(cases),
        "all_targets_reached": all(row["target_status"] == "TARGET_REACHED" for row in summaries),
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
    args = parser.parse_args()
    print(json.dumps(analyze(args.campaign_root, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
