#!/usr/bin/env python3
"""Analyze deterministic v10.1.7.5 reduced-candidate transfer cases."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _records(case_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    return list(payload.get("records", []))


def _first_fired(records: list[dict[str, Any]]) -> dict[str, Any]:
    for row in records:
        if bool(row.get("fired", False)):
            return row
    raise RuntimeError("case contains no fired kinetic record")


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _array_max_abs(value: Any) -> float:
    array = np.asarray(value if value is not None else [], dtype=float)
    if not array.size:
        return math.nan
    return float(np.max(np.abs(array)))


def _array_max(value: Any) -> float:
    array = np.asarray(value if value is not None else [], dtype=float)
    if not array.size:
        return math.nan
    return float(np.max(array))


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return math.nan
    return float(numerator / max(abs(denominator), 1.0e-12))


def _case_row(manifest_row: pd.Series) -> dict[str, Any]:
    case_dir = Path(str(manifest_row.outdir))
    records = _records(case_dir)
    fired = _first_fired(records)
    reliable = [row for row in records if "anisotropic_drive_reliable" in row]
    return {
        "candidate_id": str(manifest_row.candidate_id),
        "transition_bracket": str(manifest_row.transition_bracket),
        "endpoint": str(manifest_row.endpoint),
        "T_K": float(manifest_row.T_K),
        "mode": str(manifest_row["mode"]),
        "outdir": str(case_dir),
        "K_init_MPa_sqrt_m": _safe_float(fired.get("K_Pa_sqrt_m")) / 1.0e6,
        "active_mobile_at_fire": _safe_float(fired.get("active_mobile"), 0.0),
        "active_retained_at_fire": _safe_float(fired.get("active_retained"), 0.0),
        "active_K_shield_at_fire_MPa_sqrt_m": _safe_float(
            fired.get("active_K_shield_signed_Pa_sqrt_m"), 0.0
        ) / 1.0e6,
        "source_budget_consumed_at_fire": _safe_float(
            fired.get("campaign_source_budget_consumed"), 0.0
        ),
        "source_budget_remaining_at_fire": _safe_float(
            fired.get("campaign_source_budget_remaining"), 0.0
        ),
        "max_abs_sigma_back_channel_at_fire_Pa": _array_max_abs(
            fired.get("anisotropic_sigma_back_by_system_Pa")
        ),
        "max_sigma_emit_channel_at_fire_Pa": _array_max(
            fired.get("anisotropic_sigma_emit_by_system_Pa")
        ),
        "max_emission_rate_channel_at_fire_s": _array_max(
            fired.get("anisotropic_lambda_emit_by_system_s")
        ),
        "max_drive_factor_at_fire": _array_max(
            fired.get("anisotropic_drive_factors")
        ),
        "tensor_record_count": int(len(reliable)),
        "tensor_reliable_fraction": float(
            np.mean([bool(row.get("anisotropic_drive_reliable", False)) for row in reliable])
        ) if reliable else 0.0,
        "post_hazard_weighting_count": int(
            sum(bool(row.get("anisotropic_post_hazard_weighting_applied", False)) for row in reliable)
        ),
    }


def _mode_summary(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("T_K")
    if len(ordered) != 2:
        raise ValueError(
            f"expected two endpoint cases for {ordered.candidate_id.iloc[0]} "
            f"mode={ordered['mode'].iloc[0]}; found {len(ordered)}"
        )
    low = ordered.iloc[0]
    high = ordered.iloc[-1]
    low_K = float(low.K_init_MPa_sqrt_m)
    high_K = float(high.K_init_MPa_sqrt_m)
    rise = high_K - low_K
    return {
        "candidate_id": str(low.candidate_id),
        "transition_bracket": str(low.transition_bracket),
        "mode": str(low["mode"]),
        "low_T_K": float(low.T_K),
        "high_T_K": float(high.T_K),
        "low_K_init_MPa_sqrt_m": low_K,
        "high_K_init_MPa_sqrt_m": high_K,
        "rise_MPa_sqrt_m": rise,
        "endpoint_ratio": high_K / max(low_K, 1.0e-12),
        "low_source_budget_consumed": float(low.source_budget_consumed_at_fire),
        "high_source_budget_consumed": float(high.source_budget_consumed_at_fire),
        "emission_budget_growth": float(
            high.source_budget_consumed_at_fire - low.source_budget_consumed_at_fire
        ),
        "low_active_mobile": float(low.active_mobile_at_fire),
        "high_active_mobile": float(high.active_mobile_at_fire),
        "low_active_retained": float(low.active_retained_at_fire),
        "high_active_retained": float(high.active_retained_at_fire),
        "max_abs_K_shield_MPa_sqrt_m": float(
            np.max(np.abs(ordered.active_K_shield_at_fire_MPa_sqrt_m))
        ),
        "max_abs_sigma_back_channel_Pa": float(
            np.nanmax(ordered.max_abs_sigma_back_channel_at_fire_Pa)
        ),
        "minimum_tensor_reliable_fraction": float(
            np.min(ordered.tensor_reliable_fraction)
        ),
        "post_hazard_weighting_count": int(
            ordered.post_hazard_weighting_count.sum()
        ),
    }


def _candidate_summary(candidate: str, modes: pd.DataFrame) -> dict[str, Any]:
    lookup = {str(row["mode"]): row for _, row in modes.iterrows()}
    required = {
        "full",
        "plasticity_off",
        "blunting_off",
        "backstress_off",
        "shielding_off",
        "background_field_off",
    }
    missing = sorted(required.difference(lookup))
    if missing:
        raise ValueError(f"candidate {candidate} is missing modes: {missing}")

    full = lookup["full"]
    opening = lookup["plasticity_off"]
    blunt = lookup["blunting_off"]
    back = lookup["backstress_off"]
    shield = lookup["shielding_off"]
    background = lookup["background_field_off"]

    full_rise = float(full.rise_MPa_sqrt_m)
    opening_rise = float(opening.rise_MPa_sqrt_m)
    emission_rise = full_rise - opening_rise
    emission_fraction = _safe_div(emission_rise, full_rise)
    blunting_fraction = _safe_div(full_rise - float(blunt.rise_MPa_sqrt_m), full_rise)
    backstress_fraction = _safe_div(full_rise - float(back.rise_MPa_sqrt_m), full_rise)
    shielding_fraction = _safe_div(full_rise - float(shield.rise_MPa_sqrt_m), full_rise)
    background_retained = _safe_div(float(background.rise_MPa_sqrt_m), full_rise)

    priority_checks = {
        "full_endpoint_ratio_at_least_1p8": float(full.endpoint_ratio) >= 1.8,
        "low_toughness_in_range": 8.0 <= float(full.low_K_init_MPa_sqrt_m) <= 25.0,
        "high_toughness_below_70": float(full.high_K_init_MPa_sqrt_m) <= 70.0,
        "opening_only_ratio_at_most_1p25": float(opening.endpoint_ratio) <= 1.25,
        "emission_fraction_at_least_0p60": emission_fraction >= 0.60,
        "blunting_fraction_at_least_0p50": blunting_fraction >= 0.50,
        "shielding_fraction_abs_at_most_0p20": abs(shielding_fraction) <= 0.20,
        "background_off_retains_at_least_0p75": background_retained >= 0.75,
        "tensor_drives_reliable": float(modes.minimum_tensor_reliable_fraction.min()) == 1.0,
        "no_post_hazard_weighting": int(modes.post_hazard_weighting_count.sum()) == 0,
        "high_T_emission_exceeds_low_T": float(full.emission_budget_growth) > 0.0,
    }
    priority = bool(all(priority_checks.values()))
    score_terms = {
        "ratio": max(1.8 - float(full.endpoint_ratio), 0.0) / 0.20,
        "opening": max(float(opening.endpoint_ratio) - 1.25, 0.0) / 0.10,
        "emission": max(0.60 - emission_fraction, 0.0) / 0.15,
        "blunting": max(0.50 - blunting_fraction, 0.0) / 0.15,
        "shielding": max(abs(shielding_fraction) - 0.20, 0.0) / 0.10,
        "background": max(0.75 - background_retained, 0.0) / 0.10,
    }
    score = float(sum(value * value for value in score_terms.values()))

    return {
        "candidate_id": candidate,
        "transition_bracket": str(full.transition_bracket),
        "full_low_K_init_MPa_sqrt_m": float(full.low_K_init_MPa_sqrt_m),
        "full_high_K_init_MPa_sqrt_m": float(full.high_K_init_MPa_sqrt_m),
        "full_rise_MPa_sqrt_m": full_rise,
        "full_endpoint_ratio": float(full.endpoint_ratio),
        "plasticity_off_endpoint_ratio": float(opening.endpoint_ratio),
        "emission_coupled_rise_MPa_sqrt_m": emission_rise,
        "emission_fraction_of_full_rise": emission_fraction,
        "blunting_sensitivity_fraction": blunting_fraction,
        "backstress_sensitivity_fraction": backstress_fraction,
        "shielding_sensitivity_fraction": shielding_fraction,
        "background_off_retained_rise_fraction": background_retained,
        "full_high_minus_low_source_budget_consumed": float(full.emission_budget_growth),
        "full_max_abs_K_shield_MPa_sqrt_m": float(full.max_abs_K_shield_MPa_sqrt_m),
        "full_max_abs_sigma_back_channel_Pa": float(full.max_abs_sigma_back_channel_Pa),
        "two_d_transfer_score": score,
        "two_d_transfer_priority": priority,
        "two_d_transfer_reason": (
            "emission_opening_transfer_passed"
            if priority
            else "review_failed_transfer_checks"
        ),
        "transfer_checks_json": json.dumps(priority_checks, sort_keys=True),
        **{f"transfer_penalty_{name}": value for name, value in score_terms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = pd.read_csv(root / "transfer_run_manifest.tsv", sep="\t")
    incomplete = manifest[~manifest.status.astype(str).isin(["COMPLETE", "EXISTING"])]
    if not incomplete.empty:
        raise RuntimeError(f"run manifest contains incomplete cases:\n{incomplete}")

    cases = pd.DataFrame([_case_row(row) for _, row in manifest.iterrows()])
    mode_rows = [
        _mode_summary(group)
        for _, group in cases.groupby(["candidate_id", "mode"], sort=True)
    ]
    modes = pd.DataFrame(mode_rows).sort_values(["candidate_id", "mode"])
    candidate_rows = [
        _candidate_summary(str(candidate), group)
        for candidate, group in modes.groupby("candidate_id", sort=True)
    ]
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["two_d_transfer_score", "candidate_id"]
    )
    priority = candidates[candidates.two_d_transfer_priority].copy()

    cases.to_csv(root / "transfer_case_summary.csv", index=False)
    modes.to_csv(root / "transfer_mode_summary.csv", index=False)
    candidates.to_csv(root / "transfer_candidate_ranking.csv", index=False)
    priority.to_csv(root / "transfer_2d_priority.csv", index=False)

    report = {
        "schema": "v10.1.7.5_reduced_candidate_transfer_assessment",
        "n_cases": int(len(cases)),
        "n_candidates": int(candidates.candidate_id.nunique()),
        "n_modes": int(modes["mode"].nunique()),
        "n_transfer_priority": int(len(priority)),
        "all_tensor_drives_reliable": bool(
            np.all(cases.tensor_reliable_fraction == 1.0)
        ),
        "no_post_hazard_directional_weighting": bool(
            int(cases.post_hazard_weighting_count.sum()) == 0
        ),
        "ranking": str(root / "transfer_candidate_ranking.csv"),
        "priority_manifest": str(root / "transfer_2d_priority.csv"),
    }
    (root / "transfer_assessment.json").write_text(json.dumps(report, indent=2))

    columns = [
        "candidate_id",
        "transition_bracket",
        "full_low_K_init_MPa_sqrt_m",
        "full_high_K_init_MPa_sqrt_m",
        "full_endpoint_ratio",
        "plasticity_off_endpoint_ratio",
        "emission_fraction_of_full_rise",
        "blunting_sensitivity_fraction",
        "shielding_sensitivity_fraction",
        "background_off_retained_rise_fraction",
        "two_d_transfer_priority",
        "two_d_transfer_score",
    ]
    print(candidates[columns].to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
