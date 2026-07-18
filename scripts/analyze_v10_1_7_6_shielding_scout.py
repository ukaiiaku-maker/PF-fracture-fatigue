#!/usr/bin/env python3
"""Analyze the v10.1.7.6 four-candidate shielding-history scout.

Unlike the initial transfer report, this analysis distinguishes the fired state
from the complete pre-fire trajectory.  Shielding, back stress, active
populations, cumulative emission, blunted radius, and positive anisotropic
emission-drive fractions are evaluated over every recorded kinetic state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODES = ("full", "plasticity_off", "shielding_off", "backstress_off")


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first_present(row: dict[str, Any], names: Iterable[str], default=math.nan) -> float:
    for name in names:
        if name in row:
            result = _finite(row.get(name), math.nan)
            if math.isfinite(result):
                return result
    return float(default)


def _array(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return result[np.isfinite(result)]


def _max_abs_array(records: list[dict[str, Any]], names: Iterable[str]) -> float:
    values: list[float] = []
    for row in records:
        for name in names:
            if name in row:
                array = _array(row.get(name))
                if array.size:
                    values.append(float(np.max(np.abs(array))))
                    break
    return max(values) if values else math.nan


def _trajectory_values(
    records: list[dict[str, Any]], names: Iterable[str], default=math.nan
) -> np.ndarray:
    return np.asarray(
        [_first_present(row, names, default) for row in records], dtype=float
    )


def _first_fired(records: list[dict[str, Any]]) -> dict[str, Any]:
    for row in records:
        if bool(row.get("fired", False)):
            return row
    raise RuntimeError("kinetic audit contains no fired record")


def _positive_emission_drive_fraction(records: list[dict[str, Any]]) -> float:
    positive = 0
    total = 0
    for row in records:
        array = _array(row.get("anisotropic_sigma_emit_by_system_Pa"))
        if array.size:
            positive += int(np.count_nonzero(array > 0.0))
            total += int(array.size)
    return float(positive / total) if total else math.nan


def _load_case(manifest_row: pd.Series) -> dict[str, Any]:
    root = Path(str(manifest_row.outdir))
    payload = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
    records = list(payload.get("records", []))
    if not records:
        raise RuntimeError(f"no kinetic records in {root}")
    fired = _first_fired(records)

    shielding = _trajectory_values(
        records,
        (
            "campaign_active_K_shield_effective_Pa_sqrt_m",
            "active_K_shield_signed_Pa_sqrt_m",
            "active_K_shield_Pa_sqrt_m",
            "mpz_active_K_shield_Pa_sqrt_m",
        ),
        0.0,
    ) / 1.0e6
    mobile = _trajectory_values(
        records,
        ("developed_state_mobile_count", "mpz_mobile_count", "active_mobile"),
        0.0,
    )
    retained = _trajectory_values(
        records,
        ("developed_state_retained_count", "mpz_retained_count", "active_retained"),
        0.0,
    )
    emitted = _trajectory_values(
        records,
        ("developed_state_cumulative_emitted", "mpz_emitted_total"),
        0.0,
    )
    blunted_radius = _trajectory_values(
        records,
        ("developed_state_blunted_radius_m", "mpz_blunted_radius_m", "blunted_radius_m"),
        math.nan,
    ) * 1.0e6
    reliable = [row for row in records if "anisotropic_drive_reliable" in row]

    return {
        "candidate_id": str(manifest_row.candidate_id),
        "transition_bracket": str(manifest_row.transition_bracket),
        "endpoint": str(manifest_row.endpoint),
        "T_K": float(manifest_row.T_K),
        "mode": str(manifest_row["mode"]),
        "outdir": str(root),
        "K_init_MPa_sqrt_m": _finite(fired.get("K_Pa_sqrt_m")) / 1.0e6,
        "trajectory_record_count": int(len(records)),
        "tensor_record_count": int(len(reliable)),
        "tensor_reliable_fraction": float(
            np.mean([bool(row.get("anisotropic_drive_reliable", False)) for row in reliable])
        ) if reliable else 0.0,
        "post_hazard_weighting_count": int(
            sum(bool(row.get("anisotropic_post_hazard_weighting_applied", False)) for row in reliable)
        ),
        "max_abs_K_shield_history_MPa_sqrt_m": float(np.nanmax(np.abs(shielding))),
        "K_shield_at_fire_MPa_sqrt_m": _first_present(
            fired,
            (
                "campaign_active_K_shield_effective_Pa_sqrt_m",
                "active_K_shield_signed_Pa_sqrt_m",
                "active_K_shield_Pa_sqrt_m",
                "mpz_active_K_shield_Pa_sqrt_m",
            ),
            0.0,
        ) / 1.0e6,
        "max_abs_sigma_back_history_Pa": _max_abs_array(
            records, ("anisotropic_sigma_back_by_system_Pa",)
        ),
        "max_sigma_emit_history_Pa": _max_abs_array(
            records, ("anisotropic_sigma_emit_by_system_Pa",)
        ),
        "positive_emission_drive_fraction": _positive_emission_drive_fraction(records),
        "max_mobile_history": float(np.nanmax(mobile)),
        "mobile_at_fire": _first_present(
            fired, ("developed_state_mobile_count", "mpz_mobile_count", "active_mobile"), 0.0
        ),
        "max_retained_history": float(np.nanmax(retained)),
        "retained_at_fire": _first_present(
            fired, ("developed_state_retained_count", "mpz_retained_count", "active_retained"), 0.0
        ),
        "max_cumulative_emitted_history": float(np.nanmax(emitted)),
        "cumulative_emitted_at_fire": _first_present(
            fired, ("developed_state_cumulative_emitted", "mpz_emitted_total"), 0.0
        ),
        "max_blunted_radius_history_um": float(np.nanmax(blunted_radius))
        if np.any(np.isfinite(blunted_radius))
        else math.nan,
        "blunted_radius_at_fire_um": _first_present(
            fired,
            ("developed_state_blunted_radius_m", "mpz_blunted_radius_m", "blunted_radius_m"),
            math.nan,
        ) * 1.0e6,
        "source_budget_consumed_at_fire": _first_present(
            fired, ("campaign_source_budget_consumed",), 0.0
        ),
    }


def _mode_summary(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("T_K")
    if len(ordered) != 2:
        raise ValueError(
            f"expected two endpoints for {ordered.candidate_id.iloc[0]} "
            f"mode={ordered['mode'].iloc[0]}; found {len(ordered)}"
        )
    low = ordered.iloc[0]
    high = ordered.iloc[-1]
    low_K = float(low.K_init_MPa_sqrt_m)
    high_K = float(high.K_init_MPa_sqrt_m)
    return {
        "candidate_id": str(low.candidate_id),
        "transition_bracket": str(low.transition_bracket),
        "mode": str(low["mode"]),
        "low_T_K": float(low.T_K),
        "high_T_K": float(high.T_K),
        "low_K_init_MPa_sqrt_m": low_K,
        "high_K_init_MPa_sqrt_m": high_K,
        "rise_MPa_sqrt_m": high_K - low_K,
        "endpoint_ratio": high_K / max(low_K, 1.0e-12),
        "low_max_abs_K_shield_history_MPa_sqrt_m": float(low.max_abs_K_shield_history_MPa_sqrt_m),
        "high_max_abs_K_shield_history_MPa_sqrt_m": float(high.max_abs_K_shield_history_MPa_sqrt_m),
        "maximum_abs_K_shield_history_MPa_sqrt_m": float(
            ordered.max_abs_K_shield_history_MPa_sqrt_m.max()
        ),
        "maximum_abs_sigma_back_history_Pa": float(
            ordered.max_abs_sigma_back_history_Pa.max()
        ),
        "low_positive_emission_drive_fraction": float(low.positive_emission_drive_fraction),
        "high_positive_emission_drive_fraction": float(high.positive_emission_drive_fraction),
        "low_max_cumulative_emitted": float(low.max_cumulative_emitted_history),
        "high_max_cumulative_emitted": float(high.max_cumulative_emitted_history),
        "cumulative_emission_growth": float(
            high.max_cumulative_emitted_history - low.max_cumulative_emitted_history
        ),
        "low_max_mobile": float(low.max_mobile_history),
        "high_max_mobile": float(high.max_mobile_history),
        "low_max_retained": float(low.max_retained_history),
        "high_max_retained": float(high.max_retained_history),
        "low_max_blunted_radius_um": float(low.max_blunted_radius_history_um),
        "high_max_blunted_radius_um": float(high.max_blunted_radius_history_um),
        "minimum_tensor_reliable_fraction": float(ordered.tensor_reliable_fraction.min()),
        "post_hazard_weighting_count": int(ordered.post_hazard_weighting_count.sum()),
    }


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return float(numerator / denominator)


def _candidate_summary(candidate: str, modes: pd.DataFrame) -> dict[str, Any]:
    lookup = {str(row["mode"]): row for _, row in modes.iterrows()}
    missing = sorted(set(MODES).difference(lookup))
    if missing:
        raise ValueError(f"candidate {candidate} is missing modes: {missing}")
    full = lookup["full"]
    off = lookup["plasticity_off"]
    shielding_off = lookup["shielding_off"]
    backstress_off = lookup["backstress_off"]

    full_rise = float(full.rise_MPa_sqrt_m)
    off_rise = float(off.rise_MPa_sqrt_m)
    shield_off_rise = float(shielding_off.rise_MPa_sqrt_m)
    back_off_rise = float(backstress_off.rise_MPa_sqrt_m)
    emission_fraction = _safe_fraction(full_rise - off_rise, full_rise)
    shielding_fraction = _safe_fraction(full_rise - shield_off_rise, full_rise)
    backstress_fraction = _safe_fraction(full_rise - back_off_rise, full_rise)

    checks = {
        "positive_full_rise": full_rise > 0.0,
        "full_endpoint_ratio_at_least_1p5": float(full.endpoint_ratio) >= 1.5,
        "plasticity_off_ratio_at_most_1p25": float(off.endpoint_ratio) <= 1.25,
        "shielding_removes_at_least_half_rise": math.isfinite(shielding_fraction)
        and shielding_fraction >= 0.50,
        "substantial_shielding_history": float(full.maximum_abs_K_shield_history_MPa_sqrt_m) >= 0.25,
        "high_T_emission_exceeds_low_T": float(full.cumulative_emission_growth) > 0.0,
        "tensor_drives_reliable": float(modes.minimum_tensor_reliable_fraction.min()) == 1.0,
        "no_post_hazard_weighting": int(modes.post_hazard_weighting_count.sum()) == 0,
    }
    priority = bool(all(checks.values()))

    penalties = {
        "ratio": max(1.5 - float(full.endpoint_ratio), 0.0) / 0.20,
        "opening": max(float(off.endpoint_ratio) - 1.25, 0.0) / 0.10,
        "shielding_fraction": (
            max(0.50 - shielding_fraction, 0.0) / 0.15
            if math.isfinite(shielding_fraction)
            else 10.0
        ),
        "shielding_history": max(
            0.25 - float(full.maximum_abs_K_shield_history_MPa_sqrt_m), 0.0
        ) / 0.10,
    }
    score = float(sum(value * value for value in penalties.values()))

    return {
        "candidate_id": candidate,
        "transition_bracket": str(full.transition_bracket),
        "full_low_K_init_MPa_sqrt_m": float(full.low_K_init_MPa_sqrt_m),
        "full_high_K_init_MPa_sqrt_m": float(full.high_K_init_MPa_sqrt_m),
        "full_rise_MPa_sqrt_m": full_rise,
        "full_endpoint_ratio": float(full.endpoint_ratio),
        "plasticity_off_endpoint_ratio": float(off.endpoint_ratio),
        "shielding_off_endpoint_ratio": float(shielding_off.endpoint_ratio),
        "backstress_off_endpoint_ratio": float(backstress_off.endpoint_ratio),
        "emission_coupled_fraction_of_full_rise": emission_fraction,
        "shielding_history_fraction_of_full_rise": shielding_fraction,
        "backstress_sensitivity_fraction_of_full_rise": backstress_fraction,
        "full_max_abs_K_shield_history_MPa_sqrt_m": float(
            full.maximum_abs_K_shield_history_MPa_sqrt_m
        ),
        "full_max_abs_sigma_back_history_Pa": float(full.maximum_abs_sigma_back_history_Pa),
        "full_low_positive_emission_drive_fraction": float(
            full.low_positive_emission_drive_fraction
        ),
        "full_high_positive_emission_drive_fraction": float(
            full.high_positive_emission_drive_fraction
        ),
        "full_cumulative_emission_growth": float(full.cumulative_emission_growth),
        "full_low_max_mobile": float(full.low_max_mobile),
        "full_high_max_mobile": float(full.high_max_mobile),
        "full_low_max_retained": float(full.low_max_retained),
        "full_high_max_retained": float(full.high_max_retained),
        "full_low_max_blunted_radius_um": float(full.low_max_blunted_radius_um),
        "full_high_max_blunted_radius_um": float(full.high_max_blunted_radius_um),
        "shielding_history_scout_priority": priority,
        "shielding_history_scout_score": score,
        "shielding_history_scout_reason": (
            "shielding_history_dbtt_scout_passed"
            if priority
            else "review_failed_shielding_history_checks"
        ),
        "shielding_history_checks_json": json.dumps(checks, sort_keys=True),
        **{f"shielding_scout_penalty_{name}": value for name, value in penalties.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = pd.read_csv(root / "shielding_scout_run_manifest.tsv", sep="\t")
    invalid = manifest[~manifest.status.astype(str).isin(["COMPLETE", "EXISTING"])]
    if not invalid.empty:
        raise RuntimeError(f"incomplete cases remain:\n{invalid}")

    cases = pd.DataFrame([_load_case(row) for _, row in manifest.iterrows()])
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
        ["shielding_history_scout_score", "candidate_id"]
    )
    priority = candidates[candidates.shielding_history_scout_priority].copy()

    cases.to_csv(root / "shielding_scout_case_summary.csv", index=False)
    modes.to_csv(root / "shielding_scout_mode_summary.csv", index=False)
    candidates.to_csv(root / "shielding_scout_candidate_ranking.csv", index=False)
    priority.to_csv(root / "shielding_scout_priority.csv", index=False)

    report = {
        "schema": "v10.1.7.6_four_candidate_shielding_history_scout",
        "n_cases": int(len(cases)),
        "n_candidates": int(candidates.candidate_id.nunique()),
        "n_modes": int(modes["mode"].nunique()),
        "n_shielding_history_priority": int(len(priority)),
        "all_tensor_drives_reliable": bool(np.all(cases.tensor_reliable_fraction == 1.0)),
        "no_post_hazard_directional_weighting": bool(
            int(cases.post_hazard_weighting_count.sum()) == 0
        ),
        "ranking": str(root / "shielding_scout_candidate_ranking.csv"),
        "priority_manifest": str(root / "shielding_scout_priority.csv"),
        "trajectory_history_metrics_used": True,
    }
    (root / "shielding_scout_assessment.json").write_text(
        json.dumps(report, indent=2)
    )

    columns = [
        "candidate_id",
        "transition_bracket",
        "full_low_K_init_MPa_sqrt_m",
        "full_high_K_init_MPa_sqrt_m",
        "full_endpoint_ratio",
        "plasticity_off_endpoint_ratio",
        "shielding_off_endpoint_ratio",
        "shielding_history_fraction_of_full_rise",
        "backstress_sensitivity_fraction_of_full_rise",
        "full_max_abs_K_shield_history_MPa_sqrt_m",
        "shielding_history_scout_priority",
        "shielding_history_scout_score",
    ]
    print(candidates[columns].to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
