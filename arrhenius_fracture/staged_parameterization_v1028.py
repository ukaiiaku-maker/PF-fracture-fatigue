"""Staged exact reduced and 2-D validation gates for v10.2.8."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .analytical_screen_v1028 import (
    DBTT_FIRST_PASSAGE_TEMPERATURES_K,
    WEAKT_FIRST_PASSAGE_TEMPERATURES_K,
)
from .material_manifest import MaterialManifest
from .state_resolved_reduced_campaign_v1027 import (
    ReducedCampaignControl,
    StateResolvedProductionConfig,
    run_reduced_r_curve,
)

MODEL_ID = "v10.2.8_staged_exact_parameterization"


@dataclass(frozen=True)
class FirstPassageControl:
    Kdot_MPa_sqrt_m_s: float = 0.005
    Kmax_MPa_sqrt_m: float = 80.0
    dK_MPa_sqrt_m: float = 0.05
    max_outer_steps: int = 2_000_000

    def as_reduced_control(self) -> ReducedCampaignControl:
        return ReducedCampaignControl(
            Kdot_MPa_sqrt_m_s=self.Kdot_MPa_sqrt_m_s,
            Kmax_MPa_sqrt_m=self.Kmax_MPa_sqrt_m,
            dK_MPa_sqrt_m=self.dK_MPa_sqrt_m,
            # Any positive extension smaller than one production checkpoint gives
            # exactly one accepted advance in run_reduced_r_curve.
            target_extension_um=1.0e-12,
            max_outer_steps=self.max_outer_steps,
        ).validate()


def run_first_passage(
    manifest: MaterialManifest,
    temperature_K: float,
    production: StateResolvedProductionConfig,
    control: FirstPassageControl,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    result = run_reduced_r_curve(
        manifest,
        temperature_K,
        production,
        control.as_reduced_control(),
        mode=mode,
    )
    return {
        **result,
        "schema": MODEL_ID,
        "stage": "first_passage",
        "K_first_passage_MPa_sqrt_m": result["K_init_MPa_sqrt_m"],
        "exactly_one_checkpoint_requested": True,
    }


def _finite_complete(row: dict[str, Any]) -> bool:
    return bool(
        row.get("status") == "complete"
        and math.isfinite(float(row.get("K_init_MPa_sqrt_m", math.nan)))
    )


def score_first_passage_dbtt(
    results: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    temperatures = DBTT_FIRST_PASSAGE_TEMPERATURES_K
    full = [results[("full", T)] for T in temperatures]
    complete = all(_finite_complete(row) for row in full)
    K = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    ratio = K[-1] / K[0] if complete and K[0] > 0.0 else math.nan
    monotonic = float(np.mean(np.diff(K) >= -1.0e-9)) if complete else 0.0
    strict = bool(
        complete
        and 5.0 <= K[0] <= 30.0
        and K[-1] <= 80.0
        and ratio >= 1.15
        and monotonic >= 2.0 / 3.0
    )
    objective = (
        20.0 * max(0.0, 1.15 - (ratio if math.isfinite(ratio) else 0.0)) ** 2
        + 10.0 * max(0.0, 2.0 / 3.0 - monotonic) ** 2
        + (0.0 if complete else 1.0e6)
    )
    return {
        "first_passage_pass": strict,
        "first_passage_objective": float(objective),
        "first_passage_endpoint_ratio": ratio,
        "first_passage_monotonic_fraction": monotonic,
        **{f"K_first_{int(T)}K": float(value) for T, value in zip(temperatures, K)},
    }


def score_first_passage_weakt(
    results: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    temperatures = WEAKT_FIRST_PASSAGE_TEMPERATURES_K
    full = [results[("full", T)] for T in temperatures]
    complete = all(_finite_complete(row) for row in full)
    K = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    span = float(np.max(K) / np.min(K)) if complete and np.min(K) > 0.0 else math.nan
    strict = bool(
        complete
        and float(np.min(K)) >= 5.0
        and float(np.max(K)) <= 60.0
        and span <= 1.20
    )
    objective = (
        25.0 * max(0.0, (span if math.isfinite(span) else 10.0) - 1.20) ** 2
        + (0.0 if complete else 1.0e6)
    )
    return {
        "first_passage_pass": strict,
        "first_passage_objective": float(objective),
        "first_passage_temperature_span_ratio": span,
        **{f"K_first_{int(T)}K": float(value) for T, value in zip(temperatures, K)},
    }


def score_rcurve_dbtt(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    temperatures = DBTT_FIRST_PASSAGE_TEMPERATURES_K
    full = [results[("full", T)] for T in temperatures]
    complete = all(row.get("status") == "complete" for row in full)
    init = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    final = np.asarray([row["K_final_MPa_sqrt_m"] for row in full], dtype=float)
    rise = np.asarray([row["R_rise_MPa_sqrt_m"] for row in full], dtype=float)
    rise_fraction = np.asarray([row["R_rise_fraction"] for row in full], dtype=float)
    ratio = final[-1] / final[0] if complete and final[0] > 0.0 else math.nan
    off_ratio = (
        results[("plasticity_off", temperatures[-1])]["K_final_MPa_sqrt_m"]
        / results[("plasticity_off", temperatures[0])]["K_final_MPa_sqrt_m"]
        if all(
            results[("plasticity_off", T)].get("status") == "complete"
            for T in (temperatures[0], temperatures[-1])
        )
        and results[("plasticity_off", temperatures[0])]["K_final_MPa_sqrt_m"] > 0.0
        else math.nan
    )
    full_temp_rise = final[-1] - final[0]
    shield_temp_rise = (
        results[("shielding_off", temperatures[-1])]["K_final_MPa_sqrt_m"]
        - results[("shielding_off", temperatures[0])]["K_final_MPa_sqrt_m"]
    )
    shield_temp_fraction = (
        (full_temp_rise - shield_temp_rise) / full_temp_rise
        if full_temp_rise > 1.0e-12
        else math.nan
    )
    high_shield_rise = results[("shielding_off", temperatures[-1])]["R_rise_MPa_sqrt_m"]
    high_shield_fraction = (
        (rise[-1] - high_shield_rise) / rise[-1] if rise[-1] > 1.0e-12 else math.nan
    )
    backstress_rise = results[("backstress_off", temperatures[-1])]["R_rise_MPa_sqrt_m"]
    monotonic = float(np.mean(np.diff(final) >= -1.0e-9)) if complete else 0.0
    strict = bool(
        complete
        and ratio >= 1.50
        and rise_fraction[0] <= 0.15
        and rise_fraction[-1] >= 0.20
        and math.isfinite(off_ratio) and off_ratio <= 1.25
        and math.isfinite(shield_temp_fraction) and shield_temp_fraction >= 0.50
        and math.isfinite(high_shield_fraction) and high_shield_fraction >= 0.30
        and backstress_rise > 0.0
        and monotonic >= 0.90
    )
    objective = (
        20.0 * max(0.0, 1.50 - (ratio if math.isfinite(ratio) else 0.0)) ** 2
        + 15.0 * max(0.0, rise_fraction[0] - 0.15) ** 2
        + 15.0 * max(0.0, 0.20 - rise_fraction[-1]) ** 2
        + 15.0 * max(0.0, (off_ratio if math.isfinite(off_ratio) else 10.0) - 1.25) ** 2
        + 20.0 * max(0.0, 0.50 - (shield_temp_fraction if math.isfinite(shield_temp_fraction) else -1.0)) ** 2
        + 10.0 * max(0.0, 0.30 - (high_shield_fraction if math.isfinite(high_shield_fraction) else -1.0)) ** 2
        + (0.0 if complete else 1.0e6)
    )
    return {
        "rcurve_pass": strict,
        "rcurve_objective": float(objective),
        "full_endpoint_ratio": ratio,
        "low_R_rise_fraction": float(rise_fraction[0]),
        "high_R_rise_fraction": float(rise_fraction[-1]),
        "plasticity_off_endpoint_ratio": off_ratio,
        "shielding_fraction_of_temperature_rise": shield_temp_fraction,
        "shielding_fraction_of_high_T_R_rise": high_shield_fraction,
        "backstress_off_high_T_R_rise_MPa_sqrt_m": float(backstress_rise),
        "monotonic_temperature_fraction": monotonic,
        **{f"K_init_{int(T)}K": float(v) for T, v in zip(temperatures, init)},
        **{f"K_final_{int(T)}K": float(v) for T, v in zip(temperatures, final)},
    }


def score_rcurve_weakt(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    temperatures = WEAKT_FIRST_PASSAGE_TEMPERATURES_K
    full = [results[("full", T)] for T in temperatures]
    complete = all(row.get("status") == "complete" for row in full)
    init = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    final = np.asarray([row["K_final_MPa_sqrt_m"] for row in full], dtype=float)
    rise = np.asarray([row["R_rise_MPa_sqrt_m"] for row in full], dtype=float)
    rise_fraction = np.asarray([row["R_rise_fraction"] for row in full], dtype=float)
    init_span = float(np.max(init) / np.min(init)) if complete and np.min(init) > 0 else math.nan
    final_span = float(np.max(final) / np.min(final)) if complete and np.min(final) > 0 else math.nan
    mean_rise = float(np.mean(rise)) if complete else math.nan
    off_mean = float(np.mean([
        results[("plasticity_off", temperatures[0])]["R_rise_MPa_sqrt_m"],
        results[("plasticity_off", temperatures[-1])]["R_rise_MPa_sqrt_m"],
    ]))
    shield_mean = float(np.mean([
        results[("shielding_off", temperatures[0])]["R_rise_MPa_sqrt_m"],
        results[("shielding_off", temperatures[-1])]["R_rise_MPa_sqrt_m"],
    ]))
    plastic_fraction = (
        (mean_rise - off_mean) / mean_rise if mean_rise > 1.0e-12 else math.nan
    )
    shield_fraction = (
        (mean_rise - shield_mean) / mean_rise if mean_rise > 1.0e-12 else math.nan
    )
    strict = bool(
        complete
        and init_span <= 1.20
        and final_span <= 1.20
        and float(np.min(rise_fraction)) >= 0.05
        and float(np.max(rise_fraction)) <= 0.25
        and float(np.min(rise)) >= 0.5
        and math.isfinite(plastic_fraction) and plastic_fraction >= 0.30
        and math.isfinite(shield_fraction) and shield_fraction >= 0.15
    )
    objective = (
        20.0 * max(0.0, (init_span if math.isfinite(init_span) else 10.0) - 1.20) ** 2
        + 20.0 * max(0.0, (final_span if math.isfinite(final_span) else 10.0) - 1.20) ** 2
        + 15.0 * max(0.0, 0.05 - float(np.min(rise_fraction))) ** 2
        + 10.0 * max(0.0, float(np.max(rise_fraction)) - 0.25) ** 2
        + 5.0 * max(0.0, 0.5 - float(np.min(rise))) ** 2
        + 15.0 * max(0.0, 0.30 - (plastic_fraction if math.isfinite(plastic_fraction) else -1.0)) ** 2
        + 10.0 * max(0.0, 0.15 - (shield_fraction if math.isfinite(shield_fraction) else -1.0)) ** 2
        + (0.0 if complete else 1.0e6)
    )
    return {
        "rcurve_pass": strict,
        "rcurve_objective": float(objective),
        "full_init_temperature_span_ratio": init_span,
        "full_final_temperature_span_ratio": final_span,
        "minimum_R_rise_MPa_sqrt_m": float(np.min(rise)),
        "maximum_R_rise_MPa_sqrt_m": float(np.max(rise)),
        "minimum_R_rise_fraction": float(np.min(rise_fraction)),
        "maximum_R_rise_fraction": float(np.max(rise_fraction)),
        "plasticity_fraction_of_mean_R_rise": plastic_fraction,
        "shielding_fraction_of_mean_R_rise": shield_fraction,
        **{f"K_init_{int(T)}K": float(v) for T, v in zip(temperatures, init)},
        **{f"K_final_{int(T)}K": float(v) for T, v in zip(temperatures, final)},
    }


def two_d_validation_cases(
    dbtt_candidate_ids: list[str],
    weakt_candidate_ids: list[str],
    *,
    target_extension_um: float = 100.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = ("full", "plasticity_off", "shielding_off", "backstress_off")
    for target_class, candidates, temperatures in (
        ("DBTT", dbtt_candidate_ids, DBTT_FIRST_PASSAGE_TEMPERATURES_K),
        ("weakT", weakt_candidate_ids, WEAKT_FIRST_PASSAGE_TEMPERATURES_K),
    ):
        for candidate in candidates:
            for T in temperatures:
                rows.append({
                    "target_class": target_class,
                    "candidate_id": candidate,
                    "temperature_K": T,
                    "mode": "full",
                    "target_extension_um": target_extension_um,
                })
            for mode in modes[1:]:
                for T in (temperatures[0], temperatures[-1]):
                    rows.append({
                        "target_class": target_class,
                        "candidate_id": candidate,
                        "temperature_K": T,
                        "mode": mode,
                        "target_extension_um": target_extension_um,
                    })
    return rows


__all__ = [
    "MODEL_ID",
    "FirstPassageControl",
    "run_first_passage",
    "score_first_passage_dbtt",
    "score_first_passage_weakt",
    "score_rcurve_dbtt",
    "score_rcurve_weakt",
    "two_d_validation_cases",
]
