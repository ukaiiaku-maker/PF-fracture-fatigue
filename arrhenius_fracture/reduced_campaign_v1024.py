"""v10.2.4 reduced DBTT campaign using the production cap-free state.

Candidate evolution uses the same spatial MPZ/source/transport/backstress/
blunting/shielding implementation as v10.2.2.  The only surrogate is the
candidate-independent tensor-drive closure loaded from a cap-free 2-D atlas.
Top candidates are not accepted until they pass full 2-D endpoint ablations.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from .material_manifest import ExpFloorBarrier, MaterialManifest, TransportBarrier
from .mechanical_closure_v1024 import TensorDriveAtlas
from .physical_shielding_v1022 import install_uncapped_physical_shielding
from .reduced_shared_state_v1023 import (
    SharedReducedConfig,
    _field_snapshot,
    _state_arrays,
    build_shared_engine,
    fallback_manifest_path,
)


MODEL_ID = "v10.2.4_2d_atlas_shared_state_dbtt_campaign"
DEFAULT_TEMPERATURES_K = (300.0, 700.0, 900.0, 1200.0)
DEFAULT_ANCHORS = (
    "DBTT_A0002333",
    "DBTT_A0003837",
    "DBTT_A0002277",
)
MODES = ("full", "plasticity_off", "shielding_off", "backstress_off")

# Search only the mechanism-bearing parameters. Cleavage parameters and the
# remaining barrier-shape parameters are convex blends of the three preserved
# 2-D anchors, which prevents the campaign from escaping through an unrelated
# cleavage-only fit.
SEARCH_SCALES: dict[str, tuple[str, float]] = {
    "emit_G00_eV": ("factor", 1.7),
    "emit_gT_eV_per_K": ("absolute", 0.0030),
    "emit_sigc0_GPa": ("factor", 2.0),
    "peierls_H0_eV": ("factor", 1.6),
    "peierls_activation_entropy_kB": ("absolute", 20.0),
    "taylor_H0_eV": ("factor", 1.6),
    "taylor_activation_entropy_kB": ("absolute", 20.0),
    "taylor_corr_rho_c_m2": ("factor", 10.0),
    "taylor_corr_scale": ("factor", 3.0),
    "source_sites_per_system": ("factor", 3.0),
    "encounter_efficiency": ("factor", 3.0),
    "retained_recovery_rate_s": ("factor", 10.0),
    "source_refresh_length_um": ("factor", 3.0),
    "c_blunt": ("factor", 2.0),
}
SIGNED_FIELDS = {
    "cleave_gT_eV_per_K",
    "cleave_sT_GPa_per_K",
    "emit_gT_eV_per_K",
    "emit_sT_GPa_per_K",
    "peierls_activation_entropy_kB",
    "taylor_activation_entropy_kB",
}


def _read_one_row(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}; found {len(rows)}")
    row: dict[str, Any] = dict(rows[0])
    for key, value in list(row.items()):
        if key in {"candidate_id", "target_class"}:
            continue
        row[key] = float(value)
    return row


def load_anchor_rows() -> list[dict[str, Any]]:
    return [_read_one_row(fallback_manifest_path(candidate)) for candidate in DEFAULT_ANCHORS]


def manifest_from_row(row: dict[str, Any]) -> MaterialManifest:
    emission = ExpFloorBarrier(
        G00_eV=float(row["emit_G00_eV"]),
        gT_eV_per_K=float(row["emit_gT_eV_per_K"]),
        sigc0_Pa=float(row["emit_sigc0_GPa"]) * 1.0e9,
        sT_Pa_per_K=float(row["emit_sT_GPa_per_K"]) * 1.0e9,
        alpha=float(row["emit_exp_a"]),
        exponent=float(row["emit_exp_n"]),
        floor_fraction=float(row["emit_floor_frac"]),
        attempt_frequency_s=1.0e11,
    )
    cleavage = ExpFloorBarrier(
        G00_eV=float(row["cleave_G00_eV"]),
        gT_eV_per_K=float(row["cleave_gT_eV_per_K"]),
        sigc0_Pa=float(row["cleave_sigc0_GPa"]) * 1.0e9,
        sT_Pa_per_K=float(row["cleave_sT_GPa_per_K"]) * 1.0e9,
        alpha=float(row["cleave_exp_a"]),
        exponent=float(row["cleave_exp_n"]),
        floor_fraction=float(row["cleave_floor_frac"]),
        attempt_frequency_s=1.0e12,
    )
    return MaterialManifest(
        name=str(row.get("target_class", "DBTT")),
        candidate_id=str(row["candidate_id"]),
        cleavage=cleavage,
        emission=emission,
        peierls=TransportBarrier(
            H0_eV=float(row["peierls_H0_eV"]),
            activation_entropy_kB=float(row["peierls_activation_entropy_kB"]),
            alpha=float(row["peierls_exp_a"]),
            exponent=float(row["peierls_exp_n"]),
            attempt_frequency_s=float(row["peierls_nu0_s"]),
        ),
        taylor=TransportBarrier(
            H0_eV=float(row["taylor_H0_eV"]),
            activation_entropy_kB=float(row["taylor_activation_entropy_kB"]),
            alpha=float(row["taylor_exp_a"]),
            exponent=float(row["taylor_exp_n"]),
            attempt_frequency_s=float(row["taylor_nu0_s"]),
        ),
        taylor_corr_rho_c_m2=float(row["taylor_corr_rho_c_m2"]),
        taylor_corr_scale=float(row["taylor_corr_scale"]),
        source_sites_per_system=float(row["source_sites_per_system"]),
        encounter_efficiency=float(row["encounter_efficiency"]),
        retained_recovery_rate_s=float(row["retained_recovery_rate_s"]),
        source_refresh_length_m=float(row["source_refresh_length_um"]) * 1.0e-6,
        c_blunt=float(row["c_blunt"]),
        # Historical schema provenance only; v10.2.4 always installs uncapped law.
        max_K_shield_MPa_sqrt_m=float(row.get("max_K_shield_MPa_sqrt_m", 0.0)),
    )


def generate_candidate_rows(samples: int, seed: int = 1201) -> list[dict[str, Any]]:
    anchors = load_anchor_rows()
    numeric_fields = [
        key for key in anchors[0]
        if key not in {"candidate_id", "target_class"}
    ]
    dimension = 3 + len(SEARCH_SCALES)
    exponent = int(math.ceil(math.log2(max(int(samples), 1))))
    points = qmc.Sobol(d=dimension, scramble=True, seed=int(seed)).random_base2(exponent)
    points = points[: int(samples)]
    rows: list[dict[str, Any]] = []
    search_names = list(SEARCH_SCALES)

    for index, point in enumerate(points):
        raw_weight = np.maximum(point[:3], 1.0e-9)
        weight = raw_weight / np.sum(raw_weight)
        row: dict[str, Any] = {
            "candidate_id": f"DBTT_V1024_{index:07d}",
            "target_class": "DBTT",
            "anchor_weight_A0002333": float(weight[0]),
            "anchor_weight_A0003837": float(weight[1]),
            "anchor_weight_A0002277": float(weight[2]),
        }
        for field in numeric_fields:
            values = np.asarray([float(anchor[field]) for anchor in anchors], dtype=float)
            if field not in SIGNED_FIELDS and np.all(values > 0.0):
                value = float(np.exp(weight @ np.log(values)))
            else:
                value = float(weight @ values)
            row[field] = value

        for offset, field in enumerate(search_names, start=3):
            kind, magnitude = SEARCH_SCALES[field]
            coordinate = 2.0 * float(point[offset]) - 1.0
            if kind == "factor":
                row[field] = max(
                    float(row[field]) * math.exp(coordinate * math.log(magnitude)),
                    1.0e-30,
                )
            else:
                row[field] = float(row[field]) + coordinate * magnitude

        # No candidate is permitted to reactivate the historical cap.
        row["max_K_shield_MPa_sqrt_m"] = 0.0
        rows.append(row)
    return rows


def run_atlas_shared_front(
    manifest: MaterialManifest,
    temperature_K: float,
    cfg: SharedReducedConfig,
    atlas: TensorDriveAtlas,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Run the production shared state with a 2-D tensor-drive atlas closure."""
    cfg = cfg.validate()
    engine = build_shared_engine(manifest, cfg, mode=mode)
    target_advances = max(
        int(math.ceil(cfg.target_extension_um / cfg.checkpoint_da_um)), 1
    )
    K_left = 0.0
    outer_step = 0
    history: list[dict[str, Any]] = []
    outside_count = 0

    with install_uncapped_physical_shielding():
        while K_left < cfg.Kmax_MPa_sqrt_m and engine.n_adv < target_advances:
            outer_step += 1
            if outer_step > cfg.max_outer_steps:
                raise RuntimeError("v10.2.4 reduced front exceeded outer-step limit")
            dK = min(cfg.max_dK_step_MPa_sqrt_m, cfg.Kmax_MPa_sqrt_m - K_left)
            K_mid = K_left + 0.5 * dK
            progress = float(np.clip(engine.B, 0.0, 1.0))
            closure = atlas.evaluate(K_mid, progress)
            factors = np.asarray(closure.factors, dtype=float)
            engine.mpz._anisotropic_drive_factors = factors.copy()
            engine.mpz._anisotropic_drive_reliable = True
            engine.mpz._anisotropic_drive_serial = outer_step
            outside_count += int(closure.outside_support)

            dt_requested = dK / cfg.Kdot_MPa_sqrt_m_s
            result = engine.step(K_mid * 1.0e6, float(temperature_K), dt_requested)
            consumed = min(
                max(float(result.get("kinetic_dt_consumed_s", dt_requested)), 0.0),
                dt_requested,
            )
            fired = bool(result.get("fired", False))
            K_right = (
                K_left + cfg.Kdot_MPa_sqrt_m_s * consumed
                if fired
                else K_left + dK
            )
            history.append(
                {
                    "outer_step": outer_step,
                    "K_MPa_sqrt_m": float(K_right),
                    "K_input_MPa_sqrt_m": float(K_mid),
                    "temperature_K": float(temperature_K),
                    "mode": mode,
                    "fired": fired,
                    "checkpoint_progress_action": float(engine.B),
                    "drive_factor_0": float(factors[0]),
                    "drive_factor_1": float(factors[1]),
                    "closure_distance": float(closure.normalized_distance),
                    "closure_outside_support": bool(closure.outside_support),
                    "lambda_cleave_s": float(result.get("lambda_c", 0.0)),
                    "dN_emit": float(result.get("dN_emit", result.get("dN_emit_raw", 0.0))),
                    **_field_snapshot(engine),
                }
            )
            K_left = float(K_right)
            if fired and engine.n_adv >= target_advances:
                break
            if fired and consumed <= 0.0:
                K_left += min(dK, 1.0e-12)

    first = next(
        (float(row["K_MPa_sqrt_m"]) for row in history if row["fired"]), None
    )
    return {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "mode": mode,
        "temperature_K": float(temperature_K),
        "status": "complete" if engine.n_adv >= target_advances else "incomplete",
        "K_first_MPa_sqrt_m": first,
        "outer_steps": outer_step,
        "closure_outside_support_count": outside_count,
        "closure_outside_support_fraction": outside_count / max(outer_step, 1),
        "history": history,
        "final_arrays": _state_arrays(engine),
        "config": asdict(cfg),
        "constitutive_K_shield_clip_applied": False,
    }


def _rise(low: float, high: float) -> float:
    return float(high - low)


def score_candidate(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    def K(mode: str, T: float) -> float:
        value = results[(mode, float(T))].get("K_first_MPa_sqrt_m")
        return float(value) if value is not None else math.nan

    low_T = float(DEFAULT_TEMPERATURES_K[0])
    high_T = float(DEFAULT_TEMPERATURES_K[-1])
    full_values = [K("full", T) for T in DEFAULT_TEMPERATURES_K]
    finite = all(math.isfinite(value) for value in full_values)
    low, high = full_values[0], full_values[-1]
    full_rise = _rise(low, high) if finite else math.nan
    full_ratio = high / low if finite and low > 0.0 else math.nan
    off_low, off_high = K("plasticity_off", low_T), K("plasticity_off", high_T)
    shield_low, shield_high = K("shielding_off", low_T), K("shielding_off", high_T)
    back_low, back_high = K("backstress_off", low_T), K("backstress_off", high_T)
    off_ratio = off_high / off_low if off_low > 0.0 else math.nan
    shield_rise = _rise(shield_low, shield_high)
    back_rise = _rise(back_low, back_high)
    shielding_fraction = (
        (full_rise - shield_rise) / full_rise
        if math.isfinite(full_rise) and full_rise > 1.0e-12
        else math.nan
    )
    monotonic_fraction = float(
        np.mean(np.diff(np.asarray(full_values, dtype=float)) >= -1.0e-9)
    ) if finite else 0.0
    outside = max(
        float(result.get("closure_outside_support_fraction", 1.0))
        for result in results.values()
    )

    strict = bool(
        finite
        and 8.0 <= low <= 25.0
        and high <= 70.0
        and full_ratio >= 1.5
        and monotonic_fraction >= 0.90
        and math.isfinite(off_ratio)
        and off_ratio <= 1.25
        and math.isfinite(shielding_fraction)
        and shielding_fraction >= 0.50
        and back_rise > 0.0
        and outside <= 0.25
    )
    # Continuous objective: lower is better. Large penalties prevent cleavage-only
    # or surrogate-extrapolated candidates from ranking above the desired mechanism.
    objective = 0.0
    objective += 10.0 * max(0.0, 1.5 - (full_ratio if math.isfinite(full_ratio) else 0.0)) ** 2
    objective += 8.0 * max(0.0, (off_ratio if math.isfinite(off_ratio) else 10.0) - 1.25) ** 2
    objective += 12.0 * max(0.0, 0.50 - (shielding_fraction if math.isfinite(shielding_fraction) else -1.0)) ** 2
    objective += 5.0 * max(0.0, 8.0 - (low if math.isfinite(low) else 0.0)) ** 2
    objective += 2.0 * max(0.0, (low if math.isfinite(low) else 100.0) - 25.0) ** 2
    objective += 2.0 * max(0.0, (high if math.isfinite(high) else 100.0) - 70.0) ** 2
    objective += 5.0 * max(0.0, 0.90 - monotonic_fraction) ** 2
    objective += 5.0 * max(0.0, -back_rise) ** 2 if math.isfinite(back_rise) else 100.0
    objective += 25.0 * max(0.0, outside - 0.25) ** 2

    return {
        "strict_reduced_pass": strict,
        "objective": float(objective),
        "low_K_MPa_sqrt_m": low,
        "high_K_MPa_sqrt_m": high,
        "full_endpoint_ratio": full_ratio,
        "full_rise_MPa_sqrt_m": full_rise,
        "plasticity_off_endpoint_ratio": off_ratio,
        "shielding_off_rise_MPa_sqrt_m": shield_rise,
        "shielding_fraction_of_full_rise": shielding_fraction,
        "backstress_off_rise_MPa_sqrt_m": back_rise,
        "monotonic_fraction": monotonic_fraction,
        "maximum_closure_outside_support_fraction": outside,
        **{
            f"full_K_{int(T)}K_MPa_sqrt_m": value
            for T, value in zip(DEFAULT_TEMPERATURES_K, full_values)
        },
    }


def evaluate_candidate(
    row: dict[str, Any],
    atlas_csv: str | Path,
    cfg: SharedReducedConfig,
) -> tuple[dict[str, Any], dict[tuple[str, float], dict[str, Any]]]:
    manifest = manifest_from_row(row)
    atlas = TensorDriveAtlas.from_csv(atlas_csv)
    runs: dict[tuple[str, float], dict[str, Any]] = {}
    for T in DEFAULT_TEMPERATURES_K:
        runs[("full", float(T))] = run_atlas_shared_front(
            manifest, T, cfg, atlas, mode="full"
        )
    for mode in ("plasticity_off", "shielding_off", "backstress_off"):
        for T in (DEFAULT_TEMPERATURES_K[0], DEFAULT_TEMPERATURES_K[-1]):
            runs[(mode, float(T))] = run_atlas_shared_front(
                manifest, T, cfg, atlas, mode=mode
            )
    score = score_candidate(runs)
    return {**row, **score}, runs


def write_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        key for key in row
        if not key.startswith("anchor_weight_")
        and key not in {
            "strict_reduced_pass", "objective", "low_K_MPa_sqrt_m",
            "high_K_MPa_sqrt_m", "full_endpoint_ratio",
            "full_rise_MPa_sqrt_m", "plasticity_off_endpoint_ratio",
            "shielding_off_rise_MPa_sqrt_m", "shielding_fraction_of_full_rise",
            "backstress_off_rise_MPa_sqrt_m", "monotonic_fraction",
            "maximum_closure_outside_support_fraction",
            "full_K_300K_MPa_sqrt_m", "full_K_700K_MPa_sqrt_m",
            "full_K_900K_MPa_sqrt_m", "full_K_1200K_MPa_sqrt_m",
        }
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


__all__ = [
    "MODEL_ID",
    "DEFAULT_TEMPERATURES_K",
    "DEFAULT_ANCHORS",
    "generate_candidate_rows",
    "manifest_from_row",
    "run_atlas_shared_front",
    "score_candidate",
    "evaluate_candidate",
    "write_manifest_row",
]
