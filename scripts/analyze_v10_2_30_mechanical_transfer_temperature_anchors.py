#!/usr/bin/env python3
"""Qualify archived mechanical transfer and freeze bounded temperature anchors.

This is an archive-only analysis.  It does not launch PF, FEM/CZM, or fatigue
trajectories, and it never fits an applied-to-local transfer coefficient.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.analytical_monotonic_fracture_v10230 import (
    MonotonicControls,
    solve_first_passage,
    solve_first_passage_history,
)
from arrhenius_fracture.analytical_stationary_fatigue_v10230 import (
    AnalyticalControls,
    solve_hierarchy,
    waveform_K,
)
from arrhenius_fracture.material_manifest import (
    ExpFloorBarrier,
    MaterialManifest,
    TransportBarrier,
)

HIST = Path("/private/tmp/taylor-peierls-spatial-coupling-paper-audit")
ARCH = HIST / "analysis_outputs"
DEFAULT_OUT = ROOT / "runs/mechanical_transfer_temperature_anchors_v1"
R0_M = 1.0e-6
TEMPERATURE_GRID = (300.0, 600.0, 700.0, 900.0, 1100.0, 1200.0)
ANCHOR_ROLES = (
    "ceramic_primary", "weakT_primary", "dbtt_primary",
    "dbtt_intrinsic_control", "peak_primary", "dbtt_broad_shielding",
)
CANONICAL_ANCHOR_ROLES = (
    "CANONICAL_CERAMIC", "CANONICAL_WEAK_T", "CANONICAL_DBTT", "CANONICAL_PEAK",
)
LEGACY_NAMED_REGISTRY = ROOT / "arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv"
CURRENT_CANONICAL_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv"
EVENT_SOURCES = (
    "oneD_v2_taylor_peierls_spatial_transfer/pf_2d_spatial_transfer_event_transactions.csv",
    "oneD_v2_taylor_peierls_rcurve_search/pf_2d_taylor_peierls_event_transactions.csv",
    "oneD_v2_peak_dbtt_rcurve_search/pf_2d_peak_dbtt_R_event_transactions.csv",
    "oneD_v2_mechanics_maps_and_baselines/pf_2d_event_transactions_v2.csv",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def value(row: Mapping[str, Any], names: tuple[str, ...], default: float = math.nan) -> float:
    for name in names:
        if name in row:
            result = number(row[name])
            if math.isfinite(result):
                return result
    return default


def material_from_row(row: Mapping[str, Any]) -> MaterialManifest:
    def f(name: str, default: float = 0.0) -> float:
        return value(row, (name,), default)

    emission = ExpFloorBarrier(
        G00_eV=f("emit_G00_eV"), gT_eV_per_K=f("emit_gT_eV_per_K"),
        sigc0_Pa=f("emit_sigc0_GPa") * 1e9,
        sT_Pa_per_K=f("emit_sT_GPa_per_K") * 1e9,
        alpha=f("emit_exp_a"), exponent=f("emit_exp_n"),
        floor_fraction=f("emit_floor_frac"), attempt_frequency_s=1e11,
    )
    cleavage = ExpFloorBarrier(
        G00_eV=f("cleave_G00_eV"), gT_eV_per_K=f("cleave_gT_eV_per_K"),
        sigc0_Pa=f("cleave_sigc0_GPa") * 1e9,
        sT_Pa_per_K=f("cleave_sT_GPa_per_K") * 1e9,
        alpha=f("cleave_exp_a"), exponent=f("cleave_exp_n"),
        floor_fraction=f("cleave_floor_frac"), attempt_frequency_s=1e12,
    )
    return MaterialManifest(
        name=str(row.get("material_class", "unknown")),
        candidate_id=str(row.get("candidate_id", "unknown")),
        cleavage=cleavage, emission=emission,
        peierls=TransportBarrier(
            H0_eV=f("peierls_H0_eV"),
            activation_entropy_kB=f("peierls_activation_entropy_kB"),
            alpha=f("peierls_exp_a"), exponent=f("peierls_exp_n"),
            attempt_frequency_s=f("peierls_nu0_s", 1e12),
        ),
        taylor=TransportBarrier(
            H0_eV=f("taylor_H0_eV"),
            activation_entropy_kB=f("taylor_activation_entropy_kB"),
            alpha=f("taylor_exp_a"), exponent=f("taylor_exp_n"),
            attempt_frequency_s=f("taylor_nu0_s", 1e11),
        ),
        taylor_corr_rho_c_m2=f("taylor_corr_rho_c_m2", 5e12),
        taylor_corr_scale=f("taylor_corr_scale", 1.0),
        source_sites_per_system=f("source_sites_per_system", 1.0),
        encounter_efficiency=f("encounter_efficiency", 1.0),
        retained_recovery_rate_s=f("retained_recovery_rate_s", 0.0),
        source_refresh_length_m=f("source_refresh_length_um", 1.0) * 1e-6,
        c_blunt=f("c_blunt", 0.0),
        max_K_shield_MPa_sqrt_m=f("max_K_shield_MPa_sqrt_m", 0.0),
    )


def load_candidate_rows() -> pd.DataFrame:
    sources = [
        ROOT / "runs/joint_fracture_fatigue_archetype_atlas_v2/candidate_cross_lineage_registry.csv",
        ARCH / "oneD_v2_taylor_peierls_spatial_transfer/oneD_v2_spatial_transfer_pf_registry.csv",
        ARCH / "oneD_v2_taylor_peierls_rcurve_search/oneD_v2_taylor_peierls_pf_transfer_registry.csv",
        ARCH / "oneD_v2_peak_dbtt_rcurve_search/oneD_v2_peak_dbtt_R_pf_transfer_registry.csv",
    ]
    frames = []
    for priority, path in enumerate(sources):
        frame = pd.read_csv(path)
        frame["candidate_source_file"] = str(path)
        frame["candidate_source_sha256"] = sha256(path)
        frame["candidate_source_priority"] = priority
        frames.append(frame)
    # The cross-lineage atlas intentionally stores a merged descriptor table.
    # For the named v9.11 controls, use the exact source registry instead of
    # allowing missing fields in that merged table to acquire analytical
    # defaults.  These rows still lack the current persistent-site density;
    # that incompatibility is audited explicitly below and is not filled from
    # a template or inferred from K_init.
    legacy = pd.read_csv(LEGACY_NAMED_REGISTRY)
    legacy = legacy[legacy.option_key.isin(ANCHOR_ROLES)].copy()
    legacy["registry_role"] = legacy.option_key
    legacy["candidate_source_file"] = str(LEGACY_NAMED_REGISTRY)
    legacy["candidate_source_sha256"] = sha256(LEGACY_NAMED_REGISTRY)
    legacy["candidate_source_priority"] = len(sources)
    legacy["lineage"] = "LEGACY_V9_11_FINITE_SOURCE"
    legacy["row_sha256"] = [
        hashlib.sha256(json.dumps(
            {key: (None if pd.isna(item) else item) for key, item in row.items()
             if key not in {"row_sha256", "candidate_source_file", "candidate_source_sha256", "candidate_source_priority"}},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        for row in legacy.to_dict("records")
    ]
    frames.append(legacy)
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows = rows.sort_values("candidate_source_priority").drop_duplicates("candidate_id", keep="last")
    canonical = pd.read_csv(CURRENT_CANONICAL_REGISTRY)
    canonical["registry_role"] = canonical.candidate_id.map({
        "v913_zeroD_sobol_0242980": "CANONICAL_PEAK",
        "v913_zeroD_sobol_0202500": "CANONICAL_DBTT",
        "v913_zeroD_sobol_0129902": "CANONICAL_WEAK_T",
        "v913_zeroD_sobol_0077080": "CANONICAL_CERAMIC",
    })
    canonical["candidate_source_file"] = str(CURRENT_CANONICAL_REGISTRY)
    canonical["candidate_source_sha256"] = sha256(CURRENT_CANONICAL_REGISTRY)
    canonical["candidate_source_priority"] = len(sources) + 1
    canonical["lineage"] = "CURRENT_V10_2_30_PERSISTENT_SITE"
    canonical["row_sha256"] = [
        hashlib.sha256(json.dumps(
            {key: (None if pd.isna(item) else item) for key, item in row.items()
             if key not in {"row_sha256", "candidate_source_file", "candidate_source_sha256", "candidate_source_priority"}},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        for row in canonical.to_dict("records")
    ]
    rows = pd.concat([rows, canonical], ignore_index=True, sort=False)
    rows = rows.sort_values("candidate_source_priority").drop_duplicates("candidate_id", keep="last")
    return rows.reset_index(drop=True)


def load_audit(path: Path, cache: dict[Path, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if path not in cache:
        # The raw records include full spatial arrays and can occupy many times
        # their on-disk size once decoded.  Preserve every scalar needed by this
        # qualification plus exact signed population sums, then release the raw
        # arrays before decoding the next record.  Streaming the top-level
        # ``records`` array avoids materializing an entire 30--70 MB tensor audit
        # as nested Python objects.  This changes neither the archived history
        # nor the first-passage input.
        keep = (
            "K_Pa_sqrt_m", "source_opening_stress_Pa",
            "hazard_last_completed_threshold", "persistent_tip_radius_m",
            "active_K_shield_signed_Pa_sqrt_m", "persistent_sigma_back_Pa",
            "active_mobile", "active_retained",
        )
        slim: list[dict[str, Any]] = []
        text = path.read_text()
        marker = text.find('"records"')
        start = text.find("[", marker)
        if marker < 0 or start < 0:
            raise ValueError(f"tensor audit lacks records array: {path}")
        decoder = json.JSONDecoder()
        cursor = start + 1
        while True:
            while cursor < len(text) and text[cursor] in " \t\r\n,":
                cursor += 1
            if cursor >= len(text) or text[cursor] == "]":
                break
            record, cursor = decoder.raw_decode(text, cursor)
            item = {name: record.get(name) for name in keep}
            for population in ("mobile", "retained"):
                for sign in ("positive", "negative"):
                    name = f"{population}_{sign}_by_system_bin"
                    raw = record.get(name)
                    item[name] = (
                        float(np.sum(np.asarray(raw, dtype=float)))
                        if raw is not None else None
                    )
            slim.append(item)
        cache[path] = slim
    return cache[path]


def exact_audit_record(row: Mapping[str, Any], cache: dict[Path, list[dict[str, Any]]]) -> dict[str, Any] | None:
    raw = row.get("source_tensor_audit_file")
    step = int(value(row, ("pre_event_step",), -1))
    if not raw or step < 1:
        return None
    path = Path(str(raw))
    if not path.is_file():
        return None
    records = load_audit(path, cache)
    return records[step - 1] if step <= len(records) else None


def event_reconstruction(cache: dict[Path, list[dict[str, Any]]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for relative in EVENT_SOURCES:
        path = ARCH / relative
        frame = pd.read_csv(path)
        sources.append({"source_file": str(path), "source_sha256": sha256(path), "rows": len(frame)})
        for source_row, raw in enumerate(frame.to_dict("records")):
            audit = exact_audit_record(raw, cache)
            KJ = value(raw, ("pre_event_native_KJ_MPa_sqrt_m", "KJ_MPa_sqrt_m"))
            J = value(raw, ("pre_event_native_J_J_per_m2", "native_J_J_per_m2"))
            Kapp = value(raw, (
                "pre_event_K_app_or_common_reference_MPa_sqrt_m",
                "pre_event_native_KJ_MPa_sqrt_m", "KJ_MPa_sqrt_m",
            ))
            Klocal = number(audit.get("K_Pa_sqrt_m")) * 1e-6 if audit else math.nan
            if not math.isfinite(Klocal):
                Klocal = value(raw, ("pre_event_K_native_MPa_sqrt_m",))
            sigma_source = number(audit.get("source_opening_stress_Pa")) if audit else math.nan
            if not math.isfinite(sigma_source):
                sigma_source = value(raw, ("pre_event_source_opening_stress_Pa",))
            source_K = value(raw, ("pre_event_K_effective_local_equivalent_MPa_sqrt_m",))
            if not math.isfinite(source_K) and math.isfinite(sigma_source):
                source_K = sigma_source * math.sqrt(2 * math.pi * R0_M) * 1e-6
            radius_um = value(raw, ("pre_event_tip_radius_um",))
            Eeff = (KJ * 1e6) ** 2 / J if J > 0 and math.isfinite(KJ) else math.nan
            shape = (
                sigma_source * math.sqrt(2 * math.pi * radius_um * 1e-6) / (Klocal * 1e6)
                if radius_um > 0 and Klocal > 0 and math.isfinite(sigma_source) else math.nan
            )
            output.append({
                "result_id": f"{relative}:{source_row}", "result_kind": "PF_EVENT_TRANSACTION",
                "source_file": str(path), "source_sha256": sources[-1]["source_sha256"],
                "source_row": source_row, "material_class": raw.get("material_class"),
                "candidate_role": raw.get("candidate_role"), "candidate_id": raw.get("candidate_id"),
                "temperature_K": value(raw, ("temperature_K",)),
                "hazard_seed": value(raw, ("hazard_seed",)),
                "event_transaction_index": value(raw, ("event_transaction_index",)),
                "K_applied_MPa_sqrt_m": Kapp, "J_front_J_per_m2": J,
                "E_effective_prime_Pa": Eeff, "K_J_front_MPa_sqrt_m": KJ,
                "K_local_MPa_sqrt_m": Klocal,
                "K_source_opening_equivalent_MPa_sqrt_m": source_K,
                "K_shield_MPa_sqrt_m": value(raw, (
                    "pre_event_K_shield_MPa_sqrt_m", "pre_event_signed_shielding_MPa_sqrt_m",
                )),
                "K_exact_supplied_to_kinetic_kernel_MPa_sqrt_m": Klocal,
                "source_opening_stress_override_Pa": sigma_source,
                "pre_event_r_eff_um": radius_um,
                "pre_event_backstress_GPa": value(raw, ("pre_event_backstress_GPa",)),
                "pre_event_mobile_count": value(raw, ("pre_event_mobile_count",)),
                "pre_event_retained_count": value(raw, ("pre_event_retained_count",)),
                "finite_radius_normalized_stress_shape_at_reff": shape,
                "finite_radius_normalized_stress_shape_at_r0": (
                    source_K / Klocal if Klocal > 0 and math.isfinite(source_K) else math.nan
                ),
                "orientation_convention": "THETA_0_BCC_TWO_SIGNED_SLIP_TRACE_TENSORS",
                "constraint_convention": "PLANE_STRAIN",
                "modulus_convention": "ROW_RECONSTRUCTED_EPRIME_EQUALS_KJ_SQUARED_OVER_NATIVE_J",
                "contour_or_domain_stability": "PF_SINGLE_NATIVE_DOMAIN_VALUE__NO_CONTOUR_SERIES",
                "local_tensor_history_available": bool(audit is not None and math.isfinite(sigma_source)),
                "exact_kernel_K_available": math.isfinite(Klocal),
                "unresolved_transfer": (
                    "APPLIED_TO_PF_NATIVE_J_RESOLVED__NATIVE_J_TO_FULL_LOCAL_TENSOR_NOT_A_SCALAR_IDENTITY"
                ),
            })
    return pd.DataFrame(output), sources


def mechanics_and_shadow_reconstruction() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    specs = (
        ("oneD_v2_mechanics_maps_and_baselines/oneD_v2_pf_mechanics_map.csv", "PF_MECHANICS_MAP"),
        ("oneD_v2_mechanics_maps_and_baselines/oneD_v2_fem_native_mechanics_map.csv", "FEM_NATIVE_MECHANICS_MAP"),
        ("oneD_v2_mechanics_maps_and_baselines/oneD_v2_fem_qualified_G_map.csv", "FEM_QUALIFIED_G_MAP"),
        ("oneD_v2_native_state_closure/oneD_v2_state_domain_shadow.csv", "EXACT_STATE_DOMAIN_SHADOW"),
        ("oneD_v2_native_state_closure/oneD_v2_level3_exact_microtrajectories_v2.csv", "EXACT_MICROTRAJECTORY"),
    )
    for relative, kind in specs:
        path = ARCH / relative
        frame = pd.read_csv(path)
        digest = sha256(path)
        sources.append({"source_file": str(path), "source_sha256": digest, "rows": len(frame)})
        for source_row, raw in enumerate(frame.to_dict("records")):
            J = value(raw, ("native_J_J_per_m2", "native_J_J_m2", "native_J_aggregate_J_per_m2"))
            KJ = value(raw, (
                "native_KJ_MPa_sqrt_m", "native_KJ_aggregate_MPa_sqrt_m",
            ))
            if not math.isfinite(KJ):
                Kpa = value(raw, ("native_KJ_Pa_sqrt_m",))
                KJ = Kpa * 1e-6 if math.isfinite(Kpa) else math.nan
            Eeff = (KJ * 1e6) ** 2 / J if J > 0 and math.isfinite(KJ) else math.nan
            if kind == "FEM_NATIVE_MECHANICS_MAP":
                stability = "FEM_DOMAIN_SERIES_RECORDED__" + str(raw.get("J_status", "UNKNOWN"))
                contour_spread = value(raw, ("J_domain_spread_J_per_m2",))
            elif kind == "FEM_QUALIFIED_G_MAP":
                stability = "FEM_ENERGY_COMPLIANCE_VCCT_QUALIFIED"
                contour_spread = math.nan
            elif kind == "PF_MECHANICS_MAP":
                stability = "PF_SINGLE_ANNULAR_NATIVE_DOMAIN_RECORDED"
                contour_spread = math.nan
            else:
                stability = str(raw.get("mechanics_status", raw.get("state_status", "EXACT_FIXTURE")))
                contour_spread = math.nan
            rows.append({
                "result_id": f"{relative}:{source_row}", "result_kind": kind,
                "source_file": str(path), "source_sha256": digest, "source_row": source_row,
                "material_class": raw.get("material_class"), "candidate_role": None,
                "candidate_id": raw.get("candidate_id"),
                "temperature_K": value(raw, ("temperature_K",)), "hazard_seed": math.nan,
                "event_transaction_index": value(raw, ("interval_index", "accepted_event_index")),
                "K_applied_MPa_sqrt_m": KJ, "J_front_J_per_m2": J,
                "E_effective_prime_Pa": Eeff, "K_J_front_MPa_sqrt_m": KJ,
                "K_local_MPa_sqrt_m": KJ if kind.startswith("EXACT_") else math.nan,
                "K_source_opening_equivalent_MPa_sqrt_m": math.nan,
                "K_shield_MPa_sqrt_m": value(raw, ("signed_shielding_Pa_sqrt_m",)) * 1e-6,
                "K_exact_supplied_to_kinetic_kernel_MPa_sqrt_m": KJ if kind.startswith("EXACT_") else math.nan,
                "source_opening_stress_override_Pa": math.nan,
                "pre_event_r_eff_um": value(raw, ("tip_radius_m", "probe_radius_m")) * 1e6,
                "pre_event_backstress_GPa": value(raw, ("backstress_Pa",)) * 1e-9,
                "pre_event_mobile_count": value(raw, ("mobile_count",)),
                "pre_event_retained_count": value(raw, ("retained_count",)),
                "finite_radius_normalized_stress_shape_at_reff": math.nan,
                "finite_radius_normalized_stress_shape_at_r0": math.nan,
                "orientation_convention": "THETA_0_CRACK_LOCAL_BASIS",
                "constraint_convention": "PLANE_STRAIN",
                "modulus_convention": "ROW_RECONSTRUCTED_EPRIME_EQUALS_KJ_SQUARED_OVER_NATIVE_J",
                "contour_or_domain_stability": stability,
                "contour_J_spread_J_per_m2": contour_spread,
                "local_tensor_history_available": kind == "EXACT_STATE_DOMAIN_SHADOW",
                "exact_kernel_K_available": kind.startswith("EXACT_"),
                "unresolved_transfer": (
                    "FEM_FULL_LOCAL_TENSOR_HISTORY_UNAVAILABLE" if kind.startswith("FEM_")
                    else "MAP_NODE_HAS_NO_KINETIC_STATE" if kind.endswith("MAP") else "NONE_AT_EXACT_FIXTURE_STATE"
                ),
            })
    return pd.DataFrame(rows), sources


def signed_population(record: Mapping[str, Any], prefix: str, sign: str) -> float:
    raw = record.get(f"{prefix}_{sign}_by_system_bin")
    if raw is None:
        return math.nan
    if np.isscalar(raw):
        return number(raw)
    array = np.asarray(raw, dtype=float)
    return float(np.sum(array)) if np.all(np.isfinite(array)) else math.nan


def history_qualification(
    event_frames: list[pd.DataFrame], candidate_rows: pd.DataFrame,
    cache: dict[Path, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectories = pd.concat(event_frames, ignore_index=True, sort=False)
    required = {"source_steps_file", "source_tensor_audit_file", "candidate_id", "temperature_K"}
    if not required.issubset(trajectories.columns):
        raise RuntimeError("rich event sources lack archived history references")
    trajectories = trajectories.dropna(subset=list(required)).drop_duplicates(
        ["source_steps_file", "source_tensor_audit_file"]
    )
    lookup = candidate_rows.set_index("candidate_id", drop=False)
    predictions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for raw in trajectories.to_dict("records"):
        steps_path = Path(str(raw["source_steps_file"]))
        audit_path = Path(str(raw["source_tensor_audit_file"]))
        base = {
            "candidate_id": raw["candidate_id"], "material_class": raw.get("material_class"),
            "candidate_role": raw.get("candidate_role"), "temperature_K": float(raw["temperature_K"]),
            "source_steps_file": str(steps_path), "source_tensor_audit_file": str(audit_path),
            "steps_available": steps_path.is_file(), "tensor_audit_available": audit_path.is_file(),
            "candidate_row_available": raw["candidate_id"] in lookup.index,
        }
        if not (steps_path.is_file() and audit_path.is_file() and raw["candidate_id"] in lookup.index):
            inventory.append({**base, "history_status": "UNAVAILABLE_FAIL_CLOSED"})
            continue
        steps = pd.read_csv(steps_path)
        records = load_audit(audit_path, cache)
        fired = np.flatnonzero(steps.n_fire.to_numpy(float) > 0.0)
        if not len(fired) or len(records) < int(fired[0]) + 1:
            inventory.append({**base, "history_status": "INVALID_ALIGNMENT_FAIL_CLOSED"})
            continue
        stop = int(fired[0])
        local_history = []
        for index in range(stop + 1):
            K = number(records[index].get("K_Pa_sqrt_m")) * 1e-6
            dt = number(steps.iloc[index].get("dt_cur_s"))
            if not (math.isfinite(K) and math.isfinite(dt) and dt >= 0.0):
                raise RuntimeError(f"non-finite archived local history at {audit_path}:{index}")
            local_history.append({"dt_s": dt, "K_local_MPa_sqrt_m": K})
        event_record = records[stop]
        threshold = number(event_record.get("hazard_last_completed_threshold"), 1.0)
        if threshold <= 0.0:
            threshold = 1.0
        row = lookup.loc[raw["candidate_id"]]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        manifest = material_from_row(row)
        observed_Klocal = local_history[-1]["K_local_MPa_sqrt_m"]
        observed_Kapp = number(steps.iloc[stop].get("KJ_Pa_sqrtm")) * 1e-6
        first_results: dict[str, dict[str, Any]] = {}
        for level in ("F0", "F1", "F2"):
            result = solve_first_passage_history(
                manifest, row, float(raw["temperature_K"]), local_history, level,
                MonotonicControls(), threshold_action=threshold,
            )
            first_results[level] = result
            predictions.append({
                **base, "history_status": "ARCHIVED_LOCAL_K_HISTORY_DRIVEN_NO_FIT",
                "level": level, "physical_first_event_interval": stop,
                "physical_first_event_time_s": float(steps.dt_cur_s.iloc[:stop + 1].sum()),
                "physical_first_event_K_applied_MPa_sqrt_m": observed_Kapp,
                "physical_first_event_K_local_MPa_sqrt_m": observed_Klocal,
                "threshold_action": threshold,
                "predicted_first_passage_reached_by_physical_event": result["first_passage_reached"],
                "predicted_first_passage_time_s": result["first_passage_time_s"],
                "predicted_first_passage_K_local_MPa_sqrt_m": result["K_local_at_first_passage_MPa_sqrt_m"],
                "action_at_physical_history_censor": result["cleavage_action"],
                "no_transfer_coefficient_fitted": True,
            })
        for level in ("F1", "F2", "F2B"):
            evolved = solve_first_passage_history(
                manifest, row, float(raw["temperature_K"]), local_history, level,
                MonotonicControls(), threshold_action=1.0e300,
            )["pre_event_state"]
            states.append({
                **base, "level": level, "state_timing": "PRE_FIRST_EVENT_PRE_TRANSLATION",
                "physical_r_eff_um": number(event_record.get("persistent_tip_radius_m")) * 1e6,
                "predicted_r_eff_um": number(evolved.get("r_eff_m")) * 1e6,
                "physical_K_shield_MPa_sqrt_m": number(event_record.get("active_K_shield_signed_Pa_sqrt_m")) * 1e-6,
                "predicted_K_shield_MPa_sqrt_m": number(evolved.get("K_shield_Pa_sqrt_m")) * 1e-6,
                "physical_backstress_GPa": number(event_record.get("persistent_sigma_back_Pa")) * 1e-9,
                "predicted_backstress_GPa": number(evolved.get("backstress_Pa")) * 1e-9,
                "physical_mobile": number(event_record.get("active_mobile")),
                "predicted_mobile": number(evolved.get("mobile")),
                "physical_retained": number(event_record.get("active_retained")),
                "predicted_retained": number(evolved.get("retained")),
                "physical_Nm_plus": signed_population(event_record, "mobile", "positive"),
                "physical_Nm_minus": signed_population(event_record, "mobile", "negative"),
                "physical_Nr_plus": signed_population(event_record, "retained", "positive"),
                "physical_Nr_minus": signed_population(event_record, "retained", "negative"),
                "predicted_signed_population_status": "MOMENT_ONLY_UNSIGNED_REDUCTION",
                "coefficient_fit_to_Kinit": False,
            })
        f0 = first_results["F0"]["K_local_at_first_passage_MPa_sqrt_m"]
        f2 = first_results["F2"]["K_local_at_first_passage_MPa_sqrt_m"]
        finite = math.isfinite(f0) and math.isfinite(f2)
        inventory.append({
            **base, "history_status": "ARCHIVED_LOCAL_K_HISTORY_DRIVEN_NO_FIT",
            "physical_first_event_K_applied_MPa_sqrt_m": observed_Kapp,
            "physical_first_event_K_local_MPa_sqrt_m": observed_Klocal,
            "mechanical_transfer_error_MPa_sqrt_m": observed_Kapp - observed_Klocal,
            "local_constitutive_error_MPa_sqrt_m": observed_Klocal - f0 if math.isfinite(f0) else math.nan,
            "state_history_interaction_MPa_sqrt_m": f0 - f2 if finite else math.nan,
            "total_applied_load_error_MPa_sqrt_m": observed_Kapp - f2 if math.isfinite(f2) else math.nan,
            "decomposition_closure_error": (
                (observed_Kapp - observed_Klocal) + (observed_Klocal - f0) + (f0 - f2)
                - (observed_Kapp - f2) if finite else math.nan
            ),
        })
    return pd.DataFrame(inventory), pd.DataFrame(predictions), pd.DataFrame(states)


def choose_middle(role: str, data: pd.DataFrame) -> float:
    finite = data[np.isfinite(data.K_init_F2_MPa_sqrt_m)].sort_values("temperature_K")
    interior = finite[(finite.temperature_K > min(TEMPERATURE_GRID)) & (finite.temperature_K < max(TEMPERATURE_GRID))]
    if len(interior) == 0:
        return 900.0
    if "peak" in role.lower():
        return float(interior.loc[interior.K_init_F2_MPa_sqrt_m.idxmax(), "temperature_K"])
    y = finite.K_init_F2_MPa_sqrt_m.to_numpy(float)
    T = finite.temperature_K.to_numpy(float)
    gradients = np.abs(np.diff(y) / np.diff(T))
    if len(gradients):
        candidate = float(T[min(int(np.argmax(gradients)) + 1, len(T) - 2)])
        if candidate in set(interior.temperature_K):
            return candidate
    return float(interior.iloc[len(interior) // 2].temperature_K)


def temperature_anchor_preflight(
    candidate_rows: pd.DataFrame,
    anchor_roles: tuple[str, ...] = ANCHOR_ROLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = candidate_rows[candidate_rows.registry_role.isin(anchor_roles)].copy()
    monotonic: list[dict[str, Any]] = []
    controls = MonotonicControls(dK_MPa_sqrt_m=0.1)
    for raw in rows.to_dict("records"):
        manifest = material_from_row(raw)
        for temperature in TEMPERATURE_GRID:
            results = {
                level: solve_first_passage(manifest, raw, temperature, level, controls)
                for level in ("F0", "F1", "F2")
            }
            monotonic.append({
                "registry_role": raw["registry_role"], "candidate_id": raw["candidate_id"],
                "row_sha256": raw.get("row_sha256"), "temperature_K": temperature,
                **{f"K_init_{level}_MPa_sqrt_m": results[level]["K_init_MPa_sqrt_m"] for level in results},
                **{f"{level}_right_censored": results[level]["right_censored"] for level in results},
            })
    mono = pd.DataFrame(monotonic)
    preflight: list[dict[str, Any]] = []
    fatigue_controls = AnalyticalControls(n_phase=512)
    for raw in rows.to_dict("records"):
        candidate = mono[mono.registry_role == raw["registry_role"]]
        selected_T = (300.0, choose_middle(str(raw["registry_role"]), candidate), 1200.0)
        manifest = material_from_row(raw)
        for temperature in selected_T:
            point = candidate[candidate.temperature_K == temperature].iloc[0]
            onset = number(point.K_init_F2_MPa_sqrt_m)
            if not math.isfinite(onset):
                onset = number(point.K_init_F0_MPa_sqrt_m, 20.0)
            Kset = sorted(set(round(max(2.0, min(80.0, onset * factor)), 1) for factor in (0.80, 1.00, 1.25)))
            for Kmax in Kset:
                fatigue = solve_hierarchy(
                    manifest, raw, Kmax, 0.1, temperature, 1000.0, fatigue_controls
                )
                Kphase = waveform_K(Kmax * 1e6, 0.1, fatigue_controls.n_phase)
                radius = number(fatigue["r_eff_m"], fatigue_controls.r0_m)
                stress = np.minimum(
                    np.maximum(Kphase, 0.0) / math.sqrt(2 * math.pi * max(radius, 1e-30)),
                    fatigue_controls.sigma_cap_Pa,
                )
                raw_rate = np.asarray(manifest.cleavage.rate(stress, temperature), dtype=float)
                x = raw_rate * fatigue_controls.cleavage_tau_s
                effective = gammainc(fatigue_controls.cleavage_hits, np.minimum(x, 1e12)) / fatigue_controls.cleavage_tau_s
                barrier = np.asarray(manifest.cleavage.values_eV(stress, temperature), dtype=float)
                floor = float(np.min(barrier))
                span = max(float(np.max(barrier) - floor), 1e-12)
                # The hard launch gate is the cleavage asymptote itself (A0),
                # not the partially qualified stationary state closure.  A1/A2
                # nonconvergence is retained as a diagnostic and never promoted.
                ceiling_fraction = float(fatigue["A0_mu_open"]) * 1000.0 * fatigue_controls.cleavage_tau_s
                near_ceiling = float(np.mean(effective >= 0.95 / fatigue_controls.cleavage_tau_s))
                near_floor = float(np.mean(barrier <= floor + 0.01 * span))
                stress_cap = float(np.mean(stress >= 0.999999 * fatigue_controls.sigma_cap_Pa))
                asymptotic_gate_passed = bool(
                    math.isfinite(ceiling_fraction)
                    and ceiling_fraction < 0.95 and near_ceiling < 0.5
                    and near_floor < 0.5 and stress_cap < 0.5
                )
                accessibility_gate_passed = bool(
                    number(fatigue["A0_da_dN"]) >= 1.0e-16
                )
                persistent_density = number(raw.get("rho_source0_m2"))
                bins = number(raw.get("n_bins_recommended"))
                inactive_closure_zero = all(
                    abs(number(raw.get(name), 0.0)) <= 1.0e-30
                    for name in (
                        "source_recovery_rate_s", "retained_recovery_rate_s",
                        "source_refresh_length_um", "legacy_source_sites_active",
                        "legacy_source_refresh_active", "explicit_recovery_active",
                    )
                )
                production_row_complete = bool(
                    math.isfinite(persistent_density) and persistent_density > 0.0
                    and bins == 80.0 and inactive_closure_zero
                    and str(raw.get("lineage")) == "CURRENT_V10_2_30_PERSISTENT_SITE"
                )
                eligible = asymptotic_gate_passed and production_row_complete
                preflight.append({
                    "registry_role": raw["registry_role"], "candidate_id": raw["candidate_id"],
                    "row_sha256": raw.get("row_sha256"), "temperature_K": temperature,
                    "temperature_role": (
                        "LOWER_BRANCH" if temperature == 300.0 else
                        "UPPER_OR_POST_TRANSITION" if temperature == 1200.0 else "TRANSITION_OR_PEAK"
                    ),
                    "Kmax_MPa_sqrt_m": Kmax, "R": 0.1, "frequency_Hz": 1000.0,
                    # The qualified signed-kernel family has 80 active stations.
                    # This is a frozen numerical contract, not a search coordinate.
                    "n_bins": 80, "target_extension_um": 100.0,
                    "maximum_cycles": 1.0e12, "fresh_virgin_required": True,
                    "A0_da_dN_m_per_cycle": fatigue["A0_da_dN"],
                    "A1_da_dN_m_per_cycle": fatigue["A1_da_dN"],
                    "A2_da_dN_m_per_cycle": fatigue["A2_da_dN"],
                    "fixed_point_converged": fatigue["fixed_point_converged"],
                    "stationary_state_reduction_status": (
                        "AVAILABLE" if fatigue["fixed_point_converged"]
                        else "UNAVAILABLE_NONCONVERGED_NOT_A_PHYSICAL_REJECTION"
                    ),
                    "accessibility_floor_m_per_cycle": 1.0e-16,
                    "accessibility_gate_passed": accessibility_gate_passed,
                    "prospective_accessibility_class": (
                        "EXPECTED_ACTIVE_OR_INTERMEDIATE" if accessibility_gate_passed
                        else "EXPECTED_PHYSICAL_CYCLE_CENSOR"
                    ),
                    "cooperative_ceiling_fraction": ceiling_fraction,
                    "phase_fraction_near_cooperative_ceiling": near_ceiling,
                    "phase_fraction_near_barrier_floor": near_floor,
                    "phase_fraction_at_stress_cap": stress_cap,
                    "asymptotic_gate_passed": asymptotic_gate_passed,
                    "current_persistent_site_density_m2": persistent_density,
                    "source_lineage": raw.get("lineage"),
                    "current_production_row_complete": production_row_complete,
                    "production_row_gap": (
                        "NONE" if production_row_complete else
                        "INCOMPLETE_OR_NONCURRENT_PERSISTENT_SITE_ROW"
                    ),
                    "preflight_launch_eligible": eligible,
                    "physical_launch_status": (
                        "NOT_LAUNCHED_PREFLIGHT_ONLY" if eligible else
                        "BLOCKED_INCOMPLETE_PERSISTENT_SITE_ROW" if asymptotic_gate_passed else
                        "NOT_LAUNCHED_ASYMPTOTIC_REJECTION"
                    ),
                })
    return mono, pd.DataFrame(preflight)


def plots(out: Path, reconstruction: pd.DataFrame, preflight: pd.DataFrame) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    events = reconstruction[reconstruction.result_kind == "PF_EVENT_TRANSACTION"]
    finite = events.dropna(subset=["K_applied_MPa_sqrt_m", "K_source_opening_equivalent_MPa_sqrt_m"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for material, group in finite.groupby("material_class"):
        ax.scatter(group.K_applied_MPa_sqrt_m, group.K_source_opening_equivalent_MPa_sqrt_m,
                   s=9, alpha=.35, label=str(material))
    lo = min(finite.K_applied_MPa_sqrt_m.min(), finite.K_source_opening_equivalent_MPa_sqrt_m.min())
    hi = max(finite.K_applied_MPa_sqrt_m.max(), finite.K_source_opening_equivalent_MPa_sqrt_m.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set(xlabel=r"$K_{applied}=K_J$ (MPa$\sqrt{m}$)",
           ylabel=r"source-opening equivalent $K$ (MPa$\sqrt{m}$)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figdir / "PF_APPLIED_VERSUS_SOURCE_LOCAL_K.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for role, group in preflight.groupby("registry_role"):
        y = group.A2_da_dN_m_per_cycle.where(
            group.A2_da_dN_m_per_cycle > 0.0, group.A0_da_dN_m_per_cycle
        )
        positive = y > 0.0
        ax.scatter(group.loc[positive, "Kmax_MPa_sqrt_m"], y[positive], s=22, label=role)
    if (preflight.A0_da_dN_m_per_cycle > 0.0).any():
        ax.set_yscale("log")
    ax.set(xlabel=r"$K_{max}$ (MPa$\sqrt{m}$)",
           ylabel=r"analytical $da/dN$ (A2 where available; otherwise A0) (m/cycle)")
    ax.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(figdir / "TEMPERATURE_FATIGUE_ANCHOR_PREFLIGHT.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, list[dict[str, Any]]] = {}
    candidates = load_candidate_rows()
    events, event_sources = event_reconstruction(cache)
    maps, map_sources = mechanics_and_shadow_reconstruction()
    reconstruction = pd.concat([events, maps], ignore_index=True, sort=False)
    reconstruction.to_parquet(out / "spatial_mechanical_transfer_reconstruction.parquet", index=False)

    rich_frames = [pd.read_csv(ARCH / rel) for rel in EVENT_SOURCES[:3]]
    inventory, passage, state = history_qualification(rich_frames, candidates, cache)
    inventory.to_csv(out / "archived_local_history_inventory_and_decomposition.csv", index=False)
    passage.to_csv(out / "archived_local_history_first_passage.csv", index=False)
    state.to_parquet(out / "transient_state_semantic_comparison.parquet", index=False)
    mono, preflight = temperature_anchor_preflight(candidates)
    canonical_mono, canonical_preflight = temperature_anchor_preflight(
        candidates, CANONICAL_ANCHOR_ROLES
    )
    mono.to_csv(out / "named_row_monotonic_temperature_screen.csv", index=False)
    preflight.to_csv(out / "bounded_temperature_fatigue_anchor_preflight.csv", index=False)
    canonical_mono.to_csv(out / "canonical_monotonic_temperature_screen.csv", index=False)
    canonical_preflight.to_csv(out / "canonical_temperature_fatigue_anchor_preflight.csv", index=False)

    coverage = reconstruction.groupby("result_kind", dropna=False).agg(
        rows=("result_id", "size"),
        J_available=("J_front_J_per_m2", lambda x: int(x.notna().sum())),
        Eprime_available=("E_effective_prime_Pa", lambda x: int(x.notna().sum())),
        local_K_available=("K_local_MPa_sqrt_m", lambda x: int(x.notna().sum())),
        tensor_history_available=("local_tensor_history_available", lambda x: int(x.fillna(False).sum())),
    ).reset_index()
    coverage.to_csv(out / "mechanical_transfer_coverage.csv", index=False)
    plots(out, reconstruction, preflight)

    qualified_histories = inventory[inventory.history_status == "ARCHIVED_LOCAL_K_HISTORY_DRIVEN_NO_FIT"]
    closure = qualified_histories.decomposition_closure_error.abs().dropna()
    decision = {
        "schema": "mechanical_transfer_temperature_anchor_campaign_v1",
        "created_utc": utcnow(), "repository": str(ROOT),
        "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"),
        "historical_repository": str(HIST), "historical_HEAD": subprocess.check_output(
            ["git", "-C", str(HIST), "rev-parse", "HEAD"], text=True
        ).strip(),
        "archive_sources": event_sources + map_sources,
        "spatial_rows_reconstructed": len(reconstruction),
        "archived_local_histories_qualified": len(qualified_histories),
        "first_passage_comparisons": len(passage),
        "state_comparisons": len(state),
        "decomposition_maximum_absolute_closure_error": float(closure.max()) if len(closure) else None,
        "PF_mechanical_transfer_status": "APPLIED_TO_NATIVE_J_RECONSTRUCTED__FULL_TENSOR_SOURCE_CHANNEL_RETAINED_SEPARATELY",
        "FEM_mechanical_transfer_status": "NATIVE_J_AND_CONTOUR_STABILITY_RECONSTRUCTED__FULL_LOCAL_TENSOR_HISTORY_UNAVAILABLE",
        "transient_state_status": "MOMENT_LEVEL_PARTIALLY_QUALIFIED__SIGNED_SPATIAL_CLOSURE_UNAVAILABLE_IN_F0_F1_F2",
        "F2B_status": "CONTROLLED_IMPLEMENTATION_RETAINED_FOR_TESTING__MODEL_CLASS_NOT_REJECTED",
        "Kinit_transfer_fit_performed": False,
        "new_PF_or_FEM_calculations": False,
        "new_fatigue_trajectories": False,
        "named_row_source_registry": str(LEGACY_NAMED_REGISTRY),
        "named_row_source_registry_sha256": sha256(LEGACY_NAMED_REGISTRY),
        "canonical_row_source_registry": str(CURRENT_CANONICAL_REGISTRY),
        "canonical_row_source_registry_sha256": sha256(CURRENT_CANONICAL_REGISTRY),
        "temperature_anchor_preflight_rows": len(preflight),
        "temperature_anchor_asymptotic_gate_rows": int(preflight.asymptotic_gate_passed.sum()),
        "temperature_anchor_launch_eligible_rows": int(preflight.preflight_launch_eligible.sum()),
        "canonical_temperature_anchor_preflight_rows": len(canonical_preflight),
        "canonical_temperature_anchor_asymptotic_gate_rows": int(canonical_preflight.asymptotic_gate_passed.sum()),
        "canonical_temperature_anchor_launch_eligible_rows": int(canonical_preflight.preflight_launch_eligible.sum()),
        "next_gate": "LAUNCH_ELIGIBLE_EXACT_CANONICAL_ROWS__KEEP_LEGACY_NAMED_CONTROLS_BLOCKED",
    }
    (out / "campaign_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(
        "# Mechanical-transfer and temperature-anchor qualification\n\n"
        "This archive-only bundle reconstructs native J/K transfer, exact kernel K, source-local "
        "tensor-equivalent drive, shielding, radius, and state without fitting K_init. PF local "
        "histories are replayed through F0/F1/F2 where both step and tensor-audit records exist. "
        "FEM native J and contour stability are qualified, but FEM full local tensor histories remain "
        "unavailable. The temperature matrix is preflight evidence only; no physical trajectory was launched. "
        "The six named rows are exact legacy v9.11 rows and lack the current persistent-site density, so their "
        "analytically admissible points are not production-launch eligible and no density is fabricated.\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
