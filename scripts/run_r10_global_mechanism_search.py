#!/usr/bin/env python3
"""Global, mechanism-stratified Family-A acquisition for the R10 continuation.

This is a search/selection driver, not a constitutive implementation.  Every
emitted production row uses the existing single EXP_floor fields consumed by
``retained_material_r10.load_offspring_binding``.  The driver deliberately
crosses opening and dislocation backgrounds, records rejected domain points,
and never uses the historical guard/broad-basis extensions.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import gammainc
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from arrhenius_fracture.stochastic_avalanche_tip import clipped_exponential_mean

FAMILY = "A_ORIGINAL_SINGLE_EXP_FLOOR"
MODEL = "r10.global-mechanism-stratified-acquisition/1"
TEMPERATURES = np.array([300.0, 450.0, 600.0, 900.0, 1200.0])
LOADS = np.array([4.0, 6.0, 9.0, 13.5, 20.25, 30.0])
TREF = 300.0
HITS = 3.0
TAU_S = 1.0e-6
FREQUENCY_HZ = 1000.0
EVENT_LENGTH_M = 5.0e-6
R_RATIO = 0.1


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def read_sources(original: Path, parents: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with original.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            row["search_source_id"] = row["candidate_id"]
            rows.append(row)
    with parents.open(newline="") as stream:
        for record in csv.DictReader(stream):
            row = json.loads(record["complete_parameter_vector"])
            row["search_source_id"] = record["candidate_id"]
            rows.append(row)
    if len(rows) != 8:
        raise RuntimeError(f"expected four original and four design rows; found {len(rows)}")
    return [rereference(row, TREF) for row in rows]


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def rereference(row0: dict[str, Any], target_Tref: float) -> dict[str, Any]:
    """Change coordinates, not a surface: preserve every G(T) and sigma_c(T)."""
    row = dict(row0)
    old = f(row, "Tref_K")
    dt = target_Tref - old
    for prefix in ("cleave", "emit"):
        row[f"{prefix}_G00_eV"] = f(row, f"{prefix}_G00_eV") + f(row, f"{prefix}_gT_eV_per_K") * dt
        row[f"{prefix}_sigc0_GPa"] = f(row, f"{prefix}_sigc0_GPa") + f(row, f"{prefix}_sT_GPa_per_K") * dt
    for prefix in ("peierls", "taylor"):
        entropy = f(row, f"{prefix}_activation_entropy_kB")
        row[f"{prefix}_H0_eV"] = f(row, f"{prefix}_H0_eV") - entropy * KB_EV_PER_K * dt
    row["Tref_K"] = target_Tref
    row["coordinate_rereference_only"] = True
    row["coordinate_source_Tref_K"] = old
    return row


def logit(x: float) -> float:
    x = min(max(float(x), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(x / (1.0 - x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def signed(u: float) -> float:
    return 2.0 * float(u) - 1.0


def generate_row(i: int, u: np.ndarray, sources: list[dict[str, Any]]) -> dict[str, Any]:
    oi = min(int(u[0] * len(sources)), len(sources) - 1)
    di = min(int(u[1] * len(sources)), len(sources) - 1)
    opening, disl = sources[oi], sources[di]
    row = dict(disl)
    for key in list(opening):
        if key.startswith("cleave_"):
            row[key] = opening[key]
    row["candidate_id"] = f"R10A_GLOBAL_{i:06d}"
    row["option_key"] = row["candidate_id"]
    row["material_class"] = "SEARCH_UNCLASSIFIED"
    row["search_opening_source"] = opening["search_source_id"]
    row["search_dislocation_source"] = disl["search_source_id"]
    row["search_cross_kind"] = ("SAME_BACKGROUND" if oi == di else "CROSSED_BACKGROUND")
    row["generator"] = "SCRAMBLED_SOBOL_CROSSED_ORIGINAL_EXP_FLOOR"
    row["generator_seed"] = 10162026
    row["family"] = FAMILY
    # Positive coordinates are varied logarithmically. Thermal slopes remain
    # the original linear-free-energy coordinates; the common/contrast draws
    # prevent independent endpoint fitting while exploring lawful entropy.
    row["cleave_G00_eV"] = f(row, "cleave_G00_eV") * math.exp(0.55 * signed(u[2]))
    row["cleave_sigc0_GPa"] = f(row, "cleave_sigc0_GPa") * math.exp(0.40 * signed(u[3]))
    row["cleave_exp_a"] = f(row, "cleave_exp_a") * math.exp(0.70 * signed(u[4]))
    row["cleave_exp_n"] = f(row, "cleave_exp_n") * math.exp(0.42 * signed(u[5]))
    row["cleave_floor_frac"] = sigmoid(logit(f(row, "cleave_floor_frac")) + 1.35 * signed(u[6]))
    thermal_common = 18.0 * KB_EV_PER_K * signed(u[7])
    thermal_contrast = 8.0 * KB_EV_PER_K * signed(u[8])
    row["cleave_gT_eV_per_K"] = f(row, "cleave_gT_eV_per_K") + thermal_common + thermal_contrast
    row["emit_gT_eV_per_K"] = f(row, "emit_gT_eV_per_K") + thermal_common - thermal_contrast
    stress_thermal = 2.5e-4 * signed(u[9])
    row["cleave_sT_GPa_per_K"] = f(row, "cleave_sT_GPa_per_K") + stress_thermal
    row["emit_sT_GPa_per_K"] = f(row, "emit_sT_GPa_per_K") + stress_thermal
    row["emit_G00_eV"] = f(row, "emit_G00_eV") * math.exp(0.50 * signed(u[10]))
    shared_stress = math.exp(0.42 * signed(u[11]))
    row["emit_sigc0_GPa"] = f(row, "emit_sigc0_GPa") * shared_stress
    row["emit_exp_a"] = f(row, "emit_exp_a") * math.exp(0.75 * signed(u[12]))
    row["emit_exp_n"] = f(row, "emit_exp_n") * math.exp(0.45 * signed(u[13]))
    row["emit_floor_frac"] = sigmoid(logit(f(row, "emit_floor_frac")) + 1.35 * signed(u[14]))
    hscale = math.exp(0.48 * signed(u[15]))
    row["peierls_H0_eV"] = f(row, "peierls_H0_eV") * hscale
    row["taylor_H0_eV"] = f(row, "taylor_H0_eV") * hscale * math.exp(0.20 * signed(u[16]))
    entropy_common = 22.0 * signed(u[17])
    row["peierls_activation_entropy_kB"] = f(row, "peierls_activation_entropy_kB") + entropy_common
    row["taylor_activation_entropy_kB"] = f(row, "taylor_activation_entropy_kB") + entropy_common
    row["peierls_exp_a"] = f(row, "peierls_exp_a") * math.exp(0.55 * signed(u[18]))
    row["taylor_exp_a"] = f(row, "taylor_exp_a") * math.exp(0.55 * signed(u[19]))
    row["peierls_exp_n"] = f(row, "peierls_exp_n") * math.exp(0.32 * signed(u[20]))
    row["taylor_exp_n"] = f(row, "taylor_exp_n") * math.exp(0.32 * signed(u[21]))
    row["changed_fields"] = [
        "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a", "cleave_exp_n",
        "cleave_floor_frac", "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K",
        "emit_G00_eV", "emit_sigc0_GPa", "emit_exp_a", "emit_exp_n",
        "emit_floor_frac", "emit_gT_eV_per_K", "emit_sT_GPa_per_K",
        "peierls_H0_eV", "peierls_activation_entropy_kB", "peierls_exp_a",
        "peierls_exp_n", "taylor_H0_eV", "taylor_activation_entropy_kB",
        "taylor_exp_a", "taylor_exp_n",
    ]
    return row


def barrier(row: dict[str, Any], prefix: str, sigma_Pa: np.ndarray,
            temperature_K: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = temperature_K - f(row, "Tref_K")
    G0 = f(row, f"{prefix}_G00_eV") + f(row, f"{prefix}_gT_eV_per_K") * dt
    sigc = (f(row, f"{prefix}_sigc0_GPa") + f(row, f"{prefix}_sT_GPa_per_K") * dt) * 1e9
    floor = f(row, f"{prefix}_floor_frac") * G0
    G = floor + (G0 - floor) * np.exp(-f(row, f"{prefix}_exp_a") *
        np.power(np.maximum(sigma_Pa, 0.0) / sigc, f(row, f"{prefix}_exp_n")))
    nu = 1.0e12 if prefix == "cleave" else 1.0e11
    rate = nu * np.exp(np.clip(-G / (KB_EV_PER_K * temperature_K), -700.0, 0.0))
    return G, rate, np.array([G0, sigc])


def cooperative(raw: np.ndarray) -> np.ndarray:
    return gammainc(HITS, np.maximum(raw, 0.0) * TAU_S) / TAU_S


def phase_action(row: dict[str, Any], prefix: str, Kmax: float, T: float,
                 radius_m: float = 1.0e-6) -> float:
    phi = 2 * math.pi * (np.arange(64) + 0.5) / 64
    h = np.maximum(0.5 * (1 + R_RATIO) + 0.5 * (1 - R_RATIO) * np.cos(phi), 0.0)
    sigma = np.minimum(Kmax * 1e6 * h / math.sqrt(2 * math.pi * radius_m), 30e9)
    _, raw, _ = barrier(row, prefix, sigma, T)
    return float(np.mean(cooperative(raw)) / FREQUENCY_HZ)


def transport_rate(row: dict[str, Any], prefix: str, sigma: float, T: float) -> float:
    H0 = f(row, f"{prefix}_H0_eV")
    entropy = f(row, f"{prefix}_activation_entropy_kB")
    H = H0 - entropy * KB_EV_PER_K * (T - TREF)
    sigc = (f(row, "emit_sigc0_GPa") + f(row, "emit_sT_GPa_per_K") * (T - TREF)) * 1e9
    G = f(row, "emit_floor_frac") * H + (1 - f(row, "emit_floor_frac")) * H * math.exp(
        -f(row, f"{prefix}_exp_a") * (max(sigma, 0.0) / sigc) ** f(row, f"{prefix}_exp_n"))
    return f(row, f"{prefix}_nu0_s") * math.exp(max(-G / (KB_EV_PER_K * T), -700.0))


def domain(row: dict[str, Any]) -> tuple[bool, str]:
    for T in TEMPERATURES:
        for prefix in ("cleave", "emit"):
            G0 = f(row, f"{prefix}_G00_eV") + f(row, f"{prefix}_gT_eV_per_K") * (T - TREF)
            sigc = f(row, f"{prefix}_sigc0_GPa") + f(row, f"{prefix}_sT_GPa_per_K") * (T - TREF)
            if G0 <= 0.05 or sigc <= 0.20:
                return False, f"{prefix.upper()}_NONPOSITIVE_DOMAIN"
        for prefix in ("peierls", "taylor"):
            H = f(row, f"{prefix}_H0_eV") - f(row, f"{prefix}_activation_entropy_kB") * KB_EV_PER_K * (T - TREF)
            if H <= 0.05:
                return False, f"{prefix.upper()}_NONPOSITIVE_DOMAIN"
    return True, ""


def ramp_root(row: dict[str, Any], T: float, Kdot: float = 0.005) -> float:
    K = np.linspace(0.05, 80.0, 320)
    sigma = K * 1e6 / math.sqrt(2 * math.pi * 1e-6)
    _, raw, _ = barrier(row, "cleave", sigma, T)
    lam = cooperative(raw)
    cumulative = np.r_[0.0, np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(K) / Kdot)]
    j = int(np.searchsorted(cumulative, 1.0))
    if j >= len(K):
        return math.nan
    if j == 0:
        return float(K[0])
    fraction = (1.0 - cumulative[j - 1]) / max(cumulative[j] - cumulative[j - 1], 1e-300)
    return float(K[j - 1] + fraction * (K[j] - K[j - 1]))


def descriptors(row: dict[str, Any]) -> dict[str, Any]:
    ok, rejection = domain(row)
    out: dict[str, Any] = {"candidate_id": row["candidate_id"], "domain_pass": ok,
                           "rejection": rejection, "opening_source": row["search_opening_source"],
                           "dislocation_source": row["search_dislocation_source"],
                           "cross_kind": row["search_cross_kind"]}
    if not ok:
        return out
    roots = [ramp_root(row, T) for T in TEMPERATURES]
    out.update({f"Kroot_{int(T)}K": roots[j] for j, T in enumerate(TEMPERATURES)})
    opening = np.array([[phase_action(row, "cleave", K, T) for K in LOADS]
                        for T in (300.0, 600.0, 900.0)])
    emission = np.array([[phase_action(row, "emit", K, T) for K in LOADS]
                         for T in (300.0, 600.0, 900.0)])
    source_mult = max(f(row, "rho_source0_m2") * f(row, "reference_source_area_um2") * 1e-12, 1.0)
    state_scores = []
    for ti, T in enumerate((300.0, 600.0, 900.0)):
        for ki, K in enumerate(LOADS):
            sigma = min(K * 1e6 / math.sqrt(2 * math.pi * 1e-6), 30e9)
            mobility = math.sqrt(max(transport_rate(row, "peierls", sigma, T), 1e-300) *
                                 max(transport_rate(row, "taylor", sigma, T), 1e-300)) / FREQUENCY_HZ
            state_scores.append(emission[ti, ki] * source_mult * min(mobility, 1e6) /
                                max(opening[ti, ki], 1e-300))
    state_scores = np.asarray(state_scores)
    finite_roots = np.asarray(roots)[np.isfinite(roots)]
    root_ratio = roots[-1] / roots[0] if np.isfinite(roots[0]) and np.isfinite(roots[-1]) else math.nan
    if len(finite_roots) >= 3 and max(finite_roots[1:-1], default=-math.inf) > 1.1 * max(finite_roots[0], finite_roots[-1]):
        thermal = "PEAK_PROXY"
    elif np.isfinite(root_ratio) and root_ratio > 1.25:
        thermal = "RISING_DBTT_PROXY"
    elif np.isfinite(root_ratio) and 0.8 <= root_ratio <= 1.25:
        thermal = "WEAK_T_PROXY"
    else:
        thermal = "CERAMIC_OR_FALLING_PROXY"
    median = float(np.median(np.log10(np.maximum(state_scores, 1e-300))))
    if median < -1.0:
        mechanism = "OPENING_DIRECTED"
    elif median <= 1.0:
        mechanism = "MIXED_DIRECTED"
    else:
        mechanism = "PLASTICITY_DIRECTED"
    rates = EVENT_LENGTH_M * opening[0]
    slopes = np.diff(np.log(np.maximum(rates, 1e-300))) / np.diff(np.log(LOADS))
    out.update({
        "mechanism_acquisition_stratum": mechanism,
        "thermal_acquisition_stratum": thermal,
        "median_log10_state_drive_ratio": median,
        "state_drive_log10_span": float(np.ptp(np.log10(np.maximum(state_scores, 1e-300)))),
        "A0_rate_300K_min_m_per_cycle": float(rates.min()),
        "A0_rate_300K_max_m_per_cycle": float(rates.max()),
        "A0_slope_300K_median": float(np.median(slopes)),
        "A0_slope_300K_min": float(slopes.min()),
        "A0_slope_300K_max": float(slopes.max()),
        "fracture_accessible_count": int(np.isfinite(roots).sum()),
        "fracture_root_ratio_1200_over_300": root_ratio,
        "descriptor_complete": True,
    })
    return out


def select_batch(metrics: list[dict[str, Any]], rows: list[dict[str, Any]], n: int) -> list[int]:
    by_id = {row["candidate_id"]: i for i, row in enumerate(rows)}
    selected: list[int] = []
    per = n // 3
    for stratum in ("OPENING_DIRECTED", "MIXED_DIRECTED", "PLASTICITY_DIRECTED"):
        eligible = [m for m in metrics if m.get("mechanism_acquisition_stratum") == stratum
                    and m.get("fracture_accessible_count", 0) >= 2
                    and 1.0 <= m.get("Kroot_300K", 0.0) <= 60.0
                    and m.get("A0_rate_300K_max_m_per_cycle", 0) > 1e-14
                    and m.get("A0_rate_300K_min_m_per_cycle", 1) < 1e-4]
        eligible.sort(key=lambda m: (m.get("thermal_acquisition_stratum", ""),
                                     abs(m.get("A0_slope_300K_median", 99) - 4),
                                     -m.get("state_drive_log10_span", 0), m["candidate_id"]))
        # Round-robin thermal classes before filling on score.
        groups: dict[str, list[dict[str, Any]]] = {}
        for m in eligible:
            groups.setdefault(m["thermal_acquisition_stratum"], []).append(m)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(chosen) < per:
                    chosen.append(groups[key].pop(0))
        selected.extend(by_id[m["candidate_id"]] for m in chosen)
    if len(selected) < n:
        reserve = [m for m in metrics if m.get("descriptor_complete") and by_id[m["candidate_id"]] not in selected]
        reserve.sort(key=lambda m: (-m.get("fracture_accessible_count", 0),
                                    -m.get("state_drive_log10_span", 0), m["candidate_id"]))
        selected.extend(by_id[m["candidate_id"]] for m in reserve[:n-len(selected)])
    if len(selected) != n:
        raise RuntimeError(f"could select only {len(selected)} of {n} requested native rows")
    return selected


def assign_relative_mechanism_strata(metrics: list[dict[str, Any]]) -> None:
    """Balance acquisition without pretending the cheap proxy is a final label."""
    valid = [m for m in metrics if m.get("descriptor_complete")]
    values = np.array([m["median_log10_state_drive_ratio"] for m in valid])
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for m in valid:
        value = m["median_log10_state_drive_ratio"]
        if value <= q1:
            m["mechanism_acquisition_stratum"] = "OPENING_DIRECTED"
        elif value <= q2:
            m["mechanism_acquisition_stratum"] = "MIXED_DIRECTED"
        else:
            m["mechanism_acquisition_stratum"] = "PLASTICITY_DIRECTED"
        m["mechanism_stratum_basis"] = "GLOBAL_VALID_BANK_TERCILE_OF_STATE_DRIVE_PROXY_NOT_FINAL_CLASSIFICATION"
        m["mechanism_stratum_q1"] = float(q1)
        m["mechanism_stratum_q2"] = float(q2)


def write_csv(path: Path, records: list[dict[str, Any]], compressed: bool = False) -> None:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original-registry", type=Path, required=True)
    p.add_argument("--parent-registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log2-bank", type=int, default=17)
    p.add_argument("--native-count", type=int, default=48)
    p.add_argument("--reuse-metrics", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    source = read_sources(args.original_registry, args.parent_registry)
    points = qmc.Sobol(d=22, scramble=True, seed=10162026).random_base2(args.log2_bank)
    rows = [generate_row(i, u, source) for i, u in enumerate(points)]
    existing = args.out / "cheap_global_bank.csv.gz"
    if args.reuse_metrics:
        if not existing.is_file():
            raise FileNotFoundError(existing)
        with gzip.open(existing, "rt", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        for m in metrics:
            for key in ("Kroot_300K", "Kroot_450K", "Kroot_600K", "Kroot_900K", "Kroot_1200K",
                        "A0_rate_300K_min_m_per_cycle", "A0_rate_300K_max_m_per_cycle",
                        "A0_slope_300K_median", "A0_slope_300K_min", "A0_slope_300K_max",
                        "median_log10_state_drive_ratio", "state_drive_log10_span",
                        "fracture_root_ratio_1200_over_300", "mechanism_stratum_q1", "mechanism_stratum_q2"):
                if m.get(key, "") not in ("", None):
                    m[key] = float(m[key])
            for key in ("fracture_accessible_count",):
                if m.get(key, "") not in ("", None): m[key] = int(m[key])
            for key in ("domain_pass", "descriptor_complete"):
                m[key] = str(m.get(key, "")).lower() == "true"
    else:
        metrics = []
        for i, row in enumerate(rows):
            metrics.append(descriptors(row))
            if i and i % 8192 == 0:
                print(json.dumps({"completed": i, "total": len(points)}), flush=True)
        assign_relative_mechanism_strata(metrics)
    chosen = select_batch(metrics, rows, args.native_count)
    chosen_ids = {rows[i]["candidate_id"] for i in chosen}
    for m in metrics:
        m["selected_native_batch"] = m["candidate_id"] in chosen_ids
    write_csv(args.out / "cheap_global_bank.csv.gz", metrics, compressed=True)
    state_pool = [m for m in metrics if m.get("descriptor_complete")]
    state_pool.sort(key=lambda m: (m["mechanism_acquisition_stratum"],
                                   m["thermal_acquisition_stratum"],
                                   -m["state_drive_log10_span"], m["candidate_id"]))
    stride = max(len(state_pool) // 4096, 1)
    write_csv(args.out / "state_reduction_allocation.csv", state_pool[::stride][:4096])
    registry = []
    selected_metrics = {m["candidate_id"]: m for m in metrics if m["candidate_id"] in chosen_ids}
    for i in chosen:
        row = rows[i]
        row_hash = canonical_hash(row)
        registry.append({
            "candidate_id": row["candidate_id"], "family": FAMILY,
            "complete_row_sha256": row_hash, "cleavage_hits": HITS,
            "cleavage_tau_s": TAU_S,
            "mechanism_acquisition_stratum": selected_metrics[row["candidate_id"]]["mechanism_acquisition_stratum"],
            "thermal_acquisition_stratum": selected_metrics[row["candidate_id"]]["thermal_acquisition_stratum"],
            "complete_parameter_vector": json.dumps(row, sort_keys=True, separators=(",", ":")),
        })
    write_csv(args.out / "native_batch_registry.csv", registry)
    write_csv(args.out / "native_batch_descriptors.csv", [selected_metrics[r["candidate_id"]] for r in registry])
    counts: dict[str, int] = {}
    rejects: dict[str, int] = {}
    for m in metrics:
        key = m.get("mechanism_acquisition_stratum") or "DOMAIN_REJECT"
        counts[key] = counts.get(key, 0) + 1
        if m.get("rejection"):
            rejects[m["rejection"]] = rejects.get(m["rejection"], 0) + 1
    reward_mean = EVENT_LENGTH_M * clipped_exponential_mean(0.5, 4.0) / clipped_exponential_mean(0.5, 4.0)
    decision = {
        "schema": MODEL, "status": "ANALYTICAL_SELECTION_COMPLETE_NATIVE_BATCH_PENDING",
        "family": FAMILY, "bank_size": len(rows), "domain_and_stratum_counts": counts,
        "domain_rejections": rejects, "native_batch_size": len(registry),
        "operational_thresholds_frozen_before_native_results": {
            "mixed_absolute_slope_change": 0.5, "strong_absolute_slope_change": 1.0,
            "rate_level_factor": 1.5, "two_percent_is_not_material": True,
        },
        "A0_unit_identity": {
            "producer": "cooperative opening action per cycle",
            "consumer": "stationary renewal-reward metres per cycle",
            "expected_action_mark": 1.0, "expected_event_length_m": reward_mean,
            "conversion_m_per_action": reward_mean,
            "derivation": "E[L]/E[Xi] under unit-mean exponential Xi and mean-preserving clipped correlated reward",
            "fitted_attenuation_used": False,
        },
        "constitutive_invariants": {
            "single_EXP_floor_only": True, "empirical_temperature_law": False,
            "independent_temperature_endpoints": False, "guard_or_broad_basis": False,
            "attempt_frequencies_varied": False, "transport_stress_scale_shared_with_emission": True,
            "source_geometry_and_event_law_varied": False,
        },
        "files": {},
    }
    for name in ("cheap_global_bank.csv.gz", "state_reduction_allocation.csv",
                 "native_batch_registry.csv", "native_batch_descriptors.csv"):
        path = args.out / name
        decision["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "bytes": path.stat().st_size}
    (args.out / "analytical_selection_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
