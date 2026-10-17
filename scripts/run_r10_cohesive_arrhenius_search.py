#!/usr/bin/env python3
"""Deterministic analytical acquisition for the additive cohesive R10 family."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammainc, lambertw
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import EV_TO_J, FAMILY
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.run_r10_global_mechanism_search import (
    EVENT_LENGTH_M, FREQUENCY_HZ, HITS, LOADS, R_RATIO, TAU_S,
    canonical_hash, phase_action, read_sources, transport_rate,
)

MODEL = "r10.cohesive-arrhenius-global-acquisition/1"
SEED = 10162026
TEMPERATURES = np.array([300., 450., 600., 750., 900., 1050., 1200.])
KGRID = np.linspace(0.05, 80.0, 420)


def _log_range(u, lo, hi):
    return math.exp(math.log(lo) + float(u) * math.log(hi / lo))


def generate(index: int, u: np.ndarray, sources: list[dict]) -> dict:
    source_index = min(int(u[0] * len(sources)), len(sources) - 1)
    parent = dict(sources[source_index])
    gamma = _log_range(u[1], 0.5, 8.0)
    t_hat = _log_range(u[2], 8.0e9, 80.0e9)
    zero_barrier_eV = _log_range(u[3], 0.35, 6.0)
    activation_area = zero_barrier_eV * EV_TO_J / gamma
    if u[4] < 0.5:
        thermal_kind = "CONSTANT_COHESIVE_SCALE"
        S_gamma = 0.0
    else:
        thermal_kind = "MINIMAL_COUPLED_COHESIVE_ENTROPY"
        # Fractional Gamma change across 300--1200 K. Negative values harden.
        span_fraction = -0.25 + 1.0 * (2.0 * (u[4] - 0.5))
        S_gamma = span_fraction * gamma / 900.0
    cohesive = {
        "Gamma_ref_J_m2": gamma, "t_hat_ref_Pa": t_hat,
        "S_Gamma_J_m2_K": S_gamma, "T_ref_K": 300.0,
        "activation_area_m2": activation_area,
        "attempt_frequency_s": 1.0e12,
        "delta0_m": gamma / (math.e * t_hat),
        "thermal_subfamily": thermal_kind,
        "zero_traction_barrier_eV_300K": zero_barrier_eV,
    }
    cid = f"R10D_COH_{index:05d}"
    return {"candidate_id": cid, "parent_candidate_id": parent["candidate_id"],
            "parent_source_id": parent.get("search_source_id", parent["candidate_id"]),
            "parent": parent, "cohesive": cohesive}


def gamma_T(c: dict, T: float) -> float:
    return c["Gamma_ref_J_m2"] - (T - 300.0) * c["S_Gamma_J_m2_K"]


def cohesive_barrier_eV(stress, c: dict, T: float):
    stress = np.asarray(stress, float)
    gamma = gamma_T(c, T)
    peak = gamma / (math.e * c["delta0_m"])
    s = stress / peak
    out = np.empty_like(s)
    zero = s <= 0.0; spinodal = s >= 1.0; mid = ~(zero | spinodal)
    out[zero] = c["activation_area_m2"] * gamma / EV_TO_J
    out[spinodal] = 0.0
    if np.any(mid):
        sm = s[mid]; z = -sm / math.e
        xm = -lambertw(z, 0).real
        xs = -lambertw(z, -1).real
        eps = 1.0-sm
        exact = ((1+xm)*np.exp(-xm) - (1+xs)*np.exp(-xs)
                 - sm/math.e*(xs-xm))
        asym = 4*math.sqrt(2)/(3*math.e)*eps**1.5
        shape = np.where(eps <= 1e-7, asym, np.maximum(exact, 0.0))
        out[mid] = c["activation_area_m2"] * gamma * shape / EV_TO_J
    return out


def opening_action(c: dict, Kmax: float, T: float) -> float:
    phase = 2*math.pi*(np.arange(64)+0.5)/64
    h = np.maximum(0.5*(1+R_RATIO)+0.5*(1-R_RATIO)*np.cos(phase), 0.0)
    stress = np.minimum(Kmax*1e6*h/math.sqrt(2*math.pi*1e-6), 30e9)
    barrier = cohesive_barrier_eV(stress, c, T)
    raw = 1e12*np.exp(np.clip(-barrier/(KB_EV_PER_K*T), -700, 0))
    complete = gammainc(HITS, raw*TAU_S)/TAU_S
    return float(np.mean(complete)/FREQUENCY_HZ)


def ramp_root(c: dict, T: float, Kdot=0.005) -> float:
    stress = KGRID*1e6/math.sqrt(2*math.pi*1e-6)
    barrier = cohesive_barrier_eV(stress, c, T)
    raw = 1e12*np.exp(np.clip(-barrier/(KB_EV_PER_K*T), -700, 0))
    rate = gammainc(HITS, raw*TAU_S)/TAU_S
    action = np.r_[0., np.cumsum(0.5*(rate[1:]+rate[:-1])*np.diff(KGRID)/Kdot)]
    j = int(np.searchsorted(action, 1.0))
    if j >= len(KGRID): return math.nan
    if j == 0: return float(KGRID[0])
    f = (1-action[j-1])/max(action[j]-action[j-1], 1e-300)
    return float(KGRID[j-1]+f*(KGRID[j]-KGRID[j-1]))


def evaluate(item: dict) -> dict:
    c = item["cohesive"]; parent = item["parent"]
    result = {"candidate_id": item["candidate_id"],
              "parent_candidate_id": item["parent_candidate_id"],
              "thermal_subfamily": c["thermal_subfamily"]}
    gammas = np.array([gamma_T(c, T) for T in TEMPERATURES])
    if np.min(gammas) <= 0.2:
        return {**result, "domain_pass": False,
                "rejection": "COHESIVE_GAMMA_NONPOSITIVE_OR_TOO_SMALL"}
    roots = np.array([ramp_root(c, T) for T in TEMPERATURES])
    actions = np.array([opening_action(c, K, 300.) for K in LOADS])
    rates = EVENT_LENGTH_M * actions
    slopes = np.diff(np.log(np.maximum(rates, 1e-300))) / np.diff(np.log(LOADS))
    emission = np.array([phase_action(parent, "emit", K, 300.) for K in LOADS])
    source_mult = max(float(parent["rho_source0_m2"])
                      * float(parent["reference_source_area_um2"])*1e-12, 1.)
    state = []
    for K, o, e in zip(LOADS, actions, emission):
        stress = min(K*1e6/math.sqrt(2*math.pi*1e-6), 30e9)
        mobility = math.sqrt(max(transport_rate(parent, "peierls", stress, 300.), 1e-300)
                             * max(transport_rate(parent, "taylor", stress, 300.), 1e-300))/FREQUENCY_HZ
        state.append(e*source_mult*min(mobility, 1e6)/max(o, 1e-300))
    state_score = float(np.median(np.log10(np.maximum(state, 1e-300))))
    if state_score < -1: mechanism = "OPENING_DIRECTED"
    elif state_score <= 1: mechanism = "MIXED_DIRECTED"
    else: mechanism = "PLASTICITY_DIRECTED"
    finite = roots[np.isfinite(roots)]
    ratio = roots[-1]/roots[0] if np.isfinite(roots[[0,-1]]).all() else math.nan
    interior_peak = (len(finite) == len(roots)
                     and np.max(roots[1:-1]) > 1.10*max(roots[0], roots[-1]))
    if interior_peak: thermal = "PEAK_PROXY"
    elif math.isfinite(ratio) and ratio > 1.25: thermal = "RISING_DBTT_PROXY"
    elif len(finite) == len(roots) and np.max(roots)/np.min(roots) <= 1.25:
        thermal = "WEAK_T_PROXY"
    else: thermal = "OTHER_OR_CENSORED_PROXY"
    zone = 400e9/(1-0.22**2)*c["Gamma_ref_J_m2"]/c["t_hat_ref_Pa"]**2
    return {**result, "domain_pass": True, "rejection": "",
            "mechanism_acquisition_stratum": mechanism,
            "thermal_acquisition_stratum": thermal,
            "median_log10_state_drive_ratio": state_score,
            "fracture_accessible_count": int(np.isfinite(roots).sum()),
            "fracture_root_ratio_1200_over_300": ratio,
            **{f"Kroot_{int(T)}K": roots[i] for i, T in enumerate(TEMPERATURES)},
            "A0_rate_300K_min_m_per_cycle": float(np.min(rates)),
            "A0_rate_300K_max_m_per_cycle": float(np.max(rates)),
            "A0_slope_300K_median": float(np.median(slopes)),
            "A0_slope_300K_min": float(np.min(slopes)),
            "A0_slope_300K_max": float(np.max(slopes)),
            "cohesive_zone_scale_m": zone,
            "activation_area_m2": c["activation_area_m2"],
            "delta0_m": c["delta0_m"]}


def write_csv(path: Path, rows: list[dict], compressed=False):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def select(metrics: list[dict], count: int) -> list[dict]:
    chosen = []
    per = max(count//9, 1)
    for thermal in ("RISING_DBTT_PROXY", "PEAK_PROXY", "WEAK_T_PROXY"):
        for mechanism in ("OPENING_DIRECTED", "MIXED_DIRECTED", "PLASTICITY_DIRECTED"):
            pool = [m for m in metrics if m.get("domain_pass") and
                    m.get("thermal_acquisition_stratum") == thermal and
                    m.get("mechanism_acquisition_stratum") == mechanism and
                    m.get("fracture_accessible_count", 0) >= 5]
            pool.sort(key=lambda m: (abs(m["A0_slope_300K_median"]-4),
                                     m["candidate_id"]))
            chosen.extend(pool[:per])
    seen = {m["candidate_id"] for m in chosen}
    reserve = [m for m in metrics if m.get("domain_pass") and m["candidate_id"] not in seen]
    reserve.sort(key=lambda m: (-m.get("fracture_accessible_count", 0),
                                abs(m.get("A0_slope_300K_median", 99)-4),
                                m["candidate_id"]))
    return (chosen + reserve)[:count]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--original-registry", type=Path, required=True)
    p.add_argument("--parent-registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log2-bank", type=int, default=15)
    p.add_argument("--native-count", type=int, default=48)
    a = p.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True)
    sources = read_sources(a.original_registry, a.parent_registry)
    points = qmc.Sobol(d=5, scramble=True, seed=SEED).random_base2(a.log2_bank)
    items = [generate(i, u, sources) for i, u in enumerate(points)]
    metrics = []
    for i, item in enumerate(items):
        metrics.append(evaluate(item))
        if i and i % 4096 == 0:
            print(json.dumps({"analytically_evaluated": i, "allocated": len(items)}), flush=True)
    selected = select(metrics, a.native_count)
    selected_ids = {m["candidate_id"] for m in selected}
    for m in metrics: m["selected_native_batch"] = m["candidate_id"] in selected_ids
    write_csv(a.out/"analytical_bank.csv.gz", metrics, True)
    valid = [m for m in metrics if m.get("domain_pass")]
    valid.sort(key=lambda m: (m.get("mechanism_acquisition_stratum", ""),
                              m.get("thermal_acquisition_stratum", ""), m["candidate_id"]))
    stride = max(len(valid)//4096, 1)
    write_csv(a.out/"state_aware_allocation.csv", valid[::stride][:4096])
    by_id = {item["candidate_id"]: item for item in items}
    registry = []
    for m in selected:
        item = by_id[m["candidate_id"]]; c = item["cohesive"]; parent = item["parent"]
        c_hash = canonical_hash(c); p_hash = canonical_hash(parent)
        complete = {"family": FAMILY, "candidate_id": item["candidate_id"],
                    "parent_sha256": p_hash, "cohesive_sha256": c_hash}
        registry.append({"candidate_id": item["candidate_id"], "family": FAMILY,
            "complete_row_sha256": canonical_hash(complete),
            "parent_candidate_id": item["parent_candidate_id"],
            "parent_complete_row_sha256": p_hash,
            "cohesive_parameter_sha256": c_hash,
            "cleavage_hits": HITS, "cleavage_tau_s": TAU_S,
            "mechanism_acquisition_stratum": m["mechanism_acquisition_stratum"],
            "thermal_acquisition_stratum": m["thermal_acquisition_stratum"],
            "cohesive_parameter_vector": json.dumps(c, sort_keys=True, separators=(",", ":")),
            "parent_complete_parameter_vector": json.dumps(parent, sort_keys=True, separators=(",", ":"))})
    write_csv(a.out/"native_batch_registry.csv", registry)
    write_csv(a.out/"native_batch_descriptors.csv", selected)
    counts = {}
    for m in metrics:
        key = "DOMAIN_REJECT" if not m.get("domain_pass") else (
            m["thermal_acquisition_stratum"]+"__"+m["mechanism_acquisition_stratum"])
        counts[key] = counts.get(key, 0)+1
    protocol = {"schema": MODEL, "family": FAMILY, "seed": SEED,
        "allocated": len(items), "analytically_evaluated": len(metrics),
        "domain_valid": len(valid), "state_aware_allocated_not_executed": min(4096, len(valid)),
        "native_queued_not_executed": len(registry), "counts": counts,
        "fixed_protocol": {"temperature_K": TEMPERATURES.tolist(), "fatigue_T_K": 300,
            "R": R_RATIO, "frequency_Hz": FREQUENCY_HZ, "cleavage_hits": HITS,
            "cleavage_tau_s": TAU_S, "attempt_frequency_s": 1e12,
            "developed_window_um": [20, 60], "initial_target_um": 70,
            "promotion_minimum_amplitudes": 5, "finalist_minimum_events": 50,
            "Paris_adjacent_slope_interval": [3.4, 4.6], "connected_rate_decades": 2},
        "published_priors_before_native_selection": {
            "Gamma_ref_J_m2": [0.5, 8.0], "t_hat_ref_Pa": [8e9, 80e9],
            "zero_traction_barrier_eV_300K": [0.35, 6.0],
            "Gamma_1200_minus_300_fraction": [-0.75, 0.25],
            "delta0_derived_not_independent": True,
            "activation_area_derived_from_zero_barrier_and_Gamma": True},
        "execution_counts_are_not_allocations": True, "files": {}}
    for name in ("analytical_bank.csv.gz", "state_aware_allocation.csv",
                 "native_batch_registry.csv", "native_batch_descriptors.csv"):
        path = a.out/name
        protocol["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "bytes": path.stat().st_size}
    (a.out/"prospective_search_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True)+"\n")
    print(json.dumps(protocol, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
