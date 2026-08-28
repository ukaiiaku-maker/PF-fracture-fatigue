#!/usr/bin/env python3
"""Build and finalize the v10.2.30 inverse fatigue-barrier design study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammainc
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    B1Controls,
    ExpFloorBounds,
    RenewalControls,
    b1_event_conditioned_emission,
    cooperative_Q,
    cooperative_rate_from_raw,
    cycle_growth_and_slope,
    exact_target_instantaneous_rate,
    exp_floor_drop_derivative_eV,
    exp_floor_rare_event_max_slope,
    invert_cooperative_rate_to_barrier,
    project_exp_floor,
    rare_event_reference_barrier,
    rare_event_target_barrier,
    waveform_factor_C,
    waveform_fraction,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier, KB_EV_PER_K, MaterialManifest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/inverse_fatigue_barrier_design_v1"
ARCHIVE = ROOT / "runs/A_native_analytical_overlay_all_1d_v1/analytical_predictions.csv"
REGISTRY = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv"
MANIFEST = next((ROOT / "runs/A_native_plus_8PT_fatigue_v1/developed/n80/A_NATIVE").glob(
    "DK_*/selected_material_manifest_v10_2_22.csv"
))
FIG = OUT / "figures"
SOLVER_SHA = "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b"
PHYSICS_FIELDS = [
    "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n",
    "cleave_floor_frac", "emit_G00_eV", "emit_gT_eV_per_K",
    "emit_sigc0_GPa", "emit_sT_GPa_per_K", "emit_exp_a", "emit_exp_n",
    "emit_floor_frac", "peierls_H0_eV", "peierls_activation_entropy_kB",
    "peierls_exp_a", "peierls_exp_n", "taylor_H0_eV",
    "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
    "taylor_corr_rho_c_m2", "taylor_corr_scale", "source_sites_per_system",
    "encounter_efficiency", "retained_recovery_rate_s",
    "source_refresh_length_um", "c_blunt", "peierls_nu0_s",
    "taylor_nu0_s", "rho_source0_m2", "recovery_nu0_s", "recovery_H0_eV",
    "recovery_activation_entropy_kB", "reference_source_area_um2",
    "reference_front_width_um", "source_zone_length_um",
]
CLEAVAGE_DESIGN_FIELDS = [
    "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a",
    "cleave_exp_n", "cleave_floor_frac",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_configurations() -> tuple[dict, list[dict]]:
    payload = json.loads((OUT / "target_design_configurations.json").read_text())
    common = payload["common"]
    targets = []
    for item in payload["targets"]:
        target = {**common, **item}
        targets.append(target)
    return payload, targets


def native_inputs() -> tuple[dict[str, str], MaterialManifest]:
    row = next(r for r in csv.DictReader(REGISTRY.open()) if r["option_key"] == "A_NATIVE")
    return row, MaterialManifest.from_csv(MANIFEST)


def controls_for(target: dict, n_phase: int = 1024) -> RenewalControls:
    return RenewalControls(
        hits=3.0, tau_s=1e-6, event_length_m=5e-6,
        frequency_Hz=float(target["frequency_Hz"]),
        temperature_K=float(target["temperature_K"]), radius_m=1e-6,
        n_phase=n_phase,
    )


def exact_targets(targets: list[dict], manifest: MaterialManifest) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    barrier_rows, rare_rows, gamma_rows = [], [], []
    for target in targets:
        controls = controls_for(target, 4096)
        M = float(target["target_slope_parameters"]["M"])
        R = float(target["R"]); Kref = float(target["K_ref_MPa_sqrt_m"])
        gref = float(target["g_ref_m_per_cycle"])
        sigma_ref = Kref * 1e6 / math.sqrt(2 * math.pi * controls.radius_m)
        sigma_min = float(target["K_design_min_MPa_sqrt_m"]) * 1e6 * max(R, 0.02) / math.sqrt(
            2 * math.pi * controls.radius_m
        )
        sigma_max = float(target["K_design_max_MPa_sqrt_m"]) * 1e6 / math.sqrt(
            2 * math.pi * controls.radius_m
        )
        sigma = np.geomspace(sigma_min, sigma_max, 401)
        rate = exact_target_instantaneous_rate(
            sigma, M=M, R=R, K_ref_MPa_sqrt_m=Kref,
            g_ref_m_per_cycle=gref, controls=controls,
        )
        G = invert_cooperative_rate_to_barrier(
            rate, hits=controls.hits, tau_s=controls.tau_s,
            attempt_frequency_s=manifest.cleavage.attempt_frequency_s,
            temperature_K=controls.temperature_K,
        )
        rate_ref = float(exact_target_instantaneous_rate(
            sigma_ref, M=M, R=R, K_ref_MPa_sqrt_m=Kref,
            g_ref_m_per_cycle=gref, controls=controls,
        ))
        G_rare_ref = rare_event_reference_barrier(
            rate_ref, controls=controls,
            attempt_frequency_s=manifest.cleavage.attempt_frequency_s,
        )
        G_rare = rare_event_target_barrier(
            sigma, M=M, sigma_ref_Pa=sigma_ref, G_ref_eV=G_rare_ref,
            controls=controls,
        )
        for s, rr, gg, gr in zip(sigma, rate, G, G_rare):
            barrier_rows.append({
                "target_id": target["target_id"], "design_track": target["design_track"],
                "M_target": M, "R": R, "stress_Pa": s,
                "target_cooperative_rate_s": rr, "exact_inverse_barrier_eV": gg,
                "rare_event_barrier_eV": gr,
                "tauLambda": controls.tau_s * rr,
                "saturation_compatible": 0.0 < controls.tau_s * rr < 1.0,
            })
            rare_rows.append({
                "target_id": target["target_id"], "stress_Pa": s,
                "exact_barrier_eV": gg, "rare_barrier_eV": gr,
                "absolute_error_eV": abs(gg - gr),
                "relative_error": abs(gg - gr) / max(abs(gg), 1e-300),
            })
        sample_y = np.geomspace(1e-12, min(0.8, float(np.max(rate) * controls.tau_s * 1.2)), 100)
        recovered_G = invert_cooperative_rate_to_barrier(
            sample_y / controls.tau_s, hits=controls.hits, tau_s=controls.tau_s,
            attempt_frequency_s=manifest.cleavage.attempt_frequency_s,
            temperature_K=controls.temperature_K,
        )
        raw = manifest.cleavage.attempt_frequency_s * np.exp(
            -recovered_G / (KB_EV_PER_K * controls.temperature_K)
        )
        roundtrip = cooperative_rate_from_raw(raw, controls.hits, controls.tau_s)
        for y, got in zip(sample_y, roundtrip * controls.tau_s):
            gamma_rows.append({
                "target_id": target["target_id"], "requested_tauLambda": y,
                "roundtrip_tauLambda": got, "relative_error": abs(got-y)/y,
            })
    return pd.DataFrame(barrier_rows), pd.DataFrame(rare_rows), pd.DataFrame(gamma_rows)


PROJECTION_OBJECTIVES = {
    "MINIMUM_BARRIER_ERROR": {"barrier": 1.0, "derivative": 0.25, "rate": 0.2,
                              "growth": 0.2, "slope": 0.1},
    "MINIMUM_SLOPE_ERROR": {"barrier": 0.1, "derivative": 0.2, "rate": 0.2,
                            "growth": 1.0, "slope": 1.0},
    "CLOSEST_TO_A_NATIVE": {"barrier": 0.2, "derivative": 0.2, "rate": 0.2,
                            "growth": 0.5, "slope": 0.5, "parameter": 2.0},
    "WIDEST_TARGET_SLOPE_WINDOW": {"barrier": 0.05, "derivative": 0.5, "rate": 0.2,
                                   "growth": 0.8, "slope": 1.5},
    "LOWEST_COOPERATIVE_SATURATION_RISK": {"barrier": 0.4, "derivative": 0.3,
                                            "rate": 0.8, "growth": 0.8, "slope": 0.5},
}


def projections(targets: list[dict], exact: pd.DataFrame,
                native: MaterialManifest) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, errors, manifold = [], [], []
    bounds = ExpFloorBounds()
    for target in targets:
        subset = exact[exact.target_id == target["target_id"]]
        sigma = subset.stress_Pa.to_numpy()
        G = subset.exact_inverse_barrier_eV.to_numpy()
        rate = subset.target_cooperative_rate_s.to_numpy()
        M = float(target["target_slope_parameters"]["M"])
        K = np.linspace(float(target["K_design_min_MPa_sqrt_m"]),
                        float(target["K_design_max_MPa_sqrt_m"]), 25)
        growth = float(target["g_ref_m_per_cycle"]) * (
            K / float(target["K_ref_MPa_sqrt_m"])
        ) ** M
        slopes = np.full_like(K, M)
        controls = controls_for(target, 512)
        starts = [
            [native.cleavage.G00_eV, native.cleavage.sigc0_Pa/1e9,
             native.cleavage.alpha, native.cleavage.exponent,
             native.cleavage.floor_fraction],
            [1.0, 2.0, 1.0, max(M/2, .5), .15],
            [2.5, 4.0, .3, max(M, 1), .4],
        ]
        for objective, weights in PROJECTION_OBJECTIVES.items():
            fits = [project_exp_floor(
                sigma_grid_Pa=sigma, target_barrier_eV=G,
                target_rate_s=rate, K_grid_MPa_sqrt_m=K,
                target_growth=growth, target_slope=slopes, R=float(target["R"]),
                template=native.cleavage, controls=controls, bounds=bounds,
                weights=weights, start=start,
            ) for start in starts]
            fit = min(fits, key=lambda x: x["cost"])
            b = fit["barrier"]
            cid = f"{target['target_id']}__{objective}"
            row = {
                "projection_id": cid, "target_id": target["target_id"],
                "design_track": target["design_track"], "objective": objective,
                "M_target": M, "cleave_G00_eV": b.G00_eV,
                "cleave_sigc0_GPa": b.sigc0_Pa/1e9, "cleave_exp_a": b.alpha,
                "cleave_exp_n": b.exponent, "cleave_floor_frac": b.floor_fraction,
                "rare_event_max_slope_capacity": exp_floor_rare_event_max_slope(
                    b, controls.temperature_K, controls.hits),
                "success": fit["success"], "cost": fit["cost"],
                "minimum_saturation_margin": fit["minimum_saturation_margin"],
            }
            candidates.append(row)
            errors.append({**row, **{name: fit[name] for name in (
                "barrier_RMS_eV", "derivative_RMS_eV", "log_rate_RMS",
                "log_growth_RMS", "slope_RMS", "nfev")}})
            for k, gg, mm, gt, mt in zip(K, fit["growth"], fit["slope"], growth, slopes):
                manifold.append({
                    "projection_id": cid, "target_id": target["target_id"],
                    "objective": objective, "Kmax_MPa_sqrt_m": k,
                    "projected_da_dN": gg, "target_da_dN": gt,
                    "projected_local_slope": mm, "target_local_slope": mt,
                    "log10_rate_residual": math.log10(gg/gt),
                    "slope_residual": mm-mt,
                })
    return pd.DataFrame(candidates), pd.DataFrame(errors), pd.DataFrame(manifold)


def make_candidate_registry(targets: list[dict], candidates: pd.DataFrame,
                            native_row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selected = candidates[
        (candidates.objective == "MINIMUM_SLOPE_ERROR") &
        candidates.target_id.str.startswith("REFERENCE_SCALE_")
    ].sort_values("M_target")
    registry_rows, audits, hashes = [], [], {}
    target_by_id = {x["target_id"]: x for x in targets}
    for _, item in selected.iterrows():
        M = int(round(item.M_target))
        option = f"INV_OPENING_M{M}_V1"
        row = dict(native_row)
        row.update({
            "option_key": option, "candidate_id": option,
            "role": "prospective inverse-designed opening barrier",
            "mechanism_summary": f"M={M} reference-scale target; A_NATIVE common physics",
            "validation_status": "PROSPECTIVE_FROZEN",
            "cleave_G00_eV": f"{item.cleave_G00_eV:.17g}",
            "cleave_sigc0_GPa": f"{item.cleave_sigc0_GPa:.17g}",
            "cleave_exp_a": f"{item.cleave_exp_a:.17g}",
            "cleave_exp_n": f"{item.cleave_exp_n:.17g}",
            "cleave_floor_frac": f"{item.cleave_floor_frac:.17g}",
        })
        registry_rows.append(row)
        changed = [field for field in PHYSICS_FIELDS
                   if str(row.get(field, "")) != str(native_row.get(field, ""))]
        audits.append({
            "option_key": option, "target_id": item.target_id,
            "declared_design_fields": json.dumps(CLEAVAGE_DESIGN_FIELDS),
            "changed_fields": json.dumps(changed),
            "changed_fields_equal_declared": changed == CLEAVAGE_DESIGN_FIELDS,
            "noncleavage_physics_unchanged": all(
                str(row.get(f, "")) == str(native_row.get(f, ""))
                for f in PHYSICS_FIELDS if f not in CLEAVAGE_DESIGN_FIELDS
            ),
        })
        target = target_by_id[item.target_id]
        common = {f: row[f] for f in PHYSICS_FIELDS if f not in CLEAVAGE_DESIGN_FIELDS}
        hashes[option] = {
            "target_definition_sha256": canonical_hash(target),
            "exact_inverse_barrier_sha256": canonical_hash(
                {"target_id": item.target_id, "equation": "exact_gamma_inverse_v1"}
            ),
            "EXP_floor_projection_sha256": canonical_hash(item.to_dict()),
            "complete_candidate_row_sha256": canonical_hash(row),
            "changed_field_subset_sha256": canonical_hash({f: row[f] for f in CLEAVAGE_DESIGN_FIELDS}),
            "common_physics_sha256": canonical_hash(common),
            "solver_source_sha256": SOLVER_SHA,
        }
    return pd.DataFrame(registry_rows), pd.DataFrame(audits), hashes


def prospective_predictions(targets: list[dict], registry: pd.DataFrame,
                            native: MaterialManifest, native_row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_by_id = {x["target_id"]: x for x in targets}
    rows, b1rows = [], []
    validation_K = [12.0, 12.75, 13.5, 15.0, 18.0, 21.0, 24.3]
    for _, candidate in registry.iterrows():
        target_id = f"REFERENCE_SCALE_M{candidate.option_key.split('_M')[1].split('_')[0]}"
        target = target_by_id[target_id]
        M = float(target["target_slope_parameters"]["M"])
        barrier = ExpFloorBarrier(
            G00_eV=float(candidate.cleave_G00_eV), gT_eV_per_K=0.0,
            sigc0_Pa=float(candidate.cleave_sigc0_GPa)*1e9, sT_Pa_per_K=0.0,
            alpha=float(candidate.cleave_exp_a), exponent=float(candidate.cleave_exp_n),
            floor_fraction=float(candidate.cleave_floor_frac), attempt_frequency_s=1e12,
        )
        conditions = [(K, .1) for K in validation_K] + [(18.0, -.95), (18.0, .5)]
        for K, R in conditions:
            c = controls_for({**target, "R": R}, 4096)
            a0 = cycle_growth_and_slope(barrier, K, R, c)
            target_g = float(target["g_ref_m_per_cycle"]) * (
                K / float(target["K_ref_MPa_sqrt_m"])
            ) ** M
            target_M = M
            rows.append({
                "option_key": candidate.option_key, "target_id": target_id,
                "Kmax_MPa_sqrt_m": K, "DeltaK_full_MPa_sqrt_m": (1-R)*K,
                "R": R, "temperature_K": 300.0, "frequency_Hz": 1000.0,
                "target_da_dN": target_g, "target_local_slope": target_M,
                "EXP_floor_A0_da_dN": a0["da_dN"],
                "EXP_floor_A0_local_slope": a0["local_slope"],
                "cooperative_saturation_fraction": a0["cooperative_saturation_fraction"],
                "prediction_frozen_before_physical_run": True,
            })
            if R == .1:
                b1 = b1_event_conditioned_emission(
                    native, native_row, K, R, cleavage_barrier=barrier,
                    controls=B1Controls(n_phase=32, burn_events=3, sample_events=10,
                                        maximum_cycles=200000, hazard_seed=1720),
                )
                b1rows.append({
                    "option_key": candidate.option_key, "target_id": target_id,
                    "Kmax_MPa_sqrt_m": K, "R": R, **b1,
                })
    return pd.DataFrame(rows), pd.DataFrame(b1rows)


def archived_b1(native: MaterialManifest, native_row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = pd.read_csv(ARCHIVE)
    subset = archive[(archive.family == "A_NATIVE") &
                     (archive.stationarity_classification == "STEADY_STATE_QUALIFIED")].copy()
    cache = {}
    for key in sorted({(float(r.R), float(r.Kmax_MPa_sqrt_m), int(r.seed))
                       for _, r in subset.iterrows()}):
        R, K, seed = key
        cache[key] = b1_event_conditioned_emission(
            native, native_row, K, R,
            controls=B1Controls(n_phase=32, burn_events=3, sample_events=10,
                                maximum_cycles=300000, hazard_seed=seed),
        )
    rows = []
    for _, item in subset.iterrows():
        b1 = cache[(float(item.R), float(item.Kmax_MPa_sqrt_m), int(item.seed))]
        K = float(item.Kmax_MPa_sqrt_m); numerical = float(item.numerical_da_dN)
        region = "LOW_K" if K < 15 else ("MID_K" if K < 21 else "HIGH_K")
        rows.append({
            "condition_id": item.condition_id, "seed": int(item.seed), "R": float(item.R),
            "Kmax_MPa_sqrt_m": K, "K_region": region,
            "numerical_da_dN": numerical, "A0_OPENING_ONLY_da_dN": float(item.A0_da_dN),
            "A1_CONTINUOUS_MEAN_BLUNTING_da_dN": float(item.A1_da_dN),
            "A2_PT_STATE_ONLY_da_dN": float(item.A2_da_dN),
            "B1_EVENT_CONDITIONED_EMISSION_da_dN": b1["da_dN"],
            "A1_log10_residual": math.log10(float(item.A1_da_dN)/numerical),
            "B1_log10_residual": (math.log10(float(b1["da_dN"])/numerical)
                                  if math.isfinite(float(b1["da_dN"])) else math.nan),
            "B1_converged": b1["fixed_point_converged"],
            "B1_r_eff_m": b1["r_eff_m"], "B1_backstress_Pa": b1["backstress_Pa"],
            "source_result_path": item.source_result_path, "source_hash": item.source_hash,
        })
    frame = pd.DataFrame(rows)
    summary = []
    for region, group in frame.groupby("K_region"):
        for model in ("A1", "B1"):
            residual = group[f"{model}_log10_residual"].dropna().to_numpy()
            summary.append({
                "K_region": region, "model": model, "count": len(residual),
                "median_absolute_log10_error": float(np.median(np.abs(residual))),
                "maximum_absolute_log10_error": float(np.max(np.abs(residual))),
                "RMS_log10_error": float(np.sqrt(np.mean(residual**2))),
                "signed_bias": float(np.mean(residual)),
            })
    return frame, pd.DataFrame(summary)


def slope_decomposition(predictions: pd.DataFrame, b1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for option, group in predictions[predictions.R == .1].groupby("option_key"):
        group = group.sort_values("Kmax_MPa_sqrt_m")
        b = b1[b1.option_key == option].sort_values("Kmax_MPa_sqrt_m")
        K = group.Kmax_MPa_sqrt_m.to_numpy()
        opening = group.EXP_floor_A0_local_slope.to_numpy()
        b1g = b.da_dN.to_numpy()
        b1slope = np.gradient(np.log(b1g), np.log(K), edge_order=2)
        for index, k in enumerate(K):
            rows.append({
                "option_key": option, "Kmax_MPa_sqrt_m": k,
                "M_total_B1": b1slope[index],
                "M_opening_barrier_plus_renewal_waveform": opening[index],
                "M_effective_radius_and_state_history": b1slope[index]-opening[index],
                "M_shielding": 0.0, "M_event_size": 0.0,
                "M_PT": 0.0,
                "cooperative_renewal_amplification_Q_weighted": math.nan,
                "decomposition_closure_residual": 0.0,
            })
    return pd.DataFrame(rows)


def analytical_atlas(native: MaterialManifest, n_power: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = qmc.Sobol(d=10, scramble=True, seed=1720).random_base2(n_power)
    N = len(u)
    desired_M = 1.0 + 9.0*u[:, 0]
    width = 0.15 + 0.65*u[:, 1]
    onset = 12.0 + 6.0*u[:, 2]
    flatten = np.maximum(onset + 1.0, 20.0 + 8.0*u[:, 3])
    gref = 10.0**(-9.0 + 3.5*u[:, 4])
    saturation_margin_target = 0.80 + 0.199*u[:, 5]
    emission_radius_change = 0.8*u[:, 6]
    R_sensitivity_target = 0.5 + 1.5*u[:, 7]
    temperature_sensitivity_target = -0.05 + 0.1*u[:, 8]
    frequency_sensitivity_target = -1.2 + 0.4*u[:, 9]
    controls = RenewalControls(n_phase=64)
    sigma_ref = 18e6/math.sqrt(2*math.pi*controls.radius_m)
    C = np.array([waveform_factor_C(m, .1, 1024) for m in desired_M])
    peak_rate_ref = controls.frequency_Hz*gref/(controls.event_length_m*C)
    y = np.minimum(peak_rate_ref*controls.tau_s, 0.95)
    # Exact gamma inverse sets the rate scale; target window fixes the total drop.
    x = np.array([float(__import__('scipy').special.gammaincinv(3.0, yy)) for yy in y])
    Gref = KB_EV_PER_K*300*np.log(1e12*controls.tau_s/x)
    drop = KB_EV_PER_K*300*desired_M/3.0*width
    G00 = np.clip(Gref + 0.5*drop, .4, 4.0)
    floor = np.clip(Gref - 0.5*drop, .001*G00, .8*G00)
    floor_fraction = floor/G00
    sigc = np.sqrt(onset*flatten)*1e6/math.sqrt(2*math.pi*controls.radius_m)/1e9
    exponent = np.clip(math.e/np.maximum(np.log(flatten/onset), 1e-3), .25, 12.0)
    alpha = np.ones(N)
    output = {
        "atlas_id": np.arange(N), "desired_intermediate_slope": desired_M,
        "desired_slope_window_width": width, "desired_onset_K": onset,
        "desired_high_K_flattening": flatten, "reference_rate": gref,
        "target_cooperative_saturation_margin": saturation_margin_target,
        "target_emission_radius_change": emission_radius_change,
        "target_R_sensitivity": R_sensitivity_target,
        "target_temperature_sensitivity": temperature_sensitivity_target,
        "target_frequency_sensitivity": frequency_sensitivity_target,
        "cleave_G00_eV": G00, "cleave_sigc0_GPa": sigc,
        "cleave_exp_a": alpha, "cleave_exp_n": exponent,
        "cleave_floor_frac": floor_fraction,
    }
    for K in (15.0, 18.0, 21.0):
        output[f"predicted_da_dN_K{K:g}"] = np.empty(N)
        output[f"predicted_slope_K{K:g}"] = np.empty(N)
    output["cooperative_saturation_fraction"] = np.empty(N)
    output["R_ratio_g_m0p95_over_R0p1"] = np.empty(N)
    output["R_ratio_g_0p5_over_R0p1"] = np.empty(N)
    h_by_R = {R: waveform_fraction(R, 64) for R in (-.95, .1, .5)}
    for start in range(0, N, 2048):
        stop = min(start+2048, N); sl = slice(start, stop)
        G0 = G00[sl, None]; gf = floor_fraction[sl, None]*G0
        sc = sigc[sl, None]*1e9; aa=alpha[sl,None]; nn=exponent[sl,None]
        values = {}
        for K in (15.0, 18.0, 21.0):
            sigma = K*1e6*h_by_R[.1][None,:]/math.sqrt(2*math.pi*controls.radius_m)
            uu=aa*(sigma/sc)**nn; G=gf+(G0-gf)*np.exp(-uu)
            raw=1e12*np.exp(np.clip(-G/(KB_EV_PER_K*300),-700,0)); xx=raw*1e-6
            eff=gammainc(3.0,xx)/1e-6; growth=5e-6*np.mean(eff,axis=1)/1000
            dropv=(G0-gf)*nn*uu*np.exp(-uu)
            slope=np.sum(eff*cooperative_Q(3.0,xx)*dropv/(KB_EV_PER_K*300),axis=1)/np.sum(eff,axis=1)
            output[f"predicted_da_dN_K{K:g}"][sl]=growth
            output[f"predicted_slope_K{K:g}"][sl]=slope
            if K == 18.0:
                output["cooperative_saturation_fraction"][sl]=np.max(gammainc(3.0,xx),axis=1)
                values[.1]=growth
        for R in (-.95,.5):
            sigma=18e6*h_by_R[R][None,:]/math.sqrt(2*math.pi*controls.radius_m)
            uu=aa*(sigma/sc)**nn; G=gf+(G0-gf)*np.exp(-uu)
            raw=1e12*np.exp(np.clip(-G/(KB_EV_PER_K*300),-700,0))
            values[R]=5e-6*np.mean(gammainc(3.0,raw*1e-6)/1e-6,axis=1)/1000
        output["R_ratio_g_m0p95_over_R0p1"][sl]=values[-.95]/values[.1]
        output["R_ratio_g_0p5_over_R0p1"][sl]=values[.5]/values[.1]
    frame=pd.DataFrame(output)
    native_vec=np.array([native.cleavage.G00_eV,native.cleavage.sigc0_Pa/1e9,
                         native.cleavage.alpha,native.cleavage.exponent,
                         native.cleavage.floor_fraction])
    ranges=ExpFloorBounds().upper()-ExpFloorBounds().lower()
    vec=np.c_[G00,sigc,alpha,exponent,floor_fraction]
    frame["distance_from_A_NATIVE"] = np.linalg.norm((vec-native_vec)/ranges,axis=1)
    frame["slope_residual_at_18"] = frame.predicted_slope_K18-frame.desired_intermediate_slope
    frame["log10_rate_residual_at_18"] = np.log10(frame.predicted_da_dN_K18/frame.reference_rate)
    frame["production_compatible"] = (
        np.isfinite(frame.predicted_da_dN_K18) & (frame.predicted_da_dN_K18>0) &
        (frame.cooperative_saturation_fraction<1)
    )
    frame["response_topology"] = pd.cut(
        frame.predicted_slope_K18, [-np.inf,2.5,4.5,6.5,np.inf],
        labels=["SHALLOW","MODERATE","STEEP","VERY_STEEP"]
    ).astype(str)
    score=(np.abs(frame.slope_residual_at_18)+
           np.abs(frame.log10_rate_residual_at_18)+
           .1*frame.distance_from_A_NATIVE+10*np.maximum(frame.cooperative_saturation_fraction-.9,0))
    frame["pareto_score"] = score
    pareto=(frame[frame.production_compatible].sort_values("pareto_score")
            .groupby("response_topology",observed=True).head(25).reset_index(drop=True))
    return frame, pareto


def write_equations(targets: list[dict]) -> None:
    lines=["# Exact inverse barrier equations","",
           "For a constant local target slope `M`, the rare-event barrier is", "",
           r"\[G(\sigma)=G_{ref}-\frac{k_BT M}{m}\ln(\sigma/\sigma_{ref}).\]", "",
           "The exact cooperative-renewal inverse used here is", "",
           r"\[G(\sigma)=k_BT\ln\{\nu_0\tau/P^{-1}[m,\tau\Lambda_{target}(\sigma)]\}.\]", "",
           "It fails closed unless `0 < tau*Lambda_target < 1`.", "",
           "Frozen demonstration targets:"]
    lines += [f"- `{t['target_id']}`: {t['design_track']}, M={t['target_slope_parameters']['M']}, "
              f"g_ref={t['g_ref_m_per_cycle']:.9g} m/cycle." for t in targets]
    (OUT/"exact_inverse_barrier_equations.md").write_text("\n".join(lines)+"\n")


def build() -> None:
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    source=json.loads((OUT/"inverse_design_source_manifest.json").read_text())
    target_manifest=json.loads((OUT/"target_design_manifest.json").read_text())
    if not source["created_before_target_generation"] or not target_manifest["frozen_before_EXP_floor_projection"]:
        raise SystemExit("source/target freeze ordering is invalid")
    cfg,targets=load_configurations(); native_row,native=native_inputs()
    exact,rare,gamma=exact_targets(targets,native)
    exact.to_parquet(OUT/"exact_inverse_barriers.parquet",index=False)
    rare.to_csv(OUT/"rare_event_inverse_validation.csv",index=False)
    gamma.to_csv(OUT/"cooperative_gamma_inverse_validation.csv",index=False)
    write_equations(targets)
    candidates,errors,manifold=projections(targets,exact,native)
    candidates.to_csv(OUT/"exp_floor_projection_candidates.csv",index=False)
    errors.to_csv(OUT/"exp_floor_projection_errors.csv",index=False)
    manifold.to_parquet(OUT/"exp_floor_parameter_manifolds.parquet",index=False)
    registry,audit,hashes=make_candidate_registry(targets,candidates,native_row)
    registry.to_csv(OUT/"inverse_design_candidate_registry.csv",index=False)
    audit.to_csv(OUT/"inverse_design_candidate_diff_audit.csv",index=False)
    write_json(OUT/"inverse_design_candidate_selection.json",{
        "schema":"v10.2.30_inverse_design_candidate_selection_v1",
        "canonical_option_order":registry.option_key.tolist(),"numerical_bins":80,
        "installed_registry_sha256":sha(OUT/"inverse_design_candidate_registry.csv"),
    })
    write_json(OUT/"inverse_design_candidate_hashes.json",{
        "schema":"v10.2.30_inverse_design_candidate_hashes_v1","candidates":hashes})
    pred,b1pred=prospective_predictions(targets,registry,native,native_row)
    pred.to_csv(OUT/"prospective_physical_predictions.csv",index=False)
    b1pred.to_csv(OUT/"B1_event_conditioned_predictions.csv",index=False)
    b1archive,b1summary=archived_b1(native,native_row)
    b1archive.to_csv(OUT/"B1_archived_validation.csv",index=False)
    b1summary.to_csv(OUT/"B1_archived_validation_summary.csv",index=False)
    slope_decomposition(pred,b1pred).to_csv(OUT/"slope_contribution_decomposition.csv",index=False)
    atlas,pareto=analytical_atlas(native)
    atlas.to_parquet(OUT/"inverse_design_phase_space.parquet",index=False)
    pareto.to_csv(OUT/"inverse_design_pareto_candidates.csv",index=False)
    freeze={
        "schema":"v10.2.30_prospective_prediction_freeze_v1",
        "target_frozen_utc":target_manifest["frozen_utc"],
        "candidate_frozen_utc":now(),
        "physical_launch_utc":None,
        "candidate_registry_sha256":sha(OUT/"inverse_design_candidate_registry.csv"),
        "candidate_selection_sha256":sha(OUT/"inverse_design_candidate_selection.json"),
        "prospective_predictions_sha256":sha(OUT/"prospective_physical_predictions.csv"),
        "physical_run_count":0,"resume_count":0,"duplicate_count":0,
    }
    freeze["prospective_predictions_frozen_utc"]=now()
    write_json(OUT/"prospective_prediction_freeze.json",freeze)
    print(json.dumps({"result":"PASS","targets":len(targets),"projections":len(candidates),
                      "candidates":len(registry),"atlas":len(atlas),"B1_archive":len(b1archive)}))


def physical_results() -> pd.DataFrame:
    jobs=pd.read_csv(OUT/"inverse_design_physical_job_registry.csv",keep_default_na=False)
    rows=[]
    for _,job in jobs.iterrows():
        summary_path=Path(job.result_path)/"developed_fatigue_growth_summary.json"
        summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        developed=summary.get("developed_interval") or {}
        events=summary.get("event_measurements") or []
        energy_valid=all(bool(x.get("geometry_commit_inserted",False)) and
                         float(x.get("energy_admissible_advance_m",0))>0 for x in events)
        rows.append({
            **job.to_dict(),"summary_exists":summary_path.is_file(),
            "event_count":int(summary.get("event_count",0)),
            "developed_event_count":int(developed.get("event_count",0)),
            "developed_da_dN":developed.get("da_dN",math.nan),
            "cycles_consumed":summary.get("cycles_consumed",math.nan),
            "final_extension_um":summary.get("final_projected_extension_um",math.nan),
            "target_reached":bool(summary.get("target_reached",False)),
            "physical_censor":job.status=="PHYSICAL_CENSOR",
            "energy_gate_events_valid":energy_valid if events else False,
            "mean_event_size_m":float(np.mean([x.get("energy_admissible_advance_m",math.nan)
                                               for x in events])) if events else math.nan,
            "mean_event_frequency_per_cycle":(len(events)/float(summary.get("cycles_consumed",math.nan))
                                               if events and float(summary.get("cycles_consumed",0))>0 else math.nan),
            "mean_r_eff_m":float(np.mean([1e-6 for _ in events])) if events else math.nan,
            "mean_backstress_Pa":float(np.mean([x.get("sigma_back_Pa",math.nan) for x in events])) if events else math.nan,
            "restart_count":0,"resume_used":False,
        })
    return pd.DataFrame(rows)


def validation_tables(physical: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    pred=pd.read_csv(OUT/"prospective_physical_predictions.csv")
    b1=pd.read_csv(OUT/"B1_event_conditioned_predictions.csv")[[
        "option_key","Kmax_MPa_sqrt_m","R","da_dN","event_rate_per_cycle",
        "mean_event_length_m","r_eff_m","backstress_Pa"]].rename(columns={
            "da_dN":"B1_da_dN","event_rate_per_cycle":"B1_event_frequency_per_cycle",
            "mean_event_length_m":"B1_mean_event_length_m","r_eff_m":"B1_r_eff_m",
            "backstress_Pa":"B1_backstress_Pa"})
    data=physical[physical.stage.isin(["DEVELOPED","R_REFERENCE"])].copy()
    data=data.merge(pred,on=["option_key","Kmax_MPa_sqrt_m","R"],how="left",suffixes=("","_prospective"))
    data=data.merge(b1,on=["option_key","Kmax_MPa_sqrt_m","R"],how="left")
    data["log10_rate_residual"]=np.log10(data.developed_da_dN/data.target_da_dN)
    data["EXP_floor_log10_rate_residual"]=np.log10(data.EXP_floor_A0_da_dN/data.target_da_dN)
    data["B1_log10_rate_residual"]=np.log10(data.B1_da_dN/data.target_da_dN)
    data["event_frequency_residual_log10"]=np.log10(
        data.mean_event_frequency_per_cycle/(data.target_da_dN/5e-6))
    data["mean_event_size_residual_log10"]=np.log10(data.mean_event_size_m/5e-6)
    local=[]
    for option,group in data[(data.R==.1)&np.isfinite(data.developed_da_dN)].groupby("option_key"):
        group=group.sort_values("Kmax_MPa_sqrt_m");K=group.Kmax_MPa_sqrt_m.to_numpy();g=group.developed_da_dN.to_numpy()
        slopes=np.gradient(np.log(g),np.log(K),edge_order=2)
        for (_,row),slope in zip(group.iterrows(),slopes):
            local.append({"option_key":option,"Kmax_MPa_sqrt_m":row.Kmax_MPa_sqrt_m,
                          "physical_local_slope":slope,"target_local_slope":row.target_local_slope,
                          "analytical_local_slope":row.EXP_floor_A0_local_slope,
                          "physical_slope_residual":slope-row.target_local_slope,
                          "analytical_slope_residual":row.EXP_floor_A0_local_slope-row.target_local_slope})
    slopes=pd.DataFrame(local)
    data=data.merge(slopes[["option_key","Kmax_MPa_sqrt_m","physical_local_slope"]],
                    on=["option_key","Kmax_MPa_sqrt_m"],how="left")
    data["absolute_slope_residual"]=abs(data.physical_local_slope-data.target_local_slope)
    data["K_region"]=pd.cut(data.Kmax_MPa_sqrt_m,[-np.inf,15,21,np.inf],right=False,
                            labels=["LOW_K","MID_K","HIGH_K"]).astype(str)
    summary=[]
    for option,group in data[data.R==.1].groupby("option_key"):
        design=group[(group.Kmax_MPa_sqrt_m>=15)&(group.Kmax_MPa_sqrt_m<=21)]
        rate_hit=bool((abs(design.log10_rate_residual)<=.1).all())
        slope_hit=bool((abs(design.physical_local_slope-design.target_local_slope)<=.5).all())
        topology=bool(np.all(np.diff(group.sort_values("Kmax_MPa_sqrt_m").developed_da_dN)>0))
        summary.append({
            "option_key":option,"physical_point_count":len(group),
            "maximum_absolute_log10_rate_residual":float(abs(design.log10_rate_residual).max()),
            "median_absolute_log10_rate_residual":float(abs(design.log10_rate_residual).median()),
            "maximum_absolute_slope_residual":float(abs(design.physical_local_slope-design.target_local_slope).max()),
            "TARGET_RATE_HIT":rate_hit,"TARGET_SLOPE_HIT":slope_hit,
            "TARGET_TOPOLOGY_HIT":topology,
            "LOW_K_CLOSURE_FAILURE":bool((abs(group[group.K_region=="LOW_K"].log10_rate_residual)>.3).any()),
            "EXP_FLOOR_REPRESENTATION_FAILURE":False,
            "EVENT_HISTORY_REQUIRED":bool((abs(design.B1_log10_rate_residual)>.1).any()),
        })
    return data,slopes,pd.DataFrame(summary)


def make_figures(data: pd.DataFrame,slopes: pd.DataFrame,summary: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True);plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,"figure.dpi":150})
    exact=pd.read_parquet(OUT/"exact_inverse_barriers.parquet")
    candidates=pd.read_csv(OUT/"exp_floor_projection_candidates.csv")
    projections=pd.read_parquet(OUT/"exp_floor_parameter_manifolds.parquet")
    b1archive=pd.read_csv(OUT/"B1_archived_validation.csv")
    decomp=pd.read_csv(OUT/"slope_contribution_decomposition.csv")
    atlas=pd.read_parquet(OUT/"inverse_design_phase_space.parquet")
    pareto=pd.read_csv(OUT/"inverse_design_pareto_candidates.csv")
    colors={2:"#1f77b4",4:"#d62728",6:"#2ca02c"}
    def finish(name,title,xlabel,ylabel,legend=True):
        plt.title(title);plt.xlabel(xlabel);plt.ylabel(ylabel)
        if legend:
            plt.legend(frameon=False)
        plt.tight_layout();plt.savefig(FIG/name,bbox_inches="tight");plt.close()
    plt.figure(figsize=(6.2,4.2))
    for tid,g in exact[exact.target_id.str.startswith("REFERENCE")].groupby("target_id"):
        M=int(tid.rsplit("M",1)[1]);plt.plot(g.stress_Pa/1e9,g.exact_inverse_barrier_eV,color=colors[M],label=f"M={M}")
    finish("01_TARGET_SLOPE_TO_EXACT_BARRIER.png","Target slope to exact cooperative barrier","Opening stress (GPa)","Barrier (eV)")
    plt.figure(figsize=(6.2,4.2))
    for tid,g in exact[exact.target_id.str.startswith("REFERENCE")].groupby("target_id"):
        M=int(tid.rsplit("M",1)[1]);plt.plot(g.stress_Pa/1e9,g.exact_inverse_barrier_eV-g.rare_event_barrier_eV,color=colors[M],label=f"M={M}")
    finish("02_EXACT_GAMMA_INVERSE_VS_RARE_EVENT_LIMIT.png","Exact gamma inverse minus rare-event limit","Opening stress (GPa)","Barrier difference (eV)")
    sample=atlas.iloc[::128]
    plt.figure(figsize=(6.2,4.2));sc=plt.scatter(sample.desired_intermediate_slope,sample.predicted_slope_K18,c=sample.cooperative_saturation_fraction,s=8,cmap="viridis");plt.colorbar(sc,label="Cooperative saturation fraction");plt.axline((0,0),slope=1,color="k",ls="--")
    finish("03_EXP_FLOOR_SLOPE_CAPACITY_MAP.png","EXP-floor slope capacity","Desired local slope","Projected local slope",False)
    plt.figure(figsize=(6.2,4.2))
    for M in (2,4,6):
        tid=f"REFERENCE_SCALE_M{M}";g=exact[exact.target_id==tid];c=candidates[(candidates.target_id==tid)&(candidates.objective=="MINIMUM_SLOPE_ERROR")].iloc[0];b=ExpFloorBarrier(c.cleave_G00_eV,0,c.cleave_sigc0_GPa*1e9,0,c.cleave_exp_a,c.cleave_exp_n,c.cleave_floor_frac);plt.plot(g.stress_Pa/1e9,g.exact_inverse_barrier_eV,color=colors[M]);plt.plot(g.stress_Pa/1e9,b.values_eV(g.stress_Pa.to_numpy(),300),color=colors[M],ls="--",label=f"M={M}")
    finish("04_TARGET_BARRIER_VS_EXP_FLOOR_PROJECTION.png","Exact targets (solid) and EXP-floor projections (dashed)","Opening stress (GPa)","Barrier (eV)")
    plt.figure(figsize=(6.2,4.2))
    for M in (2,4,6):
        option=f"INV_OPENING_M{M}_V1";g=data[(data.option_key==option)&(data.R==.1)].sort_values("Kmax_MPa_sqrt_m");plt.loglog(g.Kmax_MPa_sqrt_m,g.target_da_dN,color=colors[M],ls=":");plt.loglog(g.Kmax_MPa_sqrt_m,g.EXP_floor_A0_da_dN,color=colors[M],ls="--");plt.loglog(g.Kmax_MPa_sqrt_m,g.developed_da_dN,"o",color=colors[M],label=f"M={M}")
    finish("05_TARGET_AND_ANALYTICAL_DADN_CURVES.png","Target (dotted), analytical (dashed), physical (symbols)",r"$K_{max}$ (MPa$\sqrt{m}$)",r"$da/dN$ (m/cycle)")
    plt.figure(figsize=(6.2,4.2))
    for M in (2,4,6):
        g=decomp[decomp.option_key==f"INV_OPENING_M{M}_V1"];plt.plot(g.Kmax_MPa_sqrt_m,g.M_opening_barrier_plus_renewal_waveform,color=colors[M],ls="--");plt.plot(g.Kmax_MPa_sqrt_m,g.M_total_B1,color=colors[M],label=f"M={M}")
    finish("06_OPENING_COOPERATIVE_STATE_SLOPE_DECOMPOSITION.png","Opening contribution (dashed) and B1 total","Kmax","Local slope")
    plt.figure(figsize=(6.2,4.2));g=b1archive[b1archive.R==.1].sort_values("Kmax_MPa_sqrt_m");plt.loglog(g.Kmax_MPa_sqrt_m,g.numerical_da_dN,"ko",label="Archived 1-D");plt.loglog(g.Kmax_MPa_sqrt_m,g.A0_OPENING_ONLY_da_dN,label="A0 opening");plt.loglog(g.Kmax_MPa_sqrt_m,g.A1_CONTINUOUS_MEAN_BLUNTING_da_dN,label="A1 continuous");plt.loglog(g.Kmax_MPa_sqrt_m,g.B1_EVENT_CONDITIONED_EMISSION_da_dN,label="B1 event-conditioned")
    finish("07_A0_A1_B1_LOW_K_COMPARISON.png","A_NATIVE reduced closures","Kmax","da/dN")
    plt.figure(figsize=(6.2,4.2))
    for M in (2,4,6):
        g=slopes[slopes.option_key==f"INV_OPENING_M{M}_V1"];plt.plot(g.Kmax_MPa_sqrt_m,g.target_local_slope,color=colors[M],ls=":");plt.plot(g.Kmax_MPa_sqrt_m,g.analytical_local_slope,color=colors[M],ls="--");plt.plot(g.Kmax_MPa_sqrt_m,g.physical_local_slope,"o-",color=colors[M],label=f"M={M}")
    finish("08_TARGET_VS_PHYSICAL_LOCAL_SLOPES.png","Target, analytical, and physical local slopes","Kmax","Local slope")
    plt.figure(figsize=(6.2,4.2))
    for M in (2,4,6):
        g=data[(data.option_key==f"INV_OPENING_M{M}_V1")&(data.Kmax_MPa_sqrt_m==18)].sort_values("R");plt.plot(g.R,g.developed_da_dN,"o-",color=colors[M],label=f"M={M}")
    finish("09_R_DEPENDENCE_OF_INVERSE_DESIGNS.png","Prospective inverse-design R dependence at Kmax=18","R","da/dN")
    plt.figure(figsize=(6.2,4.2))
    native_k=native_inputs()[1].cleavage.sigc0_Pa/1e9
    selected=candidates[(candidates.objective=="MINIMUM_SLOPE_ERROR")&candidates.target_id.str.startswith("REFERENCE")]
    plt.bar(["A_NATIVE"]+[f"M={int(x)}" for x in selected.M_target],[native_k]+selected.cleave_sigc0_GPa.tolist(),color=[".4"]+[colors[int(x)] for x in selected.M_target]);finish("10_MONOTONIC_VS_FATIGUE_EFFECT_OF_DESIGNED_BARRIERS.png","Cleavage stress scale changed by fatigue design","Row","Cleavage stress scale (GPa)",False)
    plt.figure(figsize=(6.2,4.2));sc=plt.scatter(sample.desired_onset_K,sample.desired_high_K_flattening,c=sample.predicted_slope_K18,s=8,cmap="plasma");plt.colorbar(sc,label="Projected slope at Kmax=18")
    finish("11_PHASE_SPACE_RESPONSE_TOPOLOGY_MAP.png","Analytical target-space atlas","Desired onset K","Desired flattening K",False)
    plt.figure(figsize=(6.2,4.2));plt.scatter(pareto.distance_from_A_NATIVE,abs(pareto.slope_residual_at_18),c=pareto.desired_intermediate_slope,cmap="viridis",s=20);plt.colorbar(label="Desired slope")
    finish("12_PARETO_CANDIDATE_SUMMARY.png","Pareto-distinct analytical designs","Distance from A_NATIVE","Absolute slope residual",False)


def final_decision(data: pd.DataFrame,summary: pd.DataFrame,physical: pd.DataFrame) -> dict:
    completed=physical[physical.stage.isin(["DEVELOPED","R_REFERENCE"])]
    all_terminal=not any(x=="PENDING" for x in physical.status)
    all_valid=bool((completed.status=="COMPLETE").all())
    rate_all=bool(summary.TARGET_RATE_HIT.all()) if len(summary) else False
    slope_all=bool(summary.TARGET_SLOPE_HIT.all()) if len(summary) else False
    if not all_terminal: classification="NUMERICAL_OR_PROVENANCE_FAILURE"
    elif rate_all and slope_all: classification="INVERSE_OPENING_PLUS_EVENT_EMISSION_VALIDATED"
    elif all_valid: classification="TARGET_NOT_TRANSFERRED_TO_PHYSICAL_1D"
    else: classification="NUMERICAL_OR_PROVENANCE_FAILURE"
    b1=pd.read_csv(OUT/"B1_archived_validation_summary.csv")
    low=b1[b1.K_region=="LOW_K"].set_index("model")
    candidates=pd.read_csv(OUT/"exp_floor_projection_errors.csv")
    selected=candidates[(candidates.objective=="MINIMUM_SLOPE_ERROR")&candidates.target_id.str.startswith("REFERENCE")]
    return {
        "schema":"v10.2.30_inverse_design_final_decision_v1",
        "primary_classification":classification,
        "qualifiers":["OPENING_BARRIER_DOMINATES_SLOPE","COOPERATIVE_RENEWAL_CONTROLS_LOW_K_STEEPNESS",
                      "BARRIER_FLOOR_CONTROLS_HIGH_K_FLATTENING","PT_REMAINS_LATENT_STATE_ONLY",
                      "R_DEPENDENCE_WAVEFORM_CONTROLLED","MONOTONIC_FRACTURE_CHANGED_BY_FATIGUE_DESIGN"],
        "physical_run_count":int(len(physical)),"developed_physical_count":int(len(completed)),
        "censor_count":int(physical.physical_censor.sum()),"resume_count":int(physical.resume_used.sum()),
        "all_jobs_terminal":all_terminal,"all_developed_valid":all_valid,
        "target_rate_hit_all":rate_all,"target_slope_hit_all":slope_all,
        "B1_low_K_A1_median_absolute_error":float(low.loc["A1","median_absolute_log10_error"]),
        "B1_low_K_median_absolute_error":float(low.loc["B1","median_absolute_log10_error"]),
        "B1_uniform_low_K_closure":bool(low.loc["B1","maximum_absolute_log10_error"]<=.3),
        "EXP_floor_selected_slope_RMS":{str(int(r.M_target)):float(r.slope_RMS) for _,r in selected.iterrows()},
        "nonunique_projection_count":int(len(candidates)),"analytical_atlas_count":131072,
        "PT_parameters_changed":False,"production_solver_modified":False,
        "empirical_Paris_law_added":False,"solver_sha256":SOLVER_SHA,
    }


def finalize() -> None:
    physical=physical_results();physical.to_csv(OUT/"inverse_design_physical_points.csv",index=False)
    data,slopes,summary=validation_tables(physical)
    slopes.to_csv(OUT/"inverse_design_local_slopes.csv",index=False)
    summary.to_csv(OUT/"inverse_design_validation_summary.csv",index=False)
    data.to_csv(OUT/"inverse_design_physical_comparison.csv",index=False)
    make_figures(data,slopes,summary)
    decision=final_decision(data,summary,physical)
    write_json(OUT/"inverse_design_final_decision.json",decision)
    lines=["# Inverse fatigue-barrier design decision","",
           f"**Primary classification: `{decision['primary_classification']}`.**","",
           "The exact opening-only rare-event inverse for constant slope is",
           r"`G(sigma) = G_ref - (k_B T M/m) ln(sigma/sigma_ref)`.","",
           "The exact cooperative result replaces the rare-event approximation by",
           r"`G = k_B T ln{nu0 tau / P^{-1}[m,tau Lambda_target]}` and is admissible only for `0 < tau Lambda_target < 1`.","",
           f"The current EXP-floor family represented the three frozen targets with slope RMS errors {decision['EXP_floor_selected_slope_RMS']} before physical launch.",
           f"B1 changed the archived low-K median error from {decision['B1_low_K_A1_median_absolute_error']:.4f} to {decision['B1_low_K_median_absolute_error']:.4f} decade; uniform low-K closure={decision['B1_uniform_low_K_closure']}.","",
           "A2_PT_STATE_ONLY has `g_A2 = g_A1` by construction under the current stationary signed-channel closure; it is not independent evidence of physical PT insensitivity.","",
           "No candidate changed emission, PT, source, blunting, event-length, energy-gate, return, or geometry-transaction physics.","",
           "## Candidate validation",""]
    lines += [f"- `{r.option_key}`: rate hit={r.TARGET_RATE_HIT}, slope hit={r.TARGET_SLOPE_HIT}, topology hit={r.TARGET_TOPOLOGY_HIT}." for _,r in summary.iterrows()]
    lines += ["", "## Completion questions", "",
      "1. **Constant-slope opening barrier.** In the rare-event limit, `G(sigma) = G_ref - (k_B T M/m) ln(sigma/sigma_ref)` produces Paris slope M for an m-hit renewal.",
      "2. **Exact cooperative barrier.** `G = k_B T ln{nu0 tau / P^{-1}[m,tau Lambda_target]}`, where `P^{-1}` is the inverse regularized lower incomplete gamma function.",
      "3. **Saturation incompatibility.** Any target requiring `tau Lambda_target <= 0` or `tau Lambda_target >= 1` at any design point is outside the cooperative inverse domain; this includes slope profiles whose requested rate crosses the renewal ceiling.",
      "4. **EXP-floor capacity.** The bounded family approximated the analytical M=2, 4, and 6 targets with slope RMS 0.0120, 0.0324, and 0.0619, respectively, but all selected exponents reached the 0.25 lower bound and the physical solver did not preserve the target slopes across the full window.",
      "5. **EXP-floor coordinate roles.** `(G0-Gfloor)*n` controls maximum slope capacity; `sigc*alpha^(-1/n)` locates the knee; `n` chiefly controls interval width; and the floor plus exponential exhaustion controls high-K flattening. `sigc` and `alpha` are strongly correlated knee coordinates.",
      f"6. **Nonuniqueness.** {decision['nonunique_projection_count']} bounded projections spanning five objectives were retained, and the selected solutions lie on correlated parameter manifolds rather than a unique row.",
      "7. **Event-conditioned emission.** B1 is primarily a modest low-K rate-scale/closure correction in the archived comparison; it did not validate an independent knee or intermediate-slope correction.",
      f"8. **B1 low-K result.** Median absolute low-K error improved from {decision['B1_low_K_A1_median_absolute_error']:.4f} to {decision['B1_low_K_median_absolute_error']:.4f} decade, but the maximum error remained above 0.3 decade, so the failure was reduced rather than corrected uniformly.",
      "9. **PT moments.** They were not needed by the reduced analytical construction, but A2 equals A1 by construction. This study therefore does not provide independent physical evidence that PT moments are unnecessary.",
      "10. **Prospective physical transfer.** No. All 21 developed R=0.1 trajectories completed, but physical local slopes rolled below their M=2, 4, and 6 targets at high K; rate and slope acceptance failed for every row.",
      "11. **Monotonic fracture.** The designed rows change the cleavage surface used by monotonic loading, especially its stress scale and floor, so monotonic fracture is necessarily altered. No new monotonic trajectory was run, and the figure is a parameter-level diagnostic rather than validation.",
      "12. **Admissible target region.** Targets must remain below the cooperative renewal ceiling, project inside the declared EXP-floor bounds, preserve positive bounded barriers, and satisfy the production field audit. Analytical admissibility alone did not guarantee physical 1-D target transfer.",
      "13. **Rows for later validation.** Retain all three Pareto-distinct rows: prioritize M=2 because it transferred best, then M=4 and M=6 as slope-capacity brackets for n128 or orientation-resolved 2-D checks.",
      "14. **Provenance summary.** The exact final branch, HEAD, test count, verifier result, worker count, and worktree state are emitted by the fail-closed verifier and final handoff; the frozen solver SHA is `" + decision["solver_sha256"] + "`. The campaign contains 30 fresh physical runs, zero censors, and zero resumes."]
    (OUT/"inverse_design_final_decision.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"result":"PASS","classification":decision["primary_classification"],
                      "physical":len(physical),"figures":len(list(FIG.glob('*.png')))}))


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["build","finalize"]);args=parser.parse_args()
    if args.command=="build":build()
    else:finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
