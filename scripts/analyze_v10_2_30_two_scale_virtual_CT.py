#!/usr/bin/env python3
"""Analyze local anchors, held-out dynamics, and virtual C(T) protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arrhenius_fracture.two_scale_virtual_ct_v10230 import (
    LogPchipRateSurface,
    fixed_load_kmax_MPa,
    integrate_virtual_ct,
)
from arrhenius_fracture.virtual_ct_v10230 import (
    CompactTensionGeometry,
    ct_geometry_factor,
    ct_load_from_k_pa_sqrt_m,
)

SOURCE = Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1")
NATIVE = "A_NATIVE"
PT03 = "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5"
PT08 = "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"
LABEL = {NATIVE: "A_NATIVE", PT03: "PT03", PT08: "PT08"}
RS = (-0.95, 0.1, 0.5)
PROTOCOLS = ("FIXED_LOAD", "LOAD_SHEDDING", "CONSTANT_KMAX")
GEOMETRIES = {
    "W10_B2.5": CompactTensionGeometry(0.010, 0.0025, 0.0045),
    "W25_B6.25": CompactTensionGeometry(0.025, 0.00625, 0.01125),
}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(option: str) -> list[dict]:
    frame = pd.read_csv(SOURCE / "A_PT03_PT08_R_developed_points.csv")
    frame = frame[(frame.option == option) & frame.stage.str.contains("PRIMARY") & (frame.seed == 1720)]
    frame = frame[frame.Kmax_MPa_sqrt_m.isin((12.0, 15.0, 18.0, 24.0))]
    if len(frame) != 12 or not frame.target_reached.all() or not frame.stable_growth.all() or frame.resumed.any():
        raise RuntimeError(f"inadmissible frozen local rows for {option}")
    return frame.sort_values(["R", "Kmax_MPa_sqrt_m"]).to_dict("records")


def collect_anchors(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(root / "two_scale_anchor_job_registry.csv")
    predictions = pd.read_csv(root / "A_NATIVE_anchor_predictions_v0.csv")
    prediction_sha = sha256(root / "A_NATIVE_anchor_predictions_v0.csv")
    results, validation = [], []
    source = pd.DataFrame(source_rows(NATIVE))
    for job in registry.itertuples():
        if job.status != "PHYSICAL_TARGET_REACHED" or bool(job.resumed) or bool(job.reused):
            raise RuntimeError(f"anchor not fresh and terminal: {job.job_id}")
        contract = json.loads(Path(job.contract_path).read_text())
        result = Path(job.result_path)
        summary_path = result / "developed_fatigue_growth_summary.json"
        summary = json.loads(summary_path.read_text())
        developed = summary.get("developed_interval") or {}
        if contract["prediction_file_sha256"] != prediction_sha:
            raise RuntimeError("anchor contract does not bind frozen predictions")
        if int(contract["created_unix_ns"]) <= int(contract["prediction_frozen_unix_ns"]):
            raise RuntimeError("anchor contract timestamp is not after prediction freeze")
        if summary_path.stat().st_mtime_ns <= int(contract["prediction_frozen_unix_ns"]):
            raise RuntimeError("anchor result predates prospective prediction")
        row = {
            "job_id": job.job_id,
            "option": NATIVE,
            "R": float(job.R),
            "Kmax_MPa_sqrt_m": float(job.Kmax_MPa_sqrt_m),
            "developed_da_dN": float(developed["da_dN"]),
            "event_count": int(summary["event_count"]),
            "cycles": float(summary["cycles_consumed"]),
            "final_extension_um": float(summary["final_projected_extension_um"]),
            "target_reached": bool(summary["target_reached"]),
            "stable_growth": bool(summary["stable_growth_provisional"]),
            "stationarity_ratio": float(summary["late_to_early_rate_ratio"]),
            "terminal_classification": job.status,
            "result_path": job.result_path,
            "seed": int(job.seed),
            "n_bins": int(job.n_bins),
            "temperature_K": float(job.temperature_K),
            "frequency_Hz": float(job.frequency_Hz),
            "launch_head": contract["launch_head"],
            "qualified_solver_head": contract["qualified_solver_head"],
            "production_solver_sha256": contract["production_solver_sha256"],
            "fresh_virgin_start": contract["fresh_virgin_start"],
            "resumed": contract["resume"],
            "prediction_file_sha256": prediction_sha,
            "prediction_frozen_unix_ns": int(contract["prediction_frozen_unix_ns"]),
            "result_summary_mtime_ns": summary_path.stat().st_mtime_ns,
        }
        results.append(row)
        prediction = predictions[
            np.isclose(predictions.R, float(job.R))
            & np.isclose(predictions.Kmax_MPa_sqrt_m, float(job.Kmax_MPa_sqrt_m))
        ].iloc[0]
        epsilon = math.log10(float(prediction.predicted_da_dN) / row["developed_da_dN"])
        Rsource = source[np.isclose(source.R, float(job.R))].sort_values("Kmax_MPa_sqrt_m")
        combined_K = np.r_[Rsource.Kmax_MPa_sqrt_m.to_numpy(float), row["Kmax_MPa_sqrt_m"]]
        combined_g = np.r_[Rsource.developed_da_dN.to_numpy(float), row["developed_da_dN"]]
        order = np.argsort(combined_K)
        monotonic = bool(np.all(np.diff(combined_g[order]) >= 0))
        abs_error = abs(epsilon)
        classification = (
            "HIGH_ACCURACY" if abs_error <= 0.025 else
            "ACCEPTABLE_WITH_REFINEMENT" if abs_error <= 0.05 else
            "INTERPOLATION_FAILURE"
        )
        validation.append(
            row
            | {
                "predicted_da_dN_v0": float(prediction.predicted_da_dN),
                "predicted_local_slope_v0": float(prediction.predicted_local_slope),
                "epsilon_log_decade": epsilon,
                "abs_epsilon_log_decade": abs_error,
                "relative_rate_error": float(prediction.predicted_da_dN) / row["developed_da_dN"] - 1,
                "classification": classification,
                "monotonic_with_neighbors": monotonic,
                "mechanism_transition_detected": False,
                "admissible": bool(row["target_reached"] and row["stable_growth"] and monotonic),
            }
        )
    results_frame, validation_frame = pd.DataFrame(results), pd.DataFrame(validation)
    results_frame.to_csv(root / "A_NATIVE_anchor_results.csv", index=False)
    validation_frame.to_csv(root / "A_NATIVE_anchor_validation.csv", index=False)
    return results_frame, validation_frame


def build_surfaces(root: Path, anchors: pd.DataFrame) -> tuple[LogPchipRateSurface, LogPchipRateSurface, dict[str, LogPchipRateSurface]]:
    native = source_rows(NATIVE)
    v0 = LogPchipRateSurface(native, option=NATIVE, version="v0")
    v1rows = native + anchors[["R", "Kmax_MPa_sqrt_m", "developed_da_dN"]].to_dict("records")
    v1 = LogPchipRateSurface(v1rows, option=NATIVE, version="v1")
    payload = {
        "schema": "log_log_PCHIP_local_rate_surface_v1",
        "option": NATIVE,
        "version": "v1",
        "interpolation": "PCHIP_IN_LN_K_AND_LN_RATE_PER_MEASURED_R;LINEAR_IN_LN_RATE_BETWEEN_R",
        "global_Paris_law": False,
        "extrapolation": False,
        "K_domain_MPa_sqrt_m": [v1.K_min, v1.K_max],
        "R_domain": [v1.R_min, v1.R_max],
        "source_rows": v1.nodes(),
        "held_out_constant_load_fit_rows": 0,
    }
    atomic_json(root / "A_NATIVE_rate_surface_v1.json", payload)
    pd.DataFrame(v1.nodes()).to_parquet(root / "A_NATIVE_rate_surface_v1.parquet", index=False)
    comparison = []
    for R in RS:
        for K in np.linspace(12, 24, 241):
            g0 = float(v0.evaluate(K, R).rate_m_per_cycle)
            g1 = float(v1.evaluate(K, R).rate_m_per_cycle)
            comparison.append({"R": R, "Kmax_MPa_sqrt_m": K, "g_v0": g0, "g_v1": g1,
                               "log10_v1_over_v0": math.log10(g1 / g0)})
    atomic_csv(root / "A_NATIVE_surface_version_comparison.csv", comparison)
    overlays = {}
    for option, filename in ((PT03, "PT03_rate_surface.json"), (PT08, "PT08_rate_surface.json")):
        surface = LogPchipRateSurface(source_rows(option), option=option, version="v0_overlay")
        overlays[option] = surface
        atomic_json(root / filename, {"schema": "log_log_PCHIP_local_rate_surface_v1", "option": option,
          "role": "DIAGNOSTIC_MECHANISTIC_OVERLAY", "version": "v0_overlay", "extrapolation": False,
          "K_domain_MPa_sqrt_m": [surface.K_min, surface.K_max], "R_domain": [surface.R_min, surface.R_max],
          "source_rows": surface.nodes(), "held_out_constant_load_fit_rows": 0})
    ratio_rows = []
    for option, surface in overlays.items():
        for R in RS:
            for K in np.linspace(12, 24, 241):
                native_rate = float(v0.evaluate(K, R).rate_m_per_cycle)
                overlay_rate = float(surface.evaluate(K, R).rate_m_per_cycle)
                ratio_rows.append({"option": option, "R": R, "Kmax_MPa_sqrt_m": K,
                                  "rate_ratio_to_A_NATIVE_v0": overlay_rate / native_rate,
                                  "log10_rate_ratio": math.log10(overlay_rate / native_rate),
                                  "diagnostic_overlay_only": True})
    atomic_csv(root / "PT_overlay_surface_ratios.csv", ratio_rows)
    return v0, v1, overlays


def dynamic_validation(root: Path, surface: LogPchipRateSurface) -> tuple[pd.DataFrame, dict]:
    events = pd.read_csv(SOURCE / "A_PT03_PT08_R_event_results.csv")
    events = events[events.stage == "CONSTANT_LOAD_CT"].copy()
    rows, residual_rows = [], []
    for (job_id, option, R), group in events.groupby(["job_id", "option", "R"]):
        group = group.sort_values("cycles_post")
        W = float(group.W_m.iloc[0])
        a0 = 0.5 * W
        K0 = float(group.Kmax_MPa_sqrt_m.iloc[0])
        f0 = ct_geometry_factor(0.5)

        def rate(extension):
            K = K0 * ct_geometry_factor((a0 + extension) / W) / f0
            return float(surface.evaluate(K, float(R)).rate_m_per_cycle)

        final_extension = float(group.projected_extension_post_m.iloc[-1])
        predicted_cycles = quad(lambda extension: 1 / rate(extension), 0, final_extension, epsrel=1e-9)[0]
        physical_cycles = float(group.cycles_post.iloc[-1])
        for event in group.itertuples():
            extension = float(event.projected_extension_post_m)
            prediction = quad(lambda value: 1 / rate(value), 0, extension, epsrel=1e-9)[0]
            residual_rows.append({"job_id": job_id, "option": option, "R": float(R),
              "projected_extension_m": extension, "physical_cycles": float(event.cycles_post),
              "predicted_cycles": prediction, "cycle_residual": prediction-float(event.cycles_post),
              "relative_cycle_residual": (prediction-float(event.cycles_post))/max(physical_cycles,1e-300)})
        residual = pd.DataFrame([x for x in residual_rows if x["job_id"] == job_id])
        rho = float(spearmanr(residual.projected_extension_m, residual.relative_cycle_residual).statistic)
        epsilon = math.log10(predicted_cycles / physical_cycles)
        span = float(residual.relative_cycle_residual.max() - residual.relative_cycle_residual.min())
        weak_trend = bool(abs(rho) >= 0.8 and span > 0.01)
        classification = (
            "STATE_HISTORY_REQUIRED" if abs(epsilon) > 0.05 else
            "QUASI_STEADY_DYNAMIC_WARNING" if weak_trend else
            "QUASI_STEADY_DYNAMIC_PASS"
        )
        rows.append({"job_id": job_id, "option": option, "R": float(R), "held_out": True,
          "used_for_surface_fit": False, "initial_Kmax_MPa_sqrt_m": K0, "final_extension_m": final_extension,
          "physical_cycles": physical_cycles, "predicted_cycles": predicted_cycles,
          "epsilon_log_cycles": epsilon, "physical_mean_da_dN": final_extension/physical_cycles,
          "predicted_mean_da_dN": final_extension/predicted_cycles, "residual_spearman_rho": rho,
          "normalized_residual_span": span, "weak_systematic_trend": weak_trend,
          "classification": classification, "result_path": group.result_path.iloc[0]})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "constant_load_dynamic_validation.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(root / "constant_load_dynamic_residuals.csv", index=False)
    native = frame[frame.option == NATIVE]
    overall = (
        "STATE_HISTORY_REQUIRED" if (native.classification == "STATE_HISTORY_REQUIRED").any() else
        "QUASI_STEADY_DYNAMIC_WARNING" if (native.classification == "QUASI_STEADY_DYNAMIC_WARNING").any() else
        "QUASI_STEADY_DYNAMIC_PASS"
    )
    payload = {"schema": "constant_load_dynamic_validation_v1", "classification": overall,
      "held_out_trajectory_count": len(frame), "native_trajectory_count": len(native),
      "maximum_native_abs_log_cycle_error": float(native.epsilon_log_cycles.abs().max()),
      "constant_load_rows_used_for_fit": 0, "records": frame.to_dict("records")}
    atomic_json(root / "constant_load_dynamic_validation.json", payload)
    # Correlation is diagnostic only; no empirical state correction is fitted.
    states_path = SOURCE / "A_PT03_PT08_R_state_histories.parquet"
    diagnostics = []
    if states_path.is_file():
        states = pd.read_parquet(states_path)
        residuals = pd.DataFrame(residual_rows)
        for job_id, residual in residuals.groupby("job_id"):
            state = states[states.job_id == job_id].sort_values("record_index").head(len(residual))
            residual = residual.sort_values("projected_extension_m").head(len(state))
            for column in ("retained_count", "mobile_count", "internal_stress_Pa", "shielding_Pa_sqrt_m",
                           "tip_radius_m", "net_source_linked_blunting_m", "physical_return"):
                values = pd.to_numeric(state[column], errors="coerce").to_numpy(float)
                target = residual.relative_cycle_residual.to_numpy(float)
                mask = np.isfinite(values) & np.isfinite(target)
                rho = float(spearmanr(values[mask], target[mask]).statistic) if mask.sum() >= 3 and np.ptp(values[mask]) else math.nan
                diagnostics.append({"job_id": job_id, "state_variable": column,
                                    "residual_spearman_rho": rho, "sample_count": int(mask.sum()),
                                    "empirical_state_law_fitted": False})
    atomic_csv(root / "state_history_residual_diagnostics.csv", diagnostics)
    return frame, payload


def integrate_native(root: Path, surface: LogPchipRateSurface, dynamic_classification: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows, summaries, controls = [], [], []
    for geometry_name, geometry in GEOMETRIES.items():
        for R in RS:
            for protocol in PROTOCOLS:
                rows, metadata = integrate_virtual_ct(surface, R=R, protocol=protocol, geometry=geometry,
                                                       relative_tolerance=1e-9, output_points=401)
                for row in rows:
                    row.update({"geometry": geometry_name, "option": NATIVE, "role": "PRIMARY"})
                curve_rows.extend(rows)
                frame = pd.DataFrame(rows)
                life_low = np.trapezoid(
                    np.where(frame.Kmax_MPa_sqrt_m <= 15, 1/frame.local_da_dN, 0), frame.a_m
                ) / metadata["total_cycles"]
                extension_flat = float((frame.Kmax_MPa_sqrt_m >= 18).sum() / len(frame))
                summaries.append({"geometry": geometry_name, "option": NATIVE, "R": R, "protocol": protocol,
                  "total_extension_m": geometry.width_m*0.2, "total_cycles": metadata["total_cycles"],
                  "elapsed_seconds_1000Hz": metadata["total_cycles"]/1000,
                  "initial_Kmax_MPa_sqrt_m": frame.Kmax_MPa_sqrt_m.iloc[0], "final_Kmax_MPa_sqrt_m": frame.Kmax_MPa_sqrt_m.iloc[-1],
                  "initial_deltaK_full": frame.deltaK_full_MPa_sqrt_m.iloc[0], "final_deltaK_full": frame.deltaK_full_MPa_sqrt_m.iloc[-1],
                  "initial_deltaK_tensile": frame.deltaK_tensile_MPa_sqrt_m.iloc[0], "final_deltaK_tensile": frame.deltaK_tensile_MPa_sqrt_m.iloc[-1],
                  "initial_Pmax_N": frame.Pmax_N.iloc[0], "final_Pmax_N": frame.Pmax_N.iloc[-1],
                  "initial_Pmin_N": frame.Pmin_N.iloc[0], "final_Pmin_N": frame.Pmin_N.iloc[-1],
                  "minimum_da_dN": frame.local_da_dN.min(), "maximum_da_dN": frame.local_da_dN.max(),
                  "minimum_local_slope": frame.local_effective_slope.min(), "maximum_local_slope": frame.local_effective_slope.max(),
                  "fraction_life_Kmax_le_15": life_low, "fraction_extension_Kmax_ge_18": extension_flat,
                  "surface_version": surface.version, "K_domain_margin_low": metadata["K_domain_margin_low"],
                  "K_domain_margin_high": metadata["K_domain_margin_high"], "integration_convergence": metadata["maximum_relative_convergence_difference"],
                  "dynamic_validation_classification": dynamic_classification, "tip_radius_used": False,
                  "closure_corrected_deltaK_reported": False})
                controls.append({"geometry": geometry_name, "R": R, "protocol": protocol, **metadata})
    curves, summary = pd.DataFrame(curve_rows), pd.DataFrame(summaries)
    curves.to_parquet(root / "virtual_CT_curves.parquet", index=False)
    summary.to_csv(root / "virtual_CT_life_summary.csv", index=False)
    curves[["geometry", "option", "protocol", "R", "a_m", "a_over_W", "Pmax_N", "Pmin_N",
            "Kmax_MPa_sqrt_m", "Kmin_MPa_sqrt_m", "cumulative_cycles"]].to_csv(root / "virtual_CT_load_histories.csv", index=False)
    curves[["geometry", "option", "protocol", "R", "a_over_W", "Kmax_MPa_sqrt_m", "local_da_dN",
            "local_effective_slope"]].to_csv(root / "virtual_CT_local_slopes.csv", index=False)
    atomic_json(root / "virtual_CT_protocols.json", {"schema": "virtual_CT_protocols_v1", "x0": .45, "x1": .65,
      "protocols": {"FIXED_LOAD": "K=12*f_CT(x)/f_CT(0.45)", "LOAD_SHEDDING": "K=24*exp(ln(12/24)*(x-.45)/.2)",
                    "CONSTANT_KMAX": "K=18"}, "geometries": {k:{"W_m":v.width_m,"B_m":v.thickness_m} for k,v in GEOMETRIES.items()},
      "integration_controls": controls, "tip_radius_used": False, "deltaK_effective_reported": False})
    return curves, summary


def seed_sensitivity(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    developed = pd.read_csv(SOURCE / "A_PT03_PT08_R_developed_points.csv")
    second = pd.read_csv(SOURCE / "A_PT03_PT08_R_second_seed_points.csv")
    rows = []
    for R in RS:
        central = developed[(developed.option == NATIVE) & np.isclose(developed.R,R) & np.isclose(developed.Kmax_MPa_sqrt_m,18)].iloc[0]
        alternate = second[(second.option == NATIVE) & np.isclose(second.R,R) & np.isclose(second.Kmax_MPa_sqrt_m,18)].iloc[0]
        factor = float(alternate.developed_da_dN / central.developed_da_dN)
        for record in summary[summary.R == R].itertuples():
            rows.append({"geometry": record.geometry, "R": R, "protocol": record.protocol,
              "sensitivity": "KMAX18_SEED_SENSITIVITY", "central_seed": 1720, "comparison_seed": 1001723,
              "observed_rate_factor": factor, "central_total_cycles": record.total_cycles,
              "uniform_rate_perturbation_total_cycles": record.total_cycles/factor,
              "log10_life_shift": math.log10(1/factor), "probabilistic_confidence_interval": False})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "virtual_CT_seed_sensitivity.csv", index=False)
    return frame


def plot_figures(root: Path, v0: LogPchipRateSurface, v1: LogPchipRateSurface, anchors: pd.DataFrame,
                 validation: pd.DataFrame, dynamic: pd.DataFrame, curves: pd.DataFrame,
                 summary: pd.DataFrame, overlays: dict[str, LogPchipRateSurface]) -> None:
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    colors = {-0.95:"#1f77b4",0.1:"#ff7f0e",0.5:"#2ca02c"}
    source = pd.DataFrame(source_rows(NATIVE))

    def save(fig, name):
        fig.tight_layout(); fig.savefig(destination / f"{name}.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5.5))
    for R in RS:
        K=np.linspace(12,24.3,300); ax.plot(K,v1.evaluate(K,R).rate_m_per_cycle,color=colors[R],label=f"R={R:g} v1")
        q=source[np.isclose(source.R,R)]; ax.scatter(q.Kmax_MPa_sqrt_m,q.developed_da_dN,color=colors[R],marker="o")
        q=anchors[np.isclose(anchors.R,R)]; ax.scatter(q.Kmax_MPa_sqrt_m,q.developed_da_dN,color=colors[R],marker="x",s=55)
    ax.set_yscale("log");ax.set_xlabel("Kmax (MPa sqrt(m))");ax.set_ylabel("da/dN (m/cycle)");ax.legend();ax.grid(alpha=.25)
    save(fig,"LOCAL_RATE_DATA_AND_PCHIP_SURFACES")
    fig,ax=plt.subplots(figsize=(8,5.5))
    for R in RS:
        K=np.linspace(12,24.3,300);ax.plot(K,v1.evaluate(K,R).local_slope,label=f"R={R:g}")
    ax.set_xlabel("Kmax (MPa sqrt(m))");ax.set_ylabel("d ln g / d ln Kmax");ax.legend();ax.grid(alpha=.25)
    save(fig,"LOCAL_EFFECTIVE_SLOPE_VS_KMAX")
    fig,ax=plt.subplots(figsize=(8,5.5)); loo=pd.read_csv(root/"A_NATIVE_rate_surface_v0_validation.csv")
    ax.scatter([f"LOO {r:g}/{k:g}" for r,k in zip(loo.R,loo.held_Kmax_MPa_sqrt_m)],loo.epsilon_log_decade,label="leave-one-out")
    ax.scatter([f"A {r:g}/{k:g}" for r,k in zip(validation.R,validation.Kmax_MPa_sqrt_m)],validation.epsilon_log_decade,label="prospective anchors")
    ax.axhline(.025,color="orange",ls="--");ax.axhline(-.025,color="orange",ls="--");ax.axhline(.05,color="red",ls=":");ax.axhline(-.05,color="red",ls=":");ax.tick_params(axis="x",rotation=70);ax.set_ylabel("log10(predicted/physical)");ax.legend();ax.grid(alpha=.2)
    save(fig,"LEAVE_ONE_OUT_AND_NEW_ANCHOR_VALIDATION")
    fig,ax=plt.subplots(figsize=(6,6));
    for option,g in dynamic.groupby("option"):ax.scatter(g.physical_cycles,g.predicted_cycles,label=LABEL[option])
    limits=[min(dynamic.physical_cycles.min(),dynamic.predicted_cycles.min()),max(dynamic.physical_cycles.max(),dynamic.predicted_cycles.max())];ax.plot(limits,limits,"k--");ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("physical cycles");ax.set_ylabel("predicted cycles");ax.legend();ax.grid(alpha=.25)
    save(fig,"DYNAMIC_CONSTANT_LOAD_PREDICTED_VS_PHYSICAL")
    primary=curves[curves.geometry=="W10_B2.5"]
    mappings = [
      ("VIRTUAL_CT_K_AND_LOAD_VS_A_OVER_W","a_over_W","Kmax_MPa_sqrt_m",False),
      ("VIRTUAL_CT_A_OVER_W_VS_CYCLES","cumulative_cycles","a_over_W",True),
      ("VIRTUAL_CT_DADN_VS_KMAX_BY_R","Kmax_MPa_sqrt_m","local_da_dN",True),
      ("VIRTUAL_CT_DADN_VS_FULL_DELTAK_BY_R","deltaK_full_MPa_sqrt_m","local_da_dN",True),
      ("VIRTUAL_CT_DADN_VS_TENSILE_DELTAK_BY_R","deltaK_tensile_MPa_sqrt_m","local_da_dN",True),
      ("FIXED_LOAD_VS_LOAD_SHEDDING_PATHS","a_over_W","Kmax_MPa_sqrt_m",False),
    ]
    for name,x,y,log in mappings:
        fig,ax=plt.subplots(figsize=(8,5.5))
        for (protocol,R),g in primary.groupby(["protocol","R"]):ax.plot(g[x],g[y],label=f"{protocol}, R={R:g}")
        if log: ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlabel(x);ax.set_ylabel(y);ax.legend(fontsize=7);ax.grid(alpha=.25);save(fig,name)
    ratios=pd.read_csv(root/"PT_overlay_surface_ratios.csv")
    fig,ax=plt.subplots(figsize=(8,5.5))
    for (option,R),g in ratios.groupby(["option","R"]):ax.plot(g.Kmax_MPa_sqrt_m,g.rate_ratio_to_A_NATIVE_v0,label=f"{LABEL[option]}, R={R:g}")
    ax.set_xlabel("Kmax");ax.set_ylabel("g_PT/g_native");ax.legend(fontsize=7);ax.grid(alpha=.25);save(fig,"PT03_PT08_RATE_RATIO_ALONG_CT_PATH")
    # Placeholder is a truthful domain-status visualization until PT fixed-load integration is admitted.
    fig,ax=plt.subplots(figsize=(8,5.5));ax.text(.5,.5,"PT cumulative-life integration pending\nvalidated endpoint-domain resolution",ha="center",va="center");ax.axis("off");save(fig,"PT03_PT08_CUMULATIVE_LIFE_DIFFERENCE")
    fig,ax=plt.subplots(figsize=(8,5.5))
    for (geometry,R),g in summary[summary.protocol=="FIXED_LOAD"].groupby(["geometry","R"]):ax.scatter(R,g.total_cycles.iloc[0],label=geometry if R==RS[0] else None)
    ax.set_yscale("log");ax.set_xlabel("R");ax.set_ylabel("cycles");ax.legend();ax.grid(alpha=.25);save(fig,"W10_VS_W25_SPECIMEN_SCALE_EFFECT")
    fig,ax=plt.subplots(figsize=(8,5.5));ax.bar(validation.classification.value_counts().index,validation.classification.value_counts().values);ax.set_ylabel("anchor count");ax.set_title("Two-scale validation state");save(fig,"TWO_SCALE_FINAL_MECHANISM_SUMMARY")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--anchors-only",action="store_true");args=parser.parse_args();root=args.root.resolve()
    anchors, validation=collect_anchors(root)
    if (validation.classification=="INTERPOLATION_FAILURE").any() or not validation.admissible.all():
        raise RuntimeError("mandatory anchor refinement required before virtual integration")
    v0,v1,overlays=build_surfaces(root,anchors)
    if args.anchors_only:return 0
    dynamic,dynamic_json=dynamic_validation(root,v0)
    curves,summary=integrate_native(root,v1,dynamic_json["classification"])
    seed_sensitivity(root,summary)
    plot_figures(root,v0,v1,anchors,validation,dynamic,curves,summary,overlays)
    atomic_json(root/"two_scale_analysis_state.json",{"phase":"NATIVE_COMPLETE_PT_INTEGRATION_PENDING","anchor_count":len(anchors),
      "anchor_high_accuracy_count":int((validation.classification=="HIGH_ACCURACY").sum()),"dynamic_classification":dynamic_json["classification"],
      "native_virtual_curve_rows":len(curves),"native_virtual_summary_rows":len(summary)})
    return 0


if __name__=="__main__":raise SystemExit(main())
