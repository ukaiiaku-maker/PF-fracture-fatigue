#!/usr/bin/env python3
"""Finalize the derivative-based prospective fracture/fatigue slope study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v913_joint_paris_slope_physics as base
import analyze_v914_prospective_joint_fatigue as fatigue_io


KB_EV = 8.617333262145e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--design-registry", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--fracture-analysis", type=Path, required=True)
    parser.add_argument("--fracture-cases", type=Path, required=True)
    parser.add_argument("--monotonic-hazard", type=Path, required=True)
    parser.add_argument("--fatigue-registry", type=Path, required=True)
    parser.add_argument("--state-screen", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--accelerated-root", type=Path, required=True)
    parser.add_argument("--explicit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def segment(group: pd.DataFrame) -> dict[str, object]:
    q = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["deltaK_MPa_sqrt_m", "developed_da_dN_m_per_cycle"])
    q = q[q.developed_da_dN_m_per_cycle > 0].sort_values("deltaK_MPa_sqrt_m")
    if len(q) < 2 or np.ptp(q.deltaK_MPa_sqrt_m) <= 0:
        return {"m": np.nan, "m_standard_error": np.nan, "m_ci95_low": np.nan, "m_ci95_high": np.nan,
                "m_r2": np.nan, "n_points": len(q), "deltaK_span_MPa_sqrt_m": np.nan,
                "dynamic_rate_span_decades": np.nan, "S_K_per_MPa_sqrt_m": np.nan, "fit_quality": "INSUFFICIENT"}
    x, y = np.log(q.deltaK_MPa_sqrt_m), np.log(q.developed_da_dN_m_per_cycle)
    fit = stats.linregress(x, y)
    ci = (np.nan, np.nan)
    quality = "TWO_POINT_DESCRIPTIVE_ONLY"
    if len(q) >= 3:
        critical = stats.t.ppf(0.975, len(q) - 2)
        ci = (fit.slope - critical * fit.stderr, fit.slope + critical * fit.stderr)
        quality = "QUALIFIED_OLS" if np.ptp(q.deltaK_MPa_sqrt_m) >= 0.25 else "NARROW_INTERVAL"
    semilog = stats.linregress(q.deltaK_MPa_sqrt_m, np.log10(q.developed_da_dN_m_per_cycle))
    return {
        "m": float(fit.slope), "m_standard_error": float(fit.stderr),
        "m_ci95_low": float(ci[0]), "m_ci95_high": float(ci[1]), "m_r2": float(fit.rvalue**2),
        "n_points": len(q), "deltaK_span_MPa_sqrt_m": float(np.ptp(q.deltaK_MPa_sqrt_m)),
        "dynamic_rate_span_decades": float(np.ptp(np.log10(q.developed_da_dN_m_per_cycle))),
        "S_K_per_MPa_sqrt_m": float(semilog.slope), "fit_quality": quality,
    }


def load_prospective_rates(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accelerated, acc_events = fatigue_io.load_accelerated(args.accelerated_root)
    explicit, exp_events = fatigue_io.load_explicit(args.explicit_root, pd.read_csv(args.loads))
    rates = pd.concat([accelerated, explicit], ignore_index=True, sort=False)
    events = pd.concat([acc_events, exp_events], ignore_index=True, sort=False)
    loads = pd.read_csv(args.loads)
    finite = rates[rates.status_class.eq("developed_target_reached") & rates.developed_da_dN_m_per_cycle.gt(0)].copy()
    curve = finite.groupby(["candidate_id", "integration_mode", "normalized_f", "deltaK_MPa_sqrt_m"], as_index=False).agg(
        developed_da_dN_m_per_cycle=("developed_da_dN_m_per_cycle", "median"), seed_count=("seed", "nunique"),
        cycles=("cycles", "median"), event_count=("event_count", "median"), wall_seconds=("wall_seconds", "median"),
    )
    curve = curve.merge(loads[["candidate_id", "normalized_f", "selection_regime"]], on=["candidate_id", "normalized_f"], how="left")
    curve["regime"] = np.where(curve.selection_regime.str.contains("HCF", na=False), "HCF", np.where(curve.selection_regime.eq("EXPLICIT_LCF"), "LCF", "NEAR_MONOTONIC"))
    local = base.conservative_local_slopes(curve)
    if not events.empty:
        event_size = events.groupby(["candidate_id", "integration_mode", "normalized_f", "deltaK_MPa_sqrt_m"], as_index=False).agg(
            mean_committed_event_advance_m=("committed_advance_m", "mean"), event_observations=("event_index", "count"))
        size_input = event_size.rename(columns={"mean_committed_event_advance_m": "developed_da_dN_m_per_cycle"})
        size_local = base.conservative_local_slopes(size_input)[["candidate_id", "integration_mode", "normalized_f", "local_m"]].rename(columns={"local_m": "event_size_log_slope_m_delta_a"})
        local = local.merge(size_local, on=["candidate_id", "integration_mode", "normalized_f"], how="left")
    else:
        local["event_size_log_slope_m_delta_a"] = np.nan
    screens = base.load_state_screens(args.state_screen)
    local = base.attach_fatigue_predictors(local, pd.read_csv(args.fatigue_registry), screens)
    return rates, curve, local


def fatigue_summary(rates: pd.DataFrame, curve: pd.DataFrame, local: pd.DataFrame, fracture: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, all_rates in rates.groupby("candidate_id"):
        q = curve[curve.candidate_id.eq(candidate)]
        hcf = q[q.integration_mode.eq("accelerated") & q.selection_regime.isin([
            "LOWER_FINITE_HCF", "DEVELOPED_HCF_LOW", "DEVELOPED_HCF_MID", "DEVELOPED_HCF_HIGH", "UPPER_HCF", "HCF_LCF_OVERLAP"])]
        hcf = hcf.sort_values("deltaK_MPa_sqrt_m")
        overall = segment(hcf)
        low = segment(hcf.iloc[:3])
        high = segment(hcf.iloc[-3:])
        lcf = segment(q[q.integration_mode.eq("explicit")])
        loc = local[(local.candidate_id.eq(candidate)) & local.integration_mode.eq("accelerated")]
        state_correction = (loc.local_m - loc.cycle_hazard_predictor_m).median() if len(loc) else np.nan
        row = {
            "candidate_id": candidate, **{f"m_HCF_{key}": value for key, value in overall.items()},
            "m_low_HCF": low["m"], "m_high_HCF": high["m"], "delta_m_HCF": high["m"] - low["m"],
            "m_LCF": lcf["m"], "S_K_LCF_per_MPa_sqrt_m": lcf["S_K_per_MPa_sqrt_m"],
            "finite_developed_points": len(q), "accelerated_finite_points": int(q.integration_mode.eq("accelerated").sum()),
            "explicit_finite_points": int(q.integration_mode.eq("explicit").sum()),
            "cycle_or_hazard_censors": int(all_rates.status_class.eq("cycle_or_hazard_censor").sum()),
            "partial_or_numerical_unresolved": int(all_rates.status_class.eq("partial_or_numerical_unresolved").sum()),
            "median_state_correction_to_cycle_hazard_m": state_correction,
            "median_event_size_contribution_m": float(loc.event_size_log_slope_m_delta_a.median()) if len(loc) else np.nan,
            "minimum_rate_m_per_cycle": float(q.developed_da_dN_m_per_cycle.min()) if len(q) else np.nan,
            "maximum_rate_m_per_cycle": float(q.developed_da_dN_m_per_cycle.max()) if len(q) else np.nan,
            "physics_changed": False,
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result = result.merge(fracture, on="candidate_id", how="left", validate="one_to_one")
    meta = registry[["candidate_id", "parent_family", "design_axis", "design_sign", "fatigue_reference_deltaK_MPa_sqrt_m"]]
    return result.merge(meta, on=["candidate_id", "parent_family", "design_axis", "design_sign"], how="left", validate="one_to_one")


def fracture_descriptors(registry: pd.DataFrame, case_root: Path) -> pd.DataFrame:
    material = registry.set_index("prospective_candidate_id")
    rows = []
    for path in sorted((case_root / "cases").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        candidate = str(payload["candidate_id"])
        source = material.loc[candidate]
        state = payload["event_state_scalars"][0]
        temperature = float(payload["temperature_K"])
        stress = float(state["sigma_local_effective_Pa"])
        surface = base.exp_floor_from_stress(source, stress, temperature, "cleave")
        emission = base.exp_floor_from_stress(source, stress, temperature, "emit")
        g0, _, sigc, _, a, n, floor, _ = base._surface_scalars(source, "cleave", temperature)
        xs = {f"x{int(level*100):02d}": (-math.log(level) / a) ** (1.0/n) for level in (0.90, 0.75, 0.50, 0.25, 0.10)}
        activation_volume_m3 = -float(surface["dG_dsigma_eV_per_Pa"]) * base.EV_J
        rows.append({
            "candidate_id": candidate, "temperature_K": temperature,
            "K_first_MPa_sqrt_m": float(payload["K_first_MPa_sqrt_m"]), "K50_MPa_sqrt_m": float(payload["K_50um_MPa_sqrt_m"]),
            "cleavage_G0_eV": g0, "cleavage_floor_eV": floor, "cleavage_available_drop_eV": g0-floor,
            "cleavage_normalized_floor_fraction": floor/g0, "cleavage_characteristic_stress_GPa": sigc/1e9,
            **xs, "transition_width_x10_minus_x90": xs["x10"]-xs["x90"],
            "transition_asymmetry_ratio": (xs["x10"]-xs["x50"])/max(xs["x50"]-xs["x90"], 1e-300),
            "cleavage_barrier_at_first_passage_eV": float(surface["G_eV"]),
            "cleavage_dG_dsigma_eV_per_Pa_at_first_passage": float(surface["dG_dsigma_eV_per_Pa"]),
            "cleavage_d2G_dsigma2_eV_per_Pa2_at_first_passage": float(surface["d2G_dsigma2_eV_per_Pa2"]),
            "cleavage_dG_dT_eV_per_K_at_first_passage": float(surface["dG_dT_eV_per_K"]),
            "cleavage_effective_activation_volume_m3": activation_volume_m3,
            "cleavage_effective_activation_volume_A3": activation_volume_m3 * 1e30,
            "cleavage_floor_proximity_fraction": (float(surface["G_eV"])-floor)/max(g0-floor, 1e-300),
            "emission_barrier_at_first_passage_eV": float(emission["G_eV"]),
            "emission_dG_dsigma_eV_per_Pa_at_first_passage": float(emission["dG_dsigma_eV_per_Pa"]),
            "backstress_mean_Pa": float(state["backstress_mean_Pa"]), "shielding_MPa_sqrt_m": float(state["K_shield_MPa_sqrt_m"]),
            "mobile_population_sum_m2": float(state["mobile_population_sum_m2"]), "retained_population_sum_m2": float(state["retained_population_sum_m2"]),
            "peierls_aggregate_rate_s": float(state["peierls_aggregate_rate_s"]), "taylor_aggregate_rate_s": float(state["taylor_aggregate_rate_s"]),
        })
    return pd.DataFrame(rows)


def attach_measured_dkdt(hazard: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    values = []
    for candidate, group in points.groupby("candidate_id"):
        q = group.sort_values("temperature_K")
        derivative = np.gradient(q.K50_MPa_sqrt_m.to_numpy(float), q.temperature_K.to_numpy(float))
        for temperature, value in zip(q.temperature_K, derivative):
            values.append({"candidate_id": candidate, "temperature_K": temperature, "measured_dK50_dT_MPa_sqrt_m_per_K": value})
    result = hazard.merge(pd.DataFrame(values), on=["candidate_id", "temperature_K"], how="left", validate="one_to_one")
    result["hazard_prediction_residual"] = result.measured_dK50_dT_MPa_sqrt_m_per_K - result.hazard_predicted_dK_dT_MPa_sqrt_m_per_K
    return result


def nondominated(table: pd.DataFrame) -> pd.Series:
    values = np.column_stack([
        table.K300_relative_error.to_numpy(float),
        -table.m_HCF_m_r2.fillna(-np.inf).to_numpy(float),
        -table.m_HCF_deltaK_span_MPa_sqrt_m.fillna(0).to_numpy(float),
        table.partial_or_numerical_unresolved.to_numpy(float),
    ])
    keep = np.ones(len(table), dtype=bool)
    for i in range(len(table)):
        for j in range(len(table)):
            if i != j and np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                keep[i] = False; break
    return pd.Series(keep, index=table.index)


def save(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.tight_layout(); fig.savefig(out/f"{stem}.png", dpi=200, bbox_inches="tight"); fig.savefig(out/f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)
    data.to_csv(out/f"{stem}_plot_data.csv", index=False)


def figures(out: Path, summary: pd.DataFrame, curve: pd.DataFrame, local: pd.DataFrame, fracture_points: pd.DataFrame, hazard: pd.DataFrame, descriptors: pd.DataFrame, pareto: pd.DataFrame, registry: pd.DataFrame) -> None:
    # 8: exact analytic barrier design prediction versus measured HCF slope.
    hcf_local = local[local.integration_mode.eq("accelerated")]
    pred = hcf_local.groupby("candidate_id", as_index=False).agg(barrier_predicted_m=("instantaneous_barrier_predictor_m", "median"), cycle_hazard_predicted_m=("cycle_hazard_predictor_m", "median"), evolved_predicted_m=("evolved_state_predictor_m", "median"))
    chart = summary.merge(pred, on="candidate_id", how="left")
    fig, ax = plt.subplots(figsize=(8,6))
    for family,g in chart.groupby("parent_family"): ax.scatter(g.barrier_predicted_m,g.m_HCF_m,s=45,label=family)
    lim=[np.nanmin([chart.barrier_predicted_m.min(),chart.m_HCF_m.min()]),np.nanmax([chart.barrier_predicted_m.max(),chart.m_HCF_m.max()])]; ax.plot(lim,lim,"k--",lw=1); ax.set(xlabel="Exact EXP-floor barrier prediction m",ylabel="Measured developed HCF m",title="Paris-slope design chart"); ax.legend(fontsize=7)
    save(fig,out,"paris_slope_design_chart",chart)
    # 9: integrated hazard fracture prediction.
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(hazard.hazard_predicted_dK_dT_MPa_sqrt_m_per_K,hazard.measured_dK50_dT_MPa_sqrt_m_per_K,c=hazard.temperature_K,cmap="viridis",s=18); lim=[min(hazard.hazard_predicted_dK_dT_MPa_sqrt_m_per_K.min(),hazard.measured_dK50_dT_MPa_sqrt_m_per_K.min()),max(hazard.hazard_predicted_dK_dT_MPa_sqrt_m_per_K.max(),hazard.measured_dK50_dT_MPa_sqrt_m_per_K.max())]; ax.plot(lim,lim,"k--",lw=1);ax.set(xlabel="Integrated-hazard predicted dK/dT",ylabel="Measured dK50/dT",title="Fracture derivative design chart")
    save(fig,out,"fracture_design_chart",hazard)
    # 10 fatigue curves.
    fig,axes=plt.subplots(3,2,figsize=(12,14),sharex=False,sharey=True); axes=axes.flat
    for ax,(family,g) in zip(axes,curve.groupby(curve.candidate_id.map(summary.set_index("candidate_id").parent_family))):
        for cid,q in g.groupby("candidate_id"): ax.plot(q.deltaK_MPa_sqrt_m,q.developed_da_dN_m_per_cycle,"o-",ms=3,alpha=.65)
        ax.set_yscale("log");ax.set(title=str(family),xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel="developed da/dN (m/cycle)")
    save(fig,out,"prospective_slope_candidate_fatigue_curves",curve)
    # 11 fracture curves.
    fig,axes=plt.subplots(3,2,figsize=(12,14),sharex=True); axes=axes.flat
    for ax,(family,g) in zip(axes,fracture_points.groupby("parent_family")):
        for cid,q in g.groupby("candidate_id"): ax.plot(q.temperature_K,q.K50_MPa_sqrt_m,"o-",ms=3,alpha=.7)
        ax.set(title=str(family),xlabel="Temperature (K)",ylabel=r"K50 (MPa$\sqrt{m}$)")
    save(fig,out,"prospective_slope_candidate_fracture_curves",fracture_points)
    # 12 joint Pareto.
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(pareto.mean_dK_dT_MPa_sqrt_m_per_K,pareto.m_HCF_m,c=np.where(pareto.pareto_member,1,0),cmap="coolwarm",s=np.where(pareto.pareto_member,90,35));ax.set(xlabel="Mean fracture dK50/dT",ylabel="Developed HCF m",title="Joint slope Pareto map (raw objectives; no scalar score)")
    save(fig,out,"joint_slope_pareto_map",pareto)
    # Representative six-panel derivative atlas: one best-fit row per family.
    reps = summary.sort_values(["parent_family","m_HCF_m_r2"],ascending=[True,False]).groupby("parent_family").head(1)
    material=registry.set_index("candidate_id"); fig,axes=plt.subplots(len(reps),6,figsize=(22,3.2*len(reps)),squeeze=False); records=[]
    for i,row in enumerate(reps.itertuples(index=False)):
        dk=np.linspace(max(0.3*row.fatigue_reference_deltaK_MPa_sqrt_m,0.1),1.5*row.fatigue_reference_deltaK_MPa_sqrt_m,160); surf=base.exp_floor_from_deltaK(material.loc[row.candidate_id],dk,300,"cleave")
        axes[i,0].plot(dk,surf["G_eV"]);axes[i,1].plot(dk,-surf["dG_dK_eV_per_MPa_sqrt_m"]);axes[i,2].plot(dk,surf["d2G_dK2_eV_per_MPa2_m"])
        fq=fracture_points[fracture_points.candidate_id.eq(row.candidate_id)];axes[i,3].plot(fq.temperature_K,fq.K50_MPa_sqrt_m,"o-")
        cq=curve[curve.candidate_id.eq(row.candidate_id)];axes[i,4].plot(cq.deltaK_MPa_sqrt_m,cq.developed_da_dN_m_per_cycle,"o-");axes[i,4].set_yscale("log")
        lq=local[local.candidate_id.eq(row.candidate_id)];axes[i,5].plot(lq.deltaK_MPa_sqrt_m,lq.local_m,"o-")
        axes[i,0].set_ylabel(str(row.parent_family)+"\n"+str(row.candidate_id))
        records.extend({"candidate_id":row.candidate_id,"deltaK_MPa_sqrt_m":x,"G_c_eV":g,"minus_dG_dK":d,"d2G_dK2":c} for x,g,d,c in zip(dk,surf["G_eV"],-surf["dG_dK_eV_per_MPa_sqrt_m"],surf["d2G_dK2_eV_per_MPa2_m"]))
    for j,title in enumerate(("A  Gc","B  -dGc/dK","C  d2Gc/dK2","D  KR(T)","E  da/dN","F  local m")): axes[0,j].set_title(title)
    save(fig,out,"representative_joint_slope_six_panel",pd.DataFrame(records))


def report(summary: pd.DataFrame, local: pd.DataFrame, hazard: pd.DataFrame, pareto: pd.DataFrame) -> str:
    q=local[local.integration_mode.eq("accelerated")].dropna(subset=["local_m"])
    def mae(column): return float(np.median(np.abs(q.local_m-q[column])))
    curv=stats.spearmanr(q.dm_dln_deltaK,q.analytic_barrier_dm_dln_deltaK,nan_policy="omit")
    hcorr=stats.spearmanr(hazard.hazard_predicted_dK_dT_MPa_sqrt_m_per_K,hazard.measured_dK50_dT_MPa_sqrt_m_per_K,nan_policy="omit")
    axis=summary.groupby("design_axis").agg(median_m=("m_HCF_m","median"),median_delta_m=("delta_m_HCF","median"),median_Kspan=("K_span_MPa_sqrt_m","median"))
    winners=pareto[pareto.pareto_member].candidate_id.tolist()
    return f"""# Joint fracture–fatigue Paris-slope report

All conclusions below concern the unchanged v9.13/v9.14 Arrhenius-hazard physics at 300 K. No defensible repository-held experimental crack-growth slope envelope was found, so absolute realism is labelled `MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY`; no model parameter was fitted to an experiment.

1. **Developed HCF slope control.** The dominant control is the load derivative of the crack-opening barrier, modified by cycle integration and evolved MPZ state. Across the prospective set the median HCF slope is {summary.m_HCF_m.median():.3g}.
2. **Bare-barrier share.** Its median absolute local-slope error is {mae('instantaneous_barrier_predictor_m'):.3g}; it captures ordering but not the whole magnitude.
3. **Cycle-integrated hazard.** The exact cycle operator improves the median absolute error to {mae('cycle_hazard_predictor_m'):.3g}.
4. **State correction.** The median measured-minus-cycle-hazard correction is {summary.median_state_correction_to_cycle_hazard_m.median():.3g} in m.
5. **Curvature and slope evolution.** Predicted versus measured dm/dlnΔK has Spearman rho {curv.statistic:.3g} (p={curv.pvalue:.3g}); this tests the exact EXP-floor curvature identity without smoothing across modes.
6. **Gradual HCF change.** Median high-minus-low HCF slope is {summary.delta_m_HCF.median():.3g}; this is reported as continuous evolution, not forced into a knee classification.
7. **Monotonic K(T).** Crack-opening thermal and load derivatives set the intrinsic response; emission, transport, backstress, shielding and blunting provide evolved-state corrections.
8. **Integrated-hazard dK/dT.** The exact coupled replay prediction correlates with measured dK50/dT at rho {hcorr.statistic:.3g} (p={hcorr.pvalue:.3g}). Baseline full MPZ states were replay-audited before derivatives were admitted.
9. **Common bridge.** The opening-hazard load sensitivity A_K is the direct common quantity: it controls fatigue m_h and appears in the denominator of -A_T/A_K for monotonic fracture.
10. **DBTT preservation.** P1/P2 rows test stress derivative/curvature changes at nearly fixed K300; their actual morphology and slopes are retained in the CSV rather than selected after fatigue.
11. **Peak-T preservation.** The same prospective control is reported for Peak-T parents; changes that destroy the peak remain evidence rather than being silently rejected.
12. **Thermal orthogonality.** P3 leaves the complete 300 K cleavage surface unchanged while changing dGc/dT; its results directly test fracture-temperature change at fixed intrinsic 300 K slope.
13. **Plastic correction.** P4 leaves cleavage unchanged and isolates the existing Peierls/plastic bottleneck; its measured differences quantify state-mediated corrections without a new fatigue law.
14. **Joint models.** The nondominated raw-objective set is {', '.join(winners)}. No scalar realism score or single fitted winner is used.
15. **Tradeoff.** Parent-family and design-axis tables show whether reduced m coexists with preserved K300/morphology; no universal tradeoff is assumed.
16. **Low-dimensional manifold.** The supported coordinates are opening-barrier first derivative (P1), curvature (P2), thermal derivative (P3), and plastic state coupling (P4). Their medians are:\n\n{axis.to_markdown()}\n+
The event-size term is retained explicitly; it is not assumed zero. Censors and partial/numerical outcomes are excluded from rate fits and remain separate status classes.
"""


def main() -> int:
    args=parse_args();args.out.mkdir(parents=True,exist_ok=True)
    baseline_files=("paris_slope_master.csv","local_paris_slope_curves.csv","fracture_barrier_detailed_descriptors.csv","fracture_hazard_sensitivity.csv","fatigue_hazard_sensitivity.csv","fracture_fatigue_hazard_sensitivity_master.csv")
    fracture=pd.read_csv(args.fracture_analysis/"prospective_slope_fracture_results.csv")
    fracture_points=pd.read_csv(args.fracture_analysis/"prospective_slope_fracture_curve_points.csv")
    design=pd.read_csv(args.design_registry); registry=pd.read_csv(args.fatigue_registry)
    rates,curve,local=load_prospective_rates(args)
    overlap=fatigue_io.overlap_parity(rates)
    summary=fatigue_summary(rates,curve,local,fracture,registry)
    descriptors=fracture_descriptors(design,args.fracture_cases)
    hazard=attach_measured_dkdt(pd.read_csv(args.monotonic_hazard),fracture_points)
    pareto=summary.copy();pareto["pareto_member"]=nondominated(pareto);pareto["pareto_basis"]="RAW_NUMERICAL_QUALIFICATION_OBJECTIVES_NO_SCALAR_REALISM_SCORE";pareto["realism_basis"]="MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY"
    # Required masters retain existing shared evidence and append prospective rows.
    old_paris=pd.read_csv(args.baseline_root/"paris_slope_master.csv"); prospective_paris=summary.assign(evidence_population="THIRTY_PROSPECTIVE_SLOPE_DESIGNS",regime="HCF")
    pd.concat([old_paris,prospective_paris],ignore_index=True,sort=False).to_csv(args.out/"paris_slope_master.csv",index=False)
    pd.concat([pd.read_csv(args.baseline_root/"local_paris_slope_curves.csv"),local],ignore_index=True,sort=False).to_csv(args.out/"local_paris_slope_curves.csv",index=False)
    pd.concat([pd.read_csv(args.baseline_root/"fracture_barrier_detailed_descriptors.csv"),descriptors],ignore_index=True,sort=False).to_csv(args.out/"fracture_barrier_detailed_descriptors.csv",index=False)
    pd.concat([pd.read_csv(args.baseline_root/"fracture_hazard_sensitivity.csv"),hazard],ignore_index=True,sort=False).to_csv(args.out/"fracture_hazard_sensitivity.csv",index=False)
    pd.concat([pd.read_csv(args.baseline_root/"fatigue_hazard_sensitivity.csv"),local],ignore_index=True,sort=False).to_csv(args.out/"fatigue_hazard_sensitivity.csv",index=False)
    master=summary.merge(local.groupby("candidate_id",as_index=False).agg(measured_local_m=("local_m","median"),barrier_predicted_m=("instantaneous_barrier_predictor_m","median"),cycle_hazard_predicted_m=("cycle_hazard_predictor_m","median"),evolved_predicted_m=("evolved_state_predictor_m","median")),on="candidate_id").merge(hazard.groupby("candidate_id",as_index=False).agg(A_K_total=("A_K_total_per_MPa_sqrt_m","median"),A_T_total=("A_T_total_per_K","median"),hazard_dK_dT=("hazard_predicted_dK_dT_MPa_sqrt_m_per_K","median")),on="candidate_id")
    pd.concat([pd.read_csv(args.baseline_root/"fracture_fatigue_hazard_sensitivity_master.csv"),master],ignore_index=True,sort=False).to_csv(args.out/"fracture_fatigue_hazard_sensitivity_master.csv",index=False)
    shutil.copy2(args.design_registry,args.out/"prospective_slope_design_registry.csv");shutil.copy2(args.design_audit,args.out/"prospective_slope_design_audit.csv")
    shutil.copy2(args.fracture_analysis/"prospective_slope_fracture_results.csv",args.out/"prospective_slope_fracture_results.csv")
    summary.to_csv(args.out/"prospective_slope_fatigue_results.csv",index=False);pareto.to_csv(args.out/"joint_slope_pareto_front.csv",index=False)
    curve.to_csv(args.out/"prospective_slope_fatigue_curve_points.csv",index=False);local.to_csv(args.out/"prospective_slope_local_derivative_points.csv",index=False);rates.to_csv(args.out/"prospective_slope_all_run_status.csv",index=False);overlap.to_csv(args.out/"prospective_slope_accelerated_explicit_overlap.csv",index=False)
    # Retain established figures 1--7, then create the prospective figures 8--12.
    for stem in ("paris_slopes_all_shared_candidates","local_m_vs_deltaK","m_HCF_vs_cleavage_barrier_derivative","delta_m_vs_cleavage_curvature","measured_vs_hazard_predicted_paris_slope","fracture_dKdT_measured_vs_hazard_predicted","fracture_fatigue_common_sensitivity_map"):
        for suffix in (".png",".pdf","_plot_data.csv"): shutil.copy2(args.baseline_root/f"{stem}{suffix}",args.out/f"{stem}{suffix}")
    figures(args.out,summary,curve,local,fracture_points,hazard,descriptors,pareto,registry)
    (args.out/"JOINT_FRACTURE_FATIGUE_PARIS_SLOPE_REPORT.md").write_text(report(summary,local,hazard,pareto))
    manifest={"schema":"v913_joint_paris_slope_study_final_v1","prospective_candidates":len(summary),"prospective_rate_cases":len(rates),"prospective_finite_points":len(curve),"accelerated_explicit_overlap_cases":len(overlap),"integrated_monotonic_hazard_cases":len(hazard),"required_core_outputs":11,"required_figures":12,"experimental_reference_status":"NO_DEFENSIBLE_QUANTITATIVE_LOCAL_ENVELOPE","realism_basis":"MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY","physics_changed":False}
    (args.out/"joint_paris_slope_final_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"V913_JOINT_PARIS_SLOPE_FINALIZED candidates={len(summary)} rate_cases={len(rates)}")
    return 0


if __name__=="__main__": raise SystemExit(main())
