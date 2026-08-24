#!/usr/bin/env python3
"""Large, vectorized, interpretable low-fidelity Paris-window search."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import gammainc
from scipy.stats import qmc, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.audit_v914_historical_analytical_model import COORDINATES

KB = 8.617333262145e-5
BOUNDS = {
    "cleave_G00_eV": (1.0, 5.5, "linear"), "cleave_sigc0_GPa": (1.0, 8.0, "linear"),
    "cleave_exp_a": (.05, 1.5, "log10"), "cleave_exp_n": (.5, 3.0, "linear"),
    "cleave_floor_frac": (.005, .5, "log10"), "emit_G00_eV": (.4, 5.0, "linear"),
    "emit_sigc0_GPa": (1.0, 9.0, "linear"), "emit_exp_a": (.03, 1.5, "log10"),
    "emit_exp_n": (.45, 2.5, "linear"), "emit_floor_frac": (.005, .2, "log10"),
    "peierls_H0_eV": (.5, 7.0, "linear"), "peierls_activation_entropy_kB": (-40., 20., "linear"),
    "peierls_exp_a": (.1, 2., "log10"), "peierls_exp_n": (.5, 2.75, "linear"),
    "taylor_H0_eV": (.08, 2., "linear"), "taylor_activation_entropy_kB": (0., 40., "linear"),
    "taylor_exp_a": (.03, 1., "log10"), "taylor_exp_n": (.5, 2.5, "linear"),
    "rho_source0_m2": (1e14, 1e17, "log10"), "taylor_corr_rho_c_m2": (1e11, 1e18, "log10"),
    "taylor_corr_scale": (.1, 10., "log10"), "c_blunt": (0., 3., "linear"),
}
assert tuple(BOUNDS) == COORDINATES


def transform(u: np.ndarray, bounds=BOUNDS) -> pd.DataFrame:
    out = {}
    for j, (name, (lo, hi, mode)) in enumerate(bounds.items()):
        out[f"u__{name}"] = u[:, j]
        out[name] = 10 ** (math.log10(lo) + u[:, j] * math.log10(hi / lo)) if mode == "log10" else lo + u[:, j] * (hi - lo)
    return pd.DataFrame(out)


def cleavage_extension_bounds(path: Path) -> dict:
    shell=pd.read_csv(path); shell=shell[shell.barrier_surface_valid.astype(bool)]
    result=dict(BOUNDS)
    for name in COORDINATES[:5]:
        lo=float(shell[name].min()); hi=float(shell[name].max()); mode=BOUNDS[name][2]
        if not hi>lo: raise ValueError(f"cleavage shell does not vary {name}")
        result[name]=(lo,hi,mode)
    return result


def normalized(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"u__{x}" for x in COORDINATES]].to_numpy(float)


def add_reference_rows(frame: pd.DataFrame, registries: list[Path]) -> pd.DataFrame:
    refs = []
    wanted = {"v914_endurance_knee_0462", "v914_endurance_knee_0554", "v914_endurance_knee_0300", "v914_endurance_knee_0917"}
    for path in registries:
        d = pd.read_csv(path)
        if "candidate_id" not in d: continue
        take = d[d.candidate_id.isin(wanted) | d.get("paris_design_role", pd.Series("", index=d.index)).astype(str).str.contains("SELECTED|PROMOT", case=False)]
        for _, row in take.iterrows():
            rec = {"candidate_id": str(row.candidate_id), "population_role": "fixed_reference"}
            for name, (lo, hi, mode) in BOUNDS.items():
                v = float(row[name]); rec[name] = v
                rec[f"u__{name}"] = (math.log10(v / lo) / math.log10(hi / lo)) if mode == "log10" else (v - lo) / (hi - lo)
            refs.append(rec)
    if refs:
        r = pd.DataFrame(refs).drop_duplicates("candidate_id")
        frame = pd.concat([frame, r], ignore_index=True, sort=False)
    return frame


def surface(G0, sigc, a, n, floor_frac, stress):
    floor = np.maximum(1e-4, floor_frac * G0)
    x = np.maximum(stress, 0.) / (sigc * 1e9)
    return floor + (G0 - floor) * np.exp(-a * np.power(x, n)), x


def evaluate(frame: pd.DataFrame, loads: np.ndarray, physics: dict, cfg: dict, offset: float, uncertainty: float):
    n, nk = len(frame), len(loads)
    K = loads[None, :]
    stress = K / .9 * 1e6 / math.sqrt(2 * math.pi * float(physics["r0_m"]))
    col = lambda x: frame[x].to_numpy(float)[:, None]
    G, x = surface(col("cleave_G00_eV"), col("cleave_sigc0_GPa"), col("cleave_exp_a"), col("cleave_exp_n"), col("cleave_floor_frac"), stress)
    raw = float(physics["cleavage_nu0_s"]) * np.exp(np.clip(-G / (KB * 300.), -700., 0.))
    tau = float(physics["cleavage_correlation_time_s"]); rt = raw * tau
    p3 = gammainc(3., rt); rate3 = np.maximum(p3 / tau, 1e-300)
    amp = np.divide(rt ** 3 * np.exp(-rt) / 2., np.maximum(p3, 1e-300))
    dGdlnK = -(col("cleave_G00_eV") * (1-col("cleave_floor_frac"))) * np.exp(-col("cleave_exp_a") * x ** col("cleave_exp_n")) * col("cleave_exp_a") * col("cleave_exp_n") * x ** col("cleave_exp_n")
    mraw = -dGdlnK / (KB * 300.); mlocal = amp * mraw
    elog, ex = surface(col("emit_G00_eV"), col("emit_sigc0_GPa"), col("emit_exp_a"), col("emit_exp_n"), col("emit_floor_frac"), stress / math.sqrt(3.))
    emit = float(physics["emission_nu0_s"]) * np.exp(np.clip(-elog / (KB * 300.), -700., 0.))
    plastic = np.log10(np.maximum(emit * col("rho_source0_m2") * 1e-17, 1e-300))
    transport = (col("peierls_H0_eV") - 300 * KB * col("peierls_activation_entropy_kB")) + (col("taylor_H0_eV") - 300 * KB * col("taylor_activation_entropy_kB"))
    competition = np.log10(rate3) - np.log10(np.maximum(emit, 1e-300))
    predicted = np.log10(rate3) + offset
    lo = float(cfg["target_log10_da_dN_min_m_per_cycle"]); hi = float(cfg["target_log10_da_dN_max_m_per_cycle"])
    # Interpretable regime proxy: finite rate, unsaturated three-hit response, and no gross cleavage/emission imbalance.
    feasible = (predicted >= lo - 2*uncertainty) & (predicted <= hi + 2*uncertainty) & (rt > 1e-10) & (rt < 30.) & (competition > -14.) & (competition < 14.)
    arrays = dict(Gstar=G, raw=raw, raw_tau=rt, rate3=rate3, three_hit_amplification=amp,
                  predicted=predicted, mraw=mraw, mlocal=mlocal, feasible=feasible,
                  competition=competition, plastic=plastic, transport=np.broadcast_to(transport, (n, nk)))
    return arrays


def best_window(loads, y, feasible, cfg):
    best = None; minimum = int(cfg["minimum_Paris_points"])
    for i in range(len(loads)):
        if not feasible[i]: continue
        for j in range(i + minimum, len(loads) + 1):
            if not np.all(feasible[i:j]): break
            x = np.log10(loads[i:j]); z = y[i:j]
            span = loads[j-1] / loads[i] - 1.; decades = float(z.max()-z.min())
            if span < float(cfg["minimum_relative_DeltaK_span"]): continue
            m, b = np.polyfit(x, z, 1); pred = b + m*x; resid = z-pred
            ss = np.sum((z-z.mean())**2); r2 = 1-np.sum(resid**2)/ss if ss > 0 else 0.
            rmse = float(np.sqrt(np.mean(resid**2)))
            if r2 < float(cfg["minimum_fit_R2"]): continue
            slope_mismatch = max(float(cfg["target_m_min"])-m, 0., m-float(cfg["target_m_max"]))
            shape = slope_mismatch + rmse + max(float(cfg["minimum_fit_R2"])-r2,0.) + max(float(cfg["preferred_da_dN_span_decades"])-decades,0.)
            score = (round(shape, 12), -span, -len(x), rmse)
            rec = dict(m=float(m), log10_C=float(b), C=float(10**b), R2=float(r2), residual_rmse_log10=rmse,
                       DeltaK_min=float(loads[i]), DeltaK_max=float(loads[j-1]), DeltaK_fractional_span=float(span),
                       da_dN_span_decades=decades, n_points=int(j-i), window_start_index=i, window_end_index=j-1,
                       shape_score=float(shape))
            if best is None or score < best[0]: best=(score,rec)
    return None if best is None else best[1]


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--target-config",type=Path,required=True); ap.add_argument("--physics",type=Path,required=True)
    ap.add_argument("--calibration-points",type=Path,required=True); ap.add_argument("--reference-registry",type=Path,action="append",default=[])
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--count",type=int,default=524288); ap.add_argument("--seed",type=int,default=1914)
    ap.add_argument("--domain",choices=("historical","cleavage-extension"),default="historical"); ap.add_argument("--cleavage-shell",type=Path)
    ap.add_argument("--deltaK-min",type=float,default=8.0); ap.add_argument("--deltaK-max",type=float,default=32.0); ap.add_argument("--deltaK-count",type=int,default=16)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.target_config.read_text()); physics=json.loads(a.physics.read_text())["common_physics"]
    cal=pd.read_csv(a.calibration_points); use=cal.developed_point_usable.astype(bool)&(pd.to_numeric(cal.developed_da_dN_m_per_cycle,errors="coerce")>0)
    delta=np.log10(cal.loc[use,"developed_da_dN_m_per_cycle"].to_numpy(float))-np.log10(np.maximum(cal.loc[use,"analytical_Gamma_3hit_s"].to_numpy(float),1e-300))
    offset=float(np.mean(delta)); uncertainty=float(np.std(delta,ddof=1))
    u=qmc.Sobol(d=len(COORDINATES),scramble=True,seed=a.seed).random(a.count)
    bounds=BOUNDS if a.domain=="historical" else cleavage_extension_bounds(a.cleavage_shell)
    population=transform(u,bounds); prefix="v914_large_analytical" if a.domain=="historical" else "v914_cleavage_extension_analytical"
    role="HISTORICAL_DOMAIN" if a.domain=="historical" else "CLEAVAGE_EXTENSION"
    population.insert(0,"candidate_id",[f"{prefix}_{i:07d}" for i in range(len(population))]); population.insert(1,"population_role",role)
    population=add_reference_rows(population,a.reference_registry)
    loads=np.geomspace(a.deltaK_min,a.deltaK_max,a.deltaK_count); arrays=evaluate(population,loads,physics,cfg,offset,uncertainty)
    rows=[]
    for i,cid in enumerate(population.candidate_id):
        win=best_window(loads,arrays["predicted"][i],arrays["feasible"][i],cfg)
        rec={"candidate_id":cid,"valid_analytical_Paris_window":win is not None,"predicted_sustained_growth_probability":float(np.mean(arrays["feasible"][i])),"prediction_uncertainty_log10":uncertainty,"m_uncertainty_proxy":uncertainty/max(math.log10(a.deltaK_max/a.deltaK_min),1e-9)}
        if win: rec.update(win); rec["level_score"]=abs(float(np.mean(arrays["predicted"][i,win["window_start_index"]:win["window_end_index"]+1]))-float(cfg["target_log10_da_dN_center_m_per_cycle"])); rec["predicted_log10_da_dN_median"]=float(np.median(arrays["predicted"][i,win["window_start_index"]:win["window_end_index"]+1]))
        rows.append(rec)
    windows=pd.DataFrame(rows); merged=population.merge(windows,on="candidate_id")
    merged.to_parquet(a.out/"analytical_large_population.parquet",index=False)
    merged.to_parquet(a.out/"large_search_candidates.parquet",index=False)
    windows.to_csv(a.out/"analytical_paris_windows.csv",index=False)
    windows.to_parquet(a.out/"large_search_paris_windows.parquet",index=False)
    valid=merged[merged.valid_analytical_Paris_window].copy(); valid["target_m_distance"]=np.maximum(float(cfg["target_m_min"])-valid.m,0)+np.maximum(valid.m-float(cfg["target_m_max"]),0)
    valid=valid.sort_values(["target_m_distance","shape_score","DeltaK_fractional_span","predicted_sustained_growth_probability"],ascending=[True,True,False,False])
    valid.head(5000).to_csv(a.out/"analytical_large_population_top.csv",index=False)
    valid.head(5000).to_csv(a.out/"large_search_top_candidates.csv",index=False)
    point_rows=[]
    for i in valid.head(5000).index:
        for k,K in enumerate(loads): point_rows.append({"candidate_id":population.iloc[i].candidate_id,"DeltaK_MPa_sqrt_m":K,"raw_analytical_rate_s":arrays["raw"][i,k],"three_hit_rate_s":arrays["rate3"][i,k],"corrected_log10_da_dN":arrays["predicted"][i,k],"analytical_local_slope":arrays["mlocal"][i,k],"feasible_regime_proxy":bool(arrays["feasible"][i,k]),"cleavage_emission_log10_ratio":arrays["competition"][i,k]})
    pd.DataFrame(point_rows).to_parquet(a.out/"analytical_top_response_points.parquet",index=False)
    correlations=[]
    if len(valid)>2:
        for name in COORDINATES: correlations.append({"parameter":name,"spearman_rho_with_m":float(spearmanr(valid[name],valid.m).statistic)})
    pd.DataFrame(correlations).to_csv(a.out/"analytical_parameter_m_correlations.csv",index=False)
    summary={"schema":"v914_large_analytical_paris_search_v1","domain":role,"sobol_count":a.count,"total_with_references":len(population),"valid_window_count":len(valid),"valid_window_fraction":len(valid)/len(population),"target_range_count":int(valid.m.between(cfg["target_m_min"],cfg["target_m_max"]).sum()),"preferred_range_count":int(valid.m.between(cfg.get("preferred_m_min",cfg["target_m_min"]),cfg.get("preferred_m_max",cfg["target_m_max"])).sum()),"calibration_offset_log10":offset,"calibration_uncertainty_log10":uncertainty,"deltaK_grid":loads.tolist(),"learned_high_dimensional_residual_used":False,"target_config":str(a.target_config.resolve())}
    (a.out/"analytical_large_search_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
