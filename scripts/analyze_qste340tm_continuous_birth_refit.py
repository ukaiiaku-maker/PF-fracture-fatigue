#!/usr/bin/env python3
"""Profile, validate, and plot the selected corrected QSTE340TM reducer."""
from __future__ import annotations

import csv, hashlib, json, math
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.refit_qste340tm_capacity_policy import (
    DATA, LOWER, OUT, UPPER, TEMPERATURE_K, SMOOTHING_RATIO,
    build_records, embrittlement_eV, evaluate, metrics, unpack,
)

OUTDIR = OUT / "continuous_birth_exact"
MODE = "continuous_birth_exact"


def rolling_slope(x, y, width=21):
    answer = np.full(len(x), np.nan)
    half = width // 2
    for i in range(len(x)):
        lo, hi = max(0, i-half), min(len(x), i+half+1)
        if hi-lo >= 7:
            answer[i] = np.polyfit(np.log10(x[lo:hi]), np.log10(y[lo:hi]), 1)[0]
    return answer


def main() -> int:
    records = build_records(DATA)
    selected = json.loads((OUTDIR / "fit_uncapped_continuous_birth.json").read_text())
    x = np.asarray(selected["parameters_vector"], dtype=float)
    observed = np.asarray([float(r["observed_dadn_m_per_cycle"]) for r in records])
    rv = np.asarray([float(r["R"]) for r in records])
    dk = np.asarray([float(r["DeltaK_MPa_sqrt_m"]) for r in records])
    cycles = np.asarray([float(r["cycles"]) for r in records])
    lo, hi = LOWER.copy(), UPPER.copy()
    hi[9] = math.log1p(3000.0)

    def fit_fixed_initial(n0):
        keep = np.arange(10) != 9
        initial = x[keep]
        def assemble(y):
            full = np.empty(10); full[keep] = y; full[9] = math.log1p(n0); return full
        def residual(y):
            st = evaluate(assemble(y), records, "uncapped", retention_timing=MODE)
            error = []
            for r in sorted(set(rv)):
                mask = rv == r
                error.extend((np.log10(st["prediction"][mask]/observed[mask]) / math.sqrt(mask.sum())).tolist())
            return np.r_[error, max(0.0, .25-float(st["raw_opening_barrier_over_kBT"].min()))*10]
        sol = least_squares(residual, initial, bounds=(lo[keep], hi[keep]), max_nfev=90,
                            x_scale="jac", ftol=1e-9, xtol=1e-9, gtol=1e-9)
        full = assemble(sol.x); st = evaluate(full, records, "uncapped", retention_timing=MODE)
        p = unpack(full)
        return {"fixed_initial_retained_count": n0, "parameters_vector": full.tolist(),
                "parameters": p, "metrics": metrics(records, st),
                "stored_energy_initial_offset_eV": float(
                    -(embrittlement_eV(n0) - embrittlement_eV(0.0))),
                "optimizer": {"success": bool(sol.success), "message": str(sol.message),
                              "nfev": int(sol.nfev)}}

    profile_path = OUTDIR / "initial_state_profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        for item in profile:
            n0 = float(item["fixed_initial_retained_count"])
            item["stored_energy_initial_offset_eV"] = float(
                -(embrittlement_eV(n0) - embrittlement_eV(0.0))
            )
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True)+"\n")
    else:
        with ThreadPool(4) as pool:
            profile = pool.map(fit_fixed_initial, [0.0, 100.0, 400.0, 1000.0, 2000.0, 3000.0])
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True)+"\n")

    def fit_holdout(held):
        training = rv != held
        training_counts = {r: int(np.sum((rv == r) & training)) for r in sorted(set(rv))}
        def residual(v):
            st = evaluate(v, records, "uncapped", retention_timing=MODE)
            pieces = []
            for r in sorted(set(rv[training])):
                mask = (rv == r) & training
                pieces.extend((np.log10(st["prediction"][mask]/observed[mask]) / math.sqrt(training_counts[r])).tolist())
            return np.r_[pieces, max(0.0, .25-float(st["raw_opening_barrier_over_kBT"].min()))*10]
        sol = least_squares(residual, x, bounds=(lo, hi), max_nfev=50,
                            x_scale="jac", ftol=1e-9, xtol=1e-9, gtol=1e-9)
        st = evaluate(sol.x, records, "uncapped", retention_timing=MODE)
        err = np.log10(st["prediction"]/observed)
        result = {"held_out_R": held, "train_points": int(training.sum()),
                "held_out_points": int((~training).sum()),
                "train_rmse_log10": float(np.sqrt(np.mean(err[training]**2))),
                "held_out_rmse_log10": float(np.sqrt(np.mean(err[~training]**2))),
                "held_out_bias_log10": float(np.mean(err[~training])),
                "parameters_vector": sol.x.tolist(), "parameters": unpack(sol.x),
                "optimizer": {"success": bool(sol.success), "message": str(sol.message),
                              "nfev": int(sol.nfev)}}
        (OUTDIR / f"holdout_R{held:g}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result
    holdout_path = OUTDIR / "complete_R_holdouts.json"
    if holdout_path.exists():
        holdouts = json.loads(holdout_path.read_text())
    else:
        with ThreadPool(4) as pool:
            holdouts = pool.map(fit_holdout, sorted(set(rv)))
        holdout_path.write_text(json.dumps(holdouts, indent=2, sort_keys=True)+"\n")

    state = evaluate(x, records, "uncapped", retention_timing=MODE)
    frozen = evaluate(x, records, "uncapped", evolve=False, retention_timing=MODE)
    colors = {0.1:"#0072B2", 0.3:"#D55E00", 0.5:"#009E73", 0.7:"#CC79A7"}
    xmin, xmax = dk.min()*.94, dk.max()*1.06
    ymin = min(observed.min(), state["prediction"].min())*.45
    ymax = max(observed.max(), state["prediction"].max())*2.2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, sorted(set(rv))):
        m = rv == r; c = colors[r]
        ax.scatter(dk[m], observed[m], s=10, color=c, alpha=.45, label="data")
        ax.plot(dk[m], state["prediction"][m], color="black", lw=2, label="corrected full state")
        ax.plot(dk[m], frozen["prediction"][m], color="#777777", ls="--", lw=1.5, label="same-parameter frozen")
        ax.set(xscale="log", yscale="log", xlim=(xmin,xmax), ylim=(ymin,ymax), title=f"R = {r:g}")
        ax.grid(True, which="both", alpha=.2)
    for ax in axes[-1]: ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
    for ax in axes[:,0]: ax.set_ylabel(r"$da/dN$ (m/cycle)")
    h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,ncol=3,loc="lower center",frameon=False)
    fig.suptitle("QSTE340TM corrected continuous-birth refit (298.15 K, assumed 10 Hz)")
    fig.tight_layout(rect=(0,.06,1,.96)); fig.savefig(OUTDIR/"common_axis_data_fit.png",dpi=240); fig.savefig(OUTDIR/"common_axis_data_fit.svg"); plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(9,8),sharex=True)
    shape = {}
    for r in sorted(set(rv)):
        m=rv==r; c=colors[r]; so=rolling_slope(dk[m],observed[m]); sp=rolling_slope(dk[m],state["prediction"][m])
        axes[0].plot(dk[m],so,color=c,lw=1.5,label=f"data R={r:g}"); axes[0].plot(dk[m],sp,color=c,ls="--",lw=2,label=f"fit R={r:g}")
        residual=np.log10(state["prediction"][m]/observed[m]); axes[1].plot(dk[m],residual,color=c,lw=1.5,label=f"R={r:g}")
        signs=np.sign(residual); shape[str(r)]={"sliding_slope_observed_min":float(np.nanmin(so)),"sliding_slope_observed_max":float(np.nanmax(so)),"sliding_slope_predicted_min":float(np.nanmin(sp)),"sliding_slope_predicted_max":float(np.nanmax(sp)),"residual_sign_changes":int(np.sum(signs[1:]*signs[:-1]<0))}
    axes[0].set(ylabel="21-point local log-log slope"); axes[1].axhline(0,color="black",lw=.8); axes[1].set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)",ylabel="log10(predicted / observed)")
    for ax in axes: ax.grid(True,alpha=.2); ax.legend(ncol=4,fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(OUTDIR/"shape_and_residual_diagnostics.png",dpi=240); fig.savefig(OUTDIR/"shape_and_residual_diagnostics.svg"); plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(9,8),sharex=True)
    for r in sorted(set(rv)):
        m=rv==r;c=colors[r];axes[0].plot(cycles[m],state["retained"][m],color=c,label=f"R={r:g}");axes[1].plot(cycles[m],state["density_m2"][m],color=c,label=f"R={r:g}")
    axes[0].set(ylabel="retained count");axes[1].set(xlabel="reported cycles",ylabel=r"total density (m$^{-2}$)")
    for ax in axes:ax.grid(True,alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUTDIR/"retained_count_density_histories.png",dpi=240);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5)); stress=np.linspace(0,30e9,500); p=unpack(x); basis=SmoothBroadBasis(p["opening_sigma_b_Pa"],p["opening_exponent"],SMOOTHING_RATIO); barrier=(p["opening_floor_over_kBT"]+p["opening_amplitude_over_kBT"]*basis.value(stress))*KB_EV_PER_K*TEMPERATURE_K
    ax.plot(stress/1e9,barrier,lw=2);ax.set(xlabel="local opening stress (GPa)",ylabel=r"$G^*_{open}$ (eV)",title="Selected corrected opening barrier at 298.15 K");ax.grid(True,alpha=.2);fig.tight_layout();fig.savefig(OUTDIR/"Gstar_vs_stress.png",dpi=240);fig.savefig(OUTDIR/"Gstar_vs_stress.svg");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    keys=[("emission_per_cycle","emission"),("peierls_per_cycle","Peierls"),("taylor_per_cycle","Taylor"),("source_per_cycle","surviving source")]
    for ax,(key,label) in zip(axes.flat,keys):
        for r in sorted(set(rv)):
            m=rv==r;ax.plot(dk[m],np.maximum(state[key][m],1e-300),color=colors[r],label=f"R={r:g}")
        ax.set(yscale="log",ylabel=f"{label} / cycle");ax.grid(True,alpha=.2)
    for ax in axes[-1]:ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
    axes[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(OUTDIR/"separate_process_rates.png",dpi=240);plt.close(fig)

    source = DATA/"normalized/qste340tm_constant_amplitude.csv"
    summary={"schema":"qste340tm-continuous-birth-final-analysis/1","selected_model":"single_smooth_broad_EXP_floor_n1","shape_diagnostics":shape,"initial_state_profile":profile,"complete_R_holdouts":holdouts,"source_csv":str(source.relative_to(ROOT)),"source_csv_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"selected_fit_sha256":hashlib.sha256((OUTDIR/"fit_uncapped_continuous_birth.json").read_bytes()).hexdigest()}
    (OUTDIR/"analysis_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"profile":[(p["fixed_initial_retained_count"],p["metrics"]["overall_rmse_log10_rate"]) for p in profile],"holdouts":[(h["held_out_R"],h["held_out_rmse_log10"],h["optimizer"]["success"]) for h in holdouts]},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
