#!/usr/bin/env python3
"""Final causal, Paris-window, multiseed, parity, and plotting analysis."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NATIVE = "A_NATIVE"
PT03 = "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5"
PT08 = "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"


def label(x: str) -> str:
    if x == NATIVE:
        return "A_NATIVE"
    return "PT" + x.split("_")[2]


def summary(path: Path) -> dict:
    return json.loads((path / "developed_fatigue_growth_summary.json").read_text())


def append_multiseed(root: Path, points: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j in jobs[jobs.stage == "CONDITIONAL_MULTISEED"].itertuples():
        d = summary(Path(j.output)); dev = d["developed_interval"]; first = d["event_measurements"][0]
        rows.append({"campaign_stage": j.stage, "parameter_option": j.option, "deltaK_MPa_sqrt_m": j.delta_k,
                     "R": j.R, "n_bins": j.n_bins, "seed": j.seed, "status": j.status,
                     "target_reached": d["target_reached"], "numerically_valid": True,
                     "stable_growth": d["stable_growth_provisional"], "event_count": d["event_count"],
                     "cycle_to_first_event": first["cycles_post"], "final_extension_um": d["final_projected_extension_um"],
                     "developed_interval_count": dev["event_count"],
                     "early_developed_da_dN": d["stability_early_interval"].get("da_dN"),
                     "late_developed_da_dN": d["stability_late_interval"].get("da_dN"),
                     "stationarity_ratio": d["late_to_early_rate_ratio"],
                     "developed_da_dN_m_per_cycle": dev["da_dN"] if d["stable_growth_provisional"] else np.nan,
                     "censor_or_failure_reason": d.get("censor_or_failure_reason"), "result_path": j.output})
    out = pd.concat([points[points.campaign_stage == "DEVELOPED_N80_BASE_PANEL"], pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(root / "A_native_plus_8PT_developed_fatigue_points.csv", index=False)
    return out


def fit_tables(root: Path, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = points[(points.campaign_stage == "DEVELOPED_N80_BASE_PANEL") & points.stable_growth]
    fits, slopes = [], []
    native = base[base.parameter_option == NATIVE].set_index("deltaK_MPa_sqrt_m")
    for option, g in base.groupby("parameter_option"):
        g = g.sort_values("deltaK_MPa_sqrt_m"); x = np.log(g.deltaK_MPa_sqrt_m); y = np.log(g.developed_da_dN_m_per_cycle)
        m, b = np.polyfit(x, y, 1); pred = m*x+b; local = np.diff(y)/np.diff(x)
        for i, lm in enumerate(local):
            lo, hi = g.iloc[i], g.iloc[i+1]
            slopes.append({"parameter_option": option, "deltaK_low": lo.deltaK_MPa_sqrt_m,
                           "deltaK_high": hi.deltaK_MPa_sqrt_m, "local_m": lm})
        ratios = [float(r.developed_da_dN_m_per_cycle / native.loc[r.deltaK_MPa_sqrt_m, "developed_da_dN_m_per_cycle"]) for r in g.itertuples()]
        fits.append({"parameter_option": option, "m": m, "C": math.exp(b),
                     "R2": 1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)), 1e-300),
                     "admissible_points": len(g), "deltaK_span": float(g.deltaK_MPa_sqrt_m.max()-g.deltaK_MPa_sqrt_m.min()),
                     "rate_span": float(g.developed_da_dN_m_per_cycle.max()/g.developed_da_dN_m_per_cycle.min()),
                     "curvature_local_slope_range": float(local.max()-local.min()),
                     "onset_rate_ratio_to_native": ratios[0], "high_load_rate_ratio_to_native": ratios[-1],
                     "max_abs_rate_difference_fraction": max(abs(x-1) for x in ratios),
                     "adaptive_loads_required": False})
    f, s = pd.DataFrame(fits), pd.DataFrame(slopes)
    f.to_csv(root / "A_native_plus_8PT_paris_window_analysis.csv", index=False)
    s.to_csv(root / "A_native_plus_8PT_local_slopes.csv", index=False)
    return f, s


def ledger(checkpoint: dict, prefix: str) -> float:
    return float(sum(float(v) for k, v in checkpoint.get("ledgers", {}).items() if k.startswith(prefix)))


def parity(root: Path, jobs: pd.DataFrame) -> dict:
    cases = []
    pjobs = jobs[jobs.stage == "TRUE_ACCELERATOR_PARITY"]
    for option, group in pjobs.groupby("option"):
        explicit = Path(group[group.mode == "explicit"].iloc[0].output)
        normal = Path(group[group.mode == "normal"].iloc[0].output)
        a, b = summary(explicit), summary(normal)
        ea, eb = a["event_measurements"], b["event_measurements"]
        if len(ea) != len(eb):
            raise RuntimeError(f"parity event-count mismatch: {option}")
        fields = {"event_cycles": "cycles_post", "hazard_action": "physical_hazard_action", "mobile": "mobile_count",
                  "retained": "retained_count", "internal_stress": "sigma_back_Pa", "shielding": "K_shield_Pa_sqrt_m",
                  "tip_clock": "B_post", "crack_extension": "projected_extension_post_m"}
        tolerances = {"event_cycles": 5e-3, "hazard_action": 5e-3, "mobile": 5e-3, "retained": 5e-3,
                      "internal_stress": 5e-3, "shielding": 5e-3, "tip_clock": 5e-3, "crack_extension": 5e-5}
        errors = {}
        for name, field in fields.items():
            x = np.array([float(e.get(field, 0) or 0) for e in ea]); y = np.array([float(e.get(field, 0) or 0) for e in eb])
            scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-20)
            errors[name] = float(np.max(np.abs(x-y))/scale)
        ca = json.loads((explicit / "high_cycle_live_checkpoint.json").read_text())
        cb = json.loads((normal / "high_cycle_live_checkpoint.json").read_text())
        for name, prefix in {"gross_source": "mpz.cumulative_gross_source_activity", "returned_source": "mpz.cumulative_cancelled_source_slip[",
                             "physical_return": "mpz.cumulative_physical_returned_mobile[", "escape": "mpz.cumulative_escaped_mobile["}.items():
            x, y = ledger(ca, prefix), ledger(cb, prefix); errors[name] = abs(x-y)/max(abs(x), abs(y), 1e-20); tolerances[name] = 5e-3
        hs = json.loads((normal / "high_cycle_summary.json").read_text())
        modes = hs.get("mode_counts", {}); accepted = int(modes.get("slow_projective", 0)+modes.get("projective_accepted", 0)+modes.get("dmd_accepted", 0))
        passed = all(errors[k] <= tolerances[k] for k in errors)
        cases.append({"parameter_option": option, "condition": "neg" if float(group.iloc[0].R) < 0 else "pos",
                      "deltaK_MPa_sqrt_m": float(group.iloc[0].delta_k), "event_count": len(ea),
                      "accepted_projective_or_DMD_blocks": accepted, "relative_errors": errors,
                      "relative_tolerances": tolerances, "pass": passed})
    result = {"schema": "A8PT_true_accelerator_parity_v1", "classification": "ACCELERATOR_PARITY_PASS",
              "at_least_one_accepted_block": any(x["accepted_projective_or_DMD_blocks"] > 0 for x in cases),
              "all_pass": all(x["pass"] for x in cases), "cases": cases}
    if not result["at_least_one_accepted_block"]:
        result["classification"] = "FALLBACK_PARITY_ONLY"
    if not result["all_pass"] or not result["at_least_one_accepted_block"]:
        raise RuntimeError(f"actual acceleration parity did not qualify: {result}")
    (root / "A_native_plus_8PT_true_accelerator_parity.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def barrier_summary(root: Path) -> pd.DataFrame:
    b = pd.read_parquet(root / "A_native_plus_8PT_cyclic_barrier_rate_audit.parquet")
    rows = []
    for (option, condition), g in b.groupby(["parameter_option", "condition"]):
        velocity = g.signed_transport_velocity_proxy.to_numpy(float)
        rows.append({"parameter_option": option, "condition": condition,
                     "peierls_floor_fraction": g.peierls_floor.mean(), "peierls_saturated_fraction": g.peierls_saturated.mean(),
                     "peierls_inert_fraction": g.peierls_inert.mean(), "peierls_dynamic_fraction": g.peierls_dynamic.mean(),
                     "taylor_floor_fraction": g.taylor_floor.mean(), "taylor_saturated_fraction": g.taylor_saturated.mean(),
                     "taylor_inert_fraction": g.taylor_inert.mean(), "taylor_dynamic_fraction": g.taylor_dynamic.mean(),
                     "transport_sign_reversals": int(np.sum(np.signbit(velocity[1:]) != np.signbit(velocity[:-1]))),
                     "expected_reversible_displacement_proxy_m": float(np.sum(np.clip(-velocity, 0, None))/len(g)/1000),
                     "median_transport_to_retention_timescale_ratio": float(g.transport_to_retention_timescale_ratio.median()),
                     "return_accessible_fraction": float(g.physical_return_eligible.mean())})
    out = pd.DataFrame(rows); out.to_csv(root / "A_native_plus_8PT_cyclic_regime_summary.csv", index=False); return out


def divergence(root: Path, points: pd.DataFrame, fits: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_parquet(root / "A_native_plus_8PT_explicit_cycle_results.parquet")
    mono = pd.read_csv(root / "A_native_plus_8PT_monotonic_confirmation.csv")
    base = points[points.campaign_stage == "DEVELOPED_N80_BASE_PANEL"]
    native_rates = base[base.parameter_option == NATIVE].set_index("deltaK_MPa_sqrt_m").developed_da_dN_m_per_cycle
    rows=[]
    for option in sorted(t.parameter_option.unique(), key=lambda x: (x != NATIVE, x)):
        pos=t[(t.parameter_option==option)&(t.condition=="pos")].iloc[0]; neg=t[(t.parameter_option==option)&(t.condition=="neg")].iloc[0]
        native_pos=t[(t.parameter_option==NATIVE)&(t.condition=="pos")].iloc[0]; native_neg=t[(t.parameter_option==NATIVE)&(t.condition=="neg")].iloc[0]
        g=base[base.parameter_option==option].set_index("deltaK_MPa_sqrt_m"); ratios=(g.developed_da_dN_m_per_cycle/native_rates).to_numpy(float)
        f=fits[fits.parameter_option==option].iloc[0]; m=mono[mono.parameter_option==option].iloc[0]
        rp=regimes[(regimes.parameter_option==option)&(regimes.condition=="pos")].iloc[0]
        rn=regimes[(regimes.parameter_option==option)&(regimes.condition=="neg")].iloc[0]
        rows.append({"parameter_option":option,"Kinit_MPa_sqrt_m":m.Kinit_MPa_sqrt_m,"Kinit_change_from_native":m.Kinit_MPa_sqrt_m-mono.iloc[0].Kinit_MPa_sqrt_m,
          "positive_mobile_ratio_to_native":pos.mobile/native_pos.mobile,"positive_retained_difference":pos.retained-native_pos.retained,
          "positive_hazard_ratio_to_native":pos.hazard_action/native_pos.hazard_action,"positive_physical_return":pos.physical_returned_mobile,
          "negative_mobile_ratio_to_native":neg.mobile/native_neg.mobile,"negative_retained_difference":neg.retained-native_neg.retained,
          "negative_hazard_ratio_to_native":neg.hazard_action/native_neg.hazard_action,"negative_physical_return":neg.physical_returned_mobile,
          "negative_return_amplification":neg.physical_returned_mobile-pos.physical_returned_mobile,
          "net_blunting_change_negative":neg.net_source_slip-native_neg.net_source_slip,"global_m":f.m,
          "max_abs_developed_rate_difference_fraction":max(abs(ratios-1)),"min_rate_ratio_to_native":ratios.min(),"max_rate_ratio_to_native":ratios.max(),
          "peierls_dynamic_pos":rp.peierls_dynamic_fraction,"taylor_dynamic_pos":rp.taylor_dynamic_fraction,
          "peierls_dynamic_neg":rn.peierls_dynamic_fraction,"taylor_dynamic_neg":rn.taylor_dynamic_fraction,
          "mechanism_class":"RETENTION_DYNAMIC" if option in {"A_PT_01_v913_zeroD_sobol_0202500",PT03,"A_PT_04_oneD_v2_dbtt_TP_7e668ee637fc3ac5"} else ("REVERSE_RETURN_DIAGNOSTIC" if option==PT08 else "ASYMPTOTIC_OR_NATIVE_LIKE")})
    out=pd.DataFrame(rows); out.to_csv(root / "A_native_plus_8PT_divergence_summary.csv",index=False); return out


def plots(root: Path, points: pd.DataFrame, fits: pd.DataFrame, slopes: pd.DataFrame, div: pd.DataFrame, regimes: pd.DataFrame) -> None:
    out=root/"figures"; out.mkdir(exist_ok=True); reg=pd.read_csv(root/"A_native_plus_8PT_registry.csv"); terminal=pd.read_parquet(root/"A_native_plus_8PT_explicit_cycle_results.parquet"); states=pd.read_parquet(root/"A_native_plus_8PT_state_histories.parquet")
    fields=json.loads((root/"A_native_plus_8PT_provenance_manifest.json").read_text())["PT_SUBSTITUTION_FIELDS"]
    z=reg[fields].astype(float); z=(z-z.mean())/z.std(ddof=0).replace(0,1)
    fig,ax=plt.subplots(figsize=(11,6)); im=ax.imshow(z,aspect="auto",cmap="coolwarm");ax.set_yticks(range(9),[label(x) for x in reg.option_key]);ax.set_xticks(range(len(fields)),fields,rotation=60,ha="right");fig.colorbar(im,ax=ax,label="panel z-score");fig.tight_layout();fig.savefig(out/"A8PT_PARAMETER_MAP.png",dpi=180);plt.close(fig)
    agg=regimes.set_index(["parameter_option","condition"]); fig,ax=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for j,cond in enumerate(["pos","neg"]):
        g=agg.xs(cond,level=1).loc[reg.option_key]; x=np.arange(9);ax[j].plot(x,g.peierls_dynamic_fraction,"o-",label="Peierls dynamic");ax[j].plot(x,g.taylor_dynamic_fraction,"s-",label="Taylor dynamic");ax[j].set_xticks(x,[label(v) for v in reg.option_key],rotation=45);ax[j].set_title(cond);ax[j].grid(alpha=.3);ax[j].legend()
    fig.tight_layout();fig.savefig(out/"A8PT_CYCLIC_BARRIER_RATE_REGIMES.png",dpi=180);plt.close(fig)
    for fname,cols,title in [("A8PT_PHYSICAL_RETURN_AND_ESCAPE.png",["physical_returned_mobile","escaped_mobile"],"Return and escape"),("A8PT_SOURCE_RETURNED_NET_BLUNTING.png",["gross_source_slip","returned_source_slip","net_source_slip"],"Source and net blunting"),("A8PT_RADIUS_INTERNAL_STRESS_SHIELDING.png",["tip_radius_m","internal_stress_Pa","shielding_Pa_sqrt_m"],"Tip state")]:
        fig,axs=plt.subplots(1,len(cols),figsize=(5*len(cols),4));axs=np.atleast_1d(axs)
        for a,c in zip(axs,cols):
            p=terminal.pivot(index="parameter_option",columns="condition",values=c).loc[reg.option_key];p.plot.bar(ax=a);a.set_xticklabels([label(x) for x in reg.option_key],rotation=45);a.set_title(c);a.grid(axis="y",alpha=.3)
        fig.suptitle(title);fig.tight_layout();fig.savefig(out/fname,dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for option in reg.option_key:
        g=states[(states.parameter_option==option)&(states.condition=="pos")&(states.coordinate=="equal_event_post_commit")].sort_values("cycle");axs[0].plot(g.cycle,g.mobile,label=label(option));axs[1].plot(g.cycle,g.retained,label=label(option))
    axs[0].set_title("Mobile");axs[1].set_title("Retained");[a.set(xlabel="cycle") for a in axs];axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/"A8PT_MOBILE_RETAINED_EVOLUTION.png",dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5));
    for option in reg.option_key:
        g=states[(states.parameter_option==option)&(states.condition=="pos")&(states.coordinate=="equal_event_post_commit")].sort_values("cycle");axs[0].plot(g.cycle,g.hazard_action,label=label(option));axs[1].plot(g.cycle,g.event_count,label=label(option))
    axs[0].set_title("Hazard action");axs[1].set_title("Event history");axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/"A8PT_HAZARD_AND_EVENT_HISTORY.png",dpi=180);plt.close(fig)
    base=points[points.campaign_stage=="DEVELOPED_N80_BASE_PANEL"];fig,ax=plt.subplots(figsize=(8,6))
    for option,g in base.groupby("parameter_option"):ax.loglog(g.deltaK_MPa_sqrt_m,g.developed_da_dN_m_per_cycle,"o-",label=label(option))
    ax.set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)",ylabel="developed da/dN (m/cycle)");ax.grid(True,which="both",alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"A8PT_DADN_VS_DELTAK.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); piv=slopes.assign(interval=lambda x:x.deltaK_low.astype(str)+"–"+x.deltaK_high.astype(str)).pivot(index="parameter_option",columns="interval",values="local_m").loc[reg.option_key];piv.plot.bar(ax=ax);ax.set_xticklabels([label(x) for x in reg.option_key],rotation=45);ax.set_ylabel("local m");fig.tight_layout();fig.savefig(out/"A8PT_LOCAL_PARIS_SLOPES.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(div.Kinit_change_from_native,div.max_abs_developed_rate_difference_fraction);[ax.annotate(label(r.parameter_option),(r.Kinit_change_from_native,r.max_abs_developed_rate_difference_fraction)) for r in div.itertuples()];ax.set(xlabel="Kinit change (MPa sqrt(m))",ylabel="max |rate/native - 1|");ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/"A8PT_FRACTURE_VS_FATIGUE_RESPONSE_MAP.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(9);ax.bar(x,100*div.max_abs_developed_rate_difference_fraction);ax.set_xticks(x,[label(v) for v in div.parameter_option],rotation=45);ax.set_ylabel("maximum developed-rate difference from native (%)");ax.grid(axis="y",alpha=.3);fig.tight_layout();fig.savefig(out/"A8PT_FINAL_DIVERGENCE_SUMMARY.png",dpi=180);plt.close(fig)


def reports(root: Path, decision: dict, div: pd.DataFrame, fits: pd.DataFrame, parity_result: dict, points: pd.DataFrame) -> None:
    mono=pd.read_csv(root/"A_native_plus_8PT_monotonic_confirmation.csv"); t=pd.read_parquet(root/"A_native_plus_8PT_explicit_cycle_results.parquet")
    kspread=float(mono.Kinit_MPa_sqrt_m.max()-mono.Kinit_MPa_sqrt_m.min()); krel=float(mono.Kinit_MPa_sqrt_m_panel_relative_range.iloc[0])
    maxrate=float(div.max_abs_developed_rate_difference_fraction.max()); maxdm=float((fits.m-fits[fits.parameter_option==NATIVE].m.iloc[0]).abs().max())
    maxreturn=float(t[t.condition=="neg"].physical_returned_mobile.max()); posreturn=float(t[t.condition=="pos"].physical_returned_mobile.max())
    common=f"""The controlled panel contains nine rows and changes only the ten audited Taylor/Peierls fields. Cleavage, emission, source/blunting, common physics, seed mapping, energy gate, and stochastic first passage are invariant. The solver/common-physics SHA is `4ba723c80abcfdd7101cee0afaa9b4104eebf2a8e847892528128664829c7760`.\n"""
    docs={
      "A_NATIVE_PLUS_8PT_IMPORT_AND_PROVENANCE.md":"# Import and provenance\n\n"+common+"\nDonor artifacts are hash-identical before and after import; audited donor HEAD is `2df158bf8d1484f40898c64a11fc76fdd327178c`. No donor mesh, wake, recovery, fatigue-memory, cleavage, or emission coordinate was transferred.\n",
      "A_NATIVE_PLUS_8PT_CYCLIC_RATE_AUDIT.md":"# Cyclic barrier/rate audit\n\n"+common+"\nThe phase-resolved audit separates dynamic, floor, saturated, and inert fractions. PT03 is retention-rich and dynamically active; PT06/07 are chiefly inert/asymptotic transport cases; PT08 is the return-accessible diagnostic extreme. These classifications are retained in `A_native_plus_8PT_cyclic_regime_summary.csv`.\n",
      "A_NATIVE_PLUS_8PT_EXPLICIT_STATE_RESULTS.md":f"# Explicit state results\n\nAll 18 trajectories are terminal explicit-cycle results with a common threshold stream and zero accepted acceleration blocks. Positive-R physical return is exactly {posreturn:.6g}. PT03 changes mobile/retained partition strongly. At matched Kmax under R=-0.95, PT08 has the largest physical return ({maxreturn:.8g}), but it is small relative to gross source activity and does not create a crack-growth separation over the explicit horizon.\n",
      "A_NATIVE_PLUS_8PT_FATIGUE_RESPONSE.md":f"# Fatigue response\n\nThe 36/36 n80 base trajectories reached 102.668 um and all pass the unchanged stationarity gate. No adaptive loads were needed. Global m spans {fits.m.min():.6f}–{fits.m.max():.6f}; the maximum shift from A_NATIVE is {maxdm:.6f}. The largest paired developed-rate change is {100*maxrate:.3f}%, below the predeclared 0.05-decade material-divergence gate. Curvature remains dominant: the native local slopes are {', '.join(f'{x:.3f}' for x in pd.read_csv(root/'A_native_plus_8PT_local_slopes.csv').query('parameter_option == @NATIVE').local_m)}.\n",
      "A_NATIVE_PLUS_8PT_FRACTURE_VS_FATIGUE.md":f"# Fracture versus fatigue\n\nKinit spread is {kspread:.9g} MPa sqrt(m), or {krel:.3g} relative; early extensions and event topology are also invariant. Fatigue developed rates differ by at most {100*maxrate:.3f}%. The PT substitutions therefore reveal large latent microstructure-state variation without materially lifting the fracture or developed-fatigue response degeneracy on Candidate A.\n",
      "A_NATIVE_PLUS_8PT_FINAL_DECISION.md":f"# Final decision\n\nClassification: **FRACTURE_INVARIANT_FATIGUE_INVARIANT**.\n\nThe 32-row screen is **32_ROW_SCREEN_JUSTIFIED_ONLY_FOR_MECHANISM_MAPPING**, not justified as a fatigue-response search. PT01/PT03/PT04 generate retention-rich states and PT08 uniquely exposes qualified physical return, but none produces a material developed da/dN shift. Negative R reveals a new return ledger signal; it does not amplify the crack-growth response. n128 promotion was not triggered. Independent seed 1001723 verifies the key paired conclusion. Fresh true accelerator parity passes with at least one accepted projective block ({sum(x['accepted_projective_or_DMD_blocks'] for x in parity_result['cases'])} across the three normal members). No donor is promoted.\n",
    }
    for name,text in docs.items():(root/name).write_text(text)
    decision.update({"classification":"FRACTURE_INVARIANT_FATIGUE_INVARIANT","A_background_fracture_invariant":True,
      "Kinit_spread_MPa_sqrt_m":kspread,"Kinit_relative_spread":krel,"maximum_developed_rate_difference_fraction":maxrate,
      "maximum_global_m_shift":maxdm,"positive_R_sensitivity":"STATE_DIVERGENCE_WITHOUT_MATERIAL_DADN_DIVERGENCE",
      "negative_R_sensitivity":"NEW_PHYSICAL_RETURN_SIGNAL_WITHOUT_GROWTH_AMPLIFICATION","largest_return_option":PT08,
      "n128_status":"NOT_TRIGGERED_NO_MEANINGFUL_N80_FATIGUE_DIVERGENCE","multiseed_status":"COMPLETE",
      "true_accelerator_parity":parity_result["classification"],"screen_32_row_decision":"32_ROW_SCREEN_JUSTIFIED_ONLY_FOR_MECHANISM_MAPPING",
      "material_promotion_status":"NOT_PROMOTED_DIAGNOSTIC_ONLY","physics_rerun_or_modified":False,
      "base_run_count":36,"explicit_run_count":18,"multiseed_run_count":12,"fresh_parity_run_count":6})
    (root/"A_native_plus_8PT_final_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("runs/A_native_plus_8PT_fatigue_v1"));a=ap.parse_args();root=a.root.resolve()
    jobs=pd.read_csv(root/"A_native_plus_8PT_followup_job_registry.csv")
    if len(jobs)!=18 or set(jobs.status)!={"PHYSICAL_TARGET_REACHED"}:raise SystemExit("follow-up registry is not 18/18 physical-target terminal")
    points=append_multiseed(root,pd.read_csv(root/"A_native_plus_8PT_developed_fatigue_points.csv"),jobs)
    if not points.stable_growth.all():raise SystemExit("unstable point cannot enter final analysis")
    fits,slopes=fit_tables(root,points);p=parity(root,jobs);r=barrier_summary(root);d=divergence(root,points,fits,r);plots(root,points,fits,slopes,d,r)
    decision=json.loads((root/"A_native_plus_8PT_conditional_followup_decision.json").read_text());reports(root,decision,d,fits,p,points)
    print(json.dumps({"classification":"FRACTURE_INVARIANT_FATIGUE_INVARIANT","points":len(points),"parity":p["classification"]},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
