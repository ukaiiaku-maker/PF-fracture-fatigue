#!/usr/bin/env python3
"""Harvest the A-native + 8 PT explicit/monotonic mechanism panel.

This is analysis-only.  Barrier values and rates are evaluated by the same
``MaterialManifest``/``ExpFloorBarrier`` implementation used by refined-v10;
no surrogate constitutive law is introduced.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammainc

from arrhenius_fracture.material_manifest import KB_EV_PER_K, MaterialManifest

COMMON_CYCLES = (1.0, 4.0, 8.0, 16.0, 32.0, 64.0)
COMMON_EXT_UM = (5.0, 10.0, 15.0)


def _sum_prefix(d: dict[str, float], prefix: str) -> float:
    return float(sum(float(v) for k, v in d.items() if k.startswith(prefix)))


def _snapshot_metrics(s: dict, initial_tip_m: float) -> dict[str, float]:
    dg, led, st = s["diagnostics"], s["ledgers"], s["stochastic"]
    gross = float(led.get("mpz.cumulative_gross_source_activity", 0.0))
    returned_slip = _sum_prefix(led, "mpz.cumulative_cancelled_source_slip[")
    physical_return = _sum_prefix(led, "mpz.cumulative_physical_returned_mobile[")
    raw_return = _sum_prefix(led, "mpz.cumulative_returned_mobile[")
    escaped = _sum_prefix(led, "mpz.cumulative_escaped_mobile[")
    mobile, retained = float(dg["mobile_count"]), float(dg["retained_count"])
    emitted = float(led.get("mpz.emitted_total", 0.0))
    return {
        "cycle": float(s["engine_time_s"]) * 1000.0,
        "extension_um": (float(s["geometry_signature"][1]) - initial_tip_m) * 1e6,
        "event_count": int(st["hazard_event_index"]),
        "mobile": mobile,
        "retained": retained,
        "retained_fraction": retained / (mobile + retained) if mobile + retained else 0.0,
        "gross_source_slip": gross,
        "returned_source_slip": returned_slip,
        "net_source_slip": gross - returned_slip,
        "net_over_gross_source_slip": (gross - returned_slip) / gross if gross else 0.0,
        "raw_left_boundary_outflow": raw_return,
        "physical_returned_mobile": physical_return,
        "physical_return_fraction": physical_return / emitted if emitted else 0.0,
        "escaped_mobile": escaped,
        "escape_fraction_of_emitted": escaped / emitted if emitted else 0.0,
        "emitted_mobile_content": emitted,
        "tip_radius_m": float(dg["tip_radius_m"]),
        "internal_stress_Pa": float(dg["sigma_back_Pa"]),
        "shielding_Pa_sqrt_m": float(dg["active_K_shield_Pa_sqrt_m"]),
        "hazard_action": float(st["hazard_action_current"]),
        "hazard_threshold": float(st["hazard_threshold_action"]),
        "B": float(st["B"]),
    }


def _load_history(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Geometry-committed checkpoints are the admissible post-event states.
    keep = [r for r in rows if r["reason"] in {
        "outer_driver_initial_committed_state", "outer_driver_geometry_committed",
        "outer_driver_step_committed", "integrator_return"}]
    out, seen = [], set()
    for r in keep:
        key = (float(r["engine_time_s"]), int(r["stochastic"]["hazard_event_index"]), r["reason"])
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def _nearest(rows: list[dict], key: str, value: float) -> tuple[dict, float]:
    r = min(rows, key=lambda x: abs(float(x[key]) - value))
    return r, float(r[key]) - value


def _terminal_class(run: Path) -> tuple[str, int, int]:
    exit_code = int((run / "exit_code.txt").read_text().strip()) if (run / "exit_code.txt").exists() else -999
    summary = json.loads((run / "high_cycle_summary.json").read_text()) if (run / "high_cycle_summary.json").exists() else {}
    accepted = int(summary.get("mode_counts", {}).get("projective_accepted", 0)) + int(summary.get("mode_counts", {}).get("dmd_accepted", 0))
    if exit_code == 0 and float(summary.get("cycles_consumed", -1)) >= 64.0 - 1e-9:
        return "TERMINAL_EXPLICIT_COMPLETE", exit_code, accepted
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    if "wall" in log.lower() and exit_code != 0:
        return "WALL_LIMIT_NONTERMINAL", exit_code, accepted
    return "NUMERICAL_FAILURE", exit_code, accepted


def harvest_explicit(root: Path, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terminals, states, events = [], [], []
    thresholds: dict[str, list[list[float]]] = {"pos": [], "neg": []}
    for cond, R, dk in (("pos", .1, 16.2), ("neg", -.95, 35.1)):
        for option in registry.option_key:
            run = root / "explicit" / cond / option
            classification, exit_code, accepted = _terminal_class(run)
            hist = _load_history(run / "high_cycle_live_history.jsonl")
            initial_tip = float(hist[0]["geometry_signature"][1])
            metrics = [_snapshot_metrics(s, initial_tip) | {"reason": s["reason"]} for s in hist]
            final = metrics[-1]
            stochastic = hist[0]["stochastic"]
            thresholds[cond].append([float(x) for x in stochastic["hazard_threshold_history"]] + [float(stochastic["hazard_threshold_action"])])
            manifest = json.loads((run / "high_cycle_run_manifest.json").read_text())
            terminals.append({
                "parameter_option": option, "condition": cond, "R": R, "deltaK_MPa_sqrt_m": dk,
                "status": classification, "exit_code": exit_code, "accepted_projective_or_DMD_blocks": accepted,
                "hazard_seed": int(manifest["hazard_seed"]), "cycles": final["cycle"], **{k:v for k,v in final.items() if k != "cycle"},
            })
            # Equal-cycle: nearest physically recorded committed state, never interpolate events.
            for target in COMMON_CYCLES:
                picked, mismatch = _nearest(metrics, "cycle", target)
                states.append({"parameter_option":option,"condition":cond,"coordinate":"equal_cycle_nearest_committed",
                               "target":target,"mismatch":mismatch,**picked})
            shared_events = sorted({m["event_count"] for m in metrics})
            for target in shared_events:
                candidates = [m for m in metrics if m["event_count"] == target]
                picked = candidates[-1]
                states.append({"parameter_option":option,"condition":cond,"coordinate":"equal_event_post_commit",
                               "target":float(target),"mismatch":0.0,**picked})
            for target in COMMON_EXT_UM:
                picked, mismatch = _nearest(metrics, "extension_um", target)
                states.append({"parameter_option":option,"condition":cond,"coordinate":"equal_extension_nearest_post_event",
                               "target":target,"mismatch":mismatch,**picked})
            audit = json.loads((run / "kinetic_tip_cell_audit_v101.json").read_text())
            for i, rec in enumerate(audit["records"], 1):
                events.append({"parameter_option":option,"condition":cond,"event_index":i,
                    "cycle_pre":float(rec.get("time_s",0))-float(rec.get("cycles_consumed",0))/1000,
                    "cycle_post":float(rec.get("time_s",0))*1000,
                    "mobile_pre":rec.get("state_mobile_count_pre"),"mobile_post":rec.get("state_mobile_count"),
                    "retained_pre":rec.get("state_retained_count_pre"),"retained_post":rec.get("state_retained_count"),
                    "sigma_back_pre_Pa":rec.get("state_sigma_back_Pa_pre"),"shielding_pre_Pa_sqrt_m":rec.get("state_active_K_shield_signed_Pa_sqrt_m_pre"),
                    "hazard_action_block":rec.get("physical_hazard_action_block"),"event_advance_m":rec.get("energy_gated_event_advance_m"),
                    "projective_accepted":sum(1 for x in rec.get("coupled_hazard_modes",[]) if x.get("mode") in {"projective_accepted","dmd_accepted"})})
    # CRN contract: complete threshold streams are identical within condition.
    for cond, streams in thresholds.items():
        if any(x != streams[0] for x in streams[1:]):
            raise RuntimeError(f"{cond} threshold/RNG stream contract diverged")
    terminal_df = pd.DataFrame(terminals)
    if len(terminal_df) != 18 or set(terminal_df.status) != {"TERMINAL_EXPLICIT_COMPLETE"}:
        raise RuntimeError("explicit matrix is not 18/18 terminal")
    if terminal_df.accepted_projective_or_DMD_blocks.sum() or sum(x["projective_accepted"] for x in events):
        raise RuntimeError("accepted projective/DMD evidence found in explicit matrix")
    return terminal_df, pd.DataFrame(states), pd.DataFrame(events)


def monotonic(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for option in registry.option_key:
        p=root/"monotonic"/option
        s=pd.read_csv(p/"steps_0300K.csv")
        fired=s[s.n_fire>0]
        if fired.empty: raise RuntimeError(f"no monotonic events: {option}")
        k=fired.KJ_Pa_sqrtm.to_numpy()/1e6; ext=fired.crack_extension_m.to_numpy()*1e6
        def kat(x):
            j=int(np.argmin(abs(ext-x))); return float(k[j])
        out.append({"parameter_option":option,"Kinit_MPa_sqrt_m":float(k[0]),"first_event_K_MPa_sqrt_m":float(k[0]),
            "event_count":int(len(fired)),"final_extension_um":float(ext[-1]),"K_at_5um_MPa_sqrt_m":kat(5),
            "K_at_10um_MPa_sqrt_m":kat(10),"K_at_15um_MPa_sqrt_m":kat(15),"short_R_curve_rise_MPa_sqrt_m":kat(15)-float(k[0]),
            "mobile":float(fired.mpz_mobile_count.iloc[-1]),"retained":float(fired.mpz_retained_count.iloc[-1]),
            "shielding_Pa_sqrt_m":float(fired.mpz_K_shield_Pa_sqrt_m.iloc[-1]),"blunting_local_slip":float(fired.mpz_local_slip_count.iloc[-1])})
    df=pd.DataFrame(out)
    for c in [x for x in df if x!="parameter_option"]:
        df[c+"_panel_range"]=float(df[c].max()-df[c].min())
        mean=float(df[c].mean()); df[c+"_panel_relative_range"]=float((df[c].max()-df[c].min())/abs(mean)) if mean else 0
        df[c+"_panel_cv"]=float(df[c].std(ddof=0)/abs(mean)) if mean else 0
    # predeclared quantitative decision: all key K metrics and extension vary <0.1%.
    key=["Kinit_MPa_sqrt_m","K_at_10um_MPa_sqrt_m","final_extension_um"]
    invariant=all(float(df[c+"_panel_relative_range"].iloc[0])<1e-3 for c in key)
    df["classification"]="A_BACKGROUND_FRACTURE_INVARIANT" if invariant else "A_BACKGROUND_NOT_FRACTURE_INVARIANT"
    return df


def barrier_audit(root: Path, registry: pd.DataFrame, terminal: pd.DataFrame) -> pd.DataFrame:
    rows=[]; phases=np.linspace(0,1,257)[:-1]
    for cond,R,dk in (("pos",.1,16.2),("neg",-.95,35.1)):
      for option in registry.option_key:
        run=root/"explicit"/cond/option
        mat=MaterialManifest.from_csv(run/"selected_material_manifest_v10_2_22.csv")
        final=terminal[(terminal.parameter_option==option)&(terminal.condition==cond)].iloc[0]
        # Actual fixed-K sinusoidal waveform and observed refined-v10 peak/backstress state.
        ratio=(1+R)/2+(1-R)/2*np.cos(2*np.pi*phases)
        peak_sigma=7.06e9
        signed=peak_sigma*ratio-float(final.internal_stress_Pa)
        rho=max(5e12+float(final.retained)/(0.625e-6*1e-6),1.0)
        spacing=1/(2*np.sqrt(rho)); phi=spacing/2.74e-10
        p=mat.peierls.as_surface(mat.emission); t=mat.taylor.as_surface(mat.emission)
        tau_p=0.5773502691896258*np.abs(signed)
        tau_t=0.5773502691896258*np.abs(signed)*phi
        gp=p.values_eV(tau_p,300); gt=t.values_eV(tau_t,300)
        rp=np.asarray(p.rate(tau_p,300)); rt1=np.asarray(t.rate(tau_t,300))
        m=1+max(mat.taylor_corr_scale,0)*max(math.sqrt(rho/max(mat.taylor_corr_rho_c_m2,1))-1,0)
        rt=gammainc(max(m,1),np.minimum(rt1,1e12))
        emit_g=mat.emission.values_eV(np.maximum(signed,0),300); emit_r=mat.emission.rate(np.maximum(signed,0),300)
        cleave_g=mat.cleavage.values_eV(np.maximum(signed,0),300); cleave_r=mat.cleavage.rate(np.maximum(signed,0),300)
        floorp=max(p.floor_min_eV,p.floor_fraction*max(p.G00_eV+p.gT_eV_per_K*(300-p.Tref_K),1e-12)); floort=max(t.floor_min_eV,t.floor_fraction*max(t.G00_eV+t.gT_eV_per_K*(300-t.Tref_K),1e-12))
        pmin,pmax=float(rp.min()),float(rp.max()); tmin,tmax=float(rt1.min()),float(rt1.max())
        for i,ph in enumerate(phases):
          def regime(g,r,floor,rmin,rmax):
            ff=abs(g-floor)<=max(1e-8,0.01*floor); sat=r>=.99*max(rmax,1e-300); inert=r<=max(1e-30,1.01*rmin)
            dyn=not(ff or sat or inert); return ff,sat,inert,dyn
          pf,ps,pi,pdyn=regime(gp[i],rp[i],floorp,pmin,pmax); tf,ts,ti,td=regime(gt[i],rt1[i],floort,tmin,tmax)
          rows.append({"parameter_option":option,"condition":cond,"state":"OBSERVED_TERMINAL_NONVIRGIN","phase":ph,
            "signed_resolved_mobile_stress_Pa":signed[i],"peierls_barrier_eV":gp[i],"peierls_forward_rate_s":rp[i] if signed[i]>=0 else 0,
            "peierls_reverse_rate_s":rp[i] if signed[i]<0 else 0,"signed_transport_velocity_proxy":math.copysign(spacing*rp[i],signed[i]),
            "taylor_barrier_eV":gt[i],"taylor_single_rate_s":rt1[i],"taylor_completion_rate_s":rt[i],
            "emission_barrier_eV":emit_g[i],"emission_rate_s":emit_r[i],"cleavage_barrier_eV":cleave_g[i],"cleavage_rate_s":cleave_r[i],
            "transport_to_retention_timescale_ratio":rt1[i]/max(rp[i],1e-300),"physical_return_eligible":bool(cond=="neg" and signed[i]<0),
            "peierls_floor":pf,"peierls_saturated":ps,"peierls_inert":pi,"peierls_dynamic":pdyn,
            "taylor_floor":tf,"taylor_saturated":ts,"taylor_inert":ti,"taylor_dynamic":td})
    return pd.DataFrame(rows)


def plots(root: Path, terminal: pd.DataFrame, state: pd.DataFrame, barrier: pd.DataFrame) -> None:
    out=root/"figures"; out.mkdir(exist_ok=True)
    labels=terminal[terminal.condition=="pos"].parameter_option.tolist()
    fig,ax=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for cond,ls in (("pos","-"),("neg","--")):
      for option in labels:
        d=state[(state.parameter_option==option)&(state.condition==cond)&(state.coordinate=="equal_event_post_commit")].sort_values("cycle")
        ax[0,0].plot(d.cycle,d.mobile,ls,label=option if cond=="pos" else None); ax[0,1].plot(d.cycle,d.retained,ls)
        ax[1,0].plot(d.cycle,d.physical_returned_mobile,ls); ax[1,1].plot(d.cycle,d.net_source_slip,ls)
    for a,t in zip(ax.flat,["Mobile","Retained","Physical return","Net source slip"]): a.set_title(t);a.set_xlabel("cycles");a.grid(alpha=.25)
    ax[0,0].legend(fontsize=6); fig.tight_layout(); fig.savefig(out/"A8PT_MOBILE_RETAINED_EVOLUTION.png",dpi=180); plt.close(fig)
    agg=barrier.groupby(["parameter_option","condition"])[["peierls_floor","peierls_dynamic","taylor_floor","taylor_dynamic"]].mean().reset_index()
    fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(agg)); ax.plot(x,agg.peierls_dynamic,"o",label="Peierls dynamic fraction");ax.plot(x,agg.taylor_dynamic,"s",label="Taylor dynamic fraction");ax.set_xticks(x,agg.parameter_option,rotation=75,ha="right",fontsize=7);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"A8PT_CYCLIC_BARRIER_RATE_REGIMES.png",dpi=180);plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("runs/A_native_plus_8PT_fatigue_v1"));a=ap.parse_args()
    root=a.root.resolve(); reg=pd.read_csv(root/"A_native_plus_8PT_registry.csv")
    terminal,state,event=harvest_explicit(root,reg)
    mono=monotonic(root,reg); barrier=barrier_audit(root,reg,terminal)
    terminal.to_parquet(root/"A_native_plus_8PT_explicit_cycle_results.parquet",index=False)
    state.to_parquet(root/"A_native_plus_8PT_state_histories.parquet",index=False)
    event.to_csv(root/"A_native_plus_8PT_event_results.csv",index=False)
    mono.to_csv(root/"A_native_plus_8PT_monotonic_confirmation.csv",index=False)
    barrier.to_parquet(root/"A_native_plus_8PT_cyclic_barrier_rate_audit.parquet",index=False)
    roles=pd.read_csv('/private/tmp/taylor-peierls-spatial-coupling-paper-audit/analysis_outputs/taylor_peierls_spatial_coupling_paper_audit/pf_tp_observed_microstructure_roles.csv')
    summary=terminal.merge(reg[["option_key","candidate_id"]],left_on="parameter_option",right_on="option_key",how="left").merge(roles,on="candidate_id",how="left")
    b=barrier.groupby(["parameter_option","condition"])[["peierls_floor","peierls_dynamic","taylor_floor","taylor_dynamic"]].mean().reset_index()
    summary=summary.merge(b,on=["parameter_option","condition"])
    summary.to_csv(root/"A_native_plus_8PT_divergence_summary.csv",index=False)
    plots(root,terminal,state,barrier)
    decision={"schema":"A_native_plus_8PT_predeveloped_mechanism_gate_v1","explicit_matrix_complete":True,
      "explicit_accepted_projective_or_DMD_blocks":0,"common_random_number_contract_verified":True,
      "positive_R_physical_return_max":float(terminal[terminal.condition=="pos"].physical_returned_mobile.max()),
      "monotonic_classification":mono.classification.iloc[0],"developed_panel_authorized":False,
      "remaining_gates":["explicit_vs_accelerated_parity"]}
    (root/"A_native_plus_8PT_predeveloped_gate.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(json.dumps(decision,indent=2))

if __name__=="__main__": main()
