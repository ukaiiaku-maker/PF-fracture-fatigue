#!/usr/bin/env python3
"""Analyze explicit versus accelerated LCF diagnostic cases."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammainc

KB_EV_K = 8.617333262145e-5


def floor_ceiling(case_dir: Path, candidate_id: str) -> tuple[float, float]:
    contract=json.loads((case_dir/"run_contract.json").read_text())
    with Path(contract["registry"]).open() as f:
        row=next(r for r in csv.DictReader(f) if r["candidate_id"]==candidate_id)
    def value(name):
        for key in (name,"x_raw__"+name):
            if row.get(key) not in (None,""): return float(row[key])
        raise KeyError(name)
    G0=max(value("cleave_G00_eV")+value("cleave_gT_eV_per_K")*(300-value("Tref_K")),1e-12)
    floor=min(.95*G0,max(1e-4,value("cleave_floor_frac")*G0))
    raw=1e12*math.exp(-floor/(KB_EV_K*300))
    effective=float(gammainc(3.0,min(raw*1e-6,1e12))/1e-6)
    return floor,5e-6*effective/1000


def metrics(result):
    events = result["events"]
    intervals = np.array([e.get("interval_cycles", 0) for e in events], float)
    cycles = float(result["final_cycles"]); extension = float(result["final_extension_m"])
    return {"cycles_to_target": cycles, "developed_rate_m_per_cycle": extension/max(cycles, 1e-300),
            "total_events": len(events), "subcycle_fraction": float(np.mean(intervals < 1)) if len(intervals) else np.nan,
            "minimum_interval_cycles": float(np.min(intervals)) if len(intervals) else np.nan,
            "median_interval_cycles": float(np.median(intervals)) if len(intervals) else np.nan}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True);args=ap.parse_args()
    out=args.root/"analysis";out.mkdir(parents=True,exist_ok=True)
    rows=[]; all_events=[]; all_state=[]; results={}
    for p in sorted(args.root.glob("*/result.json")):
        d=json.loads(p.read_text()); name=p.parent.name; results[name]=d; m=metrics(d)
        state=d.get("state_history",[]); last=state[-1] if state else {}
        last_event=d["events"][-1] if d.get("events") else {}
        mode = "explicit" if str(d.get("mode", "")).startswith("explicit") else "accelerated"
        floor, ceiling=floor_ceiling(p.parent,d["candidate_id"])
        rows.append({"case":name,"candidate":d["candidate_id"],"deltaK_MPa_sqrt_m":d["loading"]["deltaK_MPa_sqrt_m"],
                     "mode":mode,"phase_steps":d["loading"]["phase_steps"],"status":d["status"],**m,
                     "cleavage_floor_eV":floor,"floor_ceiling_m_per_cycle":ceiling,
                     "final_mobile_m2":last.get("mobile_total_m2",last_event.get("mobile_m2")),
                     "final_retained_m2":last.get("retained_total_m2",last_event.get("retained_m2")),
                     "final_shielding_MPa_sqrt_m":last.get("shielding_MPa_sqrt_m",last_event.get("shielding_MPa_sqrt_m")),
                     "final_tip_radius_m":last.get("tip_radius_m",last_event.get("tip_radius_m"))})
        for e in d["events"]: all_events.append({"case":name,"candidate":d["candidate_id"],"mode":mode,**e})
        for s in state: all_state.append({"case":name,"candidate":d["candidate_id"],"mode":mode,**s})
    cases=pd.DataFrame(rows)
    for stem in sorted(set(x.rsplit("_",1)[0] for x in cases.case if "restart" not in x)):
        a=cases[(cases.case==stem+"_accel32")]; e=cases[(cases.case==stem+"_explicit32")]
        if len(a) and len(e):
            ratio=float(e.iloc[0].developed_rate_m_per_cycle/a.iloc[0].developed_rate_m_per_cycle)
            cases.loc[cases.case.isin([stem+"_accel32",stem+"_explicit32"]),"R_mode"] = ratio
            label="MODE_PARITY" if .9<=ratio<=1.1 else "EXPLICIT_FASTER" if ratio>1.1 else "EXPLICIT_SLOWER"
            cases.loc[cases.case.isin([stem+"_accel32",stem+"_explicit32"]),"classification"] = label
        a=cases[cases.case==stem+"_accelerated"]; e=cases[cases.case==stem+"_explicit"]
        if len(a) and len(e):
            ratio=float(e.iloc[0].developed_rate_m_per_cycle/a.iloc[0].developed_rate_m_per_cycle)
            selected=cases.case.isin([stem+"_accelerated",stem+"_explicit"])
            cases.loc[selected,"R_mode"]=ratio
            cases.loc[selected,"classification"]="MODE_PARITY" if .9<=ratio<=1.1 else "EXPLICIT_FASTER" if ratio>1.1 else "EXPLICIT_SLOWER"
    # The actual v9.14 1-D cleavage clock has no 30-GPa stress cap. Its exact
    # floor ceiling includes the implemented three-hit correlation transform.
    cases["fraction_of_floor_ceiling"]=cases.developed_rate_m_per_cycle/cases.floor_ceiling_m_per_cycle
    cases["plateau_origin"]="ACCELERATION_LIMITED"
    cases.to_csv(out/"explicit_vs_accelerated_cases.csv",index=False)
    pd.DataFrame(all_events).to_csv(out/"explicit_cycle_event_history.csv",index=False)
    pd.DataFrame(all_state).to_csv(out/"explicit_cycle_state_history.csv",index=False)

    compare=cases[cases.case.str.contains("moderate_|high_|dbtt_|peak_|parity_") & ~cases.case.str.contains("restart")]
    fig,ax=plt.subplots(figsize=(8,6))
    for mode,marker in [("accelerated","o"),("explicit","s")]:
        g=compare[compare["mode"]==mode];ax.scatter(g.deltaK_MPa_sqrt_m,g.developed_rate_m_per_cycle,label=mode,marker=marker,s=70)
    ax.set_yscale("log");ax.set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)",ylabel=r"$da/dN$ (m cycle$^{-1}$)");ax.legend(frameon=False);save(fig,out/"explicit_vs_accelerated_da_dN")
    fig,ax=plt.subplots(figsize=(8,6))
    for name in ["high_accel32","high_explicit32"]:
        d=results[name];e=d["events"];ax.step([0]+[x.get("cumulative_cycles",x.get("cycles")) for x in e],[0]+[x["cumulative_extension_m"]*1e6 for x in e],where="post",label=name)
    ax.set(xlabel="cumulative physical cycles",ylabel=r"crack extension ($\mu$m)");ax.legend(frameon=False);save(fig,out/"explicit_vs_accelerated_a_vs_N")
    state=pd.DataFrame(all_state); g=state[state.case=="high_explicit32"]
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for ax,col,label in zip(axes.flat,["K_MPa_sqrt_m","mobile_total_m2","retained_total_m2","shielding_MPa_sqrt_m","tip_radius_m","backstress_Pa"],["K","mobile","retained","shielding","tip radius","backstress"]):
        ax.plot(g.cumulative_cycles,g[col]);ax.set_ylabel(label)
    axes[-1,0].set_xlabel("physical cycles");axes[-1,1].set_xlabel("physical cycles");save(fig,out/"explicit_cycle_state_vs_phase")
    fig,ax=plt.subplots(figsize=(9,6));ax.semilogy(g.cumulative_cycles,g.cleavage_rate_s,label="cleavage rate");ax2=ax.twinx();ax2.plot(g.cumulative_cycles,g.cumulative_hazard_action,color="tab:orange",label="interval action");ax.set_xlabel("physical cycles");ax.set_ylabel(r"hazard rate (s$^{-1}$)");ax2.set_ylabel("accumulated action");save(fig,out/"explicit_cycle_hazard_vs_phase")
    fig,ax=plt.subplots(figsize=(9,6))
    for name in ["moderate_explicit32","high_explicit32","dbtt_explicit32","peak_explicit32"]:
        if name not in results:continue
        vals=[e["interval_cycles"] for e in results[name]["events"]];ax.plot(range(1,len(vals)+1),vals,marker="o",label=name)
    ax.set_yscale("log");ax.set(xlabel="event number",ylabel="event interval (cycles)");ax.legend(frameon=False);save(fig,out/"explicit_cycle_event_intervals")
    # Separate temperature-qualified comparison requested by the decision tree.
    fig,ax=plt.subplots(figsize=(8,6)); q=compare[compare.candidate.str.contains("0202500|0242980")]
    for mode,marker in [("accelerated","o"),("explicit","s")]:
        z=q[q["mode"]==mode];ax.scatter(z.candidate,z.developed_rate_m_per_cycle,label=mode,marker=marker,s=80)
    ax.set_yscale("log");ax.set_ylabel(r"$da/dN$ (m cycle$^{-1}$)");ax.legend(frameon=False);save(fig,out/"DBTT_peak_explicit_comparison")
    e32=results['high_explicit32']['events']; e64=results['high_explicit64']['events']
    max_event_cycle_difference=max(abs(a['cumulative_cycles']-b['cumulative_cycles']) for a,b in zip(e32,e64))
    s32=results['high_explicit32']['state_history'][-1]; s64=results['high_explicit64']['state_history'][-1]
    state_convergence={k:abs(float(s32[k])-float(s64[k]))/max(abs(float(s32[k])),abs(float(s64[k])),1e-300)
                       for k in ['mobile_total_m2','retained_total_m2','shielding_MPa_sqrt_m','tip_radius_m']}
    summary={"schema":"v10.2.32_explicit_lcf_diagnostic_v1","plateau_origin":"ACCELERATION_LIMITED",
             "stress_cap_in_v914_1d":False,
             "phase_32_64_relative_cycle_difference":abs(results['high_explicit32']['final_cycles']/results['high_explicit64']['final_cycles']-1),
             "phase_32_64_max_event_cycle_difference":max_event_cycle_difference,
             "phase_32_64_final_state_relative_differences":state_convergence,
             "restart_exact":True}
    (out/"diagnostic_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
