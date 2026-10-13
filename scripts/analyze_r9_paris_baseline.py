#!/usr/bin/env python3
"""Analyze common-extension R9 fatigue windows and local Paris secants."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _event_at(events, target_m):
    return next(row for row in events if float(row["extension_m"]) >= float(target_m))


def _window(events, low_m, high_m, cycle_key="cycles"):
    low = _event_at(events, low_m); high = _event_at(events, high_m)
    da = float(high["extension_m"])-float(low["extension_m"])
    dN = float(high[cycle_key])-float(low[cycle_key])
    return low, high, da/dN


def _bootstrap(events, low, high, replicates=2000, block=3, seed=913):
    selected = [row for row in events if low["event_index"] < row["event_index"] <= high["event_index"]]
    prior = {int(row["event_index"]): row for row in events}
    pairs = []
    for row in selected:
        before = prior[int(row["event_index"])-1]
        pairs.append((float(row["event_advance_m"]), float(row["cycles"])-float(before["cycles"])))
    if len(pairs) < 2:
        return math.nan, math.nan
    array = np.asarray(pairs); rng = np.random.default_rng(seed)
    values = []
    starts = np.arange(len(array))
    blocks = int(math.ceil(len(array)/block))
    for _ in range(replicates):
        indices = np.concatenate([np.arange(s, s+block) % len(array)
                                  for s in rng.choice(starts, size=blocks)])[:len(array)]
        sample = array[indices]
        values.append(float(np.sum(sample[:, 0])/np.sum(sample[:, 1])))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, action="append", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(argv); a.out_root.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(path.read_text()) for path in a.case]
    rows=[]; states=[]; histories=[]
    for case in sorted(cases, key=lambda x:x["DeltaK_MPa_sqrt_m"]):
        events=[{"event_index":0,"cycles":0.,"extension_m":0.,"event_advance_m":0.},*case["events"]]
        opening_action=float(case["overlap"]["mean_hazard_action"])
        opening_cycle=0.
        for row in events[1:]:
            opening_cycle += float(row["threshold_action"])/opening_action
            row["opening_cycles"] = opening_cycle
        events[0]["opening_cycles"]=0.
        w1a,w1b,w1=_window(events,20e-6,60e-6)
        w2a,w2b,w2=_window(events,60e-6,100e-6)
        wa,wb,rate=_window(events,20e-6,100e-6)
        _,_,opening_rate=_window(events,20e-6,100e-6,"opening_cycles")
        ci_low,ci_high=_bootstrap(events,wa,wb,seed=913+int(round(case["Kmax_MPa_sqrt_m"]*1000)))
        rows.append({
            "candidate_id":case["candidate_id"],"seed":case["seed"],
            "Kmax_MPa_sqrt_m":case["Kmax_MPa_sqrt_m"],"DeltaK_MPa_sqrt_m":case["DeltaK_MPa_sqrt_m"],
            "status":case["status"],"event_count":len(case["events"]),
            "window_low_event":wa["event_index"],"window_high_event":wb["event_index"],
            "window_low_extension_m":wa["extension_m"],"window_high_extension_m":wb["extension_m"],
            "developed_da_dN_m_per_cycle":rate,"bootstrap_95_low":ci_low,"bootstrap_95_high":ci_high,
            "window_20_60_da_dN":w1,"window_60_100_da_dN":w2,
            "window_rate_ratio_second_over_first":w2/w1,
            "opening_reference_da_dN_m_per_cycle":opening_rate,
            "developed_over_opening":rate/opening_rate,
            "opening_action_per_cycle":opening_action,
        })
        last=case["events"][-1]
        states.append({
            "Kmax_MPa_sqrt_m":case["Kmax_MPa_sqrt_m"],"DeltaK_MPa_sqrt_m":case["DeltaK_MPa_sqrt_m"],
            "tip_radius_m":last["tip_radius_m"],"K_shield_Pa_sqrt_m":last["K_shield_Pa_sqrt_m"],
            "mobile_count":last["mobile_count"],"retained_count":last["retained_count"],
            "gross_source_activity":last["gross_source_activity"],
            "gross_return_activity":last["gross_return_activity"],
            "source_multiplicity":last["persistent_site_multiplicity_per_system"],
            "opening_action_per_cycle":opening_action,
            "mean_event_phase":float(np.mean([x["phase"] for x in case["events"]])),
        })
        for row in case["history"]:
            histories.append({"Kmax_MPa_sqrt_m":case["Kmax_MPa_sqrt_m"],**row})
    frame=pd.DataFrame(rows).sort_values("DeltaK_MPa_sqrt_m").reset_index(drop=True)
    slopes=[]
    for i in range(len(frame)-1):
        left,right=frame.iloc[i],frame.iloc[i+1]
        m=math.log(right.developed_da_dN_m_per_cycle/left.developed_da_dN_m_per_cycle)/math.log(right.DeltaK_MPa_sqrt_m/left.DeltaK_MPa_sqrt_m)
        mo=math.log(right.opening_reference_da_dN_m_per_cycle/left.opening_reference_da_dN_m_per_cycle)/math.log(right.DeltaK_MPa_sqrt_m/left.DeltaK_MPa_sqrt_m)
        slopes.append({"DeltaK_low":left.DeltaK_MPa_sqrt_m,"DeltaK_high":right.DeltaK_MPa_sqrt_m,
                       "rate_low":left.developed_da_dN_m_per_cycle,"rate_high":right.developed_da_dN_m_per_cycle,
                       "local_slope":m,"opening_local_slope":mo,
                       "state_correction_log_slope":m-mo,"within_m4_band":3.4<=m<=4.6,
                       "rate_width_decades":math.log10(right.developed_da_dN_m_per_cycle/left.developed_da_dN_m_per_cycle)})
    slope=pd.DataFrame(slopes)
    best={"rate_width_decades":0.,"start":None,"stop":None}
    for i in range(len(slope)):
        if not slope.loc[i,"within_m4_band"]: continue
        width=0.
        for j in range(i,len(slope)):
            if not slope.loc[j,"within_m4_band"]: break
            width+=float(slope.loc[j,"rate_width_decades"])
            if width>best["rate_width_decades"]:
                best={"rate_width_decades":width,"start":i,"stop":j,
                      "DeltaK_low":float(slope.loc[i,"DeltaK_low"]),
                      "DeltaK_high":float(slope.loc[j,"DeltaK_high"])}
    frame.to_csv(a.out_root/"growth_window_table.csv",index=False)
    slope.to_csv(a.out_root/"local_slope_table.csv",index=False)
    pd.DataFrame(states).to_csv(a.out_root/"phase_state_table.csv",index=False)
    pd.DataFrame(histories).to_csv(a.out_root/"a_N_history.csv",index=False)
    (a.out_root/"slope_width_summary.json").write_text(json.dumps({"largest_connected_m4_interval":best,
        "band":[3.4,4.6],"target_rate_width_decades":2.0,"window_rule":"first committed event at or above 20 and 100 um; common event rewards under CRN"},indent=2,sort_keys=True)+"\n")
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    for K,g in pd.DataFrame(histories).groupby("Kmax_MPa_sqrt_m"):
        ax.plot(g.cycles,g.extension_m*1e6,label=f"{K:.3g}")
    ax.set_xlabel("Cycles, N");ax.set_ylabel("Committed crack extension, μm");ax.legend(title="Kmax (MPa√m)",fontsize=8);fig.tight_layout();fig.savefig(a.out_root/"a_N.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,4.4));x=frame.DeltaK_MPa_sqrt_m
    y=frame.developed_da_dN_m_per_cycle
    ax.errorbar(x,y,yerr=np.vstack([y-frame.bootstrap_95_low,frame.bootstrap_95_high-y]),fmt="o-",capsize=3,label="state evolved")
    ax.plot(x,frame.opening_reference_da_dN_m_per_cycle,"s--",label="opening only")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("ΔK (MPa√m)");ax.set_ylabel("da/dN (m/cycle)");ax.legend();fig.tight_layout();fig.savefig(a.out_root/"fatigue_curve.png",dpi=220);plt.close(fig)
    return 0


if __name__=="__main__": raise SystemExit(main())
