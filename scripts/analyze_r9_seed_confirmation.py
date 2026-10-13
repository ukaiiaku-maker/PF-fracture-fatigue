#!/usr/bin/env python3
"""Independent-seed endpoint confirmation for the R9 finalist."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _rate(case, low=20e-6, high=60e-6):
    events=[{"event_index":0,"cycles":0.,"extension_m":0.,"threshold_action":0.},*case["events"]]
    left=next(x for x in events if x["extension_m"]>=low)
    right=next(x for x in events if x["extension_m"]>=high)
    developed=(right["extension_m"]-left["extension_m"])/(right["cycles"]-left["cycles"])
    action=float(case["overlap"]["mean_hazard_action"]); cycle=0.
    for event in events[1:]:
        cycle+=float(event["threshold_action"])/action;event["opening_cycles"]=cycle
    events[0]["opening_cycles"]=0.
    opening=(right["extension_m"]-left["extension_m"])/(right["opening_cycles"]-left["opening_cycles"])
    return developed,opening,left,right


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--case",type=Path,action="append",required=True);p.add_argument("--out-root",type=Path,required=True)
    a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in a.case:
        case=json.loads(path.read_text());rate,opening,left,right=_rate(case)
        rows.append({"seed":case["seed"],"Kmax_MPa_sqrt_m":case["Kmax_MPa_sqrt_m"],"DeltaK_MPa_sqrt_m":case["DeltaK_MPa_sqrt_m"],
                     "event_count":len(case["events"]),"window_low_event":left["event_index"],"window_high_event":right["event_index"],
                     "window_low_extension_m":left["extension_m"],"window_high_extension_m":right["extension_m"],
                     "developed_da_dN":rate,"opening_da_dN":opening,"developed_over_opening":rate/opening,"result":str(path.resolve())})
    frame=pd.DataFrame(rows).sort_values(["seed","DeltaK_MPa_sqrt_m"])
    slopes=[]
    for seed,g in frame.groupby("seed"):
        if len(g)!=2: raise SystemExit(f"seed {seed} does not have two endpoints")
        g=g.sort_values("DeltaK_MPa_sqrt_m");lo,hi=g.iloc[0],g.iloc[1]
        slopes.append({"seed":int(seed),"endpoint_slope":math.log(hi.developed_da_dN/lo.developed_da_dN)/math.log(hi.DeltaK_MPa_sqrt_m/lo.DeltaK_MPa_sqrt_m),
                       "rate_width_decades":math.log10(hi.developed_da_dN/lo.developed_da_dN),"low_rate":lo.developed_da_dN,"high_rate":hi.developed_da_dN})
    slope=pd.DataFrame(slopes)
    summary={"schema":"r9.independent-seed-confirmation/1","window":"20_to_60_um_common_committed_events",
             "seeds":slope.seed.tolist(),"endpoint_slope_mean":float(slope.endpoint_slope.mean()),
             "endpoint_slope_sample_sd":float(slope.endpoint_slope.std(ddof=1)),
             "rate_width_decades_mean":float(slope.rate_width_decades.mean()),
             "low_rate_cv":float(frame.groupby("Kmax_MPa_sqrt_m").developed_da_dN.std(ddof=1).iloc[0]/frame.groupby("Kmax_MPa_sqrt_m").developed_da_dN.mean().iloc[0]),
             "high_rate_cv":float(frame.groupby("Kmax_MPa_sqrt_m").developed_da_dN.std(ddof=1).iloc[-1]/frame.groupby("Kmax_MPa_sqrt_m").developed_da_dN.mean().iloc[-1])}
    frame.to_csv(a.out_root/"seed_endpoint_rates.csv",index=False);slope.to_csv(a.out_root/"seed_endpoint_slopes.csv",index=False)
    (a.out_root/"seed_confirmation_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return 0
if __name__=="__main__":raise SystemExit(main())
