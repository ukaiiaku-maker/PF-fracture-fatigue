#!/usr/bin/env python3
"""Consolidate same-backend monotonic controls and candidate dispositions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    a=p.parse_args(argv);rows=[]
    for path in sorted(a.root.glob("*/*K.json")):
        d=json.loads(path.read_text()); events=d["events"]
        rows.append({"campaign":path.parent.name,"candidate_id":d["candidate_id"],
            "temperature_K":d["temperature_K"],"status":d["status"],
            "K_init_MPa_sqrt_m":d["K_init_MPa_sqrt_m"],
            "K_init_lower_bound_MPa_sqrt_m":d["Kmax_MPa_sqrt_m"] if not events else None,
            "K_final_event_MPa_sqrt_m":d["K_final_event_MPa_sqrt_m"],
            "event_count":d["event_count"],"final_extension_um":d["final_extension_um"],
            "maximum_mobile_count":max((x["mobile_count"] for x in events),default=0.),
            "maximum_retained_count":max((x["retained_count"] for x in events),default=0.),
            "maximum_source_activity":max((x["source_activity"] for x in events),default=0.),
            "maximum_K_shield_MPa_sqrt_m":max((abs(x["K_shield_MPa_sqrt_m"]) for x in events),default=0.),
            "artifact":str(path.resolve())})
    frame=pd.DataFrame(rows).sort_values(["candidate_id","temperature_K"])
    frame.to_csv(a.root/"current_fracture_case_table.csv",index=False)
    decisions=[]
    for cid,g in frame.groupby("candidate_id"):
        low=g[g.temperature_K==300.]
        lowK=None if low.empty else low.iloc[0].K_init_MPa_sqrt_m
        decisions.append({"candidate_id":cid,"temperatures_K":g.temperature_K.tolist(),
            "all_complete":bool((g.status=="complete").all()),
            "censored_temperatures_K":g[g.status!="complete"].temperature_K.tolist(),
            "K300_MPa_sqrt_m":None if pd.isna(lowK) else float(lowK),
            "inherited_low_temperature_scale_gate_5_to_30":bool(pd.notna(lowK) and 5<=lowK<=30),
            "plastic_state_exercised":bool(g.maximum_source_activity.max()>0),
            "maximum_K_shield_MPa_sqrt_m":float(g.maximum_K_shield_MPa_sqrt_m.max()),
            "disposition":("JOINT_REJECT_LOW_T_FRACTURE_SCALE" if pd.notna(lowK) and lowK<5
                           else "REFERENCE_OR_REQUIRES_DENSE_CLASSIFICATION")})
    payload={"schema":"r10.current-fracture-bridge-decision/1","rows":decisions,
        "scope":"current deterministic diagnostic-mark 0.005 MPa sqrt(m)/s monotonic bridge; censored K is a lower bound",
        "automatic_class_promotion":False}
    (a.root/"current_fracture_decision.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(frame.to_string(index=False));print(json.dumps(payload,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
