#!/usr/bin/env python3
"""Consolidate completed and restartable R10 bridge trajectories."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def rate(events, low=20e-6, high=60e-6):
    rows=[{"cycles":0.,"extension_m":0.,"event_index":0},*events]
    try:
        a=next(x for x in rows if x["extension_m"]>=low)
        b=next(x for x in rows if x["extension_m"]>=high)
    except StopIteration:return None
    return (b["extension_m"]-a["extension_m"])/(b["cycles"]-a["cycles"])


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    a=p.parse_args(argv); rows=[]
    for case in sorted((a.root/"bridges").glob("*/*")):
        if not case.is_dir():continue
        result=case/"result.json";progress=case/"checkpoint/trajectory_progress.json"
        if result.is_file():
            d=json.loads(result.read_text());complete=True
        elif progress.is_file():
            d=json.loads(progress.read_text());complete=False
            d={"candidate_id":d["identity"]["candidate_id"],
               "Kmax_MPa_sqrt_m":d["identity"]["Kmax_MPa_sqrt_m"],
               "DeltaK_MPa_sqrt_m":.9*d["identity"]["Kmax_MPa_sqrt_m"],
               "seed":d["identity"]["seed"],"events":d["events"],
               "final_cycles":d["cursor"]["completed_cycles"]+d["cursor"]["phase_index"]/d["identity"]["phase_count"],
               "final_extension_m":d["geometry"]["committed_extension_m"],
               "overlap":d["overlap"],"status":"RESTARTABLE_RESOURCE_CENSOR"}
        else:continue
        rows.append({"campaign":case.parent.name,"case":case.name,
            "candidate_id":d["candidate_id"],"Kmax_MPa_sqrt_m":d["Kmax_MPa_sqrt_m"],
            "DeltaK_MPa_sqrt_m":d["DeltaK_MPa_sqrt_m"],"seed":d["seed"],
            "complete":complete,"status":d.get("status"),"event_count":len(d["events"]),
            "final_cycles":d["final_cycles"],"final_extension_um":1e6*d["final_extension_m"],
            "rate_20_60_m_per_cycle":rate(d["events"]),
            "overlap_passed":d["overlap"]["passed"],
            "overlap_state_error":d["overlap"]["state_relative_max_error"],
            "overlap_hazard_error":d["overlap"]["hazard_relative_error"],
            "artifact":str((result if complete else progress).resolve())})
    frame=pd.DataFrame(rows).sort_values(["candidate_id","Kmax_MPa_sqrt_m"])
    frame.to_csv(a.root/"current_bridge_case_table.csv",index=False)
    slope_rows=[]; candidate_windows=[]
    for candidate,group in frame[
        frame.complete & frame.rate_20_60_m_per_cycle.notna()
    ].groupby("candidate_id"):
        group=group.sort_values("DeltaK_MPa_sqrt_m")
        for (_,left),(_,right) in zip(group.iloc[:-1].iterrows(),group.iloc[1:].iterrows()):
            m=math.log(right.rate_20_60_m_per_cycle/left.rate_20_60_m_per_cycle)/math.log(
                right.DeltaK_MPa_sqrt_m/left.DeltaK_MPa_sqrt_m)
            slope_rows.append({"candidate_id":candidate,
                "DeltaK_low":left.DeltaK_MPa_sqrt_m,"DeltaK_high":right.DeltaK_MPa_sqrt_m,
                "rate_low":left.rate_20_60_m_per_cycle,"rate_high":right.rate_20_60_m_per_cycle,
                "local_slope":m,"moderate_slope":3.4<=m<=4.6})
        if len(group)>=2:
            x=np.log(group.DeltaK_MPa_sqrt_m.to_numpy(float));y=np.log(group.rate_20_60_m_per_cycle.to_numpy(float))
            fit=np.polyfit(x,y,1); width=float(np.log10(group.rate_20_60_m_per_cycle.max()/group.rate_20_60_m_per_cycle.min()))
            local=[r for r in slope_rows if r["candidate_id"]==candidate]
            candidate_windows.append({"candidate_id":candidate,"point_count":len(group),
                "global_slope":float(fit[0]),"growth_rate_width_decades":width,
                "local_slope_min":min((r["local_slope"] for r in local),default=None),
                "local_slope_max":max((r["local_slope"] for r in local),default=None),
                "moderate_two_decade_interval":bool(width>=2.0 and local and all(r["moderate_slope"] for r in local))})
    pd.DataFrame(slope_rows).to_csv(a.root/"current_bridge_local_slopes.csv",index=False)
    pd.DataFrame(candidate_windows).to_csv(a.root/"current_bridge_windows.csv",index=False)
    two_decade=[row for row in candidate_windows if row["moderate_two_decade_interval"]]
    findings=[
        "current acceleration resolves the historical zero-event low-load DBTT and Weak-T censors",
        "V2 active amplitudes fail overlap and require costly phase-exact execution",
        "original Peak has an extreme low/high kinetic transition and no demonstrated moderate interval",
        "ceramic is retained only as a physical control",
    ]
    if two_decade:
        findings.append(
            "a strict-family projected complete row passes the current developed two-decade fatigue gate; "
            "this is not a class promotion until its monotonic and spatial gates pass")
    decision={
      "schema":"r10.information-batch-decision/1",
      "completed_cases":int(frame.complete.sum()),"restartable_cases":int((~frame.complete).sum()),
      "developed_20_60_rates":frame[frame.rate_20_60_m_per_cycle.notna()][
          ["candidate_id","DeltaK_MPa_sqrt_m","rate_20_60_m_per_cycle"]].to_dict("records"),
      "current_windows":candidate_windows,
      "findings":findings,
      "next_acquisition":[
        "use measured current P40 discrepancy and recovered strict-family projection machinery to rank class-parent offspring with protected fracture anchors",
        "complete one additional amplitude for DBTT and Weak-T only after strict analytical candidates are ranked",
        "reuse sealed same-row V2 2D cases; launch new 2D only for a promoted complete row",
      ],
    }
    (a.root/"information_batch_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
    state_path=a.root/"continuation_state.json";state=json.loads(state_path.read_text())
    if "fresh four-class current-backend information batch" not in state["completed"]:
        state["completed"].append("fresh four-class current-backend information batch")
    if two_decade:
        item="strict-family projected current-backend two-decade fatigue interval"
        if item not in state["completed"]:state["completed"].append(item)
    state["running_jobs"]=[]
    state["bridge_case_table"]=str((a.root/"current_bridge_case_table.csv").resolve())
    state["information_batch_decision"]=str((a.root/"information_batch_decision.json").resolve())
    state["queued_jobs"]=decision["next_acquisition"]
    state["next"]=decision["next_acquisition"][0]
    temp=state_path.with_suffix(".json.tmp");temp.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n");temp.replace(state_path)
    print(frame.to_string(index=False));return 0


if __name__=="__main__":raise SystemExit(main())
