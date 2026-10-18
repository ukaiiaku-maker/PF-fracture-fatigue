#!/usr/bin/env python3
"""Discrepancy-driven boundary probes after the first 64-vector native block."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path):
    with Path(path).open(newline="") as stream:return list(csv.DictReader(stream))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--state-map",type=Path,required=True)
    parser.add_argument("--native-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int,default=24)
    args=parser.parse_args(argv)
    selected=read(args.selection);mapped=read(args.state_map)
    by_candidate={}
    for row in mapped:by_candidate.setdefault(row["candidate_id"],[]).append(row)
    observations={}
    for path in args.native_root.glob("*/*/result.json"):
        result=json.loads(path.read_text());observations[result["candidate_id"]]=result
    proposals=[]
    for row in selected:
        result=observations.get(row["candidate_id"])
        if not result or result["events"] or result["unsupported_crossover_encountered"]:continue
        current=float(row["reference_factor"]);arm=by_candidate[row["candidate_id"]]
        event=sorted(float(r["reference_factor"]) for r in arm
                     if float(r["reference_factor"])>current and
                     (r["one_cycle_event_reachable"].lower()=="true" or
                      r["second_cycle_event_reachable"].lower()=="true") and
                     r["unsupported"].lower()!="true")
        if event:
            factor=event[0];reason="STATE_MAP_EVENT_BOUNDARY_AFTER_NATIVE_NONADVANCE"
        else:
            factor=min(max(float(r["reference_factor"]) for r in arm)+.05,1.12)
            reason="NATIVE_NONADVANCE_REQUIRES_UPPER_BOUNDARY_EXTENSION"
        proposal=dict(row);proposal["parent_reference_factor"]=row["reference_factor"]
        proposal["reference_factor"]=factor
        proposal["Kmax_MPa_sqrt_m"]=float(row["Kmax_MPa_sqrt_m"])/current*factor
        proposal["round2_reason"]=reason
        proposals.append(proposal)
    groups={}
    for row in proposals:
        key=(row["search_design"],row["mechanism_prediction"],row["mode"],row["round2_reason"])
        groups.setdefault(key,[]).append(row)
    for group in groups.values():group.sort(key=lambda r:r["candidate_id"])
    output=[]
    while len(output)<min(args.count,len(proposals)):
        changed=False
        for key in sorted(groups):
            if groups[key]:output.append(groups[key].pop(0));changed=True
            if len(output)>=args.count:break
        if not changed:break
    for index,row in enumerate(output,1):row["acquisition_rank"]=index
    fields=[]
    for row in output:
        for key in row:
            if key not in fields:fields.append(key)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(output)
    print(json.dumps({"proposed":len(output),"reasons":{
        reason:sum(r["round2_reason"]==reason for r in output)
        for reason in sorted({r["round2_reason"] for r in output})}},indent=2))


if __name__=="__main__":raise SystemExit(main())
