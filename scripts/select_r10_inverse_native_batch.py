#!/usr/bin/env python3
"""Select a measured-discrepancy native batch from inverse-shape state runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def atomic(path: Path, payload: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def load_results(directory: Path) -> dict[str,dict]:
    output={}
    for path in directory.glob("*/*.json"):
        try:payload=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if isinstance(payload,dict) and payload.get("candidate_id"):
            output[str(payload["candidate_id"])]=payload
    return output


def state_score(payload: dict) -> float:
    history=payload.get("history") or []
    if len(history)<2:return 0.
    first,last=history[0],history[-1]
    terms=[]
    for key in ("mobile_count","retained_count","tip_radius_m","K_shield_Pa_sqrt_m",
                "source_backstress_Pa"):
        if key in first and key in last:
            scale=max(abs(float(first[key])),abs(float(last[key])),1.)
            terms.append(abs(float(last[key])-float(first[key]))/scale)
    return max(terms,default=0.)


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--state-dir",type=Path,required=True)
    parser.add_argument("--native-dir",type=Path,action="append",default=[])
    parser.add_argument("--exclude-registry",type=Path,action="append",default=[])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--update-index",type=int,required=True)
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    with args.registry.open(newline="") as stream:registry=list(csv.DictReader(stream))
    by_id={row["candidate_id"]:row for row in registry}
    state=load_results(args.state_dir);native={}
    for directory in args.native_dir:native.update(load_results(directory))
    excluded=set()
    for path in args.exclude_registry:
        with path.open(newline="") as stream:
            excluded.update(row["candidate_id"] for row in csv.DictReader(stream))
    metrics=[]
    for cid,result in state.items():
        if cid not in by_id or cid in excluded:continue
        row=by_id[cid];native_result=native.get(cid)
        state_events=int(result.get("event_cycle_count",0))
        native_events=(None if native_result is None else len(native_result.get("events",[])))
        discrepancy=(0. if native_events is None else abs(native_events-state_events)/max(native_events,state_events,1))
        analytical_slope=float(row.get("predicted_hcf_local_slope") or math.nan)
        score=(6.*discrepancy+2.*math.log10(1.+state_score(result)*1e8)
               +min(state_events,4)+(.5 if math.isfinite(analytical_slope) else 0.)
               -(4. if result.get("unsupported_crossover") else 0.))
        metrics.append({"candidate_id":cid,"shape":row["selected_shape"],
            "mechanism":row.get("mechanism_prediction","UNRESOLVED"),
            "plasticity_sha256":row.get("plasticity_physics_sha256"),
            "state_events":state_events,"native_events":native_events,
            "state_relative_change":state_score(result),
            "state_native_event_discrepancy":discrepancy,
            "unsupported":bool(result.get("unsupported_crossover")),"score":score})
    chosen=[];seen=set();seen_plastic=set()
    cells=sorted({(row["shape"],row["mechanism"]) for row in metrics})
    pools={cell:sorted([row for row in metrics if (row["shape"],row["mechanism"])==cell],
                       key=lambda row:(-row["score"],row["candidate_id"])) for cell in cells}
    while len(chosen)<min(args.count,len(metrics)):
        advanced=False
        for cell in cells:
            while pools[cell] and (pools[cell][0]["candidate_id"] in seen or
                    pools[cell][0]["plasticity_sha256"] in seen_plastic):pools[cell].pop(0)
            if pools[cell]:
                pick=pools[cell].pop(0);chosen.append(pick);seen.add(pick["candidate_id"])
                seen_plastic.add(pick["plasticity_sha256"]);advanced=True
                if len(chosen)>=args.count:break
        if not advanced:break
    if len(chosen)<args.count:
        for pick in sorted(metrics,key=lambda row:(-row["score"],row["candidate_id"])):
            if pick["candidate_id"] not in seen:
                chosen.append(pick);seen.add(pick["candidate_id"])
            if len(chosen)>=args.count:break
    selected=[by_id[row["candidate_id"]] for row in chosen]
    write_csv(args.out/"selected_registry.csv",selected)
    write_csv(args.out/"selection_metrics.csv",chosen)
    summary={"schema":"r10.inverse-shape-acquisition-update/1",
        "update_index":args.update_index,"state_results":len(state),
        "prior_native_results":len(native),"excluded_candidates":len(excluded),
        "selected_count":len(selected),"selected_ids":[r["candidate_id"] for r in selected],
        "selected_unique_plasticity_vectors":len({r.get("plasticity_physics_sha256") for r in selected}),
        "selection_uses_measured_discrepancy":True}
    atomic(args.out/"selection_summary.json",summary)
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
