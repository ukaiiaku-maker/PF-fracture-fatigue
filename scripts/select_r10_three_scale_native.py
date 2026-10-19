#!/usr/bin/env python3
"""Select mechanism-balanced native rows from the three-scale state screen."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path


def read(path):
    with path.open(newline="") as stream:return list(csv.DictReader(stream))


def write(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--metrics",type=Path,required=True);p.add_argument("--state-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--count",type=int,default=24)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    registry=read(a.registry);by_id={row["candidate_id"]:row for row in registry}
    metric={row["candidate_id"]+"__CM":row for row in read(a.metrics)}
    scores=[]
    for path in sorted(a.state_root.glob("*/*.json")):
        record=json.loads(path.read_text());history=record.get("history") or []
        if not history:continue
        first,last=history[0],history[-1];candidate=record["candidate_id"]
        m=metric[candidate];mobile=max(float(row.get("mobile_count") or 0.) for row in history)
        state=max(float(row.get("state_relative_change") or 0.) for row in history)
        radius=abs(float(last.get("tip_radius_m") or 1e-6)-1e-6)/1e-6
        shield=max(abs(float(row.get("K_shield_Pa_sqrt_m") or 0.)) for row in history)/1e6
        consequence=max(state,radius,shield/max(float(record["Kmax_MPa_sqrt_m"]),1e-30))
        scores.append({"candidate_id":candidate,"source_material_class":m["source_material_class"],
            "analytical_objective":float(m["objective"]),"analytical_slope":float(m["target_local_slope"]),
            "analytical_rate_m_per_cycle":float(m["target_rate_m_per_cycle"]),
            "state_relative_change_max":state,"tip_radius_relative_change":radius,
            "shielding_over_Kmax":shield/max(float(record["Kmax_MPa_sqrt_m"]),1e-30),
            "mobile_max":mobile,"consequence_score":consequence,
            "state_receipt":str(path)})
    ordered=[]
    # Protect class and genuinely consequential-state lanes, then analytical frontier.
    for key,reverse in (("consequence_score",True),("analytical_objective",False)):
        pool=sorted(scores,key=lambda row:row[key],reverse=reverse)
        seen=set()
        for row in pool:
            cls=row["source_material_class"]
            if cls in seen:continue
            ordered.append(row);seen.add(cls)
    ordered+=sorted(scores,key=lambda row:(-row["consequence_score"],row["analytical_objective"]))
    ordered+=sorted(scores,key=lambda row:(row["analytical_objective"],-row["consequence_score"]))
    chosen=[];seen=set()
    for row in ordered:
        if row["candidate_id"] in seen:continue
        chosen.append(row);seen.add(row["candidate_id"])
        if len(chosen)>=a.count:break
    write(a.out/"selection_metrics.csv",chosen)
    write(a.out/"selected_registry.csv",[by_id[row["candidate_id"]] for row in chosen])
    summary={"schema":"r10.three-scale-native-selection/1","selected":len(chosen),
        "unique_complete_hashes":len({by_id[row["candidate_id"]]["complete_row_sha256"] for row in chosen}),
        "classes":sorted({row["source_material_class"] for row in chosen}),
        "maximum_consequence_score":max((row["consequence_score"] for row in chosen),default=0.),
        "candidate_ids":[row["candidate_id"] for row in chosen]}
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
