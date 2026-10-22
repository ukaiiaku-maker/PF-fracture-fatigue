#!/usr/bin/env python3
"""Recompute finalist evidence from the executed QSTE Pareto archive."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_qste340tm_stateful_exp_floor import build_records
from scripts.continue_qste340tm_native_joint_search import DATA,OUT,candidate_summary,fixed_kmax,make_plot
from scripts.qualify_qste340tm_native_transfer import compare_case
from scripts.refit_qste340tm_faithful_reduction import metrics,opening_prediction,plastic_parameters,plastic_trajectory,unpack

def main():
    path=OUT/"joint_search.json";payload=json.loads(path.read_text());records=build_records(DATA);front=payload["pareto_front"];predictions={};finalists={}
    for mode,key in (("absolute","absolute_rmse_log10"),("shape","shape_rmse_log10")):
        chosen=min(front,key=lambda x:x[key]);vector=np.asarray(chosen["parameters_vector"]);trajectory=plastic_trajectory(vector,records,interval_substeps=1,waveform_steps=32);prediction=opening_prediction(vector,trajectory)[0];summary=candidate_summary(mode,records,vector,trajectory,prediction,chosen["objective"])
        frozen=plastic_trajectory(vector,records,interval_substeps=1,waveform_steps=32,evolve=False);frozen_prediction=opening_prediction(vector,frozen)[0]
        convergence={"history":{},"waveform":{}}
        for n in (1,2,4,8):
            state=plastic_trajectory(vector,records,interval_substeps=n,waveform_steps=32);pred=opening_prediction(vector,state)[0];convergence["history"][str(n)]=metrics(records,pred,state)
        for n in (32,64,128,256):
            state=plastic_trajectory(vector,records,interval_substeps=1,waveform_steps=n);pred=opening_prediction(vector,state)[0];convergence["waveform"][str(n)]=metrics(records,pred,state)
        finalists[mode]={"summary":summary,"optimizer":payload["finalists"][mode]["optimizer"],"frozen_metrics":metrics(records,frozen_prediction,frozen),"convergence":convergence,"prediction":prediction.tolist(),"frozen_prediction":frozen_prediction.tolist(),"state":{k:v.tolist() for k,v in trajectory.items() if k!="stress"}}
        predictions[mode+"_prediction"]=prediction
    payload["finalists"]=finalists;payload["fixed_Kmax_contrasts"]=fixed_kmax(records,predictions);payload["active_native_qualification"]=[]
    for mode in ("absolute","shape"):
        vector=np.asarray(finalists[mode]["summary"]["parameters_vector"]);plastic=plastic_parameters(vector);opening=unpack(vector)
        for r in (.1,.3,.5,.7):
            block=[x for x in records if x["R"]==r];row=block[len(block)//2];state=np.array([plastic.initial_mobile_count,plastic.initial_retained_count,plastic.initial_source_activity]);payload["active_native_qualification"].append(compare_case(f"selected_{mode}_mid",row["DeltaK_MPa_sqrt_m"],r,state,plastic,opening,512))
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    fields=list(records[0])+["absolute_prediction","absolute_frozen","shape_prediction","shape_frozen"]
    with (OUT/"joint_predictions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader()
        for i,row in enumerate(records):w.writerow(dict(row)|{"absolute_prediction":finalists["absolute"]["prediction"][i],"absolute_frozen":finalists["absolute"]["frozen_prediction"][i],"shape_prediction":finalists["shape"]["prediction"][i],"shape_frozen":finalists["shape"]["frozen_prediction"][i]})
    with (OUT/"fixed_Kmax_contrasts.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(payload["fixed_Kmax_contrasts"][0]),lineterminator="\n");w.writeheader();w.writerows(payload["fixed_Kmax_contrasts"])
    cases=[("best absolute",np.asarray(finalists["absolute"]["prediction"]),"#0072B2","-"),("best shape",np.asarray(finalists["shape"]["prediction"]),"#D55E00","--"),("absolute frozen",np.asarray(finalists["absolute"]["frozen_prediction"]),"#56B4E9",":")]
    make_plot(records,cases,OUT/"joint_fit_DeltaK.png","DeltaK");make_plot(records,cases,OUT/"joint_fit_Kmax.png","Kmax")
    print(json.dumps({m:{"absolute":f["summary"]["absolute_rmse_log10"],"shape":f["summary"]["shape_rmse_log10"]} for m,f in finalists.items()},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
