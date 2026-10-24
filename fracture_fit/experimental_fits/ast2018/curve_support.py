"""Final all-629 J-point support and quadrature refinement check."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import AstData
from .objective import _quadrature_weights,_select_quantiles


def _rows(path):
    with path.open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _curve_cost(observed_rows,model_rows):
    rows=sorted(observed_rows,key=lambda r:float(r["J_N_per_m"]))
    Jtarget=np.array([float(r["J_N_per_m"]) for r in rows])
    atarget=np.array([float(r["crack_extension_nm"])*1e-9 for r in rows])
    q=_quadrature_weights(Jtarget)
    events=[r for r in model_rows if r["event"]=="True"]
    if not events:return {"cost":float(np.hypot(1,20)-1),"covered_points":0,"n_points":len(rows),
                          "model_max_J_N_per_m":None}
    Jmodel=np.array([float(r["J_continuous_N_per_m"]) for r in events])
    amodel=np.array([float(r["da_m"]) for r in events])
    if np.all(np.diff(Jmodel)>-1e-9):
        residual=(np.interp(Jtarget,Jmodel,amodel)-atarget)/50e-9
        beyond=Jtarget>max(Jmodel)
        residual[beyond]+=10+(Jtarget[beyond]-max(Jmodel))/max(10,.05*max(Jmodel))
    else:
        distance=np.hypot((Jtarget[:,None]-Jmodel[None,:])/max(10,.05*max(Jtarget)),
                          (atarget[:,None]-amodel[None,:])/50e-9)
        residual=np.min(distance,axis=1)
        residual[Jtarget>max(Jmodel)]+=10
    return {"cost":float(np.sum(q*(np.hypot(1,residual)-1))),
            "covered_points":int(np.count_nonzero(Jtarget<=max(Jmodel))),
            "n_points":len(rows),"model_max_J_N_per_m":float(max(Jmodel)),
            "rms_standardized":float(np.sqrt(np.mean(residual**2)))}


def check_all(data_dir:Path,run_dir:Path,scenario:str):
    data=AstData.load(data_dir)
    config=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    search_points=int(config["curve_points"])
    curves=defaultdict(list)
    for r in data.J_curves:curves[r["curve_id"]].append(r)
    results=[]
    for curve_id,rows in sorted(curves.items()):
        condition_id=rows[0]["condition_id"]
        subset=[r for r in rows if r["sample_for_fit"].lower() in ("true","1","yes")]
        search_nodes=_select_quantiles(subset,search_points,"J_N_per_m")
        for level in ("reduced","spatial"):
            path=run_dir/f"confirmation_{scenario.lower()}/{level}/{condition_id}/ast_observer.csv"
            model=_rows(path)
            all_cost=_curve_cost(rows,model)
            subset_cost=_curve_cost(subset,model)
            search_cost=_curve_cost(search_nodes,model)
            results.append({"curve_id":curve_id,"condition_id":condition_id,
                "level":level,"all_629_support_cost":all_cost["cost"],
                "fit_subset_cost":subset_cost["cost"],
                "search_quadrature_cost":search_cost["cost"],
                "search_quadrature_points":search_cost["n_points"],
                "subset_minus_search_cost":subset_cost["cost"]-search_cost["cost"],
                "all_points":all_cost["n_points"],
                "fit_points":subset_cost["n_points"],
                "covered_all_points":all_cost["covered_points"],
                "model_max_J_N_per_m":all_cost["model_max_J_N_per_m"],
                "rms_standardized":all_cost.get("rms_standardized")})
    if sum(r["all_points"] for r in results if r["level"]=="reduced")!=629:
        raise RuntimeError("full Ast J-curve support was not retained")
    out=run_dir/f"curve_support_{scenario.lower()}.json"
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results
