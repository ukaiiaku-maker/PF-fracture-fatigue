#!/usr/bin/env python3
"""Dense held-out derivative, cycle, and barrier checks for selected shapes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import CohesiveThermalLaw
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    connected_barrier,cyclic_rate,metastable_window,rate_curve)
from scripts.run_r10_opening_alternatives import mechanics,thermal_from


def atomic(path:Path,payload:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    temporary.replace(path)


def evaluate(row:dict)->dict:
    cohesive=json.loads(row["cohesive_parameter_vector"]);thermal=thermal_from(cohesive)
    shape=row["selected_shape"];_,kenv=mechanics(thermal,300.)
    window=metastable_window(shape,thermal,300.,kenv)
    birth=float(window["birth_load_Pa"])*math.sqrt(2*math.pi*1e-6)/1e6
    fold=float(window["fold_load_Pa"])*math.sqrt(2*math.pi*1e-6)/1e6
    amplitudes=np.geomspace(birth*1.00001,fold*.99999,101)
    peak=rate_curve(amplitudes,kenv,thermal,300.,shape,
        float(cohesive["activation_area_m2"]),float(cohesive["attempt_frequency_s"]))
    cycle_indices=np.unique(np.linspace(0,100,17).astype(int))
    cycles=[cyclic_rate(float(amplitudes[index]),kenv,thermal,300.,shape,
        float(cohesive["activation_area_m2"]),float(cohesive["attempt_frequency_s"]),
        phases=32) for index in cycle_indices]
    positive=[i for i,item in enumerate(cycles) if item["ensemble_rate_m_per_cycle"]>0.]
    slopes=[None]*len(cycles)
    if len(positive)>=3:
        x=np.log([cycles[i]["Kmax_MPa_sqrt_m"] for i in positive])
        y=np.log([cycles[i]["ensemble_rate_m_per_cycle"] for i in positive])
        values=np.gradient(y,x,edge_order=2)
        for index,value in zip(positive,values):slopes[index]=float(value)
    for item,slope in zip(cycles,slopes):item["finite_difference_local_slope"]=slope
    strict=[slopes[i] is not None and 3.4<=slopes[i]<=4.6 and
        cycles[i]["ensemble_rate_m_per_cycle"]>0. for i in range(len(cycles))]
    strict_width=0.;start=None
    for index,flag in enumerate(strict+[False]):
        if flag and start is None:start=index
        elif not flag and start is not None:
            end=index-1
            if end>start:strict_width=max(strict_width,math.log10(
                cycles[end]["ensemble_rate_m_per_cycle"]/
                cycles[start]["ensemble_rate_m_per_cycle"]))
            start=None
    midpoint=.5*(float(window["birth_load_Pa"])+float(window["fold_load_Pa"]))/kenv
    checked=connected_barrier(midpoint,kenv,thermal,300.,shape,verify_quadrature=True)
    return {"candidate_id":row["candidate_id"],"complete_row_sha256":row["complete_row_sha256"],
        "shape":shape,"metastable_load_ratio":window["load_ratio"],
        "peak_rate_curve":peak,"cyclic_32phase_curve":cycles,
        "cyclic_strict_adjacent_width_log10rate":strict_width,
        "strict_internal_reference_pass":strict_width>=2.,
        "barrier_quadrature_relative_error":checked["barrier_quadrature_relative_error"]}


def main(argv=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--count",type=int,default=32)
    parser.add_argument("--workers",type=int,default=8);args=parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    with args.registry.open(newline="") as stream:rows=list(csv.DictReader(stream))[:args.count]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(evaluate,rows,chunksize=1))
    for result in results:atomic(args.out/f"{result['candidate_id']}.json",result)
    summary={"schema":"r10.inverse-shape-dense-validation/1","materials":len(results),
        "strict_passes":sum(result["strict_internal_reference_pass"] for result in results),
        "maximum_cyclic_strict_width_log10rate":max((result["cyclic_strict_adjacent_width_log10rate"] for result in results),default=0.),
        "maximum_barrier_quadrature_relative_error":max((result["barrier_quadrature_relative_error"] for result in results),default=None),
        "phase_count":32,"amplitude_points_peak_reference":101,"amplitude_points_cyclic":17}
    atomic(args.out/"summary.json",summary);print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
