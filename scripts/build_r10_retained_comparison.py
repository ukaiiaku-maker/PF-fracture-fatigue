#!/usr/bin/env python3
"""Build the retained-row comparison and A0 acquisition screen for R10."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.archived_fatigue_material_r9 import build_archived_current_engine
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import mean_cycle_inputs
from arrhenius_fracture.retained_material_r10 import (
    STRICT_FAMILY, V2_FAMILY, load_parent_binding,
    load_strict_original_binding, load_v2_binding,
)


ORIGINALS = {
    "Peak": "v913_paper_peak01_0242980_persistent_sites",
    "DBTT": "v913_paper_dbtt01_0202500_persistent_sites",
    "Weak-T": "v913_paper_weakT01_0129902_persistent_sites",
    "Ceramic-like": "v913_paper_ceramic01_0077080_persistent_sites",
}
V2 = {
    "DBTT_V2": ("DBTT", "P25_TJBSV2_S_002987"),
    "DBTT_V2_P40": ("DBTT", "P40_TJBSV2_S_038503"),
    "weakT_V2_P25": ("Weak-T", "P25_TJBSV2_S_026704"),
    "weakT_V2_P40": ("Weak-T", "P40_TJBSV2_S_038278"),
    "ceramic_V2_P25": ("Ceramic-like", "P25_TJBSV2_S_004127"),
    "ceramic_V2_P40": ("Ceramic-like", "P40_TJBSV2_S_031027"),
    "P25_control_064036": ("Control", "P25_TJBSV2_S_064036"),
    "P40_control_043821": ("Control", "P40_TJBSV2_S_043821"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_band(frame: pd.DataFrame) -> dict:
    slopes=[]
    for left, right in zip(frame.iloc[:-1].itertuples(), frame.iloc[1:].itertuples()):
        m=math.log(right.action_per_cycle/left.action_per_cycle)/math.log(right.DeltaK/left.DeltaK)
        slopes.append((left.DeltaK,right.DeltaK,m,
                       math.log10(right.action_per_cycle/left.action_per_cycle)))
    best={"width_decades":0.0,"DeltaK_low":None,"DeltaK_high":None}
    for i,item in enumerate(slopes):
        if not 3.4 <= item[2] <= 4.6: continue
        width=0.0
        for j in range(i,len(slopes)):
            if not 3.4 <= slopes[j][2] <= 4.6: break
            width += slopes[j][3]
            if width > best["width_decades"]:
                best={"width_decades":width,"DeltaK_low":slopes[i][0],
                      "DeltaK_high":slopes[j][1]}
    return best


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--original-registry",type=Path,required=True)
    p.add_argument("--parent-registry",type=Path,required=True)
    p.add_argument("--v2-rows",type=Path,required=True)
    p.add_argument("--v2-manifest",type=Path,required=True)
    p.add_argument("--v2-history",type=Path,required=True)
    p.add_argument("--v2-spatial",type=Path,required=True)
    p.add_argument("--prospective-history",type=Path,required=True,
                   help="sealed P25/P40/P55 physical-transfer evidence directory")
    p.add_argument("--out-root",type=Path,required=True)
    a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True)

    bindings=[]; metadata=[]
    for target,option in ORIGINALS.items():
        b=load_strict_original_binding(a.original_registry,option)
        bindings.append(b);metadata.append((b,target,"original_four",option))
    for candidate,role in (("P25_TRANSFER_V1_RANK1","fatigue_parent"),
                           ("P40_TRANSFER_CALIBRATED_GEN2","fatigue_parent"),
                           ("P55_TRANSFER_V1_RANK1","bounded_negative_control")):
        b=load_parent_binding(a.parent_registry,candidate)
        bindings.append(b);metadata.append((b,"Control",role,candidate))
    for alias,(target,candidate) in V2.items():
        b=load_v2_binding(a.v2_rows,a.v2_manifest,candidate)
        bindings.append(b);metadata.append((b,target,"v2_exact",alias))

    fracture=pd.read_csv(a.v2_history/"production_1d_fracture_results.csv")
    fatigue=pd.read_csv(a.v2_history/"production_300K_fatigue_case_table.csv")
    spatial=pd.read_csv(a.v2_spatial/"v2_2d_single_crack_terminal_summary.csv")
    prospective_rates=pd.read_csv(a.prospective_history/"physical_developed_rates.csv")
    prospective_slopes=pd.read_csv(a.prospective_history/"physical_local_slopes.csv")
    prospective_decision=json.loads(
        (a.prospective_history/"final_candidate_selection.json").read_text())
    comparisons=[]; analytical=[]; surfaces={}
    loads=np.geomspace(0.5,80.0,49)
    for binding,target,role,alias in metadata:
        cid=binding.optimized_row["candidate_id"]
        engine=build_archived_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
        rows=[]
        for Kmax in loads:
            waveform=FatigueWaveform(Kmax=float(Kmax)*1e6,R=0.1,frequency_Hz=1000.0)
            result=mean_cycle_inputs(engine,waveform,300.0,64)
            row={"candidate_id":cid,"alias":alias,"target_class":target,
                 "family":binding.provenance["family"],"Kmax":float(Kmax),
                 "DeltaK":0.9*float(Kmax),"action_per_cycle":result["hazard_action_per_cycle"],
                 "sigma_min_Pa":result["sigma_min_Pa"],"sigma_max_Pa":result["sigma_max_Pa"]}
            rows.append(row);analytical.append(row)
        af=pd.DataFrame(rows)
        positive=af[af.action_per_cycle>0].reset_index(drop=True)
        band=longest_band(positive) if len(positive)>1 else {"width_decades":0.0,"DeltaK_low":None,"DeltaK_high":None}
        hf=fracture[fracture.candidate_id==cid]
        hfat=fatigue[fatigue.candidate_id==cid]
        h2=spatial[spatial.candidate_id==cid]
        hpr=prospective_rates[prospective_rates.candidate_id==cid]
        hps=prospective_slopes[prospective_slopes.candidate_id==cid]
        prior=prospective_decision.get("targets",{}).get(
            {"P25_TRANSFER_V1_RANK1":"P25",
             "P40_TRANSFER_CALIBRATED_GEN2":"P40",
             "P55_TRANSFER_V1_RANK1":"P55"}.get(cid,""),{})
        comparisons.append({
            "target_class":target,"alias":alias,"candidate_id":cid,
            "family":binding.provenance["family"],"role":role,
            "complete_row_sha256":binding.provenance["complete_row_sha256"],
            "evidence_identity":("historical_result_consumer_changed_plus_fresh_A0_bridge"
                                 if len(hf) else "fresh_A0_bridge_historical_classification"),
            "historical_1d_fracture_conditions":len(hf),
            "historical_1d_K_min":None if hf.empty else float(hf.K_onset_MPa_sqrt_m.min()),
            "historical_1d_K_max":None if hf.empty else float(hf.K_onset_MPa_sqrt_m.max()),
            "historical_max_emitted":None if hf.empty else float(hf.emitted_total.max()),
            "historical_max_backstress_Pa":None if hf.empty else float(hf.backstress_Pa.max()),
            "historical_fatigue_cases":len(hfat),
            "historical_fatigue_finite_points":int(hfat.developed_da_dN.notna().sum()) if len(hfat) else 0,
            "historical_fatigue_censors":"" if hfat.empty else ";".join(sorted(set(hfat.censor_status.astype(str)))),
            "prospective_physical_points":len(hpr),
            "prospective_developed_points":int(hpr.developed_qualified.fillna(False).sum()) if len(hpr) else 0,
            "prospective_DeltaK_min":None if hpr.empty else float(hpr.applied_full_DeltaK.min()),
            "prospective_DeltaK_max":None if hpr.empty else float(hpr.applied_full_DeltaK.max()),
            "prospective_rate_min":None if hpr.empty else float(hpr.physical_rate.min()),
            "prospective_rate_max":None if hpr.empty else float(hpr.physical_rate.max()),
            "prospective_local_slope_min":None if hps.empty else float(hps.physical_slope.min()),
            "prospective_local_slope_max":None if hps.empty else float(hps.physical_slope.max()),
            "prospective_global_slope":prior.get("m_global"),
            "prospective_classification":prior.get("classification",""),
            "prospective_window_qualified":prior.get("window_qualified"),
            "prospective_seed_transfer_passed":prior.get("seed_transfer_passed"),
            "sealed_2d_cases":len(h2),
            "sealed_2d_temperatures":"" if h2.empty else ";".join(map(str,sorted(set(h2.temperature_K.astype(int))))),
            "sealed_2d_all_reached_1000um":bool(len(h2) and h2.target_reached.all()),
            "A0_moderate_slope_width_decades":band["width_decades"],
            "A0_band_DeltaK_low":band["DeltaK_low"],
            "A0_band_DeltaK_high":band["DeltaK_high"],
            "current_1d_status":(
                "QUEUED_INFORMATION_BATCH" if target != "Control" else
                "CURRENT_CONSUMER_BRIDGE_REQUIRED" if cid in prospective_decision.get("retained_candidate_ids",[]) else
                "ANALYTICAL_CONTROL_ONLY"),
            "current_2d_status":"REUSE_SEALED_2D" if len(h2) else "NOT_SELECTED",
        })
        probe=np.asarray([0.,1e9,2e9,5e9,10e9,20e9,30e9])
        surfaces[cid]={"provenance":dict(binding.provenance),"temperatures":{}}
        for T in (300.,750.,1200.):
            surfaces[cid]["temperatures"][str(int(T))]={}
            for name,surface in (("opening",binding.manifest.cleavage),("emission",binding.manifest.emission)):
                G=np.asarray(surface.values_eV(probe,T),float)
                surfaces[cid]["temperatures"][str(int(T))][name]={
                    "stress_Pa":probe.tolist(),"barrier_eV":G.tolist(),
                    "dG_dsigma_eV_per_Pa":np.gradient(G,probe).tolist(),
                    "raw_rate_s-1":np.asarray(surface.rate(probe,T),float).tolist(),
                }

    comparison=pd.DataFrame(comparisons).sort_values(["target_class","family","alias"])
    comparison.to_csv(a.out_root/"retained_material_comparison.csv",index=False)
    pd.DataFrame(analytical).to_csv(a.out_root/"analytical_opening_screen.csv",index=False)
    (a.out_root/"evaluated_surfaces.json").write_text(json.dumps(surfaces,indent=2,sort_keys=True)+"\n")
    spatial.to_csv(a.out_root/"reused_sealed_2d_terminal_summary.csv",index=False)

    inputs=[a.original_registry,a.parent_registry,a.v2_rows,a.v2_manifest,
            a.v2_history/"production_1d_fracture_results.csv",
            a.v2_history/"production_300K_fatigue_case_table.csv",
            a.v2_spatial/"v2_2d_single_crack_terminal_summary.csv",
            a.prospective_history/"final_candidate_parameter_rows.csv",
            a.prospective_history/"final_candidate_selection.json",
            a.prospective_history/"physical_developed_rates.csv",
            a.prospective_history/"physical_local_slopes.csv"]
    state={
        "schema":"r10.joint-search-continuation/1",
        "objective":"joint fracture classes plus developed moderate-slope fatigue for Weak-T, Peak-T, and DBTT",
        "branch":"codex/r10-joint-fracture-fatigue-continuation",
        "base_head":"f691e0c2ef2dfa278de9f4f216f48c15d07f4a2e",
        "backend":"R8 restored persistent/reversible current backend with R9 phase-exact stochastic fatigue",
        "barrier_family_policy":{
            "A":{"id":STRICT_FAMILY,"primary":True,"allowed":"original single EXP_floor only"},
            "B":{"id":V2_FAMILY,"primary":False,"allowed":"exact inherited V2 guard/entropy surfaces; cannot count as A"},
            "C":{"id":"C_V3_V3_1_R9_BROAD_BASIS","primary":False,"allowed":"comparison only; no optimization"},
        },
        "retained_rows":comparison[["alias","candidate_id","family","complete_row_sha256"]].to_dict("records"),
        "source_inputs":[{"path":str(x.resolve()),"sha256":sha(x)} for x in inputs],
        "completed":["retained assets recovered and hashed","exact surfaces evaluated",
                     "A0 current-consumer amplitude screen",
                     "P25/P40/P55 historical physical-transfer evidence recovered"],
        "running_jobs":[],
        "queued_jobs":[
            {"tier":"current_full_state_1d_sparse","class":"DBTT","candidate_id":"P25_TJBSV2_S_002987"},
            {"tier":"current_full_state_1d_sparse","class":"Weak-T","candidate_id":"P25_TJBSV2_S_026704"},
            {"tier":"current_full_state_1d_sparse","class":"Peak-T","candidate_id":"v913_zeroD_sobol_0242980"},
            {"tier":"current_full_state_1d_control","class":"Ceramic-like","candidate_id":"v913_zeroD_sobol_0077080"},
        ],
        "unresolved":[
            "original four have no inherited V2 direct-surface extension",
            "historical V2 fatigue pilots were resource/state-closure censored",
            "same-row current 2D requires narrow complete-row binding after 1D selection",
        ],
        "next":"execute queued sparse full-state 1D points selected from A0 action levels, then update class Pareto sets",
    }
    (a.out_root/"continuation_state.json").write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    print(comparison.to_string(index=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
