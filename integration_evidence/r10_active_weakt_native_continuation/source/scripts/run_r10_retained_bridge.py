#!/usr/bin/env python3
"""Run restartable current-backend trajectories for one exact retained row."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import run_trajectory
from arrhenius_fracture.retained_material_r10 import (
    load_offspring_binding,load_parent_binding,load_strict_original_binding,load_v2_binding,
)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=("original","parent","offspring","v2"),required=True)
    p.add_argument("--identifier",required=True)
    p.add_argument("--original-registry",type=Path)
    p.add_argument("--parent-registry",type=Path)
    p.add_argument("--offspring-registry",type=Path)
    p.add_argument("--v2-rows",type=Path)
    p.add_argument("--v2-manifest",type=Path)
    p.add_argument("--loads",required=True)
    p.add_argument("--temperature-K",type=float,default=300.)
    p.add_argument("--seed",type=int,default=4810)
    p.add_argument("--target-events",type=int,default=12)
    p.add_argument("--target-extension-um",type=float,default=70.)
    p.add_argument("--maximum-events",type=int,default=40)
    p.add_argument("--maximum-cycles",type=float,default=1e12)
    p.add_argument("--phase-count",type=int,default=64)
    p.add_argument("--out-root",type=Path,required=True)
    a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True)
    if a.kind=="original":
        binding=load_strict_original_binding(a.original_registry,a.identifier)
    elif a.kind=="parent": binding=load_parent_binding(a.parent_registry,a.identifier)
    elif a.kind=="offspring": binding=load_offspring_binding(a.offspring_registry,a.identifier)
    else: binding=load_v2_binding(a.v2_rows,a.v2_manifest,a.identifier)
    loads=[float(x) for x in a.loads.split(",")]
    preflight={
        "schema":"r10.retained-current-bridge/1","binding":dict(binding.provenance),
        "backend":"R8_RESTORED_PERSISTENT_REVERSIBLE_CURRENT_BACKEND",
        "trajectory":"R9_PHASE_EXACT_STOCHASTIC_WITH_TRANSACTIONAL_MEAN_OR_NATIVE_HIGH_CYCLE_BLOCKS",
        "native_numerical_controls":{
            "DMD_chain_growth_factor":float(os.environ.get("V10230_DMD_CHAIN_GROWTH_FACTOR","2")),
            "all_projected_segments_require_existing_state_hazard_ledger_positivity_and_event_guard_validation":True,
        },
        "kernel":{"path":str(R4_KERNEL_FAMILY.resolve()),"sha256":sha(R4_KERNEL_FAMILY)},
        "loading":{"Kmax_MPa_sqrt_m":loads,"temperature_K":a.temperature_K,
                   "R":0.1,"frequency_Hz":1000.,"phase_count":a.phase_count},
        "stochastic":{"seed":a.seed,"owner":"PCG64_SeedSequence_[seed,1]",
                      "mark":"unit_exponential","reward":"threshold_scaled_bounded"},
        "stops":{"target_events":a.target_events,"target_extension_um":a.target_extension_um,
                 "maximum_events":a.maximum_events,"maximum_cycles":a.maximum_cycles},
        "local_strength_Pa":30e9,
        "local_strength_role":"independent qualified mechanical coordinate; not barrier knee",
    }
    (a.out_root/"production_preflight.json").write_text(json.dumps(preflight,indent=2,sort_keys=True)+"\n")
    summaries=[]
    for K in loads:
        case=a.out_root/("Kmax_"+format(K,".9g").replace(".","p"))
        result_path=case/"result.json"
        if result_path.is_file(): result=json.loads(result_path.read_text())
        else:
            case.mkdir(parents=True,exist_ok=True)
            result=run_trajectory(
                binding,kernel_family_path=R4_KERNEL_FAMILY,Kmax_MPa_sqrt_m=K,
                seed=a.seed,checkpoint_directory=case/"checkpoint",
                target_events=a.target_events,target_extension_m=a.target_extension_um*1e-6,
                maximum_events=a.maximum_events,maximum_cycles=a.maximum_cycles,
                phase_count=a.phase_count,temperature_K=a.temperature_K,R=.1,frequency_Hz=1000.,
            )
            result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        summaries.append({"Kmax_MPa_sqrt_m":K,"DeltaK_MPa_sqrt_m":.9*K,
                          "status":result["status"],"event_count":len(result["events"]),
                          "final_cycles":result["final_cycles"],
                          "final_extension_m":result["final_extension_m"],
                          "overlap":result["overlap"],"result":str(result_path.resolve())})
        print(json.dumps(summaries[-1],sort_keys=True),flush=True)
    (a.out_root/"campaign_summary.json").write_text(json.dumps({"preflight":preflight,"cases":summaries},indent=2,sort_keys=True)+"\n")
    return 0


if __name__=="__main__":raise SystemExit(main())
