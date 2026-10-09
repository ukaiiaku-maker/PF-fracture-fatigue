#!/usr/bin/env python3
"""Four actual V12/disabled-V5 peers in separate clean historical/current workers.

The historical base is identified by the commit introducing the corrected
Stage-II V3 qualifier, not by the revoked provisional scaffold branch. Complete
component differences are retained, including support certification/provenance;
this harness never strips model identity or turns metadata differences green.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

SCHEMA = "v12.disabled-v5-clean-historical-neutrality/1"
STAGE_II_V3_BASE = "326e3f5973ef623781ab8568798c495a3f68238c"
BASE_AUTHORITY = {
    "commit_subject":"Correct Stage II physical branching evidence",
    "qualifier":"scripts/qualify_v12_production_integration_v3.py",
    "qualifier_schema":"v12.production-integration-qualified-evidence/3",
    "gate":"STAGE_II_BASE_V3_LOCAL",
    "test":"tests/test_v12_stage2_base_v3.py",
}
CASES = {
    "monotonic":{"endpoint_m":[5.25e-4,0.],"seed":3621},
    "fixed_mesh_oblique":{"endpoint_m":[5.25e-4,2.5e-5],"seed":3621},
    "checkpoint_restart":{"endpoint_m":[5.25e-4,0.],"seed":3621},
    "unload_reload":{"openings_m":[0.,4e-7],"seed":3621},
}


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def hash_value(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def normalize(value):
    import numpy as np
    if isinstance(value,np.ndarray):
        array=np.ascontiguousarray(value)
        return {"array_sha256":hashlib.sha256(array.tobytes()).hexdigest(),
                "shape":list(array.shape),"dtype":array.dtype.str}
    if isinstance(value,np.generic): return normalize(value.item())
    if hasattr(value,"to_dict"): return normalize(value.to_dict())
    if is_dataclass(value): return {field.name:normalize(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,Mapping): return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [normalize(v) for v in value]
    if isinstance(value,(set,frozenset)):
        return {"set_values":sorted((normalize(v) for v in value),key=lambda item:encoded(item))}
    if isinstance(value,float) and not math.isfinite(value): return {"nonfinite":str(value)}
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return {"type":type(value).__qualname__,"state":normalize(value.__dict__)}


def differences(first,second,path=""):
    if isinstance(first,dict) and isinstance(second,dict):
        result=[]
        for key in sorted(set(first)|set(second)):
            name=path+"/"+key
            if key not in first or key not in second:
                result.append({"path":name,"base":first.get(key,"<MISSING>"),"current":second.get(key,"<MISSING>")})
            else: result.extend(differences(first[key],second[key],name))
        return result
    if first==second: return []
    return [{"path":path,"base":first,"current":second}]


def compare_case(base,current):
    if base["case_id"]!=current["case_id"] or base["input_configuration"]!=current["input_configuration"]:
        raise ValueError("neutrality peers have different physical inputs")
    changed=differences(base["terminal_components"],current["terminal_components"])
    model_equal=(base["model_identity"]==current["model_identity"]=="sharp_wake_mechanically_separating_v12")
    return {"case_id":base["case_id"],"model_identity_equal":model_equal,
        "base_model_identity":base["model_identity"],"current_model_identity":current["model_identity"],
        "base_complete_fingerprint":base["complete_fingerprint"],
        "current_complete_fingerprint":current["complete_fingerprint"],
        "complete_native_fingerprints_equal":base["complete_fingerprint"]==current["complete_fingerprint"],
        "full_component_differences":changed,
        "full_components_equal":not changed,
        "base_failure":base["failure"],"current_failure":current["failure"],
        "checkpoint_continuation_equal":base["restart_exact"] and current["restart_exact"],
        "passed":bool(model_equal and not changed and base["failure"] is None and current["failure"] is None
                      and base["restart_exact"] and current["restart_exact"])}


def clean_head(repository):
    sha=subprocess.check_output(("git","rev-parse","HEAD"),cwd=repository,text=True).strip()
    if subprocess.check_output(("git","status","--porcelain"),cwd=repository,text=True).strip():
        raise RuntimeError("neutrality worker must be a clean committed worktree")
    return sha


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")


def worker(repository,out,mode):
    head=clean_head(repository)
    sys.path.insert(0,str(repository))
    import numpy as np
    from arrhenius_fracture.checkpoint_v11 import write_checkpoint,restore_checkpoint
    from arrhenius_fracture.sharp_wake_backend_v12 import V12_MODEL_ID
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
    from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem as equilibrate
    from arrhenius_fracture.v12_production_driver import build_loaded_state,execute_event
    if mode=="disabled":
        from arrhenius_fracture.voiding_production_v5 import advance_disabled_v5_stage2 as event
    else: event=execute_event
    result=[]
    def components(state):
        # Every dataclass field is retained. Attribute absence is visible in
        # the comparison rather than silently mapped onto an active capability.
        payload={field.name:normalize(getattr(state,field.name)) for field in fields(state) if field.name != "mesh"}
        payload["mesh"]=normalize(state.mesh)
        return payload
    for case,cfg in CASES.items():
        state=build_loaded_state(V12_MODEL_ID,seed=cfg["seed"])
        if getattr(state,"void_state",None) is not None: raise ValueError("neutrality state has active void capability")
        initial=components(state); operations=[]; failure=None; restart_exact=True
        def advance(value):
            if case=="unload_reload":
                for opening in cfg["openings_m"]:
                    u=value.displacement.copy()
                    u[2*np.asarray(value.boundary.top_nodes)+1]=opening/2
                    u[2*np.asarray(value.boundary.bot_nodes)+1]=-opening/2
                    value=equilibrate(replace(value,displacement=u))
                return value,{"operation":"equilibrium_unload_reload","openings_m":cfg["openings_m"]}
            advanced,audit=event(value,tuple(cfg["endpoint_m"]),transaction_identity="clean-neutrality:"+case)
            return advanced,{"operation":"V12_topology_transaction","executed_operations":audit["operations"],
                "energy_release_J_per_m":audit["energy_release_J_per_m"],
                "hazard_dissipation_J_per_m":audit["hazard_dissipation_J_per_m"]}
        try:
            if case=="checkpoint_restart":
                checkpoint=out/"checkpoints"/(case+"_initial.json")
                write_checkpoint(state,checkpoint); restored=restore_checkpoint(checkpoint)
                direct,direct_trace=advance(state); resumed,resume_trace=advance(restored)
                restart_exact=fingerprint(direct)==fingerprint(resumed) and direct_trace==resume_trace
                state=resumed; operations.append(resume_trace)
            else:
                state,trace=advance(state); operations.append(trace)
        except Exception as error:
            failure={"type":type(error).__name__,"message":str(error)}
        write_checkpoint(state,out/"checkpoints"/(case+"_terminal.json"))
        row={"case_id":case,"input_configuration":cfg,"executed_code_sha":head,
            "model_identity":state.sharp_wake_model_id,"initial_components":initial,
            "terminal_components":components(state),"complete_fingerprint":fingerprint(state),
            "operations":operations,"failure":failure,"restart_exact":restart_exact,
            "void_state_attribute_present":hasattr(state,"void_state"),
            "void_state_inactive":getattr(state,"void_state",None) is None,
            "event_classification":"software_forced_geometry_neutrality_screen_not_physical_cleavage"}
        result.append(row)
    write_json(out/"rows.json",{"schema":SCHEMA,"executed_code_sha":head,"mode":mode,"rows":result,
        "harness_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if clean_head(repository)!=head: raise RuntimeError("neutrality worker head changed during execution")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--base-worktree",type=Path)
    parser.add_argument("--current-worktree",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--worker",choices=("base","disabled"))
    parser.add_argument("--repository",type=Path)
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError("refusing to overwrite neutrality evidence")
    if args.worker:
        worker(args.repository.resolve(),out,args.worker);return
    if args.base_worktree is None: raise ValueError("an actual clean detached Stage-II V3 base worktree is required")
    base=args.base_worktree.resolve();current=args.current_worktree.resolve()
    if clean_head(base)!=STAGE_II_V3_BASE: raise ValueError("wrong authoritative Stage-II V3 base")
    current_sha=clean_head(current)
    source=(base/BASE_AUTHORITY["qualifier"]).read_text()
    if BASE_AUTHORITY["qualifier_schema"] not in source or BASE_AUTHORITY["gate"] not in source:
        raise ValueError("historical qualifier does not establish Stage-II V3 identity")
    for mode,repository in (("base",base),("disabled",current)):
        subprocess.run((sys.executable,str(Path(__file__).resolve()),str(out/mode),"--worker",mode,
            "--repository",str(repository)),cwd=repository,check=True)
    a=json.loads((out/"base/rows.json").read_text());b=json.loads((out/"disabled/rows.json").read_text())
    rows=[compare_case(first,second) for first,second in zip(a["rows"],b["rows"])]
    payload={"schema":SCHEMA,"base_sha":STAGE_II_V3_BASE,"current_sha":current_sha,
        "base_authority":BASE_AUTHORITY,"case_registry":CASES,"rows":rows,
        "decision":"PASS" if all(r["passed"] for r in rows) else "FAIL_FULL_STATE_EQUALITY",
        "comparison_exclusions":[],"model_identity_retained":True}
    write_json(out/"neutrality_comparison.json",payload)
    write_json(out/"sha256_manifest.json",{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps({"decision":payload["decision"],"cases":len(rows)}),flush=True)


if __name__=="__main__":main()
