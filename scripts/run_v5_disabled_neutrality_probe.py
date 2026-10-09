#!/usr/bin/env python3
"""Execute Stage-II trajectories without importing code from the caller tree."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

WORKTREE=Path.cwd()
sys.path.insert(0,str(WORKTREE))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.sharp_wake_backend_v12 import V11_MODEL_ID
from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
from arrhenius_fracture.v12_production_driver import (
    _observables, build_loaded_state, execute_event, run_trajectory,
)

def physical(observation):
    if isinstance(observation,dict):
        return {key:physical(value) for key,value in observation.items()
                if key not in {"fingerprint","model","source_commit","implementation_git_sha"}}
    if isinstance(observation,list): return [physical(value) for value in observation]
    return observation

def main(argv=None):
    output=Path((argv or sys.argv[1:])[0]); rows=[]
    for name,geometry in (("monotonic","straight"),("oblique","oblique")):
        result=run_trajectory(V11_MODEL_ID,geometry)
        rows.append({"trajectory":name,"initial":physical(result["initial"]),
                     "terminal":physical(result["final"]),"events":physical(result["events"])})
    state=build_loaded_state(V11_MODEL_ID); temporary=Path(tempfile.mkdtemp(prefix="v5-neutral-restart-"))/"state.json"
    write_checkpoint(state,temporary); restored=restore_checkpoint(temporary)
    uninterrupted,event_a=execute_event(state,(5.25e-4,0.),transaction_identity="neutral-restart")
    restarted,event_b=execute_event(restored,(5.25e-4,0.),transaction_identity="neutral-restart")
    rows.append({"trajectory":"checkpoint_restart","initial":physical(_observables(state)),
      "terminal":physical(_observables(restarted)),"uninterrupted_terminal":physical(_observables(uninterrupted)),
      "events":physical([event_a,event_b]),"internal_restart_equal":physical(_observables(uninterrupted))==physical(_observables(restarted))})
    state=build_loaded_state(V11_MODEL_ID); displacement=np.asarray(state.displacement).copy()
    unloaded=displacement.copy(); unloaded[2*np.asarray(state.boundary.top_nodes)+1]=0.; unloaded[2*np.asarray(state.boundary.bot_nodes)+1]=0.
    state=equilibrate_fixed_load_with_production_fem(replace(state,displacement=unloaded))
    state=equilibrate_fixed_load_with_production_fem(replace(state,displacement=displacement))
    rows.append({"trajectory":"unload_reload","initial_loading_m":4e-7,"unloaded_opening_m":0.,
                 "terminal":physical(_observables(state)),"events":[]})
    payload={"schema":"v5.disabled-neutrality-probe/1","voiding_enabled":False,"rows":rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(hashlib.sha256(output.read_bytes()).hexdigest())
    return 0

if __name__=="__main__": raise SystemExit(main())
