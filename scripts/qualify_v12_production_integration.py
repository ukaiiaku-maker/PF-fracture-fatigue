#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
from arrhenius_fracture.stage2_criterion_v12 import *

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
"V11_SELECTABLE_NEUTRALITY":["tests/test_topology_transaction_v11.py::test_explicit_v11_selection_is_physically_identical_to_default","tests/test_sharp_wake_backend_v12.py::test_default_remains_v11_and_selection_is_explicit"],
"V12_PRODUCTION_STATE_OWNERSHIP_QUALIFIED":["tests/test_topology_transaction_v11.py::test_v12_initialization_and_checkpoint_own_certified_support","tests/test_sharp_wake_backend_v12.py"],
"PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED":["tests/test_topology_transaction_v11.py::test_injected_transaction_failure_leaves_exact_accepted_state","tests/test_topology_transaction_v11.py::test_v12_geometry_failure_injection_preserves_accepted_state","tests/test_topology_transaction_v11.py::test_active_tip_remesh_failure_preserves_accepted_state"],
"V12_PRODUCTION_CHECKPOINT_RESTART_QUALIFIED":["tests/test_topology_transaction_v11.py::test_v12_interrupted_restart_continues_exactly"],
"V12_BOUNDED_PRODUCTION_PROPAGATION_QUALIFIED":["tests/test_topology_transaction_v11.py::test_v12_event_commits_exact_graph_length_and_certified_mechanical_state","tests/test_topology_transaction_v11.py::test_short_event_is_resolved_by_bounded_tip_refinement_not_length_change","tests/test_v12_mechanically_separating_wake.py::test_kink_partition_and_sequential_history_are_mechanically_equivalent","tests/test_v12_mechanically_separating_wake.py::test_y_arm_subdivision_and_sequential_history_are_mechanically_equivalent","tests/test_v12_mechanically_separating_wake.py::test_offaxis_and_endpoint_phases_never_silently_fall_back"],
}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 out=Path(sys.argv[1] if len(sys.argv)>1 else "artifacts/v12_production_integration"); out.mkdir(parents=True,exist_ok=True)
 rows=[]; checks={}; env={**os.environ,"PYTHONPATH":str(ROOT)}
 for gate,tests in GROUPS.items():
  p=subprocess.run([sys.executable,"-m","pytest","-q",*tests],cwd=ROOT,env=env,text=True,capture_output=True)
  checks[gate]=p.returncode==0
  rows.append({"gate":gate,"passed":checks[gate],"returncode":p.returncode,"tests":" ".join(tests),"stdout_sha256":hashlib.sha256(p.stdout.encode()).hexdigest(),"stderr_sha256":hashlib.sha256(p.stderr.encode()).hexdigest()})
 gates=qualify(checks); gates["V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED"]=prerequisite(gates)
 with (out/"qualification_matrix.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 sources=("arrhenius_fracture/sharp_wake_backend_v12.py","arrhenius_fracture/topology_transaction_v11.py","arrhenius_fracture/checkpoint_v11.py","arrhenius_fracture/stage2_criterion_v12.py","tests/test_topology_transaction_v11.py","tests/test_sharp_wake_backend_v12.py")
 ownership={"schema":CRITERION_ID,"sources":{p:digest(ROOT/p) for p in sources},"stale_element_ids_authoritative_after_remesh":False,"absolute_K_consumed":False}
 (out/"state_ownership_manifest.json").write_text(json.dumps(ownership,indent=2,sort_keys=True)+"\n")
 head=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip(); report={"schema":CRITERION_ID,"implementation_git_sha":head,"checks":checks,"gates":gates,"rollback_stages":ROLLBACK_STAGES,"restart_fields":RESTART_FIELDS,"v11_neutrality_fields":V11_NEUTRALITY_FIELDS,"bounded_cases":BOUNDED_CASES}
 (out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (out/"environment_attestation.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"implementation_git_sha":head},indent=2,sort_keys=True)+"\n")
 files=("qualification_matrix.csv","state_ownership_manifest.json","qualification.json","environment_attestation.json")
 (out/"sha256_manifest.json").write_text(json.dumps({p:digest(out/p) for p in files},indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2,sort_keys=True)); return 0 if gates["V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
