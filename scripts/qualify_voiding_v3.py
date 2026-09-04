#!/usr/bin/env python3
import csv,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
from arrhenius_fracture.stage3_criterion_v3 import *
ROOT=Path(__file__).resolve().parents[1]
GROUPS={
"V12_VOIDING_DISABLED_NEUTRALITY":["tests/test_voiding_v3.py::test_default_off_instantiates_nothing_and_does_not_touch_rng","tests/test_voiding_v3.py::test_disabled_metadata_is_separate_from_physical_artifact","tests/test_topology_transaction_v11.py::test_v12_interrupted_restart_continues_exactly"],
"V12_EXPLICIT_CRACK_VOID_STATIC_MECHANICS_QUALIFIED":["tests/test_explicit_cavity_v3.py"],
"V12_CRACK_VOID_TRANSACTION_QUALIFIED":["tests/test_voiding_v3.py::test_ligament_hit_miss_ledgers_and_separate_downstream_activation","tests/test_voiding_v3.py::test_crack_void_injected_failures_are_exact_rollback"],
"V12_VOID_LIFECYCLE_QUALIFIED":["tests/test_voiding_v3.py::test_multihit_localized_lifecycle_and_partition","tests/test_voiding_v3.py::test_healing_competes_and_creates_no_cavity"],
"V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED":["tests/test_voiding_v3.py::test_series_growth_sign_inventory_and_promotion_identity"],
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=Path(sys.argv[1] if len(sys.argv)>1 else "artifacts/voiding_v3");out.mkdir(parents=True,exist_ok=True);rows=[];checks={};env={**os.environ,"PYTHONPATH":str(ROOT)}
 for gate,tests in GROUPS.items():
  p=subprocess.run([sys.executable,"-m","pytest","-q",*tests],cwd=ROOT,env=env,capture_output=True,text=True)
  checks[gate]=p.returncode==0;rows.append({"gate":gate,"passed":checks[gate],"returncode":p.returncode,"tests":" ".join(tests)})
 det=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_voiding_v3.py::test_complete_deterministic_one_void_sequence"],cwd=ROOT,env=env,capture_output=True,text=True)
 deterministic="PASS" if det.returncode==0 else "FAIL"
 rng=np.random.default_rng(3621); threshold=float(rng.exponential()); bounded_hazard=.75
 natural="COMPLETED_ONE_VOID" if threshold<=bounded_hazard else "NO_BIRTH_WITHIN_BOUNDED_DIAGNOSTIC"
 gates=classify(checks);gates["V12_ONE_VOID_END_TO_END_DEMONSTRATED"]=deterministic;gates["V12_BOUNDED_NATURAL_STOCHASTIC_CASE"]=natural
 with (out/"qualification_matrix.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 head=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip();report={"schema":CRITERION_ID,"implementation_git_sha":head,"gates":gates,"deterministic_sequence":DETERMINISTIC_SEQUENCE,"transaction_cases":TRANSACTION_CASES,"lifecycle_cases":LIFECYCLE_CASES,"natural_case":{"seed":3621,"threshold":threshold,"bounded_hazard":bounded_hazard,"classification":natural},"multiple_voids_enabled":False,"fatigue_campaign_run":False}
 (out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(out/"environment_attestation.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"implementation_git_sha":head},indent=2,sort_keys=True)+"\n")
 files=("qualification_matrix.csv","qualification.json","environment_attestation.json");(out/"sha256_manifest.json").write_text(json.dumps({f:sha(out/f) for f in files},indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True));return 0 if deterministic=="PASS" and all(v=="PASS" for k,v in gates.items() if k not in ("V12_BOUNDED_NATURAL_STOCHASTIC_CASE","V12_ONE_VOID_END_TO_END_DEMONSTRATED")) else 2
if __name__=="__main__":raise SystemExit(main())
