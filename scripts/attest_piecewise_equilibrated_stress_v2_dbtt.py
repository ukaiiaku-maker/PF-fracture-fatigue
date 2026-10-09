#!/usr/bin/env python3
"""Run only the frozen D/E central DBTT piecewise-traction decision."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2_contract import (
    ACCEPTANCE_LIMITS, CONTRACT_ID, OPERATOR_ID,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle, require_bound_identity
from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from scripts.attest_cavity_source_shape_regular_v5_dbtt import _run_level, _threshold_identity

OUTPUT = ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/central_dbtt_v7_readiness.json"
CONTRACT = ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/v7_contract.json"
QUALIFICATION = ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/qualification.json"
V6 = ROOT / "artifacts/v6_cavity_source_raw_traction_closure/central_dbtt_v6_readiness.json"


def build_record():
    contract=json.loads(CONTRACT.read_text()); qualification=json.loads(QUALIFICATION.read_text())
    retained=json.loads(V6.read_text())
    if contract["contract"] != CONTRACT_ID or qualification["contract"] != CONTRACT_ID or not qualification["passed"]:
        raise RuntimeError("prospective piecewise verifier qualification is unavailable")
    bundle=material_bundle("DBTT"); initial,_=build_production_void_state(bundle=bundle,enabled=True)
    path=tuple(initial.crack_network.branch(ROOT_BRANCH_ID).path)
    reference={"material_identity":dict(require_bound_identity(initial)),"threshold_identity":None,
               "rng_state":initial.rng_state,"source_coordinate_m":[7.55e-4,0.0],
               "nominal_cavity_geometry":((7.0e-4,0.0),5.5e-5)}
    rows=[_run_level(bundle,path,sectors=128,local_level=level,reference=reference,
                     final_traction_verification=True) for level in ("D","E")]
    connected_thresholds=rows[0]["threshold_identity"]
    for row in rows: row["thresholds_match_reference"]=row["threshold_identity"]==connected_thresholds
    old_rows=retained["fixed_geometry_local_family"]["new_rows"]
    preserved=[]
    for fresh,old in zip(rows,old_rows):
        preserved.append({
            "local_level":fresh["local_level"],
            "raw_scalar_exact":fresh["raw_adjacent_element_traction_normalized"]==old["raw_adjacent_element_traction_normalized"],
            "accepted_state_fingerprint_exact":fresh["accepted_input_state_fingerprint"]==old["accepted_input_state_fingerprint"],
            "complete_boundary_fingerprint_exact":fresh["discrete_cavity_boundary_fingerprints"]["complete_boundary"]==old["discrete_cavity_boundary_fingerprints"]["complete_boundary"],
            "source_window_fingerprint_exact":fresh["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"]==old["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"],
            "arc_rows_recompute_scalar_within_2e_15":bool(np.isclose(
                fresh["arc_resolved_cavity_traction_v1"]["recomputed_raw_adjacent_element_traction_normalized"],
                fresh["raw_adjacent_element_traction_normalized"], rtol=0.0, atol=2.0e-15,
            )),
            "owned_edge_count":fresh["arc_resolved_cavity_traction_v1"]["edge_count"],
        })
    d,e=(row["piecewise_equilibrated_boundary_stress_v2"] for row in rows)
    values=[d.get("normalized_boundary_traction",float("inf")),e.get("normalized_boundary_traction",float("inf"))]
    mesh_change=abs(values[1]-values[0])
    mesh_relative_change=mesh_change/max(abs(values[1]),1.0e-300)
    shared_v6={key:value for key,value in retained["fixed_geometry_local_family"]["predicates"].items()
               if key not in ("raw_traction_E","strict_raw_traction_order_C_D_E")}
    limits=ACCEPTANCE_LIMITS
    predicates={
        "prospective_qualification":qualification["passed"],
        "retained_D_E_state_geometry_and_raw_scalar":all(
            row["raw_scalar_exact"] and row["source_window_fingerprint_exact"]
            and row["arc_rows_recompute_scalar_within_2e_15"] and row["owned_edge_count"]==512
            for row in preserved
        ) and rows[0]["discrete_cavity_boundary_fingerprints"]["complete_boundary"]
            == rows[1]["discrete_cavity_boundary_fingerprints"]["complete_boundary"]
            and all(row["accepted_input_state_unchanged"] for row in rows),
        "retained_shared_v6_gates":all(shared_v6.values()),
        "piecewise_available":d.get("available") is True and e.get("available") is True,
        "piecewise_boundary_traction":max(values)<=limits["normalized_equilibrated_boundary_traction"],
        "piecewise_mesh_change":mesh_change<=limits["traction_mesh_relative_change"]
            and mesh_relative_change<=limits["traction_mesh_relative_change"],
        "piecewise_fit":all(row.get("stress_fit_residual",float("inf"))<=limits["maximum_stress_fit_residual"] for row in (d,e)),
        "piecewise_rank_condition":all(row.get("rank",-1)==row.get("required_rank",-2)
            and row.get("condition",float("inf"))<=limits["maximum_scaled_condition"] for row in (d,e)),
        "piecewise_equilibrium":all(row.get("equilibrium_residual",float("inf"))<=limits["maximum_equilibrium_residual"] for row in (d,e)),
        "piecewise_interface_continuity":all(row.get("interface_traction_jump_residual",float("inf"))<=limits["maximum_interface_traction_jump_relative"] for row in (d,e)),
        "piecewise_resultant_moment":all(row.get("resultant_residual",float("inf"))<=limits["maximum_resultant_residual_relative"]
            and row.get("moment_residual",float("inf"))<=limits["maximum_moment_residual_relative"] for row in (d,e)),
        "zero_cavity_traction_not_imposed":all(row.get("zero_boundary_traction_imposed") is False for row in (d,e)),
    }
    passed=all(predicates.values())
    failures=[key.upper() for key,value in predicates.items() if not value]
    return {"schema":"v7.central-dbtt-piecewise-equilibrated-stress-readiness/1",
            "executed_code_sha":os.environ.get("V7_SOURCE_COMMIT") or subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip(),
            "contract":CONTRACT_ID,"operator":OPERATOR_ID,"qualification":qualification,
            "retained_v6_executed_code_sha":retained["executed_code_sha"],"preservation_checks":preserved,
            "shared_v6_predicates":shared_v6,"rows":rows,
            "piecewise_traction_D":values[0],"piecewise_traction_E":values[1],
            "piecewise_traction_absolute_D_to_E":mesh_change,
            "piecewise_traction_relative_D_to_E":mesh_relative_change,"predicates":predicates,
            "TRACTION_FREE_BOUNDARY_VERIFICATION":("PASS_PIECEWISE_EQUILIBRATED_STRESS" if passed else "FAIL"),
            "RAW_CST_TRACTION":"RETAINED_DIAGNOSTIC_FAIL","DBTT_SOURCE_READINESS":("PASS" if passed else "BLOCKED_WITH_EXACT_FAILURE"),
            "exact_failure":failures,"actual_production_event_executed":False,"oracle_generated":False,
            "oneD_changed":False,"paired_trajectories_run":False,"fatigue_started":False,
            "next_bounded_step":("ACTUAL_PRODUCTION_EVENT" if passed else "STOP_NUMERICAL_POINT_SOURCE_QUALIFICATION")}


def main():
    record=build_record(); destination=Path(os.environ.get("V7_READINESS_OUTPUT",str(OUTPUT)))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("DBTT_SOURCE_READINESS="+record["DBTT_SOURCE_READINESS"])
    print("PIECEWISE_TRACTION_D="+str(record["piecewise_traction_D"]))
    print("PIECEWISE_TRACTION_E="+str(record["piecewise_traction_E"]))
    print("PIECEWISE_CHANGE_D_TO_E="+str(record["piecewise_traction_absolute_D_to_E"]))
    print("FAILURES="+"+".join(record["exact_failure"])); return 0


if __name__=="__main__": raise SystemExit(main())
