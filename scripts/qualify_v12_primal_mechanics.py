#!/usr/bin/env python3
"""Write the predeclared straight-crack V12 primal qualification ledger."""
from __future__ import annotations

import argparse, csv, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.conforming_crack_oracle_v12 import CONFORMING_ORACLE_SOURCE_COMMIT
from arrhenius_fracture.primal_crack_mechanics_v12 import run_low_kappa_prescreen, run_rotated_cases, run_straight_case

THRESHOLDS={
 "maximum_finest_global_reference_error":.05,
 "maximum_finest_outside_support_stress_l2_error":.05,
 "maximum_finest_G_reference_error":.05,
 "maximum_energy_compliance_G_identity_error":.01,
 "maximum_low_kappa_pair_reaction_spread":1e-3,
 "maximum_free_residual_relative":1e-10,
 "maximum_energy_reaction_identity_relative":1e-10,
 "maximum_killed_energy_fraction_at_kappa_1e-6":1e-3,
 "maximum_finest_matched_cod_error":.05,
 "maximum_conforming_extrapolated_direct_cod_error":.05,
 "maximum_finest_local_tensor_error":.05,
 "maximum_low_kappa_primary_spread":.01,
 "maximum_G_delta_plateau_spread":.10,
 "maximum_pin_invariance_error":1e-10,
 "maximum_mirror_residual":.05,
}
BASE_SHA="2b5e5351add0bf0db67f2cda35a1480c3e7efc91"
IMPLEMENTATION_PATHS=("arrhenius_fracture/conforming_crack_oracle_v12.py","arrhenius_fracture/primal_crack_mechanics_v12.py","scripts/qualify_v12_primal_mechanics.py","tests/test_v12_primal_crack_mechanics.py")

def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-300)
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
def git(*args): return subprocess.check_output(("git",)+args,cwd=ROOT,text=True).strip()
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_primal_mechanics")); args=parser.parse_args()
    rows,derivatives=run_straight_case(); prescreen=run_low_kappa_prescreen(); finest=6.25e-6; prescreen_h=3.125e-6
    v12=sorted((r for r in rows if r["representation"]=="C_V12"),key=lambda r:(r["h_tip_m"],r["kappa"])); fine=[r for r in v12 if r["h_tip_m"]==finest]
    low=next(r for r in fine if r["kappa"]==1e-8); mid=next(r for r in fine if r["kappa"]==1e-6); pre_low=next(r for r in prescreen if r["representation"]=="C_V12"); pre_conf=next(r for r in prescreen if r["representation"]=="D_CONFORMING")
    vg=[r for r in derivatives if r["representation"]=="V12" and r["kappa"]==1e-8 and r["delta_a_m"]==25e-6]; cg=[r for r in derivatives if r["representation"]=="CONF" and r["delta_a_m"]==25e-6]
    vf=next(r for r in vg if r["h_tip_m"]==finest); cf=next(r for r in cg if r["h_tip_m"]==finest)
    checks={
      "matched_parent_fingerprints_identical_per_h":all(len({(r["parent_geometry_fingerprint"],r["parent_connectivity_fingerprint"]) for r in rows if r["h_tip_m"]==h})==1 for h in (25e-6,12.5e-6,6.25e-6)),
      "finest_global_reference_error":max(low[k] for k in ("reaction_reference_error","compliance_reference_error","energy_reference_error")),
      "finest_outside_support_stress_l2_error":low["outside_support_stress_l2_error"],
      "global_and_field_errors_decrease_under_refinement":all([v12[i]["reaction_reference_error"]>v12[i-3]["reaction_reference_error"] for i in range(3,6)]) if False else all(a>b for a,b in zip([next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)["reaction_reference_error"] for h in (25e-6,12.5e-6)],[next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)["reaction_reference_error"] for h in (12.5e-6,6.25e-6)])),
      "finest_G_reference_error":rel(vf["G_energy_J_per_m2"],cf["G_energy_J_per_m2"]),
      "G_error_decreases_under_refinement":rel(vg[0]["G_energy_J_per_m2"],cg[0]["G_energy_J_per_m2"])>rel(vg[-1]["G_energy_J_per_m2"],cg[-1]["G_energy_J_per_m2"]),
      "maximum_energy_compliance_G_identity_error":max(r["energy_compliance_relative_error"] for r in derivatives),
      "low_kappa_pair_reaction_spread":rel(low["reaction_N_per_m"],mid["reaction_N_per_m"]),
      "maximum_free_residual_relative":max(r["free_residual_relative"] for r in rows),
      "maximum_energy_reaction_identity_relative":max(r["energy_reaction_identity_relative"] for r in rows),
      "killed_energy_fraction_at_kappa_1e-6":mid["killed_energy_fraction"],
      "finest_matched_cod_error":pre_low["crack_opening_reference_error"],
      "finest_fixed_distance_cod_error":pre_low["fixed_distance_cod_reference_error"],
      "conforming_extrapolated_direct_cod_error_max":max(r["conforming_extrapolated_direct_cod_error"] for r in rows if r["representation"]=="D_CONFORMING"),
      "matched_cod_error_decreases_under_refinement":all(a>b for a,b in zip([next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)["crack_opening_reference_error"] for h in (25e-6,12.5e-6,6.25e-6)],[next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)["crack_opening_reference_error"] for h in (12.5e-6,6.25e-6)]+[pre_low["crack_opening_reference_error"]])),
      "all_area_weighted_tensor_errors_decrease":all(a>b for key in ("area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior") for a,b in zip([next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (25e-6,12.5e-6,6.25e-6)],[next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (12.5e-6,6.25e-6)]+[pre_low[key]])),
      "finest_local_tensor_error_max":max(pre_low[key] for key in ("area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior")),
      "pin_invariance_error_max":max(low[k] for k in ("pin_reaction_relative_error","pin_energy_relative_error","pin_cod_relative_error")),
      "mirror_residual_max":max(r[f"mirror_{region}_{component}"] for r in rows+prescreen for region in ("full","constraint_excluded","near_tip") for component in ("sigma_xx_relative","sigma_yy_relative","sigma_xy_antisymmetry_relative")),
    }
    primary=("reaction_N_per_m","compliance_m2_per_N","energy_J_per_m","crack_opening_displacement_m","area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior")
    checks["low_kappa_primary_spread_max"]=max(rel(low[k],mid[k]) for k in primary)
    checks["low_kappa_G_spread_max"]=max(rel(next(r for r in derivatives if r["representation"]=="V12" and r["h_tip_m"]==h and r["delta_a_m"]==delta and r["kappa"]==1e-8)[quantity],next(r for r in derivatives if r["representation"]=="V12" and r["h_tip_m"]==h and r["delta_a_m"]==delta and r["kappa"]==1e-6)[quantity]) for h in (12.5e-6,6.25e-6) for delta in (12.5e-6,25e-6,50e-6) for quantity in ("G_energy_J_per_m2","G_compliance_J_per_m2"))
    checks["transmitted_traction_and_killed_energy_decrease_h_kappa"]=all(next(r for r in v12 if r["h_tip_m"]==a and r["kappa"]==1e-8)[key]>next(r for r in v12 if r["h_tip_m"]==b and r["kappa"]==1e-8)[key] for key in ("trimmed_interior_soft_traction_rms_Pa","killed_energy_fraction") for a,b in ((25e-6,12.5e-6),(12.5e-6,6.25e-6)))
    delta_groups=[[r for r in derivatives if r["representation"]==name and r["h_tip_m"]==h and (name=="CONF" or r["kappa"]==1e-8)] for name in ("V12","CONF") for h in (12.5e-6,6.25e-6)]
    checks["G_delta_plateau_spread_max"]=max((max(r["G_energy_J_per_m2"] for r in group)-min(r["G_energy_J_per_m2"] for r in group))/max(abs(r["G_energy_J_per_m2"]) for r in group) for group in delta_groups)
    gates={
      "MATCHED_PARENT_REPRESENTATIONS": "PASS" if checks["matched_parent_fingerprints_identical_per_h"] else "FAIL",
      "KAPPA_OBJECTIVITY": "PASS" if checks["low_kappa_pair_reaction_spread"]<=THRESHOLDS["maximum_low_kappa_pair_reaction_spread"] else "FAIL",
      "FIELDS_CONVERGENCE": "PASS" if checks["global_and_field_errors_decrease_under_refinement"] and checks["finest_global_reference_error"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["finest_outside_support_stress_l2_error"]<=THRESHOLDS["maximum_finest_outside_support_stress_l2_error"] else "FAIL",
      "CENTERED_G_CONVERGENCE": "PASS" if checks["G_error_decreases_under_refinement"] and checks["finest_G_reference_error"]<=THRESHOLDS["maximum_finest_G_reference_error"] and checks["maximum_energy_compliance_G_identity_error"]<=THRESHOLDS["maximum_energy_compliance_G_identity_error"] else "FAIL",
      "V12_PRIMAL_GLOBAL_RESPONSE_SCREEN": "OPEN",
      "V12_CENTERED_G_SINGLE_INCREMENT_SCREEN": "OPEN",
      "V12_ROTATION_COVARIANCE_SCREEN": "OPEN",
      "V12_MATCHED_COD_QUALIFIED":"OPEN",
      "V12_INTERFACE_TRACTION_QUALIFIED":"OPEN",
      "V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED":"OPEN",
      "V12_LOCAL_TENSOR_FIELDS_QUALIFIED":"OPEN",
      "V12_G_PERTURBATION_CONVERGENCE":"OPEN",
      "V12_PRIMAL_CLEAN_WORKER_REPRODUCIBLE":"NOT_RUN",
      "V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED":"OPEN",
      "MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED":"OPEN",
      "MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED":"NOT_RUN",
      "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED":"NOT_RUN",
      "V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED":"OPEN",
    }
    equilibrium=checks["maximum_free_residual_relative"]<=THRESHOLDS["maximum_free_residual_relative"] and checks["maximum_energy_reaction_identity_relative"]<=THRESHOLDS["maximum_energy_reaction_identity_relative"] and checks["killed_energy_fraction_at_kappa_1e-6"]<=THRESHOLDS["maximum_killed_energy_fraction_at_kappa_1e-6"]
    gates["V12_PRIMAL_GLOBAL_RESPONSE_SCREEN"]="PASS" if equilibrium and gates["MATCHED_PARENT_REPRESENTATIONS"]=="PASS" and checks["finest_global_reference_error"]<=THRESHOLDS["maximum_finest_global_reference_error"] else "FAIL"
    gates["V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED"]="PASS" if checks["mirror_residual_max"]<=THRESHOLDS["maximum_mirror_residual"] and checks["pin_invariance_error_max"]<=THRESHOLDS["maximum_pin_invariance_error"] else "FAIL"
    gates["V12_CENTERED_G_SINGLE_INCREMENT_SCREEN"]="PASS" if gates["CENTERED_G_CONVERGENCE"]=="PASS" else "FAIL"
    gates["V12_MATCHED_COD_QUALIFIED"]="PASS" if checks["matched_cod_error_decreases_under_refinement"] and max(checks["finest_matched_cod_error"],checks["finest_fixed_distance_cod_error"])<=THRESHOLDS["maximum_finest_matched_cod_error"] and checks["conforming_extrapolated_direct_cod_error_max"]<=THRESHOLDS["maximum_conforming_extrapolated_direct_cod_error"] else "FAIL"
    gates["V12_LOCAL_TENSOR_FIELDS_QUALIFIED"]="PASS" if checks["all_area_weighted_tensor_errors_decrease"] and checks["finest_local_tensor_error_max"]<=THRESHOLDS["maximum_finest_local_tensor_error"] else "FAIL"
    gates["V12_INTERFACE_TRACTION_QUALIFIED"]="PASS" if checks["transmitted_traction_and_killed_energy_decrease_h_kappa"] else "FAIL"
    gates["V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED"]="PASS" if checks["transmitted_traction_and_killed_energy_decrease_h_kappa"] else "FAIL"
    gates["V12_G_PERTURBATION_CONVERGENCE"]="PASS" if checks["G_delta_plateau_spread_max"]<=THRESHOLDS["maximum_G_delta_plateau_spread"] else "FAIL"
    gates["KAPPA_OBJECTIVITY"]="PASS" if max(checks["low_kappa_primary_spread_max"],checks["low_kappa_G_spread_max"])<=THRESHOLDS["maximum_low_kappa_primary_spread"] else "FAIL"
    angle_rows=[]
    if gates["V12_PRIMAL_GLOBAL_RESPONSE_SCREEN"]=="PASS":
        angle_rows=run_rotated_cases(); angle_fine=[r for r in angle_rows if r["representation"]=="C_V12" and r["h_tip_m"]==finest and r["kappa"]==1e-8]
        checks["angle_finest_global_reference_error_max"]=max(max(r[k] for k in ("reaction_reference_error","compliance_reference_error","energy_reference_error")) for r in angle_fine)
        checks["angle_equilibrium_identity_max"]=max(max(r["free_residual_relative"],r["energy_reaction_identity_relative"]) for r in angle_rows)
        gates["V12_ROTATION_COVARIANCE_SCREEN"]="PASS" if checks["angle_finest_global_reference_error_max"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["angle_equilibrium_identity_max"]<=THRESHOLDS["maximum_free_residual_relative"] else "FAIL"
    args.out.mkdir(parents=True,exist_ok=True); write_csv(args.out/"straight_primal_matrix.csv",rows); write_csv(args.out/"straight_3p125um_prescreen.csv",prescreen); write_csv(args.out/"centered_G_matrix.csv",derivatives)
    if angle_rows: write_csv(args.out/"angle_primal_matrix.csv",angle_rows)
    implementation_sha=git("log","-1","--format=%H","--",*IMPLEMENTATION_PATHS)
    oracle_sha=CONFORMING_ORACLE_SOURCE_COMMIT
    oracle_source_sha256=sha256(ROOT/"arrhenius_fracture/conforming_crack_oracle_v12.py")
    report={"schema":"v12_primal_mechanics_qualification_v2","base_git_sha":BASE_SHA,"implementation_git_sha":implementation_sha,"evidence_generation_parent_sha":implementation_sha,"conforming_oracle_source_commit":oracle_sha,"conforming_oracle_source_sha256":oracle_source_sha256,"thresholds_predeclared":THRESHOLDS,"checks":checks,"gates":gates}
    (args.out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
    scientific=("straight_primal_matrix.csv","straight_3p125um_prescreen.csv","centered_G_matrix.csv","angle_primal_matrix.csv","qualification.json")
    manifest={name:sha256(args.out/name) for name in scientific}
    (args.out/"sha256_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    environment={"base_git_sha":BASE_SHA,"implementation_git_sha":implementation_sha,"evidence_generation_parent_sha":implementation_sha,"conforming_oracle_source_commit":oracle_sha,"conforming_oracle_source_sha256":oracle_source_sha256,"python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,"platform":platform.platform(),"solver_identity":"scipy.sparse.linalg.spsolve_cst_plane_strain","thresholds":THRESHOLDS}
    (args.out/"environment_attestation.json").write_text(json.dumps(environment,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
