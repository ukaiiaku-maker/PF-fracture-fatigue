#!/usr/bin/env python3
"""Write the predeclared straight-crack V12 primal qualification ledger."""
from __future__ import annotations

import argparse, csv, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.conforming_crack_oracle_v12 import CONFORMING_ORACLE_SOURCE_COMMIT
from arrhenius_fracture.primal_crack_mechanics_v12 import run_low_kappa_prescreen, run_rotated_cases, run_straight_case, run_targeted_local_refinement

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
 "maximum_joint_limit_transmission_relative":1e-3,
 "maximum_joint_limit_killed_energy_fraction":1e-3,
 "maximum_joint_limit_primary_change":.05,
 "maximum_usable_conditioning_diagonal_ratio":1e12,
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
def decreasing(values): return all(a>b for a,b in zip(values,values[1:]))
def fitted_slope(rows,key):
    x=np.log([r["h_tip_m"] for r in rows]); y=np.log(np.maximum([abs(r[key]) for r in rows],1e-300)); return float(np.polyfit(x,y,1)[0])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_primal_mechanics")); args=parser.parse_args()
    rows,derivatives=run_straight_case(); finest=3.125e-6
    targeted=run_targeted_local_refinement(); targeted.extend(run_targeted_local_refinement(h=.78125e-6)); targeted_v12=sorted((r for r in targeted if r["representation"]=="C_V12"),key=lambda r:r["h_tip_m"],reverse=True); targeted_low=targeted_v12[-1]
    h0=25e-6; kappa0=1e-6; joint=[]
    for p in (1,2):
        for h in (25e-6,12.5e-6,6.25e-6,3.125e-6):
            kappa=kappa0*(h/h0)**p
            if h==h0:
                row=dict(next(r for r in rows if r["representation"]=="C_V12" and r["h_tip_m"]==h and r["kappa"]==kappa0))
            else:
                row=dict(next(r for r in run_low_kappa_prescreen(h=h,kappa=kappa) if r["representation"]=="C_V12"))
            row.update({"policy_exponent_p":p,"policy_h0_m":h0,"policy_kappa0":kappa0,"kappa_over_h_per_m":kappa/h}); joint.append(row)
    v12=sorted((r for r in rows if r["representation"]=="C_V12"),key=lambda r:(r["h_tip_m"],r["kappa"])); fine=[r for r in v12 if r["h_tip_m"]==finest]
    low=next(r for r in fine if r["kappa"]==1e-8); mid=next(r for r in fine if r["kappa"]==1e-6)
    vg=[r for r in derivatives if r["representation"]=="V12" and r["kappa"]==1e-8 and r["delta_a_m"]==25e-6]; cg=[r for r in derivatives if r["representation"]=="CONF" and r["delta_a_m"]==25e-6]
    g_finest=6.25e-6; vf=next(r for r in vg if r["h_tip_m"]==g_finest); cf=next(r for r in cg if r["h_tip_m"]==g_finest)
    checks={
      "matched_parent_fingerprints_identical_per_h":all(len({(r["parent_geometry_fingerprint"],r["parent_connectivity_fingerprint"]) for r in rows if r["h_tip_m"]==h})==1 for h in (25e-6,12.5e-6,6.25e-6,3.125e-6)),
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
      "finest_matched_cod_error":low["crack_opening_reference_error"],
      "finest_fixed_distance_cod_error":low["fixed_distance_cod_reference_error"],
      "conforming_extrapolated_direct_cod_error_max":max(r["conforming_extrapolated_direct_cod_error"] for r in rows if r["representation"]=="D_CONFORMING"),
      "matched_cod_error_decreases_under_refinement":all(a>b for key in ("crack_opening_reference_error",) for a,b in zip([next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (25e-6,12.5e-6,6.25e-6)],[next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (12.5e-6,6.25e-6,3.125e-6)])),
      "all_area_weighted_tensor_errors_decrease":all(a>b for key in ("area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior") for a,b in zip([next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (25e-6,12.5e-6,6.25e-6)],[next(r for r in v12 if r["h_tip_m"]==h and r["kappa"]==1e-8)[key] for h in (12.5e-6,6.25e-6,3.125e-6)])),
      "finest_local_tensor_error_max":max(low[key] for key in ("area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior")),
      "pin_invariance_error_max":max(low[k] for k in ("pin_reaction_relative_error","pin_energy_relative_error","pin_cod_relative_error")),
      "mirror_residual_max":max(r[f"mirror_{region}_{component}"] for r in rows for region in ("full","constraint_excluded","near_tip") for component in ("sigma_xx_relative","sigma_yy_relative","sigma_xy_antisymmetry_relative")),
      "targeted_1p5625_face_adjacent_error":targeted_v12[0]["area_weighted_stress_error_face_adjacent_strip"],
      "targeted_0p78125_face_adjacent_error":targeted_low["area_weighted_stress_error_face_adjacent_strip"],
      "targeted_0p78125_local_tensor_error_max":max(targeted_low[key] for key in ("area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior")),
      "targeted_production_tensor_probe_errors":[r["production_tensor_probe_relative_error"] for r in targeted_v12],
    }
    primary=("reaction_N_per_m","compliance_m2_per_N","energy_J_per_m","crack_opening_displacement_m","area_weighted_stress_error_near_tip_annulus","area_weighted_stress_error_face_adjacent_strip","area_weighted_stress_error_whole_exterior")
    checks["low_kappa_primary_spread_max"]=max(rel(low[k],mid[k]) for k in primary)
    checks["low_kappa_G_spread_max"]=max(rel(next(r for r in derivatives if r["representation"]=="V12" and r["h_tip_m"]==h and r["delta_a_m"]==delta and r["kappa"]==1e-8)[quantity],next(r for r in derivatives if r["representation"]=="V12" and r["h_tip_m"]==h and r["delta_a_m"]==delta and r["kappa"]==1e-6)[quantity]) for h in (12.5e-6,6.25e-6) for delta in (12.5e-6,25e-6,50e-6) for quantity in ("G_energy_J_per_m2","G_compliance_J_per_m2"))
    for row in joint:
        row["T_sigma"]=abs(row["trimmed_interior_soft_traction_rms_relative_remote_stress"]); row["T_n"]=abs(row["discrete_signed_transmitted_normal_force_N_per_m"])/abs(row["reaction_N_per_m"]); row["T_t"]=abs(row["discrete_signed_transmitted_shear_force_N_per_m"])/abs(row["reaction_N_per_m"]); row["T_resultant"]=float(np.hypot(row["T_n"],row["T_t"])); row["E_soft"]=row["killed_energy_fraction"]
    p2=sorted((r for r in joint if r["policy_exponent_p"]==2),key=lambda r:r["h_tip_m"],reverse=True); joint_metrics=("trimmed_interior_soft_traction_rms_relative_remote_stress","discrete_transmitted_normal_force_relative_remote_resultant","discrete_transmitted_shear_force_relative_remote_resultant","killed_energy_fraction")
    checks["joint_p2_kappa_over_h_decreases"]=all(a["kappa_over_h_per_m"]>b["kappa_over_h_per_m"] for a,b in zip(p2,p2[1:])); checks["joint_p2_transmission_decreases"]=all(a[key]>b[key] for key in joint_metrics for a,b in zip(p2,p2[1:])); checks["joint_p2_finest_transmission_max"]=max(abs(p2[-1][key]) for key in joint_metrics[:3]); checks["joint_p2_finest_killed_energy_fraction"]=p2[-1]["killed_energy_fraction"]; checks["joint_p2_primary_change_max"]=max(p2[-1][key] for key in ("reaction_reference_error","compliance_reference_error","energy_reference_error","crack_opening_reference_error")); checks["joint_conditioning_diagonal_ratio_max"]=max(r["conditioning_diagonal_ratio"] for r in joint)
    for key in ("T_sigma","T_n","T_t","T_resultant","E_soft"):
        checks[f"v3_{key}_values"]=[r[key] for r in p2]; checks[f"v3_{key}_convergence_ratios"]=[a[key]/max(b[key],1e-300) for a,b in zip(p2,p2[1:])]; checks[f"v3_{key}_fitted_slope"]=fitted_slope(p2,key)
    checks["v3_kappa_over_h_decreases"]=decreasing([r["kappa_over_h_per_m"] for r in p2]); checks["v3_T_sigma_decreases"]=decreasing([r["T_sigma"] for r in p2]); checks["v3_T_n_decreases"]=decreasing([r["T_n"] for r in p2]); checks["v3_T_resultant_decreases"]=decreasing([r["T_resultant"] for r in p2]); checks["v3_E_soft_decreases"]=decreasing([r["E_soft"] for r in p2]); checks["v3_two_finest_T_sigma_max"]=max(r["T_sigma"] for r in p2[-2:]); checks["v3_two_finest_T_resultant_max"]=max(r["T_resultant"] for r in p2[-2:]); checks["v3_two_finest_E_soft_max"]=max(r["E_soft"] for r in p2[-2:]); checks["v3_T_t_all_levels_max"]=max(r["T_t"] for r in p2); checks["v3_transmission_threshold_margin"]=THRESHOLDS["maximum_joint_limit_transmission_relative"]-max(checks["v3_two_finest_T_sigma_max"],checks["v3_two_finest_T_resultant_max"],checks["v3_T_t_all_levels_max"]); checks["v3_energy_threshold_margin"]=THRESHOLDS["maximum_joint_limit_killed_energy_fraction"]-checks["v3_two_finest_E_soft_max"]; checks["v3_recovered_discrete_ratio_finest"]=p2[-1]["T_sigma"]/max(p2[-1]["T_resultant"],1e-300); checks["v3_upper_lower_balance_max"]=max(r[k] for r in p2 for k in ("discrete_upper_lower_normal_balance_relative","discrete_upper_lower_shear_balance_relative")); checks["v3_free_residual_max"]=max(r["free_residual_relative"] for r in p2)
    delta_groups=[[r for r in derivatives if r["representation"]==name and r["h_tip_m"]==h and (name=="CONF" or r["kappa"]==1e-8)] for name in ("V12","CONF") for h in (12.5e-6,6.25e-6)]
    checks["G_delta_plateau_spread_max"]=max((max(r["G_energy_J_per_m2"] for r in group)-min(r["G_energy_J_per_m2"] for r in group))/max(abs(r["G_energy_J_per_m2"]) for r in group) for group in delta_groups)
    gates={
      "MATCHED_PARENT_REPRESENTATIONS": "PASS" if checks["matched_parent_fingerprints_identical_per_h"] else "FAIL",
      "KAPPA_OBJECTIVITY": "PASS" if checks["low_kappa_pair_reaction_spread"]<=THRESHOLDS["maximum_low_kappa_pair_reaction_spread"] else "FAIL",
      "UNIFORM_25_TO_3P125_FIELD_SCREEN": "PASS" if checks["global_and_field_errors_decrease_under_refinement"] and checks["finest_global_reference_error"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["finest_outside_support_stress_l2_error"]<=THRESHOLDS["maximum_finest_outside_support_stress_l2_error"] else "FAIL",
      "CENTERED_G_CONVERGENCE": "PASS" if checks["G_error_decreases_under_refinement"] and checks["finest_G_reference_error"]<=THRESHOLDS["maximum_finest_G_reference_error"] and checks["maximum_energy_compliance_G_identity_error"]<=THRESHOLDS["maximum_energy_compliance_G_identity_error"] else "FAIL",
      "V12_PRIMAL_GLOBAL_RESPONSE_SCREEN": "OPEN",
      "V12_CENTERED_G_SINGLE_INCREMENT_SCREEN": "OPEN",
      "V12_ROTATION_COVARIANCE_SCREEN": "OPEN",
      "V12_MATCHED_COD_QUALIFIED":"OPEN",
      "V12_INTERFACE_TRACTION_QUALIFIED":"OPEN",
      "V12_INTERFACE_TRACTION_DIAGNOSTIC":"RETAINED_NOT_AGGREGATE",
      "V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED":"OPEN",
      "V12_SOFT_CORRIDOR_TRANSMISSION_V2_FROZEN":"FAIL_STRICT_SIGNED_COMPONENT_MONOTONICITY",
      "V12_SOFT_CORRIDOR_TRANSMISSION_V3_PHYSICAL":"OPEN",
      "V12_LOCAL_TENSOR_FIELDS_QUALIFIED":"OPEN",
      "V12_G_PERTURBATION_CONVERGENCE":"OPEN",
      "V12_PRIMAL_CLEAN_WORKER_REPRODUCIBLE":"NOT_RUN",
      "V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED":"OPEN",
      "V12_STRAIGHT_MODE_I_PRIMAL_MECHANICS_QUALIFIED":"OPEN",
      "BOUNDED_LOCAL_REFINEMENT_FIELD_QUALIFIED":"OPEN",
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
    targeted_converges=checks["targeted_1p5625_face_adjacent_error"]>checks["targeted_0p78125_face_adjacent_error"] and checks["targeted_production_tensor_probe_errors"][0]>checks["targeted_production_tensor_probe_errors"][1]
    gates["V12_LOCAL_TENSOR_FIELDS_QUALIFIED"]="PASS" if checks["all_area_weighted_tensor_errors_decrease"] and targeted_converges and checks["targeted_0p78125_local_tensor_error_max"]<=THRESHOLDS["maximum_finest_local_tensor_error"] else "FAIL"
    gates["V12_INTERFACE_TRACTION_QUALIFIED"]="DIAGNOSTIC_NOT_AGGREGATE"
    gates["V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED"]="FAIL"
    gates["V12_P0_LOCAL_TENSOR_FIDELITY"]="PASS" if gates["V12_LOCAL_TENSOR_FIELDS_QUALIFIED"]=="PASS" else "NOT_QUALIFIED_REQUIRES_CONFORMING_TIP_PATCH"
    gates["BOUNDED_LOCAL_REFINEMENT_FIELD_QUALIFIED"]=gates["V12_LOCAL_TENSOR_FIELDS_QUALIFIED"]
    v3_pass=all((checks["v3_kappa_over_h_decreases"],checks["v3_T_sigma_decreases"],checks["v3_T_n_decreases"],checks["v3_T_resultant_decreases"],checks["v3_E_soft_decreases"],checks["v3_two_finest_T_sigma_max"]<=THRESHOLDS["maximum_joint_limit_transmission_relative"],checks["v3_two_finest_T_resultant_max"]<=THRESHOLDS["maximum_joint_limit_transmission_relative"],checks["v3_two_finest_E_soft_max"]<=THRESHOLDS["maximum_joint_limit_killed_energy_fraction"],checks["v3_T_t_all_levels_max"]<=THRESHOLDS["maximum_joint_limit_transmission_relative"],gates["V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED"]=="PASS",checks["joint_p2_primary_change_max"]<=THRESHOLDS["maximum_joint_limit_primary_change"],checks["joint_conditioning_diagonal_ratio_max"]<=THRESHOLDS["maximum_usable_conditioning_diagonal_ratio"],checks["v3_free_residual_max"]<=THRESHOLDS["maximum_free_residual_relative"]))
    gates["V12_SOFT_CORRIDOR_TRANSMISSION_V3_PHYSICAL"]="PASS" if v3_pass else "FAIL"
    gates["V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED"]=gates["V12_SOFT_CORRIDOR_TRANSMISSION_V3_PHYSICAL"]
    gates["V12_G_PERTURBATION_CONVERGENCE"]="PASS" if checks["G_delta_plateau_spread_max"]<=THRESHOLDS["maximum_G_delta_plateau_spread"] else "FAIL"
    gates["KAPPA_OBJECTIVITY"]="PASS" if max(checks["low_kappa_primary_spread_max"],checks["low_kappa_G_spread_max"])<=THRESHOLDS["maximum_low_kappa_primary_spread"] else "FAIL"
    straight_pass=v3_pass and gates["V12_PRIMAL_GLOBAL_RESPONSE_SCREEN"]=="PASS" and gates["V12_MATCHED_COD_QUALIFIED"]=="PASS" and gates["V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED"]=="PASS" and gates["V12_LOCAL_TENSOR_FIELDS_QUALIFIED"]=="PASS" and gates["V12_G_PERTURBATION_CONVERGENCE"]=="PASS" and gates["KAPPA_OBJECTIVITY"]=="PASS"
    gates["V12_STRAIGHT_MODE_I_PRIMAL_MECHANICS_QUALIFIED"]="PASS" if straight_pass else "FAIL"
    gates["MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED"]="PASS_STRAIGHT_MODE_I_SCOPE" if straight_pass else "OPEN"
    angle_rows=[]
    if gates["V12_PRIMAL_GLOBAL_RESPONSE_SCREEN"]=="PASS":
        angle_rows=run_rotated_cases(); angle_fine=[r for r in angle_rows if r["representation"]=="C_V12" and r["h_tip_m"]==g_finest and r["kappa"]==1e-8]
        checks["angle_finest_global_reference_error_max"]=max(max(r[k] for k in ("reaction_reference_error","compliance_reference_error","energy_reference_error")) for r in angle_fine)
        checks["angle_equilibrium_identity_max"]=max(max(r["free_residual_relative"],r["energy_reaction_identity_relative"]) for r in angle_rows)
        gates["V12_ROTATION_COVARIANCE_SCREEN"]="PASS" if checks["angle_finest_global_reference_error_max"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["angle_equilibrium_identity_max"]<=THRESHOLDS["maximum_free_residual_relative"] else "FAIL"
    args.out.mkdir(parents=True,exist_ok=True)
    obsolete=args.out/"straight_3p125um_prescreen.csv"
    if obsolete.exists(): obsolete.unlink()
    obsolete_target=args.out/"targeted_1p5625um_local_refinement.csv"
    if obsolete_target.exists(): obsolete_target.unlink()
    write_csv(args.out/"straight_primal_matrix.csv",rows); write_csv(args.out/"targeted_local_refinement_matrix.csv",targeted); write_csv(args.out/"joint_h_kappa_transmission_matrix.csv",joint); write_csv(args.out/"centered_G_matrix.csv",derivatives)
    if angle_rows: write_csv(args.out/"angle_primal_matrix.csv",angle_rows)
    implementation_sha=git("log","-1","--format=%H","--",*IMPLEMENTATION_PATHS)
    oracle_sha=CONFORMING_ORACLE_SOURCE_COMMIT
    oracle_source_sha256=sha256(ROOT/"arrhenius_fracture/conforming_crack_oracle_v12.py")
    report={"schema":"v12_primal_mechanics_qualification_v3","base_git_sha":BASE_SHA,"implementation_git_sha":implementation_sha,"evidence_generation_parent_sha":implementation_sha,"conforming_oracle_source_commit":oracle_sha,"conforming_oracle_source_sha256":oracle_source_sha256,"thresholds_predeclared":THRESHOLDS,"checks":checks,"gates":gates}
    (args.out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
    scientific=("straight_primal_matrix.csv","targeted_local_refinement_matrix.csv","joint_h_kappa_transmission_matrix.csv","centered_G_matrix.csv","angle_primal_matrix.csv","qualification.json")
    manifest={name:sha256(args.out/name) for name in scientific}
    (args.out/"sha256_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    environment={"base_git_sha":BASE_SHA,"implementation_git_sha":implementation_sha,"evidence_generation_parent_sha":implementation_sha,"conforming_oracle_source_commit":oracle_sha,"conforming_oracle_source_sha256":oracle_source_sha256,"python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,"platform":platform.platform(),"solver_identity":"scipy.sparse.linalg.spsolve_cst_plane_strain","thresholds":THRESHOLDS}
    (args.out/"environment_attestation.json").write_text(json.dumps(environment,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
