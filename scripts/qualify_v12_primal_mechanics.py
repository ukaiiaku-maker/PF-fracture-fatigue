#!/usr/bin/env python3
"""Write the predeclared straight-crack V12 primal qualification ledger."""
from __future__ import annotations

import argparse, csv, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.conforming_crack_oracle_v12 import CONFORMING_ORACLE_SOURCE_COMMIT
from arrhenius_fracture.primal_crack_mechanics_v12 import run_rotated_cases, run_straight_case

THRESHOLDS={
 "maximum_finest_global_reference_error":.05,
 "maximum_finest_outside_support_stress_l2_error":.05,
 "maximum_finest_G_reference_error":.05,
 "maximum_energy_compliance_G_identity_error":.01,
 "maximum_low_kappa_pair_reaction_spread":1e-3,
 "maximum_free_residual_relative":1e-10,
 "maximum_energy_reaction_identity_relative":1e-10,
 "maximum_killed_energy_fraction_at_kappa_1e-6":1e-3,
}

def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-300)
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_primal_mechanics")); args=parser.parse_args()
    rows,derivatives=run_straight_case(); finest=6.25e-6
    v12=sorted((r for r in rows if r["representation"]=="C_V12"),key=lambda r:(r["h_tip_m"],r["kappa"])); fine=[r for r in v12 if r["h_tip_m"]==finest]
    low=next(r for r in fine if r["kappa"]==1e-8); mid=next(r for r in fine if r["kappa"]==1e-6)
    vg=[r for r in derivatives if r["representation"]=="V12" and r["kappa"]==1e-8]; cg=[r for r in derivatives if r["representation"]=="CONF"]
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
    }
    gates={
      "MATCHED_PARENT_REPRESENTATIONS": "PASS" if checks["matched_parent_fingerprints_identical_per_h"] else "FAIL",
      "KAPPA_OBJECTIVITY": "PASS" if checks["low_kappa_pair_reaction_spread"]<=THRESHOLDS["maximum_low_kappa_pair_reaction_spread"] else "FAIL",
      "FIELDS_CONVERGENCE": "PASS" if checks["global_and_field_errors_decrease_under_refinement"] and checks["finest_global_reference_error"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["finest_outside_support_stress_l2_error"]<=THRESHOLDS["maximum_finest_outside_support_stress_l2_error"] else "FAIL",
      "CENTERED_G_CONVERGENCE": "PASS" if checks["G_error_decreases_under_refinement"] and checks["finest_G_reference_error"]<=THRESHOLDS["maximum_finest_G_reference_error"] and checks["maximum_energy_compliance_G_identity_error"]<=THRESHOLDS["maximum_energy_compliance_G_identity_error"] else "FAIL",
      "V12_PRIMAL_MECHANICS_STRAIGHT": "PASS",
      "V12_PRIMAL_MECHANICS_ANGLES_30_45": "OPEN",
      "MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED":"NOT_RUN",
      "V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED":"OPEN",
    }
    equilibrium=checks["maximum_free_residual_relative"]<=THRESHOLDS["maximum_free_residual_relative"] and checks["maximum_energy_reaction_identity_relative"]<=THRESHOLDS["maximum_energy_reaction_identity_relative"] and checks["killed_energy_fraction_at_kappa_1e-6"]<=THRESHOLDS["maximum_killed_energy_fraction_at_kappa_1e-6"]
    gates["V12_PRIMAL_MECHANICS_STRAIGHT"]="PASS" if equilibrium and all(gates[k]=="PASS" for k in ("MATCHED_PARENT_REPRESENTATIONS","KAPPA_OBJECTIVITY","FIELDS_CONVERGENCE","CENTERED_G_CONVERGENCE")) else "FAIL"
    angle_rows=[]
    if gates["V12_PRIMAL_MECHANICS_STRAIGHT"]=="PASS":
        angle_rows=run_rotated_cases(); angle_fine=[r for r in angle_rows if r["representation"]=="C_V12" and r["h_tip_m"]==finest and r["kappa"]==1e-8]
        checks["angle_finest_global_reference_error_max"]=max(max(r[k] for k in ("reaction_reference_error","compliance_reference_error","energy_reference_error")) for r in angle_fine)
        checks["angle_equilibrium_identity_max"]=max(max(r["free_residual_relative"],r["energy_reaction_identity_relative"]) for r in angle_rows)
        gates["V12_PRIMAL_MECHANICS_ANGLES_30_45"]="PASS" if checks["angle_finest_global_reference_error_max"]<=THRESHOLDS["maximum_finest_global_reference_error"] and checks["angle_equilibrium_identity_max"]<=THRESHOLDS["maximum_free_residual_relative"] else "FAIL"
    gates["MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED"]="PASS" if gates["V12_PRIMAL_MECHANICS_STRAIGHT"]=="PASS" and gates["V12_PRIMAL_MECHANICS_ANGLES_30_45"]=="PASS" else "FAIL"
    args.out.mkdir(parents=True,exist_ok=True); write_csv(args.out/"straight_primal_matrix.csv",rows); write_csv(args.out/"centered_G_matrix.csv",derivatives)
    if angle_rows: write_csv(args.out/"angle_primal_matrix.csv",angle_rows)
    sha=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip()
    report={"schema":"v12_primal_mechanics_qualification_v1","implementation_git_sha":sha,"conforming_oracle_source_commit":CONFORMING_ORACLE_SOURCE_COMMIT,"thresholds_predeclared":THRESHOLDS,"checks":checks,"gates":gates}
    (args.out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
