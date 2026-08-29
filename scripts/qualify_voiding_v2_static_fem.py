#!/usr/bin/env python3
"""Computed-only V2 static cavity qualification; unimplemented gates stay OPEN."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from arrhenius_fracture.voiding_v2 import (
    build_explicit_hole_mesh, build_solid_plate_mesh, solve_static_hole,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def status(value: float, tolerance: float, relation: str) -> str:
    passed=(value <= tolerance if relation=="max" else value >= tolerance)
    return "PASS" if passed else "FAIL"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("analysis_outputs/voiding_v2_static_fem"))
    args=parser.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    solver="arrhenius_fracture.fem.production_CST_plane_strain"
    refinements=[("coarse",2e-4,48),("medium",1.3333333333333334e-4,72),("fine",1e-4,96)]
    hole_rows=[]
    for label,h,n in refinements:
        hole=build_explicit_hole_mesh(.008,.008,(.004,0),.0005,h,n)
        result=solve_static_hole(hole,8e-6)
        xy=hole.mesh.nodes; edges=hole.cavity_edges
        perimeter=float(np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]],axis=1).sum())
        ordered=xy[np.arange(n)]; area=.5*abs(float(np.sum(ordered[:,0]*np.roll(ordered[:,1],-1)-ordered[:,1]*np.roll(ordered[:,0],-1))))
        remote=abs(result.reaction_top_N_per_m)/.008
        hole_rows.append({"case":"circular_hole","mesh":label,"target_h_m":h,"boundary_segments":n,
            "element_count":hole.mesh.ne,"area_m2":area,"area_relative_error":abs(area-np.pi*.0005**2)/(np.pi*.0005**2),
            "perimeter_m":perimeter,"perimeter_relative_error":abs(perimeter-2*np.pi*.0005)/(2*np.pi*.0005),
            "reaction_top_N_per_m":result.reaction_top_N_per_m,"reaction_balance_relative":result.symmetry_error,
            "stored_energy_J_per_m":result.stored_energy_J_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,
            "free_residual_relative":result.free_residual_norm_N_per_m/abs(result.reaction_top_N_per_m),
            "traction_l2_normalized":result.traction_l2_normalized,"hoop_stress_concentration":result.hoop_stress_concentration,
            "minimum_angle_deg":hole.validation["minimum_angle_deg"],"minimum_quality":hole.validation["minimum_quality"],
            "maximum_aspect_ratio":hole.validation["maximum_aspect_ratio"],
            "actual_internal_components":hole.validation["actual_internal_components"],
            "triangle_disk_intersections":hole.validation["triangle_disk_intersections"],"orphan_nodes":hole.validation["orphan_nodes"],
            "remote_stress_Pa":remote,"solver_identity":solver,"git_sha":git_sha})
    write_csv(out/"circular_hole_fem_convergence.csv",hole_rows)

    interaction=[]
    for label,h,n in refinements:
        baseline=solve_static_hole(build_solid_plate_mesh(.008,.008,h),8e-6,
                                   crack_tip_m=(.002,0),wake_half_width_m=.75*h)
        interaction.append({"case":"crack_only","mesh":label,"center_x_m":"","offset_y_m":0,
            "ligament_over_R":"INF","tip_sigma_yy_Pa":baseline.crack_tip_sigma_yy_Pa,
            "tip_probe_relative_to_same_mesh_crack_only":1.0,"compliance_m2_per_N":baseline.compliance_m2_per_N,
            "traction_l2_normalized":"","free_residual_relative":baseline.free_residual_norm_N_per_m/abs(baseline.reaction_top_N_per_m),
            "solver_identity":solver,"git_sha":git_sha})
        for case,center in (("centered",(.003,0)),("far",(.006,0)),("offset_positive",(.004,.00025)),("offset_negative",(.004,-.00025))):
            hole=build_explicit_hole_mesh(.008,.008,center,.00025,h,n)
            result=solve_static_hole(hole,8e-6,crack_tip_m=(.002,0),wake_half_width_m=.75*h)
            ligament=(center[0]-.00025-.002)/.00025
            interaction.append({"case":case,"mesh":label,"center_x_m":center[0],"offset_y_m":center[1],
                "ligament_over_R":ligament,"tip_sigma_yy_Pa":result.crack_tip_sigma_yy_Pa,
                "tip_probe_relative_to_same_mesh_crack_only":result.crack_tip_sigma_yy_Pa/baseline.crack_tip_sigma_yy_Pa,
                "compliance_m2_per_N":result.compliance_m2_per_N,"traction_l2_normalized":result.traction_l2_normalized,
                "free_residual_relative":result.free_residual_norm_N_per_m/abs(result.reaction_top_N_per_m),
                "solver_identity":solver,"git_sha":git_sha})
    write_csv(out/"prescribed_crack_void_interaction.csv",interaction)

    # Actual centered finite differences. Crack derivative holds one mesh/void fixed;
    # void derivative remeshes only R while holding crack geometry fixed.
    h=refinements[1][1]; n=refinements[1][2]; da=5e-5; dR=2.5e-5
    fixed=build_explicit_hole_mesh(.008,.008,(.004,0),.00025,h,n)
    um=solve_static_hole(fixed,8e-6,crack_tip_m=(.002-da,0),wake_half_width_m=.75*h).stored_energy_J_per_m
    up=solve_static_hole(fixed,8e-6,crack_tip_m=(.002+da,0),wake_half_width_m=.75*h).stored_energy_J_per_m
    rm=solve_static_hole(build_explicit_hole_mesh(.008,.008,(.004,0),.00025-dR,h,n),8e-6,
                         crack_tip_m=(.002,0),wake_half_width_m=.75*h).stored_energy_J_per_m
    rp=solve_static_hole(build_explicit_hole_mesh(.008,.008,(.004,0),.00025+dR,h,n),8e-6,
                         crack_tip_m=(.002,0),wake_half_width_m=.75*h).stored_energy_J_per_m
    energy=[{"derivative":"G_crack","held_fixed":"R_void","perturbation_m":da,"minus_energy_J_per_m":um,
             "plus_energy_J_per_m":up,"value_J_per_m2":(up-um)/(2*da),"solver_identity":solver,"git_sha":git_sha},
            {"derivative":"G_void","held_fixed":"crack_geometry","perturbation_m":dR,"minus_energy_J_per_m":rm,
             "plus_energy_J_per_m":rp,"value_J_per_m2":(rp-rm)/(2*dR),"solver_identity":solver,"git_sha":git_sha}]
    write_csv(out/"virtual_energy_derivatives.csv",energy)

    fine=hole_rows[-1]
    offset=[r for r in interaction if r["mesh"]=="fine" and r["case"].startswith("offset_")]
    far=[r for r in interaction if r["mesh"]=="fine" and r["case"]=="far"][0]
    checks=[
      {"metric":"actual_cavity_components","value":fine["actual_internal_components"],"tolerance":1,"relation":"equal","status":"PASS" if fine["actual_internal_components"]==1 else "FAIL"},
      {"metric":"triangle_disk_intersections","value":fine["triangle_disk_intersections"],"tolerance":0,"relation":"equal","status":"PASS" if fine["triangle_disk_intersections"]==0 else "FAIL"},
      {"metric":"free_residual_relative","value":fine["free_residual_relative"],"tolerance":1e-10,"relation":"max","status":status(fine["free_residual_relative"],1e-10,"max")},
      {"metric":"traction_l2_normalized","value":fine["traction_l2_normalized"],"tolerance":.1,"relation":"max","status":status(fine["traction_l2_normalized"],.1,"max")},
      {"metric":"Kirsch_SC_error","value":abs(fine["hoop_stress_concentration"]-3)/3,"tolerance":.1,"relation":"max","status":status(abs(fine["hoop_stress_concentration"]-3)/3,.1,"max")},
      {"metric":"minimum_angle_deg","value":fine["minimum_angle_deg"],"tolerance":10,"relation":"min","status":status(fine["minimum_angle_deg"],10,"min")},
      {"metric":"offset_mirror_tip_probe_relative","value":abs(offset[0]["tip_sigma_yy_Pa"]-offset[1]["tip_sigma_yy_Pa"])/max(abs(offset[0]["tip_sigma_yy_Pa"]),1e-300),"tolerance":.02,"relation":"max","status":"PASS" if abs(offset[0]["tip_sigma_yy_Pa"]-offset[1]["tip_sigma_yy_Pa"])/abs(offset[0]["tip_sigma_yy_Pa"])<=.02 else "FAIL"},
      {"metric":"far_void_tip_probe_relative_error","value":abs(far["tip_probe_relative_to_same_mesh_crack_only"]-1),"tolerance":.1,"relation":"max","status":status(abs(far["tip_probe_relative_to_same_mesh_crack_only"]-1),.1,"max")},
      {"metric":"virtual_derivative_mesh_and_step_convergence","value":"NOT_RUN","tolerance":"required","relation":"","status":"OPEN"},
    ]
    for row in checks:
        row.update({"calculation_source":"executed V2 static FEM runner","solver_identity":solver,"git_sha":git_sha})
    write_csv(out/"acceptance_checks.csv",checks)
    gate="PASS" if all(r["status"]=="PASS" for r in checks) else "OPEN"
    decision={"schema":"voiding-v2-static-fem-decision/1","EXPLICIT_VOID_MECHANICS_QUALIFIED":gate,
              "reason":"all computed acceptance checks must pass" if gate=="PASS" else "one or more computed checks failed or remain OPEN",
              "checks":checks,"git_sha":git_sha,"solver_identity":solver,
              "not_run":{"production_crack_void_topology":"OUT_OF_SCOPE","stochastic_lifecycle_integration":"OUT_OF_SCOPE","promotion_and_resolved_growth":"OUT_OF_SCOPE","fatigue":"OUT_OF_SCOPE"}}
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    files=[]
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name!="manifest.json": files.append({"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size})
    (out/"manifest.json").write_text(json.dumps({"schema":"voiding-v2-artifacts/1","files":files},indent=2)+"\n")


if __name__=="__main__": main()
