#!/usr/bin/env python3
"""Authoritative split-gate V2 cavity and V11 causal-crack static qualification."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy

from arrhenius_fracture.causal_sharp_wake_v11 import causal_segment_support, mechanical_fingerprint
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.conforming_crack_oracle_v11 import (
    build_conforming_slit_mesh, recovered_face_traction_relative, solve_conforming_slit,
)
from arrhenius_fracture.fem import plane_strain_D
from arrhenius_fracture.interaction_integral_v10214 import compute_signed_interaction_integral
from arrhenius_fracture.interaction_integral_v1029 import _hermite_plateau_q
from arrhenius_fracture.matched_crack_qualification_v11 import run_matched_qualification
from arrhenius_fracture.voiding_v2 import build_explicit_hole_mesh, fill_explicit_hole_mesh, solve_static_hole

OPENING=8e-6; MAT=ElasticProperties(E=210e9,nu=.3); D=plane_strain_D(MAT)
REFS=(("coarse",2e-4,48),("medium",1.3333333333333334e-4,72),("fine",1e-4,96),("finer",7.5e-5,128))
ORACLE_REFS=(("coarse",2.5e-4),("medium",1.25e-4),("fine",6.25e-5))
P0=np.asarray((.0005,0.)); BASE_TIP=np.asarray((.002,0.)); CONTOURS=(.0006,.0008,.0010)
SOLVER="production CST assembly/constitutive law + benchmark symmetric rigid constraint + scipy.spsolve(SuperLU)"

def write_csv(path,rows):
    fields=list(rows[0]); fields.extend(sorted({k for r in rows for k in r}-set(fields)))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)

def fingerprint(ids): return hashlib.sha256(np.ascontiguousarray(ids,dtype=np.int64).tobytes()).hexdigest()

def causal_mask(mesh,p1):
    selected,represented=causal_segment_support(mesh,P0,np.asarray(p1,float)); mask=np.zeros(mesh.ne,bool); mask[selected]=True
    tri=mesh.nodes[mesh.elems[selected]]; length=float(np.linalg.norm(np.asarray(p1)-P0))
    return mask,{"physical_p0_x_m":P0[0],"physical_p0_y_m":P0[1],"physical_p1_x_m":p1[0],"physical_p1_y_m":p1[1],
      "physical_crack_length_m":length,"selected_element_count":len(selected),"selected_element_ids":" ".join(map(str,selected)),
      "selected_element_area_m2":float(mesh.area_e[selected].sum()),"represented_intersection_length_sum_m":float(represented.sum()),
      "maximum_normal_support_distance_m":float(np.max(np.abs(tri[:,:,1]))),
      "endpoint_support_overshoot_m":float(np.max(tri[:,:,0])-p1[0]),"support_fingerprint":fingerprint(selected)}

def contour(mesh,result,mask,p1,r_outer,*,status="EXECUTED_WITH_EXACT_P0_EXCLUSION"):
    radii=np.linalg.norm(mesh.nodes-p1,axis=1); q=_hermite_plateau_q(radii,.3*r_outer,r_outer)
    grad_support=np.flatnonzero(np.ptp(q[mesh.elems],axis=1)>1e-14)
    raw_overlap=np.intersect1d(grad_support,np.flatnonzero(mask)); prefiltered=np.setdiff1d(grad_support,np.flatnonzero(mask))
    cent=mesh.nodes[mesh.elems[prefiltered]].mean(axis=1); theta=np.arctan2(cent[:,1]-p1[1],cent[:,0]-p1[0])
    angular_bins=len(np.unique(np.floor((theta+np.pi)/(2*np.pi)*24).astype(int)%24))
    boundary_clearance=min(p1[0]-mesh.nodes[:,0].min(),mesh.nodes[:,0].max()-p1[0],
                           p1[1]-mesh.nodes[:,1].min(),mesh.nodes[:,1].max()-p1[1])
    admissible=bool(r_outer<boundary_clearance and len(prefiltered)>=50 and angular_bins>=16)
    if not admissible:
        return {"outer_radius_m":r_outer,"inner_radius_m":.3*r_outer,"q_gradient_element_count":len(grad_support),
          "raw_q_gradient_crack_overlap_count":len(raw_overlap),"active_element_count":0,"angular_bins_of_24":angular_bins,
          "external_boundary_clearance_m":boundary_clearance,"status":"NOT_ADMISSIBLE_AT_CURRENT_RESOLUTION",
          "K_I_Pa_sqrt_m":math.nan,"K_II_Pa_sqrt_m":math.nan,"skipped_exact_elements":len(mask.nonzero()[0])}
    value=compute_signed_interaction_integral(mesh,result.displacement,result.sigma_gp,np.zeros(mesh.nn),p1,np.asarray((1.,0.)),
      MAT,r_outer,cfg=SimpleNamespace(r_inner_factor=.3,r_outer_factor=1.),crack_segments=[(P0,p1)],D=D,
      exclude_element_mask=mask)
    return {"outer_radius_m":r_outer,"inner_radius_m":.3*r_outer,"q_gradient_element_count":len(grad_support),
      "raw_q_gradient_crack_overlap_count":len(raw_overlap),"active_element_count":value.diagnostics["mode_I"]["active_elements"],"angular_bins_of_24":angular_bins,
      "external_boundary_clearance_m":boundary_clearance,"status":status,
      "K_I_Pa_sqrt_m":value.K_I_Pa_sqrt_m,"K_II_Pa_sqrt_m":value.K_II_Pa_sqrt_m,
      "exact_mask_total_elements":value.diagnostics["mode_I"]["exact_mask_total_elements"],
      "skipped_exact_q_gradient_overlap":value.diagnostics["mode_I"]["skipped_exact_elements"],
      "skipped_damage_elements":value.diagnostics["mode_I"]["skipped_damage_elements"],
      "skipped_branch_cut_elements":value.diagnostics["mode_I"]["skipped_branch_cut_elements"],
      "skipped_line_of_sight_elements":value.diagnostics["mode_I"]["skipped_line_of_sight_elements"]}

def spread(values): return float(np.ptp(values)/max(np.max(np.abs(values)),1e-300))

def safe_spread(values):
    values=np.asarray(values,float)
    return spread(values) if values.size>=2 and np.all(np.isfinite(values)) else math.inf

def qualify_conforming_oracle(source):
    rows=[]; contour_rows=[]
    for label,h in ORACLE_REFS:
        slit=build_conforming_slit_mesh(.008,.008,tuple(P0),tuple(BASE_TIP),h)
        pin=int(slit.hole.boundary.left_bot); base=solve_conforming_slit(slit,OPENING,pin_node=pin)
        minus_slit=build_conforming_slit_mesh(.008,.008,tuple(P0),(BASE_TIP[0]-h,0.),h)
        plus_slit=build_conforming_slit_mesh(.008,.008,tuple(P0),(BASE_TIP[0]+h,0.),h)
        minus=solve_conforming_slit(minus_slit,OPENING,pin_node=int(minus_slit.hole.boundary.left_bot))
        plus=solve_conforming_slit(plus_slit,OPENING,pin_node=int(plus_slit.hole.boundary.left_bot))
        ge=-(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*h)
        for radius in CONTOURS:
            cr=contour(slit.hole.mesh,base,np.zeros(slit.hole.mesh.ne,bool),BASE_TIP,radius,
                       status="EXECUTED_CONFORMING_TRACTION_FREE_SLIT")
            contour_rows.append({"mesh":label,"h_m":h,**cr,"git_sha":source})
        rows.append({"mesh":label,"h_m":h,"nodes":slit.hole.mesh.nn,"elements":slit.hole.mesh.ne,
          "shared_crack_endpoints":True,"duplicated_interior_face_nodes":slit.hole.validation["duplicated_interior_face_nodes"],
          "weak_crack_face_residual_relative":base.weak_cavity_residual_relative,
          "recovered_crack_face_traction_relative":recovered_face_traction_relative(slit,base),
          "energy_J_per_m":base.stored_energy_J_per_m,"G_energy_J_per_m2":ge,"delta_a_m":h,
          "reaction_N_per_m":base.reaction_top_N_per_m,"compliance_m2_per_N":base.compliance_m2_per_N,"git_sha":source})
    per_mesh_plateau={label:safe_spread([r["K_I_Pa_sqrt_m"] for r in contour_rows if r["mesh"]==label and np.isfinite(r["K_I_Pa_sqrt_m"])]) for label,_ in ORACLE_REFS}
    pin_slit=build_conforming_slit_mesh(.008,.008,tuple(P0),tuple(BASE_TIP),ORACLE_REFS[-1][1])
    left=solve_conforming_slit(pin_slit,OPENING,pin_node=int(pin_slit.hole.boundary.left_bot))
    right=solve_conforming_slit(pin_slit,OPENING,pin_node=int(pin_slit.hole.boundary.right_bot))
    pin_error=abs(left.stored_energy_J_per_m-right.stored_energy_J_per_m)/max(abs(left.stored_energy_J_per_m),1e-300)
    checks={"all_contours_admissible":all(r["status"].startswith("EXECUTED") for r in contour_rows),
      "maximum_contour_KI_plateau_relative":max(per_mesh_plateau.values()),
      "energy_G_mesh_spread_relative":safe_spread([r["G_energy_J_per_m2"] for r in rows[-2:]]),
      "weak_face_residual_max_relative":max(r["weak_crack_face_residual_relative"] for r in rows),
      "recovered_face_traction_decreases":rows[-1]["recovered_crack_face_traction_relative"]<rows[-2]["recovered_crack_face_traction_relative"],
      "rigid_pin_energy_relative_error":pin_error}
    qualified=bool(checks["all_contours_admissible"] and checks["maximum_contour_KI_plateau_relative"]<=.1 and
      checks["energy_G_mesh_spread_relative"]<=.1 and checks["weak_face_residual_max_relative"]<=1e-10 and
      checks["recovered_face_traction_decreases"] and pin_error<=1e-10)
    return rows,contour_rows,checks,qualified

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("analysis_outputs/voiding_v2_causal_static")); a=p.parse_args()
    out=a.out; out.mkdir(parents=True,exist_ok=True); source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    cavity=[]; support=[]; contours=[]; derivatives=[]; derivative_states=[]
    for label,h,n in REFS:
        hole=build_explicit_hole_mesh(.008,.008,(.004,0),.0005,h,n); r=solve_static_hole(hole,OPENING)
        cavity.append({"mesh":label,"h_m":h,"segments":n,"weak_residual":r.weak_cavity_residual_relative,
          "recovered_traction":r.traction_l2_normalized,"Kirsch_error":abs(r.hoop_stress_concentration-3)/3,
          "free_residual":r.free_residual_norm_N_per_m/abs(r.reaction_top_N_per_m),"minimum_angle_deg":hole.validation["minimum_angle_deg"],
          "mirror_xx":r.mirror_sigma_xx_relative,"mirror_yy":r.mirror_sigma_yy_relative,"mirror_xy":r.mirror_sigma_xy_antisym_relative,
          "topology_components":hole.validation["actual_internal_components"],"disk_intersections":hole.validation["triangle_disk_intersections"],
          "exact_boundary_nodes":hole.validation["polygon_exact_node_set_match"],"git_sha":source})
        control=fill_explicit_hole_mesh(build_explicit_hole_mesh(.008,.008,(.004,0),.00025,h,n))
        base_mask,audit=causal_mask(control.mesh,BASE_TIP); base=solve_static_hole(control,OPENING,crack_tip_m=tuple(BASE_TIP),element_kill_mask=base_mask)
        support.append({"mesh":label,"h_m":h,**audit,"support_distance_over_h":audit["maximum_normal_support_distance_m"]/h,"git_sha":source})
        local=[]
        for radius in CONTOURS:
            row=contour(control.mesh,base,base_mask,BASE_TIP,radius); contours.append({"mesh":label,"h_m":h,**row,"git_sha":source}); local.append(row)
        valid=[r for r in local if r["status"].startswith("EXECUTED")]
        ki=float(np.median([r["K_I_Pa_sqrt_m"] for r in valid])) if valid else math.nan
        kii=float(np.median([r["K_II_Pa_sqrt_m"] for r in valid])) if valid else math.nan
        local_plateau=spread([r["K_I_Pa_sqrt_m"] for r in valid]) if valid else math.inf
        gk=(ki*ki+kii*kii)/MAT.Eprime if valid and local_plateau<=.1 else math.nan
        crack_length=float(np.linalg.norm(BASE_TIP-P0))
        base_fingerprint=audit["support_fingerprint"]
        base_mechanical_fingerprint=mechanical_fingerprint(control.mesh,base_mask.astype(float))
        for fraction in (.025,.05,.10):
            da=fraction*crack_length; values=[]; state_audits=[]
            for state_name,sign in (("minus",-1),("plus",1)):
                endpoint=BASE_TIP.copy(); endpoint[0]+=sign*da
                mask,state_audit=causal_mask(control.mesh,endpoint)
                solved=solve_static_hole(control,OPENING,crack_tip_m=tuple(endpoint),element_kill_mask=mask)
                values.append(solved); state_audits.append((state_name,mask,state_audit,solved))
            minus,plus=values; dU=(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*da)
            dC=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*da); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N)
            resolved=(state_audits[0][2]["support_fingerprint"]!=base_fingerprint and state_audits[1][2]["support_fingerprint"]!=base_fingerprint and
                      state_audits[0][2]["support_fingerprint"]!=state_audits[1][2]["support_fingerprint"])
            ge=-dU if resolved else math.nan; gc=OPENING**2*dC/(2*cm**2) if resolved else math.nan
            derivative_states.append({"mesh":label,"h_m":h,"delta_a_over_physical_crack_length":fraction,"state":"base",
              **audit,"new_support_vs_base":"","removed_support_vs_base":"","reaction_N_per_m":base.reaction_top_N_per_m,
              "compliance_m2_per_N":base.compliance_m2_per_N,"energy_J_per_m":base.stored_energy_J_per_m,
              "mechanical_fingerprint":base_mechanical_fingerprint,"git_sha":source})
            for state_name,mask,state_audit,solved in state_audits:
                derivative_states.append({"mesh":label,"h_m":h,"delta_a_over_physical_crack_length":fraction,"state":state_name,
                  **state_audit,"new_support_vs_base":" ".join(map(str,np.setdiff1d(np.flatnonzero(mask),np.flatnonzero(base_mask)))),
                  "removed_support_vs_base":" ".join(map(str,np.setdiff1d(np.flatnonzero(base_mask),np.flatnonzero(mask)))),
                  "reaction_N_per_m":solved.reaction_top_N_per_m,"compliance_m2_per_N":solved.compliance_m2_per_N,
                  "energy_J_per_m":solved.stored_energy_J_per_m,
                  "mechanical_fingerprint":mechanical_fingerprint(control.mesh,mask.astype(float)),"git_sha":source})
            derivatives.append({"mesh":label,"h_m":h,"delta_a_over_physical_crack_length":fraction,"delta_a_m":da,
              "support_resolution_status":"RESOLVED_DISTINCT_P0_SUPPORT" if resolved else "NOT_RESOLVED_NO_P0_SUPPORT_CHANGE","G_energy_J_per_m2":ge,
              "G_compliance_J_per_m2":gc,"energy_compliance_error":abs(ge-gc)/max(abs(ge),abs(gc),1e-300),
              "INTERACTION_INTEGRAL_G":"NOT_CONVERGED_NO_CONTOUR_PLATEAU" if not np.isfinite(gk) else gk,
              "ENERGY_VS_INTERACTION_G":"NOT_EVALUATED_REQUIRES_CONVERGED_INTERACTION_G" if not np.isfinite(gk) else abs(ge-gk)/max(abs(ge),abs(gk),1e-300),
              "physical_length_normalization":True,"git_sha":source})
    write_csv(out/"cavity_only.csv",cavity); write_csv(out/"causal_support.csv",support)
    write_csv(out/"causal_crack_contours.csv",contours); write_csv(out/"causal_crack_derivatives.csv",derivatives)
    write_csv(out/"causal_crack_derivative_states.csv",derivative_states)
    oracle,oracle_contours,oracle_checks,oracle_pass=qualify_conforming_oracle(source)
    write_csv(out/"conforming_crack_oracle.csv",oracle)
    write_csv(out/"conforming_crack_oracle_contours.csv",oracle_contours)
    matched,matched_oracle_checks,matched_oracle_pass,matched_p0_checks,matched_p0_pass=run_matched_qualification(source)
    for name,rows in matched.items(): write_csv(out/(name+".csv"),rows)
    cavity_pass=(max(r["recovered_traction"] for r in cavity[-2:])<=.1 and max(r["Kirsch_error"] for r in cavity[-2:])<=.1 and
      max(r["weak_residual"] for r in cavity)<=1e-10 and all(r["topology_components"]==1 and r["disk_intersections"]==0 and r["exact_boundary_nodes"] for r in cavity))
    all_admissible=all(r["status"].startswith("EXECUTED") for r in contours)
    plateau=max(safe_spread([r["K_I_Pa_sqrt_m"] for r in contours if r["mesh"]==label and np.isfinite(r["K_I_Pa_sqrt_m"])]) for label,_,_ in REFS)
    symmetry_values=[abs(r["K_II_Pa_sqrt_m"])/max(abs(r["K_I_Pa_sqrt_m"]),1e-300) for r in contours if np.isfinite(r["K_I_Pa_sqrt_m"])]
    symmetry=max(symmetry_values) if symmetry_values else math.inf
    resolved_der=[r for r in derivatives if r["support_resolution_status"].startswith("RESOLVED")]
    fine_der=[r for r in resolved_der if r["mesh"]=="finer"]; perturb=safe_spread([r["G_energy_J_per_m2"] for r in fine_der])
    mesh_values=[r["G_energy_J_per_m2"] for r in resolved_der if r["delta_a_over_physical_crack_length"]==.05 and r["mesh"] in ("fine","finer")]
    mesh_spread=safe_spread(mesh_values); ec=max((r["energy_compliance_error"] for r in resolved_der),default=math.inf)
    eg="NOT_EVALUATED_REQUIRES_CONVERGED_INTERACTION_G"
    support_converges=support[-1]["maximum_normal_support_distance_m"]<support[-2]["maximum_normal_support_distance_m"]
    checks={"contours_all_admissible":all_admissible,"contour_KI_plateau_relative":plateau,"centered_KII_over_KI":symmetry,
      "energy_compliance_error":ec,"energy_interaction_error":eg,"perturbation_spread":perturb,"mesh_spread":mesh_spread,
      "causal_support_distance_decreases":support_converges}
    crack_pass=bool(all_admissible and plateau<=.1 and symmetry<=.05 and ec<=.1 and perturb<=.1 and mesh_spread<=.1 and support_converges)
    combined="NOT_RUN_REQUIRES_CAUSAL_CRACK_ONLY_PASS" if not crack_pass else "OPEN_NOT_RUN"
    decision={"schema":"voiding-v2-causal-static-decision/1","EXPLICIT_CAVITY_ONLY_STATIC_FEM_QUALIFIED":"PASS" if cavity_pass else "OPEN",
      "V11_CONFORMING_CRACK_REFERENCE_QUALIFIED":"PASS" if matched_oracle_pass else "OPEN",
      "V11_CAUSAL_P0_WAKE_MESH_OBJECTIVE":"PASS" if matched_p0_pass else "OPEN",
      "V11_CAUSAL_CRACK_ONLY_STATIC_FEM_QUALIFIED":"PASS" if matched_p0_pass and crack_pass else "OPEN",
      "P0_ABSOLUTE_INTERACTION_INTEGRAL_QUALIFIED":"PASS" if matched_p0_pass else "NOT_QUALIFIED_FINITE_WIDTH_STRIP_REQUIRES_INTERFACE_CONFIG_FORCE_CORRECTION",
      "P0_INTERACTION_INTEGRAL_SIGNED_PERTURBATION_ROLE":"RETAIN_PREVIOUSLY_QUALIFIED_ROLE_ONLY",
      "PRESCRIBED_CAUSAL_CRACK_VOID_INTERACTION_QUALIFIED":combined,"EXPLICIT_VOID_MECHANICS_QUALIFIED":"OPEN",
      "conforming_crack_reference_checks":matched_oracle_checks,"causal_p0_matched_checks":matched_p0_checks,
      "legacy_unmatched_conforming_diagnostic_checks":oracle_checks,"causal_crack_checks":checks,"superseded_diagnostic":{"type":"CENTROID_BAND_EXTENSION_DERIVATIVE_NUMERICALLY_STABLE",
       "status":"DIAGNOSTIC","evidence_commit":"f0e2959ecf1187685e9d7723b2b4314791f9c353"},"git_sha":source,"solver_identity":SOLVER}
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    env={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"required_python":"3.12"}
    (out/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    files=[]
    for q in sorted(out.glob("*")):
        if q.name!="manifest.json": files.append({"path":q.name,"sha256":hashlib.sha256(q.read_bytes()).hexdigest(),"bytes":q.stat().st_size})
    (out/"manifest.json").write_text(json.dumps({"schema":"voiding-v2-causal-static-artifacts/1","files":files},indent=2)+"\n")

if __name__=="__main__": main()
