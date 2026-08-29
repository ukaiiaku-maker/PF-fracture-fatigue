#!/usr/bin/env python3
"""Authoritative split-gate V2 cavity and V11 causal-crack static qualification."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy

from arrhenius_fracture.causal_sharp_wake_v11 import causal_segment_support
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import plane_strain_D
from arrhenius_fracture.interaction_integral_v10214 import compute_signed_interaction_integral
from arrhenius_fracture.interaction_integral_v1029 import _hermite_plateau_q
from arrhenius_fracture.voiding_v2 import build_explicit_hole_mesh, fill_explicit_hole_mesh, solve_static_hole

OPENING=8e-6; MAT=ElasticProperties(E=210e9,nu=.3); D=plane_strain_D(MAT)
REFS=(("coarse",2e-4,48),("medium",1.3333333333333334e-4,72),("fine",1e-4,96),("finer",7.5e-5,128))
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

def contour(mesh,result,mask,p1,r_outer):
    radii=np.linalg.norm(mesh.nodes-p1,axis=1); q=_hermite_plateau_q(radii,.3*r_outer,r_outer)
    grad_support=np.flatnonzero(np.ptp(q[mesh.elems],axis=1)>1e-14)
    raw_overlap=np.intersect1d(grad_support,np.flatnonzero(mask)); active=np.setdiff1d(grad_support,np.flatnonzero(mask))
    cent=mesh.nodes[mesh.elems[active]].mean(axis=1); theta=np.arctan2(cent[:,1]-p1[1],cent[:,0]-p1[0])
    angular_bins=len(np.unique(np.floor((theta+np.pi)/(2*np.pi)*24).astype(int)%24))
    boundary_clearance=min(p1[0]-mesh.nodes[:,0].min(),mesh.nodes[:,0].max()-p1[0],
                           p1[1]-mesh.nodes[:,1].min(),mesh.nodes[:,1].max()-p1[1])
    admissible=bool(r_outer<boundary_clearance and len(active)>=50 and angular_bins>=16)
    if not admissible:
        return {"outer_radius_m":r_outer,"inner_radius_m":.3*r_outer,"q_gradient_element_count":len(grad_support),
          "raw_q_gradient_crack_overlap_count":len(raw_overlap),"active_element_count":len(active),"angular_bins_of_24":angular_bins,
          "external_boundary_clearance_m":boundary_clearance,"status":"NOT_ADMISSIBLE_AT_CURRENT_RESOLUTION",
          "K_I_Pa_sqrt_m":math.nan,"K_II_Pa_sqrt_m":math.nan,"skipped_exact_elements":len(mask.nonzero()[0])}
    value=compute_signed_interaction_integral(mesh,result.displacement,result.sigma_gp,np.zeros(mesh.nn),p1,np.asarray((1.,0.)),
      MAT,r_outer,cfg=SimpleNamespace(r_inner_factor=.3,r_outer_factor=1.),crack_segments=[(P0,p1)],D=D,
      exclude_element_mask=mask)
    return {"outer_radius_m":r_outer,"inner_radius_m":.3*r_outer,"q_gradient_element_count":len(grad_support),
      "raw_q_gradient_crack_overlap_count":len(raw_overlap),"active_element_count":len(active),"angular_bins_of_24":angular_bins,
      "external_boundary_clearance_m":boundary_clearance,"status":"ADMISSIBLE_EXACT_CRACK_ELEMENTS_EXCLUDED",
      "K_I_Pa_sqrt_m":value.K_I_Pa_sqrt_m,"K_II_Pa_sqrt_m":value.K_II_Pa_sqrt_m,
      "skipped_exact_elements":value.diagnostics["mode_I"]["skipped_exact_elements"]}

def spread(values): return float(np.ptp(values)/max(np.max(np.abs(values)),1e-300))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("analysis_outputs/voiding_v2_causal_static")); a=p.parse_args()
    out=a.out; out.mkdir(parents=True,exist_ok=True); source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    cavity=[]; support=[]; contours=[]; derivatives=[]
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
        valid=[r for r in local if r["status"].startswith("ADMISSIBLE")]
        ki=float(np.median([r["K_I_Pa_sqrt_m"] for r in valid])) if valid else math.nan
        kii=float(np.median([r["K_II_Pa_sqrt_m"] for r in valid])) if valid else math.nan
        gk=(ki*ki+kii*kii)/MAT.Eprime if valid else math.nan
        for fraction in (.025,.05,.10):
            da=fraction*BASE_TIP[0]; values=[]
            for sign in (-1,1):
                endpoint=BASE_TIP.copy(); endpoint[0]+=sign*da; mask,_=causal_mask(control.mesh,endpoint)
                values.append(solve_static_hole(control,OPENING,crack_tip_m=tuple(endpoint),element_kill_mask=mask))
            minus,plus=values; dU=(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*da)
            dC=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*da); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N)
            ge=-dU; gc=OPENING**2*dC/(2*cm**2)
            derivatives.append({"mesh":label,"h_m":h,"delta_a_over_a":fraction,"delta_a_m":da,"G_energy_J_per_m2":ge,
              "G_compliance_J_per_m2":gc,"energy_compliance_error":abs(ge-gc)/max(abs(ge),abs(gc),1e-300),
              "G_interaction_J_per_m2":gk,"energy_interaction_error":abs(ge-gk)/max(abs(ge),abs(gk),1e-300),
              "physical_length_normalization":True,"git_sha":source})
    write_csv(out/"cavity_only.csv",cavity); write_csv(out/"causal_support.csv",support)
    write_csv(out/"causal_crack_contours.csv",contours); write_csv(out/"causal_crack_derivatives.csv",derivatives)
    cavity_pass=(max(r["recovered_traction"] for r in cavity[-2:])<=.1 and max(r["Kirsch_error"] for r in cavity[-2:])<=.1 and
      max(r["weak_residual"] for r in cavity)<=1e-10 and all(r["topology_components"]==1 and r["disk_intersections"]==0 and r["exact_boundary_nodes"] for r in cavity))
    all_admissible=all(r["status"].startswith("ADMISSIBLE") for r in contours)
    plateau=max(spread([r["K_I_Pa_sqrt_m"] for r in contours if r["mesh"]==label and np.isfinite(r["K_I_Pa_sqrt_m"])]) for label,_,_ in REFS)
    symmetry=max(abs(r["K_II_Pa_sqrt_m"])/max(abs(r["K_I_Pa_sqrt_m"]),1e-300) for r in contours if np.isfinite(r["K_I_Pa_sqrt_m"]))
    fine_der=[r for r in derivatives if r["mesh"]=="finer"]; perturb=spread([r["G_energy_J_per_m2"] for r in fine_der])
    mesh_values=[r["G_energy_J_per_m2"] for r in derivatives if r["delta_a_over_a"]==.05 and r["mesh"] in ("fine","finer")]
    mesh_spread=spread(mesh_values); ec=max(r["energy_compliance_error"] for r in derivatives)
    eg=max(r["energy_interaction_error"] for r in derivatives if r["mesh"] in ("fine","finer") and r["delta_a_over_a"]==.05)
    support_converges=support[-1]["maximum_normal_support_distance_m"]<support[-2]["maximum_normal_support_distance_m"]
    checks={"contours_all_admissible":all_admissible,"contour_KI_plateau_relative":plateau,"centered_KII_over_KI":symmetry,
      "energy_compliance_error":ec,"energy_interaction_error":eg,"perturbation_spread":perturb,"mesh_spread":mesh_spread,
      "causal_support_distance_decreases":support_converges}
    crack_pass=bool(all_admissible and plateau<=.1 and symmetry<=.05 and ec<=.1 and eg<=.1 and perturb<=.1 and mesh_spread<=.1 and support_converges)
    combined="NOT_RUN_REQUIRES_CAUSAL_CRACK_ONLY_PASS" if not crack_pass else "OPEN_NOT_RUN"
    decision={"schema":"voiding-v2-causal-static-decision/1","EXPLICIT_CAVITY_ONLY_STATIC_FEM_QUALIFIED":"PASS" if cavity_pass else "OPEN",
      "V11_CAUSAL_CRACK_ONLY_STATIC_FEM_QUALIFIED":"PASS" if crack_pass else "OPEN",
      "PRESCRIBED_CAUSAL_CRACK_VOID_INTERACTION_QUALIFIED":combined,"EXPLICIT_VOID_MECHANICS_QUALIFIED":"OPEN",
      "causal_crack_checks":checks,"superseded_diagnostic":{"type":"CENTROID_BAND_EXTENSION_DERIVATIVE_NUMERICALLY_STABLE",
       "status":"DIAGNOSTIC","evidence_commit":"f0e2959ecf1187685e9d7723b2b4314791f9c353"},"git_sha":source,"solver_identity":SOLVER}
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    env={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"required_python":"3.12"}
    (out/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    files=[]
    for q in sorted(out.glob("*")):
        if q.name!="manifest.json": files.append({"path":q.name,"sha256":hashlib.sha256(q.read_bytes()).hexdigest(),"bytes":q.stat().st_size})
    (out/"manifest.json").write_text(json.dumps({"schema":"voiding-v2-causal-static-artifacts/1","files":files},indent=2)+"\n")

if __name__=="__main__": main()
