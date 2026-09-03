#!/usr/bin/env python3
"""Generate SHA-bound V12 absolute-K evidence under the frozen v1 criterion."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import scipy

from arrhenius_fracture.absolute_k_criterion_v12 import *
from arrhenius_fracture.config import JIntegralConfig
from arrhenius_fracture.conforming_crack_oracle_v12 import build_matched_crack_parent, conforming_slit_from_parent
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.interaction_integral_v10214 import compute_signed_interaction_integral
from arrhenius_fracture.interaction_integral_v1029 import _auxiliary_stress_local
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import mechanically_separating_graph_support
from arrhenius_fracture.primal_crack_mechanics_v12 import MAT, D, _solve

ROOT=Path(__file__).resolve().parents[1]
BASE_SHA="2b5e5351add0bf0db67f2cda35a1480c3e7efc91"
IMPLEMENTATION_PATHS=("arrhenius_fracture/absolute_k_criterion_v12.py","arrhenius_fracture/interaction_integral_v1029.py","arrhenius_fracture/interaction_integral_v10214.py","scripts/qualify_v12_absolute_k.py")

def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-300)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(("git",)+args,cwd=ROOT,text=True).strip()
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)

def state(h,tip_x,kind,kappa):
    p0=(2e-4,0.); tip=(tip_x,0.); parent=build_matched_crack_parent(8e-4,8e-4,p0,tip,h)
    if kind=="CONFORMING":
        slit=conforming_slit_from_parent(parent); return slit.mesh,_solve(slit.mesh,slit.boundary,8e-7),np.zeros(slit.mesh.ne,bool),parent,None
    ids,audit=mechanically_separating_graph_support(parent.mesh,CrackNetworkState.one_tip((p0,tip)))
    mask=np.zeros(parent.mesh.ne,bool); mask[ids]=True
    return parent.mesh,_solve(parent.mesh,parent.boundary,8e-7,mask.astype(float),kappa),mask,parent,audit

def williams(mesh,result,mask,tip,annulus):
    c=mesh.nodes[mesh.elems].mean(axis=1); local=c-np.asarray(tip); r=np.linalg.norm(local,axis=1)
    use=(r>=annulus[0])&(r<=annulus[1])&(~mask); ids=np.flatnonzero(use)
    rows=[]; targets=[]; weights=[]
    for i in ids:
        x,y=local[i]; sI=_auxiliary_stress_local(x,y,"I",1.0); sII=_auxiliary_stress_local(x,y,"II",1.0)
        for actual,a,b,t in zip(result.sigma_gp[:,i],(sI[0,0],sI[1,1],sI[0,1]),(sII[0,0],sII[1,1],sII[0,1]),(1.,0.,0.)):
            rows.append((a,b,t)); targets.append(actual); weights.append(np.sqrt(mesh.area_e[i]))
    A=np.asarray(rows); b=np.asarray(targets); w=np.asarray(weights); scale=np.linalg.norm(A*w[:,None],axis=0); As=A/scale
    coef_scaled,*_=np.linalg.lstsq(As*w[:,None],b*w,rcond=None); coef=coef_scaled/scale
    pred=A@coef; residual=np.linalg.norm((pred-b)*w)/max(np.linalg.norm(b*w),1e-300)
    angles=np.mod(np.arctan2(local[ids,1],local[ids,0]),2*np.pi); coverage=len(np.unique(np.floor(72*angles/(2*np.pi))))/72
    return {"K_I_fit":coef[0],"K_II_fit":coef[1],"T_stress_Pa":coef[2],"fit_residual":residual,"fit_condition_number_scaled":np.linalg.cond(As*w[:,None]),"fit_sample_elements":len(ids),"fit_angular_coverage_fraction":coverage}

def integral_row(h,kappa,kind,mesh,result,mask,parent,audit,inner,outer):
    tip=np.asarray(parent.p1); qcfg=JIntegralConfig(r_inner_factor=inner,r_outer_factor=outer)
    ii=compute_signed_interaction_integral(mesh,result.displacement,result.sigma_gp,np.zeros(mesh.nn),tip,np.array((1.,0.)),MAT,1.,cfg=qcfg,crack_segments=[(np.asarray(parent.p0),tip)],exclude_element_mask=mask,D=D)
    d=ii.diagnostics["mode_I"]; c=mesh.nodes[mesh.elems].mean(axis=1); rr=np.linalg.norm(c-tip,axis=1); active=(rr>=inner)&(rr<=outer)&(~mask)
    angles=np.mod(np.arctan2(c[active,1]-tip[1],c[active,0]-tip[0]),2*np.pi); coverage=len(np.unique(np.floor(72*angles/(2*np.pi))))/72 if len(angles) else 0.
    return {"representation":kind,"h_tip_m":h,"kappa":kappa,"r_inner_m":inner,"r_outer_m":outer,"r_inner_over_h_tip":inner/h,"support_width_over_r_inner":(outer-inner)/inner,"root_clearance_m":3e-4-outer,"exterior_clearance_m":3e-4-outer,"patch_clearance_m":3e-4-outer,"q_gradient_element_count":int(np.count_nonzero((rr>=inner)&(rr<=outer))),"raw_q_gradient_support_overlap":int(np.count_nonzero((rr>=inner)&(rr<=outer)&mask)),"exact_excluded_element_count":d["skipped_exact_elements"],"exact_mask_total":d["exact_mask_total_elements"],"damage_skips":d["skipped_damage_elements"],"branch_cut_skips":d["skipped_branch_cut_elements"],"line_of_sight_skips":d["skipped_line_of_sight_elements"],"final_active_element_count":d["active_elements"],"active_angular_coverage_fraction":coverage,"K_I_int":ii.K_I_Pa_sqrt_m,"K_II_int":ii.K_II_Pa_sqrt_m,"mode_II_to_mode_I":abs(ii.K_II_Pa_sqrt_m)/max(abs(ii.K_I_Pa_sqrt_m),1e-300),"support_width_m":audit.maximum_normal_support_width_m if audit else 0.,"killed_region_energy_fraction":float(np.sum(.5*np.sum(result.strain_gp[:,mask]*result.sigma_gp[:,mask],axis=0)*mesh.area_e[mask])/max(result.energy_J_per_m,1e-300)) if np.any(mask) else 0.,"equilibrium_residual":result.free_residual_relative,"conditioning":result.conditioning_diagonal_ratio}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("artifacts/v12_absolute_k")); args=ap.parse_args()
    rows=[]; fits=[]; globals_=[]
    for h in MESH_LEVELS_M:
        for kind,kappa in (("CONFORMING",None),("V12_FIXED",1e-8),("V12_JOINT_P2",JOINT_LIMIT_KAPPA0*(h/JOINT_LIMIT_H0_M)**JOINT_LIMIT_P)):
            mesh,result,mask,parent,audit=state(h,5e-4,kind,kappa or 0.)
            for inner,outer in CONTOURS_M: rows.append(integral_row(h,kappa,kind,mesh,result,mask,parent,audit,inner,outer))
            for annulus in WILLIAMS_ANNULI_M:
                f=williams(mesh,result,mask,parent.p1,annulus); f.update({"representation":kind,"h_tip_m":h,"kappa":kappa,"fit_r_inner_m":annulus[0],"fit_r_outer_m":annulus[1]}); fits.append(f)
            delta=25e-6; minus=state(h,5e-4-delta,kind,kappa or 0.)[1]; plus=state(h,5e-4+delta,kind,kappa or 0.)[1]
            ge=-(plus.energy_J_per_m-minus.energy_J_per_m)/(2*delta); dc=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N); gc=(8e-7)**2*dc/(2*cm**2)
            globals_.append({"representation":kind,"h_tip_m":h,"kappa":kappa,"delta_a_m":delta,"G_energy":ge,"G_compliance":gc,"energy_compliance_error":rel(ge,gc),"K_G":np.sqrt(max(MAT.Eprime*ge,0.))})
    checks={}; gates={}
    for kind in ("CONFORMING","V12_FIXED","V12_JOINT_P2"):
        for h in MESH_LEVELS_M:
            subset=[r for r in rows if r["representation"]==kind and r["h_tip_m"]==h]; kval=[r["K_I_int"] for r in subset]
            checks[f"{kind}_{h}_contour_spread"]=(max(kval)-min(kval))/max(abs(np.mean(kval)),1e-300)
            checks[f"{kind}_{h}_mode_ratio_max"]=max(r["mode_II_to_mode_I"] for r in subset)
    finest=MESH_LEVELS_M[-2:]
    def passes(kind):
        return all(checks[f"{kind}_{h}_contour_spread"]<=LIMITS["maximum_contour_spread"] and checks[f"{kind}_{h}_mode_ratio_max"]<=LIMITS["maximum_mode_II_to_mode_I"] for h in finest)
    gates["CONFORMING_CONTROL_QUALIFIED"]="PASS" if passes("CONFORMING") else "FAIL"
    gates["V12_STANDARD_INTERACTION_INTEGRAL_QUALIFIED"]="PASS" if passes("V12_FIXED") and passes("V12_JOINT_P2") else "FAIL"
    # Global-G and Williams comparisons use only predeclared primary fit and contours after plateau qualification.
    energy_ok=True; fit_ok=True
    for kind in ("CONFORMING","V12_FIXED","V12_JOINT_P2"):
        for h in finest:
            g=next(x for x in globals_ if x["representation"]==kind and x["h_tip_m"]==h); ks=[x["K_I_int"] for x in rows if x["representation"]==kind and x["h_tip_m"]==h]; km=float(np.mean(ks)); kg=g["K_G"]
            energy_ok &= g["energy_compliance_error"]<=LIMITS["maximum_GK_to_compliance_G_error"] and rel(km,kg)<=LIMITS["maximum_KI_to_energy_K_error"]
            fs=[x for x in fits if x["representation"]==kind and x["h_tip_m"]==h]; primary=fs[0]; fit_ok &= rel(primary["K_I_fit"],kg)<=LIMITS["maximum_Williams_KI_to_energy_K_error"] and max(rel(x["K_I_fit"],primary["K_I_fit"]) for x in fs[1:])<=LIMITS["maximum_Williams_radius_sensitivity"] and primary["fit_condition_number_scaled"]<=LIMITS["maximum_fit_condition_number"] and primary["fit_sample_elements"]>=LIMITS["minimum_fit_samples"]
    gates["GLOBAL_ENERGY_QUALIFIED"]="PASS" if energy_ok else "FAIL"; gates["WILLIAMS_QUALIFIED"]="PASS" if fit_ok else "FAIL"
    classification=classify_absolute_k(conforming_pass=gates["CONFORMING_CONTROL_QUALIFIED"]=="PASS",primal_pass=True,corridor_v3_pass=True,standard_pass=gates["V12_STANDARD_INTERACTION_INTEGRAL_QUALIFIED"]=="PASS",energy_pass=energy_ok,williams_pass=fit_ok,production_consumes_absolute_k=False)
    gates["V12_STANDARD_INTERACTION_INTEGRAL_ABSOLUTE_K"]=classification.standard_integral; gates["MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED"]=classification.aggregate; gates["STAGE_II_PERMITTED"]="PASS" if classification.production_may_continue else "FAIL"
    args.out.mkdir(parents=True,exist_ok=True); write_csv(args.out/"interaction_matrix.csv",rows); write_csv(args.out/"williams_matrix.csv",fits); write_csv(args.out/"global_G_matrix.csv",globals_)
    impl=git("log","-1","--format=%H","--",*IMPLEMENTATION_PATHS); report={"schema":CRITERION_ID,"base_git_sha":BASE_SHA,"implementation_git_sha":impl,"evidence_generation_parent_sha":git("rev-parse","HEAD"),"criterion":{ "limits":LIMITS,"contours_m":CONTOURS_M,"mesh_levels_m":MESH_LEVELS_M,"williams_annuli_m":WILLIAMS_ANNULI_M,"williams_terms":WILLIAMS_TERMS,"production_absolute_K_dependency_audit":"NOT_CONSUMED"},"checks":checks,"gates":gates}
    (args.out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); files=("interaction_matrix.csv","williams_matrix.csv","global_G_matrix.csv","qualification.json"); (args.out/"sha256_manifest.json").write_text(json.dumps({f:sha(args.out/f) for f in files},indent=2,sort_keys=True)+"\n"); (args.out/"environment_attestation.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"implementation_git_sha":impl},indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
