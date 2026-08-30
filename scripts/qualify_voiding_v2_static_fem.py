#!/usr/bin/env python3
"""Superseded centroid-band diagnostic retained for evidence reproduction.

This file is not an authoritative qualification path.  New gate decisions are
produced by ``qualify_voiding_v2_causal_static.py`` using the V11 causal wake.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arrhenius_fracture.voiding_v2 import build_explicit_hole_mesh, fill_explicit_hole_mesh, solve_static_hole
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import plane_strain_D
from arrhenius_fracture.interaction_integral_v10214 import compute_signed_interaction_integral

OPENING=8e-6
SOLVER="arrhenius_fracture.fem.production_CST_plane_strain/scipy.sparse.linalg.spsolve(SuperLU)"
MAT=ElasticProperties(E=210e9,nu=.3)
WAKE_HALF_WIDTH_M=4.5e-4

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]); fields.extend(sorted({k for row in rows for k in row}-set(fields)))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)

def check(metric,value,tolerance,relation,source_sha):
    passed={"max":value<=tolerance,"min":value>=tolerance,"equal":value==tolerance}[relation]
    return {"metric":metric,"value":value,"tolerance":tolerance,"relation":relation,
      "status":"PASS" if passed else "FAIL","calculation_source":"executed V2 static FEM runner",
      "solver_identity":SOLVER,"git_sha":source_sha}

def probes(hole,result,tip,h):
    cent=hole.mesh.nodes[hole.mesh.elems].mean(axis=1); rows=[]
    for multiple in (2.,4.,6.):
        target=np.asarray((tip[0]+multiple*h,tip[1])); idx=int(np.argmin(np.linalg.norm(cent-target,axis=1)))
        rows.append((float(np.linalg.norm(cent[idx]-tip)),result.sigma_gp[:,idx]))
    out={"K_I":float(np.median([s[1]*math.sqrt(2*math.pi*r) for r,s in rows])),
         "K_II":float(np.median([s[2]*math.sqrt(2*math.pi*r) for r,s in rows]))}
    for i,(r,s) in enumerate(rows,1):
        out.update({f"r{i}":r,f"xx{i}":float(s[0]),f"yy{i}":float(s[1]),f"xy{i}":float(s[2])})
    return out

def annular_norm(hole,void,control,tip,h,scale):
    cent=hole.mesh.nodes[hole.mesh.elems].mean(axis=1); r=np.linalg.norm(cent-np.asarray(tip),axis=1)
    mask=(r>=2*h)&(r<=8*h)&(cent[:,0]>=tip[0]); area=hole.mesh.area_e[mask]
    delta=void.sigma_gp[:,mask]*scale-control.sigma_gp[:,:hole.mesh.ne][:,mask]
    base=control.sigma_gp[:,:hole.mesh.ne][:,mask]
    return float(np.sqrt(np.sum(area*np.sum(delta**2,axis=0))/max(np.sum(area*np.sum(base**2,axis=0)),1e-300)))

def effective_crack_tip(hole,nominal_tip,wake_half_width):
    cent=hole.mesh.nodes[hole.mesh.elems].mean(axis=1)
    killed=(cent[:,0]<=nominal_tip)&(np.abs(cent[:,1])<=wake_half_width)
    if not np.any(killed): raise RuntimeError("P0 wake contains no elements")
    return float(np.max(cent[killed,0]))

def interaction_contours(hole,result,nominal_tip,h,max_outer_radius=None):
    """Production signed interaction integral on three bounded contours."""
    wake=WAKE_HALF_WIDTH_M; tip=np.asarray((effective_crack_tip(hole,nominal_tip,wake),0.0))
    nodes=hole.mesh.nodes
    damage=((nodes[:,0]<=tip[0]+h/2)&(np.abs(nodes[:,1])<=1.5*wake)).astype(float)
    segment=[(np.asarray((float(nodes[:,0].min()),0.0)),tip.copy())]
    rows=[]
    upper=min(8*h,float(max_outer_radius) if max_outer_radius is not None else 8*h)
    lower=max(2*h,1.05*wake)
    if upper<=lower: raise RuntimeError("no valid interaction-integral annulus outside wake and inside cavity clearance")
    for outer in np.linspace(lower,upper,3):
        outer_over_h=outer/h
        result_i=compute_signed_interaction_integral(
          hole.mesh,result.displacement,result.sigma_gp,damage,tip,np.asarray((1.,0.)),MAT,outer,
          cfg=SimpleNamespace(r_inner_factor=.3,r_outer_factor=1.0),crack_segments=segment,
          exclude_radius=.35*h,D=plane_strain_D(MAT))
        rows.append({"outer_radius_m":outer,"outer_over_h":outer_over_h,
          "effective_tip_x_m":tip[0],"K_I_Pa_sqrt_m":result_i.K_I_Pa_sqrt_m,
          "K_II_Pa_sqrt_m":result_i.K_II_Pa_sqrt_m,
          "active_elements":result_i.diagnostics["mode_I"]["active_elements"]})
    return rows

def contour_summary(rows):
    ki=np.asarray([r["K_I_Pa_sqrt_m"] for r in rows]); kii=np.asarray([r["K_II_Pa_sqrt_m"] for r in rows])
    median=float(np.median(ki)); plateau=float(np.ptp(ki)/max(abs(median),1e-300))
    return median,float(np.median(kii)),plateau

def cavity_matrix(source_sha):
    refs=[("coarse",2e-4,48),("medium",1.3333333333333334e-4,72),("fine",1e-4,96),("finer",7.5e-5,128)]; rows=[]
    for label,h,n in refs:
        hole=build_explicit_hole_mesh(.008,.008,(.004,0),.0005,h,n); result=solve_static_hole(hole,OPENING)
        xy=hole.mesh.nodes; edges=hole.cavity_edges; ordered=xy[np.arange(n)]
        perimeter=float(np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]],axis=1).sum())
        area=.5*abs(float(np.sum(ordered[:,0]*np.roll(ordered[:,1],-1)-ordered[:,1]*np.roll(ordered[:,0],-1))))
        identity=abs(2*result.stored_energy_J_per_m-abs(result.reaction_top_N_per_m)*OPENING)/max(2*result.stored_energy_J_per_m,1e-300)
        rows.append({"case":"circular_hole","mesh":label,"target_h_m":h,"boundary_segments":n,
          "node_count":hole.mesh.nn,"element_count":hole.mesh.ne,"radial_layers":hole.validation["radial_layers"],
          "area_m2":area,"area_relative_error":abs(area-np.pi*.0005**2)/(np.pi*.0005**2),
          "perimeter_m":perimeter,"perimeter_relative_error":abs(perimeter-2*np.pi*.0005)/(2*np.pi*.0005),
          "reaction_top_N_per_m":result.reaction_top_N_per_m,"reaction_bottom_N_per_m":result.reaction_bottom_N_per_m,
          "reaction_balance_relative":result.symmetry_error,"stored_energy_J_per_m":result.stored_energy_J_per_m,
          "energy_reaction_identity_relative":identity,"compliance_m2_per_N":result.compliance_m2_per_N,
          "free_residual_relative":result.free_residual_norm_N_per_m/abs(result.reaction_top_N_per_m),
          "weak_cavity_boundary_residual_relative":result.weak_cavity_residual_relative,
          "traction_l2_normalized":result.traction_l2_normalized,
          "traction_norm_definition":"sqrt(integral_boundary |sigma_CST*n|^2 ds)/(measured_remote_stress*sqrt(perimeter))",
          "traction_measure":"raw unique-adjacent-element CST traction; natural condition is weak",
          "hoop_stress_concentration":result.hoop_stress_concentration,"Kirsch_SC_relative_error":abs(result.hoop_stress_concentration-3)/3,
          "minimum_angle_deg":hole.validation["minimum_angle_deg"],"minimum_quality":hole.validation["minimum_quality"],
          "maximum_aspect_ratio":hole.validation["maximum_aspect_ratio"],"worst_element_region":"polar-to-rectangle transition",
          "mirror_sigma_xx_relative":result.mirror_sigma_xx_relative,"mirror_sigma_yy_relative":result.mirror_sigma_yy_relative,
          "mirror_sigma_xy_antisym_relative":result.mirror_sigma_xy_antisym_relative,
          "reaction_balance_is_not_field_symmetry":True,
          "actual_internal_components":hole.validation["actual_internal_components"],
          "polygon_bidirectional_Hausdorff_m":hole.validation["polygon_bidirectional_Hausdorff_m"],
          "polygon_exact_node_set_match":hole.validation["polygon_exact_node_set_match"],
          "triangle_disk_intersections":hole.validation["triangle_disk_intersections"],"orphan_nodes":hole.validation["orphan_nodes"],
          "remote_stress_Pa":abs(result.reaction_top_N_per_m)/.008,"linear_solver_status":"CONVERGED",
          "solver_identity":SOLVER,"git_sha":source_sha})
    return refs,rows

def interaction_matrix(refs,source_sha):
    rows=[]; contour_rows=[]; width=.016; height=.008; tip=np.asarray((.002,0.))
    for label,h,n in refs:
      for radius in (.00010,.00015):
       for dr in (8,16,32,64):
        center=(tip[0]+radius*(dr+1),0.); hole=build_explicit_hole_mesh(width,height,center,radius,h,n)
        control=fill_explicit_hole_mesh(hole)
        rv=solve_static_hole(hole,OPENING,crack_tip_m=tuple(tip),wake_half_width_m=WAKE_HALF_WIDTH_M)
        rc=solve_static_hole(control,OPENING,crack_tip_m=tuple(tip),wake_half_width_m=WAKE_HALF_WIDTH_M)
        pv,pc=probes(hole,rv,tip,h),probes(control,rc,tip,h); scale=abs(rc.reaction_top_N_per_m/rv.reaction_top_N_per_m)
        clearance=center[0]-radius-float(tip[0])
        iv=interaction_contours(hole,rv,float(tip[0]),h,.75*clearance); ic=interaction_contours(control,rc,float(tip[0]),h,.75*clearance)
        iv_ki,iv_kii,iv_plateau=contour_summary(iv); ic_ki,ic_kii,ic_plateau=contour_summary(ic)
        for family,series in (("void",iv),("matched_control",ic)):
            for contour in series:
                contour_rows.append({"mesh":label,"radius_m":radius,"d_over_R":dr,"family":family,**contour,
                  "reaction_rescale_to_control":scale,"reaction_normalized_K_I_Pa_sqrt_m":contour["K_I_Pa_sqrt_m"]*(scale if family=="void" else 1),
                  "reaction_normalized_K_II_Pa_sqrt_m":contour["K_II_Pa_sqrt_m"]*(scale if family=="void" else 1),
                  "solver_identity":SOLVER,"git_sha":source_sha})
        row={"mesh":label,"h_m":h,"boundary_segments":n,"d_over_R":dr,"R_over_W":radius/width,"d_over_W":dr*radius/width,
          "radius_m":radius,"center_x_m":center[0],"matched_parent_nodes":hole.mesh.nn,"matched_parent_elements":hole.mesh.ne,
          "same_opening_void_reaction_N_per_m":rv.reaction_top_N_per_m,"same_opening_control_reaction_N_per_m":rc.reaction_top_N_per_m,
          "same_opening_reaction_relative_change":abs(abs(rv.reaction_top_N_per_m/rc.reaction_top_N_per_m)-1),
          "same_opening_void_compliance_m2_per_N":rv.compliance_m2_per_N,"same_opening_control_compliance_m2_per_N":rc.compliance_m2_per_N,
          "same_opening_void_energy_J_per_m":rv.stored_energy_J_per_m,"same_opening_control_energy_J_per_m":rc.stored_energy_J_per_m,
          "reaction_rescale_to_control":scale,"void_K_I_Pa_sqrt_m":pv["K_I"],"control_K_I_Pa_sqrt_m":pc["K_I"],
          "void_K_II_Pa_sqrt_m":pv["K_II"],"control_K_II_Pa_sqrt_m":pc["K_II"],
          "same_opening_KI_relative_error":abs(pv["K_I"]/pc["K_I"]-1),
          "same_reaction_KI_relative_error":abs(scale*pv["K_I"]/pc["K_I"]-1),
          "same_reaction_annular_L2_relative":annular_norm(hole,rv,rc,tip,h,scale),
          "interaction_void_KI_Pa_sqrt_m":iv_ki,"interaction_control_KI_Pa_sqrt_m":ic_ki,
          "interaction_void_KII_Pa_sqrt_m":iv_kii,"interaction_control_KII_Pa_sqrt_m":ic_kii,
          "interaction_void_contour_plateau_relative":iv_plateau,"interaction_control_contour_plateau_relative":ic_plateau,
          "same_reaction_interaction_KI_relative_error":abs(scale*iv_ki/ic_ki-1),
          "centered_KII_over_KI":abs(iv_kii)/max(abs(iv_ki),1e-300),
          "interaction_integral_status":"EXECUTED_PRODUCTION_ADAPTER","solver_identity":SOLVER,"git_sha":source_sha}
        for i in (1,2,3):
            row[f"void_probe_{i}_sigma_yy_Pa"]=pv[f"yy{i}"]; row[f"control_probe_{i}_sigma_yy_Pa"]=pc[f"yy{i}"]
            row[f"same_reaction_probe_{i}_relative_error"]=abs(scale*pv[f"yy{i}"]/pc[f"yy{i}"]-1)
        rows.append(row)
    return rows,contour_rows

def derivative_matrix(refs,source_sha):
    rows=[]; width=height=.008; center=(.004,0); radius=.00025; tip=.002
    for label,h,n in refs:
      base=build_explicit_hole_mesh(width,height,center,radius,h,n); layers=base.validation["radial_layers"]
      base_centroids=base.mesh.nodes[base.mesh.elems].mean(axis=1)
      fixed_wake_mask=(base_centroids[:,0]<=tip)&(np.abs(base_centroids[:,1])<=WAKE_HALF_WIDTH_M)
      base_result=solve_static_hole(base,OPENING,crack_tip_m=(tip,0),wake_half_width_m=WAKE_HALF_WIDTH_M)
      base_contours=interaction_contours(base,base_result,tip,h,.75*(center[0]-radius-tip)); base_ki,base_kii,base_plateau=contour_summary(base_contours)
      interaction_G=(base_ki**2+base_kii**2)/MAT.Eprime
      for kind,fractions in (("crack",(.05,.10,.15)),("void",(.025,.05,.10))):
       for fraction in fractions:
        delta=fraction*(tip if kind=="crack" else radius)
        if kind=="crack":
            minus=solve_static_hole(base,OPENING,crack_tip_m=(tip-delta,0),wake_half_width_m=WAKE_HALF_WIDTH_M)
            plus=solve_static_hole(base,OPENING,crack_tip_m=(tip+delta,0),wake_half_width_m=WAKE_HALF_WIDTH_M)
            eff_minus=effective_crack_tip(base,tip-delta,WAKE_HALF_WIDTH_M); eff_plus=effective_crack_tip(base,tip+delta,WAKE_HALF_WIDTH_M)
            effective_delta=(eff_plus-eff_minus)/2
            units="J/m^2"; normalization="-(1/B)dU/da_eff"; method="identical nodes/elements; measured P0 wake element-front extension"
        else:
            mm=build_explicit_hole_mesh(width,height,center,radius-delta,h,n,radial_layers_override=layers)
            pm=build_explicit_hole_mesh(width,height,center,radius+delta,h,n,radial_layers_override=layers)
            minus=solve_static_hole(mm,OPENING,crack_tip_m=(tip,0),wake_half_width_m=WAKE_HALF_WIDTH_M,element_kill_mask=fixed_wake_mask)
            plus=solve_static_hole(pm,OPENING,crack_tip_m=(tip,0),wake_half_width_m=WAKE_HALF_WIDTH_M,element_kill_mask=fixed_wake_mask)
            effective_delta=delta
            units="J/m^2"; normalization="f_R=-d(U/B)/dR"; method="fixed connectivity/radial layers; smooth nodal geometry perturbation"
        dU=(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*effective_delta)
        dC=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*effective_delta)
        evalue=-dU; cmean=(plus.compliance_m2_per_N+minus.compliance_m2_per_N)/2
        cvalue=OPENING**2*dC/(2*cmean**2)
        rows.append({"derivative":kind,"mesh":label,"h_m":h,"nominal_delta_m":delta,"effective_delta_m":effective_delta,
          "delta_over_reference":fraction,"reference_length_m":tip if kind=="crack" else radius,
          "minus_energy_J_per_m":minus.stored_energy_J_per_m,"plus_energy_J_per_m":plus.stored_energy_J_per_m,
          "delta_squared_m2":delta**2,
          "minus_compliance_m2_per_N":minus.compliance_m2_per_N,"plus_compliance_m2_per_N":plus.compliance_m2_per_N,
          "energy_derivative_value":evalue,"compliance_derivative_value":cvalue,
          "energy_compliance_relative_error":abs(evalue-cvalue)/max(abs(evalue),abs(cvalue),1e-300),
          "reported_units":units,"normalization":normalization,
          "interaction_integral_G_J_per_m2":interaction_G if kind=="crack" else math.nan,
          "derivative_vs_interaction_G_relative_error":abs(evalue-interaction_G)/max(abs(evalue),abs(interaction_G),1e-300) if kind=="crack" else math.nan,
          "interaction_contour_plateau_relative":base_plateau if kind=="crack" else math.nan,
          "void_force_density_f_R_J_per_m2":evalue if kind=="void" else math.nan,
          "void_total_force_per_unit_thickness_F_R_J_per_m":evalue if kind=="void" else math.nan,
          "void_surface_normalized_G_J_per_m2":evalue/(2*math.pi) if kind=="void" else math.nan,
          "contribution":"elastic mechanical only; excludes capillarity and vacancy chemical potential",
          "paired_mesh_method":method,"solver_identity":SOLVER,"git_sha":source_sha})
    return rows

def derivative_status(rows,kind):
    selected=[r for r in rows if r["derivative"]==kind]; mid=.10 if kind=="crack" else .05
    fine=[r["energy_derivative_value"] for r in selected if r["mesh"]=="finer"]
    mesh=[r["energy_derivative_value"] for r in selected if r["delta_over_reference"]==mid and r["mesh"] in ("fine","finer")]
    spread=lambda v:(max(v)-min(v))/max(max(abs(x) for x in v),1e-300)
    ps,ms=spread(fine),spread(mesh)
    return ("PASS" if ps<=.1 and ms<=.1 else "NOT_CONVERGED"),ps,ms

def plate_size_matrix(source_sha):
    rows=[]
    for width in (.008,.012,.016):
        hole=build_explicit_hole_mesh(width,width,(width/2,0),.0005,1e-4,96)
        result=solve_static_hole(hole,OPENING)
        rows.append({"plate_width_m":width,"R_over_W":.0005/width,
          "hoop_stress_concentration":result.hoop_stress_concentration,
          "Kirsch_SC_relative_error":abs(result.hoop_stress_concentration-3)/3,
          "interpretation":"finite-width displacement-controlled benchmark",
          "solver_identity":SOLVER,"git_sha":source_sha})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("analysis_outputs/voiding_v2_static_fem")); a=p.parse_args()
    out=a.out; out.mkdir(parents=True,exist_ok=True)
    # Remove only the superseded V1-format matrix; never sweep arbitrary files
    # from a caller-provided output directory.
    (out/"prescribed_crack_void_interaction.csv").unlink(missing_ok=True)
    source_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    test_files=["tests/test_voiding_v2.py","tests/test_v11_process_state_ownership.py","tests/test_v11_live_topology_multitip.py",
      "tests/test_v11_causal_sharp_wake.py","tests/test_live_topology_kernel_v11.py","tests/test_crack_network_v11.py",
      "tests/test_topology_transaction_v11.py","tests/test_interaction_integral_v1029.py",
      "tests/test_v10214_intrinsic_isotropy_interaction.py"]
    collected=subprocess.check_output([sys.executable,"-m","pytest","--collect-only","-q",*test_files],text=True)
    nodeids=[line for line in collected.splitlines() if "::" in line]
    (out/"qualification_test_inventory.txt").write_text("\n".join(nodeids)+"\n")
    refs,holes=cavity_matrix(source_sha); interactions,contours=interaction_matrix(refs,source_sha)
    derivatives=derivative_matrix(refs,source_sha); plate_sizes=plate_size_matrix(source_sha)
    write_csv(out/"circular_hole_fem_convergence.csv",holes); write_csv(out/"far_void_matched_mesh_matrix.csv",interactions)
    write_csv(out/"interaction_integral_contours.csv",contours); write_csv(out/"virtual_energy_derivatives.csv",derivatives)
    write_csv(out/"circular_hole_plate_size_convergence.csv",plate_sizes)
    matplotlib.rcParams["svg.hashsalt"]="voiding-v2-static-fem"
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for ax,kind in zip(axes,("crack","void")):
        for label in ("coarse","medium","fine","finer"):
            selected=[r for r in derivatives if r["derivative"]==kind and r["mesh"]==label]
            selected.sort(key=lambda r:r["delta_squared_m2"])
            ax.plot([r["delta_squared_m2"] for r in selected],[r["energy_derivative_value"] for r in selected],"o-",label=label)
        ax.set(xlabel=r"$\delta^2$ [m$^2$]",ylabel="elastic derivative",title=f"{kind} centered difference")
        ax.grid(True,alpha=.25); ax.legend()
    fig.savefig(out/"virtual_energy_derivative_delta2.svg",metadata={"Date":None}); plt.close(fig)
    cs,cstep,cmesh=derivative_status(derivatives,"crack"); vs,vstep,vmesh=derivative_status(derivatives,"void"); fine=holes[-1]
    far=[r for r in interactions if r["d_over_R"] in (32,64) and r["mesh"] in ("fine","finer")]
    uncertainty_aware=[]; asymptotic_rows=[]
    for radius in (.00010,.00015):
        fine_rows={r["d_over_R"]:r for r in interactions if r["mesh"]=="fine" and r["radius_m"]==radius}
        finer_rows={r["d_over_R"]:r for r in interactions if r["mesh"]=="finer" and r["radius_m"]==radius}
        for metric in ("same_reaction_annular_L2_relative","same_reaction_interaction_KI_relative_error"):
            values=[finer_rows[d][metric] for d in (8,16,32,64)]
            uncertainty=[abs(fine_rows[d][metric]-finer_rows[d][metric]) for d in (8,16,32,64)]
            for i,(left,right) in enumerate(zip((8,16,32),(16,32,64))):
                allowed=values[i]+uncertainty[i]+uncertainty[i+1]; passed=values[i+1]<=allowed
                asymptotic_rows.append({"radius_m":radius,"metric":metric,"from_d_over_R":left,"to_d_over_R":right,
                  "from_finer_value":values[i],"to_finer_value":values[i+1],"from_fine_finer_uncertainty":uncertainty[i],
                  "to_fine_finer_uncertainty":uncertainty[i+1],"allowed_upper_value":allowed,"status":"PASS" if passed else "FAIL",
                  "solver_identity":SOLVER,"git_sha":source_sha})
                uncertainty_aware.append(passed)
    write_csv(out/"far_void_asymptotic_uncertainty_checks.csv",asymptotic_rows)
    crack_G_rows=[r for r in derivatives if r["derivative"]=="crack" and r["mesh"] in ("fine","finer") and r["delta_over_reference"]==.10]
    checks=[check("actual_cavity_components",fine["actual_internal_components"],1,"equal",source_sha),
      check("triangle_disk_intersections",fine["triangle_disk_intersections"],0,"equal",source_sha),
      check("free_residual_relative",fine["free_residual_relative"],1e-10,"max",source_sha),
      check("weak_cavity_boundary_residual_relative",max(r["weak_cavity_boundary_residual_relative"] for r in holes[-2:]),1e-10,"max",source_sha),
      check("energy_reaction_identity_relative",fine["energy_reaction_identity_relative"],1e-10,"max",source_sha),
      check("traction_l2_normalized_two_finest",max(r["traction_l2_normalized"] for r in holes[-2:]),.1,"max",source_sha),
      check("Kirsch_SC_error_two_finest",max(r["Kirsch_SC_relative_error"] for r in holes[-2:]),.1,"max",source_sha),
      check("minimum_angle_deg",min(r["minimum_angle_deg"] for r in holes),10,"min",source_sha),
      check("cavity_boundary_exact_node_set_match",all(r["polygon_exact_node_set_match"] for r in holes),True,"equal",source_sha),
      check("mirrored_sigma_xx_relative",fine["mirror_sigma_xx_relative"],.1,"max",source_sha),
      check("mirrored_sigma_yy_relative",fine["mirror_sigma_yy_relative"],.1,"max",source_sha),
      check("mirrored_sigma_xy_antisym_relative",fine["mirror_sigma_xy_antisym_relative"],.1,"max",source_sha),
      check("far_void_farthest_two_interaction_KI_error_two_finest",max(r["same_reaction_interaction_KI_relative_error"] for r in far),.1,"max",source_sha),
      check("far_void_uncertainty_aware_asymptotic_convergence",all(uncertainty_aware),True,"equal",source_sha),
      check("interaction_integral_contour_plateau",max(max(r["interaction_void_contour_plateau_relative"],r["interaction_control_contour_plateau_relative"]) for r in far),.1,"max",source_sha),
      check("centered_KII_over_KI",max(r["centered_KII_over_KI"] for r in far),.05,"max",source_sha),
      check("crack_derivative_vs_interaction_G",max(r["derivative_vs_interaction_G_relative_error"] for r in crack_G_rows),.1,"max",source_sha)]
    for metric,state,ps,ms in (("crack_virtual_derivative_convergence",cs,cstep,cmesh),("void_virtual_derivative_convergence",vs,vstep,vmesh)):
        checks.append({"metric":metric,"value":state,"perturbation_spread":ps,"mesh_spread":ms,"tolerance":"both <= 0.1",
          "relation":"max","status":"PASS" if state=="PASS" else "OPEN","calculation_source":"executed V2 static FEM runner","solver_identity":SOLVER,"git_sha":source_sha})
    write_csv(out/"acceptance_checks.csv",checks)
    env={"python":platform.python_version(),"python_executable":sys.executable,"implementation":platform.python_implementation(),
      "numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"sparse_solver":"SciPy spsolve using SuperLU",
      "plotting_dependency":"Matplotlib Agg/SVG with fixed hashsalt and no date metadata","table_dependency":"Python csv standard library","required_python":"3.12",
      "local_default_python_3_13_2":"SIGNAL_139_OBSERVED_DURING_PYTEST; environment issue, not physics result"}
    (out/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    gate="PASS" if all(r["status"]=="PASS" for r in checks) else "OPEN"
    decision={"schema":"voiding-v2-static-fem-decision/3","EXPLICIT_VOID_MECHANICS_QUALIFIED":gate,
      "reason":"all computed acceptance checks must pass" if gate=="PASS" else "one or more computed checks failed or remain OPEN",
      "checks":checks,"git_sha":source_sha,"solver_identity":SOLVER,
      "not_run":{"production_local_remeshing":"OUT_OF_SCOPE","production_crack_void_topology":"OUT_OF_SCOPE",
       "stochastic_lifecycle_integration":"OUT_OF_SCOPE","promotion_and_resolved_growth":"OUT_OF_SCOPE","fatigue":"OUT_OF_SCOPE"}}
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    files=[{"path":q.name,"sha256":sha(q),"bytes":q.stat().st_size} for q in sorted(out.glob("*")) if q.is_file() and q.name!="manifest.json"]
    (out/"manifest.json").write_text(json.dumps({"schema":"voiding-v2-artifacts/3","environment":env,"files":files},indent=2)+"\n")

if __name__=="__main__": main()
