#!/usr/bin/env python3
"""Fail-closed V2 static explicit-cavity mechanics qualification."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arrhenius_fracture.voiding_v2 import build_explicit_hole_mesh, fill_explicit_hole_mesh, solve_static_hole

OPENING=8e-6
SOLVER="arrhenius_fracture.fem.production_CST_plane_strain/scipy.sparse.linalg.spsolve(SuperLU)"

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

def cavity_matrix(source_sha):
    refs=[("coarse",2e-4,48),("medium",1.3333333333333334e-4,72),("fine",1e-4,96)]; rows=[]
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
          "traction_l2_normalized":result.traction_l2_normalized,
          "traction_norm_definition":"sqrt(integral_boundary |sigma_CST*n|^2 ds)/(measured_remote_stress*sqrt(perimeter))",
          "traction_measure":"raw unique-adjacent-element CST traction; natural condition is weak",
          "hoop_stress_concentration":result.hoop_stress_concentration,"Kirsch_SC_relative_error":abs(result.hoop_stress_concentration-3)/3,
          "minimum_angle_deg":hole.validation["minimum_angle_deg"],"minimum_quality":hole.validation["minimum_quality"],
          "maximum_aspect_ratio":hole.validation["maximum_aspect_ratio"],"worst_element_region":"polar-to-rectangle transition",
          "actual_internal_components":hole.validation["actual_internal_components"],
          "triangle_disk_intersections":hole.validation["triangle_disk_intersections"],"orphan_nodes":hole.validation["orphan_nodes"],
          "remote_stress_Pa":abs(result.reaction_top_N_per_m)/.008,"linear_solver_status":"CONVERGED",
          "solver_identity":SOLVER,"git_sha":source_sha})
    return refs,rows

def interaction_matrix(refs,source_sha):
    rows=[]; width=.016; height=.008; tip=np.asarray((.002,0.))
    for label,h,n in refs:
      for radius in (.00010,.00015):
       for dr in (8,16,32,64):
        center=(tip[0]+radius*(dr+1),0.); hole=build_explicit_hole_mesh(width,height,center,radius,h,n)
        control=fill_explicit_hole_mesh(hole)
        rv=solve_static_hole(hole,OPENING,crack_tip_m=tuple(tip),wake_half_width_m=.75*h)
        rc=solve_static_hole(control,OPENING,crack_tip_m=tuple(tip),wake_half_width_m=.75*h)
        pv,pc=probes(hole,rv,tip,h),probes(control,rc,tip,h); scale=abs(rc.reaction_top_N_per_m/rv.reaction_top_N_per_m)
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
          "interaction_integral_status":"NOT_RUN","solver_identity":SOLVER,"git_sha":source_sha}
        for i in (1,2,3):
            row[f"void_probe_{i}_sigma_yy_Pa"]=pv[f"yy{i}"]; row[f"control_probe_{i}_sigma_yy_Pa"]=pc[f"yy{i}"]
            row[f"same_reaction_probe_{i}_relative_error"]=abs(scale*pv[f"yy{i}"]/pc[f"yy{i}"]-1)
        rows.append(row)
    return rows

def derivative_matrix(refs,source_sha):
    rows=[]; width=height=.008; center=(.004,0); radius=.00025; tip=.002
    for label,h,n in refs:
      base=build_explicit_hole_mesh(width,height,center,radius,h,n); layers=base.validation["radial_layers"]
      for kind,multiples in (("crack",(1.5,2.,3.)),("void",(.25,.5,.75))):
       for multiple in multiples:
        delta=multiple*h
        if kind=="crack":
            minus=solve_static_hole(base,OPENING,crack_tip_m=(tip-delta,0),wake_half_width_m=.75*h)
            plus=solve_static_hole(base,OPENING,crack_tip_m=(tip+delta,0),wake_half_width_m=.75*h)
            units="J/m^2"; normalization="-(1/B)dU/da"; method="identical nodes/elements; P0 wake extent perturbation"
        else:
            mm=build_explicit_hole_mesh(width,height,center,radius-delta,h,n,radial_layers_override=layers)
            pm=build_explicit_hole_mesh(width,height,center,radius+delta,h,n,radial_layers_override=layers)
            minus=solve_static_hole(mm,OPENING,crack_tip_m=(tip,0),wake_half_width_m=.75*h)
            plus=solve_static_hole(pm,OPENING,crack_tip_m=(tip,0),wake_half_width_m=.75*h)
            units="J/m"; normalization="-dU/dR (per unit thickness model)"; method="fixed connectivity/radial layers; smooth nodal geometry perturbation"
        dU=(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*delta)
        dC=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta)
        evalue=-dU; cmean=(plus.compliance_m2_per_N+minus.compliance_m2_per_N)/2
        cvalue=OPENING**2*dC/(2*cmean**2)
        rows.append({"derivative":kind,"mesh":label,"h_m":h,"delta_m":delta,"delta_over_h":multiple,
          "minus_energy_J_per_m":minus.stored_energy_J_per_m,"plus_energy_J_per_m":plus.stored_energy_J_per_m,
          "delta_squared_m2":delta**2,
          "minus_compliance_m2_per_N":minus.compliance_m2_per_N,"plus_compliance_m2_per_N":plus.compliance_m2_per_N,
          "energy_derivative_value":evalue,"compliance_derivative_value":cvalue,
          "energy_compliance_relative_error":abs(evalue-cvalue)/max(abs(evalue),abs(cvalue),1e-300),
          "reported_units":units,"normalization":normalization,
          "void_surface_normalized_G_J_per_m2":evalue/(2*math.pi) if kind=="void" else math.nan,
          "contribution":"elastic mechanical only; excludes capillarity and vacancy chemical potential",
          "paired_mesh_method":method,"solver_identity":SOLVER,"git_sha":source_sha})
    return rows

def derivative_status(rows,kind):
    selected=[r for r in rows if r["derivative"]==kind]; mid=2. if kind=="crack" else .5
    fine=[r["energy_derivative_value"] for r in selected if r["mesh"]=="fine"]
    mesh=[r["energy_derivative_value"] for r in selected if r["delta_over_h"]==mid]
    spread=lambda v:(max(v)-min(v))/max(max(abs(x) for x in v),1e-300)
    ps,ms=spread(fine),spread(mesh)
    return ("PASS" if ps<=.1 and ms<=.1 else "NOT_CONVERGED"),ps,ms

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("analysis_outputs/voiding_v2_static_fem")); a=p.parse_args()
    out=a.out; out.mkdir(parents=True,exist_ok=True)
    # Remove only the superseded V1-format matrix; never sweep arbitrary files
    # from a caller-provided output directory.
    (out/"prescribed_crack_void_interaction.csv").unlink(missing_ok=True)
    source_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    test_files=["tests/test_voiding_v2.py","tests/test_v11_process_state_ownership.py","tests/test_v11_live_topology_multitip.py",
      "tests/test_v11_causal_sharp_wake.py","tests/test_live_topology_kernel_v11.py","tests/test_crack_network_v11.py",
      "tests/test_topology_transaction_v11.py"]
    collected=subprocess.check_output([sys.executable,"-m","pytest","--collect-only","-q",*test_files],text=True)
    nodeids=[line for line in collected.splitlines() if "::" in line]
    (out/"qualification_test_inventory.txt").write_text("\n".join(nodeids)+"\n")
    refs,holes=cavity_matrix(source_sha); interactions=interaction_matrix(refs,source_sha); derivatives=derivative_matrix(refs,source_sha)
    write_csv(out/"circular_hole_fem_convergence.csv",holes); write_csv(out/"far_void_matched_mesh_matrix.csv",interactions)
    write_csv(out/"virtual_energy_derivatives.csv",derivatives)
    matplotlib.rcParams["svg.hashsalt"]="voiding-v2-static-fem"
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for ax,kind in zip(axes,("crack","void")):
        for label in ("coarse","medium","fine"):
            selected=[r for r in derivatives if r["derivative"]==kind and r["mesh"]==label]
            selected.sort(key=lambda r:r["delta_squared_m2"])
            ax.plot([r["delta_squared_m2"] for r in selected],[r["energy_derivative_value"] for r in selected],"o-",label=label)
        ax.set(xlabel=r"$\delta^2$ [m$^2$]",ylabel="elastic derivative",title=f"{kind} centered difference")
        ax.grid(True,alpha=.25); ax.legend()
    fig.savefig(out/"virtual_energy_derivative_delta2.svg",metadata={"Date":None}); plt.close(fig)
    cs,cstep,cmesh=derivative_status(derivatives,"crack"); vs,vstep,vmesh=derivative_status(derivatives,"void"); fine=holes[-1]
    far=[r for r in interactions if r["d_over_R"]==64 and r["mesh"] in ("medium","fine")]
    monotonic=[]
    for radius in (.00010,.00015):
        values=[r["same_reaction_KI_relative_error"] for r in interactions if r["mesh"]=="fine" and r["radius_m"]==radius]
        monotonic.append(all(values[i+1]<=values[i]+1e-12 for i in range(3)))
    checks=[check("actual_cavity_components",fine["actual_internal_components"],1,"equal",source_sha),
      check("triangle_disk_intersections",fine["triangle_disk_intersections"],0,"equal",source_sha),
      check("free_residual_relative",fine["free_residual_relative"],1e-10,"max",source_sha),
      check("energy_reaction_identity_relative",fine["energy_reaction_identity_relative"],1e-10,"max",source_sha),
      check("traction_l2_normalized_two_finest",max(r["traction_l2_normalized"] for r in holes[-2:]),.1,"max",source_sha),
      check("Kirsch_SC_error_two_finest",max(r["Kirsch_SC_relative_error"] for r in holes[-2:]),.1,"max",source_sha),
      check("minimum_angle_deg",min(r["minimum_angle_deg"] for r in holes),10,"min",source_sha),
      check("far_void_same_reaction_KI_error_two_finest",max(r["same_reaction_KI_relative_error"] for r in far),.1,"max",source_sha),
      check("far_void_monotonic_with_separation",all(monotonic),True,"equal",source_sha)]
    for metric,state,ps,ms in (("crack_virtual_derivative_convergence",cs,cstep,cmesh),("void_virtual_derivative_convergence",vs,vstep,vmesh)):
        checks.append({"metric":metric,"value":state,"perturbation_spread":ps,"mesh_spread":ms,"tolerance":"both <= 0.1",
          "relation":"max","status":"PASS" if state=="PASS" else "OPEN","calculation_source":"executed V2 static FEM runner","solver_identity":SOLVER,"git_sha":source_sha})
    checks.append({"metric":"interaction_integral_K_I_K_II","value":"NOT_RUN","tolerance":"required","relation":"","status":"OPEN",
      "calculation_source":"executed V2 static FEM runner","solver_identity":SOLVER,"git_sha":source_sha})
    write_csv(out/"acceptance_checks.csv",checks)
    env={"python":platform.python_version(),"python_executable":sys.executable,"implementation":platform.python_implementation(),
      "numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"sparse_solver":"SciPy spsolve using SuperLU",
      "plotting_dependency":"Matplotlib Agg/SVG with fixed hashsalt and no date metadata","table_dependency":"Python csv standard library","required_python":"3.12",
      "local_default_python_3_13_2":"SIGNAL_139_OBSERVED_DURING_PYTEST; environment issue, not physics result"}
    (out/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    gate="PASS" if all(r["status"]=="PASS" for r in checks) else "OPEN"
    decision={"schema":"voiding-v2-static-fem-decision/2","EXPLICIT_VOID_MECHANICS_QUALIFIED":gate,
      "reason":"all computed acceptance checks must pass" if gate=="PASS" else "one or more computed checks failed or remain OPEN",
      "checks":checks,"git_sha":source_sha,"solver_identity":SOLVER,
      "not_run":{"production_local_remeshing":"OUT_OF_SCOPE","production_crack_void_topology":"OUT_OF_SCOPE",
       "stochastic_lifecycle_integration":"OUT_OF_SCOPE","promotion_and_resolved_growth":"OUT_OF_SCOPE","fatigue":"OUT_OF_SCOPE"}}
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    files=[{"path":q.name,"sha256":sha(q),"bytes":q.stat().st_size} for q in sorted(out.glob("*")) if q.is_file() and q.name!="manifest.json"]
    (out/"manifest.json").write_text(json.dumps({"schema":"voiding-v2-artifacts/2","environment":env,"files":files},indent=2)+"\n")

if __name__=="__main__": main()
