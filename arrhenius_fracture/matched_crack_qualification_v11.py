"""Matched-mesh conforming/P0 crack qualification measurements for PR #57."""
from __future__ import annotations

from types import SimpleNamespace
import hashlib, math
import numpy as np

from .causal_sharp_wake_v11 import causal_segment_support, mechanical_fingerprint
from .config import ElasticProperties
from .conforming_crack_oracle_v11 import (
    build_matched_crack_parent, conforming_slit_from_parent,
    recovered_face_traction_relative, solve_conforming_slit,
)
from .fem import plane_strain_D
from .interaction_integral_v1029 import compute_signed_interaction_integral, _hermite_plateau_q
from .voiding_v2 import solve_static_hole

WIDTH=HEIGHT=.008; OPENING=8e-6; P0=np.array((.0005,0.)); TIP=np.array((.002,0.))
MAT=ElasticProperties(E=210e9,nu=.3); D=plane_strain_D(MAT)
HS=(25e-6,12.5e-6,6.25e-6); KAPPAS=(1e-4,1e-6,1e-8); RADII=(.0006,.0008,.001)

def _hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def _rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-300)
def _spread(a): return np.ptp(a)/max(np.max(np.abs(a)),1e-300)

def _mask(mesh,tip):
    ids,lengths=causal_segment_support(mesh,P0,np.asarray(tip,float)); mask=np.zeros(mesh.ne,bool); mask[ids]=1
    tri=mesh.nodes[mesh.elems[ids]]
    return mask,ids,lengths,{"selected_element_count":len(ids),"selected_element_ids":" ".join(map(str,ids)),
      "selected_area_m2":float(mesh.area_e[ids].sum()),"represented_intersection_length_m":float(lengths.sum()),
      "endpoint_footprint_error_m":float(np.max(tri[:,:,0])-tip[0]),"support_fingerprint":_hash(ids),
      "mechanical_fingerprint":mechanical_fingerprint(mesh,mask.astype(float))}

def _contour(mesh,result,mask,tip,radius):
    q=_hermite_plateau_q(np.linalg.norm(mesh.nodes-tip,axis=1),.3*radius,radius)
    support=np.flatnonzero(np.ptp(q[mesh.elems],axis=1)>1e-14); overlap=np.intersect1d(support,np.flatnonzero(mask))
    active=np.setdiff1d(support,np.flatnonzero(mask)); cent=mesh.nodes[mesh.elems[active]].mean(axis=1)
    theta=np.arctan2(cent[:,1]-tip[1],cent[:,0]-tip[0]); bins=len(np.unique(np.floor((theta+np.pi)/(2*np.pi)*24).astype(int)%24))
    value=compute_signed_interaction_integral(mesh,result.displacement,result.sigma_gp,np.zeros(mesh.nn),tip,np.array((1.,0.)),MAT,
      radius,cfg=SimpleNamespace(r_inner_factor=.3,r_outer_factor=1.),crack_segments=[(P0,tip)],D=D,exclude_element_mask=mask)
    return {"outer_radius_m":radius,"q_gradient_element_count":len(support),"excluded_p0_elements":len(overlap),
      "active_angular_bins_of_24":bins,"active_element_count":value.diagnostics["mode_I"]["active_elements"],
      "K_I_Pa_sqrt_m":value.K_I_Pa_sqrt_m,"K_II_Pa_sqrt_m":value.K_II_Pa_sqrt_m}

def _cod_parent(parent,result,h):
    x=.5*(P0[0]+TIP[0]); nodes=parent.hole.mesh.nodes; u=result.displacement.reshape(-1,2)
    top=int(np.argmin((nodes[:,0]-x)**2+(nodes[:,1]-h)**2)); bot=int(np.argmin((nodes[:,0]-x)**2+(nodes[:,1]+h)**2))
    return float(u[top,1]-u[bot,1])

def _cod_slit(slit,result):
    x=.5*(P0[0]+TIP[0]); nodes=slit.hole.mesh.nodes; u=result.displacement.reshape(-1,2)
    ids=np.flatnonzero(np.isclose(nodes[:,0],x)&np.isclose(nodes[:,1],0.))
    return float(np.ptp(u[ids,1]))

def _p0_tractions(mesh,result,mask):
    remote=abs(result.reaction_top_N_per_m)/WIDTH; sig=result.sigma_gp
    normal=float(np.sqrt(np.mean(sig[1,mask]**2))/remote); shear=float(np.sqrt(np.mean(sig[2,mask]**2))/remote)
    adjacency={}
    for ei,e in enumerate(mesh.elems):
        for a,b in ((e[0],e[1]),(e[1],e[2]),(e[2],e[0])): adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ei)
    interface=[]
    for edge,eids in adjacency.items():
        if len(eids)==2 and bool(mask[eids[0]])!=bool(mask[eids[1]]): interface.append(eids[0] if not mask[eids[0]] else eids[1])
    interface=np.asarray(interface,int)
    interface_norm=float(np.sqrt(np.mean(sig[1,interface]**2+sig[2,interface]**2))/remote) if len(interface) else math.nan
    return normal,shear,interface_norm

def run_matched_qualification(source_sha: str):
    geometry=[]; conforming=[]; conforming_contours=[]; matrix=[]; p0_contours=[]; staircase=[]; slopes=[]
    parents={h:build_matched_crack_parent(WIDTH,HEIGHT,tuple(P0),tuple(TIP),h) for h in HS}
    conforming_cache={}; p0_cache={}
    def csolve(h,tip):
        key=(h,float(tip[0]));
        if key not in conforming_cache:
            slit=conforming_slit_from_parent(parents[h],tuple(tip)); conforming_cache[key]=(slit,solve_conforming_slit(slit,OPENING,pin_node=slit.hole.boundary.left_bot))
        return conforming_cache[key]
    def psolve(h,kappa,tip):
        key=(h,kappa,float(tip[0]));
        if key not in p0_cache:
            parent=parents[h]; mask,ids,lengths,audit=_mask(parent.hole.mesh,tip)
            result=solve_static_hole(parent.hole,OPENING,crack_tip_m=tuple(tip),element_kill_mask=mask,
              rigid_pin_node=parent.hole.boundary.left_bot,residual_stiffness_kappa=kappa)
            p0_cache[key]=(result,mask,ids,lengths,audit)
        return p0_cache[key]
    for h,parent in parents.items():
        slit,base=csolve(h,TIP); normalized=slit.parent_node_of_node[slit.hole.mesh.elems]
        geometry.append({"h_tip_m":h,"parent_nodes":parent.hole.mesh.nn,"parent_elements":parent.hole.mesh.ne,
          "parent_geometry_fingerprint":parent.geometry_fingerprint,"parent_connectivity_fingerprint":parent.connectivity_fingerprint,
          "p0_geometry_fingerprint":_hash(parent.hole.mesh.nodes),"p0_connectivity_fingerprint":_hash(parent.hole.mesh.elems),
          "conforming_normalized_geometry_fingerprint":_hash(slit.hole.mesh.nodes[slit.parent_node_of_node]),
          "conforming_normalized_connectivity_fingerprint":_hash(normalized),"internal_face_components_from_incidence":slit.hole.validation["actual_internal_components"],"git_sha":source_sha})
        contour_local=[]
        for radius in RADII:
            row=_contour(slit.hole.mesh,base,np.zeros(slit.hole.mesh.ne,bool),TIP,radius); row.update({"h_tip_m":h,"git_sha":source_sha}); conforming_contours.append(row); contour_local.append(row)
        ki=float(np.median([r["K_I_Pa_sqrt_m"] for r in contour_local])); kii=float(np.median([r["K_II_Pa_sqrt_m"] for r in contour_local])); gk=(ki*ki+kii*kii)/MAT.Eprime
        for delta in (50e-6,75e-6):
            if delta < 4*h: continue
            _,minus=csolve(h,TIP-np.array((delta,0.))); _,plus=csolve(h,TIP+np.array((delta,0.)))
            ge=-(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*delta)
            dc=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N)
            gc=OPENING**2*dc/(2*cm**2)
            conforming.append({"h_tip_m":h,"delta_a_m":delta,"delta_a_over_h_tip":delta/h,"G_energy_J_per_m2":ge,
              "G_compliance_J_per_m2":gc,"G_K_J_per_m2":gk,"energy_compliance_relative_error":_rel(ge,gc),
              "energy_K_relative_error":_rel(ge,gk),"compliance_K_relative_error":_rel(gc,gk),
              "weak_face_residual_relative":base.weak_cavity_residual_relative,"full_face_traction_relative":recovered_face_traction_relative(slit,base),
              "fixed_trim_face_traction_relative":recovered_face_traction_relative(slit,base,trim_tip_distance_m=150e-6),
              "four_h_trim_face_traction_relative":recovered_face_traction_relative(slit,base,trim_tip_distance_m=4*h),
              "crack_opening_displacement_m":_cod_slit(slit,base),"reaction_N_per_m":base.reaction_top_N_per_m,
              "compliance_m2_per_N":base.compliance_m2_per_N,"contour_KI_spread_relative":_spread([r["K_I_Pa_sqrt_m"] for r in contour_local]),"git_sha":source_sha})
        for kappa in KAPPAS:
            result,mask,ids,lengths,audit=psolve(h,kappa,TIP); normal,shear,interface=_p0_tractions(parent.hole.mesh,result,mask)
            rows=[]
            for radius in RADII:
                row=_contour(parent.hole.mesh,result,mask,TIP,radius); cref=next(r for r in conforming_contours if r["h_tip_m"]==h and r["outer_radius_m"]==radius)
                row.update({"h_tip_m":h,"kappa":kappa,"conforming_KI_relative_error":_rel(row["K_I_Pa_sqrt_m"],cref["K_I_Pa_sqrt_m"]),
                  "interface_traction_relative":interface,"killed_energy_fraction":result.killed_element_energy_J_per_m/result.stored_energy_J_per_m,"git_sha":source_sha})
                p0_contours.append(row); rows.append(row)
            matrix.append({"h_tip_m":h,"kappa":kappa,"crack_opening_displacement_m":_cod_parent(parent,result,h),
              "residual_normal_traction_relative":normal,"residual_shear_traction_relative":shear,"interface_traction_relative":interface,
              "killed_element_energy_J_per_m":result.killed_element_energy_J_per_m,"killed_energy_fraction":result.killed_element_energy_J_per_m/result.stored_energy_J_per_m,
              "reaction_N_per_m":result.reaction_top_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.stored_energy_J_per_m,
              "free_residual_relative":result.free_residual_norm_N_per_m/abs(result.reaction_top_N_per_m),"conditioning_diagonal_ratio":result.conditioning_diagonal_ratio,
              "contour_KI_spread_relative":_spread([r["K_I_Pa_sqrt_m"] for r in rows]),**audit,"git_sha":source_sha})
        # Resolve the staircase at every local cell boundary over eight cells.
        previous_ids=np.empty(0,int)
        endpoints=TIP[0]+h*np.arange(-4,5)
        mesh_staircase=[]
        for endpoint in endpoints:
            tip=np.array((endpoint,0.)); result,mask,ids,lengths,audit=psolve(h,1e-8,tip)
            row={"h_tip_m":h,"kappa":1e-8,"physical_graph_length_m":endpoint-P0[0],"physical_p1_x_m":endpoint,
              "added_support":" ".join(map(str,np.setdiff1d(ids,previous_ids))),"removed_support":" ".join(map(str,np.setdiff1d(previous_ids,ids))),
              "reaction_N_per_m":result.reaction_top_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.stored_energy_J_per_m,
              **audit,"git_sha":source_sha}; staircase.append(row); mesh_staircase.append(row); previous_ids=ids
        for i,row in enumerate(mesh_staircase):
            row["one_sided_support_event_G_J_per_m2"]=(
              -(row["energy_J_per_m"]-mesh_staircase[i-1]["energy_J_per_m"])/h if i else math.nan)
            row["centered_resolved_G_J_per_m2"]=(
              -(mesh_staircase[i+1]["energy_J_per_m"]-mesh_staircase[i-1]["energy_J_per_m"])/(2*h)
              if 0<i<len(mesh_staircase)-1 and row["support_fingerprint"]!=mesh_staircase[i-1]["support_fingerprint"]
              and row["support_fingerprint"]!=mesh_staircase[i+1]["support_fingerprint"] else math.nan)
        # Fixed physical regression windows with at least four local cells.
        phase_values=[]
        window=max(50e-6,4*h)
        for phase in (-.5,0.,.5):
            center=TIP[0]+phase*h; x=np.linspace(center-window/2,center+window/2,5); energies=[]
            for endpoint in x: energies.append(psolve(h,1e-8,np.array((endpoint,0.)))[0].stored_energy_J_per_m)
            slope=-float(np.polyfit(x,np.asarray(energies),1)[0]); phase_values.append(slope)
            slopes.append({"h_tip_m":h,"kappa":1e-8,"window_m":window,"window_over_h_tip":window/h,
              "window_over_physical_crack_length":window/(TIP[0]-P0[0]),"phase_offset_over_h":phase,
              "G_regression_J_per_m2":slope,"git_sha":source_sha})
        for row in slopes[-3:]: row["phase_averaged_G_J_per_m2"]=float(np.mean(phase_values))
    # Centered global energy/compliance release for every residual-stiffness case.
    for row in matrix:
        h=row["h_tip_m"]; kappa=row["kappa"]; delta=max(50e-6,4*h)
        minus=psolve(h,kappa,TIP-np.array((delta,0.)))[0]; plus=psolve(h,kappa,TIP+np.array((delta,0.)))[0]
        row["global_G_delta_a_m"]=delta
        row["global_G_energy_J_per_m2"]=-(plus.stored_energy_J_per_m-minus.stored_energy_J_per_m)/(2*delta)
        dc=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N)
        row["global_G_compliance_J_per_m2"]=OPENING**2*dc/(2*cm**2)
    finest_conforming=[r for r in conforming if r["h_tip_m"] in HS[-2:]]
    conforming_checks={"two_finest_pairwise_G_agreement_max_relative":max(max(r["energy_compliance_relative_error"],r["energy_K_relative_error"],r["compliance_K_relative_error"]) for r in finest_conforming),
      "common_delta_mesh_spread_max_relative":max(_spread([r["G_energy_J_per_m2"] for r in finest_conforming if r["delta_a_m"]==d]) for d in (50e-6,75e-6)),
      "weak_face_residual_max_relative":max(r["weak_face_residual_relative"] for r in conforming),
      "fixed_trim_traction_decreases":conforming[-1]["fixed_trim_face_traction_relative"]<conforming[-3]["fixed_trim_face_traction_relative"]}
    oracle_pass=all((conforming_checks["two_finest_pairwise_G_agreement_max_relative"]<=.1,
      conforming_checks["common_delta_mesh_spread_max_relative"]<=.1,conforming_checks["weak_face_residual_max_relative"]<=1e-10,
      conforming_checks["fixed_trim_traction_decreases"]))
    # These strict matched-reference checks intentionally decide, without tuning.
    fine_matrix=[r for r in matrix if r["h_tip_m"]==HS[-1]]; low=fine_matrix[-1]
    cref=next(r for r in conforming if r["h_tip_m"]==HS[-1] and r["delta_a_m"]==50e-6)
    fine_phase=next(r["phase_averaged_G_J_per_m2"] for r in slopes if r["h_tip_m"]==HS[-1] and r["phase_offset_over_h"]==0.)
    p0_checks={"reaction_reference_error":_rel(low["reaction_N_per_m"],cref["reaction_N_per_m"]),
      "compliance_reference_error":_rel(low["compliance_m2_per_N"],cref["compliance_m2_per_N"]),
      "crack_opening_reference_error":_rel(low["crack_opening_displacement_m"],cref["crack_opening_displacement_m"]),
      "phase_energy_G_reference_error":_rel(fine_phase,cref["G_energy_J_per_m2"]),
      "residual_normal_traction_relative":low["residual_normal_traction_relative"],
      "interface_traction_relative":low["interface_traction_relative"],
      "killed_energy_fraction":low["killed_energy_fraction"],
      "kappa_low_pair_reaction_spread":_rel(fine_matrix[-1]["reaction_N_per_m"],fine_matrix[-2]["reaction_N_per_m"]),
      "absolute_KI_contour_spread":low["contour_KI_spread_relative"],
      "phase_averaged_G_mesh_spread":_spread([r["phase_averaged_G_J_per_m2"] for r in slopes if r["phase_offset_over_h"]==0.])}
    p0_pass=bool(oracle_pass and max(p0_checks.values())<=.1)
    return {"matched_geometry":geometry,"conforming_hardened":conforming,"conforming_contours":conforming_contours,
      "p0_residual_stiffness_matrix":matrix,"p0_contours":p0_contours,"p0_staircase":staircase,"p0_phase_slopes":slopes},conforming_checks,oracle_pass,p0_checks,p0_pass

__all__=["run_matched_qualification"]
