"""Matched-parent, analysis-only V11/V12/conforming primal mechanics screen."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import numpy as np
from scipy.sparse.linalg import eigsh, spsolve

from .causal_sharp_wake_v11 import causal_segment_support
from .config import ElasticProperties
from .conforming_crack_oracle_v12 import build_matched_crack_parent, conforming_slit_from_parent
from .crack_network_v11 import CrackNetworkState
from .fem import assemble_mechanics, plane_strain_D
from .mechanically_separating_sharp_wake_v12 import mechanically_separating_graph_support
from .mesh import rebuild_tri_mesh

MAT=ElasticProperties(E=210e9,nu=.3); D=plane_strain_D(MAT)


@dataclass(frozen=True)
class PrimalResult:
    displacement: np.ndarray; sigma_gp: np.ndarray; strain_gp: np.ndarray
    reaction_N_per_m: float; compliance_m2_per_N: float; energy_J_per_m: float
    free_residual_relative: float; energy_reaction_identity_relative: float
    conditioning_diagonal_ratio: float
    crack_opening_displacement_m: float


def _solve(mesh,boundary,opening_m,damage=None,kappa=0.):
    if damage is not None: mesh=replace(mesh,element_damage_gp=np.asarray(damage,float))
    u=np.zeros(mesh.ndof); ep=np.zeros((3,mesh.ne)); rho=np.zeros(mesh.ne); d=np.zeros(mesh.nn)
    K,R,*_=assemble_mechanics(mesh,u,ep,rho,d,D,MAT,kappa=kappa)
    prescribed=np.zeros(mesh.ndof,bool); values=np.zeros(mesh.ndof)
    prescribed[2*boundary.top_nodes+1]=1; values[2*boundary.top_nodes+1]=opening_m/2
    prescribed[2*boundary.bot_nodes+1]=1; values[2*boundary.bot_nodes+1]=-opening_m/2
    prescribed[2*boundary.left_bot]=1; prescribed[2*boundary.right_bot]=1
    free=~prescribed; u[prescribed]=values[prescribed]
    Kff=K[np.ix_(free,free)]; rhs=-K[np.ix_(free,prescribed)]@u[prescribed]; u[free]=spsolve(Kff,rhs)
    residual=K@u; reaction=float(np.sum(residual[2*boundary.top_nodes+1])); energy=float(.5*u@(K@u))
    strain=np.einsum('eij,ej->ei',mesh.B_e,u[np.c_[2*mesh.elems,2*mesh.elems+1].reshape(mesh.ne,6)]).T
    degradation=np.ones(mesh.ne) if damage is None else (1-np.asarray(damage))**2+kappa
    sigma=(degradation[:,None]*(strain.T@D.T)).T
    free_res=float(np.linalg.norm(residual[free])/max(abs(reaction),1e-300))
    identity=abs(energy-.5*reaction*opening_m)/max(abs(energy),1e-300)
    diag=Kff.diagonal(); cond=float(np.max(diag)/np.min(diag))
    return PrimalResult(u,sigma,strain,reaction,opening_m/abs(reaction),energy,free_res,identity,cond,float("nan"))


def _solve_vector(mesh,boundary,opening_m,normal,damage=None,kappa=0.):
    """Solve a rotated specimen with full-vector platen displacements."""
    if damage is not None: mesh=replace(mesh,element_damage_gp=np.asarray(damage,float))
    u=np.zeros(mesh.ndof); ep=np.zeros((3,mesh.ne)); rho=np.zeros(mesh.ne); d=np.zeros(mesh.nn)
    K,R,*_=assemble_mechanics(mesh,u,ep,rho,d,D,MAT,kappa=kappa)
    prescribed=np.zeros(mesh.ndof,bool); values=np.zeros(mesh.ndof); normal=np.asarray(normal,float)
    for nodes,sign in ((boundary.top_nodes,1.),(boundary.bot_nodes,-1.)):
        for component in (0,1):
            prescribed[2*nodes+component]=1; values[2*nodes+component]=sign*.5*opening_m*normal[component]
    free=~prescribed; u[prescribed]=values[prescribed]; Kff=K[np.ix_(free,free)]
    u[free]=spsolve(Kff,-K[np.ix_(free,prescribed)]@u[prescribed]); residual=K@u
    top=np.ravel(np.c_[2*boundary.top_nodes,2*boundary.top_nodes+1]); reaction_vec=residual[top].reshape(-1,2).sum(axis=0)
    reaction=float(reaction_vec@normal); energy=float(.5*u@(K@u)); strain=np.einsum('eij,ej->ei',mesh.B_e,u[np.c_[2*mesh.elems,2*mesh.elems+1].reshape(mesh.ne,6)]).T
    degradation=np.ones(mesh.ne) if damage is None else (1-np.asarray(damage))**2+kappa; sigma=(degradation[:,None]*(strain.T@D.T)).T
    return PrimalResult(u,sigma,strain,reaction,opening_m/abs(reaction),energy,float(np.linalg.norm(residual[free])/abs(reaction)),abs(energy-.5*reaction*opening_m)/abs(energy),float(np.max(Kff.diagonal())/np.min(Kff.diagonal())),float("nan"))


def _relative(a,b): return float(abs(a-b)/max(abs(a),abs(b),1e-300))
def _hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _parent_cod(result,mesh,p0,tip,h):
    x=.5*(p0[0]+tip[0]); points=((x,h),(x,-h)); ids=[]
    for point in points: ids.append(int(np.argmin(np.sum((mesh.nodes-np.asarray(point))**2,axis=1))))
    u=result.displacement.reshape(-1,2); return float(u[ids[0],1]-u[ids[1],1])


def _slit_cod(result,slit,p0,tip):
    x=.5*(p0[0]+tip[0]); ids=np.flatnonzero(np.isclose(slit.mesh.nodes[:,0],x)&np.isclose(slit.mesh.nodes[:,1],0.))
    return float(np.ptp(result.displacement.reshape(-1,2)[ids,1]))


def run_straight_case(h_values=(25e-6,12.5e-6,6.25e-6),kappas=(1e-4,1e-6,1e-8),opening_m=8e-7):
    width=height=8e-4; p0=np.array((2e-4,0.)); tip=np.array((5e-4,0.)); delta=25e-6
    rows=[]; derivatives=[]
    cache={}
    for h in h_values:
        parent=build_matched_crack_parent(width,height,tuple(p0),tuple(tip),h); mesh=parent.mesh
        v11_ids,_=causal_segment_support(mesh,p0,tip); v11=np.zeros(mesh.ne); v11[v11_ids]=1
        v12_ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((tuple(p0),tuple(tip))))
        v12=np.zeros(mesh.ne); v12[v12_ids]=1; slit=conforming_slit_from_parent(parent)
        intact=_solve(mesh,parent.boundary,opening_m); conforming=_solve(slit.mesh,slit.boundary,opening_m)
        conforming=replace(conforming,crack_opening_displacement_m=_slit_cod(conforming,slit,p0,tip))
        for representation,result,mask in (("A_INTACT",intact,np.zeros(mesh.ne,bool)),("B_V11",_solve(mesh,parent.boundary,opening_m,v11,1e-8),v11.astype(bool)),("D_CONFORMING",conforming,np.zeros(mesh.ne,bool))):
            rows.append(_row(h,None,representation,result,conforming,mesh,mask,audit if representation=="B_V11" else None,parent))
        for kappa in kappas:
            result=_solve(mesh,parent.boundary,opening_m,v12,kappa); result=replace(result,crack_opening_displacement_m=_parent_cod(result,mesh,p0,tip,h)); cache[(h,kappa,float(tip[0]))]=(result,v12)
            rows.append(_row(h,kappa,"C_V12",result,conforming,mesh,v12.astype(bool),audit,parent))
    for h in h_values[-2:]:
        for kappa in kappas:
            values={}
            for sign in (-1,1):
                moved=tip+np.array((sign*delta,0.)); parent=build_matched_crack_parent(width,height,tuple(p0),tuple(moved),h)
                ids,_=mechanically_separating_graph_support(parent.mesh,CrackNetworkState.one_tip((tuple(p0),tuple(moved))))
                damage=np.isin(np.arange(parent.mesh.ne),ids).astype(float); values[("v12",sign)]=_solve(parent.mesh,parent.boundary,opening_m,damage,kappa)
                slit=conforming_slit_from_parent(parent); values[("conf",sign)]=_solve(slit.mesh,slit.boundary,opening_m)
            for name in ("v12","conf"):
                minus,plus=values[(name,-1)],values[(name,1)]; ge=-(plus.energy_J_per_m-minus.energy_J_per_m)/(2*delta)
                dc=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N)
                gc=opening_m**2*dc/(2*cm**2)
                derivatives.append({"h_tip_m":h,"kappa":kappa if name=="v12" else None,"representation":name.upper(),"delta_a_m":delta,"G_energy_J_per_m2":ge,"G_compliance_J_per_m2":gc,"energy_compliance_relative_error":_relative(ge,gc)})
    return rows,derivatives


def run_rotated_cases(angles=(30.,45.),h_values=(25e-6,12.5e-6,6.25e-6),kappas=(1e-4,1e-6,1e-8),opening_m=8e-7):
    """Matched rotation-covariance screen after the straight gate passes."""
    width=height=8e-4; base_p0=np.array((2e-4,0.)); base_tip=np.array((5e-4,0.)); center=np.array((4e-4,0.)); rows=[]
    for angle in angles:
        theta=np.deg2rad(angle); rotation=np.array(((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta)))); normal=rotation@np.array((0.,1.))
        for h in h_values:
            parent=build_matched_crack_parent(width,height,tuple(base_p0),tuple(base_tip),h); slit=conforming_slit_from_parent(parent)
            rotate=lambda nodes: (nodes-center)@rotation.T+center
            mesh=rebuild_tri_mesh(rotate(parent.mesh.nodes),parent.mesh.elems); slit_mesh=rebuild_tri_mesh(rotate(slit.mesh.nodes),slit.mesh.elems)
            p0=rotate(base_p0[None,:])[0]; tip=rotate(base_tip[None,:])[0]
            v11_ids,_=causal_segment_support(mesh,p0,tip); v11=np.isin(np.arange(mesh.ne),v11_ids).astype(float)
            v12_ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((tuple(p0),tuple(tip))),allow_offgrid_active_tips_for_screen=True); v12=np.isin(np.arange(mesh.ne),v12_ids).astype(float)
            conforming=_solve_vector(slit_mesh,slit.boundary,opening_m,normal)
            cases=(("A_INTACT",None,_solve_vector(mesh,parent.boundary,opening_m,normal),np.zeros(mesh.ne)),("B_V11",1e-8,_solve_vector(mesh,parent.boundary,opening_m,normal,v11,1e-8),v11),("D_CONFORMING",None,conforming,np.zeros(mesh.ne)))
            for name,kappa,result,mask in cases:
                rows.append(_angle_row(angle,h,kappa,name,result,conforming,mask,audit if name=="B_V11" else None,parent))
            for kappa in kappas:
                rows.append(_angle_row(angle,h,kappa,"C_V12",_solve_vector(mesh,parent.boundary,opening_m,normal,v12,kappa),conforming,v12,audit,parent))
    return rows


def _angle_row(angle,h,kappa,name,result,reference,mask,audit,parent):
    return {"angle_deg":angle,"h_tip_m":h,"kappa":kappa,"representation":name,"parent_geometry_fingerprint":parent.geometry_fingerprint,"parent_connectivity_fingerprint":parent.connectivity_fingerprint,"selected_element_count":int(np.count_nonzero(mask)),"reaction_N_per_m":result.reaction_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.energy_J_per_m,"reaction_reference_error":_relative(result.reaction_N_per_m,reference.reaction_N_per_m),"compliance_reference_error":_relative(result.compliance_m2_per_N,reference.compliance_m2_per_N),"energy_reference_error":_relative(result.energy_J_per_m,reference.energy_J_per_m),"outside_support_stress_l2_error":float(np.linalg.norm(result.sigma_gp[:,~mask.astype(bool)]-reference.sigma_gp[:,~mask.astype(bool)])/max(np.linalg.norm(reference.sigma_gp[:,~mask.astype(bool)]),1e-300)) if name!="D_CONFORMING" else 0.,"free_residual_relative":result.free_residual_relative,"energy_reaction_identity_relative":result.energy_reaction_identity_relative,"conditioning_diagonal_ratio":result.conditioning_diagonal_ratio,"support_width_m":float(audit.maximum_normal_support_width_m) if audit else 0.,"support_footprint_m":float(audit.active_tip_signed_footprint_m) if audit else 0.}


def _row(h,kappa,name,result,reference,mesh,mask,audit,parent):
    outside=~mask; stress_error=float(np.linalg.norm(result.sigma_gp[:,outside]-reference.sigma_gp[:,outside])/max(np.linalg.norm(reference.sigma_gp[:,outside]),1e-300)) if name!="D_CONFORMING" else 0.
    killed=float(np.sum(.5*np.sum(result.strain_gp[:,mask]*(result.sigma_gp[:,mask]),axis=0)*mesh.area_e[mask])/max(result.energy_J_per_m,1e-300)) if np.any(mask) else 0.
    remote=abs(result.reaction_N_per_m)/float(np.ptp(mesh.nodes[:,0])); traction=float(np.sqrt(np.mean(result.sigma_gp[1,mask]**2+result.sigma_gp[2,mask]**2))/max(remote,1e-300)) if np.any(mask) else 0.
    return {"h_tip_m":h,"kappa":kappa,"representation":name,"parent_geometry_fingerprint":parent.geometry_fingerprint,"parent_connectivity_fingerprint":parent.connectivity_fingerprint,"selected_element_count":int(np.count_nonzero(mask)),"support_fingerprint":_hash(np.flatnonzero(mask)),"reaction_N_per_m":result.reaction_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.energy_J_per_m,"crack_opening_displacement_m":result.crack_opening_displacement_m,"crack_opening_reference_error":_relative(result.crack_opening_displacement_m,reference.crack_opening_displacement_m) if np.isfinite(result.crack_opening_displacement_m) else None,"free_residual_relative":result.free_residual_relative,"energy_reaction_identity_relative":result.energy_reaction_identity_relative,"conditioning_diagonal_ratio":result.conditioning_diagonal_ratio,"reaction_reference_error":_relative(result.reaction_N_per_m,reference.reaction_N_per_m),"compliance_reference_error":_relative(result.compliance_m2_per_N,reference.compliance_m2_per_N),"energy_reference_error":_relative(result.energy_J_per_m,reference.energy_J_per_m),"outside_support_stress_l2_error":stress_error,"corridor_traction_relative":traction,"killed_energy_fraction":killed,"support_width_m":float(audit.maximum_normal_support_width_m) if audit else 0.,"support_footprint_m":float(audit.active_tip_signed_footprint_m) if audit else 0.}


__all__=["PrimalResult","run_rotated_cases","run_straight_case"]
