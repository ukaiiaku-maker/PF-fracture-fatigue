"""Matched-parent, analysis-only V11/V12/conforming primal mechanics screen."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import cKDTree

from .causal_sharp_wake_v11 import causal_segment_support, mechanical_fingerprint
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
    cod_fit_residual_m: float=float("nan")
    cod_fit_distances_m: tuple[float,...]=()
    direct_face_cod_m: float=float("nan")
    pin_reaction_relative_error: float=float("nan")
    pin_energy_relative_error: float=float("nan")
    pin_cod_relative_error: float=float("nan")


def _solve(mesh,boundary,opening_m,damage=None,kappa=0.):
    return _solve_normal_opening(mesh,boundary,opening_m,np.array((0.,1.)),damage,kappa,boundary.left_bot)


def _solve_normal_opening(mesh,boundary,opening_m,normal,damage=None,kappa=0.,pin_node=None):
    """Normal platen opening with free tangential motion and one tangent pin."""
    if damage is not None: mesh=replace(mesh,element_damage_gp=np.asarray(damage,float))
    u=np.zeros(mesh.ndof); ep=np.zeros((3,mesh.ne)); rho=np.zeros(mesh.ne); d=np.zeros(mesh.nn)
    K,R,*_=assemble_mechanics(mesh,u,ep,rho,d,D,MAT,kappa=kappa)
    normal=np.asarray(normal,float); normal/=np.linalg.norm(normal); tangent=np.array((normal[1],-normal[0])); basis=np.column_stack((tangent,normal))
    transform=sparse.kron(sparse.eye(mesh.nn,format="csr"),basis,format="csr"); Kq=(transform.T@K@transform).tocsr(); q=np.zeros(mesh.ndof)
    prescribed=np.zeros(mesh.ndof,bool); values=np.zeros(mesh.ndof)
    prescribed[2*boundary.top_nodes+1]=1; values[2*boundary.top_nodes+1]=opening_m/2
    prescribed[2*boundary.bot_nodes+1]=1; values[2*boundary.bot_nodes+1]=-opening_m/2
    pin=boundary.left_bot if pin_node is None else int(pin_node); prescribed[2*pin]=1
    free=~prescribed; q[prescribed]=values[prescribed]; Kff=Kq[np.ix_(free,free)]; q[free]=spsolve(Kff,-Kq[np.ix_(free,prescribed)]@q[prescribed]); u=np.asarray(transform@q)
    residual=K@u; reaction=float(np.sum(residual.reshape(-1,2)[boundary.top_nodes]@normal)); energy=float(.5*u@(K@u))
    strain=np.einsum('eij,ej->ei',mesh.B_e,u[np.c_[2*mesh.elems,2*mesh.elems+1].reshape(mesh.ne,6)]).T
    degradation=np.ones(mesh.ne) if damage is None else (1-np.asarray(damage))**2+kappa
    sigma=(degradation[:,None]*(strain.T@D.T)).T
    residual_q=np.asarray(transform.T@residual); free_res=float(np.linalg.norm(residual_q[free])/max(abs(reaction),1e-300))
    identity=abs(energy-.5*reaction*opening_m)/max(abs(energy),1e-300)
    diag=Kff.diagonal(); cond=float(np.max(diag)/np.min(diag))
    return PrimalResult(u,sigma,strain,reaction,opening_m/abs(reaction),energy,free_res,identity,cond,float("nan"))


def _solve_vector(mesh,boundary,opening_m,normal,damage=None,kappa=0.):
    return _solve_normal_opening(mesh,boundary,opening_m,normal,damage,kappa,boundary.left_bot)


def _relative(a,b): return float(abs(a-b)/max(abs(a),abs(b),1e-300))
def _hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _stress_tensor(voigt): return np.array(((voigt[0],voigt[2]),(voigt[2],voigt[1])))


def _area_weighted_error(mesh,value,reference,region):
    ids=np.flatnonzero(region)
    if not len(ids): return float("nan")
    difference=value[:,ids]-reference[:,ids]; area=mesh.area_e[ids]
    numerator=np.sum(area*np.sum(difference*difference,axis=0)); denominator=np.sum(area*np.sum(reference[:,ids]**2,axis=0))
    return float(np.sqrt(numerator/max(denominator,1e-300)))


def _field_errors(mesh,result,reference,mask,p0,tip):
    cent=mesh.nodes[mesh.elems].mean(axis=1); p0=np.asarray(p0); tip=np.asarray(tip); tangent=(tip-p0)/np.linalg.norm(tip-p0); normal=np.array((-tangent[1],tangent[0])); axial=(cent-p0)@tangent; transverse=np.abs((cent-p0)@normal)
    root_r=np.linalg.norm(cent-p0,axis=1); tip_r=np.linalg.norm(cent-tip,axis=1); physical_exterior=(~mask)&(root_r>=50e-6)&(tip_r>=50e-6)
    regions={"whole_exterior":physical_exterior,"near_tip_annulus":physical_exterior&(tip_r<=150e-6),"face_adjacent_strip":physical_exterior&(transverse<=50e-6)&(axial>=50e-6)&(axial<=np.linalg.norm(tip-p0)-50e-6)}
    return {f"area_weighted_stress_error_{name}":_area_weighted_error(mesh,result.sigma_gp,reference.sigma_gp,region) for name,region in regions.items()}


def _interface_tractions(mesh,result,mask,p0,tip):
    mask=np.asarray(mask,bool); cent=mesh.nodes[mesh.elems].mean(axis=1); tangent=(np.asarray(tip)-np.asarray(p0)); length=float(np.linalg.norm(tangent)); tangent/=length; crack_normal=np.array((-tangent[1],tangent[0])); adjacency={}
    for ei,elem in enumerate(mesh.elems):
        for a,b in ((elem[0],elem[1]),(elem[1],elem[2]),(elem[2],elem[0])): adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ei)
    groups={name:[] for name in ("root","trimmed_interior","active_tip")}
    for (a,b),eids in adjacency.items():
        if len(eids)!=2 or mask[eids[0]]==mask[eids[1]]: continue
        soft=eids[0] if mask[eids[0]] else eids[1]; intact=eids[1] if mask[eids[0]] else eids[0]; edge=mesh.nodes[b]-mesh.nodes[a]; edge_length=float(np.linalg.norm(edge)); interface_normal=np.array((-edge[1],edge[0]))/edge_length
        if (cent[intact]-cent[soft])@interface_normal<0: interface_normal=-interface_normal
        ti=_stress_tensor(result.sigma_gp[:,intact])@interface_normal; ts=_stress_tensor(result.sigma_gp[:,soft])@interface_normal; jump=ti-ts; midpoint=.5*(mesh.nodes[a]+mesh.nodes[b]); axial=float((midpoint-np.asarray(p0))@tangent)
        group="root" if axial<50e-6 else "active_tip" if axial>length-50e-6 else "trimmed_interior"
        groups[group].append((edge_length,ti,ts,jump))
    output={}
    for group,values in groups.items():
        total=sum(v[0] for v in values)
        for label,index in (("intact",1),("soft",2),("jump",3)):
            vectors=np.asarray([v[index] for v in values]) if values else np.empty((0,2)); weights=np.asarray([v[0] for v in values]) if values else np.empty(0)
            local=np.c_[vectors@crack_normal,vectors@tangent] if values else np.empty((0,2))
            output[f"{group}_{label}_traction_rms_Pa"]=float(np.sqrt(np.sum(weights*np.sum(local*local,axis=1))/total)) if total else float("nan")
            output[f"{group}_{label}_normal_force_N_per_m"]=float(np.sum(weights*local[:,0])) if total else float("nan")
            output[f"{group}_{label}_shear_force_N_per_m"]=float(np.sum(weights*local[:,1])) if total else float("nan")
        output[f"{group}_interface_length_m"]=total
    return output


def _mirror_residuals(mesh,result):
    cent=mesh.nodes[mesh.elems].mean(axis=1); distance,pair=cKDTree(cent).query(np.c_[cent[:,0],-cent[:,1]])
    valid=distance<1e-10; area=mesh.area_e[valid]; sigma=result.sigma_gp[:,valid]; reflected=result.sigma_gp[:,pair[valid]]; scale=np.sqrt(np.sum(area*np.sum(sigma*sigma,axis=0)))
    return {"mirror_sigma_xx_relative":float(np.sqrt(np.sum(area*(sigma[0]-reflected[0])**2))/max(scale,1e-300)),"mirror_sigma_yy_relative":float(np.sqrt(np.sum(area*(sigma[1]-reflected[1])**2))/max(scale,1e-300)),"mirror_sigma_xy_antisymmetry_relative":float(np.sqrt(np.sum(area*(sigma[2]+reflected[2])**2))/max(scale,1e-300))}


def _parent_cod(result,mesh,p0,tip,h):
    x=.5*(p0[0]+tip[0]); points=((x,h),(x,-h)); ids=[]
    for point in points: ids.append(int(np.argmin(np.sum((mesh.nodes-np.asarray(point))**2,axis=1))))
    u=result.displacement.reshape(-1,2); return float(u[ids[0],1]-u[ids[1],1])


def _slit_cod(result,slit,p0,tip):
    x=.5*(p0[0]+tip[0]); ids=np.flatnonzero(np.isclose(slit.mesh.nodes[:,0],x)&np.isclose(slit.mesh.nodes[:,1],0.))
    return float(np.ptp(result.displacement.reshape(-1,2)[ids,1]))


def _extrapolated_cod(result,mesh,p0,tip,h,normal=np.array((0.,1.))):
    normal=np.asarray(normal,float); tangent=np.array((normal[1],-normal[0])); midpoint=.5*(np.asarray(p0)+np.asarray(tip)); u=result.displacement.reshape(-1,2)
    distances=h*np.arange(2.,6.); intercepts=[]; residuals=[]
    for sign in (1.,-1.):
        values=[]
        for distance in distances:
            target=midpoint+sign*distance*normal; node=int(np.argmin(np.sum((mesh.nodes-target)**2,axis=1))); values.append(float(u[node]@normal))
        fit=np.polyfit(distances,np.asarray(values),2); intercepts.append(float(fit[-1])); residuals.extend(np.asarray(values)-np.polyval(fit,distances))
    return float(intercepts[0]-intercepts[1]),float(np.sqrt(np.mean(np.asarray(residuals)**2))),tuple(map(float,distances))


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
        direct=_slit_cod(conforming,slit,p0,tip); cod,residual,distances=_extrapolated_cod(conforming,slit.mesh,p0,tip,h)
        conforming=replace(conforming,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,direct_face_cod_m=direct)
        for representation,result,mask in (("A_INTACT",intact,np.zeros(mesh.ne,bool)),("B_V11",_solve(mesh,parent.boundary,opening_m,v11,1e-8),v11.astype(bool)),("D_CONFORMING",conforming,np.zeros(mesh.ne,bool))):
            rows.append(_row(h,None,representation,result,conforming,mesh,mask,audit if representation=="B_V11" else None,parent))
        for kappa in kappas:
            result=_solve(mesh,parent.boundary,opening_m,v12,kappa); cod,residual,distances=_extrapolated_cod(result,mesh,p0,tip,h)
            alternate=_solve_normal_opening(mesh,parent.boundary,opening_m,np.array((0.,1.)),v12,kappa,parent.boundary.right_bot); alternate_cod,_,_=_extrapolated_cod(alternate,mesh,p0,tip,h)
            result=replace(result,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,pin_reaction_relative_error=_relative(result.reaction_N_per_m,alternate.reaction_N_per_m),pin_energy_relative_error=_relative(result.energy_J_per_m,alternate.energy_J_per_m),pin_cod_relative_error=_relative(cod,alternate_cod)); cache[(h,kappa,float(tip[0]))]=(result,v12)
            rows.append(_row(h,kappa,"C_V12",result,conforming,mesh,v12.astype(bool),audit,parent))
    for h in h_values[-2:]:
        for kappa in kappas:
            for delta in (12.5e-6,25e-6,50e-6):
                if not np.isclose(delta/h,round(delta/h)): continue
                values={}; metadata={}
                for sign in (-1,1):
                    moved=tip+np.array((sign*delta,0.)); parent=build_matched_crack_parent(width,height,tuple(p0),tuple(moved),h)
                    ids,audit=mechanically_separating_graph_support(parent.mesh,CrackNetworkState.one_tip((tuple(p0),tuple(moved))))
                    damage=np.isin(np.arange(parent.mesh.ne),ids).astype(float); values[("v12",sign)]=_solve(parent.mesh,parent.boundary,opening_m,damage,kappa)
                    metadata[sign]={"length":float(moved[0]-p0[0]),"graph_fingerprint":audit.graph_fingerprint,"support_fingerprint":audit.support_fingerprint,"selected_area_m2":audit.selected_area_m2,"mechanical_fingerprint":mechanical_fingerprint(parent.mesh,damage),"selected_ids":set(map(int,ids))}
                    slit=conforming_slit_from_parent(parent); values[("conf",sign)]=_solve(slit.mesh,slit.boundary,opening_m)
                for name in ("v12","conf"):
                    minus,plus=values[(name,-1)],values[(name,1)]; ge=-(plus.energy_J_per_m-minus.energy_J_per_m)/(2*delta)
                    dc=(plus.compliance_m2_per_N-minus.compliance_m2_per_N)/(2*delta); cm=.5*(plus.compliance_m2_per_N+minus.compliance_m2_per_N); gc=opening_m**2*dc/(2*cm**2)
                    row={"h_tip_m":h,"kappa":kappa if name=="v12" else None,"representation":name.upper(),"delta_a_m":delta,"G_energy_J_per_m2":ge,"G_compliance_J_per_m2":gc,"energy_compliance_relative_error":_relative(ge,gc)}
                    for sign,label in ((-1,"minus"),(1,"plus")):
                        state=values[(name,sign)]; row.update({f"{label}_physical_crack_length_m":metadata[sign]["length"],f"{label}_reaction_N_per_m":state.reaction_N_per_m,f"{label}_compliance_m2_per_N":state.compliance_m2_per_N,f"{label}_energy_J_per_m":state.energy_J_per_m})
                        if name=="v12": row.update({f"{label}_graph_fingerprint":metadata[sign]["graph_fingerprint"],f"{label}_support_fingerprint":metadata[sign]["support_fingerprint"],f"{label}_selected_area_m2":metadata[sign]["selected_area_m2"],f"{label}_mechanical_fingerprint":metadata[sign]["mechanical_fingerprint"]})
                    if name=="v12": row["mechanically_changed_element_count"]=len(metadata[-1]["selected_ids"]^metadata[1]["selected_ids"])
                    derivatives.append(row)
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
            assert np.min(np.linalg.norm(mesh.nodes-p0,axis=1))<1e-12 and np.min(np.linalg.norm(mesh.nodes-tip,axis=1))<1e-12
            v11_ids,_=causal_segment_support(mesh,p0,tip); v11=np.isin(np.arange(mesh.ne),v11_ids).astype(float)
            v12_ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((tuple(p0),tuple(tip)))); v12=np.isin(np.arange(mesh.ne),v12_ids).astype(float)
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
    extra={}
    if name=="C_V12": extra.update(_field_errors(mesh,result,reference,mask,parent.p0,parent.p1)); extra.update(_interface_tractions(mesh,result,mask,parent.p0,parent.p1)); extra.update(_mirror_residuals(mesh,result)); extra.update({"pin_reaction_relative_error":result.pin_reaction_relative_error,"pin_energy_relative_error":result.pin_energy_relative_error,"pin_cod_relative_error":result.pin_cod_relative_error})
    return {"h_tip_m":h,"kappa":kappa,"representation":name,"parent_geometry_fingerprint":parent.geometry_fingerprint,"parent_connectivity_fingerprint":parent.connectivity_fingerprint,"selected_element_count":int(np.count_nonzero(mask)),"support_fingerprint":_hash(np.flatnonzero(mask)),"reaction_N_per_m":result.reaction_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.energy_J_per_m,"crack_opening_displacement_m":result.crack_opening_displacement_m,"direct_face_cod_m":result.direct_face_cod_m,"cod_fit_residual_m":result.cod_fit_residual_m,"cod_fit_distances_m":" ".join(map(str,result.cod_fit_distances_m)),"cod_extrapolation_order":2,"crack_opening_reference_error":_relative(result.crack_opening_displacement_m,reference.crack_opening_displacement_m) if np.isfinite(result.crack_opening_displacement_m) else None,"conforming_extrapolated_direct_cod_error":_relative(result.crack_opening_displacement_m,result.direct_face_cod_m) if name=="D_CONFORMING" else None,"free_residual_relative":result.free_residual_relative,"energy_reaction_identity_relative":result.energy_reaction_identity_relative,"conditioning_diagonal_ratio":result.conditioning_diagonal_ratio,"reaction_reference_error":_relative(result.reaction_N_per_m,reference.reaction_N_per_m),"compliance_reference_error":_relative(result.compliance_m2_per_N,reference.compliance_m2_per_N),"energy_reference_error":_relative(result.energy_J_per_m,reference.energy_J_per_m),"outside_support_stress_l2_error":stress_error,"KILLED_REGION_STRESS_RMS_DIAGNOSTIC":traction,"killed_energy_fraction":killed,"support_width_m":float(audit.maximum_normal_support_width_m) if audit else 0.,"support_footprint_m":float(audit.active_tip_signed_footprint_m) if audit else 0.,**extra}


__all__=["PrimalResult","run_rotated_cases","run_straight_case"]
