"""Matched-parent, analysis-only V11/V12/conforming primal mechanics screen."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import cKDTree

from .causal_sharp_wake_v11 import causal_segment_support, mechanical_fingerprint
from .config import ElasticProperties
from .conforming_crack_oracle_v12 import MatchedCrackParent, build_matched_crack_parent, conforming_slit_from_parent
from .crack_network_v11 import CrackNetworkState
from .fem import assemble_mechanics, plane_strain_D
from .mechanically_separating_sharp_wake_v12 import mechanically_separating_graph_support
from .mesh import BoundaryData, rebuild_tri_mesh
from .anisotropic_emission_v10174 import AnisotropicEmissionConfig, probe_tensor_ahead

MAT=ElasticProperties(E=210e9,nu=.3); D=plane_strain_D(MAT)


def recover_element_fields(mesh, displacement, constitutive=D, degradation=None):
    """Recover CST engineering strain and stress from interleaved nodal DOFs."""
    displacement=np.asarray(displacement,float)
    edofs=np.empty((mesh.ne,6),int)
    edofs[:,0::2]=2*mesh.elems
    edofs[:,1::2]=2*mesh.elems+1
    strain=np.einsum("eij,ej->ei",mesh.B_e,displacement[edofs]).T
    scale=np.ones(mesh.ne) if degradation is None else np.asarray(degradation,float)
    sigma=(scale[:,None]*(strain.T@np.asarray(constitutive,float).T)).T
    return strain,sigma


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
    fixed_distance_cod_m: float=float("nan")
    cod_samples_json: str=""


def _solve(mesh,boundary,opening_m,damage=None,kappa=0.):
    return _solve_normal_opening(mesh,boundary,opening_m,np.array((0.,1.)),damage,kappa,_symmetry_pin(mesh,np.array((0.,1.)),"min"))


def _symmetry_pin(mesh,normal,side):
    normal=np.asarray(normal,float); tangent=np.array((normal[1],-normal[0])); axial=mesh.nodes@tangent; target=np.min(axial) if side=="min" else np.max(axial); edge=np.flatnonzero(np.isclose(axial,target,atol=1e-12)); normal_coordinate=mesh.nodes@normal
    return int(edge[np.argmin(np.abs(normal_coordinate[edge]-np.median(normal_coordinate)))])


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
    degradation=np.ones(mesh.ne) if damage is None else (1-np.asarray(damage))**2+kappa
    strain,sigma=recover_element_fields(mesh,u,D,degradation)
    residual_q=np.asarray(transform.T@residual); free_res=float(np.linalg.norm(residual_q[free])/max(abs(reaction),1e-300))
    identity=abs(energy-.5*reaction*opening_m)/max(abs(energy),1e-300)
    diag=Kff.diagonal(); cond=float(np.max(diag)/np.min(diag))
    return PrimalResult(u,sigma,strain,reaction,opening_m/abs(reaction),energy,free_res,identity,cond,float("nan"))


def _solve_vector(mesh,boundary,opening_m,normal,damage=None,kappa=0.):
    return _solve_normal_opening(mesh,boundary,opening_m,normal,damage,kappa,_symmetry_pin(mesh,normal,"min"))


def _relative(a,b): return float(abs(a-b)/max(abs(a),abs(b),1e-300))
def _hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def _json(value): return json.dumps(value,sort_keys=True,separators=(",",":"),default=lambda item:np.asarray(item).tolist())


def _stress_tensor(voigt): return np.array(((voigt[0],voigt[2]),(voigt[2],voigt[1])))


def _area_weighted_error(mesh,value,reference,region):
    ids=np.flatnonzero(region)
    if not len(ids): return float("nan")
    difference=value[:,ids]-reference[:,ids]; area=mesh.area_e[ids]
    numerator=np.sum(area*np.sum(difference*difference,axis=0)); denominator=np.sum(area*np.sum(reference[:,ids]**2,axis=0))
    return float(np.sqrt(numerator/max(denominator,1e-300)))


def _field_errors(mesh,result,reference,mask,p0,tip):
    cent=mesh.nodes[mesh.elems].mean(axis=1); p0=np.asarray(p0); tip=np.asarray(tip); tangent=(tip-p0)/np.linalg.norm(tip-p0); normal=np.array((-tangent[1],tangent[0])); axial=(cent-p0)@tangent; transverse=np.abs((cent-p0)@normal)
    root_r=np.linalg.norm(cent-p0,axis=1); tip_r=np.linalg.norm(cent-tip,axis=1); graph_length=np.linalg.norm(tip-p0); fixed_tube=(axial>=0)&(axial<=graph_length)&(transverse<50e-6); physical_exterior=(root_r>=50e-6)&(tip_r>=50e-6)&(~fixed_tube)
    support_distance=np.maximum(0.,transverse-.5*np.ptp(cent[mask]@normal) if np.any(mask) else transverse)
    regions={"whole_exterior":physical_exterior,"near_tip_annulus":physical_exterior&(tip_r<=150e-6),"face_adjacent_strip":physical_exterior&(transverse>=50e-6)&(transverse<=100e-6)&(axial>=50e-6)&(axial<=graph_length-50e-6),"h_scaled_regularization_layer":(~mask)&(support_distance>0)&(support_distance<=4*mesh.hbar_tip)&(root_r>=50e-6)&(tip_r>=50e-6)}
    output={}; remote_stress=abs(result.reaction_N_per_m)/max(float(np.ptp(mesh.nodes[:,0])),1e-300)
    local=np.array(((normal[0]**2,normal[1]**2,2*normal[0]*normal[1]),(tangent[0]**2,tangent[1]**2,2*tangent[0]*tangent[1]),(normal[0]*tangent[0],normal[1]*tangent[1],normal[0]*tangent[1]+normal[1]*tangent[0])))
    for name,region in regions.items():
        ids=np.flatnonzero(region); area=mesh.area_e[ids]; difference=result.sigma_gp[:,ids]-reference.sigma_gp[:,ids]; ref=reference.sigma_gp[:,ids]
        diff2=float(np.sum(area*np.sum(difference*difference,axis=0))); ref2=float(np.sum(area*np.sum(ref*ref,axis=0))); val2=float(np.sum(area*np.sum(result.sigma_gp[:,ids]**2,axis=0))); region_area=float(np.sum(area))
        output[f"area_weighted_stress_error_{name}"]=float(np.sqrt(diff2/max(ref2,1e-300))); output[f"area_weighted_stress_region_{name}_element_count"]=len(ids); output[f"area_weighted_stress_region_{name}_area_m2"]=region_area
        output[f"stress_norm_v12_{name}_Pa_sqrt_m2"]=float(np.sqrt(val2)); output[f"stress_norm_conforming_{name}_Pa_sqrt_m2"]=float(np.sqrt(ref2)); output[f"stress_difference_norm_{name}_Pa_sqrt_m2"]=float(np.sqrt(diff2)); output[f"stress_difference_remote_normalized_{name}"]=float(np.sqrt(diff2)/max(remote_stress*np.sqrt(region_area),1e-300))
        local_difference=local@difference; local_reference=local@ref
        for component,index in (("nn",0),("tt",1),("nt",2)):
            num=float(np.sum(area*local_difference[index]**2)); den=float(np.sum(area*local_reference[index]**2)); output[f"stress_component_{component}_error_{name}"]=float(np.sqrt(num/max(den,1e-300))); output[f"stress_component_{component}_reference_norm_{name}_Pa_sqrt_m2"]=float(np.sqrt(den))
        strain_difference=result.strain_gp[:,ids]-reference.strain_gp[:,ids]; num_energy=float(np.sum(area*np.einsum("ie,ij,je->e",strain_difference,D,strain_difference))); ref_energy=float(np.sum(area*np.einsum("ie,ij,je->e",reference.strain_gp[:,ids],D,reference.strain_gp[:,ids]))); output[f"elastic_energy_norm_error_{name}"]=float(np.sqrt(num_energy/max(ref_energy,1e-300))); output[f"elastic_energy_reference_denominator_{name}_J_per_m"]=ref_energy
    profiles=[]
    for distance in (25e-6,50e-6,75e-6,100e-6,150e-6):
        for station_fraction in (.2,.4,.6,.8):
            target_axial=station_fraction*graph_length; band=(np.abs(transverse-distance)<=max(mesh.hbar_tip,.15*distance))&(np.abs(axial-target_axial)<=max(mesh.hbar_tip,25e-6))
            if np.any(band): profiles.append({"normal_distance_m":distance,"axial_station_fraction":station_fraction,"element_count":int(np.count_nonzero(band)),"relative_stress_error":_area_weighted_error(mesh,result.sigma_gp,reference.sigma_gp,band)})
    output["distance_resolved_profiles_json"]=_json(profiles)
    probe_config=AnisotropicEmissionConfig(probe_radius_m=50e-6,sector_half_angle_deg=25.,damage_cutoff=.85,min_elements=3)
    vprobe=probe_tensor_ahead(mesh,result.sigma_gp,mask.astype(float),tip,tangent,probe_config); cprobe=probe_tensor_ahead(mesh,reference.sigma_gp,np.zeros(mesh.ne),tip,tangent,probe_config)
    output["production_tensor_probe_v12_json"]=_json(vprobe); output["production_tensor_probe_conforming_json"]=_json(cprobe)
    if vprobe.get("reliable") and cprobe.get("reliable"):
        output["production_tensor_probe_relative_error"]=float(np.linalg.norm(np.asarray(vprobe["tensor"])-np.asarray(cprobe["tensor"]))/max(np.linalg.norm(cprobe["tensor"]),1e-300))
    return output


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


def _discrete_corridor_transfer(mesh,result,mask,p0,tip):
    """Independent nodal-force balance on the trimmed soft-corridor boundary."""
    mask=np.asarray(mask,bool); p0=np.asarray(p0); tip=np.asarray(tip); tangent=(tip-p0)/np.linalg.norm(tip-p0); normal=np.array((-tangent[1],tangent[0])); cent=mesh.nodes[mesh.elems].mean(axis=1); axial=(cent-p0)@tangent
    selected=mask&(axial>=50e-6)&(axial<=np.linalg.norm(tip-p0)-50e-6); nodal=np.zeros((mesh.nn,2))
    for ei in np.flatnonzero(selected):
        force=mesh.area_e[ei]*(mesh.B_e[ei].T@result.sigma_gp[:,ei])
        for local,node in enumerate(mesh.elems[ei]): nodal[node]+=force[2*local:2*local+2]
    interface=set()
    adjacency={}
    for ei,elem in enumerate(mesh.elems):
        for a,b in ((elem[0],elem[1]),(elem[1],elem[2]),(elem[2],elem[0])): adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ei)
    for edge,eids in adjacency.items():
        if len(eids)==2 and mask[eids[0]]!=mask[eids[1]]:
            midpoint=mesh.nodes[list(edge)].mean(axis=0); x=float((midpoint-p0)@tangent)
            if 50e-6<=x<=np.linalg.norm(tip-p0)-50e-6: interface.update(edge)
    upper=np.array([node for node in interface if (mesh.nodes[node]-p0)@normal>=0],int); lower=np.array([node for node in interface if (mesh.nodes[node]-p0)@normal<0],int)
    fu=nodal[upper].sum(axis=0) if len(upper) else np.zeros(2); fl=nodal[lower].sum(axis=0) if len(lower) else np.zeros(2)
    return {"discrete_transmitted_normal_force_N_per_m":float(abs(fu@normal)+abs(fl@normal)),"discrete_transmitted_shear_force_N_per_m":float(abs(fu@tangent)+abs(fl@tangent)),"discrete_interface_node_count":len(interface)}


def _mirror_residuals(mesh,result,tip,pin):
    # Area-average element stresses at unique physical coordinates. Duplicate
    # conforming-face nodes are intentionally combined before reflection.
    sums={}; weights={}
    for ei,elem in enumerate(mesh.elems):
        for node in elem:
            key=tuple(np.round(mesh.nodes[node],15)); sums[key]=sums.get(key,np.zeros(3))+mesh.area_e[ei]*result.sigma_gp[:,ei]; weights[key]=weights.get(key,0.)+mesh.area_e[ei]
    keys=sorted(sums); coords=np.asarray(keys); values=np.asarray([sums[k]/weights[k] for k in keys]).T; area=np.asarray([weights[k] for k in keys]); lookup={k:i for i,k in enumerate(keys)}; pairs=np.asarray([lookup.get(tuple(np.round((x,-y),15)),-1) for x,y in coords]); paired=pairs>=0
    distance_pin=np.linalg.norm(coords-mesh.nodes[pin],axis=1); distance_tip=np.linalg.norm(coords-np.asarray(tip),axis=1)
    regions={"full":paired,"constraint_excluded":paired&(distance_pin>=50e-6),"near_tip":paired&(distance_tip<=150e-6)&(distance_tip>=50e-6)}; output={}
    for region,mask in regions.items():
        ids=np.flatnonzero(mask); reflected=values[:,pairs[ids]]; sigma=values[:,ids]; w=area[ids]; scale=np.sqrt(np.sum(w*np.sum(sigma*sigma,axis=0)))
        output[f"mirror_{region}_sigma_xx_relative"]=float(np.sqrt(np.sum(w*(sigma[0]-reflected[0])**2))/max(scale,1e-300)); output[f"mirror_{region}_sigma_yy_relative"]=float(np.sqrt(np.sum(w*(sigma[1]-reflected[1])**2))/max(scale,1e-300)); output[f"mirror_{region}_sigma_xy_antisymmetry_relative"]=float(np.sqrt(np.sum(w*(sigma[2]+reflected[2])**2))/max(scale,1e-300)); output[f"mirror_{region}_sample_count"]=len(ids)
    return output


def _parent_cod(result,mesh,p0,tip,h):
    x=.5*(p0[0]+tip[0]); points=((x,h),(x,-h)); ids=[]
    for point in points: ids.append(int(np.argmin(np.sum((mesh.nodes-np.asarray(point))**2,axis=1))))
    u=result.displacement.reshape(-1,2); return float(u[ids[0],1]-u[ids[1],1])


def _slit_cod(result,slit,p0,tip):
    x=.5*(p0[0]+tip[0]); ids=np.flatnonzero(np.isclose(slit.mesh.nodes[:,0],x)&np.isclose(slit.mesh.nodes[:,1],0.))
    return float(np.ptp(result.displacement.reshape(-1,2)[ids,1]))


def _extrapolated_cod(result,mesh,p0,tip,h,normal=np.array((0.,1.)),support_width=0.,distances=None):
    normal=np.asarray(normal,float); tangent=np.array((normal[1],-normal[0])); midpoint=.5*(np.asarray(p0)+np.asarray(tip)); u=result.displacement.reshape(-1,2)
    distances=np.asarray(distances if distances is not None else support_width+h*np.arange(1.,5.),float); intercepts=[]; residuals=[]; samples=[]
    for sign in (1.,-1.):
        values=[]
        for distance in distances:
            target=midpoint+sign*distance*normal; node=int(np.argmin(np.sum((mesh.nodes-target)**2,axis=1))); values.append(float(u[node]@normal))
            samples.append({"side":"upper" if sign>0 else "lower","x_m":float(mesh.nodes[node,0]),"y_m":float(mesh.nodes[node,1]),"signed_graph_distance_m":float(sign*distance),"support_clearance_m":float(distance-support_width),"normal_displacement_m":values[-1]})
        fit=np.polyfit(distances,np.asarray(values),2); intercepts.append(float(fit[-1])); residuals.extend(np.asarray(values)-np.polyval(fit,distances))
    residual=float(np.sqrt(np.mean(np.asarray(residuals)**2))); details={"polynomial_order":2,"distances_m":list(map(float,distances)),"fit_residual_m":residual,"upper_extrapolated_m":intercepts[0],"lower_extrapolated_m":intercepts[1],"samples":samples}
    return float(intercepts[0]-intercepts[1]),residual,tuple(map(float,distances)),details


def run_straight_case(h_values=(25e-6,12.5e-6,6.25e-6,3.125e-6),kappas=(1e-4,1e-6,1e-8),opening_m=8e-7):
    width=height=8e-4; p0=np.array((2e-4,0.)); tip=np.array((5e-4,0.)); delta=25e-6
    rows=[]; derivatives=[]
    cache={}
    for h in h_values:
        parent=build_matched_crack_parent(width,height,tuple(p0),tuple(tip),h); mesh=parent.mesh
        v11_ids,_=causal_segment_support(mesh,p0,tip); v11=np.zeros(mesh.ne); v11[v11_ids]=1
        v12_ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((tuple(p0),tuple(tip))))
        v12=np.zeros(mesh.ne); v12[v12_ids]=1; slit=conforming_slit_from_parent(parent)
        intact=_solve(mesh,parent.boundary,opening_m); conforming=_solve(slit.mesh,slit.boundary,opening_m)
        direct=_slit_cod(conforming,slit,p0,tip); width_support=audit.maximum_normal_support_width_m; cod,residual,distances,details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=width_support); fixed,_,_,fixed_details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=width_support,distances=(50e-6,62.5e-6,75e-6,87.5e-6))
        conforming=replace(conforming,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,direct_face_cod_m=direct,fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details}))
        for representation,result,mask in (("A_INTACT",intact,np.zeros(mesh.ne,bool)),("B_V11",_solve(mesh,parent.boundary,opening_m,v11,1e-8),v11.astype(bool)),("D_CONFORMING",conforming,np.zeros(mesh.ne,bool))):
            rows.append(_row(h,None,representation,result,conforming,mesh,mask,audit if representation=="B_V11" else None,parent))
        for kappa in kappas:
            result=_solve(mesh,parent.boundary,opening_m,v12,kappa); cod,residual,distances,details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); fixed,_,_,fixed_details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m,distances=(50e-6,62.5e-6,75e-6,87.5e-6))
            alternate=_solve_normal_opening(mesh,parent.boundary,opening_m,np.array((0.,1.)),v12,kappa,_symmetry_pin(mesh,np.array((0.,1.)),"max")); alternate_cod,_,_,_=_extrapolated_cod(alternate,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m)
            result=replace(result,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,pin_reaction_relative_error=_relative(result.reaction_N_per_m,alternate.reaction_N_per_m),pin_energy_relative_error=_relative(result.energy_J_per_m,alternate.energy_J_per_m),pin_cod_relative_error=_relative(cod,alternate_cod),fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details})); cache[(h,kappa,float(tip[0]))]=(result,v12)
            rows.append(_row(h,kappa,"C_V12",result,conforming,mesh,v12.astype(bool),audit,parent))
    # The frozen G campaign remains on its predeclared 12.5/6.25 um pair; the
    # unified straight matrix nevertheless carries all non-G diagnostics at
    # every level through 3.125 um.
    derivative_h_values=tuple(value for value in h_values if value in (12.5e-6,6.25e-6)) if len(h_values)>1 else tuple(h_values)
    for h in derivative_h_values:
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


def run_low_kappa_prescreen(h=3.125e-6,opening_m=8e-7,kappa=1e-8):
    width=height=8e-4; p0=np.array((2e-4,0.)); tip=np.array((5e-4,0.)); parent=build_matched_crack_parent(width,height,tuple(p0),tuple(tip),h); mesh=parent.mesh
    ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((tuple(p0),tuple(tip)))); damage=np.isin(np.arange(mesh.ne),ids).astype(float); slit=conforming_slit_from_parent(parent)
    conforming=_solve(slit.mesh,slit.boundary,opening_m); direct=_slit_cod(conforming,slit,p0,tip); cod,residual,distances,details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); fixed,_,_,fixed_details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m,distances=(50e-6,62.5e-6,75e-6,87.5e-6)); conforming=replace(conforming,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,direct_face_cod_m=direct,fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details}))
    result=_solve(mesh,parent.boundary,opening_m,damage,kappa); cod,residual,distances,details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); fixed,_,_,fixed_details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m,distances=(50e-6,62.5e-6,75e-6,87.5e-6)); alternate=_solve_normal_opening(mesh,parent.boundary,opening_m,np.array((0.,1.)),damage,kappa,_symmetry_pin(mesh,np.array((0.,1.)),"max")); alternate_cod,_,_,_=_extrapolated_cod(alternate,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); result=replace(result,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details}),pin_reaction_relative_error=_relative(result.reaction_N_per_m,alternate.reaction_N_per_m),pin_energy_relative_error=_relative(result.energy_J_per_m,alternate.energy_J_per_m),pin_cod_relative_error=_relative(cod,alternate_cod))
    return [_row(h,None,"D_CONFORMING",conforming,conforming,mesh,np.zeros(mesh.ne,bool),None,parent),_row(h,kappa,"C_V12",result,conforming,mesh,damage.astype(bool),audit,parent)]


def _locally_refined_parent(h=1.5625e-6,far_h=12.5e-6):
    """Shared deterministic graded parent with a bounded fine intersection."""
    width=height=8e-4; p0=np.array((2e-4,0.)); tip=np.array((5e-4,0.)); x0,x1=1.5e-4,5.5e-4; y0,y1=-1.5e-4,1.5e-4
    coarse_x=np.arange(0.,width+.5*far_h,far_h); coarse_y=np.arange(-height/2,height/2+.5*far_h,far_h); fine_x=np.arange(x0,x1+.5*h,h); fine_y=np.arange(y0,y1+.5*h,h)
    xs=np.unique(np.round(np.r_[coarse_x[(coarse_x<x0)|(coarse_x>x1)],fine_x,p0[0],tip[0]],15)); ys=np.unique(np.round(np.r_[coarse_y[(coarse_y<y0)|(coarse_y>y1)],fine_y,0.],15)); gx,gy=np.meshgrid(xs,ys); nodes=np.c_[gx.ravel(),gy.ravel()]; nx=len(xs)-1; ny=len(ys)-1
    def node(i,j): return j*(nx+1)+i
    elems=[]
    for j in range(ny):
        for i in range(nx):
            a,b,c,d=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1); elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    elems=np.asarray(elems,int)
    mesh=rebuild_tri_mesh(nodes,elems,tip_centers=tip); top=np.flatnonzero(np.isclose(nodes[:,1],height/2)); bot=np.flatnonzero(np.isclose(nodes[:,1],-height/2)); left=int(np.argmin(np.sum((nodes-np.array((0.,-height/2)))**2,axis=1))); right=int(np.argmin(np.sum((nodes-np.array((width,-height/2)))**2,axis=1))); boundary=BoundaryData(top,bot,left,right,np.empty(0,int))
    return MatchedCrackParent(mesh,boundary,tuple(p0),tuple(tip),h,_hash(nodes),_hash(elems))


def run_targeted_local_refinement(h=1.5625e-6,opening_m=8e-7,kappa=1e-8):
    parent=_locally_refined_parent(h); mesh=parent.mesh; p0=np.asarray(parent.p0); tip=np.asarray(parent.p1); ids,audit=mechanically_separating_graph_support(mesh,CrackNetworkState.one_tip((parent.p0,parent.p1))); damage=np.isin(np.arange(mesh.ne),ids).astype(float); slit=conforming_slit_from_parent(parent)
    conforming=_solve(slit.mesh,slit.boundary,opening_m); direct=_slit_cod(conforming,slit,p0,tip); cod,residual,distances,details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); fixed,_,_,fixed_details=_extrapolated_cod(conforming,slit.mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m,distances=(50e-6,62.5e-6,75e-6,87.5e-6)); conforming=replace(conforming,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,direct_face_cod_m=direct,fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details}))
    result=_solve(mesh,parent.boundary,opening_m,damage,kappa); cod,residual,distances,details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); fixed,_,_,fixed_details=_extrapolated_cod(result,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m,distances=(50e-6,62.5e-6,75e-6,87.5e-6)); alternate=_solve_normal_opening(mesh,parent.boundary,opening_m,np.array((0.,1.)),damage,kappa,_symmetry_pin(mesh,np.array((0.,1.)),"max")); alternate_cod,_,_,_=_extrapolated_cod(alternate,mesh,p0,tip,h,support_width=audit.maximum_normal_support_width_m); result=replace(result,crack_opening_displacement_m=cod,cod_fit_residual_m=residual,cod_fit_distances_m=distances,fixed_distance_cod_m=fixed,cod_samples_json=_json({"h_scaled":details,"fixed_physical":fixed_details}),pin_reaction_relative_error=_relative(result.reaction_N_per_m,alternate.reaction_N_per_m),pin_energy_relative_error=_relative(result.energy_J_per_m,alternate.energy_J_per_m),pin_cod_relative_error=_relative(cod,alternate_cod))
    return [_row(h,None,"D_CONFORMING",conforming,conforming,mesh,np.zeros(mesh.ne,bool),None,parent),_row(h,kappa,"C_V12",result,conforming,mesh,damage.astype(bool),audit,parent)]


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
    extra.update(_mirror_residuals(mesh if name!="D_CONFORMING" else mesh,result,parent.p1,_symmetry_pin(mesh,np.array((0.,1.)),"min")))
    if name=="C_V12": extra.update(_field_errors(mesh,result,reference,mask,parent.p0,parent.p1)); extra.update(_interface_tractions(mesh,result,mask,parent.p0,parent.p1)); extra.update(_discrete_corridor_transfer(mesh,result,mask,parent.p0,parent.p1)); extra.update({"pin_reaction_relative_error":result.pin_reaction_relative_error,"pin_energy_relative_error":result.pin_energy_relative_error,"pin_cod_relative_error":result.pin_cod_relative_error})
    if name=="C_V12":
        remote_stress=abs(result.reaction_N_per_m)/max(float(np.ptp(mesh.nodes[:,0])),1e-300); remote_force=abs(result.reaction_N_per_m)
        for key,value in tuple(extra.items()):
            if "soft_traction_rms_Pa" in key: extra[key.replace("_Pa","_relative_remote_stress")]=value/max(remote_stress,1e-300)
            elif "soft_normal_force_N_per_m" in key or "soft_shear_force_N_per_m" in key: extra[key.replace("_N_per_m","_relative_remote_resultant")]=value/max(remote_force,1e-300)
            elif key.startswith("discrete_transmitted_") and key.endswith("_N_per_m"): extra[key.replace("_N_per_m","_relative_remote_resultant")]=value/max(remote_force,1e-300)
        extra["effective_interface_stiffness_N_per_m2"]=extra["discrete_transmitted_normal_force_N_per_m"]/max(abs(result.crack_opening_displacement_m),1e-300)
    return {"h_tip_m":h,"kappa":kappa,"representation":name,"parent_geometry_fingerprint":parent.geometry_fingerprint,"parent_connectivity_fingerprint":parent.connectivity_fingerprint,"selected_element_count":int(np.count_nonzero(mask)),"support_fingerprint":_hash(np.flatnonzero(mask)),"reaction_N_per_m":result.reaction_N_per_m,"compliance_m2_per_N":result.compliance_m2_per_N,"energy_J_per_m":result.energy_J_per_m,"crack_opening_displacement_m":result.crack_opening_displacement_m,"fixed_distance_cod_m":result.fixed_distance_cod_m,"fixed_distance_cod_reference_error":_relative(result.fixed_distance_cod_m,reference.fixed_distance_cod_m) if np.isfinite(result.fixed_distance_cod_m) else None,"cod_samples":result.cod_samples_json,"direct_face_cod_m":result.direct_face_cod_m,"cod_fit_residual_m":result.cod_fit_residual_m,"cod_fit_distances_m":" ".join(map(str,result.cod_fit_distances_m)),"cod_extrapolation_order":2,"crack_opening_reference_error":_relative(result.crack_opening_displacement_m,reference.crack_opening_displacement_m) if np.isfinite(result.crack_opening_displacement_m) else None,"conforming_extrapolated_direct_cod_error":_relative(result.crack_opening_displacement_m,result.direct_face_cod_m) if name=="D_CONFORMING" else None,"free_residual_relative":result.free_residual_relative,"energy_reaction_identity_relative":result.energy_reaction_identity_relative,"conditioning_diagonal_ratio":result.conditioning_diagonal_ratio,"reaction_reference_error":_relative(result.reaction_N_per_m,reference.reaction_N_per_m),"compliance_reference_error":_relative(result.compliance_m2_per_N,reference.compliance_m2_per_N),"energy_reference_error":_relative(result.energy_J_per_m,reference.energy_J_per_m),"outside_support_stress_l2_error":stress_error,"KILLED_REGION_STRESS_RMS_DIAGNOSTIC":traction,"killed_energy_fraction":killed,"support_width_m":float(audit.maximum_normal_support_width_m) if audit else 0.,"support_footprint_m":float(audit.active_tip_signed_footprint_m) if audit else 0.,**extra}


__all__=["PrimalResult","recover_element_fields","run_low_kappa_prescreen","run_rotated_cases","run_straight_case","run_targeted_local_refinement"]
