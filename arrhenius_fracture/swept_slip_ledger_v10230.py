"""Incremental plastic distortion from signed dislocation swept distance.

Resident density and accumulated plastic slip are deliberately separate.  For
plane strain per unit out-of-plane thickness, a bin increment is

    dgamma = b * sum_q(s_q * rho_q * dx_q),

where ``s_q`` is the Burgers sign and ``dx_q`` the signed glide distance.  The
line-length/thickness factors cancel; no additional specimen-thickness factor
is present.  Opposite signs moving in opposite directions therefore add.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MODEL_ID="v10.2.30_incremental_signed_swept_slip_ledger_v1"


@dataclass
class SweptSlipLedger:
    plastic_shear: np.ndarray
    cumulative_abs_swept_area_per_volume: np.ndarray
    nucleation_patch_shear: np.ndarray

    @classmethod
    def zeros(cls,n_systems:int,n_bins:int):
        shape=(int(n_systems),int(n_bins))
        return cls(np.zeros(shape),np.zeros(shape),np.zeros(shape))

    def advance_motion(self, resident_density_m2, signed_distance_m, burgers_m:float, burgers_signs=(-1.,1.)) -> np.ndarray:
        rho=np.asarray(resident_density_m2,float); distance=np.asarray(signed_distance_m,float)
        if rho.shape!=distance.shape or rho.ndim!=3 or rho.shape[1]!=len(burgers_signs):
            raise ValueError("resident density and signed distance must have shape (systems,signs,bins)")
        signs=np.asarray(burgers_signs,float)[None,:,None]
        increment=float(burgers_m)*np.sum(signs*rho*distance,axis=1)
        self.plastic_shear += increment
        self.cumulative_abs_swept_area_per_volume += abs(float(burgers_m))*np.sum(rho*np.abs(distance),axis=1)
        return increment

    def advance_velocity(self,resident_density_m2,signed_velocity_m_s,dt_s:float,burgers_m:float,burgers_signs=(-1.,1.)) -> np.ndarray:
        return self.advance_motion(resident_density_m2,np.asarray(signed_velocity_m_s)*float(dt_s),burgers_m,burgers_signs)

    def advance_orowan_rate(self,mobile_density_m2,base_velocity_m_s,dt_s:float,burgers_m:float) -> np.ndarray:
        """Advance the native two-sign state using its transport convention.

        The immutable fatigue state stores two Burgers-sign populations but one
        signed base velocity per system/bin. Its spatial transport velocity is
        ``sign_b * base_velocity``. Both populations therefore add to the
        Orowan rate ``b * (rho_minus + rho_plus) * base_velocity``.
        """
        rho=np.asarray(mobile_density_m2,float); velocity=np.asarray(base_velocity_m_s,float)
        if rho.ndim!=3 or rho.shape[1]!=2 or velocity.shape!=(rho.shape[0],rho.shape[2]):
            raise ValueError("mobile density must be (systems,2,bins) and velocity (systems,bins)")
        increment=float(burgers_m)*np.sum(np.maximum(rho,0.),axis=1)*velocity*float(dt_s)
        self.plastic_shear += increment
        self.cumulative_abs_swept_area_per_volume += abs(float(burgers_m))*np.sum(np.maximum(rho,0.),axis=1)*np.abs(velocity)*float(dt_s)
        return increment

    def nucleate(self,system:int,burgers_sign:int,count:float,burgers_m:float,initial_swept_area_m2:float|None,representative_volume_m3_per_unit_thickness:float) -> float:
        """Book a finite inserted patch once; point/line nucleation books zero."""
        if initial_swept_area_m2 is None or float(initial_swept_area_m2)==0.:
            return 0.
        volume=float(representative_volume_m3_per_unit_thickness)
        if volume<=0.: raise ValueError("representative volume must be positive")
        increment=float(burgers_sign)*float(count)*float(burgers_m)*float(initial_swept_area_m2)/volume
        self.plastic_shear[int(system),0] += increment
        self.nucleation_patch_shear[int(system),0] += increment
        return increment

    def beta_p(self,directions,normals) -> np.ndarray:
        d=np.asarray(directions,float); n=np.asarray(normals,float)
        if d.shape!=n.shape or d.shape!=(self.plastic_shear.shape[0],2): raise ValueError("orientation shape mismatch")
        d=d/np.linalg.norm(d,axis=1)[:,None]; n=n/np.linalg.norm(n,axis=1)[:,None]
        return np.einsum('sb,si,sj->bij',self.plastic_shear,d,n)

    def engineering_eigenstrain(self,directions,normals) -> np.ndarray:
        beta=self.beta_p(directions,normals)
        return np.stack((beta[:,0,0],beta[:,1,1],beta[:,0,1]+beta[:,1,0]),axis=0)

    def audit(self):
        return {"schema":MODEL_ID,"resident_line_content_used_as_accumulated_slip":False,
          "motion_identity":"dgamma=b*sum(sign_b*rho*dx_signed)","plane_strain_thickness_factor":1.0,
          "nucleation_without_specified_patch_area_adds_slip":False,"plastic_distortion_is_irreversible_under_exit_or_annihilation":True,
          "plastic_shear_sum":float(np.sum(self.plastic_shear)),"absolute_swept_measure_sum":float(np.sum(self.cumulative_abs_swept_area_per_volume))}


def fixed_width_slip_patch(mesh,*,start_xy_m,end_xy_m,direction,normal,slip_displacement_m:float,physical_band_width_m:float):
    """Regularized displacement discontinuity with fixed physical support."""
    start=np.asarray(start_xy_m,float); end=np.asarray(end_xy_m,float); tangent=end-start; length=float(np.linalg.norm(tangent)); tangent/=length
    cent=np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(1); rel=cent-start; longitudinal=rel@tangent; closest=start+longitudinal[:,None]*tangent
    selected=(longitudinal>=0)&(longitudinal<=length)&(np.linalg.norm(cent-closest,axis=1)<=.5*float(physical_band_width_m))
    if not np.any(selected): raise ValueError("fixed physical slip patch selects no elements")
    d=np.asarray(direction,float); d/=np.linalg.norm(d); n=np.asarray(normal,float); n/=np.linalg.norm(n)
    gamma=float(slip_displacement_m)/float(physical_band_width_m)
    voigt=gamma*np.array([d[0]*n[0],d[1]*n[1],d[0]*n[1]+d[1]*n[0]])
    ep=np.zeros((3,mesh.ne)); ep[:,selected]=voigt[:,None]
    return ep,{"slip_displacement_m":float(slip_displacement_m),"physical_band_width_m":float(physical_band_width_m),
      "plastic_shear":gamma,"selected_elements":int(np.sum(selected)),"represented_area_m2":float(np.sum(mesh.area_e[selected])),
      "numerical_band_tied_to_mesh":False}

__all__=["MODEL_ID","SweptSlipLedger","fixed_width_slip_patch"]
