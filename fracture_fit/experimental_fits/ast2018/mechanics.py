"""Scalar cantilever mechanics and the published Ast observation operators.

All dimensional inputs are SI. This module is deliberately independent of the
native kinetic implementation so both reduced and spatial histories use exactly
the same experimental observation calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Iterable

import numpy as np

from .cantilever_loading import BendingOpeningShape


_GAUSS4_NODES, _GAUSS4_WEIGHTS = np.polynomial.legendre.leggauss(4)


def unit_force_resolved_stress(specimen:"Specimen",a_m:float,radius_m:float,
                               x_m,drive_factors):
    """Resolved external stress per newton, including optional bending shape."""
    x=np.asarray(x_m,dtype=float)
    factors=np.asarray(drive_factors,dtype=float)
    shape=np.array([specimen.opening_shape(a_m,radius_m+xi) for xi in x])
    return np.abs(factors)[...,None]*shape*specimen.geometry_factor(a_m)/np.sqrt(
        2*pi*np.maximum(radius_m+x,1e-30))


def birth_plastic_displacement(activation_increments,conversion,physics,
                                specimen:"Specimen",a_m:float,radius_m:float,
                                drive_factors)->float:
    """Virtual work of forming finite tip-radius dislocation nuclei.

    One emitted line quantum spans the specimen front and initially sweeps the
    current tip radius on its slip plane. Thus b*B*r is its plastic shear
    volume. This uses the native line-content conversion once; translation
    never changes the cumulative activation ledger and creates no birth work.
    """
    counts=np.asarray(activation_increments,dtype=float)
    conversion=np.asarray(conversion,dtype=float)
    factors=np.asarray(drive_factors,dtype=float)
    if counts.shape!=conversion.shape or counts.shape!=factors.shape or np.any(counts< -1e-9):
        raise ValueError("invalid source activation increments")
    if not np.any(counts>0):return 0.0
    radius=max(float(radius_m),float(physics.b_m))
    unit=unit_force_resolved_stress(specimen,a_m,radius,np.array([0.0]),factors)[:,0]
    return float(np.sum(np.maximum(counts,0)*conversion*unit)*
                 physics.b_m*specimen.B_m*radius)


def transport_plastic_velocity(specimen:"Specimen",a_m:float,radius_m:float,
                               b_m:float,strip_width_m:float,dx_m:float,
                               mobile_m2,velocity_m_s,drive_factors)->float:
    """Shared unit-force projection of signed Orowan shear transport."""
    mobile=np.asarray(mobile_m2,dtype=float)
    velocity=np.asarray(velocity_m_s,dtype=float)
    if mobile.shape!=velocity.shape or mobile.ndim!=2 or np.any(mobile<0):
        raise ValueError("transport fields must share nonnegative system/bin shape")
    x=(np.arange(mobile.shape[1])+.5)*dx_m
    unit=unit_force_resolved_stress(specimen,a_m,radius_m,x,drive_factors)
    return float(np.sum(unit*b_m*mobile*velocity)*
                 specimen.B_m*strip_width_m*dx_m)


@dataclass(frozen=True)
class Specimen:
    B_m: float
    W_m: float
    L_m: float
    a0_m: float
    E_Pa: float
    nu: float
    initial_compliance_m_per_N: float
    instrument_compliance_m_per_N: float = 0.0
    bending_opening_shape: BendingOpeningShape | None = None

    def __post_init__(self) -> None:
        if not (self.B_m > 0 and self.W_m > self.a0_m > 0 and self.L_m > 0):
            raise ValueError("invalid cantilever geometry")
        if not (self.E_Pa > 0 and -1 < self.nu < 0.5):
            raise ValueError("invalid isotropic elastic constants")
        if self.initial_compliance_m_per_N <= 0 or self.instrument_compliance_m_per_N < 0:
            raise ValueError("invalid compliance")
        if self.bending_opening_shape is not None:
            field=self.bending_opening_shape
            if not (np.isclose(field.height_m,self.W_m,rtol=0,atol=1e-12) and
                    np.isclose(field.thickness_m,self.B_m,rtol=0,atol=1e-12) and
                    np.isclose(field.span_m,self.L_m,rtol=0,atol=1e-12)):
                raise ValueError("elastic field geometry differs from scalar specimen")

    @property
    def Eprime_Pa(self) -> float:
        return self.E_Pa / (1 - self.nu**2)

    @property
    def shear_modulus_Pa(self) -> float:
        return self.E_Pa / (2 * (1 + self.nu))

    def opening_shape(self, a_m: float, radius_m: float) -> float:
        if self.bending_opening_shape is None:
            return 1.0
        return self.bending_opening_shape.factor(a_m-self.a0_m,radius_m)

    def geometry_factor(self, a_m: float) -> float:
        if not self.a0_m <= a_m < self.W_m:
            raise ValueError("crack length outside observed cantilever range")
        x = a_m / self.W_m
        f = 1.24 + 28.08*x - 59.44*x*x + 90.40*x*x*x
        return self.L_m * f / (self.B_m * self.W_m**1.5)

    def elastic_compliance(self, a_m: float) -> float:
        """Integrate dC/da from the measured/assumed initial-notch compliance.

        We never extrapolate the geometry polynomial below the initial notch.
        Gauss-Legendre quadrature is exact to roundoff here because kappa² is a
        sixth-order polynomial in crack length.
        """
        if a_m == self.a0_m:
            return self.initial_compliance_m_per_N
        if not self.a0_m < a_m < self.W_m:
            raise ValueError("crack length outside observed cantilever range")
        mid = 0.5*(self.a0_m+a_m)
        half = 0.5*(a_m-self.a0_m)
        integral = sum(w * self.geometry_factor(mid+half*x)**2
                       for x,w in zip(_GAUSS4_NODES, _GAUSS4_WEIGHTS))*half
        return self.initial_compliance_m_per_N + 2*self.B_m*integral/self.Eprime_Pa

    def equilibrium_force(self, imposed_u_m: float, plastic_u_m: float, a_m: float) -> float:
        return max((imposed_u_m-plastic_u_m)/
                   (self.elastic_compliance(a_m)+self.instrument_compliance_m_per_N), 0.0)

    def nominal_K_MPa_sqrt_m(self, force_N: float) -> float:
        return force_N*self.geometry_factor(self.a0_m)/1e6

    def current_K_MPa_sqrt_m(self, force_N: float, a_m: float) -> float:
        return force_N*self.geometry_factor(a_m)/1e6

    def initial_displacement_rate(self, Kdot_MPa_sqrt_m_per_s: float) -> float:
        return (Kdot_MPa_sqrt_m_per_s*1e6 *
                (self.initial_compliance_m_per_N+self.instrument_compliance_m_per_N) /
                self.geometry_factor(self.a0_m))


@dataclass(frozen=True)
class HistoryPoint:
    time_s: float
    imposed_u_m: float
    force_N: float
    a_m: float
    plastic_u_m: float
    local_drive_K_MPa_sqrt_m: float
    event: bool = False
    physical_failure: bool = False


@dataclass(frozen=True)
class ObservedPoint:
    time_s: float
    u_m: float
    da_m: float
    force_N: float
    K_nominal_MPa_sqrt_m: float
    K_el_current_MPa_sqrt_m: float
    K_local_drive_MPa_sqrt_m: float
    A_pl_J: float
    J_continuous_N_per_m: float
    K_Jcontinuous_MPa_sqrt_m: float
    relative_stiffness: float
    event: bool
    physical_failure: bool
    plastic_u_compliance_m: float = 0.0
    plastic_closure_error_m: float = 0.0
    elastic_crack_energy_increment_J: float = 0.0
    contact_gap_m: float = 0.0


def observe_like_ast(history: Iterable[HistoryPoint], specimen: Specimen) -> list[ObservedPoint]:
    """Apply Eqs. 1–6 to a sampled, ordered loading history.

    Plastic area is trapezoidal force work against compliance-derived plastic
    displacement. The separate crack-compliance energy term cannot enter Jpl.
    The equilibrium convention includes instrument compliance in total C.
    """
    points = list(history)
    if not points:
        return []
    if abs(points[0].imposed_u_m) > 1e-15 or abs(points[0].force_N) > 1e-15:
        raise ValueError("history must begin at zero displacement and force")
    out: list[ObservedPoint] = []
    Apl = 0.0
    Jpl = 0.0
    old = None
    old_upl = 0.0
    C0 = specimen.elastic_compliance(specimen.a0_m)
    for p in points:
        Ce = specimen.elastic_compliance(p.a_m)
        Ctotal = Ce+specimen.instrument_compliance_m_per_N
        upl = p.imposed_u_m-p.force_N*Ctotal
        crack_energy = 0.0
        if old is not None:
            if p.time_s < old.time_s or p.imposed_u_m < old.imposed_u_m - 1e-15 or p.a_m < old.a_m - 1e-15:
                raise ValueError("history must be monotone in time, loading and crack length")
            Apl += 0.5*(old.force_N+p.force_N)*(upl-old_upl)
            old_C = specimen.elastic_compliance(old.a_m)+specimen.instrument_compliance_m_per_N
            crack_energy = 0.5*old.force_N*p.force_N*(Ctotal-old_C)
            ligament = specimen.W_m-old.a_m
            da = p.a_m-old.a_m
            if not 0 <= da < ligament:
                raise ValueError("crack increment consumes ligament")
            dApl = 0.5*(old.force_N+p.force_N)*(upl-old_upl)
            Jpl = (Jpl + 2*dApl/(specimen.B_m*ligament))*(1-da/ligament)
        Kcur = specimen.current_K_MPa_sqrt_m(p.force_N,p.a_m)
        J = (Kcur*1e6)**2/specimen.Eprime_Pa + Jpl
        if J < -1e-8:
            raise ValueError("negative continuous J")
        out.append(ObservedPoint(p.time_s,p.imposed_u_m,p.a_m-specimen.a0_m,p.force_N,
                    specimen.nominal_K_MPa_sqrt_m(p.force_N),Kcur,
                    p.local_drive_K_MPa_sqrt_m,Apl,J,
                    sqrt(max(J,0)*specimen.Eprime_Pa)/1e6,
                    C0/Ce,p.event,p.physical_failure,upl,
                    upl-p.plastic_u_m,crack_energy,
                    max(p.plastic_u_m-p.imposed_u_m,0.0)))
        old = p
        old_upl = upl
    return out


def post_event_J_curve(observed):
    """Baseline and every committed post-event J sample in crack-growth order."""
    points=list(observed)
    if not points:
        return np.empty(0),np.empty(0)
    def field(point,key):
        return point[key] if isinstance(point,dict) else getattr(point,key)
    def event(point):
        value=field(point,"event")
        return value is True or value=="True"
    a=[0.0];J=[float(field(points[0],"J_continuous_N_per_m"))]
    for point in points:
        if not event(point):
            continue
        advance=float(field(point,"da_m"))
        if advance<a[-1]-1e-15:
            raise ValueError("post-event crack extension decreased")
        if advance<=a[-1]+1e-15:
            J[-1]=float(field(point,"J_continuous_N_per_m"))
        else:
            a.append(advance);J.append(float(field(point,"J_continuous_N_per_m")))
    return np.asarray(a),np.asarray(J)


def continuous_onset_K(observed: Iterable[ObservedPoint], specimen: Specimen,
                       target_extension_m: float = 0.2e-6) -> float:
    """Interpolate post-event J(a) only within achieved crack-growth support."""
    pts = list(observed)
    if not pts or max(p.da_m for p in pts) < target_extension_m - 1e-15:
        return float("nan")
    event_points = [p for p in pts if p.event and p.da_m > 0]
    if len(event_points) < 2:
        return float("nan")
    a,J=post_event_J_curve(pts)
    if len(a)<2:
        return float("nan")
    Jq = float(np.interp(target_extension_m, a, J))
    return sqrt(max(Jq, 0.0)*specimen.Eprime_Pa)/1e6


def basic_J_at_failure(observed: Iterable[ObservedPoint], specimen: Specimen,
                       *, use_current_crack: bool = False) -> float:
    """Return basic-J K-equivalent only for an identified final failure.

    The abbreviated published equation leaves the basic-J crack-length
    convention ambiguous. The default uses a0 in both terms; the alternative
    can be reported as a sensitivity, never silently selected by fit quality.
    """
    failures = [p for p in observed if p.physical_failure]
    if not failures:
        return float("nan")
    p = failures[0]
    a = specimen.a0_m + p.da_m if use_current_crack else specimen.a0_m
    K = specimen.current_K_MPa_sqrt_m(p.force_N, a)
    J = (K*1e6)**2/specimen.Eprime_Pa + 2*p.A_pl_J/(specimen.B_m*(specimen.W_m-a))
    return sqrt(max(J,0)*specimen.Eprime_Pa)/1e6


def lefm_Kq_at_failure(observed: Iterable[ObservedPoint], specimen: Specimen) -> float:
    """Apply the initial 95% loading line, requiring a physical failure event."""
    pts = list(observed)
    failures = [i for i,p in enumerate(pts) if p.physical_failure]
    if not failures:
        return float("nan")
    pts = pts[:failures[0]+1]
    C = specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N
    # The offset line is a 5% reduced slope through the physical load origin.
    # Search only after the load has entered its rising branch.
    for old,p in zip(pts[1:-1],pts[2:]):
        d0 = old.force_N-0.95*old.u_m/C
        d1 = p.force_N-0.95*p.u_m/C
        if old.u_m > 0 and d0 >= 0 and d1 < 0:
            f = d0/(d0-d1)
            force = old.force_N+f*(p.force_N-old.force_N)
            return specimen.nominal_K_MPa_sqrt_m(force)
    return max(p.K_nominal_MPa_sqrt_m for p in pts)
