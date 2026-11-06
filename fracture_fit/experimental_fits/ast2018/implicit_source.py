"""Numerically resolved version of the pinned persistent-site emission root.

The native root uses ``max(upper, 1)`` as its absolute stopping scale.  For
Ast's small per-step activation counts that creates a spurious activation
floor which accumulates with the number of load steps.  This implementation
keeps the same backward-Euler equation and mechanical blocking bound, but
brackets by the explicit zero-backstress count and stops relative to the
actual activation bracket.
"""
from __future__ import annotations

import math
from typing import Callable

from scipy.optimize import brentq


def mechanically_blocked_at_source_bound(*, activations: float,
    drive_stress_Pa: float, rho_initial_m2: float,
    rho_increment_per_activation_m2: float,
    backstress_prefactor_Pa_sqrt_m2: float) -> bool:
    """Identify the complementary zero-drive root at the exact blocking bound.

    A floating-point square root can leave one ulp of positive stress when
    ``activations`` is exactly the mechanical upper bound. That is a blocked
    source, not a huge active-rate residual. Interior roots remain subject to
    the unchanged rate-equation tolerance.
    """
    A=float(activations);drive=float(drive_stress_Pa)
    rho0=float(rho_initial_m2);rho_per=float(rho_increment_per_activation_m2)
    kback=float(backstress_prefactor_Pa_sqrt_m2)
    if not all(math.isfinite(v) for v in (A,drive,rho0,rho_per,kback)):
        return False
    if A<=0 or drive<=0 or rho0<0 or rho_per<=0 or kback<=0:
        return False
    upper=max(((drive/kback)**2-rho0)/rho_per,0.0)
    if upper<=0 or abs(A-upper)>2*math.ulp(upper):
        return False
    stress=drive-kback*math.sqrt(max(rho0+rho_per*A,0.0))
    return abs(stress)<=8*math.ulp(drive)


def solve_backstress_limited_activations(
    *, multiplicity: float, dt_s: float, drive_stress_Pa: float,
    rho_initial_m2: float, rho_increment_per_activation_m2: float,
    backstress_prefactor_Pa_sqrt_m2: float,
    rate_function: Callable[[float], float], tolerance: float = 1e-10,
    max_iterations: int = 96,
) -> float:
    M=max(float(multiplicity),0.0)
    dt=max(float(dt_s),0.0)
    drive=max(float(drive_stress_Pa),0.0)
    rho0=max(float(rho_initial_m2),0.0)
    rho_per=max(float(rho_increment_per_activation_m2),0.0)
    kback=max(float(backstress_prefactor_Pa_sqrt_m2),0.0)
    if M<=0 or dt<=0 or drive<=0:
        return 0.0
    if rho_per<=0 or kback<=0 or not (0<tolerance<1):
        raise RuntimeError("invalid persistent-site source root parameters")
    sigma0=drive-kback*math.sqrt(rho0)
    if sigma0<=0:
        return 0.0
    rate0=max(float(rate_function(sigma0)),0.0)
    if not math.isfinite(rate0) or rate0<=0:
        return 0.0
    upper=max(((drive/kback)**2-rho0)/rho_per,0.0)
    if upper<=0:
        return 0.0
    explicit=M*rate0*dt
    if explicit<=0:
        return 0.0
    hi=min(upper,explicit)
    if not math.isfinite(hi) or hi<=0:
        raise RuntimeError("nonfinite persistent-site activation bracket")

    def residual(value: float) -> float:
        if value>=upper:
            return value
        rho=rho0+rho_per*max(value,0.0)
        stress=drive-kback*math.sqrt(max(rho,0.0))
        rate=0.0 if stress<=0 else max(float(rate_function(stress)),0.0)
        if not math.isfinite(rate):
            raise RuntimeError("nonfinite persistent-site rate")
        return value-M*rate*dt

    fhi=residual(hi)
    if fhi<0.0:
        hi=upper
        fhi=residual(hi)
    if fhi<0.0:
        raise RuntimeError("persistent-site root failed to bracket")
    if fhi==0.0:
        return hi
    # Brent's safeguarded interpolation resolves the same monotone root with
    # far fewer barrier calls than fixed bisection in this stiff rate range.
    root=float(brentq(residual,0.0,hi,xtol=math.ulp(0.0),
                      rtol=tolerance,maxiter=int(max_iterations)))
    # The rate is zero at zero net stress, so the residual jumps where the
    # stress vanishes. A sign change with an unresolved residual whose net
    # stress reaches zero within the root tolerance is the blocked
    # complementary root; return the bound exactly so the caller recognizes
    # it as blocked. (Near-balanced drive and backstress move the numerical
    # zero-stress point off the analytic bound by cancellation.) Continuous
    # roots are left unchanged.
    if abs(residual(root))>2e-8*max(root,explicit):
        beyond=root*(1+4*tolerance)+math.ulp(root)
        if drive-kback*math.sqrt(rho0+rho_per*beyond)<=0.0:
            return upper
    return root
