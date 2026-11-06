"""Conservative finite-residence rewards for the Ast MPZ reduction.

The two-state operator is backward Euler with a mobile outflow rate |v|/L.
Its reward is h times the *new* mobile occupancy, matching the population
operator rather than integrating an unresolved endpoint power spike.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResidenceStep:
    mobile: np.ndarray
    retained: np.ndarray
    mobile_time_m2_s: np.ndarray
    escaped_m2: np.ndarray
    captured_m2: np.ndarray
    released_m2: np.ndarray
    residual_m2: np.ndarray


def residence_step(mobile, retained, velocity_m_s, length_m,
                   encounter_s, release_s, h_s):
    """Solve one coupled mobile/retained BE step with an open mobile edge."""
    m=np.asarray(mobile,dtype=float)
    r=np.asarray(retained,dtype=float)
    v=np.asarray(velocity_m_s,dtype=float)
    capture=np.asarray(encounter_s,dtype=float)
    release=np.asarray(release_s,dtype=float)
    if not (m.shape==r.shape==v.shape==capture.shape==release.shape):
        raise ValueError("residence fields must share shape")
    if (length_m<=0 or h_s<0 or np.any(m<0) or np.any(r<0) or
        np.any(capture<0) or np.any(release<0)):
        raise ValueError("invalid residence state or rates")
    if h_s==0:
        zeros=np.zeros_like(m)
        return ResidenceStep(m.copy(),r.copy(),zeros,zeros,zeros,zeros,zeros)
    escape=np.abs(v)/length_m
    a=1+h_s*(escape+capture)
    d=1+h_s*release
    det=1+h_s*(escape+capture+release)+h_s*h_s*escape*release
    mn=(d*m+h_s*release*r)/det
    rn=(h_s*capture*m+a*r)/det
    mobile_time=h_s*mn
    outflow=escape*mobile_time
    captured=capture*mobile_time
    released=release*h_s*rn
    residual=m+r-mn-rn-outflow
    scale=np.maximum(m+r,1e-300)
    if np.any(np.abs(residual)>2e-12*scale):
        raise ArithmeticError("residence step lost line-content conservation")
    return ResidenceStep(mn,rn,mobile_time,outflow,captured,released,residual)
