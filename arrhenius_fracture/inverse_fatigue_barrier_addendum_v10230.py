"""Analysis-only multi-R and generalized-stress tools for inverse design.

The production constitutive model is deliberately not imported by this module.
It formalizes loading-path, NS1, and FA1 diagnostics while retaining the
qualified scalar EXP-floor surface as the NS0/FA1-isotropic limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import quad

from .crystal import bcc_cleavage_traces, bcc_slip_traces
from .material_manifest import ExpFloorBarrier


MODEL_ID = "v10.2.30_inverse_fatigue_barrier_multi_R_anisotropy_v1"


def signed_waveform(phi: np.ndarray | float, R: float) -> np.ndarray:
    phase = np.asarray(phi, dtype=float)
    return 0.5 * (1.0 + float(R)) + 0.5 * (1.0 - float(R)) * np.cos(phase)


def opening_waveform(phi: np.ndarray | float, R: float) -> np.ndarray:
    return np.maximum(signed_waveform(phi, R), 0.0)


def recover_load_ratio(Kmax: float, Kmin: float) -> float:
    if not math.isfinite(Kmax) or Kmax <= 0.0 or not math.isfinite(Kmin):
        raise ValueError("Kmax must be finite and positive and Kmin finite")
    return float(Kmin) / float(Kmax)


def waveform_factor_quadrature(M: float, R: float) -> float:
    if M <= 0.0 or not math.isfinite(M):
        raise ValueError("M must be finite and positive")
    value, _ = quad(lambda x: float(opening_waveform(x, R) ** M),
                    0.0, 2.0 * math.pi, epsabs=1e-13, epsrel=1e-13, limit=400)
    return value / (2.0 * math.pi)


def waveform_factor_midpoint(M: float, R: float, n_phase: int = 131072) -> float:
    phase = 2.0 * math.pi * (np.arange(int(n_phase)) + 0.5) / int(n_phase)
    return float(np.mean(opening_waveform(phase, R) ** float(M)))


def waveform_factor_peak_asymptotic(M: float, R: float) -> float:
    if M <= 0.0 or R >= 1.0:
        raise ValueError("peak asymptotic requires M>0 and R<1")
    return 1.0 / math.sqrt(math.pi * float(M) * (1.0 - float(R)))


def fixed_deltaK_rate(rate_fixed_Kmax: Callable[[float, float], float],
                      deltaK: float, R: float) -> float:
    if R >= 1.0:
        raise ValueError("fixed-DeltaK mapping requires R<1")
    return float(rate_fixed_Kmax(float(deltaK) / (1.0 - float(R)), float(R)))


def logarithmic_K_slope(rate_fixed_Kmax: Callable[[float, float], float],
                        Kmax: float, R: float, step: float = 1e-5) -> float:
    lo, hi = Kmax * math.exp(-step), Kmax * math.exp(step)
    return (math.log(rate_fixed_Kmax(hi, R)) -
            math.log(rate_fixed_Kmax(lo, R))) / (2.0 * step)


def R_sensitivity(rate_fixed_Kmax: Callable[[float, float], float],
                  Kmax: float, R: float, step: float = 1e-5) -> float:
    return (math.log(rate_fixed_Kmax(Kmax, R + step)) -
            math.log(rate_fixed_Kmax(Kmax, R - step))) / (2.0 * step)


def fixed_deltaK_R_sensitivity(rate_fixed_Kmax: Callable[[float, float], float],
                               deltaK: float, R: float,
                               step: float = 1e-5) -> float:
    return (math.log(fixed_deltaK_rate(rate_fixed_Kmax, deltaK, R + step)) -
            math.log(fixed_deltaK_rate(rate_fixed_Kmax, deltaK, R - step))) / (2.0 * step)


def derivative_identity(rate_fixed_Kmax: Callable[[float, float], float],
                        Kmax: float, R: float, step: float = 1e-5) -> Mapping[str, float]:
    deltaK = (1.0 - R) * Kmax
    left = fixed_deltaK_R_sensitivity(rate_fixed_Kmax, deltaK, R, step)
    direct = R_sensitivity(rate_fixed_Kmax, Kmax, R, step)
    slope = logarithmic_K_slope(rate_fixed_Kmax, Kmax, R, step)
    right = direct + slope / (1.0 - R)
    return {"fixed_deltaK_derivative": left, "fixed_Kmax_direct": direct,
            "local_slope_remapping": slope / (1.0 - R), "right_hand_side": right,
            "closure_residual": left - right}


def mode_I_tensor_factor(theta_rad: float) -> np.ndarray:
    """Dimensionless Williams mode-I tensor at polar angle theta."""
    c, s = math.cos(theta_rad / 2.0), math.sin(theta_rad / 2.0)
    s3, c3 = math.sin(3.0 * theta_rad / 2.0), math.cos(3.0 * theta_rad / 2.0)
    return np.array([[c * (1.0 - s * s3), c * s * c3],
                     [c * s * c3, c * (1.0 + s * s3)]], dtype=float)


@dataclass(frozen=True)
class GeneralizedStress:
    tau_Pa: float
    tau_ng1_Pa: float
    sigma_n_Pa: float
    pressure_Pa: float

    def vector(self) -> np.ndarray:
        return np.array([self.tau_Pa, self.tau_ng1_Pa,
                         self.sigma_n_Pa, self.pressure_Pa], dtype=float)


def generalized_slip_stress(K_Pa_sqrt_m: float, radius_m: float,
                            trace: Mapping[str, object]) -> GeneralizedStress:
    t = np.asarray(trace["t"], dtype=float); n = np.asarray(trace["n"], dtype=float)
    theta = math.atan2(float(t[1]), float(t[0]))
    sigma = (max(float(K_Pa_sqrt_m), 0.0) /
             math.sqrt(2.0 * math.pi * float(radius_m))) * mode_I_tensor_factor(theta)
    tau = float(t @ sigma @ n)
    sigma_t = float(t @ sigma @ t); sigma_n = float(n @ sigma @ n)
    return GeneralizedStress(tau, sigma_t - sigma_n, sigma_n,
                             -0.5 * float(np.trace(sigma)))


def generalized_stress_paths(Kmax_MPa_sqrt_m: float, R: float,
                             orientation_deg: float, n_phase: int = 128,
                             radius_m: float = 1e-6) -> list[dict]:
    phase = 2.0 * math.pi * (np.arange(n_phase) + 0.5) / n_phase
    signed = signed_waveform(phase, R)
    opening = np.maximum(signed, 0.0)
    rows: list[dict] = []
    for system, trace in enumerate(bcc_slip_traces(orientation_deg)):
        for j, (hs, ho) in enumerate(zip(signed, opening)):
            xi = generalized_slip_stress(Kmax_MPa_sqrt_m * 1e6 * ho, radius_m, trace)
            rows.append({"orientation_deg":orientation_deg,"R":R,"phase_index":j,
                         "phase_rad":phase[j],"system_index":system,"system_name":trace["name"],
                         "K_signed_MPa_sqrt_m":Kmax_MPa_sqrt_m*hs,
                         "K_open_MPa_sqrt_m":Kmax_MPa_sqrt_m*ho,
                         "tau_Pa":xi.tau_Pa,"tau_ng1_Pa":xi.tau_ng1_Pa,
                         "sigma_n_Pa":xi.sigma_n_Pa,"pressure_Pa":xi.pressure_Pa})
    return rows


def linear_non_schmid_drive(xi: GeneralizedStress,
                            coefficients: Sequence[float]) -> float:
    c = np.asarray(coefficients, dtype=float)
    if c.shape != (4,):
        raise ValueError("NS1 coefficient vector must have four entries")
    return float(c @ xi.vector())


def generalized_exp_floor_gradient(barrier: ExpFloorBarrier, xi: GeneralizedStress,
                                   coefficients: Sequence[float], T_K: float) -> np.ndarray:
    c = np.asarray(coefficients, dtype=float)
    chi = linear_non_schmid_drive(xi, c)
    if chi <= 0.0:
        return np.zeros(4)
    G0 = max(barrier.G00_eV + barrier.gT_eV_per_K*(T_K-barrier.Tref_K), 1e-12)
    sigc = max(barrier.sigc0_Pa + barrier.sT_Pa_per_K*(T_K-barrier.Tref_K), 1.0)
    floor = min(barrier.floor_max_fraction*G0,
                max(barrier.floor_min_eV, barrier.floor_fraction*G0))
    n = max(barrier.exponent, 1e-9)
    u = max(barrier.alpha, 0.0)*(chi/sigc)**n
    dG_dchi = -(G0-floor)*n*u*math.exp(-u)/chi
    return dG_dchi*c


def fa1_plane_opening_stresses(K_Pa_sqrt_m: float, radius_m: float,
                               orientation_deg: float,
                               include_110: bool = True) -> list[dict]:
    rows=[]
    for plane in bcc_cleavage_traces(orientation_deg, include_110=include_110):
        t=np.asarray(plane["t"]);n=np.asarray(plane["n"])
        theta=math.atan2(float(t[1]),float(t[0]))
        sigma=max(K_Pa_sqrt_m,0.0)/math.sqrt(2*math.pi*radius_m)*mode_I_tensor_factor(theta)
        snn=max(float(n@sigma@n),0.0)
        rows.append({"plane_name":plane["name"],"family":plane["family"],
                     "orientation_deg":orientation_deg,"angle_deg":plane["angle_deg"],
                     "gamma_rel":plane["gamma_rel"],"sigma_nn_open_Pa":snn,
                     "signed_configurational_score":snn/math.sqrt(plane["gamma_rel"])})
    return rows


def svd_identifiability(jacobian: np.ndarray, rank_tolerance: float = 1e-8) -> Mapping[str, object]:
    J=np.asarray(jacobian,dtype=float)
    if J.ndim != 2 or not np.isfinite(J).all():
        raise ValueError("finite two-dimensional Jacobian required")
    _, s, vh=np.linalg.svd(J,full_matrices=False)
    cutoff=rank_tolerance*(s[0] if len(s) else 1.0)
    rank=int(np.sum(s>cutoff))
    condition=float(s[0]/s[-1]) if len(s) and s[-1]>0 else math.inf
    return {"singular_values":s,"right_vectors":vh,"rank":rank,
            "condition_number":condition,"cutoff":cutoff}


__all__ = ["GeneralizedStress", "MODEL_ID", "derivative_identity",
           "fa1_plane_opening_stresses", "fixed_deltaK_rate",
           "generalized_exp_floor_gradient", "generalized_slip_stress",
           "generalized_stress_paths", "linear_non_schmid_drive",
           "mode_I_tensor_factor", "opening_waveform", "recover_load_ratio",
           "signed_waveform", "svd_identifiability", "waveform_factor_midpoint",
           "waveform_factor_peak_asymptotic", "waveform_factor_quadrature"]
