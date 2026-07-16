"""
Arrhenius barrier functions for plasticity and fracture.

Provides clean, tested implementations of:
- Rational H*(sigma), v*(sigma,T) for W plasticity (from fit)
- Polynomial S*(sigma,T) entropy
- Fracture barrier G*_f(sigma,T)
- Hazard-based toughness mapping Kc(T)
"""

import numpy as np
from typing import Callable, Optional
from .config import (
    KB, EV_TO_J, ElasticProperties, PlasticityBarrier,
    FractureBarrier, HazardConfig, PhaseFieldConfig
)


class PlasticityModel:
    """
    Plasticity barrier model.

    Modes:
      rational_Hv: legacy H*(sigma), v*(sigma,T) model with explicit
                   -sigma*v mechanical work.
      exp_floor:   full stress-biased DeltaG(sigma,T) from BarrierModel_Export.json.
                   Here v* is derived from -dDeltaG/dsigma and is not subtracted
                   again from the barrier.
    """

    def __init__(self, params: PlasticityBarrier, mat: ElasticProperties):
        self.p = params
        self.mat = mat

    @property
    def model_type(self) -> str:
        return str(getattr(self.p, 'model_type', 'rational_Hv')).lower()

    @property
    def uses_embedded_stress_barrier(self) -> bool:
        return self.model_type == 'exp_floor'

    def _H_rational(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        x = (np.abs(sigma) / self.p.sig0) ** self.p.n
        den = np.maximum(1 + self.p.chiH * (x - 1), 1e-12)
        return self.p.H0_J / den

    def _v_rational(self, sigma: np.ndarray, T: float) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        v0 = self.p.v0_a * T**2 + self.p.v0_b * T + self.p.v0_c
        v0 = max(v0, 1e-36)
        x = (np.abs(sigma) / self.p.sig0) ** self.p.n
        den = np.maximum(1 + self.p.psiV * (x - 1), 1e-12)
        return v0 / den

    def _exp_sigc(self, T: float) -> float:
        sigc = self.p.exp_sigc0_Pa + self.p.exp_sT_Pa_per_K * (T - self.p.exp_Tref_K)
        sigc *= max(float(getattr(self.p, 'exp_stress_scale', 1.0)), 1e-30)
        return max(sigc, 1e6)

    def _exp_G0_eV(self, T: float) -> float:
        escale = float(getattr(self.p, 'exp_energy_scale', 1.0))
        sscale = float(getattr(self.p, 'exp_entropy_scale', escale))
        return escale * self.p.exp_G00_eV + sscale * self.p.exp_gT_eV_per_K * (T - self.p.exp_Tref_K)

    def _exp_floor_eV(self, T: float) -> float:
        G0 = max(self._exp_G0_eV(T), 1e-8)
        floor = max(float(self.p.exp_Gfloor_min_eV), float(self.p.exp_Gfloor_fraction) * G0)
        floor = min(float(self.p.exp_Gfloor_max_fraction) * G0, floor)
        return max(floor, 0.0)

    def _G_exp_floor_eV(self, sigma: np.ndarray, T: float) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        sigc = self._exp_sigc(T)
        G0 = max(self._exp_G0_eV(T), 1e-8)
        Gfloor = self._exp_floor_eV(T)
        a = max(float(self.p.exp_a), 0.0)
        n = max(float(self.p.exp_n), 1e-8)
        x = (np.abs(sigma) / sigc) ** n
        return Gfloor + (G0 - Gfloor) * np.exp(-a * x)

    def _v_exp_floor(self, sigma: np.ndarray, T: float) -> np.ndarray:
        """Local activation volume v* = -dDeltaG/dsigma [m^3]."""
        sigma = np.asarray(sigma, dtype=float)
        b = self.mat.b
        sigc = self._exp_sigc(T)
        G0 = max(self._exp_G0_eV(T), 1e-8)
        Gfloor = self._exp_floor_eV(T)
        a = max(float(self.p.exp_a), 0.0)
        n = max(float(self.p.exp_n), 1e-8)
        sig_eval = np.maximum(np.abs(sigma), float(self.p.exp_sigma_deriv_min_frac) * sigc)
        x = (sig_eval / sigc) ** n
        v_eV_per_Pa = (G0 - Gfloor) * a * n / sigc * (sig_eval / sigc) ** (n - 1.0) * np.exp(-a * x)
        v = np.maximum(v_eV_per_Pa * EV_TO_J, 0.0)
        vmin = float(getattr(self.p, 'exp_v_min_b3', 1e-3)) * b**3
        vmax = float(getattr(self.p, 'exp_v_max_b3', 1e4)) * b**3
        return np.clip(v, max(vmin, 1e-40), max(vmax, vmin))

    def H(self, sigma: np.ndarray) -> np.ndarray:
        """Activation enthalpy-like diagnostic [J]."""
        if self.uses_embedded_stress_barrier:
            return self._G_exp_floor_eV(sigma, self.p.exp_Tref_K) * EV_TO_J
        return self._H_rational(sigma)

    def v(self, sigma: np.ndarray, T: float) -> np.ndarray:
        """Activation volume [m^3]; exp_floor derives v* from -dG/dsigma."""
        if self.uses_embedded_stress_barrier:
            return self._v_exp_floor(sigma, T)
        return self._v_rational(sigma, T)

    def S(self, sigma: np.ndarray, T: float) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        if not self.uses_embedded_stress_barrier:
            return np.zeros_like(sigma)
        return -float(getattr(self.p, 'exp_entropy_scale', 1.0)) * self.p.exp_gT_eV_per_K * EV_TO_J * np.ones_like(sigma)

    def A(self, sigma: np.ndarray, T: float) -> np.ndarray:
        if self.uses_embedded_stress_barrier:
            return self.G_barrier(sigma, T)
        sigma = np.asarray(sigma, dtype=float)
        return self.H(sigma) - T * self.S(sigma, T)

    def G_barrier(self, sigma: np.ndarray, T: float) -> np.ndarray:
        """Full stress-biased free-energy barrier [J]."""
        sigma = np.asarray(sigma, dtype=float)
        if self.uses_embedded_stress_barrier:
            return np.maximum(self._G_exp_floor_eV(sigma, T) * EV_TO_J, 0.0)
        return np.maximum(
            self.H(sigma) - T * self.S(sigma, T) - np.abs(sigma) * self.v(sigma, T),
            0)

    def sigma_peierls(self, rho: np.ndarray, T: float,
                      dot_ep: float = 1e-6) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        b = self.mat.b
        delta = 1.0 / np.sqrt(np.maximum(rho, 1e6))
        G0 = self.G_barrier(np.array([0.0]), T)[0]
        v0 = self.v(np.array([0.0]), T)[0]
        prefactor = self.p.eta0 * (b / delta)**4
        G_needed = KB * T * np.log(np.maximum(prefactor / max(dot_ep, 1e-30), 1e-300))
        G_needed = np.maximum(G_needed, 0)
        return np.maximum((G0 - G_needed) / max(v0, 1e-36), 0)


class FractureModel:
    """
    Fracture barrier model and toughness mapping.

    Computes the free energy barrier for crack advance:
        G*_f(sigma,T) = H_f(sigma) - T*S_f(sigma) - sigma*v_f(sigma,T)

    And maps it to a temperature-dependent fracture toughness Kc(T)
    through either:
        1. Lambert-W spinodal closure (analytical)
        2. Integrated hazard (numerical)
    """

    def __init__(self, barrier: FractureBarrier, mat: ElasticProperties,
                 hazard: HazardConfig = None, pf: PhaseFieldConfig = None):
        self.fb = barrier
        self.mat = mat
        self.hazard = hazard or HazardConfig()
        self.pf = pf or PhaseFieldConfig()
        self._Kdot_eff = None  # calibrated on first use
        self._T_cal = None

    def _calibrate_Kdot(self, ell: float, T_cal: float = 300.0):
        """
        Calibrate Kdot_eff so that the hazard integral gives B=Btarget
        at the Lambert-W spinodal toughness for T_cal.
        """
        r0 = ell / 2
        Kc_ref = self.toughness_lambertw(T_cal, r0)
        self._T_cal = T_cal

        if T_cal <= 0 or Kc_ref <= self.pf.K_floor:
            self._Kdot_eff = 1.0  # fallback
            return

        b = self.mat.b
        haz = self.hazard

        # Integrate hazard up to Kc_ref
        phi_tip = 1.0 / np.sqrt(2 * np.pi * max(r0, 1e-12))
        NK = 2500
        K = np.linspace(0, max(Kc_ref, 1e-6), NK)
        sigma = phi_tip * K
        sigma = np.maximum(sigma, 0)

        G = self.fb.G_barrier(sigma, T_cal, b)
        log_lam = np.log(haz.Gamma0) - G / (KB * T_cal)
        log_lam = np.clip(log_lam, -745, haz.log_clip)
        lam_A = np.exp(log_lam)

        I = np.trapezoid(lam_A, K)
        self._Kdot_eff = max(I / max(haz.Btarget, 1e-12), 1e-30)

    def Gc_of_T(self, T: float, ell: float,
                method: str = 'lambertw') -> float:
        """
        Compute effective fracture energy Gc(T).

        This is the SINGLE point where Arrhenius barrier physics enters
        the phase-field fracture model. The PF mobility is constant.

        Parameters
        ----------
        method : 'lambertw' or 'hazard'
            - 'lambertw': Uses spinodal closure. With negative S, Kc increases
              with T (DBTT-like). This is the correct approach for materials
              where the barrier itself carries the temperature trend.
            - 'hazard': Integrates Arrhenius rate over K-ramp. Kc always
              DECREASES with T (ceramic-like) because exp(-G*/kBT) always
              increases with T regardless of entropy sign. For DBTT behavior
              with hazard, temperature must enter through plasticity/shielding,
              not the barrier alone.
        """
        r0 = ell / 2  # baseline core radius

        if method == 'hazard':
            if self._Kdot_eff is None:
                self._calibrate_Kdot(ell)
            Kc = self._toughness_hazard(T, r0)
        else:
            Kc = self.toughness_lambertw(T, r0)

        # Convert to Gc
        Gc_arr = Kc**2 / self.mat.Eprime

        # Regularize
        Gc0 = self.pf.Gc0_athermal
        Gc_floor = max(self.pf.Gc_baseline,
                       (self.pf.K_floor**2) / self.mat.Eprime)
        Gc_ceiling = Gc0

        if self.pf.regularization == 'floor_and_ceiling':
            return np.clip(Gc_arr, Gc_floor, Gc_ceiling)
        elif self.pf.regularization == 'floor_only':
            return max(Gc_floor, Gc_arr)
        else:
            return max(1e-30, Gc_arr)

    def _toughness_hazard(self, T: float, r0: float) -> float:
        """
        Dual-channel hazard mapping for Kc(T).

        Integrates the Arrhenius fracture event hazard under a monotonic
        K-ramp with rate Kdot_eff (calibrated at T_cal).
        Kc is defined by B(Kc) = Btarget.

        Channel A: thermally activated (Arrhenius)
        Channel L: athermal lattice floor (series combination)
        """
        b = self.mat.b
        Eprime = self.mat.Eprime
        haz = self.hazard

        # Tip stress amplification
        phi_tip = 1.0 / np.sqrt(2 * np.pi * max(r0, 1e-12))

        if T <= 0:
            # At T=0, only athermal channel
            K0_lat = haz.K0_lattice_MPa * 1e6
            return max(K0_lat, self.pf.K_floor)

        Kdot_eff = max(self._Kdot_eff, 1e-30)

        # K grid
        K_max = max(5 * self.pf.K_floor, 50e6)
        NK = 4000
        K = np.linspace(0, K_max, NK)
        sigma = phi_tip * K
        sigma = np.maximum(sigma, 0)

        # Arrhenius channel
        G = self.fb.G_barrier(sigma, T, b)
        log_lam = np.log(haz.Gamma0) - G / (KB * T)
        log_lam = np.clip(log_lam, -745, haz.log_clip)
        lam_A = np.exp(log_lam)

        # Integrated hazard: B(K) = ∫_0^K lambda(K') / Kdot_eff dK'
        B_cum = np.zeros(NK)
        dK = np.diff(K)
        for i in range(1, NK):
            B_cum[i] = B_cum[i-1] + lam_A[i-1] * dK[i-1] / Kdot_eff

        # Find Kc_A where B crosses Btarget
        if B_cum[-1] <= haz.Btarget:
            Kc_A = K[-1]  # hazard never reaches target → high toughness
        else:
            idx = np.searchsorted(B_cum, haz.Btarget)
            if idx == 0:
                Kc_A = K[0]
            elif idx >= NK:
                Kc_A = K[-1]
            else:
                frac = ((haz.Btarget - B_cum[idx-1]) /
                        max(B_cum[idx] - B_cum[idx-1], 1e-30))
                Kc_A = K[idx-1] + frac * (K[idx] - K[idx-1])

        # Series combination with lattice floor
        K0_lat = haz.K0_lattice_MPa * 1e6
        dK_lat = haz.dK_lattice_MPa * 1e6

        if K0_lat <= 0:
            Kc_eff = Kc_A
        elif dK_lat <= 0:
            Kc_eff = max(Kc_A, K0_lat)
        else:
            m = max(Kc_A, K0_lat)
            Kc_eff = m + dK_lat * np.log(
                np.exp((Kc_A - m) / dK_lat) +
                np.exp((K0_lat - m) / dK_lat)
            )

        return max(Kc_eff, self.pf.K_floor)

    def toughness_lambertw(self, T: float, r0: float) -> float:
        """
        Lambert-W spinodal mapping (analytical toughness).

        Uses the principal branch W_0 to solve:
            sigma_tip * v(sigma_tip) = kB*T
        """
        b = self.mat.b
        v0 = self.fb.v(np.array([0.0]), T, b)[0]
        v0 = max(v0, 1e-36)

        # For the simple model, use 1/sigma0_v as beta
        beta = 1.0 / self.fb.sigma0_v

        if T <= 0:
            return self.pf.K_floor

        arg = -beta * KB * T / v0
        arg = max(arg, -1/np.e + 1e-12)

        try:
            from scipy.special import lambertw as _lambertw
            W0 = float(np.real(_lambertw(arg, k=0)))
        except Exception:
            # Fallback Newton
            W0 = _lambertw_newton(arg)

        sig_star = -W0 / beta
        phi_tip = 1.0 / np.sqrt(2 * np.pi * max(r0, 1e-12))
        Kc = sig_star / phi_tip  # note: phi_tip is the inverse factor

        return max(Kc, self.pf.K_floor)


def _lambertw_newton(z: float, tol: float = 1e-12, maxiter: int = 50) -> float:
    """Newton iteration for Lambert W_0(z)."""
    if z >= 0:
        w = np.log(1 + z)
    else:
        w = -1.0 + 0.1

    for _ in range(maxiter):
        ew = np.exp(w)
        f = w * ew - z
        fp = ew * (w + 1)
        if abs(fp) < 1e-30:
            break
        dw = f / fp
        w -= dw
        if abs(dw) < tol:
            break
    return w
