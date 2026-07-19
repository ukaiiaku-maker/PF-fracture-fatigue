"""
Configuration dataclasses for Arrhenius fracture simulation.

Replaces the ~100+ loose variables in the MATLAB code with structured,
documented, validated configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


# Physical constants
KB = 1.380649e-23       # Boltzmann constant [J/K]
EV_TO_J = 1.602176634e-19  # eV -> J


@dataclass
class GeometryConfig:
    """Specimen geometry."""
    Lx: float = 2e-3           # width [m]
    Ly: float = 4e-3           # height [m]
    a0: float = 0.5e-3         # initial notch length from left edge [m]
    notch_half_thickness: float = 0.08e-3  # half-thickness of notch band [m]


@dataclass
class MeshConfig:
    """Mesh parameters.

    By default the regularization length is tied to the mesh,
    ell = ell_factor*hbar.  For convergence studies, set ell_absolute_m
    to keep the physical process-zone length fixed while h is
    refined.  In that case ell_factor is used only for reporting.
    """
    nx: int = 80               # nodes in x
    ny: int = 160              # nodes in y
    jitter: float = 0.30       # random perturbation (0..0.49 of cell size)
    ell_factor: float = 4.0    # ell = ell_factor * hbar when ell_absolute_m is None
    ell_absolute_m: Optional[float] = None  # fixed physical ell [m] for mesh-convergence studies
    # Tip grading (adaptive mesh): if tip_h_fine > 0, the grid is spaced
    # ~tip_h_fine near the crack tip and coarsens by tip_ratio per element
    # toward the far field.  This resolves a small process zone (h << L_pz)
    # at a node count ~ log(domain/h_fine), not domain/h_fine.  nx/ny then act
    # as an upper bound on the march (rarely reached).
    tip_h_fine: float = 0.0    # fine spacing at the tip [m]; 0 -> uniform
    tip_ratio: float = 1.15    # geometric coarsening ratio per element


@dataclass
class ElasticProperties:
    """Linear elastic material properties."""
    E: float = 410e9           # Young's modulus [Pa] (W)
    nu: float = 0.28           # Poisson's ratio
    b: float = 2.74e-10        # Burgers vector [m] (W)
    Tm: float = 3695.0         # melting temperature [K] (W)

    @property
    def G(self) -> float:
        """Shear modulus."""
        return self.E / (2 * (1 + self.nu))

    @property
    def Eprime(self) -> float:
        """Plane-strain modulus E' = E/(1-ν²)."""
        return self.E / (1 - self.nu**2)

    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter (= shear modulus)."""
        return self.G


@dataclass
class PlasticityBarrier:
    """
    Arrhenius plasticity barrier parameters.

    The plastic flow rate is:
        dot_ep = eta0 * (b/delta)^4 * exp(-G*(sigma,T) / (kB*T))

    where G*(sigma,T) = H*(sigma) - T*S*(sigma,T) - sigma*v*(sigma,T)

    H*, v* use rational forms fitted to W data.
    S* uses a polynomial form fitted to W data.

    These parameters are FROM THE FIT and should not be modified by
    fracture presets. Modifying them destroys the plastic shielding
    that drives DBTT behavior.
    """
    H0_J: float = 0.51 * 1.602176634e-19  # H at sig0 [J]; H(0) = H0/(1-chi) ≈ 1.7 eV
    eta0: float = 1e12         # attempt frequency [1/s]
    sig0: float = 2e9          # stress scale [Pa] (~2 GPa Peierls stress for W)
    n: float = 0.25            # rational function exponent
    chiH: float = 0.70         # H rational function mixing; H(0) = H0/(1-chiH)
    psiV: float = 0.70         # v rational function mixing; v(0) = v0/(1-psiV)
    # v0(T) polynomial coefficients: v0 = a*T^2 + b*T + c
    v0_a: float = 0.0
    v0_b: float = 0.0
    v0_c: float = 7.5 * (2.74e-10)**3  # v at sig0 [m^3]; v(0) = v0/(1-psi) ≈ 25 b^3

    # Alternative plasticity barrier model.  In exp_floor mode DeltaG(sigma,T)
    # is a full stress-biased free-energy barrier imported from the nanopillar
    # BarrierModel_Export.json.  The code derives local v* = -dDeltaG/dsigma
    # and must not subtract sigma*v* a second time.
    model_type: Literal['rational_Hv', 'exp_floor'] = 'rational_Hv'

    # EXP_floor parameters copied from one BarrierModel_Export.json system.
    exp_system: str = 'W[100]'
    exp_Tref_K: float = 300.0
    exp_Tmin_K: float = 0.0
    exp_Tmax_K: float = 5000.0
    exp_G00_eV: float = 1.0
    exp_gT_eV_per_K: float = 0.0
    exp_sigc0_Pa: float = 2.0e9
    exp_sT_Pa_per_K: float = 0.0
    exp_a: float = 0.25
    exp_n: float = 1.0
    exp_Gfloor_fraction: float = 0.02
    exp_Gfloor_min_eV: float = 1.0e-4
    exp_Gfloor_max_fraction: float = 0.95

    # Taylor scaling of a nanopillar nucleation barrier.  energy_scale reduces
    # the Tref enthalpic level; entropy_scale scales gT separately so the
    # fitted ~-40 kB entropy can be preserved even with a 0.1-0.3 Taylor barrier.
    exp_energy_scale: float = 0.2
    exp_entropy_scale: float = 1.0
    exp_stress_scale: float = 1.0

    # Derivative regularization for v* = -dDeltaG/dsigma.  Some fits have
    # n<1, which is singular at zero stress, so evaluate derivatives above a
    # small stress floor and cap the derived volume.
    exp_sigma_deriv_min_frac: float = 1.0e-4
    exp_v_min_b3: float = 1.0e-3
    exp_v_max_b3: float = 1.0e4


@dataclass
class FractureBarrier:
    """
    Fracture activation barrier parameters.

    INDEPENDENT of plasticity parameters. These control Gc(T) only.

    The fracture barrier is:
        G*_f(sigma,T) = H*_f(sigma) - T*S*_f(sigma) - sigma*v*_f(sigma,T)

    For BDT behavior (Kc increases with T), S*_f < 0 is needed.
    Physical basis: the crack-tip transition state can be more ordered
    than the ground state if the lattice around the crack tip is
    geometrically constrained.

    DESIGN: Only 5 essential parameters instead of the original 9+.
    The stress dependence exponents are fixed at physically motivated values.
    """
    # Enthalpy: H_f(sigma) = H0 / (1 + (sigma/sigma0_H)^2)
    H0_eV: float = 2.0            # zero-stress barrier [eV]
    sigma0_H_GPa: float = 3.0     # stress for barrier collapse [GPa]

    # Activation volume: v_f(sigma) = v0 / (1 + (sigma/sigma0_v)^1.5)
    # v0 is in units of b^3. For cleavage, this is a single bond-breaking
    # displacement (~1-3 b^3), NOT a kink-pair sweep area (~25 b^3).
    v0_b3: float = 2.0            # zero-stress volume [b^3]
    sigma0_v_GPa: float = 3.0     # stress for volume collapse [GPa]

    # Entropy: S_f(sigma) = -S0_neg * (1 + sigma/sigma0_S)  if negative
    #          S_f = 0 otherwise
    S0_neg_kB: float = 3.0        # magnitude of negative entropy [kB]
    sigma0_S_GPa: float = 2.0     # stress scale for entropy [GPa]
    use_negative_entropy: bool = True  # True = BDT-like, False = no entropy
    # Stress dependence of the activation entropy.  'affine' is the legacy form
    # S = -S0*(1 + sigma/sigma0_S), which is NON-zero (=-S0) at sigma=0 and so
    # makes the ZERO-STRESS barrier collapse with T -- cleavage then fires
    # thermally at any stress at high T and there is no ductile regime.
    # 'gated' uses a saturating (Hill) gate S = -S0 * x/(1+x), x=sigma/sigma0_S,
    # which -> 0 as sigma->0 (cold zero-stress barrier preserved, no thermal
    # collapse) and -> -S0 at high stress (threshold rises with T only where the
    # tip is loaded).  This is the form needed for a DBTT crossover and is the
    # same physics implicated in low-T fatigue limits: the constrained
    # transition-state entropy exists only under load.
    entropy_stress_form: str = 'affine'   # 'affine' | 'gated' | 'physical'
    entropy_gate_power: float = 1.0       # Hill exponent n in x^n/(1+x^n)
    # --- physically-grounded composite entropy (entropy_stress_form='physical') ---
    # S*(sigma,T) = S_T(T) + S_sigma(sigma), built from three documented pieces:
    #
    #  S_T(T): experimentally/atomistically-derived BASELINE activation entropy.
    #    For W dislocation glide/bypass this is a polynomial in T (Veverka &
    #    Dillon fatigue draft, eq. set; Allera et al. Nat. Commun. 2025) that
    #    saturates at min/max.  Implemented as S_T = clip(c0 + c1*T + c2*T^2,
    #    S_T_min, S_T_max), in kB.  This is the piece the ORIGINAL emission model
    #    had (T-dependent, no stress dependence).
    #
    #  S_sigma(sigma): Schoeck thermoelastic / interaction entropy (Schoeck 1980,
    #    eqs 36/40; fatigue draft eq 37).  The activation entropy of a defect
    #    moving through its own internal stress field scales with the stored
    #    elastic energy and -(1/mu) dmu/dT.  It is ZERO at zero stress (a
    #    homogeneous strain gives no harmonic frequency change -- Schoeck sec. 4)
    #    and becomes MORE NEGATIVE with stress (the design rule for a plateau /
    #    transition: negative S with a positive-magnitude stress slope).
    #    Implemented as S_sigma = -(S_sigma_max_kB) * x^n/(1+x^n),
    #    x = sigma/sigma0_S -- the same gate, but now ADDED to S_T rather than
    #    replacing it, and with an independently swept magnitude.
    #
    # The emission channel gets a large S_T (dislocation core: ~kB per b, Schoeck
    # eq 34) plus the Schoeck stress term; the cleavage channel gets S_T ~ 0
    # (no core-mode reorganization) and only the weak modulus term.
    S_T_c0_kB: float = 0.0       # baseline polynomial constant [kB]
    S_T_c1_kB_per_K: float = 0.0 # baseline polynomial linear slope [kB/K]
    S_T_c2_kB_per_K2: float = 0.0# baseline polynomial quadratic [kB/K^2]
    S_T_min_kB: float = -40.0    # baseline saturation floor [kB]
    S_T_max_kB: float = 0.0      # baseline saturation ceiling [kB]
    S_sigma_max_kB: float = 0.0  # magnitude of the Schoeck stress term [kB]
    # --- entropy-enthalpy compensation form (entropy_stress_form='meyer_neldel') ---
    # The Meyer-Neldel rule / Schoeck modulus argument: the activation entropy of
    # a thermally activated process is PROPORTIONAL to its activation enthalpy,
    #     S*(sigma,T) = sign * H*(sigma,T) / T_MN ,
    # with a single compensation temperature T_MN (equivalently a coupling
    # beta = 1/T_MN).  Schoeck (1980) gives beta = -(1/mu) dmu/dT ~ 1.5e-4 /K for
    # W (mu ~ 160 GPa, dmu/dT ~ -25 MPa/K), i.e. T_MN ~ 6000-7000 K, S* ~ +3-4 kB
    # at zero stress.  This form is intrinsically BOUNDED (it inherits H's
    # stress-collapse) -- unlike an unconstrained polynomial fit it cannot run to
    # hundreds of kB at GPa tip stresses.  sign=+1 is the physical
    # modulus-softening case (S>0, barrier eases with T); sign=-1 explores the
    # constrained-transition-state case used for a toughness upturn.
    S_MN_T_MN_K: float = 6500.0  # compensation temperature [K]
    S_MN_sign: float = 1.0       # +1 modulus-softening; -1 constrained TS

    # --- EXP_floor barrier (experimental nanopillar form) ------------------
    # When barrier_kind=='exp_floor' the whole barrier is taken directly from
    # the nanopillar fit instead of the H - T*S - sigma*v decomposition:
    #   dG*(s,T) = Gfloor(T) + [G0(T)-Gfloor(T)] * exp[-a*(s/sigc(T))^n]
    #   G0(T)    = G00 + gT*(T-Tref);   sigc(T) = sigc0 + sT*(T-Tref)
    #   Gfloor(T)= min(fmax*G0, max(Gfloor_min, ffrac*G0))
    # The activation entropy S* = -dG*/dT then emerges with the correct
    # stress dependence (largest at s=0 ~ -gT, decaying toward the floor under
    # load) -- this is the emission/nucleation channel.  a,n come from the
    # per-temperature stress fits (BarrierModel export); the values below are
    # PLACEHOLDERS until the export is supplied.
    barrier_kind: str = 'classic'        # 'classic' | 'exp_floor'
    ef_G00_eV: float = 1.94022           # W[100] exact (BarrierModel export)
    ef_gT_eV_per_K: float = 0.00393367
    ef_sigc0_Pa: float = 2.29797e9
    ef_sT_Pa_per_K: float = -656405.0
    ef_a: float = 0.0845685              # fitted (weak stress sensitivity)
    ef_n: float = 1.0
    ef_Tref_K: float = 481.33
    ef_floor_frac: float = 0.02
    ef_floor_min_eV: float = 1e-4
    ef_floor_max_frac: float = 0.95
    # Temperature mode for the EXP-floor free-energy surface.
    #   linear:   G0=G00+gT*(T-Tref), sigc=sigc0+sT*(T-Tref).
    #   mu_scale: G0=G00*[mu(T)/mu(Tref)]^pG and
    #             sigc=sigc0*[mu(T)/mu(Tref)]^psig, with a simple
    #             local shear-modulus proxy mu/mu_ref = 1+dlnmu_dT*(T-Tref).
    # The mu_scale option is intended for cleavage sweeps where we want the
    # temperature dependence to resemble elastic modulus softening instead of
    # importing a nanopillar nucleation entropy slope directly.
    ef_T_mode: str = 'linear'            # 'linear' | 'mu_scale'
    ef_mu_dlnmu_dT_per_K: float = -1.5e-4
    ef_G0_mu_power: float = 1.0
    ef_sigc_mu_power: float = 1.0
    # --- high-stress entropy crossover (fatigue-paper hypothesis) ----------
    # The nanopillar fit gives S*<0, but it is only valid up to ~sigc.  At the
    # crack tip (stresses >> sigc) the activation entropy is argued to become
    # LESS negative or POSITIVE (the high-stress critical nucleus is small and
    # localized -- the Ryu-Cai entropic-nucleation regime).  We add a saturating
    # stress crossover that LEAVES THE FIT UNCHANGED at low stress and at Tref,
    # and shifts S* by +ef_S_hs_kB above ef_sigma_S:
    #   dS_hs(sigma) = ef_S_hs_kB * x^p/(1+x^p),  x = sigma/ef_sigma_S
    #   S*(sigma,T)  = S*_fit(sigma,T) + dS_hs(sigma)
    #   dG*(sigma,T) = dG*_fit(sigma,T) - (T-Tref)*dS_hs(sigma)   [floored >=0]
    # ef_S_hs_kB=0 recovers the pure data fit.  Making S* positive at the tip
    # (ef_S_hs_kB > |S*_fit| ~ 45) lets emission ramp with T -> a DBTT.  An
    # optional linear-in-T amplitude (ef_S_hs_dT) tilts the crossover with T.
    ef_S_hs_kB: float = 0.0              # high-stress entropy shift [kB]
    ef_sigma_S_GPa: float = 6.0          # crossover stress [GPa]
    ef_S_hs_power: float = 2.0           # crossover sharpness
    ef_S_hs_dT_per_K: float = 0.0        # optional dependence of shift on T [kB/K]
    ef_S_hs_Tref_K: float = 481.33       # ref T for the shift's T-dependence
    # Monotonicity guard: clamp G*(sigma) at its argmin so the barrier is
    # non-increasing in stress (the raw form rises again at high sigma -- an
    # extrapolation artifact of the sigma*v work-term collapse).  Bit-identical
    # below the argmin; only the pathological overshoot regime changes.
    monotone_stress: bool = True

    @property
    def H0_J(self) -> float:
        return self.H0_eV * EV_TO_J

    @property
    def sigma0_H(self) -> float:
        return self.sigma0_H_GPa * 1e9

    @property
    def sigma0_v(self) -> float:
        return self.sigma0_v_GPa * 1e9

    @property
    def sigma0_S(self) -> float:
        return self.sigma0_S_GPa * 1e9

    @property
    def S0_neg_J_per_K(self) -> float:
        return self.S0_neg_kB * KB

    def H(self, sigma: np.ndarray, T: float = 0.0) -> np.ndarray:
        """Activation enthalpy [J]."""
        sigma = np.asarray(sigma, dtype=float)
        return self.H0_J / (1 + (np.abs(sigma) / self.sigma0_H)**2)

    def v(self, sigma: np.ndarray, T: float = 0.0, b: float = 2.74e-10) -> np.ndarray:
        """Activation volume [m^3]."""
        sigma = np.asarray(sigma, dtype=float)
        v0 = self.v0_b3 * b**3
        return v0 / (1 + (np.abs(sigma) / self.sigma0_v)**1.5)

    def _exp_floor(self, sigma: np.ndarray, T: float):
        """EXP_floor barrier and its entropy.  Returns (dG_J, S_JperK), both
        arrays shaped like sigma.  Faithful to the nanopillar export model:
        S* = -d(dG*)/dT at fixed sigma, computed analytically (includes the
        G0(T), sigc(T) and floor temperature dependences)."""
        sigma = np.abs(np.asarray(sigma, dtype=float))
        Tmode = str(getattr(self, 'ef_T_mode', 'linear')).lower()
        if Tmode in ('mu', 'mu_scale', 'modulus', 'shear_modulus'):
            # Minimal shear-modulus-like temperature law for cleavage sweeps.
            # dlnmu_dT < 0 gives softening with T.  Clamp away from zero so
            # exploratory high-T sweeps remain numerically well posed.
            dln = float(getattr(self, 'ef_mu_dlnmu_dT_per_K', -1.5e-4))
            mu_ratio = max(0.05, 1.0 + dln * (T - self.ef_Tref_K))
            pG = float(getattr(self, 'ef_G0_mu_power', 1.0))
            ps = float(getattr(self, 'ef_sigc_mu_power', 1.0))
            G0 = self.ef_G00_eV * (mu_ratio ** pG)
            sigc = self.ef_sigc0_Pa * (mu_ratio ** ps)
            dmu_ratio_dT = dln
            dG0_dT = self.ef_G00_eV * pG * (mu_ratio ** (pG - 1.0)) * dmu_ratio_dT
            dsigc_dT = self.ef_sigc0_Pa * ps * (mu_ratio ** (ps - 1.0)) * dmu_ratio_dT
        else:
            G0 = self.ef_G00_eV + self.ef_gT_eV_per_K * (T - self.ef_Tref_K)
            sigc = self.ef_sigc0_Pa + self.ef_sT_Pa_per_K * (T - self.ef_Tref_K)
            dG0_dT = self.ef_gT_eV_per_K
            dsigc_dT = self.ef_sT_Pa_per_K
        G0 = max(G0, 1e-9); sigc = max(sigc, 1.0)
        a, n = float(self.ef_a), float(self.ef_n)
        x = np.maximum(sigma, 0.0) / sigc
        # x^(n-1)*sigma is finite at sigma=0 (=x^n*sigc); use that form
        xn = np.power(np.where(x > 0, x, 1e-300), n)
        expT = np.exp(-a * xn)
        # d(expTerm)/dT through sigc(T): exp[-a(s/sigc)^n]
        # derivative = expT*(a*n)*x^n*(1/sigc)*dsigc/dT.
        dExp_dT = expT * (a * n) * xn * (1.0 / sigc) * dsigc_dT
        dExp_dT = np.where(sigma > 0, dExp_dT, 0.0)
        raw_floor = max(self.ef_floor_min_eV, self.ef_floor_frac * G0)
        Gfloor = min(self.ef_floor_max_frac * G0, raw_floor)
        if (self.ef_floor_frac * G0 >= self.ef_floor_min_eV) and \
           (raw_floor <= self.ef_floor_max_frac * G0):
            dFloor = self.ef_floor_frac * dG0_dT
        elif raw_floor > self.ef_floor_max_frac * G0:
            dFloor = self.ef_floor_max_frac * dG0_dT
        else:
            dFloor = 0.0
        amp = G0 - Gfloor
        dAmp = dG0_dT - dFloor
        dG_eV = Gfloor + amp * expT
        dGdT_eV = dFloor + dAmp * expT + amp * dExp_dT
        # --- high-stress entropy crossover (preserves fit at low sigma & Tref) ---
        S_hs_kB = float(getattr(self, 'ef_S_hs_kB', 0.0))
        if S_hs_kB != 0.0:
            sS = max(float(getattr(self, 'ef_sigma_S_GPa', 6.0)) * 1e9, 1.0)
            p = max(float(getattr(self, 'ef_S_hs_power', 2.0)), 1e-6)
            xs = np.power(np.maximum(sigma, 0.0) / sS, p)
            gate = xs / (1.0 + xs)
            # amplitude may tilt linearly with T
            amp_kB = S_hs_kB + float(getattr(self, 'ef_S_hs_dT_per_K', 0.0)) \
                * (T - float(getattr(self, 'ef_S_hs_Tref_K', self.ef_Tref_K)))
            dS_hs_eV_per_K = amp_kB * gate * (KB / EV_TO_J)   # eV/K, +ve = less negative S
            # dG correction vanishes at Tref (fit magnitude preserved there)
            dG_eV = dG_eV - (T - self.ef_Tref_K) * dS_hs_eV_per_K
            # S* = -dG/dT picks up +dS_hs (T-derivative of the added -(T-Tref)*dS term)
            dGdT_eV = dGdT_eV - dS_hs_eV_per_K
        dG_J = np.maximum(dG_eV, 0.0) * EV_TO_J
        S_JperK = -dGdT_eV * EV_TO_J * np.ones_like(sigma)
        return dG_J, S_JperK

    def S(self, sigma: np.ndarray, T: float = 0.0) -> np.ndarray:
        """Activation entropy [J/K]."""
        sigma = np.asarray(sigma, dtype=float)
        if str(getattr(self, 'barrier_kind', 'classic')) == 'exp_floor':
            return self._exp_floor(sigma, T)[1]
        if not self.use_negative_entropy:
            return np.zeros_like(sigma)
        form = str(getattr(self, 'entropy_stress_form', 'affine')).lower()
        if form == 'meyer_neldel':
            # Entropy-enthalpy compensation: S* = sign * H*(sigma,T) / T_MN.
            # Bounded by construction (tracks H, which collapses under stress).
            H_J = self.H(sigma, T)
            T_MN = max(float(getattr(self, 'S_MN_T_MN_K', 6500.0)), 1e-9)
            sgn = float(getattr(self, 'S_MN_sign', 1.0))
            return sgn * H_J / T_MN * np.ones_like(sigma)
        if form == 'physical':
            # S_T(T): saturating polynomial baseline (experimental W data).
            S_T_kB = (self.S_T_c0_kB + self.S_T_c1_kB_per_K * T
                      + self.S_T_c2_kB_per_K2 * T * T)
            S_T_kB = np.clip(S_T_kB, self.S_T_min_kB, self.S_T_max_kB)
            # S_sigma(sigma): Schoeck thermoelastic term, zero at sigma=0,
            # more negative under load (added to baseline, not replacing).
            n = max(float(getattr(self, 'entropy_gate_power', 1.0)), 1e-6)
            x = np.power(np.abs(sigma) / self.sigma0_S, n)
            gate = x / (1.0 + x)
            S_sigma_kB = -self.S_sigma_max_kB * gate
            return (S_T_kB + S_sigma_kB) * KB * np.ones_like(sigma)
        if form == 'gated':
            n = max(float(getattr(self, 'entropy_gate_power', 1.0)), 1e-6)
            x = np.power(np.abs(sigma) / self.sigma0_S, n)
            gate = x / (1.0 + x)
            return -self.S0_neg_J_per_K * gate
        return -self.S0_neg_J_per_K * (1 + np.abs(sigma) / self.sigma0_S)

    def dG_dsigma_numeric(self, sigma: np.ndarray | float, T: float = 0.0,
                          b: float = 2.74e-10) -> np.ndarray:
        """Numerical derivative dG*/dsigma [J/Pa] at fixed T.

        This is used only for diagnostics: v* = -dG*/dsigma and the
        fatigue-paper stationarity audit dG*/dsigma ~ 0.  It intentionally
        differentiates the same free-energy surface used by the rate law,
        including EXP-floor and any monotone envelope.
        """
        sig = np.abs(np.asarray(sigma, dtype=float))
        out = np.empty_like(sig, dtype=float)
        for idx, val in np.ndenumerate(sig):
            h = max(1.0e5, 1.0e-5 * max(float(val), 1.0))
            sm = max(float(val) - h, 0.0)
            sp = float(val) + h
            Gp = float(self.G_barrier(np.array([sp]), T, b)[0])
            Gm = float(self.G_barrier(np.array([sm]), T, b)[0])
            denom = max(sp - sm, 1.0)
            out[idx] = (Gp - Gm) / denom
        return out

    def diagnostics(self, sigma: np.ndarray | float, T: float = 0.0,
                    b: float = 2.74e-10) -> dict:
        """Free-energy barrier diagnostics at fixed stress and temperature."""
        sig = np.asarray(sigma, dtype=float)
        G = self.G_barrier(sig, T, b)
        S = self.S(sig, T)
        dGds = self.dG_dsigma_numeric(sig, T, b)
        vstar = -dGds
        return {
            'G_eV': G / EV_TO_J,
            'S_kB': S / KB,
            'dG_dsigma_eV_per_GPa': dGds / EV_TO_J * 1.0e9,
            'vstar_b3': vstar / max(b**3, 1.0e-300),
        }

    def G_barrier(self, sigma: np.ndarray, T: float = 0.0, b: float = 2.74e-10) -> np.ndarray:
        """Free energy barrier G* = H - T*S - sigma*v [J].

        MONOTONICITY GUARD (default on): the raw functional form is
        non-monotonic in stress -- the work term sigma*v(sigma) =
        v0*sigma/(1+(sigma/sigma0_v)^1.5) peaks near sigma0_v and then decays
        ~ sigma^(-1/2) while H saturates, so G* has a minimum (~0.25 eV near
        ~8 GPa for defaults) and RISES again at higher stress.  Physically a
        higher tip stress cannot make cleavage harder; the rise is an
        extrapolation artifact beyond the fitted stress window.  We therefore
        return the running-min envelope, implemented by clamping the
        evaluation stress at the barrier's argmin sigma*:
            G*_eff(sigma) = G*( min(|sigma|, sigma*) ).
        Below sigma* this is bit-identical to the raw form.  Set
        monotone_stress=False to recover the raw (non-monotone) barrier.
        """
        sigma = np.abs(np.asarray(sigma, dtype=float))
        if str(getattr(self, 'barrier_kind', 'classic')) == 'exp_floor':
            # experimental nanopillar barrier used directly (already monotone
            # decreasing in stress, so no argmin guard needed)
            return self._exp_floor(sigma, T)[0]
        if bool(getattr(self, 'monotone_stress', True)):
            s_star = self._sigma_barrier_argmin(T, b)
            sigma = np.minimum(sigma, s_star)
        return np.maximum(
            self.H(sigma, T) - T * self.S(sigma, T) - sigma * self.v(sigma, T, b),
            0.0
        )

    def _sigma_barrier_argmin(self, T: float, b: float) -> float:
        """Stress at which the raw barrier G*(sigma) is minimal (cached per T,b)."""
        key = (round(float(T), 3), float(b),
               self.H0_eV, self.sigma0_H_GPa, self.v0_b3, self.sigma0_v_GPa,
               self.S0_neg_kB, self.sigma0_S_GPa, self.use_negative_entropy,
               str(getattr(self, 'entropy_stress_form', 'affine')),
               float(getattr(self, 'entropy_gate_power', 1.0)),
               self.S_T_c0_kB, self.S_T_c1_kB_per_K, self.S_T_c2_kB_per_K2,
               self.S_T_min_kB, self.S_T_max_kB, self.S_sigma_max_kB,
               float(getattr(self, 'S_MN_T_MN_K', 6500.0)),
               float(getattr(self, 'S_MN_sign', 1.0)))
        cache = self.__dict__.setdefault('_argmin_cache', {})
        if key not in cache:
            sg = np.linspace(0.0, 100e9, 4001)
            Gg = (self.H(sg, T) - T * self.S(sg, T) - sg * self.v(sg, T, b))
            cache[key] = float(sg[int(np.argmin(Gg))])
        return cache[key]


@dataclass
class LoadingConfig:
    """Quasi-static loading parameters."""
    dt: float = 1e-5           # pseudo-time step [s]
    n_steps: int = 1000        # maximum load steps
    n_stagger: int = 3         # stagger iterations per step
    dU_top: float = 1e-6       # displacement increment per step [m]
    snap_every: int = 5        # snapshot frequency


@dataclass
class AutoStopConfig:
    """Automatic stopping criteria."""
    enabled: bool = True
    min_step: int = 5          # don't stop before this step
    drop_factor: float = 0.05  # "near zero" threshold
    n_quiet_required: int = 3  # consecutive near-zero steps to trigger


@dataclass
class FractureResistanceConfig:
    """Intrinsic fracture-resistance regularization parameters."""
    Gc_baseline: float = 1.0   # baseline fracture energy [J/m²]
    Gc0_athermal: float = 7.0  # athermal cleavage energy 2*gamma_s [J/m²]
    K_floor: float = 0.2e6     # minimum toughness [Pa*sqrt(m)]
    regularization: Literal['floor_and_ceiling', 'floor_only', 'none'] = 'floor_and_ceiling'


@dataclass
class DislocationConfig:
    """Dislocation density evolution parameters.

    Uses Variational_form-consistent coefficients:
        drho/dt = (k_store/b)*sqrt(rho)*dot_ep - k_dyn*rho*dot_ep - gamma_static
    """
    rho0: float = 5e12             # initial density [m^-2]
    rho_cap: float = 1e20          # overflow ceiling only; production PT kinetics must remain far below it
    k_store: float = np.sqrt(2)    # storage coefficient
    k_dyn: float = 1.0             # dynamic recovery coefficient
    dot_ep_max: float = 1e3        # plastic strain rate cap [1/s] for explicit quasi-static update

    # Plastic update mode.  The original explicit_rate mode integrates the
    # Arrhenius rate over the pseudo-time increment dt.  That is useful for
    # creep tests, but in a quasi-static fracture calculation it can make the
    # whole specimen creep during each load step.  The flow_stress mode instead
    # inverts the Arrhenius-Taylor law at a reference imposed strain rate and
    # performs a bounded radial return only when seq exceeds that rate-dependent
    # flow stress.
    plastic_update_mode: Literal['explicit_rate', 'flow_stress'] = 'flow_stress'
    flow_epsdot_ref: float = 1e-5
    taylor_athermal_alpha: float = 0.2  # athermal Taylor floor sigma_T >= alpha*G*b*sqrt(rho); 0 disables

    # Thermodynamic coupling for plastic/fracture competition.
    #   off:       legacy kinetic/radial-return update.
    #   onsager:   continuous dissipative flow driven by the thermodynamic
    #              overstress; the Arrhenius rate sets the mobility and the
    #              update cannot overshoot the local elastic relaxation distance.
    #   time_cone: deterministic hazard/time-cone update; the event clock
    #              advances only inside the thermodynamically admissible cone
    #              (positive overstress and non-negative local dissipation).
    # Both thermodynamic modes use the same free-energy admissibility distance,
    # so differences are due to continuous Onsager flow vs event-clock kinetics,
    # not arbitrary work caps.
    thermo_consistency_mode: Literal['off', 'onsager', 'time_cone'] = 'off'
    thermo_event_strain: float = 1e-4      # characteristic equivalent plastic strain per hazard event
    thermo_onsager_max_fraction: float = 1.0  # max fraction of local relaxation distance per update
    thermo_use_avg_stress_work: bool = True   # compute Wp from accepted stress path, not requested pre-return stress

    # Adaptive kinetic substepping.  These limits are not empirical gates on
    # the final state; they are rollback criteria for resolving a stiff
    # Arrhenius/Onsager hazard step.  If a trial substep exceeds the requested
    # hazard or equivalent-strain increment, the kinetic clock is subcycled.
    thermo_adaptive_substepping: bool = False
    thermo_max_substeps: int = 64
    thermo_max_dep_increment: float = 5.0e-5
    thermo_max_hazard_increment: float = 0.25

    # Full incremental energy audit.  This does not by itself cap plasticity;
    # it reports whether the coupled update closes the thermodynamic balance.
    thermo_energy_audit: bool = True
    thermo_energy_abs_tol: float = 1.0e-12
    thermo_energy_rel_tol: float = 0.05

    # Numerical/physical regularization for the continuum implementation of
    # the Arrhenius-Taylor hazard.  The raw microscopic factor phi=delta/b can
    # be hundreds for rho~1e12 m^-2, which collapses the barrier everywhere in
    # a continuum FE cell.  Cap it to represent a finite local amplification
    # averaged over the element/process zone.
    phi_plastic_max: float = 20.0

    # Additive lattice/Peierls resistance branch, treated as a stress floor.
    # This is the missing low-temperature plasticity barrier: Taylor depinning
    # alone is too soft and makes the whole body flow before the crack-tip
    # competition can develop.
    use_peierls_floor: bool = True
    peierls_H0_eV: float = 1.7
    peierls_v0_b3: float = 5.0
    peierls_S_kB: float = 0.0
    peierls_epsdot_ref: float = 1e-5
    # --- Startup auto-calibration of the additive (rho-independent, phi=1)
    # Peierls floor for the system of interest.  When enabled, peierls_H0_eV is
    # solved so the floor sigma_Peierls(T_cal) equals peierls_floor_min_MPa at
    # the calibration temperature (default: the hottest requested T), holding the
    # entropy peierls_S_kB at a PHYSICAL value.  The natural scale is
    # S ~ -k*ln(eta0/epsdot_ref) ~ -37 kB (the athermal-cancellation point);
    # values in [-37,-10] kB give a floor that decreases with T (physical Peierls
    # collapse) yet never drops below the regularizing minimum.  This replaces the
    # hand-tuned peierls_H0_eV with a per-material calculation done once at start.
    peierls_autocalibrate: bool = False
    peierls_floor_min_MPa: float = 1.0   # target floor at the hottest operating T
    peierls_cal_T_K: float = 0.0         # 0 -> use max requested temperature
    # --- Correlated multi-hit Taylor renewal (fixes high-density softening) ---
    # At small forest spacing, forest junctions inside one correlation length are
    # not independent strain sources; a correlated segment must complete m(rho)
    # cooperative depinning hits per renewal time t_c before it glides.  This
    # replaces the independent-site N_site*h1 hazard (whose rho^2 attempt-rate
    # prefactor causes an unphysical high-rho softening) with a Poisson-tail
    # completion probability.  m=1 / low-rho recovers the independent model.
    taylor_multihit: bool = False
    taylor_corr_rho_c: float = 1e14      # density where forest spacing ~ correlation length (n_c=1)
    taylor_renewal_time_s: float = 1e-9  # correlated-segment renewal time t_c [s]
    taylor_m_max: float = 5.0            # max cooperative hit number at high density
    taylor_m_exponent: float = 1.0       # sharpness p of m(rho) crossover

    # --- Production emission-derived Peierls--Taylor kinetics (v9.3) -----
    # The bulk FEM and moving process zone use the crack-tip emission
    # EXP-floor surface as their parent free-energy landscape. Peierls glide
    # and Taylor depinning are scaled descendants and act as sequential rates.
    # No additive Peierls/Taylor flow stress or athermal Taylor floor is used.
    bulk_kinetics_model: str = 'emission_derived_peierls_taylor_multihit'

    # Parent emission EXP-floor surface. The 2-D sharp-front driver overwrites
    # these values from the active --emit-* material row before plasticity runs.
    pt_emit_G00_eV: float = 1.94
    pt_emit_gT_eV_per_K: float = 0.003934
    pt_emit_sigc0_Pa: float = 2.298e9
    pt_emit_sT_Pa_per_K: float = -6.564e5
    pt_emit_Tref_K: float = 481.33
    pt_emit_exp_a: float = 0.0845685
    pt_emit_exp_n: float = 1.0
    pt_emit_floor_frac: float = 0.02
    pt_emit_floor_min_eV: float = 1.0e-4
    pt_emit_floor_max_frac: float = 0.95

    # Production relative scales selected from the prior Arrhenius chain.
    # The active emission surface is scale 1.0; do not reapply the historical
    # 0.75 export factor to an already-effective emission candidate.
    pt_peierls_energy_ratio: float = 0.005
    pt_peierls_entropy_ratio: float = 0.005
    pt_peierls_stress_ratio: float = 1.0
    pt_peierls_nu0_s: float = 1.0e12
    pt_taylor_energy_ratio: float = 0.02
    pt_taylor_entropy_ratio: float = 0.02
    pt_taylor_stress_ratio: float = 1.0
    pt_taylor_nu0_s: float = 1.0e11

    # Correlated Taylor completion. The hit order grows with forest density.
    # pt_taylor_m_cap is the finite obstacle count in one correlation domain,
    # not a total-density cap; infinity leaves the cooperative order uncapped.
    pt_taylor_corr_rho_c: float = 1.0e14
    pt_taylor_renewal_time_s: float = 1.0e-9
    pt_taylor_m_exponent: float = 1.0
    pt_taylor_m_scale: float = 1.0
    pt_taylor_m_cap: float = float('inf')

    # Separate mobile-carrier and forest-density roles. Since the current bulk
    # FEM stores one density field, a trapping partition maps forest density to
    # mobile density; the moving-PZ path already stores them separately.
    pt_mobile_fraction: float = 0.01
    pt_mobile_saturation_density_m2: float = 1.0e14
    pt_mobile_density_floor_m2: float = 1.0e6
    pt_jump_fraction: float = 1.0
    pt_jump_length_min_m: float = 2.5e-10
    pt_equivalent_strain_factor: float = 0.5773502691896258
    pt_peierls_stress_fraction: float = 0.5773502691896258
    pt_taylor_stress_fraction: float = 0.5773502691896258
    pt_taylor_phi_max: float = 20.0

    # Explicit integration stabilizers.  These are not intended to fit the
    # toughness; they keep the pseudo-time radial-return update from creating
    # unphysical rho/Wp blow-ups after crack propagation.  The strain cap is
    # comparable to the nominal applied strain increment in a quick sweep.
    max_plastic_strain_increment: float = 2.5e-4
    max_rho_relative_increment: float = 0.25

    # Diagnostic ablation toggles.  These are not calibration knobs; they
    # isolate which coupling fails first in a single-case debug run.
    enable_plasticity: bool = True
    freeze_rho: bool = False

    # --- Nucleation-source plasticity ('sources-only' picture) --------------
    # Default 'bulk': Frank-Read-style multiplication (drho_store) creates new
    # dislocation content anywhere plastic strain accrues.  In Hank's picture
    # bulk sources are weak; new content is generated ONLY at heterogeneous
    # sources (here, the crack tip, where the emission/nucleation barrier is
    # exceeded), and that content is then TRANSPORTED into the bulk by mobility
    # and recovered by the existing sinks.  The pre-existing (floor) density is
    # free to redistribute but cannot multiply.
    #   bulk_mult_frac:  multiplies drho_store (1.0=full Frank-Read, 0.0=off).
    #   tip_source_rho_per_emit: density deposited per emitted dislocation,
    #       spread over the near-tip pile-up area, applied EVERY step from the
    #       emission rate (continuous source), not only on cleavage advance.
    #       0.0 disables the continuous source (legacy advance-only deposit).
    #   rho_transport_c: mobility-scaled conservative diffusivity coefficient
    #       D = rho_transport_c * dot_ep * L_pz^2 [m^2/s]; 0.0 = no transport.
    bulk_mult_frac: float = 1.0
    tip_source_rho_per_emit: float = 0.0
    rho_transport_c: float = 0.0
    # --- finite-content (exhaustion) plasticity --------------------------
    # The mobile density cannot mediate unbounded strain: each dislocation
    # sweeps a fixed mean path L_sink to a sink (free surface / interface) and
    # is then absorbed.  A population therefore carries a STRAIN BUDGET
    #   gamma_max = rho * b * L_sink
    # before it is exhausted, after which the stress must rise again unless new
    # content is nucleated at a source.  Per increment, the content reaching
    # sinks is  d(rho) = -dgamma / (b * L_sink)  (independent of rho).  The
    # kinetic rate already dies as rho -> 0 (prefactor ~ 1/delta^4 ~ rho^2), so
    # exhaustion shuts off plastic flow by itself.  Requires lowering the rho
    # evaluation/clip floor so content can actually deplete below the legacy 1e6.
    exhaustion_enabled: bool = False     # consume mobile content per unit strain
    glide_to_sink_m: float = 1e-5        # mean glide path to a sink L_sink [m]
    mobile_rho_floor: float = 1e6        # rho eval/clip floor (lower to exhaust)
    peierls_floor_min_Pa: float = 0.0    # rho-independent athermal resistance floor [Pa]

    # Static recovery (diffusion climb)
    use_static_recovery: bool = True
    Tfrac_on: float = 0.3         # T/Tm threshold for static recovery
    # W lattice diffusivity: Dl = Dl0a*exp(-Ea/kT) + Dl0b*exp(-Eb/kT)
    Dl0a: float = 0.04e-4
    Ea_eV: float = 5.45
    Dl0b: float = 46e-4
    Eb_eV: float = 6.90
    kprime: float = 1.0           # recovery prefactor
    kpp: float = 0.5              # exponent prefactor
    gamma_cap: float = 1e26       # recovery rate cap [1/(m^2*s)]


@dataclass
class TipMemoryConfig:
    """
    Reduced crack-tip memory model.

    The state variables are intentionally limited to two interpretable fields:
      r_tip    : effective front-local tip radius (blunting/sharpening)
      z_shield : plastic/wake shielding of the excess crack-tip amplification

    The memory changes only the local crack-tip stress/energy amplification used
    by the fracture update.  It does not introduce a second fracture barrier,
    does not modify intrinsic Gc, and does not alter the sharp-front event kinetics.
    """
    enabled: bool = True
    mode: Literal['off', 'weak_stage1', 'stage1'] = 'stage1'

    # Global gain for ablations.  stage1=1, weak_stage1=0.25, off=0.
    state_gain: float = 1.0

    # r_tip bounds (relative to ell)
    rtip_min_factor: float = 0.25
    rtip_max_factor: float = 50.0

    # Tip-radius update rates.  The plastic-strain term captures geometric
    # blunting; the normalized-work term ties blunting to Wp*ell/Gc.
    blunt_per_plastic_strain: float = 0.50
    blunt_per_work: float = 0.35
    sharpen_per_damage: float = 0.35
    blunt_per_emission: float = 0.25

    # Shielding update.  The normalized-work term is usually the most robust
    # predictor of shielding, while dDamage contributes wake/branch shielding.
    shield_max: float = 0.85
    shield_from_plastic: float = 0.25
    shield_from_work: float = 0.75
    shield_from_damage: float = 0.15
    shield_from_emission: float = 0.20
    wake_length_factor: float = 8.0  # optional decay length under crack advance, in units of ell
    # Per-step relaxation of tip-memory state OUTSIDE the active front:
    # rtip -> rtip_ref and shield -> 0 at fractional rate wake_relax*(1-front_w).
    # Prevents the M_tip ratchet web (old front positions retaining their
    # amplification forever and re-firing).  0 disables (legacy ratchet).
    wake_relax: float = 0.0

    # Local crack-tip amplification.  M=1+(M_base-1)*sqrt(r_ref/r_tip)*(1-z).
    amp_min: float = 0.20
    amp_max: float = 5.00
    M_max: float = 4.0
    lambda_tip: float = 5.0
    kappa_tip_max: float = 4.0

    # Coupling of the amplified local stress to the local event drive.  Energy
    # density scales approximately as stress^2, so exponent=2 is the default.
    couple_to_damage_drive: bool = True
    drive_exponent: float = 2.0

    # Thermodynamic bookkeeping for memory.  If the memory state modifies
    # fracture driving or shielding, it must have a conjugate storage and/or
    # dissipation term.  These coefficients give a minimal quadratic
    # crack-front-local energy density scale of (Gc/ell).
    use_memory_energetics: bool = True
    memory_energy_r_coeff: float = 0.05      # 0.5*k_r*((r-r_ref)/ell)^2
    memory_energy_z_coeff: float = 0.05      # 0.5*k_z*z^2
    memory_dissipation_r_coeff: float = 0.02 # R_r*|dr|/ell
    memory_dissipation_z_coeff: float = 0.02 # R_z*|dz|


@dataclass
class ProcessZoneKineticsConfig:
    """Physical crack-tip process-zone kinetics.

    This replaces free Wp->Gc tuning as the preferred path.  Crack-tip
    dislocation emission is computed from the same Arrhenius barrier family used
    for plasticity, but with a crack-tip effective stress reduced by Taylor/
    process-zone back stress.  Emitted dislocations build a blunting/shielding
    memory state, add localized process-zone density, and may drive retained
    q_Gc only through an explicitly audited internal state.

    These are still coarse-grained sensitivity parameters, but each one has a
    physical interpretation: emission barrier scaling, dislocation mobility /
    back stress, recovery kinetics, and crack-advance erasure of memory.
    """
    enabled: bool = True

    # Tip emission hazard.  Uses plast_model.G_barrier(sigma_tip_eff,T).
    emission_enabled: bool = True
    emission_H_scale: float = 1.0          # stress argument scale for barrier evaluation
    emission_eta0: float = 1e13           # attempt frequency [1/s]
    emission_event_strain: float = 1e-4   # nominal event strain used for diagnostics
    emission_probability_cap: float = 0.25 # keeps event-clock increments resolvable

    # Shielded tip stress used by crack-growth and emission hazards.
    # The dislocation-emission branch should feel a stress-like backstress
    # associated with the process-zone density.  The crack-growth branch feels
    # an energy-release-rate shielding term from the same state, but also an
    # embrittling stored-energy release term below.
    backstress_model: Literal['sqrt_taylor', 'arrhenius_taylor', 'max'] = 'arrhenius_taylor'
    backstress_alpha: float = 0.35         # fallback alpha in alpha*G*b*sqrt(rho)
    backstress_rate_ref: float = 1e-4      # reference rate for Arrhenius-Taylor backstress inversion [1/s]
    backstress_scale: float = 1.0          # multiplier on process-zone backstress
    memory_backstress_factor: float = 0.5  # additional shielding from z_shield*sigma_tip
    min_effective_stress_frac: float = 0.0

    # Crack-extension process-zone terms.  These separate shielding/backstress
    # from stored-energy embrittlement.  Crack extension sees
    #   Gc_net = Gc0 + q_blunt - G_stored_release(rho); G_shield enters G_eff only
    # rather than a one-signed process-zone toughening term.
    crack_shielding_enabled: bool = True
    crack_shielding_coeff: float = 1.0     # G_shield ~ coeff*tau_back^2/Eprime*ell
    stored_energy_enabled: bool = True
    stored_energy_coeff: float = 0.5       # e_stored ~ coeff*G*b^2*rho*log(...) [J/m^3]
    stored_energy_release_efficiency: float = 0.25
    stored_energy_release_cap_factor: float = float('inf')  # legacy optional cap: G_release <= factor*Gc0; disabled by default
    Gc_net_floor_factor: float = 0.05      # keep local Gc positive and bounded
    # Front-localize the stored-energy embrittlement.  Stored cold-work energy
    # lowers the *net* fracture resistance only where the crack actually cuts
    # through the process zone (the connected, advancing front), NOT throughout
    # the whole dislocation cloud.  When True (default) the G_stored_release that
    # is subtracted from Gc_net is multiplied by the strict crack-front weight,
    # the same mask already applied to the crack-drive force.  Without this, a
    # broad high-rho zone collapses resistance throughout the body and produces
    # diffuse failure instead of a connected sharp crack.
    crack_stored_release_front_masked: bool = True

    # Emission / mobility / storage separation.  Tip nucleation creates an
    # emitted population.  A separate mobility hazard controls whether emitted
    # dislocations glide/escape or remain as a stored process-zone pile-up.
    # Only the retained/stored fraction contributes directly to rho_pz, shielding
    # memory, and stored-energy embrittlement.
    mobility_enabled: bool = True
    mobility_model: Literal['same_barrier', 'peierls_taylor'] = 'same_barrier'
    mobility_eta0: float = 1e12
    mobility_H_scale: float = 1.0
    mobility_probability_cap: float = 0.10
    mobility_backstress_factor: float = 1.0
    mobility_escape_fraction: float = 0.50
    storage_min_fraction: float = 0.02
    storage_max_fraction: float = 0.98
    storage_backstress_boost: float = 1.0

    # Correlated multi-hit Arrhenius/Taylor depinning.  This replaces the
    # previous default source-availability rho_sat law.  At high process-zone
    # density, an emitted/mobile dislocation must overcome multiple correlated
    # Taylor/Peierls obstacles over a finite slip/depinning path.  The effective
    # barrier is n_hits * DeltaG rather than a density cap, so the suppression is
    # temperature dependent through the Arrhenius factor and thermodynamically
    # monotone.  The length is a physical coarse-graining length for the crack-tip
    # emitted segment, not a saturation density.
    multihit_enabled: bool = True
    multihit_apply_to: Literal['emission', 'mobility', 'both', 'off'] = 'both'
    multihit_path_length_nm: float = 50.0       # emitted segment/slip path for correlated hits [nm]
    multihit_path_length_b: float = 200.0       # fallback if path_length_nm <= 0
    multihit_density_power: float = 1.0         # n ~ (L_path sqrt(rho))^power
    multihit_max_hits: int = 12                 # numerical underflow guard; does not increase rate
    multihit_min_hits: int = 1

    # Legacy source-availability / pile-up saturation.  Disabled by default
    # because rho_sat is phenomenological.  Keep only for ablation/comparison.
    source_availability_enabled: bool = False
    source_availability_model: Literal['hill'] = 'hill'
    source_rho_sat: float = 0.0                 # optional legacy process-zone capacity [m^-2]
    source_rho_sat_fraction: float = 0.20       # fallback fraction of dislocation rho_cap
    source_availability_power: float = 2.0      # Hill exponent
    source_availability_floor: float = 0.0      # residual source availability at high rho
    source_backstress_scale: float = 1.0        # extra backstress contribution in source exhaustion
    rho_source_saturation_cap_enabled: bool = False  # optional diagnostic hard cap at rho_sat

    # Emission-driven density/memory/toughening mapping.  These are the coarse
    # grain links from emitted dislocation events to process-zone state.
    rho_increment_per_event: float = 5e13  # [m^-2] added at P_emit=1 near front
    rho_emission_front_only: bool = True
    qgc_from_emission_factor: float = 0.02 # dq_Gc ~= factor*Gc0*P_emit near front
    qgc_driver: Literal['plastic_work', 'emission', 'mixed'] = 'emission'

    # Recovery and escape of the process-zone dislocation density.  This is the
    # physical alternative to simply lowering caps.  Static recovery is Arrhenius;
    # dynamic recovery is plastic-flow assisted and acts mainly inside the process zone.
    recovery_enabled: bool = True
    recovery_model: Literal['arrhenius', 'climb_diffusion'] = 'arrhenius'
    recovery_eta0: float = 1e6             # [1/s] prefactor for static recovery
    recovery_Q_eV: float = 1.4             # activation energy for recovery/climb/escape
    recovery_rho_power: float = 1.0
    dynamic_recovery_coeff: float = 2.0    # multiplier for rho*dot_ep recovery
    emission_recovery_front_only: bool = True

    # Crack advance erases/convects crack-tip memory and shielding.
    crack_advance_memory_erasure: float = 1.0



@dataclass
class CohesiveConfig:
    """Athermal cohesive DBTT branch.

    In this mode, crack opening is athermal (cohesive traction).
    Temperature dependence enters through Arrhenius plasticity,
    blunting, and tip emission competition.
    """
    enabled: bool = False
    Gc: float = 7.0                    # cohesive work [J/m²]
    strength_factor: float = 0.45      # C in sigma_coh = C*sqrt(E'*Gc/l_coh)
    length_factor: float = 0.02        # l_coh = factor * ell
    gate_width: float = 0.08           # logistic gate width
    gate_floor: float = 0.0            # residual damage drive below threshold

    # Tip emission competition (Rice-Thomson)
    use_emission: bool = False
    emit_H_factor: float = 0.55        # H_emit = factor * H_plastic
    emit_v_factor: float = 1.0
    emit_eta0: float = 1e13            # attempt frequency [1/s]
    emit_shield_max: float = 0.85
    emit_shield_floor: float = 0.10


@dataclass
class HazardConfig:
    """Dual-channel hazard mapping for toughness Kc(T)."""
    Gamma0: float = 1e6               # attempt rate [1/s]
    log_clip: float = 80              # numerical clip for log(Gamma)
    Btarget: float = 1.0              # integrated hazard target
    K0_lattice_MPa: float = 0.0       # athermal floor [MPa*sqrt(m)]
    dK_lattice_MPa: float = 0.0       # floor transition width [MPa*sqrt(m)]


@dataclass
class JIntegralConfig:
    """Domain-integral J computation parameters."""
    # Annular domain: r_inner < r < r_outer around crack tip
    r_inner_factor: float = 2.0        # r_inner = factor * ell
    r_outer_factor: float = 8.0        # r_outer = factor * ell
    # Smoothing function type
    q_type: Literal['linear', 'plateau'] = 'plateau'


@dataclass
class DiagnosticsConfig:
    """Diagnostics and output control."""
    enabled: bool = True
    damage_threshold: float = 0.95
    plastic_threshold: float = 1e-14   # dot_ep threshold for "active" [1/s]
    make_plots: bool = True
    save_fields: bool = True
    save_every: int = 5              # snapshot cadence; final step is always saved
    save_field_pngs: bool = True     # save crack/rho/Gc/M_tip snapshot images
    max_snapshot_cols: int = 4       # number of snapshots shown in PNG summary
    x_axis: Literal['Uapp', 'step'] = 'Uapp'
    # Live progress monitoring (console + atomic progress.json/progress.log in
    # the output dir) so a long run can be checked from outside the process.
    progress: bool = True
    progress_interval_s: float = 15.0   # wall-clock heartbeat cadence within a step
    progress_every: int = 1             # console step-summary cadence (steps)


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    # Temperature list
    T_list: list = field(default_factory=lambda: [500, 700, 900, 1100, 1300, 1500])

    # Sub-configs
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    material: ElasticProperties = field(default_factory=ElasticProperties)
    plasticity_barrier: PlasticityBarrier = field(default_factory=PlasticityBarrier)
    fracture_barrier: FractureBarrier = field(default_factory=FractureBarrier)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    auto_stop: AutoStopConfig = field(default_factory=AutoStopConfig)
    fracture_resistance: FractureResistanceConfig = field(default_factory=FractureResistanceConfig)
    dislocations: DislocationConfig = field(default_factory=DislocationConfig)
    tip_memory: TipMemoryConfig = field(default_factory=TipMemoryConfig)
    process_zone: ProcessZoneKineticsConfig = field(default_factory=ProcessZoneKineticsConfig)
    cohesive: CohesiveConfig = field(default_factory=CohesiveConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    j_integral: JIntegralConfig = field(default_factory=JIntegralConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Fail-fast diagnostic controls.  These stop a diagnostic run as soon as
    # it exits the physically interpretable regime, rather than wasting
    # compute after the crack has already failed or the plastic state has
    # blown up.  Enabled by CLI --stop-on-invalid.
    stop_on_invalid: bool = False
    invalid_rho_max: float = 5e17
    invalid_wp_wext_pct: float = 2000.0
    invalid_Gc_factor: float = 1.001  # stop only if local Gc exceeds the cap
    invalid_min_step: int = 3          # avoid step-1 false positives in diagnostic runs
    invalid_Wext_min: float = 1e-12    # Wp/Wext is meaningless until Wext is positive
    invalid_K_MPa: float = 100.0
    invalid_sigma_eq_GPa: float = 50.0   # fail-fast for local stress singularities
    invalid_dep_eq_increment: float = 1.0e-2  # accepted per-step plastic strain sanity limit
    invalid_d_frac: float = 0.85

    # Fracture physics mode
    # 'arrhenius_gc': Gc(T) prescribed from Arrhenius barrier (toughness is input)
    # 'emergent': constant Gc, toughness EMERGES from plasticity-fracture competition
    # 'cohesive_dbtt': cohesive traction + Arrhenius plasticity
    fracture_mode: Literal['arrhenius_gc', 'emergent', 'cohesive_dbtt'] = 'arrhenius_gc'

    # Toughness mapping method for Arrhenius Gc branch
    # 'lambertw': spinodal closure (DBTT-like: Kc increases with T when S<0)
    # 'hazard': integrated Arrhenius hazard (ceramic-like: Kc decreases with T)
    toughness_method: Literal['lambertw', 'hazard'] = 'lambertw'

    # Output
    save_to_disk: bool = True
    output_prefix: str = 'hist_pf_arrhenius_W'
    output_dir: str = 'results_arrhenius'


def make_dbtt_config() -> SimulationConfig:
    """Preset: DBTT-like behavior (Kc increases with T).

    Uses Lambert-W spinodal mapping, where negative entropy makes
    the barrier (and hence Kc) grow with temperature.
    Regularization is 'none' so the Lambert-W trend is not clipped.
    """
    cfg = SimulationConfig()
    cfg.fracture_mode = 'arrhenius_gc'
    cfg.toughness_method = 'lambertw'
    cfg.fracture_barrier = FractureBarrier(
        H0_eV=2.0, sigma0_H_GPa=3.0,
        v0_b3=2.0, sigma0_v_GPa=3.0,
        S0_neg_kB=3.0, sigma0_S_GPa=2.0,
        use_negative_entropy=True,
    )
    cfg.fracture_resistance.regularization = 'none'
    return cfg


def make_ceramic_config() -> SimulationConfig:
    """Preset: ceramic-like (Kc weakly decreasing or constant with T).

    Uses hazard mapping, where the Arrhenius rate always increases with T
    → lower Kc at higher T. Floor regularization prevents collapse.
    """
    cfg = SimulationConfig()
    cfg.fracture_mode = 'arrhenius_gc'
    cfg.toughness_method = 'hazard'
    cfg.fracture_barrier = FractureBarrier(
        H0_eV=5.0, sigma0_H_GPa=10.0,
        v0_b3=1.0, sigma0_v_GPa=10.0,
        S0_neg_kB=1.0, sigma0_S_GPa=5.0,
        use_negative_entropy=False,
    )
    cfg.fracture_resistance.regularization = 'floor_only'
    return cfg


def make_cohesive_dbtt_config() -> SimulationConfig:
    """Preset: athermal cohesive opening + Arrhenius plasticity for DBTT."""
    cfg = SimulationConfig()
    cfg.fracture_mode = 'cohesive_dbtt'
    cfg.cohesive = CohesiveConfig(
        enabled=True,
        Gc=7.0,
        strength_factor=0.45,
        length_factor=0.02,
        use_emission=True,
    )
    cfg.fracture_resistance.regularization = 'none'
    cfg.fracture_resistance.Gc_baseline = 7.0
    return cfg


def make_emergent_config() -> SimulationConfig:
    """Preset: emergent toughness from plasticity-fracture competition.

    Physics:
        Gc is CONSTANT at the athermal cleavage energy 2*gamma_s.
        Temperature enters ONLY through the Arrhenius-Taylor plasticity model
        (Mirzaei & Dillon), where the flow stress is:

            sigma = (2b*sqrt(rho)/v*) * [A + kT*ln(eps_dot/(16*rho^2*b^4))]

        This naturally produces:
            - Taylor hardening (sigma ~ sqrt(rho)) at moderate rho
            - Peak stress at critical rho (where bracket = 4kT)
            - Strain softening beyond peak (log term dominates)

        The softening onset depends on T through A/(kT):
            High A/(kT):  peak at high rho → wide hardening regime
            Low A/(kT):   peak at low rho → early softening

    The fracture behavior regime depends on H0 and v*:

        H0 (barrier height at sig0; H(0) = H0/(1-chi)):
            H0 >> kBT_max  →  always brittle, Kc ≈ Kc_intrinsic
            H0 ~ kBT_DBTT  →  DBTT (Kc increases with T, then softening at high T)
            H0 << kBT_min  →  always ductile (plastic fracture at all T)

        v* (activation volume):
            Small v*  →  high flow stress → small plastic zone → sharp DBTT
            Large v*  →  low flow stress → large plastic zone → broad transition

    To sweep parameters:
        python -m arrhenius_fracture.sweep --vary H0
        python -m arrhenius_fracture.sweep --vary v_star
        python -m arrhenius_fracture.sweep --vary both
    """
    cfg = SimulationConfig()
    cfg.fracture_mode = 'emergent'
    cfg.T_list = [300, 500, 700, 900, 1100]

    # Dislocation depinning barrier (eq 13 from Mirzaei & Dillon)
    # NOT the Peierls/kink-pair barrier (which is additive, eq 16)
    # A ~ 0.1-0.5 eV, v* ~ 1-3 b^3 for forest depinning
    # These produce alpha ~ A/(2*v*mu) ~ 0.1-0.2
    cfg.plasticity_barrier.H0_J = 0.09 * EV_TO_J    # H at sig0 → H(0) = 0.3 eV
    cfg.plasticity_barrier.v0_c = 0.3 * (2.74e-10)**3  # v at sig0 → v(0) = 1 b^3
    cfg.plasticity_barrier.sig0 = 2e9
    cfg.plasticity_barrier.chiH = 0.70
    cfg.plasticity_barrier.psiV = 0.70

    # Constant intrinsic fracture energy (W cleavage: 2*gamma_s ≈ 7 J/m²)
    cfg.fracture_resistance.Gc_baseline = 7.0
    cfg.fracture_resistance.Gc0_athermal = 7.0
    cfg.fracture_resistance.regularization = 'none'

    # Quasi-static loading: dt * dU must give strain rate << plastic flow rate
    # Max plastic flow rate: eta0*(b/delta)^4 ~ 0.14/s at rho0=5e12
    # Loading velocity: dU/dt = 5e-8 m/s → strain rate ~ 2.5e-5/s
    # Ratio: flow/loading ~ 5600 → plasticity can fully equilibrate
    cfg.loading.dt = 1.0           # 1 s per step (quasi-static)
    cfg.loading.dU_top = 5e-8      # 50 nm per step
    cfg.loading.n_steps = 400      # up to 20 µm total opening
    cfg.loading.n_stagger = 4      # more stagger iterations for coupling

    # Finer mesh for process zone resolution
    cfg.mesh.nx = 80
    cfg.mesh.ny = 160
    cfg.mesh.ell_factor = 4.0

    # Auto-stop: look for significant force drop
    cfg.auto_stop.enabled = True
    cfg.auto_stop.min_step = 20    # don't stop too early
    cfg.auto_stop.drop_factor = 0.10
    cfg.auto_stop.n_quiet_required = 5

    # Diagnostics: snapshot every 10 steps
    cfg.diagnostics.save_every = 10
    cfg.diagnostics.save_fields = True

    # Regularized plasticity: include the missing Peierls/lattice branch and
    # cap the element-averaged Taylor stress amplification.
    cfg.dislocations.use_peierls_floor = True
    cfg.dislocations.peierls_H0_eV = 1.7
    cfg.dislocations.peierls_v0_b3 = 5.0
    cfg.dislocations.phi_plastic_max = 20.0

    # Emergent plastic shielding should be conservative because the explicit
    # plastic update overestimates Wp before stress relaxation.

    # Stabilize damage evolution so the crack does not jump through the
    # entire J-integral annulus in one pseudo-time increment.

    # Tip memory: enable to see blunting/sharpening effects
    cfg.tip_memory.enabled = True

    return cfg
