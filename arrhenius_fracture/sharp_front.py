"""
Sharp-front dual-hazard fracture driver.

This module REPLACES the AT2 <-> hazard coupling layer (source mode, Griffith
license, consume halo, frontier gates, fired-memory relief) with a single
crack-advance law.  The phase-field d is retained ONLY as a stiffness-kill
indicator for broken material; it never evolves variationally and there is no
second fracture criterion to reconcile.

Architecture
------------
The crack is a sharp front on the y=0 ligament.  One engine, two drivers:

  1D : K(t) is a prescribed ramp.  No FEM.  Validates the DBTT crossover of
       the two hazards in milliseconds; this is the Lambert-W/renewal model
       in its native form.
  2D : the existing elastic-plastic FEM supplies K via the J-integral (the
       mesh-objective load measure) and relaxes the far field plastically.
       The SAME engine consumes K_J(t).  When the cleavage clock completes,
       the front advances one ligament increment (stiffness kill); nothing
       else can advance the crack.

Single-front physics (the intended divergence, with explicit signs)
-------------------------------------------------------------------
State per front: N_em (emitted-dislocation ledger), B (cleavage first-passage
action), r_eff (blunted tip radius).

  sigma_tip = K / sqrt(2*pi*r_eff)          analytic; never de-smeared FEM
  r_eff     = r0 + c_blunt * b * N_em       emission blunts the tip       (-)
  sigma_back= beta * G*b/(2*pi*(1-nu)*Lpz) * N_em   pile-up back stress,
              LINEAR in N_em: suppresses further EMISSION only          (-)
  e_stored  = 0.5*G*b^2 * (rho0 + N_em/Lpz^2)
  dG_emb    = e_stored * v_emb              stored PZ energy LOWERS the
              CLEAVAGE barrier (embrittlement)                          (+)

  lambda_e  = nu0_e * exp(-G*_e(sigma_tip - sigma_back, T)/kT)
  lambda_c  = nu0_c * exp(-(G*_f(sigma_tip, T) - dG_emb)/kT)
  multi-hit renewal: lambda_eff = gammainc(m, lambda_c*tau_c)/tau_c with a
              REAL correlation window tau_c (not dt)
  B        += lambda_eff * dt ;  front advances when B >= 1

So plastic-zone development makes emission monotonically HARDER (back stress,
linear in N) and cleavage monotonically EASIER (embrittlement) while blunting
provides the transient shielding.  The two channels diverge; the system
cannot deadlock: at fixed K, emission self-arrests and the cleavage clock
then completes in finite time.  DBTT = crossover of which hazard runs first
under a given K-ramp.

On advance the crack enters fresh material: the tip re-sharpens
(r_eff -> r0), a wake fraction of the ledger is left behind
(N_em *= wake_retain), and B resets (renewal).

Rate-shelf co-design (the lesson from the old runs): the engine audits
lambda_max = nu0 * exp(S/kB at the relevant stress) against the loading
timescale and REFUSES silently shelved clocks -- it prints the shelf and the
minimum time-to-fire so a dead clock is a reported condition, not a mystery.

Usage
-----
  # 1D validation sweep (seconds):
  python -m arrhenius_fracture.sharp_front --mode 1d \
      --temperatures 300 400 500 600 700 800 900 --Kdot 0.02 --out runs/sf1d

  # 2D FEM-coupled run:
  python -m arrhenius_fracture.sharp_front --mode 2d \
      --temperatures 300 700 --steps 120 --nx 60 --ny 120 --out runs/sf2d
"""

from __future__ import annotations

import argparse
import json
import os
import copy
import numpy as np

from .config import (SimulationConfig, FractureBarrier, make_emergent_config,
                     KB, EV_TO_J)
from .materials import PlasticityModel
from .material_manifest import MaterialManifest, default_manifest_path
from .unified_mpz import MPZConfig
from .unified_front import UnifiedMPZFrontEngine


def _write_run_args(args, out, extra=None):
    """Persist the exact CLI/config state that produced an output directory.

    This is intentionally lightweight and JSON-only; it makes branch/no-branch
    sweeps auditable without scraping terminal logs.
    """
    try:
        os.makedirs(out, exist_ok=True)
        payload = dict(vars(args))
        if extra:
            payload.update(extra)
        with open(os.path.join(out, 'run_args.json'), 'w') as fp:
            json.dump(payload, fp, indent=2, sort_keys=True, default=str)
    except Exception:
        # run arguments are diagnostic only; never abort a physics run because
        # the audit file could not be written.
        pass


# ----------------------------------------------------------------------------
# Engine configuration
# ----------------------------------------------------------------------------

class FrontConfig:
    """Parameters of the sharp-front dual-hazard engine.

    All physics knobs of the advance law live here -- there are no gates,
    licenses, or consumption channels anywhere else.
    """

    def __init__(self):
        # --- geometry of the analytic tip field ---
        self.r0 = 1.0e-6            # sharp process-zone radius [m] (sigma_tip = K/sqrt(2 pi r))
        self.sigma_cap = 30.0e9     # cohesive ceiling on sigma_tip [Pa]; <=0 disables

        # --- cleavage hazard ---
        self.nu0_c = 1.0e12         # attempt frequency [1/s]
        self.m_hits = 3.0           # cooperative bonds per advance increment
        self.tau_c = 1.0e-6         # REAL correlation window [s] (never dt)
        self.tau_B = 0.0            # optional sub-critical anneal time [s]; 0 = pure first passage

        # --- emission hazard ---
        self.nu0_e = 1.0e11         # emission attempt frequency [1/s]
        self.dN_cap = float('inf')  # optional diagnostic cap on emitted dislocations per dt; inf = off

        # --- ledger couplings (signs are the model) ---
        self.c_blunt = 1.0          # tip-radius gain per emitted b   (shields both, transiently)
        self.beta_back = 1.0        # pile-up back-stress coefficient (suppresses EMISSION, linear in N)
        self.L_pz = 1.0e-6          # pile-up / process-zone length [m]
        self.v_emb_b3 = 500.0       # embrittlement release volume [b^3] (lowers CLEAVAGE barrier)
        self.emb_sat_frac = 1.0     # embrittlement saturation: dG_emb capped at
                                    # emb_sat_frac * (shielded cleavage barrier).
                                    # 1.0 -> uncapped (as shipped: embrittlement can
                                    # zero the barrier, so it always wins the high-T
                                    # limit and the upper shelf collapses -> ceramic).
                                    # <1 -> embrittlement saturates BELOW the barrier,
                                    # giving shielding the dynamic range to hold a tough
                                    # upper shelf (DBTT) at large chi_shield.
        self.rho0 = 5.0e12          # background density [1/m^2]
        # --- dynamic recovery / saturation of the emitted-dislocation ledger ---
        # As shipped N_em only GROWS (emission), with no sink, so dG_emb and
        # sigma_back both run away at high T and embrittlement always wins the
        # high-T limit.  That is a MISSING-PHYSICS artifact, not a result.  These
        # two optional terms restore a physical ceiling so both channels saturate
        # together from the same mechanism (no per-barrier cap needed):
        #   production *= max(1 - N_em/N_sat, 0)      (source exhaustion / max storable rho)
        #   annihilation = recover_k * N_em * dt       (dynamic recovery, Kocks-Mecking-like)
        # Defaults (inf, 0) reproduce the as-shipped unbounded-growth behavior.
        self.N_sat = float('inf')   # saturation density of the ledger [count]; inf = off
        self.recover_k = 0.0        # linear recovery/annihilation rate [1/s]; 0 = off
        self.wake_retain = 0.3      # ledger fraction left in the wake on advance
        self.k_shield = 0.0         # optional direct K-shielding per dislocation (default off:
                                    # blunting already carries the shielding; keep one channel)
        self.chi_shield = 0.0       # back-stress shielding of the CLEAVAGE driving stress:
                                    # sigma_eff_cleave = sigma_tip - chi_shield*sigma_back.
                                    # This is the sigma -> sigma - sigma_back substitution of the
                                    # introduction, applied to the crack-opening hazard.  It is
                                    # LINEAR in N_em (via sigma_back), so it competes head-to-head
                                    # with the linear embrittlement dG_emb (via v_emb_b3).  The
                                    # ratio chi_shield <-> v_emb_b3 is the regime axis:
                                    #   chi_shield = 0            -> embrittlement wins (ceramic, dKc/dT<0)
                                    #   chi_shield ~ balance      -> weak-T (dKc/dT~0)
                                    #   chi_shield large          -> shielding wins (DBTT, dKc/dT>0)
                                    # Default 0 reproduces the as-shipped (embrittlement-only) model.

        # --- advance increment (1D bookkeeping; 2D uses the mesh spacing) ---
        self.da = 2.0e-5            # [m]
        # Backward-compatible placeholder.  v3 does NOT cap crack velocity;
        # unstable advance is resolved numerically by adaptive event stepping.
        self.v_rayleigh = float('inf')


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------

class FrontEngine:
    """Single-criterion sharp-front state machine.

    step(K, T, dt) integrates both hazards over dt at stress intensity K and
    returns a dict of rates/state.  The ONLY way the crack advances is the
    first-passage completion of the cleavage clock (returned as 'fired').
    """

    def __init__(self, fcfg: FrontConfig, cleave_barrier: FractureBarrier,
                 emit_barrier: FractureBarrier, G_shear: float, nu: float,
                 b: float):
        self.f = fcfg
        self.cb = cleave_barrier
        self.eb = emit_barrier
        self.G = float(G_shear)
        self.nu = float(nu)
        self.b = float(b)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self):
        self.N_em = 0.0
        self.B = 0.0
        self.a_adv = 0.0          # cumulative advance [m]
        self.n_adv = 0
        self.W_emit = 0.0         # emission dissipation ledger [J/m]
        self.t = 0.0
        self.K_prev = None        # previous-step K, for analytic hazard integ.

    def clone_split(self, daughter_fraction=0.5):
        """Create a daughter front by conserving/splitting the tip ledger.

        Branching creates two physical tips from one parent tip.  Cloning the full
        N_em/B/W_emit state into both tips would double the stored history;
        resetting the daughter to zero erases the pre-branch plastic/shielding
        state.  The compromise used here is conservative: split the scalar
        renewal/plastic ledgers between parent and daughter according to the
        competing-hazard branch fraction.  Geometric advance counters are local
        to each front, so the daughter starts with zero advance.
        """
        frac = float(np.clip(daughter_fraction, 0.0, 1.0))
        child = copy.deepcopy(self)

        N0 = float(self.N_em); B0 = float(self.B); W0 = float(self.W_emit)
        child.N_em = N0 * frac
        child.B = B0 * frac
        child.W_emit = W0 * frac
        child.a_adv = 0.0
        child.n_adv = 0
        child.K_prev = self.K_prev
        child.t = self.t

        keep = 1.0 - frac
        self.N_em = N0 * keep
        self.B = B0 * keep
        self.W_emit = W0 * keep
        return child

    # -- derived fields --------------------------------------------------------
    def r_eff(self):
        return self.f.r0 + self.f.c_blunt * self.b * self.N_em

    def sigma_tip(self, K):
        K_eff = max(K - self.f.k_shield * self.N_em * self.G * self.b
                    / np.sqrt(2.0 * np.pi * self.f.L_pz), 0.0)
        s = K_eff / np.sqrt(2.0 * np.pi * self.r_eff())
        if self.f.sigma_cap > 0:
            s = min(s, self.f.sigma_cap)
        return s

    def sigma_back(self):
        return (self.f.beta_back * self.G * self.b
                / (2.0 * np.pi * (1.0 - self.nu) * self.f.L_pz)) * self.N_em

    def e_stored(self):
        rho = self.f.rho0 + self.N_em / (self.f.L_pz ** 2)
        return 0.5 * self.G * self.b ** 2 * rho

    def dG_emb(self):
        return self.e_stored() * self.f.v_emb_b3 * self.b ** 3

    # -- hazards ----------------------------------------------------------------
    def lambda_emit(self, sig_tip, T):
        s_eff = max(sig_tip - self.sigma_back(), 0.0)
        Gstar = float(self.eb.G_barrier(np.array([s_eff]), T, self.b)[0])
        x = -Gstar / max(KB * T, 1e-30)
        return self.f.nu0_e * np.exp(np.clip(x, -700.0, 0.0)), s_eff, Gstar

    def lambda_cleave(self, sig_tip, T):
        # Shielding enters the crack-opening hazard as the introduction's
        # sigma -> sigma - sigma_back substitution: the crack-tip pile-up back
        # stress opposes the local driving stress for bond rupture.  chi_shield
        # scales how much of the (linear-in-N_em) back stress shields cleavage.
        sig_eff = max(sig_tip - self.f.chi_shield * self.sigma_back(), 0.0)
        Gstar = float(self.cb.G_barrier(np.array([sig_eff]), T, self.b)[0])
        dGe = min(self.dG_emb(), self.f.emb_sat_frac * Gstar)
        Geff = max(Gstar - dGe, 0.0)
        x = -Geff / max(KB * T, 1e-30)
        lam_raw = self.f.nu0_c * np.exp(np.clip(x, -700.0, 0.0))
        m = max(self.f.m_hits, 1.0)
        if m > 1.0 + 1e-12:
            from scipy.special import gammainc
            tau = max(self.f.tau_c, 1e-30)
            lam = gammainc(m, min(lam_raw * tau, 1e12)) / tau
        else:
            lam = lam_raw
        return lam, lam_raw, Geff

    def cleavage_diagnostics(self, sig_tip, T):
        """Return raw/effective cleavage free-energy diagnostics at sig_tip."""
        sig_eff = max(float(sig_tip) - self.f.chi_shield * self.sigma_back(), 0.0)
        d = self.cb.diagnostics(np.array([sig_eff]), T, self.b)
        Gstar = float(d['G_eV'][0] * EV_TO_J)
        dGe = min(self.dG_emb(), self.f.emb_sat_frac * Gstar)
        Geff = max(Gstar - dGe, 0.0)
        return {
            'sigma_cleave_eff_Pa': float(sig_eff),
            'G_cleave_raw_eV': float(Gstar / EV_TO_J),
            'G_cleave_eff_eV': float(Geff / EV_TO_J),
            'S_cleave_kB': float(d['S_kB'][0]),
            'dGcleave_dsigma_eV_per_GPa': float(d['dG_dsigma_eV_per_GPa'][0]),
            'vstar_cleave_b3': float(d['vstar_b3'][0]),
            'cleave_barrier_kind_code': 1.0 if str(getattr(self.cb, 'barrier_kind', 'classic')) == 'exp_floor' else 0.0,
        }

    def predict_clock_increment(self, K, T, dt):
        """Estimate the *incremental* cleavage-clock advance for timestep control.

        This returns only the proposed-step contribution, ΔB ≈ λ_c Δt, using
        the log-mean cleavage rate between the previous accepted K and the
        trial K.  It deliberately does NOT include the already-accumulated
        renewal clock ``self.B``.  The adaptive controller should limit how
        much new clock is bundled into one FEM/J solve; it must not prevent the
        cumulative clock from reaching first passage.

        Returning B_old + ΔB would create an artificial arrest at the target
        value: once B approached the adaptive target, every subsequent load
        step would be rejected/shrunk and the crack could never fire.
        """
        K = float(max(K, 0.0)); dt = float(max(dt, 0.0))
        if dt <= 0.0:
            return 0.0
        sig2 = self.sigma_tip(K)
        lam_c, _, _ = self.lambda_cleave(sig2, T)
        if self.K_prev is not None and self.K_prev > 0 and K > 0:
            sig1 = self.sigma_tip(self.K_prev)
            lam_c_prev, _, _ = self.lambda_cleave(sig1, T)
            lo, hi = sorted((max(lam_c_prev, 0.0), max(lam_c, 0.0)))
            if lo <= 0.0:
                lam_eff = 0.5 * hi
            elif abs(hi - lo) <= 1e-12 * hi:
                lam_eff = hi
            else:
                lam_eff = (hi - lo) / np.log(hi / lo)
        else:
            lam_eff = max(lam_c, 0.0)
        return float(max(lam_eff * dt, 0.0))

    # -- one explicit step --------------------------------------------------------
    def step(self, K, T, dt):
        sig = self.sigma_tip(K)

        # Audit possible numerical/constitutive limiters explicitly.  These
        # diagnostics are written into the returned info so sweeps can verify that
        # no cap/floor/gate is controlling the response.
        K_eff_audit = max(K - self.f.k_shield * self.N_em * self.G * self.b
                          / np.sqrt(2.0 * np.pi * self.f.L_pz), 0.0)
        sigma_tip_uncapped = K_eff_audit / np.sqrt(2.0 * np.pi * self.r_eff())
        sigma_cap_active = bool(self.f.sigma_cap > 0 and sigma_tip_uncapped > self.f.sigma_cap)

        lam_e, sig_em_eff, Ge = self.lambda_emit(sig, T)
        prod_raw = lam_e * dt
        prod = min(prod_raw, self.f.dN_cap)
        dN_cap_active = bool(np.isfinite(self.f.dN_cap) and prod_raw > self.f.dN_cap)
        N_sat_factor = 1.0
        if np.isfinite(self.f.N_sat) and self.f.N_sat > 0.0:
            N_sat_factor = max(1.0 - self.N_em / self.f.N_sat, 0.0)
            prod *= N_sat_factor   # source exhaustion / max rho
        ann = self.f.recover_k * self.N_em * dt                 # dynamic recovery
        # each forward event dissipates ~ sigma_eff * b * Lpz per unit thickness
        self.W_emit += sig_em_eff * self.b * self.f.L_pz * prod
        self.N_em = max(self.N_em + prod - ann, 0.0)

        # cleavage clock at the (post-emission) tip state
        sig2 = self.sigma_tip(K)
        lam_c, lam_c_raw, Gc_eff = self.lambda_cleave(sig2, T)
        if self.f.tau_B > 0 and dt > 0:
            self.B *= np.exp(-min(dt / self.f.tau_B, 80.0))
        # Hazard accumulation over the step.  lam_c varies EXPONENTIALLY with
        # tip stress, and stress ramps across the step, so the forward-Euler
        # rectangle  B += lam_c(K)*dt  is first-order and SYSTEMATICALLY
        # mis-counts: as dt shrinks the answer keeps drifting (Kc rises without
        # a limit) instead of converging.  Integrate analytically instead.  For
        # K ramping linearly K_prev->K over dt, lam_c(K(t)) is ~exponential in
        # t, so  ∫lam_c dt = dt * logmean(lam_c_prev, lam_c)  where the
        # logarithmic mean (lam2-lam1)/ln(lam2/lam1) is EXACT for an
        # exponentially varying rate and ->trapezoid as the two approach.
        if self.K_prev is not None and self.K_prev > 0 and K > 0:
            sig1 = self.sigma_tip(self.K_prev)
            lam_c_prev, _, _ = self.lambda_cleave(sig1, T)
            lo, hi = sorted((max(lam_c_prev, 0.0), max(lam_c, 0.0)))
            if lo <= 0.0:
                lam_eff = hi * 0.5            # one endpoint inactive: half-weight
            elif abs(hi - lo) <= 1e-12 * hi:
                lam_eff = hi                  # ~equal: logmean -> the value
            else:
                lam_eff = (hi - lo) / np.log(hi / lo)   # logarithmic mean
        else:
            lam_eff = lam_c                   # first step: no previous K
        self.B += lam_eff * dt
        self.K_prev = K
        self.t += dt

        # Renewal COUNT, not a single-fire cap.  In time dt at rate lam_c the
        # first-passage process completes ~lam_c*dt events; once K exceeds Kc
        # this is >> 1 and the crack must run away (unstable fast fracture).
        # Advancing only once per step and discarding the excess throttles the
        # crack to da_phys/dt -- a numerical speed limit, not physics.  Advance
        # floor(B) increments and keep the fractional remainder.
        # Pre-renewal state: the tip plasticity that actually DROVE this advance.
        # Capture it before the renewal reset so diagnostics and the 2D wake
        # deposit reflect the driving state, not the post-shed remainder.
        N_em_pre = self.N_em
        sigma_back_pre = self.sigma_back()
        r_eff_pre = self.r_eff()
        dG_emb_pre = self.dG_emb()

        n_fire_available = int(np.floor(min(self.B, 1e7)))   # cap guards overflow on runaway
        max_fire = float(getattr(self.f, 'max_advances_per_step', float('inf')))
        if np.isfinite(max_fire):
            n_fire = min(n_fire_available, max(int(max_fire), 0))
        else:
            n_fire = n_fire_available
        if not np.isfinite(self.B):
            # a non-finite K (e.g. singular FEM solve) propagated into the clock;
            # refuse to advance rather than crash, and surface it to the driver.
            self.B = 0.0
            n_fire = 0
        # No velocity cap is applied.  If n_fire is large, the tip is genuinely
        # unstable at the accepted FEM/J state; the 2-D driver should resolve this
        # by adaptive event stepping and remeshing, not by limiting n_fire here.
        fired = n_fire >= 1
        v_crack = self.f.da * n_fire / dt if dt > 0 else 0.0
        N_retained = N_em_pre
        N_shed = 0.0
        if fired:
            self.B -= n_fire                       # keep remainder (renewal)
            # each advance enters fresh material and partially sheds the wake
            # ledger; wake retention compounds once per increment.  The density
            # LEFT BEHIND in the cracked material is what the tip sheds,
            # N_pre*(1 - wake_retain**n_fire) -- NOT a fraction of what it keeps.
            retain = np.clip(self.f.wake_retain, 0.0, 1.0) ** n_fire
            N_retained = N_em_pre * retain
            N_shed = N_em_pre * (1.0 - retain)
            self.N_em = N_retained
            self.a_adv += self.f.da * n_fire
            self.n_adv += n_fire

        return {
            'fired': bool(fired), 'n_fire': int(n_fire),
            'n_fire_available': int(n_fire_available), 'v_crack': v_crack,
            'sigma_tip': sig2, 'sigma_back': self.sigma_back(),
            'lambda_e': lam_e, 'lambda_c': lam_c, 'lambda_c_raw': lam_c_raw,
            'B': self.B, 'N_em': self.N_em, 'r_eff': self.r_eff(),
            'dG_emb_eV': self.dG_emb() / EV_TO_J, 'G_cleave_eff_eV': Gc_eff / EV_TO_J,
            **self.cleavage_diagnostics(sig2, T),
            'G_emit_eV': Ge / EV_TO_J, 'W_emit': self.W_emit,
            'sigma_tip_uncapped': float(sigma_tip_uncapped),
            'sigma_cap_active': bool(sigma_cap_active),
            'dN_emit_raw': float(prod_raw),
            'dN_cap_active': bool(dN_cap_active),
            'N_sat_factor': float(N_sat_factor),
            'N_sat_active': bool(np.isfinite(self.f.N_sat) and self.f.N_sat > 0.0 and N_sat_factor < 0.999999),
            # pre-renewal / wake-ledger audit (the driving state + conservation)
            'N_em_pre_renewal': N_em_pre, 'N_em_retained': N_retained,
            'N_em_shed_to_wake': N_shed,
            'sigma_back_pre_renewal': sigma_back_pre, 'r_eff_pre_renewal': r_eff_pre,
            'dG_emb_pre_renewal_eV': dG_emb_pre / EV_TO_J,
        }

    # -- rate-shelf audit -----------------------------------------------------------
    def shelf_audit(self, T, t_total):
        """Report the saturated clock rates so a dead clock is never silent."""
        scap = self.f.sigma_cap if self.f.sigma_cap > 0 else 100e9
        lam_c_max, _, _ = self.lambda_cleave(scap, T)
        lam_e_max, _, _ = self.lambda_emit(scap, T)
        t_min_fire = 1.0 / max(lam_c_max, 1e-300)
        ok = t_min_fire < t_total
        return {'lambda_c_max': float(lam_c_max), 'lambda_e_max': float(lam_e_max),
                't_min_fire_s': float(t_min_fire), 'clock_completable': bool(ok)}


# ----------------------------------------------------------------------------
# Barrier construction
# ----------------------------------------------------------------------------

def default_emission_barrier(b: float) -> FractureBarrier:
    """Stress-biased emission barrier (reuses the generic H - T*S - sigma*v form).

    Emission of a tip dislocation in W: high zero-stress barrier, weak negative
    entropy, modest activation volume.  Thermal activation makes the emission
    threshold stress drop with T much faster than the cleavage threshold (which
    is partially entropy-compensated), which is the DBTT crossover.
    """
    # Co-designed with the cleavage barrier for a DBTT crossover (gated entropy).
    # Emission is the COLD-HARD / HOT-SOFT channel: high zero-stress barrier
    # (frozen below the DBTT) plus a STRONG stress-gated negative entropy so its
    # threshold drops steeply with T.  Below the DBTT it is frozen and cleavage
    # wins (brittle); above it emission undercuts cleavage and blunts the tip
    # (ductile).  Gated entropy keeps the zero-stress barrier T-independent so
    # nothing fires thermally at zero load.
    eb = FractureBarrier()
    eb.H0_eV = 1.8
    eb.sigma0_H_GPa = 2.5
    eb.v0_b3 = 0.6
    eb.sigma0_v_GPa = 2.5
    eb.use_negative_entropy = True
    eb.entropy_stress_form = 'physical'
    eb.sigma0_S_GPa = 3.0
    # S_T(T): experimental W dislocation-glide baseline (Allera 2025 / Veverka &
    # Dillon). Illustrative polynomial: ~-25 kB cold, less negative with T,
    # saturating. The dislocation core carries ~kB per b (Schoeck eq 34), hence
    # a large baseline for the EMISSION channel.
    eb.S_T_c0_kB = -20.0
    eb.S_T_c1_kB_per_K = 0.02
    eb.S_T_min_kB = -40.0
    eb.S_T_max_kB = 0.0
    # S_sigma: Schoeck thermoelastic stress term (more negative under load).
    eb.S_sigma_max_kB = 8.0
    return eb



def apply_cleavage_barrier_args(cb: FractureBarrier, args) -> FractureBarrier:
    """Apply CLI/namespace cleavage-barrier overrides.

    The code now supports both the legacy H-TS-sigma*v cleavage barrier
    (``classic``) and a direct EXP-floor free-energy surface
    (``exp_floor``).  In exp_floor mode entropy and activation volume are
    diagnostics from derivatives of DeltaG*(sigma,T), not independent terms.
    """
    # Backward-compatible classic overrides.
    if getattr(args, 'cleave_H0_eV', None) is not None:
        cb.H0_eV = float(args.cleave_H0_eV)
    if getattr(args, 'cleave_S0_kB', None) is not None:
        cb.S0_neg_kB = float(args.cleave_S0_kB)
    if getattr(args, 'cleave_sigma0_S', None) is not None:
        cb.sigma0_S_GPa = float(args.cleave_sigma0_S)
    if getattr(args, 'cleave_S_sigma_max_kB', None) is not None:
        cb.S_sigma_max_kB = float(args.cleave_S_sigma_max_kB)
    if getattr(args, 'cleave_entropy_form', None) is not None:
        cb.entropy_stress_form = str(args.cleave_entropy_form)
    # Shared entropy-form override used by the old sweep code.
    if getattr(args, 'entropy_form', None) is not None:
        cb.entropy_stress_form = str(args.entropy_form)
        cb.use_negative_entropy = True
    if getattr(args, 'entropy_gate_power', None) is not None:
        cb.entropy_gate_power = float(args.entropy_gate_power)

    kind = getattr(args, 'cleave_barrier_kind', None)
    if kind is not None:
        cb.barrier_kind = str(kind)
    # EXP-floor controls.  These are no-ops unless barrier_kind == exp_floor,
    # but we always populate them so run_args.json fully records the surface.
    mapping = {
        'cleave_G00_eV': 'ef_G00_eV',
        'cleave_gT_eV_per_K': 'ef_gT_eV_per_K',
        'cleave_sigc0_GPa': 'ef_sigc0_Pa',
        'cleave_sT_GPa_per_K': 'ef_sT_Pa_per_K',
        'cleave_exp_a': 'ef_a',
        'cleave_exp_n': 'ef_n',
        'cleave_floor_frac': 'ef_floor_frac',
        'cleave_floor_min_eV': 'ef_floor_min_eV',
        'cleave_floor_max_frac': 'ef_floor_max_frac',
        'cleave_Tref_K': 'ef_Tref_K',
        'cleave_exp_T_mode': 'ef_T_mode',
        'cleave_mu_dlnmu_dT_per_K': 'ef_mu_dlnmu_dT_per_K',
        'cleave_G0_mu_power': 'ef_G0_mu_power',
        'cleave_sigc_mu_power': 'ef_sigc_mu_power',
        'cleave_S_hs_kB': 'ef_S_hs_kB',
        'cleave_sigma_S_GPa': 'ef_sigma_S_GPa',
        'cleave_S_hs_power': 'ef_S_hs_power',
        'cleave_S_hs_dT_per_K': 'ef_S_hs_dT_per_K',
        'cleave_S_hs_Tref_K': 'ef_S_hs_Tref_K',
    }
    for src, dst in mapping.items():
        if getattr(args, src, None) is None:
            continue
        val = getattr(args, src)
        if src in ('cleave_sigc0_GPa', 'cleave_sT_GPa_per_K', 'cleave_sigma_S_GPa'):
            val = float(val) * 1.0e9
        setattr(cb, dst, val)
    if getattr(args, 'cleave_monotone_stress', None) is not None:
        cb.monotone_stress = bool(args.cleave_monotone_stress)
    return cb

def apply_emission_barrier_args(eb: FractureBarrier, args) -> FractureBarrier:
    """Apply CLI/namespace emission EXP-floor overrides.

    This mirrors ``apply_cleavage_barrier_args`` so the sharp-front engine can
    use the fully tuned EXP-floor surface for the local emission hazard rather
    than only the legacy physical-entropy barrier preset.  Parameters are
    direct effective values entering the barrier equation.
    """
    kind = getattr(args, 'emit_barrier_kind', None)
    if kind is not None:
        eb.barrier_kind = str(kind)
    mapping = {
        'emit_G00_eV': 'ef_G00_eV',
        'emit_gT_eV_per_K': 'ef_gT_eV_per_K',
        'emit_sigc0_GPa': 'ef_sigc0_Pa',
        'emit_sT_GPa_per_K': 'ef_sT_Pa_per_K',
        'emit_exp_a': 'ef_a',
        'emit_exp_n': 'ef_n',
        'emit_floor_frac': 'ef_floor_frac',
        'emit_floor_min_eV': 'ef_floor_min_eV',
        'emit_floor_max_frac': 'ef_floor_max_frac',
        'emit_Tref_K': 'ef_Tref_K',
    }
    for src, dst in mapping.items():
        val = getattr(args, src, None)
        if val is None:
            continue
        if src in ('emit_sigc0_GPa', 'emit_sT_GPa_per_K'):
            val = float(val) * 1.0e9
        setattr(eb, dst, val)
    return eb


def default_cleavage_barrier() -> FractureBarrier:
    # Cleavage: NO dislocation-core entropy (no string-mode reorganization,
    # Schoeck eq 34 does not apply to bond rupture). Only the weak modulus
    # -(1/mu)dmu/dT thermoelastic term, so a small baseline and a small stress
    # term. This asymmetry (large emission entropy, tiny cleavage entropy) is
    # the physically-grounded replacement for the old symmetric gated/gated
    # setup, per Schoeck 1980 and the fatigue draft.
    cb = FractureBarrier()
    cb.H0_eV = 2.2
    cb.sigma0_H_GPa = 2.5
    cb.v0_b3 = 0.6
    cb.sigma0_v_GPa = 2.5
    cb.use_negative_entropy = True
    cb.entropy_stress_form = 'physical'
    cb.sigma0_S_GPa = 3.0
    cb.S_T_c0_kB = -2.0          # weak modulus-softening baseline only
    cb.S_T_c1_kB_per_K = 0.0
    cb.S_T_min_kB = -3.0
    cb.S_T_max_kB = 0.0
    cb.S_sigma_max_kB = 1.0      # weak Schoeck stress term
    return cb


def build_engine(args, mat) -> FrontEngine:
    f = FrontConfig()
    f.r0 = args.r_pz
    f.sigma_cap = args.sigma_cap_GPa * 1e9
    f.m_hits = args.multihit_m
    f.tau_c = args.multihit_tau
    f.nu0_c = args.nu0_cleave
    f.nu0_e = args.nu0_emit
    f.beta_back = args.beta_back
    f.c_blunt = args.c_blunt
    f.L_pz = args.L_pz
    f.v_emb_b3 = args.v_emb_b3
    f.wake_retain = args.wake_retain
    f.chi_shield = getattr(args, 'chi_shield', 0.0)
    f.emb_sat_frac = getattr(args, 'emb_sat_frac', 1.0)
    f.N_sat = getattr(args, 'N_sat', float('inf'))
    f.recover_k = getattr(args, 'recover_k', 0.0)
    f.v_rayleigh = getattr(args, 'v_rayleigh', float('inf'))
    f.max_advances_per_step = 1
    f.dN_cap = float('inf')
    f.da = args.da
    if getattr(args, 'rho0', None) is not None:
        f.rho0 = float(args.rho0)

    cb = apply_cleavage_barrier_args(default_cleavage_barrier(), args)
    eb = default_emission_barrier(mat.b)
    if getattr(args, 'emit_H0_eV', None) is not None:
        eb.H0_eV = args.emit_H0_eV

    material_manifest = getattr(args, 'material_manifest', None)
    material_class = getattr(args, 'material_class', None)
    if material_manifest or material_class:
        manifest_path = material_manifest or default_manifest_path(material_class)
        manifest = MaterialManifest.from_csv(manifest_path)
        mpz_cfg = MPZConfig(
            length_m=float(getattr(args, 'mpz_length_um', 100.0)) * 1.0e-6,
            n_bins=int(getattr(args, 'mpz_n_bins', 200)),
            source_bin_count=int(getattr(args, 'mpz_source_bins', 2)),
            blunting_length_m=float(getattr(args, 'mpz_blunting_length_um', 0.5)) * 1.0e-6,
            wake_length_m=float(getattr(args, 'wake_length_um', 100.0)) * 1.0e-6,
            wake_n_bins=int(getattr(args, 'wake_n_bins', 0)),
            wake_shielding=bool(getattr(args, 'wake_shielding', True)),
            wake_shield_projection=float(getattr(args, 'wake_shield_projection', 1.0)),
        )
        f.L_pz = mpz_cfg.length_m
        return UnifiedMPZFrontEngine(
            f, cb, eb, mat.G, mat.nu, mat.b, manifest, mpz_cfg
        )
    return FrontEngine(f, cb, eb, mat.G, mat.nu, mat.b)

# ----------------------------------------------------------------------------
# 1D driver: prescribed K-ramp (the validation / calibration model)
# ----------------------------------------------------------------------------

def run_1d(args):
    cfg = make_emergent_config()
    mat = cfg.material
    os.makedirs(args.out, exist_ok=True)
    _write_run_args(args, args.out, {'driver': 'sharp_front_1d'})

    Kdot = args.Kdot * 1e6      # [Pa sqrt(m)/s]
    dt = args.dt
    t_max = args.Kmax * 1e6 / Kdot

    results = []
    print("=" * 72)
    print("  SHARP-FRONT 1D K-RAMP  (single advance law; no AT2, no gates)")
    print(f"  Kdot = {args.Kdot} MPa*sqrt(m)/s   dt = {dt} s   K_max = {args.Kmax} MPa*sqrt(m)")
    print("=" * 72)

    for T in args.temperatures:
        eng = build_engine(args, mat)
        audit = eng.shelf_audit(T, t_max)
        if not audit['clock_completable']:
            print(f"  [T={T:.0f}K] WARNING rate shelf: lambda_c_max="
                  f"{audit['lambda_c_max']:.3g}/s; clock cannot complete within "
                  f"{t_max:.3g}s. Co-design nu0/tau_c/Kdot.")
        trace = []
        Kc = None
        info = {}
        nstep = int(np.ceil(t_max / dt))
        for i in range(nstep):
            K = Kdot * (i + 1) * dt
            info = eng.step(K, T, dt)
            if (i % max(1, nstep // 400)) == 0 or info['fired']:
                trace.append((eng.t, K, info['sigma_tip'], info['sigma_back'],
                              info['lambda_e'], info['lambda_c'], info['B'],
                              info['N_em'], info['r_eff']))
            if info['fired'] and Kc is None:
                Kc = K
                if not args.continue_after_init:
                    break
            if info['fired'] and eng.n_adv >= args.n_advances:
                break
        # Ductile classification: brittle vs ductile is decided by whether the
        # emitted pile-up significantly SHIELDS the tip, not by a raw emission
        # count.  The old `N_em > 50` cutoff cut through the middle of the
        # brittle cluster (N_em ~ 40-55 at all T in the affine case) and flipped
        # on rounding.  The physical discriminant is the back-stress shielding
        # fraction sigma_back/sigma_tip (emission winning the blunting race) OR
        # substantial tip blunting r_eff/r0, both continuous and with a wide gap
        # between the brittle cluster (~0.1) and genuine ductility (>0.3).
        sig_tip = max(info.get('sigma_tip', 0.0), 1.0)
        shield_frac = info.get('sigma_back', 0.0) / sig_tip
        blunt_ratio = info.get('r_eff', eng.f.r0) / eng.f.r0
        ductile = (shield_frac > args.ductile_shield) or (blunt_ratio > args.ductile_blunt)
        res = {
            'T': T, 'Kc_MPa_sqrt_m': None if Kc is None else Kc / 1e6,
            'N_em_at_fire': info.get('N_em', 0.0), 'W_emit_J_per_m': eng.W_emit,
            'sigma_tip_at_fire_GPa': info.get('sigma_tip', 0.0) / 1e9,
            'sigma_back_at_fire_GPa': info.get('sigma_back', 0.0) / 1e9,
            'shield_frac': shield_frac, 'blunt_ratio': blunt_ratio,
            'r_eff_at_fire_nm': info.get('r_eff', 0.0) * 1e9,
            'dG_emb_eV': info.get('dG_emb_eV', 0.0),
            'mode': 'no-fracture' if Kc is None else ('ductile' if ductile else 'brittle'),
            'shelf': audit,
        }
        results.append(res)
        np.savetxt(os.path.join(args.out, f'trace_{int(T)}K.csv'),
                   np.array(trace) if trace else np.zeros((0, 9)),
                   delimiter=',',
                   header='t_s,K_Pa_sqrtm,sigma_tip_Pa,sigma_back_Pa,lambda_e,lambda_c,B,N_em,r_eff_m',
                   comments='')
        kc_str = 'none (no fracture in window)' if Kc is None else f"{Kc/1e6:.3f} MPa*sqrt(m)"
        print(f"  T={T:5.0f}K  Kc={kc_str:30s} N_em={res['N_em_at_fire']:8.1f} "
              f"sigma_back={res['sigma_back_at_fire_GPa']:.2f} GPa  "
              f"r_eff={res['r_eff_at_fire_nm']:.1f} nm  [{res['mode']}]")

    with open(os.path.join(args.out, 'kc_vs_T.json'), 'w') as fp:
        json.dump(results, fp, indent=2)

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        Ts = [r['T'] for r in results]
        Kcs = [r['Kc_MPa_sqrt_m'] if r['Kc_MPa_sqrt_m'] is not None else np.nan
               for r in results]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(Ts, Kcs, 'o-')
        ax[0].set_xlabel('T [K]'); ax[0].set_ylabel('Kc [MPa*sqrt(m)]')
        ax[0].set_title('Emergent toughness (first-passage cleavage)')
        Ns = [r['N_em_at_fire'] for r in results]
        ax[1].semilogy(Ts, np.maximum(Ns, 1e-2), 's-')
        ax[1].set_xlabel('T [K]'); ax[1].set_ylabel('N_em at fracture')
        ax[1].set_title('Emitted-dislocation ledger (ductility measure)')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'kc_vs_T.png'), dpi=140)
        print(f"  saved {os.path.join(args.out, 'kc_vs_T.png')}")
    except Exception as e:                                    # pragma: no cover
        print(f"  (plot skipped: {e})")
    return results


# ----------------------------------------------------------------------------
# 2D driver: FEM supplies K_J; the SAME engine advances the front
# ----------------------------------------------------------------------------

def _render_field_snapshots(out, T, mesh, snaps, max_cols=4):
    """Render the per-temperature field-snapshot panel for the 2D driver.

    Mirrors the original ``plot_field_snapshots`` layout (tripcolor over the
    mesh triangulation, one colorbar per row) but is self-contained -- it takes
    a list of captured field dicts rather than a SimulationHistory.

    Each snapshot dict carries the spatial fields that actually exist in the
    sharp-front model: damage d (nodal stiffness-kill indicator), log10 rho
    (per-element), sigma1 FEM (per-element principal stress), equivalent
    plastic strain (per-element), plus the FRONT scalars (sigma_tip, B, N_em,
    a_tip) annotated in the title since they are front quantities, not fields.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError:
        print("  matplotlib not available, skipping field snapshots")
        return None
    if not snaps:
        return None

    if len(snaps) > max_cols:
        idx = np.linspace(0, len(snaps) - 1, max_cols, dtype=int)
        pick = [snaps[i] for i in idx]
    else:
        pick = snaps

    tri = None
    if 'nodes' not in pick[0]:
        tri = mtri.Triangulation(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3, mesh.elems)
    rows = ['damage', 'rho', 'sig1', 'epeq']
    # Shared color scales across columns so the panels are comparable and a
    # near-zero field does not get autoscaled into a full-range artifact.
    epeq_max = max(float(np.max([s['epeq_gp'].max() for s in pick])), 1e-6)
    s1_max = max(float(np.max([s['s1_gp'].max() for s in pick])) / 1e6, 1.0)
    nrow, ncol = len(rows), len(pick)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow),
                             squeeze=False, constrained_layout=True)
    last_im = {}
    for j, snap in enumerate(pick):
        tri_j = tri
        if tri_j is None:
            tri_j = mtri.Triangulation(snap['nodes'][:, 0] * 1e3,
                                       snap['nodes'][:, 1] * 1e3, snap['elems'])
        for i, row in enumerate(rows):
            ax = axes[i, j]
            if row == 'damage':
                im = ax.tripcolor(tri_j, snap['d'], shading='gouraud', cmap='inferno',
                                  vmin=0, vmax=1, rasterized=True)
                title = 'damage d'
            elif row == 'rho':
                vals = np.log10(np.maximum(snap['rho_gp'], 1.0))
                im = ax.tripcolor(tri_j, vals, shading='flat', cmap='viridis',
                                  vmin=10, vmax=max(16, float(np.nanmax(vals))),
                                  rasterized=True)
                title = 'log10 rho'
            elif row == 'sig1':
                im = ax.tripcolor(tri_j, snap['s1_gp'] / 1e6, shading='flat',
                                  cmap='magma', vmin=0.0, vmax=s1_max, rasterized=True)
                title = 'sigma1 FEM (MPa)'
            else:  # epeq
                im = ax.tripcolor(tri_j, snap['epeq_gp'], shading='flat',
                                  cmap='magma', vmin=0.0, vmax=epeq_max, rasterized=True)
                title = 'eq. plastic strain'
            last_im[row] = im
            # Overlay the explicit sharp-front inventory.  Very short branch
            # stubs can be difficult to see in the nodal/element damage field,
            # so the rendered panel now shows the actual front polylines too.
            for item in snap.get('front_paths', []):
                try:
                    fid, parent, P = item
                    P = np.asarray(P, float)
                    if P.ndim == 2 and P.shape[0] >= 2:
                        ax.plot(P[:, 0] * 1e3, P[:, 1] * 1e3, lw=1.25)
                        ax.plot(P[-1, 0] * 1e3, P[-1, 1] * 1e3, marker='o', ms=2.5)
                except Exception:
                    pass
            ax.set_aspect('equal')
            ax.set_title(f"{title}\nstep {snap['step']}  KJ={snap['KJ']/1e6:.2f}  "
                         f"N_em={snap['N_em']:.0f}  a={snap['a_tip']*1e3:.2f}mm",
                         fontsize=8)
            ax.set_xlabel('x (mm)')
            if j == 0:
                ax.set_ylabel('y (mm)')
    for i, row in enumerate(rows):
        if row in last_im:
            fig.colorbar(last_im[row], ax=axes[i, :], shrink=0.78, pad=0.01)
    fig.suptitle(f'Sharp-front field snapshots — T = {T:.0f} K', fontsize=13)
    path = os.path.join(out, f'field_snapshots_{int(T)}K.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def _render_diagnostics(out, T, hist):
    """Per-temperature diagnostic suite (sharp-front analogues of the legacy
    main.py plots).  Only quantities that genuinely exist in the sharp-front
    model are plotted; the AT2-specific panels (local-Gc toughening state,
    M_tip amplification, phase-field energy, plasticity-projection fractions)
    have no analogue here and are deliberately omitted.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if not hist['Uapp']:
        return []
    U = np.array(hist['Uapp']) * 1e3  # mm
    saved = []

    def _save(fig, name):
        p = os.path.join(out, f'{name}_{int(T)}K.png')
        fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 1. load-displacement
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(U, np.array(hist['Ftop']) / 1e3, 'b-')
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel('Reaction force (kN)')
    ax.set_title(f'Load-displacement (T = {T:.0f} K)'); ax.grid(alpha=0.3)
    _save(fig, 'load_displacement')

    # 2. toughness + crack-growth resistance curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(U, np.array(hist['KJ']) / 1e6, 'b-', label=r'$K_J$ (domain integral)')
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel(r'$K_J$ (MPa$\sqrt{m}$)')
    ax.set_title(f'Toughness / crack growth (T = {T:.0f} K)'); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(U, np.array(hist['a_tip']) * 1e3, 'r--', label='crack length a')
    ax2.set_ylabel('crack length a (mm)', color='r')
    ax.legend(loc='upper left'); _save(fig, 'toughness')

    # 3. dislocation density distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, lab in (('rho_mean', 'mean'), ('rho_p95', 'p95'),
                   ('rho_p99', 'p99'), ('rho_max', 'max')):
        ax.semilogy(U, np.maximum(hist[k], 1.0), label=lab)
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel(r'$\rho$ (m$^{-2}$)')
    ax.set_title(f'Dislocation density distribution (T = {T:.0f} K)')
    ax.legend(); ax.grid(alpha=0.3, which='both'); _save(fig, 'dislocations')

    # 4. energy balance.  psi_e_gp is now degraded by g_d (FEM fix), so the
    # global balance closes: W_ext = U_el + W_p + W_emit + (fracture work) to
    # discretization error.  Verified: U_el tracks W_ext exactly in the brittle
    # pre-fracture regime where all external work is stored elastically.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(U, hist['W_ext'], 'k-', label=r'$W_{ext}$ (external work)')
    ax.plot(U, hist['U_el'], label=r'$U_{el}$ (stored elastic)')
    ax.plot(U, hist['W_p'], label=r'$W_p$ (bulk plastic)')
    ax.plot(U, hist['W_emit'], label=r'$W_{emit}$ (tip emission)')
    # residual = W_ext - (U_el + W_p + W_emit): the energy that went into
    # creating new fracture surface (plus discretization error).
    resid = (np.array(hist['W_ext']) - np.array(hist['U_el'])
             - np.array(hist['W_p']) - np.array(hist['W_emit']))
    ax.plot(U, resid, 'r--', alpha=0.7, label=r'$W_{ext}-U_{el}-W_p-W_{emit}$ (fracture+err)')
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel('Energy (J/m)')
    ax.set_title(f'Energy balance (T = {T:.0f} K)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, 'energetics')

    # 5. hazard clocks (THE sharp-front diagnostic; no legacy analogue)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(U, np.maximum(hist['lambda_c'], 1e-30), 'b-', label=r'$\lambda_c$ (cleavage)')
    ax.semilogy(U, np.maximum(hist['lambda_e'], 1e-30), 'r-', label=r'$\lambda_e$ (emission)')
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel(r'rate (1/s)')
    ax.set_title(f'Renewal hazard clocks (T = {T:.0f} K)'); ax.grid(alpha=0.3, which='both')
    ax2 = ax.twinx()
    ax2.plot(U, hist['n_fire'], 'k:', label='advances/step')
    ax2.set_ylabel('advances per step'); ax.legend(loc='upper left'); _save(fig, 'hazard_clocks')

    # 6. tip state (sharp-front analogue of tip_memory)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(U, np.array(hist['sigma_tip']) / 1e9, 'b-', label=r'$\sigma_{tip}$ (GPa)')
    ax.plot(U, np.array(hist['sigma_back']) / 1e9, 'r-', label=r'$\sigma_{back}$ (GPa)')
    ax.plot(U, hist['r_eff_over_r0'], 'g-', label=r'$r_{eff}/r_0$ (blunting)')
    ax.set_xlabel('Applied opening (mm)'); ax.set_ylabel('stress (GPa) / ratio')
    ax.set_title(f'Tip state: drive, back-stress, blunting (T = {T:.0f} K)')
    ax2 = ax.twinx()
    ax2.semilogy(U, np.maximum(hist['N_em'], 1e-3), 'm:', label=r'$N_{em}$')
    ax2.set_ylabel(r'$N_{em}$ (emitted ledger)', color='m')
    ax.legend(loc='upper left'); ax.grid(alpha=0.3); _save(fig, 'tip_state')

    return saved


def _render_toughness_vs_T(out, summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = [s for s in summary if s.get('Kc_first_MPa_sqrt_m') is not None]
    if not rows:
        return None
    Ts = [s['T'] for s in summary]
    Kc = [s['Kc_first_MPa_sqrt_m'] if s['Kc_first_MPa_sqrt_m'] is not None else np.nan
          for s in summary]
    modes = [s['mode'] for s in summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ts, Kc, 'o-', color='navy', label=r'$K_c$ (first advance, domain $K_J$)')
    for t, k, m in zip(Ts, Kc, modes):
        if not np.isnan(k):
            ax.annotate(m, (t, k), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'$K_c$ (MPa$\sqrt{m}$)')
    ax.set_title('Fracture toughness vs temperature (sharp-front)')
    ax.grid(alpha=0.3); ax.legend()
    p = os.path.join(out, 'toughness_vs_temperature.png')
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    return p


def eng_r_pz_hint(args):
    """Length-scale hint for the default physical advance increment."""
    return float(getattr(args, 'r_pz', 1e-6))


def _remesh_following_tip(geom, mesh_cfg, seed, tip_xy, old_mesh,
                          rho_gp, ep_gp, u, crack_paths, kill_r, half_h,
                          extra_elem_fields=None):
    """Rebuild the adaptive mesh refined at the CURRENT tip, transfer history
    fields, and re-apply the crack exactly from the saved polyline(s).

    The crack is re-laid from `crack_paths` (geometry), NOT interpolated from the
    old damage field -- so the sharp crack is preserved with no smearing.  The
    per-element history (rho, ep) and the nodal displacement (u, warm start) are
    transferred by nearest neighbour.  Returns the new mesh + transferred state.
    """
    from .mesh import make_tri_mesh, make_boundary_data
    from scipy.spatial import cKDTree
    import numpy as _np

    new_mesh = make_tri_mesh(geom, mesh_cfg, seed=seed, tip_center=tip_xy)
    new_bnd = make_boundary_data(new_mesh, geom)

    old_c = old_mesh.nodes[old_mesh.elems].mean(axis=1)
    new_c = new_mesh.nodes[new_mesh.elems].mean(axis=1)
    _, ie = cKDTree(old_c).query(new_c)            # nearest old element per new element
    rho_new = _np.ascontiguousarray(rho_gp[ie])
    ep_new = _np.ascontiguousarray(ep_gp[:, ie])
    extra_new = None
    if extra_elem_fields is not None:
        extra_new = []
        for fld in extra_elem_fields:
            extra_new.append(_np.ascontiguousarray(_np.asarray(fld)[ie]))
    _, inn = cKDTree(old_mesh.nodes).query(new_mesh.nodes)   # nearest old node
    u_old = u.reshape(-1, 2)
    u_new = u_old[inn].reshape(-1).copy()

    # re-apply notch + all crack polylines on the new mesh (exact, no smearing)
    d_new = _np.zeros(new_mesh.nn)
    x = new_mesh.nodes[:, 0]; y = new_mesh.nodes[:, 1]
    d_new[(x <= geom.a0) & (_np.abs(y) <= half_h)] = 1.0
    cxe = new_c[:, 0]; cye = new_c[:, 1]
    erad = _np.sqrt(_np.maximum(new_mesh.area_e, 1e-30))
    for path in crack_paths:
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            seg = p1 - p0; L2 = float(seg @ seg) + 1e-30
            tt = _np.clip(((cxe - p0[0]) * seg[0] + (cye - p0[1]) * seg[1]) / L2, 0.0, 1.0)
            ddx = cxe - (p0[0] + tt * seg[0]); ddy = cye - (p0[1] + tt * seg[1])
            rad = _np.maximum(kill_r, 0.7 * erad)
            d_new[new_mesh.elems[(ddx * ddx + ddy * ddy) <= rad ** 2]] = 1.0
    if extra_elem_fields is not None:
        return new_mesh, new_bnd, d_new, rho_new, ep_new, u_new, extra_new
    return new_mesh, new_bnd, d_new, rho_new, ep_new, u_new


def run_2d(args):
    from .mesh import make_tri_mesh, make_boundary_data
    from .fem import plane_strain_D, assemble_mechanics, solve_dirichlet
    from .plasticity import (update_plasticity, build_elem_adjacency,
                             transport_rho_step)
    from .j_integral import compute_J_integral
    from .fatigue_v1 import FatigueWaveform, build_controller_from_namespace
    from .crack_backend import build_crack_backend
    from .coalescence import first_path_intersection

    cfg = make_emergent_config()
    cfg.mesh.nx = args.nx
    cfg.mesh.ny = args.ny
    cfg.mesh.tip_h_fine = getattr(args, 'tip_h_fine', 0.0) or 0.0
    cfg.mesh.tip_ratio = getattr(args, 'tip_ratio', 1.15)
    cfg.loading.n_steps = args.steps
    cfg.loading.dU_top = args.dU
    cfg.loading.dt = args.dt
    # sources-only plasticity controls (default: legacy bulk multiplication)
    cfg.dislocations.bulk_mult_frac = getattr(args, 'bulk_mult_frac', 1.0)
    cfg.dislocations.tip_source_rho_per_emit = getattr(args, 'tip_source_rho_per_emit', 0.0)
    cfg.dislocations.rho_transport_c = getattr(args, 'rho_transport_c', 0.0)
    cfg.dislocations.exhaustion_enabled = getattr(args, 'exhaustion', False)
    cfg.dislocations.glide_to_sink_m = getattr(args, 'glide_to_sink_m', 1e-5)
    cfg.dislocations.mobile_rho_floor = getattr(args, 'mobile_rho_floor', 1e6)
    cfg.dislocations.peierls_floor_min_Pa = getattr(args, 'peierls_floor_MPa', 0.0) * 1e6
    mat = cfg.material

    # v9.3 production bulk plasticity: use the active crack-tip emission
    # EXP-floor surface as the parent for sequential Peierls and correlated
    # multi-hit Taylor kinetics. The older additive flow-stress construction is
    # retained only when explicitly selected as a legacy ablation.
    cfg.dislocations.bulk_kinetics_model = str(getattr(
        args, 'bulk_kinetics_model',
        'emission_derived_peierls_taylor_multihit'))
    _eb_bulk = apply_emission_barrier_args(default_emission_barrier(mat.b), args)
    cfg.dislocations.pt_emit_G00_eV = float(_eb_bulk.ef_G00_eV)
    cfg.dislocations.pt_emit_gT_eV_per_K = float(_eb_bulk.ef_gT_eV_per_K)
    cfg.dislocations.pt_emit_sigc0_Pa = float(_eb_bulk.ef_sigc0_Pa)
    cfg.dislocations.pt_emit_sT_Pa_per_K = float(_eb_bulk.ef_sT_Pa_per_K)
    cfg.dislocations.pt_emit_Tref_K = float(_eb_bulk.ef_Tref_K)
    cfg.dislocations.pt_emit_exp_a = float(_eb_bulk.ef_a)
    cfg.dislocations.pt_emit_exp_n = float(_eb_bulk.ef_n)
    cfg.dislocations.pt_emit_floor_frac = float(_eb_bulk.ef_floor_frac)
    cfg.dislocations.pt_emit_floor_min_eV = float(_eb_bulk.ef_floor_min_eV)
    cfg.dislocations.pt_emit_floor_max_frac = float(_eb_bulk.ef_floor_max_frac)
    cfg.dislocations.pt_peierls_energy_ratio = float(getattr(args, 'peierls_energy_scale', 0.005))
    cfg.dislocations.pt_peierls_entropy_ratio = float(getattr(args, 'peierls_entropy_scale', 0.005))
    cfg.dislocations.pt_peierls_stress_ratio = float(getattr(args, 'peierls_stress_scale', 1.0))
    cfg.dislocations.pt_peierls_nu0_s = float(getattr(args, 'nu0_peierls', 1.0e12))
    cfg.dislocations.pt_taylor_energy_ratio = float(getattr(args, 'taylor_energy_scale', 0.02))
    cfg.dislocations.pt_taylor_entropy_ratio = float(getattr(args, 'taylor_entropy_scale', 0.02))
    cfg.dislocations.pt_taylor_stress_ratio = float(getattr(args, 'taylor_stress_scale', 1.0))
    cfg.dislocations.pt_taylor_nu0_s = float(getattr(args, 'nu0_taylor', 1.0e11))
    cfg.dislocations.pt_taylor_corr_rho_c = float(getattr(args, 'pt_taylor_corr_rho_c', 1.0e14))
    cfg.dislocations.pt_taylor_renewal_time_s = float(getattr(args, 'pt_taylor_renewal_time_s', 1.0e-9))
    cfg.dislocations.pt_taylor_m_exponent = float(getattr(args, 'pt_taylor_m_exponent', 1.0))
    cfg.dislocations.pt_taylor_m_scale = float(getattr(args, 'pt_taylor_m_scale', 1.0))
    cfg.dislocations.pt_taylor_m_cap = float(getattr(args, 'pt_taylor_m_cap', float('inf')))
    cfg.dislocations.pt_mobile_fraction = float(getattr(args, 'pt_mobile_fraction', 0.01))
    cfg.dislocations.pt_mobile_saturation_density_m2 = float(getattr(args, 'pt_mobile_saturation_density_m2', 1.0e14))
    cfg.dislocations.pt_mobile_density_floor_m2 = float(getattr(args, 'pt_mobile_density_floor_m2', 1.0e6))
    cfg.dislocations.pt_jump_fraction = float(getattr(args, 'pt_jump_fraction', 1.0))
    cfg.dislocations.pt_jump_length_min_m = float(getattr(args, 'pt_jump_length_min_m', mat.b))
    cfg.dislocations.pt_taylor_phi_max = float(getattr(args, 'pt_taylor_phi_max', 20.0))
    os.makedirs(args.out, exist_ok=True)
    fatigue_mode = bool(getattr(args, 'fatigue_cycles', False))
    fatigue_controller = build_controller_from_namespace(args) if fatigue_mode else None
    if fatigue_mode:
        fatigue_controller.write_config(os.path.join(args.out, 'fatigue_cycle_controller_config.json'))

    # Branching/material preset.  The W preset is the physical near-isotropic
    # baseline.  The branchy preset is an explicit second material class for
    # stress-testing the competing-hazard branch machinery: stronger elastic/
    # cleavage anisotropy, lower co-criticality threshold, and optional {110}
    # secondary cleavage traces.  User-supplied explicit values still override.
    branch_preset = str(getattr(args, 'crystal_material', 'w') or 'w').lower()
    if branch_preset in ('branchy', 'bcc-branchy', 'model-branchy'):
        if getattr(args, 'crystal_C44', None) is None:
            args.crystal_C44 = 320.0e9
        if getattr(args, 'cleave_gamma_aniso', None) is None:
            args.cleave_gamma_aniso = 2.0
        if getattr(args, 'branch_overdrive_ratio', None) is None:
            args.branch_overdrive_ratio = 0.80
        if getattr(args, 'branch_ratio', None) is None:
            args.branch_ratio = 0.85
        if not getattr(args, 'crystal_include_110', False):
            args.crystal_include_110 = True
        if getattr(args, 'gamma_110_rel', None) is None:
            args.gamma_110_rel = 1.15
    else:
        if getattr(args, 'cleave_gamma_aniso', None) is None:
            args.cleave_gamma_aniso = 0.3
        if getattr(args, 'branch_overdrive_ratio', None) is None:
            args.branch_overdrive_ratio = 0.9
        if getattr(args, 'branch_ratio', None) is None:
            args.branch_ratio = 0.92
        if getattr(args, 'gamma_110_rel', None) is None:
            args.gamma_110_rel = 1.3
    _write_run_args(args, args.out, {'driver': 'sharp_front_2d',
                                     'crystal_material_resolved': branch_preset})

    mesh = make_tri_mesh(cfg.geometry, cfg.mesh, seed=42)
    bnd = make_boundary_data(mesh, cfg.geometry)
    # Elasticity: isotropic by default; optional cubic BCC anisotropy carrying a
    # crystal orientation, so the near-tip field is orientation-biased (a
    # prerequisite for crystallographic deflection). Both assemble_mechanics
    # calls below use this same D, so assembly and stress recovery stay consistent.
    if getattr(args, 'crystal_aniso', False):
        from .crystal import cubic_plane_strain_D, W_C11, W_C12, W_C44, zener_ratio
        C11 = float(getattr(args, 'crystal_C11', W_C11) or W_C11)
        C12 = float(getattr(args, 'crystal_C12', W_C12) or W_C12)
        C44 = float(getattr(args, 'crystal_C44', W_C44) or W_C44)
        theta = float(getattr(args, 'crystal_theta_deg', 0.0) or 0.0)
        D = cubic_plane_strain_D(C11, C12, C44, theta)
        print(f"  ANISOTROPIC elasticity: cubic BCC, Zener A={zener_ratio(C11,C12,C44):.2f}, "
              f"crystal theta={theta:.1f} deg")
    else:
        D = plane_strain_D(mat)
    plast_model = PlasticityModel(cfg.plasticity_barrier, mat)

    half_h = cfg.geometry.notch_half_thickness
    y = mesh.nodes[:, 1]
    x = mesh.nodes[:, 0]

    # Mesh-independent length scales.  The crack physics must not depend on the
    # element size: one cleavage event advances the tip by a PHYSICAL increment
    # da_phys (not one element), and the J-integral contour radius is an
    # ABSOLUTE length (not 4*hbar) so K_J is comparable across mesh refinement.
    # Element resolution only sets how finely these physical lengths are drawn.
    da_phys = args.da_phys if args.da_phys is not None else max(5.0 * eng_r_pz_hint(args), 2e-6)
    # Local J-contour control.  Older versions passed `rJ` as the domain-integral
    # length parameter ell, so the actual outer contour was 8*rJ.  That made a
    # newborn branch of length O(da_phys) invisible to its own J-domain.  In the
    # multi-front model we instead choose an actual desired OUTER radius and pass
    # ell = r_outer/8 to compute_J_integral.  The default is still conservative:
    # several elements across the domain and several physical crack increments,
    # but close enough that short branch arms can become independently resolved.
    if getattr(args, 'rJ_outer', None) is not None:
        r_J_outer = float(args.rJ_outer)
    elif args.rJ is not None:
        # Back-compatible interpretation: user supplied the old ell-like length.
        r_J_outer = 8.0 * float(args.rJ)
    else:
        _h_guess = (getattr(args, 'tip_h_fine', 0.0) or 0.0)
        _h_guess = _h_guess if _h_guess > 0 else 0.0
        r_J_outer = max(3.0 * da_phys, 2.5 * args.L_pz, 6.0 * _h_guess, 3.0e-6)

    # Cluster / global J scale.  This is deliberately separate from the local
    # branch-tip contour.  The original multifront runs used the legacy domain
    # length r_J as `ell`, which means the actual outer domain radius is about
    # 8*r_J.  The local-J patch accidentally made the primary crack use a much
    # smaller contour.  In the clustered decomposition, parent/unresolved
    # clusters use this larger group-J domain, while only independently resolved
    # daughter tips use the small local contour.
    if getattr(args, 'rJ_cluster', None) is not None:
        r_J_cluster_ell = float(args.rJ_cluster)
    elif args.rJ is not None:
        r_J_cluster_ell = float(args.rJ)
    else:
        r_J_cluster_ell = max(10.0 * args.L_pz, 5e-6)
    r_J_cluster_outer = 8.0 * r_J_cluster_ell

    # the resolution that matters for the process zone is the TIP-LOCAL element
    # size; for a graded mesh this is far finer than the global mean.
    h_res = mesh.hbar_tip if (cfg.mesh.tip_h_fine and mesh.hbar_tip > 0) else mesh.hbar

    print("=" * 72)
    print("  SHARP-FRONT 2D  (FEM supplies K_J; engine is the only advance law)")
    if fatigue_mode:
        print(f"  FATIGUE MODE: for each active v8 front, local J-derived Kmax is cycled "
              f"with R={getattr(args, 'R', 0.1):g}, f={getattr(args, 'frequency_Hz', 1e3):g} Hz; "
              f"cycle blocks update the front-local process-zone ledger before the existing "
              f"cleavage renewal clock is allowed to fire.")
    graded = " GRADED" if cfg.mesh.tip_h_fine else ""
    print(f"  mesh{graded} {mesh.nn} nodes, hbar_tip={h_res:.3e} m "
          f"(global {mesh.hbar:.3e} m)  (resolution only)")
    print(f"  da_phys={da_phys:.3e} m, local J outer radius={r_J_outer:.3e} m, "
          f"cluster J outer radius={r_J_cluster_outer:.3e} m  (physical, mesh-independent)")
    if r_J_outer <= da_phys:
        print(f"  WARNING: local J outer radius <= da_phys; independent daughter-tip J is under-resolved.")
    if r_J_cluster_outer <= r_J_outer:
        print(f"  WARNING: cluster J outer radius <= local J outer radius; group/local decomposition is ill-conditioned.")
    if h_res > da_phys:
        print(f"  NOTE: hbar_tip > da_phys -- mesh too coarse to resolve the advance "
              f"increment; crack will advance in >=1-element jumps. Refine to hbar_tip<{da_phys:.1e} m.")
    if h_res > args.L_pz:
        print(f"  NOTE: hbar_tip > L_pz ({args.L_pz:.1e} m) -- process zone unresolved; "
              f"pile-up/back-stress terms are mesh-floored until you refine.")
    else:
        print(f"  process zone RESOLVED: hbar_tip={h_res:.2e} m < L_pz={args.L_pz:.1e} m")
    print("=" * 72)

    # element centroids (used by tip-source deposit) and face adjacency (for
    # conservative density transport).  Geometry-only, built once.
    cx_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
    cy_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
    src_rho_per_emit = float(getattr(cfg.dislocations, 'tip_source_rho_per_emit', 0.0))
    rho_transport_c = float(getattr(cfg.dislocations, 'rho_transport_c', 0.0))
    adj = build_elem_adjacency(mesh) if rho_transport_c > 0.0 else None
    if src_rho_per_emit > 0.0 or rho_transport_c > 0.0 or \
            getattr(cfg.dislocations, 'bulk_mult_frac', 1.0) != 1.0:
        print(f"  sources-only plasticity: bulk_mult_frac="
              f"{getattr(cfg.dislocations, 'bulk_mult_frac', 1.0):g}, "
              f"tip_source_rho_per_emit={src_rho_per_emit:g}, "
              f"rho_transport_c={rho_transport_c:g}")

    summary = []
    for T in args.temperatures:
        # Reset the mesh to the notch-centered refinement at the start of every
        # temperature.  Tip-following remeshing MUTATES `mesh` during a run
        # (refining wherever the crack tip ends up), so without this rebuild the
        # next temperature would inherit the previous crack's far-field
        # refinement -- leaving the notch in coarse mesh and corrupting Kc(T).
        mesh = make_tri_mesh(cfg.geometry, cfg.mesh, seed=42)
        bnd = make_boundary_data(mesh, cfg.geometry)
        # mesh-derived globals (set once before the loop) must be refreshed too,
        # else the notch stamp d[(x<=a0)&...] uses the PREVIOUS crack's remeshed
        # coordinates -> wrong initial crack -> corrupted Kc(T).
        x = mesh.nodes[:, 0]; y = mesh.nodes[:, 1]
        cx_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
        cy_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
        adj = build_elem_adjacency(mesh) if rho_transport_c > 0.0 else None
        eng = build_engine(args, mat)
        eng.f.da = da_phys  # PHYSICAL advance per cleavage event (mesh-independent)
        audit = eng.shelf_audit(T, args.steps * cfg.loading.dt)
        if not audit['clock_completable']:
            print(f"  [T={T:.0f}K] WARNING rate shelf: clock cannot complete; "
                  f"lambda_c_max={audit['lambda_c_max']:.3g}/s")

        # ================================================================
        # MULTI-FRONT INVENTORY (refactor): the crack is a list of active
        # sharp fronts.  Every front carries its own engine, tip, heading,
        # polyline and process-zone ledger.  Each step: one shared FEM/J
        # solve, per-front local driving, per-front first-passage clocks,
        # branch birth from ANY co-critical firing front, shared finite
        # advance budget across all co-firing fronts, per-front wake/source
        # deposit, and remesh centered on the leading tip.
        # ================================================================
        a_tip = cfg.geometry.a0
        a_killed = a_tip
        deflect = bool(getattr(args, 'crystal_aniso', False))
        max_fronts = int(getattr(args, 'max_fronts', 32) or 32)

        d = np.zeros(mesh.nn)
        d[(x <= a_tip) & (np.abs(y) <= half_h)] = 1.0
        u = np.zeros(mesh.ndof)
        ep_gp = np.zeros((3, mesh.ne))
        rho_gp = np.full(mesh.ne, eng.f.rho0)

        # Spatial process-zone state used by the v8 fatigue adapter.  The old
        # scalar N_em ledger remains the V1 reduction, but the 2-D driver now
        # deposits the emitted/stored/mobile content into element fields around
        # each front/wake and feeds the weighted near-tip stored count back into
        # the front engine before cycle hazards are evaluated.
        # The moving-PZ engine owns a conservative front-local state.  The old
        # 2-D projection fields remain available for legacy fatigue runs and
        # diagnostics, but must not overwrite the moving state.
        pz_spatial_state = (fatigue_mode
                            and getattr(eng, 'state_model', 'legacy_scalar') != 'moving_pz'
                            and (not bool(getattr(args, 'no_pz_spatial_state', False))))
        pz_store_gp = np.zeros(mesh.ne)   # retained PZ dislocation/event density proxy [count/m^2]
        pz_mobile_gp = np.zeros(mesh.ne)  # mobile emitted content before escape/storage [count/m^2]
        pz_escape_gp = np.zeros(mesh.ne)  # escaped/glided content audit [count/m^2]
        pz_emit_gp = np.zeros(mesh.ne)    # cumulative emitted content audit [count/m^2]
        cyclic_mechanics_enabled = fatigue_mode and (not bool(getattr(args, 'no_cyclic_mechanics', False)))
        cyclic_mechanics_updates = 0
        cyclic_plastic_work_acc = 0.0
        Kc_first = None
        Kc_first_step = None
        rows = []
        branch_rows = []
        fronts_rows = []
        snaps = []
        hist = {k: [] for k in ('Uapp', 'Ftop', 'KJ', 'W_ext', 'U_el', 'W_p',
                                'W_emit', 'rho_mean', 'rho_p95', 'rho_p99',
                                'rho_max', 'lambda_c', 'lambda_e', 'B', 'n_fire',
                                'sigma_tip', 'sigma_back', 'r_eff_over_r0',
                                'N_em', 'a_tip', 'n_fronts')}
        W_ext_acc = 0.0; W_p_acc = 0.0; Ftop_prev = 0.0; Uapp_prev = 0.0

        # Replaceable crack-geometry backend.  All hazard, fatigue, plasticity,
        # directional competition, and branch bookkeeping remain above this layer.
        crack_backend = build_crack_backend(args, cfg.geometry)
        cohesive_network = crack_backend.cohesive_network
        if crack_backend.name != 'sharp_wake':
            # Topology changes are committed one at a time. Any additional
            # completed renewals remain in B and are consumed on subsequent
            # event solves; no hazard is discarded. Child engines inherit this.
            eng.f.max_advances_per_step = 1
            print(f"  crack backend: {crack_backend.name} (Arrhenius event criterion; CZM geometry; one topology event/solve)")

        def _backend_crack_segments():
            logs = getattr(crack_backend, 'advance_log', None)
            if not logs:
                return None
            return [(np.array([r['x0'], r['y0']], dtype=float),
                     np.array([r['x1'], r['y1']], dtype=float)) for r in logs]

        def _pz_kernel(point_xy, engine_for_scale=None, radius_factor=None):
            """Gaussian element weights around a sharp-front process zone.

            The fields are element densities [count/m^2].  Integrating a density
            against area gives the corresponding front-local count.  The kernel
            is only a projection operator; it is not a crack-growth gate.
            """
            if engine_for_scale is None:
                Lpz = float(args.L_pz)
            else:
                Lpz = float(engine_for_scale.f.L_pz)
            rf = float(radius_factor if radius_factor is not None
                       else getattr(args, 'pz_deposit_radius_factor', 1.0))
            Lpz = max(Lpz * max(rf, 1e-6), mesh.hbar, 1e-12)
            dx = cx_e - float(point_xy[0]); dy = cy_e - float(point_xy[1])
            r2 = dx * dx + dy * dy
            w = np.exp(-0.5 * r2 / (Lpz * Lpz))
            # truncate the far Gaussian tail so remote elements do not acquire a
            # nonzero density in long fatigue sweeps.
            w = np.where(r2 <= (3.0 * Lpz) ** 2, w, 0.0)
            return w, Lpz

        def _field_count_near(field, point_xy, engine_for_scale=None):
            if (not pz_spatial_state) or field is None or field.size == 0:
                return 0.0
            w, _ = _pz_kernel(point_xy, engine_for_scale)
            return float(np.sum(np.maximum(field, 0.0) * mesh.area_e * w))

        def _deposit_pz_density(field, point_xy, dN_count, engine_for_scale=None):
            if (not pz_spatial_state) or field is None:
                return 0.0
            dN_count = float(dN_count)
            if not np.isfinite(dN_count) or dN_count <= 0.0:
                return 0.0
            w, _ = _pz_kernel(point_xy, engine_for_scale)
            denom = float(np.sum(w * mesh.area_e))
            if denom <= 0.0:
                return 0.0
            field[:] = field + dN_count * w / denom
            return dN_count

        def _sync_front_engine_from_pz(front):
            """Project the spatial stored field onto the scalar V1 ledger.

            This keeps V1/v8 consistency: V1 is the zero-dimensional reduction of
            the same ledger, while v8 lets the surrounding wake and neighboring
            branches contribute through the spatial near-tip field.
            """
            if (not pz_spatial_state or front is None
                    or getattr(front.get('eng'), 'state_model', 'legacy_scalar') == 'moving_pz'):
                return 0.0
            N_field = _field_count_near(pz_store_gp, front['xy'], front['eng'])
            coupling = float(np.clip(getattr(args, 'pz_field_coupling', 1.0), 0.0, 1.0))
            if coupling > 0.0 and np.isfinite(N_field):
                front['eng'].N_em = max((1.0 - coupling) * float(front['eng'].N_em)
                                        + coupling * N_field, 0.0)
            front['pz_store_count'] = N_field
            front['pz_mobile_count'] = _field_count_near(pz_mobile_gp, front['xy'], front['eng'])
            front['pz_escape_count'] = _field_count_near(pz_escape_gp, front['xy'], front['eng'])
            return N_field

        def _commit_front_pz_fields(front, info_dict):
            if (not pz_spatial_state) or front is None or info_dict is None:
                return
            # Retained content drives back-stress/blunting/embrittlement; mobile
            # and escape fields are explicit diagnostics/state sources for the
            # Peierls/Taylor pathways.
            _deposit_pz_density(pz_emit_gp, front['xy'], info_dict.get('dN_emit_block', 0.0), front['eng'])
            _deposit_pz_density(pz_store_gp, front['xy'], info_dict.get('dN_store_block', 0.0), front['eng'])
            _deposit_pz_density(pz_mobile_gp, front['xy'], info_dict.get('dN_mobile_block', 0.0), front['eng'])
            _deposit_pz_density(pz_escape_gp, front['xy'], info_dict.get('dN_escape_block', 0.0), front['eng'])
            # Also expose retained content to the existing Taylor/plasticity
            # density field so the old full-field dislocation-storage machinery
            # is not bypassed.  This conversion is deliberately explicit and
            # auditable rather than hidden in the front scalar ledger.
            scale = float(getattr(args, 'pz_store_to_rho_scale', 1.0))
            if scale > 0.0:
                _deposit_pz_density(rho_gp, front['xy'], scale * info_dict.get('dN_store_block', 0.0), front['eng'])

        def _phase_scale_array(nphase):
            nphase = max(int(nphase), 4)
            ph = (np.arange(nphase, dtype=float) + 0.5) * (2.0 * np.pi / nphase)
            Rcyc = float(getattr(args, 'R', 0.1) or 0.0)
            q = 0.5 * (1.0 + Rcyc) + 0.5 * (1.0 - Rcyc) * np.cos(ph)
            if not bool(getattr(args, 'no_closure_clip', False)):
                q = np.maximum(q, 0.0)
            return q

        def _run_cyclic_mechanics_block(Umax, cycles, T_local):
            """Resolve the 2-D body through one representative cycle block.

            This is intentionally coupled to the existing FEM/plasticity tools: the
            sharp-front crack geometry is held fixed, but the displacement boundary
            condition is cycled through R -> Kmax, and update_plasticity evolves
            ep_gp/rho_gp at each phase.  The crack can still advance only after
            the sharp-front renewal clock fires later in the step.
            """
            nonlocal u, ep_gp, rho_gp, sigma_gp, psi_gp, s1_gp, Ftop, dot_ep
            nonlocal cyclic_mechanics_updates, cyclic_plastic_work_acc
            if (not cyclic_mechanics_enabled) or cycles <= 0.0:
                return 0.0
            freq = max(float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3), 1e-300)
            ncyc_phase = int(getattr(args, 'cyclic_mechanics_phases', 0) or 0)
            if ncyc_phase <= 0:
                ncyc_phase = min(max(int(getattr(args, 'n_phase', 96) or 96), 8), 24)
            qvals = _phase_scale_array(ncyc_phase)
            dt_phase = float(cycles) / freq / len(qvals)
            stag = max(int(getattr(args, 'cyclic_mechanics_stagger', 1) or 1), 1)
            dWp_cycle = 0.0
            for q in qvals:
                Uy_top_c = 0.5 * Umax * float(q)
                Uy_bot_c = -0.5 * Umax * float(q)
                for _it in range(stag):
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
                    u, Ftop = solve_dirichlet(Kmat, Rint, u, bnd, Uy_top_c, Uy_bot_c)
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
                    try:
                        ep_gp, rho_gp, dot_ep, pinfo = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T_local, dt_phase,
                            plast_model, cfg.dislocations, return_info=True)
                        dWp_cycle += float(np.sum(pinfo.get('dWp_accepted_gp', 0.0) * mesh.area_e))
                    except TypeError:
                        ep_gp, rho_gp, dot_ep = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T_local, dt_phase,
                            plast_model, cfg.dislocations)
                if rho_transport_c > 0.0 and adj is not None:
                    D_e = rho_transport_c * np.maximum(dot_ep, 0.0) * (eng.f.L_pz ** 2)
                    rho_gp = transport_rho_step(rho_gp, adj, mesh.area_e, D_e, dt_phase)
                    rho_gp = np.clip(rho_gp, 1e6, cfg.dislocations.rho_cap)
            # Return to the maximum of the block before evaluating J-derived Kmax.
            Uy_top_m = 0.5 * Umax; Uy_bot_m = -0.5 * Umax
            Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
            u, Ftop = solve_dirichlet(Kmat, Rint, u, bnd, Uy_top_m, Uy_bot_m)
            Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
            cyclic_mechanics_updates += 1
            cyclic_plastic_work_acc += max(dWp_cycle, 0.0)
            return max(dWp_cycle, 0.0)

        fronts = None
        branch_on = False
        remesh_on = False
        if deflect:
            from .crystal import (bcc_cleavage_traces, near_tip_stress_tensor,
                                   pick_cleavage_plane, cleavage_branch_candidates,
                                   admissible_openings, cleave_direction_competition)
            _theta = float(getattr(args, 'crystal_theta_deg', 0.0) or 0.0)
            compete = bool(getattr(args, 'crystal_compete', False))
            gamma_aniso = float(getattr(args, 'cleave_gamma_aniso', 0.3) or 0.0)
            r_branch_O = float(getattr(args, 'branch_overdrive_ratio', 0.9) or 0.9)
            cleave_planes = bcc_cleavage_traces(
                _theta,
                include_110=bool(getattr(args, 'crystal_include_110', False)),
                gamma_110_rel=float(getattr(args, 'gamma_110_rel', 1.3) or 1.3))
            fwd0 = np.array([1.0, 0.0])
            gate_global = bool(getattr(args, 'plane_gate_global', False))
            h_tip = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
            kill_r = max(h_tip, 0.5e-6)
            elem_rad = np.sqrt(np.maximum(mesh.area_e, 1e-30))
            cxe = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
            cye = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
            remesh_on = (crack_backend.name == 'sharp_wake') and (not bool(getattr(args, 'no_tip_remesh', False)))
            _hf = getattr(cfg.mesh, 'tip_h_fine', 0.0) or h_tip
            R_fine_est = min(0.15 * max(cfg.geometry.Lx, cfg.geometry.Ly),
                             max(40 * _hf, 0.05e-3))
            R_trigger = 0.4 * R_fine_est
            branch_on = bool(getattr(args, 'crystal_branch', False))
            branch_ratio = float(getattr(args, 'branch_ratio', 0.92) or 0.92)
            _xlo, _xhi = 0.0, cfg.geometry.Lx - 2e-5
            _ylo, _yhi = -cfg.geometry.Ly / 2 + 2e-5, cfg.geometry.Ly / 2 - 2e-5

            _next_id = [1]

            def _new_front(xy, fwd, engine, plane, parent=-1, spawn_step=0,
                           spawn_w=1.0, ratio=0.0, fid=0):
                xy = np.array(xy, float); fwd = np.array(fwd, float)
                return {'xy': xy.copy(), 'fwd': fwd.copy(), 'eng': engine,
                        'path': [xy.copy()], 'last_plane': plane, 'active': True,
                        'id': fid, 'parent': parent, 'N_em_prev': 0.0,
                        'spawn_step': spawn_step, 'spawn_weight': spawn_w,
                        'spawn_ratio': ratio, 'KJ': 0.0, 't_win': fwd.copy(),
                        'win_plane': plane, 'cands': None, 'info': None,
                        'fired': False, 'born_this_step': False,
                        # resolved=False means a newborn daughter is still inside
                        # the parent-tip J/process-zone domain. It is drawn as a
                        # sharp segment, but it shares the parent's local event
                        # budget until its arclength/separation exceed the local
                        # J-contour handoff length. This avoids reading a bogus
                        # independent J-integral from a sub-contour stub.
                        'resolved': True, 'birth_xy': xy.copy(),
                        'J_source': 'cluster' if parent < 0 else 'local',
                        'J_source_code': 0 if parent < 0 else 1,
                        'cluster_id': fid if parent < 0 else parent,
                        'adv_since_branch': 0.0, 'last_advance_m': 0.0,
                        'n_geom_adv': 0.0, 'birth_budget_w': 0.0,
                        'birth_parent': -1, 'cluster_hold_reason': 'root',
                        # Branch birth is now its own renewal/first-passage
                        # process. A moderately competitive secondary lobe is
                        # not enough: it must accumulate branch_B to threshold
                        # under an absolute secondary driving rate.
                        'branch_B': 0.0, 'branch_dB_last': 0.0,
                        'branch_lambda_secondary': 0.0,
                        'branch_K_secondary': 0.0,
                        'branch_J_secondary_signed': 0.0,
                        'branch_J_secondary_effective': 0.0,
                        'branch_J_secondary_active_elems': 0,
                        'branch_metric_ratio': 0.0,
                        'branch_clock_ready': False,
                        'branch_clock_angle_deg': np.nan,
                        'branch_veto_reason': 'none',
                        # Persistent diagnostics for actual branch birth.  The
                        # branch clock is reset/partly consumed on spawn, so keep
                        # pre-spawn values separately for audit.
                        'branch_B_before_spawn': 0.0,
                        'branch_dB_at_spawn': 0.0,
                        'branch_lambda_secondary_at_spawn': 0.0,
                        'branch_K_secondary_at_spawn': 0.0,
                        'branch_J_secondary_signed_at_spawn': 0.0,
                        'branch_J_secondary_effective_at_spawn': 0.0,
                        'branch_metric_ratio_at_spawn': 0.0,
                        'branch_spawn_step': -1,
                        'branch_spawn_reason': 'none',
                        # Stagnation/retirement bookkeeping. Retired fronts remain
                        # in the crack path/damage field but are no longer evaluated
                        # as active crack tips.
                        'last_fired_step': -1,
                        'stagnant_count': 0,
                        'stagnant_lag_m': 0.0,
                        'stagnant_K_ratio': np.inf,
                        'stagnant_lambda': 0.0,
                        'retire_step': -1,
                        'inactive_reason': 'active',
                        # Crack-network coalescence bookkeeping. A front is
                        # retired only after an actually committed increment
                        # intersects an existing crack segment.
                        'coalesced': False,
                        'merge_target_front_id': -1,
                        'merge_step': -1,
                        'merge_x_m': np.nan,
                        'merge_y_m': np.nan,
                        'starved_recorded': False}

            fronts = [_new_front([a_tip, 0.0], [1.0, 0.0], eng, cleave_planes[0],
                                 parent=-1, fid=0)]
            # Regions where a daughter branch was admitted but then proved
            # mechanically starved. New branch births are vetoed near these
            # zones so the parent cannot make a comb of repeated dead stubs.
            starved_branch_zones = []
            refine_center = fronts[0]['xy'].copy()
            refine_centers = np.array([refine_center.copy()])

            def _active():
                return [f for f in fronts if f['active']]

            def _find_front(fid):
                for ff in fronts:
                    if ff['id'] == fid:
                        return ff
                return None

            # Directional-J convention for the sharp-front/path-selection code.
            # compute_J_integral() still returns a legacy |J| K for older diagnostics,
            # but crack advance and branch birth must use only positive signed
            # configurational work relative to the root-crack convention.  Otherwise
            # a backward or badly oriented virtual extension can be converted from
            # negative J into a spurious positive K by abs(J).
            _J_SIGN_REF = {'sign': 0.0, 'step': -1, 'J_root_signed': 0.0}

            def _effective_JK_from_info(Jinfo):
                J_signed = float(Jinfo.get('J_signed', Jinfo.get('J', 0.0)) or 0.0)
                if bool(getattr(args, 'allow_abs_directional_J', False)):
                    J_eff = abs(J_signed)
                    sign_ref = 0.0
                else:
                    if _J_SIGN_REF['sign'] == 0.0 and abs(J_signed) > 1e-30:
                        _J_SIGN_REF['sign'] = 1.0 if J_signed > 0.0 else -1.0
                        _J_SIGN_REF['step'] = int(step)
                        _J_SIGN_REF['J_root_signed'] = J_signed
                    sign_ref = _J_SIGN_REF['sign'] if _J_SIGN_REF['sign'] != 0.0 else 1.0
                    J_eff = max(sign_ref * J_signed, 0.0)
                K_eff = float(np.sqrt(max(J_eff, 0.0) * mat.Eprime))
                try:
                    Jinfo['J_effective_signed_positive'] = float(J_eff)
                    Jinfo['KJ_effective_signed_positive'] = float(K_eff)
                    Jinfo['J_sign_ref'] = float(sign_ref)
                    Jinfo['J_sign_ref_step'] = int(_J_SIGN_REF['step'])
                    Jinfo['J_root_signed_ref'] = float(_J_SIGN_REF['J_root_signed'])
                except Exception:
                    pass
                return float(J_eff), K_eff, J_signed

            def _global_forward_ok(tvec):
                # For the edge-crack-through-ligament geometry the physically
                # admissible terminal extension should keep a positive component in
                # the ligament direction.  This rejects backward growth into the wake
                # and near-vertical side branches along the tensile axis.  Set
                # --min-global-forward <= -1 to disable for geometries where reverse
                # growth/back-branching is intentionally allowed.
                gmin = float(getattr(args, 'min_global_forward', 0.05) or 0.0)
                if gmin <= -1.0:
                    return True
                t = np.array(tvec, dtype=float)
                nrm = float(np.linalg.norm(t))
                if nrm <= 1e-30:
                    return False
                return float((t / nrm) @ fwd0) >= gmin

            def _filter_global_forward(cands):
                if not cands:
                    return []
                return [c for c in cands if _global_forward_ok(c.get('t', np.array([1.0, 0.0])))]

            def _front_arclen(front):
                p = front.get('path', [])
                if len(p) < 2:
                    return 0.0
                return float(sum(np.linalg.norm(np.array(p[i+1]) - np.array(p[i]))
                                 for i in range(len(p)-1)))

            def _dist_point_to_segments(pt, segs):
                pt = np.array(pt, float)
                best = np.inf
                for a, b in segs:
                    a = np.array(a, float); b = np.array(b, float)
                    v = b - a
                    L2 = float(v @ v)
                    if L2 <= 1e-30:
                        dpt = float(np.linalg.norm(pt - a))
                    else:
                        t = float(np.clip(((pt - a) @ v) / L2, 0.0, 1.0))
                        dpt = float(np.linalg.norm(pt - (a + t * v)))
                    best = min(best, dpt)
                return best

            def _front_segments(front):
                p = front.get('path', [])
                return [(p[i], p[i + 1]) for i in range(len(p) - 1)]

            def _ancestor_front(front):
                par = _find_front(front.get('birth_parent', -999))
                if par is None:
                    par = _find_front(front.get('parent', -999))
                return par

            def _unresolved_descendants(fid):
                out = []
                stack = [fid]
                seen = set()
                while stack:
                    pid = stack.pop()
                    if pid in seen:
                        continue
                    seen.add(pid)
                    kids = [ff for ff in fronts if ff.get('parent') == pid or ff.get('birth_parent') == pid]
                    for kk in kids:
                        if kk.get('active', True) and (not kk.get('resolved', True)):
                            out.append(kk)
                        stack.append(kk.get('id'))
                return out

            def _cluster_has_unresolved_near(front):
                # Suppress repeated births from a cluster until earlier daughters
                # are geometrically/J-domain separable.  This prevents a comb of
                # unresolved side stubs from one still-unresolved process zone.
                fid = front.get('id')
                if _unresolved_descendants(fid):
                    return True
                # Also block if this tip sits within another unresolved branch cluster.
                for ff in fronts:
                    if ff.get('id') == fid or ff.get('resolved', True):
                        continue
                    par = _ancestor_front(ff)
                    if par is None:
                        continue
                    # Same connected local cluster if the candidate tip is close to
                    # the unresolved branch or to its parent.
                    d1 = float(np.linalg.norm(front['xy'] - ff['xy']))
                    d2 = float(np.linalg.norm(front['xy'] - par['xy']))
                    if min(d1, d2) <= max(r_J_outer, float(getattr(args, 'branch_resolve_length', 0.0) or r_J_outer)):
                        return True
                return False

            def _local_J_domain_clean(front, h_local_current, probe=False):
                # A local J contour should not be trusted until it is populated and
                # not intersected/strongly contaminated by another crack/wake.  In v6
                # we also return the probed local K so branch handoff can require a
                # mechanically meaningful local driving force, not only a clean
                # geometric contour.
                local_clear = max(float(getattr(args, 'local_J_clearance_factor', 1.0) or 1.0) * r_J_outer,
                                  2.0 * da_phys)
                other = _segments_except({front['id']})
                dmin = _dist_point_to_segments(front['xy'], other) if other else np.inf
                if dmin < local_clear:
                    return False, 'overlap', 0, dmin, 0.0
                if not probe:
                    return True, 'clean_geom', 0, dmin, 0.0
                try:
                    _, _Kraw, Jinfo = compute_J_integral(
                        mesh, u, sigma_gp, psi_gp, d, front['xy'], front.get('fwd', np.array([1.0, 0.0])),
                        mat, ell=max(r_J_outer / 8.0, 1.25 * h_local_current),
                        crack_segments=_all_segments(), exclude_radius=2.0 * kill_r)
                    _Jeff, _Ktmp, _Jsigned = _effective_JK_from_info(Jinfo)
                    _Ktmp = max(float(_Ktmp), 0.0)
                    front['J_signed_probe'] = float(_Jsigned)
                    front['J_effective_probe'] = float(_Jeff)
                    ne = int(Jinfo.get('n_active_elements', 0))
                except Exception:
                    return False, 'probe_failed', 0, dmin, 0.0
                min_ne = int(getattr(args, 'min_J_active_elems', 12) or 12)
                if ne < min_ne:
                    return False, 'few_J_elements', ne, dmin, _Ktmp
                return True, 'clean', ne, dmin, _Ktmp

            def _refresh_branch_resolution(front):
                # Overlap-based parent/cluster handoff.  Branch arclength alone is
                # insufficient: a 20 um branch at 30 deg can still be inside a
                # 12 um local J contour around the parent wake.  Keep it in the
                # parent/cluster budget until the local J domain is geometrically
                # separable and populated.
                if front.get('resolved', True):
                    return
                Lh = float(getattr(args, 'branch_resolve_length', 0.0) or 0.0)
                if Lh <= 0.0:
                    Lh = max(2.0 * r_J_outer, 4.0 * da_phys)
                blen = _front_arclen(front)
                parent = _ancestor_front(front)
                parent_segs = _front_segments(parent) if parent is not None else []
                parent_sep = _dist_point_to_segments(front['xy'], parent_segs) if parent_segs else blen
                front['branch_len_m'] = blen
                front['parent_sep_m'] = parent_sep
                if blen < Lh:
                    front['cluster_hold_reason'] = 'short_branch'
                    return
                if parent_sep < max(r_J_outer, 0.5 * Lh):
                    front['cluster_hold_reason'] = 'parent_overlap'
                    return
                ok, reason, ne, dmin, Kloc_probe = _local_J_domain_clean(front, h_local, probe=True)
                front['J_active_elems_probe'] = ne
                front['nearest_crack_m'] = dmin
                front['KJ_local_probe'] = Kloc_probe
                if not ok:
                    front['cluster_hold_reason'] = reason
                    return
                # A clean contour is not sufficient if the probed local partial
                # driving force is essentially zero relative to the parent/cluster
                # drive.  In that case a newborn branch would become a resolved
                # but mechanically starved 20 um side stub.  Keep it in the cluster
                # budget until the local J-derived K is a meaningful fraction of
                # the parent cluster K.  Set --local-J-handoff-min-K-ratio 0 to
                # disable this handoff-validity check.
                parentK = 0.0 if parent is None else max(float(parent.get('KJ', parent.get('KJ_trial', 0.0))), 0.0)
                k_ratio_req = max(float(getattr(args, 'local_J_handoff_min_K_ratio', 0.25) or 0.0), 0.0)
                front['KJ_local_over_parent'] = (Kloc_probe / max(parentK, 1e-300)) if parentK > 0 else np.inf
                if k_ratio_req > 0.0 and parentK > 0.0 and Kloc_probe < k_ratio_req * parentK:
                    front['cluster_hold_reason'] = 'weak_local_J'
                    if not front.get('starved_recorded', False):
                        _record_starved_zone(front.get('birth_xy', front['xy']),
                                             parent_id=(parent.get('id', -1) if parent is not None else -1),
                                             child_id=front.get('id', -1), reason='weak_local_J_handoff')
                        front['starved_recorded'] = True
                    return
                front['resolved'] = True
                front['cluster_hold_reason'] = 'resolved'
                # Handoff the common process-zone memory from the parent if the
                # child ledger is lower. This prevents a resolved child from
                # becoming an artificial virgin crack after co-growing with the
                # parent inside the same process zone.
                if parent is not None:
                    w = float(front.get('birth_budget_w', front.get('spawn_weight', 0.5)) or 0.5)
                    front['eng'].N_em = max(front['eng'].N_em, w * parent['eng'].N_em)
                    front['eng'].W_emit = max(front['eng'].W_emit, w * parent['eng'].W_emit)
                    front['eng'].B = max(front['eng'].B, min(parent['eng'].B, 0.999) * w)

            def _all_segments():
                segs = []
                for f in fronts:
                    p = f['path']
                    segs += [(p[i], p[i + 1]) for i in range(len(p) - 1)]
                return segs

            def _segments_except(skip_ids=()):
                skip_ids = set(skip_ids or [])
                segs = []
                for f in fronts:
                    if f['id'] in skip_ids:
                        continue
                    p = f['path']
                    segs += [(p[i], p[i + 1]) for i in range(len(p) - 1)]
                return segs

            def _unresolved_children_of(fid):
                return [ff for ff in fronts if (not ff.get('resolved', True))
                        and ff.get('birth_parent') == fid and ff.get('active', True)]

            def _J_source_for_front(front):
                # Parent/root tips and parents with unresolved daughters use an
                # outer cluster J.  Only separated resolved daughters get their
                # own local J contour.  This implements the intended global-to-
                # local decomposition: one group energy-release rate for an
                # unresolved kink/branch cluster, local partial derivatives only
                # after separation.
                mode = str(getattr(args, 'j_decomposition', 'cluster') or 'cluster')
                if mode == 'local':
                    return 'local'
                if front.get('parent', -1) < 0:
                    return 'cluster'
                if _unresolved_descendants(front['id']):
                    return 'cluster'
                if not front.get('resolved', True):
                    return 'cluster'
                ok, _reason, _ne, _dmin, _Ktmp = _local_J_domain_clean(front, h_local, probe=False)
                return 'local' if ok else 'cluster'

            def _J_params_for_front(front):
                src = _J_source_for_front(front)
                if src == 'cluster':
                    # Do not let the branch-safe line-of-sight mask split the
                    # cluster by its own unresolved child stubs; the cluster
                    # contour is meant to enclose them as one process zone.
                    skip = {front['id']} | {c['id'] for c in _unresolved_descendants(front['id'])}
                    return src, max(r_J_cluster_ell, 3.0 * h_local), _segments_except(skip)
                return src, max(r_J_outer / 8.0, 1.25 * h_local), _all_segments()

            def _clip_to_domain(p0, p1):
                seg = p1 - p0
                tmax = 1.0
                for c, lo, hi in ((0, _xlo, _xhi), (1, _ylo, _yhi)):
                    if seg[c] > 1e-30:
                        tmax = min(tmax, (hi - p0[c]) / seg[c])
                    elif seg[c] < -1e-30:
                        tmax = min(tmax, (lo - p0[c]) / seg[c])
                tmax = max(tmax, 0.0)
                return p0 + tmax * seg

            def _first_coalescence_hit(front, p0, p1):
                if not bool(getattr(args, 'coalesce_cracks', True)):
                    return None
                return first_path_intersection(fronts, front, p0, p1)

            def _kill_segment(p0, p1):
                seg = p1 - p0
                L2 = float(seg @ seg) + 1e-30
                tt = np.clip(((cxe - p0[0]) * seg[0] + (cye - p0[1]) * seg[1]) / L2, 0.0, 1.0)
                ddx = cxe - (p0[0] + tt * seg[0]); ddy = cye - (p0[1] + tt * seg[1])
                rad = np.maximum(kill_r, 0.7 * elem_rad)
                d[mesh.elems[(ddx * ddx + ddy * ddy) <= rad ** 2]] = 1.0

            def _advance_polyline(front, direction, length):
                nonlocal mesh, bnd, d, u, x, y, cxe, cye, cx_e, cy_e, elem_rad, h_tip, kill_r, adj
                nonlocal rho_gp, ep_gp, pz_store_gp, pz_mobile_gp, pz_escape_gp, pz_emit_gp
                nonlocal sigma_gp, psi_gp, dot_ep, seq_gp, s1_gp
                front['last_advance_m'] = 0.0
                if length <= 0.0:
                    return 0.0
                if not _global_forward_ok(direction):
                    front['advance_veto_reason'] = 'not_global_forward'
                    front['inactive_reason'] = front.get('inactive_reason', 'active')
                    return 0.0
                p0 = front['xy'].copy()
                p1_req = _clip_to_domain(p0, p0 + float(length) * direction)
                req = float(np.linalg.norm(p1_req - p0))
                if req <= 0.0:
                    return 0.0

                # Exact crack-network coalescence: clip the physical increment
                # to the first existing crack-path intersection. No artificial
                # attraction is introduced; the rule acts only after the
                # mechanically selected proposed path actually intersects a wake.
                merge_hit = _first_coalescence_hit(front, p0, p1_req)
                if merge_hit is not None:
                    p1_req = np.asarray(merge_hit['xy'], float)
                    req = float(np.linalg.norm(p1_req - p0))
                    if req <= 1e-14:
                        return 0.0

                result = crack_backend.advance(
                    mesh=mesh, boundary=bnd, damage=d, displacement=u,
                    p0=p0, p1=p1_req, direction=np.asarray(direction, float),
                    front_id=int(front.get('id', 0)), kill_r=kill_r,
                )
                if not result.inserted or result.moved <= 0.0:
                    front['advance_veto_reason'] = str(result.reason)
                    front['czm_angle_error_deg'] = float(result.angle_error_deg)
                    return 0.0

                # Local h-refinement can add bulk elements.  In that case each
                # child element inherits the full material/process-zone history
                # of its parent old element.  This is exact for the piecewise-
                # constant Gauss-point state representation used by this solver.
                if result.elem_parent_map is not None:
                    parent_map = np.asarray(result.elem_parent_map, dtype=int)
                    rho_gp = np.ascontiguousarray(rho_gp[parent_map])
                    ep_gp = np.ascontiguousarray(ep_gp[:, parent_map])
                    pz_store_gp = np.ascontiguousarray(pz_store_gp[parent_map])
                    pz_mobile_gp = np.ascontiguousarray(pz_mobile_gp[parent_map])
                    pz_escape_gp = np.ascontiguousarray(pz_escape_gp[parent_map])
                    pz_emit_gp = np.ascontiguousarray(pz_emit_gp[parent_map])
                    # Current-step mechanics/plastic-rate fields were evaluated
                    # on the pre-refinement mesh.  Children inherit their parent
                    # value for energy accounting and same-step diagnostics; the
                    # next staggered solve recomputes them on the refined mesh.
                    sigma_gp = np.ascontiguousarray(sigma_gp[:, parent_map])
                    psi_gp = np.ascontiguousarray(psi_gp[parent_map])
                    dot_ep = np.ascontiguousarray(dot_ep[parent_map])
                    seq_gp = np.ascontiguousarray(seq_gp[parent_map])
                    s1_gp = np.ascontiguousarray(s1_gp[parent_map])

                mesh, bnd, d, u = result.mesh, result.boundary, result.damage, result.displacement
                moved = float(result.moved)
                # The backend may steer to a nearby mesh edge. The authoritative
                # geometric tip is therefore the actual selected endpoint.
                if crack_backend.name == 'sharp_wake':
                    p1_actual = p1_req
                else:
                    _log = crack_backend.advance_log[-1]
                    p1_actual = np.array([_log['x1'], _log['y1']], dtype=float)
                front['xy'] = p1_actual
                front['fwd'] = (p1_actual - p0) / max(moved, 1e-300)
                front['path'].append(p1_actual.copy())
                front['last_advance_m'] = moved
                front['adv_since_branch'] = front.get('adv_since_branch', 0.0) + moved
                front['n_geom_adv'] = front.get('n_geom_adv', 0.0) + moved / max(da_phys, 1e-300)
                front['advance_veto_reason'] = 'none'
                front['czm_angle_error_deg'] = float(result.angle_error_deg)

                if merge_hit is not None:
                    front['active'] = False
                    front['inactive_reason'] = 'coalesced'
                    front['coalesced'] = True
                    front['merge_target_front_id'] = int(merge_hit['target_front_id'])
                    front['merge_step'] = int(step)
                    front['merge_x_m'] = float(p1_actual[0])
                    front['merge_y_m'] = float(p1_actual[1])
                    front['retire_step'] = int(step)
                    print(f"  COALESCENCE front {front['id']} -> crack {front['merge_target_front_id']} "
                          f"at x={p1_actual[0]*1e3:.3f}mm y={p1_actual[1]*1e3:.3f}mm")

                # Topology-only node splitting keeps element count/order stable,
                # so ep/rho/PZ Gauss-point histories remain exactly attached.
                # Refresh only geometry-derived caches and adjacency.
                x = mesh.nodes[:, 0]; y = mesh.nodes[:, 1]
                cxe = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
                cye = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
                cx_e, cy_e = cxe, cye
                elem_rad = np.sqrt(np.maximum(mesh.area_e, 1e-30))
                h_tip = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
                kill_r = max(h_tip, 0.5e-6)
                if rho_transport_c > 0.0:
                    adj = build_elem_adjacency(mesh)

                if p1_actual[0] >= _xhi - 1e-9 or p1_actual[0] <= _xlo + 1e-9 \
                        or p1_actual[1] >= _yhi - 1e-9 or p1_actual[1] <= _ylo + 1e-9:
                    front['active'] = False
                    front['inactive_reason'] = 'domain_exit'
                    front['retire_step'] = int(step)
                return moved

            def _deposit_wake_at(point_xy, N_shed, engine_for_scale):
                if N_shed <= 0.0:
                    return
                r_dep = max(engine_for_scale.f.L_pz, mesh.hbar)
                near = (cx_e - point_xy[0]) ** 2 + (cy_e - point_xy[1]) ** 2 <= r_dep ** 2
                if near.any():
                    rho_gp[near] += float(N_shed) / (np.pi * r_dep ** 2)

            def _branch_metric(c):
                if c is None:
                    return 0.0
                return float(c.get('overdrive', max(c.get('sigma_nn', 0.0), 0.0)))

            def _branch_weights(cands, default_secondary=0.5):
                if not cands or len(cands) < 2:
                    return 1.0, 0.0
                mode = str(getattr(args, 'branch_share_mode', 'hazard') or 'hazard')
                if mode == 'equal':
                    return 0.5, 0.5
                m1 = max(_branch_metric(cands[0]), 0.0)
                m2 = max(_branch_metric(cands[1]), 0.0)
                if m1 <= 0.0 and m2 <= 0.0:
                    return 1.0 - default_secondary, default_secondary
                sharp = max(float(getattr(args, 'branch_hazard_sharpness', 2.0) or 2.0), 0.25)
                p1 = m1 ** sharp; p2 = m2 ** sharp
                tot = p1 + p2 + 1e-300
                return float(p1 / tot), float(p2 / tot)

            def _branch_angle_deg(c):
                try:
                    return float(np.rad2deg(np.arctan2(c['t'][1], c['t'][0])))
                except Exception:
                    return np.nan

            def _angle_close_deg(a, b, tol):
                if not np.isfinite(a) or not np.isfinite(b):
                    return False
                d = abs(((a - b + 180.0) % 360.0) - 180.0)
                return d <= tol

            def _front_waveform(Kmax):
                return FatigueWaveform(
                    Kmax=max(float(Kmax), 0.0),
                    R=float(getattr(args, 'R', 0.1) or 0.0),
                    frequency_Hz=float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3),
                    closure_clip=not bool(getattr(args, 'no_closure_clip', False)),
                )

            def _fatigue_cycles_remaining():
                if not fatigue_mode:
                    return float('inf')
                cmax = float(getattr(args, 'cycles_max', float('inf')) or float('inf'))
                if not np.isfinite(cmax):
                    return float('inf')
                return max(cmax - float(fatigue_cycles_total_accepted), 0.0)

            def _diag_with_remaining(pred, req):
                # The physical horizon is the only mandatory cap in hazard_limited
                # mode.  This lets VHCF cases jump directly to the requested cycle
                # horizon if all hazard/process-zone rates are negligible, while
                # still shrinking the block when any monitored clock accumulates
                # too quickly.  A finite --max-block-cycles remains available as
                # an optional convergence/debug cap; set it to inf to remove it.
                rem = _fatigue_cycles_remaining()
                old_max = float(fatigue_controller.cfg.max_block_cycles)
                try:
                    if np.isfinite(rem):
                        if np.isfinite(old_max):
                            fatigue_controller.cfg.max_block_cycles = min(old_max, rem)
                        else:
                            fatigue_controller.cfg.max_block_cycles = rem
                    return fatigue_controller.choose_block_cycles_diagnostic(pred, req)
                finally:
                    fatigue_controller.cfg.max_block_cycles = old_max

            def _fatigue_global_cycles(active_fronts, T_local):
                """Choose one cycle block shared by all independently active fronts.

                V1 and v8 must use the same cycle-integrated hazard kernel, but a
                2-D multi-front run has a single physical cycle count.  We therefore
                predict the per-cycle hazard for each resolved active front, take the
                most restrictive adaptive block, then force every front to commit
                exactly that many cycles.  This prevents one branch from silently
                jumping through a million cycles while another front sees only a
                few fractional cycles.
                """
                if (not fatigue_mode) or fatigue_controller is None:
                    return 0.0
                req = float(getattr(args, 'block_cycles', getattr(args, 'fatigue_block_cycles', 1.0e4)) or 1.0)
                rem0 = _fatigue_cycles_remaining()
                if rem0 <= 0.0:
                    return 0.0
                maxb0 = float(getattr(args, 'max_block_cycles', req) or req)
                if str(getattr(args, 'cycle_block_mode', 'requested_cap')).lower() == 'hazard_limited':
                    base0 = maxb0 if np.isfinite(maxb0) else rem0
                else:
                    base0 = req
                cycles = float(min(base0, rem0) if np.isfinite(rem0) else base0)
                any_pred = False
                global_limiter = 'none'
                global_unlimited = cycles
                for ff in active_fronts:
                    if not ff.get('resolved', True):
                        continue
                    Kff = max(float(ff.get('KJ_trial', ff.get('KJ', 0.0))), 0.0)
                    _sync_front_engine_from_pz(ff)
                    pred = fatigue_controller.integrate_one_cycle(ff['eng'], _front_waveform(Kff), T_local)
                    ff['fatigue_pred_trial'] = pred
                    ff['fatigue_wave_trial'] = _front_waveform(Kff)
                    diag = _diag_with_remaining(pred, req)
                    ff['fatigue_cycle_limiter_trial'] = str(diag.get('limiter', 'unknown'))
                    ff['fatigue_cycle_unlimited_trial'] = float(diag.get('unlimited_cycles', diag.get('cycles', 0.0)))
                    cand = float(diag.get('cycles', 0.0))
                    if cand <= cycles:
                        cycles = cand
                        global_limiter = f"front{ff.get('id','?')}:{ff['fatigue_cycle_limiter_trial']}"
                        global_unlimited = float(ff['fatigue_cycle_unlimited_trial'])
                    any_pred = True
                if not any_pred:
                    return 0.0
                for ff in active_fronts:
                    ff['fatigue_global_cycle_limiter'] = global_limiter
                    ff['fatigue_global_cycle_unlimited'] = global_unlimited
                return max(float(cycles), 0.0)

            def _record_starved_zone(point_xy, parent_id=-1, child_id=-1, reason='starved'):
                rad = float(getattr(args, 'branch_starved_suppression_radius', 80e-6) or 0.0)
                if rad <= 0.0:
                    return
                pz = np.array(point_xy, float)
                for z in starved_branch_zones:
                    if float(np.linalg.norm(pz - z['xy'])) < 0.5 * rad:
                        return
                starved_branch_zones.append({'xy': pz.copy(), 'parent_id': int(parent_id),
                                             'child_id': int(child_id), 'step': int(step),
                                             'radius': rad, 'reason': str(reason)})

            def _near_starved_zone(point_xy):
                if not starved_branch_zones:
                    return False
                pz = np.array(point_xy, float)
                for z in starved_branch_zones:
                    if float(np.linalg.norm(pz - z['xy'])) <= float(z.get('radius', 0.0)):
                        return True
                return False

            def _update_branch_clock(front, dt_local, T_local, cycles_local=None):
                """Accumulate a first-passage clock for the secondary branch lobe.

                Branching is no longer admitted by a ratio alone. The secondary
                lobe receives an absolute cleavage hazard from a true secondary
                directional-J probe at the parent tip.  Birth can occur only if this
                signed-positive J probe is admissible and this clock reaches
                threshold while the parent/front also fires.
                """
                front['branch_clock_ready'] = False
                front['branch_dB_last'] = 0.0
                front['branch_lambda_secondary'] = 0.0
                front['branch_K_secondary'] = 0.0
                front['branch_J_secondary_signed'] = 0.0
                front['branch_J_secondary_effective'] = 0.0
                front['branch_J_secondary_active_elems'] = 0
                front['branch_metric_ratio'] = 0.0
                front['branch_veto_reason'] = 'none'
                cands = front.get('cands') or []
                if (not branch_on) or len(cands) < 2:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'no_secondary'
                    return False
                if _near_starved_zone(front['xy']):
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'near_starved_branch'
                    return False
                m1 = max(_branch_metric(cands[0]), 0.0)
                m2 = max(_branch_metric(cands[1]), 0.0)
                if m1 <= 0.0 or m2 <= 0.0:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'nonpositive_metric'
                    return False
                ratio = float(m2 / max(m1, 1e-300))
                front['branch_metric_ratio'] = ratio
                ratio_req = max(float(getattr(args, 'branch_fp_min_ratio', 0.95) or 0.0), 0.0)
                if ratio < ratio_req:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'ratio_below_fp_threshold'
                    return False
                # Reset the branch clock if the secondary lobe changes direction;
                # otherwise a different plane can inherit an old near-critical clock.
                ang = _branch_angle_deg(cands[1])
                tol = max(float(getattr(args, 'branch_clock_angle_tol_deg', 15.0) or 15.0), 0.0)
                old_ang = float(front.get('branch_clock_angle_deg', np.nan))
                if not _angle_close_deg(ang, old_ang, tol):
                    front['branch_B'] = 0.0
                    front['branch_clock_angle_deg'] = ang
                K_parent = max(float(front.get('KJ', front.get('KJ_trial', 0.0))), 0.0)
                if not _global_forward_ok(cands[1].get('t', np.array([1.0, 0.0]))):
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_not_global_forward'
                    return False
                # Absolute branch-birth driving force comes from a true secondary
                # directional-J probe in the current FEM field, not from K_parent
                # multiplied by the local opening-stress/overdrive ratio.
                try:
                    srcJ_sec, ellJ_sec, segsJ_sec = _J_params_for_front(front)
                    _, _Kraw_sec, Jinfo_sec = compute_J_integral(
                        mesh, u, sigma_gp, psi_gp, d, front['xy'], cands[1]['t'],
                        mat, ell=ellJ_sec,
                        crack_segments=segsJ_sec, exclude_radius=2.0 * kill_r)
                    Jeff_sec, K_sec, Jsig_sec = _effective_JK_from_info(Jinfo_sec)
                    front['branch_J_secondary_signed'] = float(Jsig_sec)
                    front['branch_J_secondary_effective'] = float(Jeff_sec)
                    front['branch_J_secondary_active_elems'] = int(Jinfo_sec.get('n_active_elements', 0))
                except Exception:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_J_probe_failed'
                    return False
                front['branch_K_secondary'] = float(K_sec)
                if K_sec <= 0.0:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_signed_J_nonpositive'
                    return False
                k_ratio_req = max(float(getattr(args, 'branch_secondary_min_K_ratio', 0.85) or 0.0), 0.0)
                k_abs_req = max(float(getattr(args, 'branch_secondary_min_K_MPa', 0.0) or 0.0), 0.0) * 1e6
                if K_parent > 0.0 and K_sec < k_ratio_req * K_parent:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_K_ratio_low'
                    return False
                if K_sec < k_abs_req:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_K_abs_low'
                    return False
                sig_sec = front['eng'].sigma_tip(K_sec)
                lam_sec, _lam_raw, _Geff = front['eng'].lambda_cleave(sig_sec, T_local)
                lam_sec = max(float(lam_sec), 0.0)
                lam_min = max(float(getattr(args, 'branch_secondary_min_lambda', 0.0) or 0.0), 0.0)
                front['branch_lambda_secondary'] = lam_sec
                if lam_sec < lam_min:
                    front['branch_B'] = 0.0
                    front['branch_veto_reason'] = 'secondary_lambda_low'
                    return False
                if fatigue_mode and fatigue_controller is not None:
                    cyc = max(float(cycles_local if cycles_local is not None
                                    else front.get('fatigue_cycles_last', 0.0)), 0.0)
                    if cyc <= 0.0:
                        dB = 0.0
                    else:
                        pred_sec = fatigue_controller.integrate_one_cycle(
                            front['eng'], _front_waveform(K_sec), T_local)
                        dB = pred_sec.mu_cleave * cyc
                        # In fatigue mode the branch secondary clock reports the
                        # equivalent rate averaged over the cycling time.
                        lam_sec = pred_sec.mu_cleave * float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3)
                        front['branch_lambda_secondary'] = lam_sec
                else:
                    dB = lam_sec * max(float(dt_local), 0.0)
                front['branch_dB_last'] = dB
                front['branch_B'] = float(front.get('branch_B', 0.0)) + dB
                target = max(float(getattr(args, 'branch_clock_target', 1.0) or 1.0), 1e-12)
                if front['branch_B'] >= target:
                    front['branch_clock_ready'] = True
                    front['branch_veto_reason'] = 'ready'
                    return True
                front['branch_veto_reason'] = 'clock_not_ready'
                return False

            def _record_mechanically_starved_fronts():
                Lh = float(getattr(args, 'branch_resolve_length', 0.0) or 0.0)
                if Lh <= 0.0:
                    Lh = max(2.0 * r_J_outer, 4.0 * da_phys)
                max_len = max(float(getattr(args, 'branch_starved_max_length_factor', 4.0) or 4.0) * Lh,
                              Lh + 2.0 * da_phys)
                kr = max(float(getattr(args, 'branch_starved_K_ratio', 0.25) or 0.0), 0.0)
                lam_thr = max(float(getattr(args, 'branch_starved_lambda', 1e-20) or 0.0), 0.0)
                for ff in fronts:
                    if ff.get('id', 0) == 0 or ff.get('starved_recorded', False):
                        continue
                    bl = _front_arclen(ff)
                    if bl > max_len:
                        continue
                    par = _ancestor_front(ff)
                    parentK = 0.0 if par is None else max(float(par.get('KJ', par.get('KJ_trial', 0.0))), 0.0)
                    Krel = (max(float(ff.get('KJ', 0.0)), 0.0) / max(parentK, 1e-300)) if parentK > 0 else np.inf
                    lam = 0.0
                    if ff.get('info') is not None:
                        lam = max(float(ff['info'].get('lambda_c', 0.0)), 0.0)
                    if (parentK > 0 and Krel < kr) or (lam_thr > 0 and lam < lam_thr):
                        _record_starved_zone(ff.get('birth_xy', ff['xy']),
                                             parent_id=ff.get('birth_parent', ff.get('parent', -1)),
                                             child_id=ff.get('id', -1), reason='weak_resolved_branch')
                        ff['starved_recorded'] = True


            def _retire_stagnant_branches():
                """Deactivate branches that are mechanically stagnant and far behind.

                This is a resource-management rule, not a crack-growth cap.  The
                crack segment remains in the damage field and in all crack-segment
                masks, but its terminal tip is no longer evaluated once it is both
                strongly shielded by a leading front and persistently unable to
                accumulate cleavage hazard.
                """
                if not bool(getattr(args, 'retire_stagnant_branches', False)):
                    return 0
                active_fronts = _active()
                if len(active_fronts) <= 1:
                    return 0
                lead = max(active_fronts, key=lambda ff: ff['xy'][0])
                leadK = max(float(lead.get('KJ', lead.get('KJ_trial', 0.0))), 0.0)
                Lh = float(getattr(args, 'branch_resolve_length', 0.0) or 0.0)
                if Lh <= 0.0:
                    Lh = max(2.0 * r_J_outer, 4.0 * da_phys)
                lag_thr = max(float(getattr(args, 'branch_stagnant_lag', 0.0) or 0.0), 0.0)
                if lag_thr <= 0.0:
                    lag_thr = max(4.0 * Lh, 50.0 * da_phys)
                kr = max(float(getattr(args, 'branch_stagnant_K_ratio', 0.20) or 0.0), 0.0)
                lam_thr = max(float(getattr(args, 'branch_stagnant_lambda', 1e-30) or 0.0), 0.0)
                no_fire_steps = max(int(getattr(args, 'branch_stagnant_no_fire_steps', 80) or 0), 0)
                persist_steps = max(int(getattr(args, 'branch_stagnant_steps', 20) or 0), 1)
                min_len = max(float(getattr(args, 'branch_stagnant_min_length', 0.0) or 0.0), 0.0)
                if min_len <= 0.0:
                    min_len = max(Lh, 5.0 * da_phys)
                retired = 0
                for ff in active_fronts:
                    if ff.get('id', 0) == 0:
                        continue
                    if not ff.get('resolved', True):
                        # Unresolved daughters still belong to a parent cluster and
                        # may co-grow; do not retire them by independent-tip criteria.
                        ff['stagnant_count'] = 0
                        continue
                    bl = _front_arclen(ff)
                    if bl < min_len:
                        ff['stagnant_count'] = 0
                        continue
                    lag = max(float(lead['xy'][0] - ff['xy'][0]), 0.0)
                    Kff = max(float(ff.get('KJ', ff.get('KJ_trial', 0.0))), 0.0)
                    krel = (Kff / max(leadK, 1e-300)) if leadK > 0.0 else 1.0
                    lam = 0.0
                    if ff.get('info') is not None:
                        lam = max(float(ff['info'].get('lambda_c', 0.0)), 0.0)
                    last_fire = int(ff.get('last_fired_step', -1))
                    no_recent_fire = (last_fire < 0) or ((int(step) - last_fire) >= no_fire_steps)
                    weak_K = (kr <= 0.0) or (krel <= kr)
                    weak_lam = (lam_thr <= 0.0) or (lam <= lam_thr)
                    far_behind = lag >= lag_thr
                    ff['stagnant_lag_m'] = lag
                    ff['stagnant_K_ratio'] = krel
                    ff['stagnant_lambda'] = lam
                    if far_behind and weak_K and weak_lam and no_recent_fire:
                        ff['stagnant_count'] = int(ff.get('stagnant_count', 0)) + 1
                    else:
                        ff['stagnant_count'] = 0
                    if ff['stagnant_count'] >= persist_steps:
                        ff['active'] = False
                        ff['inactive_reason'] = 'stagnant_branch'
                        ff['retire_step'] = int(step)
                        _record_starved_zone(ff.get('birth_xy', ff['xy']),
                                             parent_id=ff.get('birth_parent', ff.get('parent', -1)),
                                             child_id=ff.get('id', -1), reason='stagnant_branch_retired')
                        ff['starved_recorded'] = True
                        retired += 1
                return retired

        # Snapshot capture uses a bounded online buffer rather than a cadence
        # derived from ``args.steps``.  First-passage runs often terminate far
        # earlier than the nominal step horizon; a fixed horizon-based cadence
        # would then save only the initial and fracture states.  We initially
        # capture every accepted step and progressively thin the buffer while
        # increasing the capture stride.  This preserves snapshots distributed
        # across the *actual* run duration with bounded memory.
        snapshot_target = max(int(getattr(args, 'save_snapshots', 0) or 0), 0)
        snapshot_buffer_limit = max(8, 4 * max(snapshot_target, int(getattr(args, 'snapshot_cols', 1) or 1)))
        snapshot_stride = 1
        dt_cur = cfg.loading.dt
        adaptive_events = bool(getattr(args, 'adaptive_events', False)) or (crack_backend.name != 'sharp_wake')
        adaptive_target = max(float(getattr(args, 'adaptive_event_target', 0.35) or 0.35), 1e-6)
        if crack_backend.name != 'sharp_wake':
            # A cohesive topology update is an event surface.  Keep accepted clock
            # increments below unity so at most one renewal/mesh edge is consumed
            # by an accepted monotonic step.  The excess hazard is not discarded;
            # the adaptive controller lands on successive event surfaces.
            adaptive_target = min(adaptive_target, 0.8)
        adaptive_min_frac = max(float(getattr(args, 'adaptive_min_frac', 1e-8) or 1e-8), 1e-12)
        adaptive_safety = float(np.clip(getattr(args, 'adaptive_safety', 0.7) or 0.7, 0.05, 0.95))
        adaptive_grow = max(float(getattr(args, 'adaptive_grow', 4.0) or 4.0), 1.0)
        step = 0
        Uapp_accepted = 0.0
        carry_frac = 1.0
        fatigue_cycles_total_accepted = 0.0
        fatigue_cycles_max = float(getattr(args, 'cycles_max', float('inf')) or float('inf'))
        crack_extension_start_a = float(a_tip)
        target_crack_extension_m = float(getattr(args, 'target_crack_extension_um', float('inf')) or float('inf')) * 1e-6
        snapshot_by_ext_m = max(float(getattr(args, 'snapshot_by_crack_extension_um', 0.0) or 0.0) * 1e-6, 0.0)
        next_snapshot_ext_m = snapshot_by_ext_m if snapshot_by_ext_m > 0.0 else float('inf')
        max_da_per_block_m = float(getattr(args, 'max_da_per_block_um', float('inf')) or float('inf')) * 1e-6
        prev_a_tip_for_block = float(a_tip)

        while step < args.steps:
            if fatigue_mode and np.isfinite(fatigue_cycles_max) and fatigue_cycles_total_accepted >= fatigue_cycles_max - 1e-12 * max(fatigue_cycles_max, 1.0):
                print(f"  [T={T:.0f}K] reached fatigue cycle horizon {fatigue_cycles_total_accepted:.6g} cycles")
                break
            trial_frac = min(1.0, carry_frac * adaptive_grow) if adaptive_events else 1.0
            while True:
                u_saved = u.copy()
                ep_saved = ep_gp.copy()
                rho_saved = rho_gp.copy()
                Uapp_saved = Uapp_accepted
                step_trial = step + 1
                dt_cur = cfg.loading.dt * trial_frac
                # In fatigue comparison/tuning runs it is often useful to
                # ramp to a target amplitude once and then spend subsequent
                # accepted steps as additional cycle blocks at the same Kmax.
                # This is a load-hold test harness, not a propagation cap: the
                # existing v8 first-passage advance/branching laws still decide
                # whether the front moves.
                if fatigue_mode and bool(getattr(args, 'fatigue_hold_load', False)) and step > 0:
                    dU_step = 0.0
                else:
                    dU_step = cfg.loading.dU_top * trial_frac
                Uapp = Uapp_saved + dU_step
                Uy_top, Uy_bot = 0.5 * Uapp, -0.5 * Uapp

                sigma_gp = np.zeros((3, mesh.ne)); psi_gp = np.zeros(mesh.ne); Ftop = 0.0
                for it in range(args.n_stagger):
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
                    u, Ftop = solve_dirichlet(Kmat, Rint, u, bnd, Uy_top, Uy_bot)
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat, cohesive_network=cohesive_network)
                    if fatigue_mode and cyclic_mechanics_enabled:
                        # In fatigue mode the plastic strain/dislocation state is
                        # advanced by the explicit cyclic mechanics block after the
                        # Kmax/J predictor chooses an accepted cycle count.  Do not
                        # also apply a monotonic max-load plastic increment here.
                        dot_ep = np.zeros(mesh.ne)
                    else:
                        ep_gp, rho_gp, dot_ep = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations)

                h_local = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
                pred_primary = 0.0
                if deflect:
                    predicted_clock = 0.0
                    for fi, f in enumerate(_active()):
                        if not f.get('resolved', True):
                            # Unresolved daughters remain inside the parent J/process-zone
                            # domain and are advanced only when their parent fires.
                            continue
                        sig2 = near_tip_stress_tensor(sigma_gp, mesh, f['xy'], 3.0 * h_local)
                        gate_fwd = fwd0 if gate_global else f['fwd']
                        if compete:
                            sel, _all = cleave_direction_competition(
                                sig2, _theta, gate_fwd, min_forward=0.2,
                                gamma_aniso=gamma_aniso, branch_ratio=r_branch_O)
                            cands = sel if sel else [f['last_plane']]
                        else:
                            cands = cleavage_branch_candidates(
                                sig2, cleave_planes, forward=gate_fwd, min_forward=0.2,
                                branch_ratio=branch_ratio)
                            if not cands:
                                cands = [f['last_plane']]
                        cands = _filter_global_forward(cands)
                        if not cands:
                            cands = [dict(f.get('last_plane', {}), t=fwd0.copy(),
                                          n=np.array([0.0, 1.0]), sigma_nn=0.0,
                                          overdrive=0.0, gamma_rel=1.0,
                                          name='global_forward_fallback')]
                        f['cands_trial'] = cands
                        f['win_trial'] = cands[0]
                        f['t_trial'] = cands[0]['t']
                        srcJ, ellJ, segsJ = _J_params_for_front(f)
                        _, _Kraw, Jinfo = compute_J_integral(
                            mesh, u, sigma_gp, psi_gp, d, f['xy'], f['t_trial'],
                            mat, ell=ellJ,
                            crack_segments=segsJ, exclude_radius=2.0 * kill_r)
                        _Jeff, KJf, _Jsigned = _effective_JK_from_info(Jinfo)
                        f['J_signed_trial'] = float(_Jsigned)
                        f['J_effective_trial'] = float(_Jeff)
                        f['J_sign_ref'] = float(Jinfo.get('J_sign_ref', 0.0))
                        f['KJ_trial'] = max(KJf, 0.0)
                        f['J_source_trial'] = srcJ
                        f['J_source_code_trial'] = 0 if srcJ == 'cluster' else 1
                        f['J_active_elems_trial'] = int(Jinfo.get('n_active_elements', 0))
                        if fatigue_mode:
                            pc = 0.0
                        else:
                            pc = f['eng'].predict_clock_increment(f['KJ_trial'], T, dt_cur)
                        predicted_clock += pc
                        if f['id'] == 0:
                            pred_primary = pc
                    fatigue_cycles_trial = _fatigue_global_cycles(_active(), T) if fatigue_mode else 0.0
                    if fatigue_mode:
                        predicted_clock = 0.0
                        for ff in _active():
                            pred = ff.get('fatigue_pred_trial')
                            if pred is not None:
                                dBff = float(pred.mu_cleave) * fatigue_cycles_trial
                                predicted_clock += dBff
                                if ff.get('id') == 0:
                                    pred_primary = dBff
                    KJ = fronts[0]['KJ_trial'] if fronts[0]['active'] else \
                        (fronts[0].get('KJ', 0.0))
                else:
                    J, KJ, _ = compute_J_integral(
                        mesh, u, sigma_gp, psi_gp, d, np.array([a_tip, 0.0]),
                        np.array([1.0, 0.0]), mat, ell=max(r_J_cluster_ell, 3.0 * h_local),
                        crack_segments=_backend_crack_segments())
                    KJ = max(KJ, 0.0)
                    if fatigue_mode:
                        wave_trial = FatigueWaveform(
                            Kmax=max(float(KJ), 0.0),
                            R=float(getattr(args, 'R', 0.1) or 0.0),
                            frequency_Hz=float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3),
                            closure_clip=not bool(getattr(args, 'no_closure_clip', False)),
                        )
                        pred_single_trial = fatigue_controller.integrate_one_cycle(eng, wave_trial, T)
                        diag_single_trial = _diag_with_remaining(
                            pred_single_trial, float(getattr(args, 'block_cycles', 1.0e4) or 1.0))
                        fatigue_cycles_trial = float(diag_single_trial.get('cycles', 0.0))
                        fatigue_cycle_limiter_trial = str(diag_single_trial.get('limiter', 'unknown'))
                        fatigue_cycle_unlimited_trial = float(diag_single_trial.get('unlimited_cycles', fatigue_cycles_trial))
                        predicted_clock = pred_single_trial.mu_cleave * fatigue_cycles_trial
                    else:
                        predicted_clock = eng.predict_clock_increment(KJ, T, dt_cur)
                    pred_primary = predicted_clock

                if adaptive_events and predicted_clock > adaptive_target and trial_frac > adaptive_min_frac:
                    u = u_saved; ep_gp = ep_saved; rho_gp = rho_saved; Uapp = Uapp_saved
                    shrink = adaptive_safety * adaptive_target / max(predicted_clock, 1e-300)
                    trial_frac = max(adaptive_min_frac, min(0.5 * trial_frac, trial_frac * shrink))
                    continue
                break

            step = step_trial
            Uapp_accepted = Uapp
            carry_frac = trial_frac
            adaptive_frac_used = trial_frac
            adaptive_pred_clock_total = predicted_clock
            fatigue_cycles_accepted = float(locals().get('fatigue_cycles_trial', 0.0)) if fatigue_mode else 0.0

            # If requested, resolve the full 2-D body through the accepted cycle
            # block before the front hazards are committed.  This uses the old
            # elastoplastic Arrhenius/Taylor machinery rather than replacing it
            # with a scalar K(t) surrogate.  The crack geometry remains fixed here;
            # only the existing sharp-front renewal law below can advance it.
            if fatigue_mode and cyclic_mechanics_enabled and fatigue_cycles_accepted > 0.0:
                _run_cyclic_mechanics_block(Uapp, fatigue_cycles_accepted, T)
                # Recompute J-derived Kmax from the post-cycle residual stress and
                # plastic/dislocation state.  Keep the already accepted cycle block
                # size so this is a mechanics correction, not a hidden adaptive gate.
                if deflect:
                    predicted_clock = 0.0; pred_primary = 0.0
                    for fi, f in enumerate(_active()):
                        if not f.get('resolved', True):
                            continue
                        sig2 = near_tip_stress_tensor(sigma_gp, mesh, f['xy'], 3.0 * h_local)
                        gate_fwd = fwd0 if gate_global else f['fwd']
                        if compete:
                            sel, _all = cleave_direction_competition(
                                sig2, _theta, gate_fwd, min_forward=0.2,
                                gamma_aniso=gamma_aniso, branch_ratio=r_branch_O)
                            cands = sel if sel else [f['last_plane']]
                        else:
                            cands = cleavage_branch_candidates(
                                sig2, cleave_planes, forward=gate_fwd, min_forward=0.2,
                                branch_ratio=branch_ratio)
                            if not cands:
                                cands = [f['last_plane']]
                        cands = _filter_global_forward(cands)
                        if not cands:
                            cands = [dict(f.get('last_plane', {}), t=fwd0.copy(),
                                          n=np.array([0.0, 1.0]), sigma_nn=0.0,
                                          overdrive=0.0, gamma_rel=1.0,
                                          name='global_forward_fallback')]
                        f['cands_trial'] = cands
                        f['win_trial'] = cands[0]
                        f['t_trial'] = cands[0]['t']
                        srcJ, ellJ, segsJ = _J_params_for_front(f)
                        _, _Kraw, Jinfo = compute_J_integral(
                            mesh, u, sigma_gp, psi_gp, d, f['xy'], f['t_trial'],
                            mat, ell=ellJ, crack_segments=segsJ, exclude_radius=2.0 * kill_r)
                        _Jeff, KJf, _Jsigned = _effective_JK_from_info(Jinfo)
                        f['J_signed_trial'] = float(_Jsigned)
                        f['J_effective_trial'] = float(_Jeff)
                        f['J_sign_ref'] = float(Jinfo.get('J_sign_ref', 0.0))
                        f['KJ_trial'] = max(float(KJf), 0.0)
                        f['J_source_trial'] = srcJ
                        f['J_source_code_trial'] = 0 if srcJ == 'cluster' else 1
                        f['J_active_elems_trial'] = int(Jinfo.get('n_active_elements', 0))
                        _sync_front_engine_from_pz(f)
                        pred = fatigue_controller.integrate_one_cycle(f['eng'], _front_waveform(f['KJ_trial']), T)
                        f['fatigue_pred_trial'] = pred
                        f['fatigue_wave_trial'] = _front_waveform(f['KJ_trial'])
                        dBff = float(pred.mu_cleave) * fatigue_cycles_accepted
                        predicted_clock += dBff
                        if f.get('id') == 0:
                            pred_primary = dBff
                    KJ = fronts[0]['KJ_trial'] if fronts[0]['active'] else fronts[0].get('KJ', 0.0)
                else:
                    J, KJ, _ = compute_J_integral(
                        mesh, u, sigma_gp, psi_gp, d, np.array([a_tip, 0.0]),
                        np.array([1.0, 0.0]), mat, ell=max(r_J_cluster_ell, 3.0 * h_local),
                        crack_segments=_backend_crack_segments())
                    KJ = max(float(KJ), 0.0)
                    pred_single_trial = fatigue_controller.integrate_one_cycle(eng, FatigueWaveform(
                        Kmax=max(float(KJ), 0.0),
                        R=float(getattr(args, 'R', 0.1) or 0.0),
                        frequency_Hz=float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3),
                        closure_clip=not bool(getattr(args, 'no_closure_clip', False))), T)
                    predicted_clock = pred_single_trial.mu_cleave * fatigue_cycles_accepted
                    pred_primary = predicted_clock
                adaptive_pred_clock_total = predicted_clock

            newborns = []

            if deflect:
                for f in _active():
                    f['born_this_step'] = False
                    f['last_advance_m'] = 0.0
                    f['birth_budget_w'] = 0.0
                    f['birth_parent'] = f.get('birth_parent', -1)
                    f.pop('birth_child_ids', None)
                    f.pop('birth_self_w', None)
                    if not f.get('resolved', True):
                        parent = _find_front(f.get('birth_parent', -999))
                        f['KJ'] = 0.0 if parent is None else parent.get('KJ_trial', parent.get('KJ', 0.0))
                        f['J_source'] = 'unresolved_parent_cluster'
                        f['J_source_code'] = 2
                        f['J_active_elems'] = 0
                        f['t_win'] = f.get('fwd', np.array([1.0, 0.0]))
                        f['win_plane'] = f.get('last_plane')
                        f['cands'] = [f.get('last_plane')]
                        f['info'] = {'fired': False, 'n_fire': 0, 'lambda_c': 0.0,
                                     'lambda_e': 0.0, 'B': float(f['eng'].B),
                                     'N_em': float(f['eng'].N_em),
                                     'sigma_tip': 0.0, 'sigma_back': 0.0,
                                     'r_eff': f['eng'].r_eff(),
                                     'N_em_pre_renewal': float(f['eng'].N_em),
                                     'N_em_shed_to_wake': 0.0}
                        f['fired'] = False
                        continue
                    f['KJ'] = f['KJ_trial']; f['t_win'] = f['t_trial']
                    f['J_source'] = f.get('J_source_trial', _J_source_for_front(f))
                    f['J_source_code'] = int(f.get('J_source_code_trial', 0 if f['J_source'] == 'cluster' else 1))
                    f['J_active_elems'] = int(f.get('J_active_elems_trial', 0))
                    f['win_plane'] = f['win_trial']; f['cands'] = f['cands_trial']
                    if fatigue_mode:
                        _sync_front_engine_from_pz(f)
                        f['info'] = fatigue_controller.cycle_step_front(
                            f['eng'], _front_waveform(f['KJ']), T,
                            requested_cycles=fatigue_cycles_accepted,
                            force_cycles=fatigue_cycles_accepted)
                        f['info']['cycle_limiter'] = str(f.get('fatigue_global_cycle_limiter', f.get('fatigue_cycle_limiter_trial', f['info'].get('cycle_limiter', 'unknown'))))
                        f['info']['cycle_unlimited'] = float(f.get('fatigue_global_cycle_unlimited', f.get('fatigue_cycle_unlimited_trial', f['info'].get('cycle_unlimited', fatigue_cycles_accepted))))
                        f['fatigue_cycles_last'] = fatigue_cycles_accepted
                        _commit_front_pz_fields(f, f['info'])
                    else:
                        f['info'] = f['eng'].step(f['KJ'], T, dt_cur)
                        f['fatigue_cycles_last'] = 0.0
                    f['fired'] = bool(f['info']['fired'])
                    if f['fired']:
                        f['last_fired_step'] = int(step)
                info = fronts[0]['info']
                if info is None:      # primary retired: synthesize a quiet record
                    info = {'sigma_tip': 0.0, 'sigma_back': 0.0, 'lambda_c': 0.0,
                            'lambda_e': 0.0, 'B': 0.0, 'N_em': fronts[0]['eng'].N_em,
                            'n_fire': 0, 'r_eff': fronts[0]['eng'].r_eff(),
                            'N_em_pre_renewal': fronts[0]['eng'].N_em,
                            'N_em_shed_to_wake': 0.0}
                KJ = fronts[0]['KJ']

                # Integrate branch-specific first-passage clocks for the secondary
                # lobe of every resolved front. A branch can only be born later if
                # this clock is ready AND the parent/front cleavage clock fired.
                if branch_on:
                    for _bf in _active():
                        if _bf.get('resolved', True) and _bf.get('info') is not None:
                            _update_branch_clock(_bf, dt_cur, T, cycles_local=fatigue_cycles_accepted)

                # ---- branch birth: ANY co-critical firing front may branch ----
                if branch_on:
                    branch_spacing_m = float(getattr(args, 'branch_spacing', 10.0)
                                             or 0.0) * da_phys
                    for f in _active():
                        if (not f.get('resolved', True)) or (not f['fired']):
                            continue
                        # branch-spacing gate: a front cannot re-branch until it has
                        # propagated a minimum distance since birth / its last branch.
                        # Without this a stationary tip with two co-critical cleavage
                        # planes re-branches every step (a degenerate branch storm).
                        if f.get('adv_since_branch', 0.0) < branch_spacing_m:
                            continue
                        # New v5: do not spawn a new daughter from a cluster while an
                        # earlier daughter is still inside the same unresolved J domain.
                        if _cluster_has_unresolved_near(f):
                            continue
                        # New v7: do not repeatedly spawn near a known dead/stalled
                        # side-branch process zone, and require a branch-specific
                        # first-passage clock rather than a ratio-only criterion.
                        if _near_starved_zone(f['xy']):
                            f['branch_veto_reason'] = 'near_starved_branch'
                            continue
                        if not f.get('branch_clock_ready', False):
                            continue
                        cands = f['cands']
                        if not (cands and len(cands) >= 2):
                            continue
                        # Cap simultaneously active tips rather than the
                        # cumulative number ever created. Coalesced or retired
                        # daughters therefore free capacity for a later branch,
                        # while morphology remains bounded at max_fronts.
                        n_active_now = sum(1 for ff in fronts if ff.get('active', True))
                        if n_active_now + len(newborns) >= max_fronts:
                            break
                        second = cands[1]
                        w1, w2 = _branch_weights(cands)
                        if w2 <= 1e-6:
                            continue
                        ratio = _branch_metric(cands[1]) / max(_branch_metric(cands[0]), 1e-300)
                        # Preserve pre-consumption branch-clock state for diagnostics.
                        branch_B_before = float(f.get('branch_B', 0.0))
                        f['branch_B_before_spawn'] = branch_B_before
                        f['branch_dB_at_spawn'] = float(f.get('branch_dB_last', 0.0))
                        f['branch_lambda_secondary_at_spawn'] = float(f.get('branch_lambda_secondary', 0.0))
                        f['branch_K_secondary_at_spawn'] = float(f.get('branch_K_secondary', 0.0))
                        f['branch_J_secondary_signed_at_spawn'] = float(f.get('branch_J_secondary_signed', 0.0))
                        f['branch_J_secondary_effective_at_spawn'] = float(f.get('branch_J_secondary_effective', 0.0))
                        f['branch_metric_ratio_at_spawn'] = float(f.get('branch_metric_ratio', ratio))
                        f['branch_spawn_step'] = int(step)
                        f['branch_spawn_reason'] = 'secondary_first_passage'
                        # Consume one completed branch first-passage event. Any
                        # fractional excess remains, but cap it below threshold so
                        # the same front cannot immediately generate a second branch
                        # without renewed secondary hazard accumulation.
                        _btarget = max(float(getattr(args, 'branch_clock_target', 1.0) or 1.0), 1e-12)
                        f['branch_B'] = max(float(f.get('branch_B', 0.0)) - _btarget, 0.0)
                        f['branch_B'] = min(f['branch_B'], 0.99 * _btarget)
                        f['branch_clock_ready'] = False
                        child_eng = f['eng'].clone_split(w2)
                        child = _new_front(f['xy'], second['t'], child_eng, second,
                                           parent=f['id'], spawn_step=step,
                                           spawn_w=w2, ratio=ratio, fid=_next_id[0])
                        _next_id[0] += 1
                        child['resolved'] = False
                        child['birth_xy'] = f['xy'].copy()
                        if hasattr(crack_backend, 'register_branch_front'):
                            crack_backend.register_branch_front(f['id'], child['id'], child['birth_xy'])
                        child['J_source'] = 'unresolved_parent_cluster'
                        child['J_source_code'] = 2
                        child['cluster_id'] = f['id']
                        child['born_this_step'] = True
                        child['birth_parent'] = f['id']
                        child['birth_budget_w'] = w2
                        child['branch_B_before_spawn'] = branch_B_before
                        child['branch_dB_at_spawn'] = float(f.get('branch_dB_at_spawn', 0.0))
                        child['branch_lambda_secondary_at_spawn'] = float(f.get('branch_lambda_secondary_at_spawn', 0.0))
                        child['branch_K_secondary_at_spawn'] = float(f.get('branch_K_secondary_at_spawn', 0.0))
                        child['branch_J_secondary_signed_at_spawn'] = float(f.get('branch_J_secondary_signed_at_spawn', 0.0))
                        child['branch_J_secondary_effective_at_spawn'] = float(f.get('branch_J_secondary_effective_at_spawn', 0.0))
                        child['branch_metric_ratio_at_spawn'] = float(f.get('branch_metric_ratio_at_spawn', ratio))
                        child['branch_spawn_step'] = int(step)
                        child['branch_spawn_reason'] = 'born_from_parent_secondary_first_passage'
                        f['birth_self_w'] = w1
                        f.setdefault('birth_child_ids', []).append(child['id'])
                        f['adv_since_branch'] = 0.0     # parent must re-propagate too
                        lam1 = max(float(f['info'].get('lambda_c', 0.0)), 0.0)
                        child['info'] = {
                            'fired': bool(f['info'].get('fired', False)),
                            'n_fire': int(f['info'].get('n_fire', 0)),
                            'lambda_c': lam1 * w2 / max(w1, 1e-300), 'lambda_e': 0.0,
                            'B': float(child_eng.B), 'N_em': float(child_eng.N_em),
                            'N_em_shed_to_wake': 0.0,
                            'sigma_tip': f['info'].get('sigma_tip', 0.0),
                            'sigma_back': f['info'].get('sigma_back', 0.0),
                            'r_eff': child_eng.r_eff(),
                            'N_em_pre_renewal': float(child_eng.N_em)}
                        child['fired'] = True
                        child['KJ'] = f['KJ']; child['t_win'] = second['t']
                        newborns.append(child)
                        print(f"  BRANCH at ({f['xy'][0]*1e3:.3f},{f['xy'][1]*1e3:.3f})mm "
                              f"from front {f['id']}: O2/O1={ratio:.3f}, "
                              f"split=({w1:.2f},{w2:.2f}) -> front {child['id']}")
                    if newborns:
                        fronts.extend(newborns)

                # ---- advance all fired fronts with LOCAL event budgets ----
                fired_ids = {f['id'] for f in fronts if f.get('fired') and f['info'] is not None}
                # Unresolved daughters are still inside the parent's local J/process-zone
                # domain. If the parent fires, they co-grow under that parent's finite
                # event budget until the handoff length is reached.
                for f in fronts:
                    if (not f.get('resolved', True)) and f.get('birth_parent') in fired_ids:
                        parent = _find_front(f.get('birth_parent'))
                        if parent is not None and parent.get('info') is not None:
                            f['fired'] = True
                            f['last_fired_step'] = int(step)
                            f['info'] = dict(parent['info'])
                            f['info']['lambda_c'] = parent['info'].get('lambda_c', 0.0) * float(f.get('birth_budget_w', 0.5))
                            f['KJ'] = parent.get('KJ', 0.0)
                            f['J_source'] = 'unresolved_parent_cluster'
                            f['J_source_code'] = 2
                            f['cluster_id'] = parent.get('id', f.get('cluster_id', -1))
                fired_fronts = [f for f in fronts if f.get('fired') and f['info'] is not None]
                if fired_fronts and Kc_first is None:
                    Kc_first = (fronts[0]['KJ'] if fronts[0].get('fired')
                                else fired_fronts[0]['KJ'])
                    Kc_first_step = len(hist['sigma_back'])
                share = str(getattr(args, 'branch_energy_share', 'hazard-budget'))
                lens = {}

                # Earlier multi-front code split ONE global da_phys*n_fire budget
                # among every front that happened to fire in the same accepted
                # load/time step.  That starves branches as soon as several fronts
                # are active: each tip receives a sub-mesh, visually invisible
                # increment and then its J contour sees only a stub.  The energy
                # conservation constraint is local to a bifurcation: a parent and
                # its newborn daughter must split the PARENT event budget.
                # Independent fronts that each completed their own renewal clock
                # have independent local event budgets and should not globally
                # divide one crack increment.
                allocated_newborn_ids = set()
                for f in fired_fronts:
                    if f.get('born_this_step'):
                        # Birth-step daughters are allocated from their parent
                        # below; if the parent was retired or not in fired_fronts,
                        # fall back to the daughter's own budget after this loop.
                        continue
                    parent_budget = da_phys * float(max(int(f['info'].get('n_fire', 1)), 1))
                    children = [c for c in fired_fronts
                                if (not c.get('resolved', True)) and c.get('birth_parent') == f['id']]
                    if share != 'none' and children:
                        w_self = float(f.get('birth_self_w', 1.0))
                        child_ws = [max(float(c.get('birth_budget_w', 0.0)), 0.0) for c in children]
                        total_w = max(w_self + sum(child_ws), 1e-300)
                        lens[id(f)] = parent_budget * w_self / total_w
                        for c, wc in zip(children, child_ws):
                            lens[id(c)] = parent_budget * wc / total_w
                            allocated_newborn_ids.add(c['id'])
                    else:
                        lens[id(f)] = parent_budget

                for f in fired_fronts:
                    if id(f) not in lens:
                        # A pathological orphan daughter should almost never occur,
                        # but do not silently pin it.  Give it its own local clock
                        # budget rather than a global shared budget.
                        lens[id(f)] = da_phys * float(max(int(f['info'].get('n_fire', 1)), 1))
                for f in fired_fronts:
                    moved_now = _advance_polyline(f, f['t_win'], lens[id(f)])
                    if moved_now <= 0.0:
                        # The Arrhenius clock reached first passage but the mesh
                        # backend could not realize the topology event.  Do NOT
                        # consume that physical event.  Restore the completed
                        # renewal to the clock and undo only the renewal/wake
                        # bookkeeping; emission/time evolution remains accepted.
                        info_f = f.get('info') or {}
                        n_restore = max(int(info_f.get('n_fire', 1)), 1)
                        eng_f = f['eng']
                        if hasattr(eng_f, 'restore_geometry_veto'):
                            eng_f.restore_geometry_veto(n_restore)
                        else:
                            eng_f.B += float(n_restore)
                            if 'N_em_pre_renewal' in info_f:
                                eng_f.N_em = float(info_f['N_em_pre_renewal'])
                            eng_f.a_adv = max(float(eng_f.a_adv) - eng_f.f.da * n_restore, 0.0)
                            eng_f.n_adv = max(int(eng_f.n_adv) - n_restore, 0)
                        info_f['geometry_vetoed'] = True
                        info_f['fired'] = False
                        info_f['n_fire'] = 0
                        info_f['B'] = float(eng_f.B)
                        info_f['N_em'] = float(eng_f.N_em)
                        f['fired'] = False
                        print(f"  GEOMETRY VETO front {f['id']}: {f.get('advance_veto_reason','unknown')} "
                              f"-- renewal retained in B={eng_f.B:.3f}")
                for f in fronts:
                    _refresh_branch_resolution(f)
                _record_mechanically_starved_fronts()
                _retire_stagnant_branches()
                if fronts:
                    a_killed = max(a_killed, max(f['xy'][0] for f in fronts))
                    lead = max(fronts, key=lambda f: f['xy'][0])
                    tip_xy = lead['xy']; fwd = lead['fwd']; a_tip = float(lead['xy'][0])

                # ---- per-front source deposit ----
                if src_rho_per_emit > 0.0:
                    def _deposit_source_at(point_xy, dN_local, engine_for_scale):
                        if dN_local <= 0.0:
                            return
                        r_dep = max(engine_for_scale.f.L_pz, mesh.hbar)
                        near = (cx_e - point_xy[0]) ** 2 + (cy_e - point_xy[1]) ** 2 <= r_dep ** 2
                        if near.any():
                            rho_gp[near] += src_rho_per_emit * dN_local / (np.pi * r_dep ** 2)
                    for f in fronts:
                        if f['info'] is None:
                            continue
                        dN = max(float(f['info']['N_em']) - float(f.get('N_em_prev', 0.0)), 0.0) \
                            + float(f['info'].get('N_em_shed_to_wake', 0.0))
                        _deposit_source_at(f['xy'], dN, f['eng'])
                        f['N_em_prev'] = float(f['info']['N_em'])
                # ---- per-front wake deposit ----
                for f in fronts:
                    if f['info'] is None:
                        continue
                    _deposit_wake_at(f['xy'], float(f['info'].get('N_em_shed_to_wake', 0.0)), f['eng'])

                # ---- per-front diagnostic rows + back-compat branch row ----
                for f in fronts:
                    if f['info'] is None:
                        continue
                    _reason_map = {'root':0, 'short_branch':1, 'parent_overlap':2,
                                   'overlap':3, 'few_J_elements':4, 'probe_failed':5,
                                   'clean_geom':6, 'resolved':7, 'weak_local_J':8}
                    fronts_rows.append((step, f['id'], f['parent'], f['xy'][0], f['xy'][1],
                                        f['KJ'], f['info']['B'], f['info']['N_em'],
                                        f['info']['lambda_c'], int(f['info'].get('n_fire', 0)),
                                        1 if f['active'] else 0,
                                        1 if f.get('resolved', True) else 0,
                                        float(f.get('branch_len_m', _front_arclen(f))),
                                        int(f.get('J_source_code', -1)),
                                        int(f.get('cluster_id', f.get('id', -1))),
                                        int(f.get('J_active_elems', 0)),
                                        float(f.get('J_signed_trial', np.nan)),
                                        float(f.get('J_effective_trial', np.nan)),
                                        float(f.get('J_sign_ref', np.nan)),
                                        float(f.get('parent_sep_m', np.nan)),
                                        float(f.get('nearest_crack_m', np.nan)),
                                        int(_reason_map.get(str(f.get('cluster_hold_reason','')), -1)),
                                        float(f.get('n_geom_adv', 0.0)),
                                        float(f.get('KJ_local_probe', np.nan)),
                                        float(f.get('KJ_local_over_parent', np.nan)),
                                        float(f.get('branch_B', 0.0)),
                                        float(f.get('branch_dB_last', 0.0)),
                                        float(f.get('branch_lambda_secondary', 0.0)),
                                        float(f.get('branch_K_secondary', 0.0)),
                                        float(f.get('branch_J_secondary_signed', 0.0)),
                                        float(f.get('branch_J_secondary_effective', 0.0)),
                                        float(f.get('branch_metric_ratio', 0.0)),
                                        1 if f.get('branch_clock_ready', False) else 0,
                                        len(starved_branch_zones),
                                        float(f.get('branch_B_before_spawn', 0.0)),
                                        float(f.get('branch_dB_at_spawn', 0.0)),
                                        float(f.get('branch_lambda_secondary_at_spawn', 0.0)),
                                        float(f.get('branch_K_secondary_at_spawn', 0.0)),
                                        float(f.get('branch_J_secondary_signed_at_spawn', 0.0)),
                                        float(f.get('branch_J_secondary_effective_at_spawn', 0.0)),
                                        float(f.get('branch_metric_ratio_at_spawn', 0.0)),
                                        float(f.get('stagnant_lag_m', 0.0)),
                                        float(f.get('stagnant_K_ratio', np.nan)),
                                        float(f.get('stagnant_lambda', 0.0)),
                                        int(f.get('stagnant_count', 0)),
                                        int(f.get('retire_step', -1)),
                                        1 if str(f.get('inactive_reason', 'active')) == 'stagnant_branch' else 0,
                                        1 if bool(f.get('coalesced', False)) else 0,
                                        int(f.get('merge_target_front_id', -1)),
                                        int(f.get('merge_step', -1)),
                                        float(f.get('merge_x_m', np.nan)),
                                        float(f.get('merge_y_m', np.nan)),
                                        float(f['info'].get('cycles', f.get('fatigue_cycles_last', 0.0))),
                                        float(f['info'].get('dB_block', 0.0)),
                                        float(f['info'].get('dN_store_block', 0.0)),
                                        float(f['info'].get('dN_mobile_block', 0.0)),
                                        float(f['info'].get('dN_escape_block', 0.0)),
                                        float(f['info'].get('dN_peierls_block', 0.0)),
                                        float(f['info'].get('dN_taylor_block', 0.0)),
                                        float(f['info'].get('storage_fraction', 0.0)),
                                        float(f['info'].get('mu_emit', 0.0)),
                                        float(f['info'].get('mu_escape', 0.0)),
                                        float(f.get('pz_store_count', 0.0)),
                                        float(f.get('pz_mobile_count', 0.0)),
                                        float(f.get('pz_escape_count', 0.0))))
                daughters = [f for f in fronts if f['id'] != 0]
                dlead = max(daughters, key=lambda f: f['xy'][0]) if daughters else None
                branch_rows.append((
                    step, Uapp, fronts[0]['KJ'],
                    dlead['KJ'] if dlead else np.nan,
                    len(fronts[0].get('cands') or []),
                    float(fronts[0]['win_plane'].get('angle_deg', np.nan)),
                    (float(dlead['win_plane'].get('angle_deg', np.nan)) if dlead else np.nan),
                    _branch_metric(fronts[0].get('win_plane')),
                    (_branch_metric(dlead['win_plane']) if dlead else 0.0),
                    (dlead['spawn_ratio'] if dlead else 0.0),
                    1 if daughters else 0, 1 if newborns else 0,
                    1.0, (dlead['spawn_weight'] if dlead else 0.0),
                    int(fronts[0]['info'].get('n_fire', 0)) if fronts[0]['info'] else 0,
                    int(dlead['info'].get('n_fire', 0)) if (dlead and dlead['info']) else 0,
                    float(fronts[0].get('last_advance_m', 0.0)),
                    (float(dlead.get('last_advance_m', 0.0)) if dlead else 0.0),
                    fronts[0]['xy'][0], fronts[0]['xy'][1],
                    dlead['xy'][0] if dlead else np.nan,
                    dlead['xy'][1] if dlead else np.nan,
                    fronts[0]['info']['lambda_c'] if fronts[0]['info'] else np.nan,
                    dlead['info']['lambda_c'] if (dlead and dlead['info']) else np.nan,
                    int(newborns[0]['parent']) if newborns else -1,
                    int(newborns[0]['id']) if newborns else -1,
                    float((_find_front(newborns[0]['parent']).get('branch_B_before_spawn', 0.0)
                           if (newborns and _find_front(newborns[0]['parent']) is not None) else 0.0)),
                    float((newborns[0].get('branch_lambda_secondary_at_spawn', 0.0) if newborns else 0.0)),
                    float((newborns[0].get('branch_K_secondary_at_spawn', 0.0) if newborns else 0.0)),
                    float((newborns[0].get('branch_metric_ratio_at_spawn', 0.0) if newborns else 0.0))))
            else:
                if fatigue_mode:
                    wave_acc = FatigueWaveform(
                        Kmax=max(float(KJ), 0.0),
                        R=float(getattr(args, 'R', 0.1) or 0.0),
                        frequency_Hz=float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3),
                        closure_clip=not bool(getattr(args, 'no_closure_clip', False)),
                    )
                    _pseudo_front = {'xy': np.array([a_tip, 0.0]), 'eng': eng}
                    _sync_front_engine_from_pz(_pseudo_front)
                    info = fatigue_controller.cycle_step_front(
                        eng, wave_acc, T, requested_cycles=fatigue_cycles_accepted,
                        force_cycles=fatigue_cycles_accepted)
                    info['cycle_limiter'] = str(locals().get('fatigue_cycle_limiter_trial', info.get('cycle_limiter', 'unknown')))
                    info['cycle_unlimited'] = float(locals().get('fatigue_cycle_unlimited_trial', info.get('cycle_unlimited', fatigue_cycles_accepted)))
                    _commit_front_pz_fields(_pseudo_front, info)
                else:
                    info = eng.step(KJ, T, dt_cur)
                if info['fired']:
                    if Kc_first is None:
                        Kc_first = KJ; Kc_first_step = len(hist['sigma_back'])
                    n_events = max(int(info.get('n_fire', 1)), 1)
                    for _iev in range(n_events):
                        if crack_backend.name == 'sharp_wake':
                            len1_m = da_phys
                            a_tip = min(a_tip + len1_m, cfg.geometry.Lx - 2e-5)
                            if a_tip - a_killed >= mesh.hbar or a_tip >= cfg.geometry.Lx - 3e-5:
                                band = ((x > a_killed - 0.5 * mesh.hbar) & (x <= a_tip)
                                        & (np.abs(y) <= half_h))
                                d[band] = 1.0
                                a_killed = a_tip
                        else:
                            p0 = np.array([a_tip, 0.0])
                            p1_req = np.array([min(a_tip + da_phys, cfg.geometry.Lx - 2e-5), 0.0])
                            rr = crack_backend.advance(
                                mesh=mesh, boundary=bnd, damage=d, displacement=u,
                                p0=p0, p1=p1_req, direction=np.array([1.0, 0.0]),
                                front_id=0, kill_r=max(mesh.hbar_tip, 0.5e-6))
                            if not rr.inserted:
                                print(f"  CZM advance vetoed: {rr.reason} angle_error={rr.angle_error_deg:.2f} deg")
                                break
                            mesh, bnd, d, u = rr.mesh, rr.boundary, rr.damage, rr.displacement
                            _log = crack_backend.advance_log[-1]
                            a_tip = float(_log['x1'])
                            x = mesh.nodes[:, 0]; y = mesh.nodes[:, 1]
                            cx_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
                            cy_e = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
                            if rho_transport_c > 0.0:
                                adj = build_elem_adjacency(mesh)

            # ---- density transport (common to both paths) ----
            if rho_transport_c > 0.0 and adj is not None:
                D_e = rho_transport_c * np.maximum(dot_ep, 0.0) * (eng.f.L_pz ** 2)
                rho_gp = transport_rho_step(rho_gp, adj, mesh.area_e, D_e, dt_cur)
                rho_gp = np.clip(rho_gp, 1e6, cfg.dislocations.rho_cap)

            # Spatial PZ fields are true fields, not front-only scalars.  Mobile
            # content can recover/escape and, when the existing transport tool is
            # enabled, move conservatively with the same plasticity-weighted
            # mobility used for rho_gp.
            if pz_spatial_state:
                freq_eff = max(float(getattr(args, 'frequency_Hz', 1.0e3) or 1.0e3), 1e-300)
                block_time = (fatigue_cycles_accepted / freq_eff) if fatigue_mode else dt_cur
                rec_m = max(float(getattr(args, 'pz_mobile_recovery_per_s', 0.0) or 0.0), 0.0)
                if rec_m > 0.0 and block_time > 0.0:
                    pz_mobile_gp *= np.exp(-min(rec_m * block_time, 80.0))
                pz_tr = getattr(args, 'pz_field_transport_c', None)
                pz_tr = rho_transport_c if pz_tr is None else float(pz_tr)
                if pz_tr > 0.0 and adj is not None:
                    Dpz = pz_tr * np.maximum(dot_ep, 0.0) * (eng.f.L_pz ** 2)
                    pz_mobile_gp = np.maximum(transport_rho_step(pz_mobile_gp, adj, mesh.area_e, Dpz, dt_cur), 0.0)
                    pz_store_gp = np.maximum(transport_rho_step(pz_store_gp, adj, mesh.area_e, 0.25 * Dpz, dt_cur), 0.0)

            W_emit_primary = fronts[0]['eng'].W_emit if deflect else eng.W_emit
            # Long-growth state diagnostics used by the rows below.  These are
            # defined for both monotonic/calibration runs and fatigue-cycle runs.
            crack_extension_m = max(float(a_tip) - float(crack_extension_start_a), 0.0)
            da_block_m = max(float(a_tip) - float(prev_a_tip_for_block), 0.0)

            pz_store_total = float(np.sum(pz_store_gp * mesh.area_e)) if pz_spatial_state else 0.0
            pz_mobile_total = float(np.sum(pz_mobile_gp * mesh.area_e)) if pz_spatial_state else 0.0
            pz_escape_total = float(np.sum(pz_escape_gp * mesh.area_e)) if pz_spatial_state else 0.0
            pz_emit_total = float(np.sum(pz_emit_gp * mesh.area_e)) if pz_spatial_state else 0.0
            rows.append((step, Uapp, Ftop, KJ, info['sigma_tip'], info['sigma_back'],
                         info['lambda_c'], info['lambda_e'], info['B'], info['N_em'],
                         a_tip, crack_extension_m, da_block_m, W_emit_primary, info['n_fire'],
                         info['N_em_pre_renewal'], info['N_em_shed_to_wake'],
                         adaptive_frac_used, dt_cur,
                         pred_primary, max(adaptive_pred_clock_total - pred_primary, 0.0),
                         adaptive_pred_clock_total,
                         float(info.get('cycles', fatigue_cycles_accepted if fatigue_mode else 0.0)),
                         float(info.get('cycle_unlimited', fatigue_cycles_accepted if fatigue_mode else 0.0)),
                         # string limiters are encoded into a small numeric code for np.savetxt:
                         # 0 unknown/fixed, 1 cleavage, 2 store, 3 emit, 4 mobile, 5 escape, 6 peierls, 7 taylor, 8 block/max, 9 min.
                         float({'cleavage_clock':1,'stored_pz':2,'emitted_pz':3,'mobile_pz':4,'escape_pz':5,'peierls_clock':6,'taylor_clock':7,'block_cycles':8,'max_block_cycles':8,'min_block_cycles':9}.get(str(info.get('cycle_limiter','')).split(':')[-1],0)),
                         float(info.get('dB_block', 0.0)),
                         float(info.get('dN_store_block', 0.0)),
                         float(info.get('dN_mobile_block', 0.0)),
                         float(info.get('dN_escape_block', 0.0)),
                         float(info.get('dN_peierls_block', 0.0)),
                         float(info.get('dN_taylor_block', 0.0)),
                         float(info.get('storage_fraction', 0.0)),
                         float(info.get('mu_emit', 0.0)),
                         float(info.get('mu_escape', 0.0)),
                         float(info.get('mu_cleave_pred', 0.0)),
                         pz_store_total, pz_mobile_total, pz_escape_total, pz_emit_total,
                         int(cyclic_mechanics_updates), cyclic_plastic_work_acc,
                         float(info.get('G_cleave_raw_eV', info.get('G_cleave_eff_eV', 0.0))),
                         float(info.get('G_cleave_eff_eV', 0.0)),
                         float(info.get('S_cleave_kB', 0.0)),
                         float(info.get('dGcleave_dsigma_eV_per_GPa', 0.0)),
                         float(info.get('vstar_cleave_b3', 0.0)),
                         float(info.get('sigma_cleave_eff_Pa', info.get('sigma_tip', 0.0))),
                         float(info.get('cleave_barrier_kind_code', 0.0)),
                         float(info.get('front_state_model_code', 0.0)),
                         float(info.get('mpz_K_shield_Pa_sqrt_m', 0.0)),
                         float(info.get('mpz_mobile_count', 0.0)),
                         float(info.get('mpz_retained_count', info.get('N_em', 0.0))),
                         float(info.get('mpz_available_site_fraction', 1.0)),
                         float(info.get('mpz_local_slip_count', 0.0)),
                         float(info.get('mpz_escaped_total', 0.0)),
                         float(info.get('mpz_recovered_total', 0.0)),
                         float(info.get('mpz_wake_retained_total', 0.0))))
            if fatigue_mode:
                fatigue_cycles_total_accepted += max(float(fatigue_cycles_accepted), 0.0)

            if np.isfinite(max_da_per_block_m) and max_da_per_block_m > 0.0 and da_block_m > max_da_per_block_m:
                print(f"  [T={T:.0f}K] WARNING crack advanced {da_block_m*1e6:.3g} um in one accepted block; \
                      reduce --target-dB or --target-da-per-block-um for tighter long-growth resolution")
            prev_a_tip_for_block = float(a_tip)

            W_ext_acc += 0.5 * (Ftop + Ftop_prev) * (Uapp - Uapp_prev)
            Ftop_prev, Uapp_prev = Ftop, Uapp
            U_el = float(np.sum(psi_gp * mesh.area_e))
            dWp = float(np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)) * dt_cur
            W_p_acc += max(dWp, 0.0)
            rstat = rho_gp[rho_gp > 0]
            if rstat.size == 0:
                rstat = np.array([eng.f.rho0])
            n_fronts_now = len(fronts) if deflect else 1
            W_emit_tot = (sum(f['eng'].W_emit for f in fronts) if deflect else eng.W_emit)
            hist['Uapp'].append(Uapp); hist['Ftop'].append(Ftop)
            hist['KJ'].append(KJ); hist['W_ext'].append(W_ext_acc)
            hist['U_el'].append(U_el); hist['W_p'].append(W_p_acc)
            hist['W_emit'].append(W_emit_tot)
            hist['rho_mean'].append(float(np.mean(rstat)))
            hist['rho_p95'].append(float(np.percentile(rstat, 95)))
            hist['rho_p99'].append(float(np.percentile(rstat, 99)))
            hist['rho_max'].append(float(np.max(rstat)))
            hist['lambda_c'].append(info['lambda_c']); hist['lambda_e'].append(info['lambda_e'])
            hist['B'].append(info['B']); hist['n_fire'].append(info['n_fire'])
            hist['sigma_tip'].append(info['sigma_tip'])
            hist['sigma_back'].append(info['sigma_back'])
            hist['r_eff_over_r0'].append(info['r_eff'] / eng.f.r0)
            hist['N_em'].append(info['N_em']); hist['a_tip'].append(a_tip)
            hist['n_fronts'].append(n_fronts_now)

            any_fired = (any(f.get('fired') for f in fronts) if deflect else info['fired'])
            snapshot_ext_due = snapshot_by_ext_m > 0.0 and crack_extension_m + 1e-18 >= next_snapshot_ext_m
            snapshots_enabled = snapshot_target > 0 or snapshot_by_ext_m > 0.0
            regular_snapshot_due = snapshot_target > 0 and (step == 1 or (step % snapshot_stride) == 0)
            if snapshots_enabled and (regular_snapshot_due or any_fired or snapshot_ext_due or step == args.steps):
                ep_xx, ep_yy, ep_xy = ep_gp[0], ep_gp[1], ep_gp[2]
                epeq = np.sqrt(2.0 / 3.0 * (ep_xx**2 + ep_yy**2 + 2.0 * ep_xy**2))
                snap_front_paths = []
                if deflect:
                    try:
                        snap_front_paths = [(f['id'], f['parent'], np.array(f['path'], float).copy())
                                            for f in fronts if len(f.get('path', [])) >= 1]
                    except Exception:
                        snap_front_paths = []
                mpz_front_states = []
                if not bool(getattr(args, 'no_mpz_state_output', False)):
                    state_fronts = fronts if deflect else [{'id': 0, 'parent': -1,
                                                            'xy': np.array([a_tip, 0.0]),
                                                            'direction': np.array([1.0, 0.0]),
                                                            'active': True, 'eng': eng}]
                    for sf_state in state_fronts:
                        sf_eng = sf_state.get('eng')
                        if getattr(sf_eng, 'state_model', 'legacy_scalar') == 'moving_pz' and hasattr(sf_eng, 'export_process_zone_state'):
                            mpz_front_states.append({
                                'front_id': int(sf_state.get('id', 0)),
                                'parent_id': int(sf_state.get('parent', -1)),
                                'xy_m': np.asarray(sf_state.get('xy', [a_tip, 0.0]), float).tolist(),
                                'direction': np.asarray(sf_state.get('direction', [1.0, 0.0]), float).tolist(),
                                'active': bool(sf_state.get('active', True)),
                                'state': sf_eng.export_process_zone_state(),
                            })
                snaps.append({
                    'step': step, 'KJ': KJ, 'N_em': info['N_em'], 'a_tip': a_tip,
                    'd': d.copy(), 'rho_gp': rho_gp.copy(),
                    's1_gp': s1_gp.copy(), 'epeq_gp': epeq.copy(),
                    'nodes': mesh.nodes.copy(), 'elems': mesh.elems.copy(),
                    'front_paths': snap_front_paths,
                    'mpz_front_states': mpz_front_states,
                })
                if snapshot_ext_due:
                    while next_snapshot_ext_m <= crack_extension_m + 1e-18:
                        next_snapshot_ext_m += snapshot_by_ext_m

                # Keep memory bounded while retaining coverage of the full
                # accepted-step history.  Always preserve the most recent
                # snapshot, including a first-passage event snapshot.
                if snapshot_target > 0 and len(snaps) > snapshot_buffer_limit:
                    latest_snap = snaps[-1]
                    thinned_snaps = snaps[::2]
                    if thinned_snaps[-1] is not latest_snap:
                        thinned_snaps.append(latest_snap)
                    snaps[:] = thinned_snaps
                    snapshot_stride *= 2
            if step % args.print_every == 0 or any_fired:
                nf_str = f"  nfr={n_fronts_now}" if deflect else ""
                print(f"  [T={T:.0f}K] step {step:4d}  KJ={KJ/1e6:7.3f}  "
                      f"sig_tip={info['sigma_tip']/1e9:6.2f}GPa  B={info['B']:7.3f}  "
                      f"N_em={info['N_em']:9.2f}  a={a_tip*1e3:.3f}mm{nf_str}"
                      + ("  << ADVANCE" if any_fired else ""))

            # Optional diagnostic stopping criterion used by the 1-D/2-D K-sweep
            # comparison harness.  The multifront production driver keeps running
            # unless this flag is explicitly set.
            if bool(getattr(args, 'stop_after_first_fire', False)) and any_fired:
                print(f"  [T={T:.0f}K] stopping after first fire at step {step} "
                      f"for diagnostic comparison")
                break

            # Optional diagnostic stopping criterion used by the 1-D/2-D K-sweep
            # comparison harness. The production multifront driver keeps running
            # unless this flag is explicitly set.
            if bool(getattr(args, 'stop_after_first_fire', False)) and any_fired:
                print(f"  [T={T:.0f}K] stopping after first fire at step {step} "
                      f"for diagnostic comparison")
                break

            if np.isfinite(target_crack_extension_m) and target_crack_extension_m > 0.0 and crack_extension_m >= target_crack_extension_m:
                print(f"  [T={T:.0f}K] reached target crack extension {crack_extension_m*1e6:.3f} um at step {step}")
                break

            tip_extent_x = a_tip
            if tip_extent_x >= cfg.geometry.Lx - 3e-5:
                print(f"  [T={T:.0f}K] ligament severed at step {step}")
                break

            if deflect and remesh_on and any_fired:
                active_tips = np.array([f['xy'] for f in _active()]) if _active() else np.array([tip_xy])
                moved = 0.0
                for pt in active_tips:
                    moved = max(moved, float(np.min(np.hypot(refine_centers[:,0] - pt[0],
                                                             refine_centers[:,1] - pt[1]))))
                if moved > R_trigger:
                    paths = [f['path'] for f in fronts]
                    if pz_spatial_state:
                        _remesh_out = _remesh_following_tip(
                            cfg.geometry, cfg.mesh, 42, active_tips, mesh,
                            rho_gp, ep_gp, u, paths, kill_r, half_h,
                            extra_elem_fields=[pz_store_gp, pz_mobile_gp, pz_escape_gp, pz_emit_gp])
                        mesh, bnd, d, rho_gp, ep_gp, u, _extra_pz = _remesh_out
                        pz_store_gp, pz_mobile_gp, pz_escape_gp, pz_emit_gp = _extra_pz
                    else:
                        mesh, bnd, d, rho_gp, ep_gp, u = _remesh_following_tip(
                            cfg.geometry, cfg.mesh, 42, active_tips, mesh,
                            rho_gp, ep_gp, u, paths, kill_r, half_h)
                    x = mesh.nodes[:, 0]; y = mesh.nodes[:, 1]
                    cxe = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
                    cye = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
                    cx_e, cy_e = cxe, cye
                    elem_rad = np.sqrt(np.maximum(mesh.area_e, 1e-30))
                    h_tip = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
                    kill_r = max(h_tip, 0.5e-6)
                    if rho_transport_c > 0.0:
                        adj = build_elem_adjacency(mesh)
                    refine_center = tip_xy.copy()
                    refine_centers = active_tips.copy()

        tag = f"{int(T):04d}K"
        if crack_backend.name != 'sharp_wake':
            crack_backend.write_diagnostics(os.path.join(args.out, f'czm_{tag}'))
        np.savetxt(os.path.join(args.out, f'steps_{tag}.csv'), np.array(rows),
                   delimiter=',',
                   header='step,Uapp_m,Ftop_N,KJ_Pa_sqrtm,sigma_tip_Pa,sigma_back_Pa,'
                          'lambda_c,lambda_e,B,N_em,a_tip_m,crack_extension_m,da_block_m,W_emit_J_per_m,'
                          'n_fire,N_em_pre_renewal,N_em_shed_to_wake,'
                          'adaptive_frac,dt_cur_s,adaptive_dB1,adaptive_dB2,adaptive_dB_total,'
                          'fatigue_cycles,cycle_unlimited,cycle_limiter_code,dB_block,dN_store_block,dN_mobile_block,dN_escape_block,'
                          'dN_peierls_block,dN_taylor_block,storage_fraction,'
                          'mu_emit_per_cycle,mu_escape_per_cycle,mu_cleave_pred_per_cycle,'
                          'pz_store_total,pz_mobile_total,pz_escape_total,pz_emit_total,'
                          'cyclic_mechanics_updates,cyclic_plastic_work_J,'
                          'G_cleave_raw_eV,G_cleave_eff_eV,S_cleave_kB,'
                          'dGcleave_dsigma_eV_per_GPa,vstar_cleave_b3,'
                          'sigma_cleave_eff_Pa,cleave_barrier_kind_code,'
                          'front_state_model_code,mpz_K_shield_Pa_sqrt_m,mpz_mobile_count,'
                          'mpz_retained_count,mpz_available_site_fraction,mpz_local_slip_count,'
                          'mpz_escaped_total,mpz_recovered_total,mpz_wake_retained_total',
                   comments='')
        if branch_rows:
            np.savetxt(os.path.join(args.out, f'branch_diagnostics_{tag}.csv'),
                       np.array(branch_rows), delimiter=',',
                       header='step,Uapp_m,KJ1_Pa_sqrtm,KJ2_Pa_sqrtm,n_candidates,'
                              'angle1_deg,angle2_deg,metric1,metric2,metric2_over_metric1,'
                              'branch_active,branch_spawned,share_w1,share_w2,n_fire1,n_fire2,'
                              'advance1_m,advance2_m,front1_x_m,front1_y_m,front2_x_m,front2_y_m,'
                              'lambda_c1,lambda_c2,spawn_parent_id,spawn_child_id,'
                              'branch_B_before_spawn,branch_lambda_secondary_at_spawn,'
                              'branch_K_secondary_at_spawn,branch_metric_ratio_at_spawn',
                       comments='')
        if fronts_rows:
            np.savetxt(os.path.join(args.out, f'fronts_{tag}.csv'),
                       np.array(fronts_rows), delimiter=',',
                       header='step,front_id,parent_id,x_m,y_m,KJ_Pa_sqrtm,B,N_em,'
                              'lambda_c,n_fire,active,resolved,branch_len_m,'
                              'J_source_code,cluster_id,J_active_elems,J_signed_trial,J_effective_trial,J_sign_ref,parent_sep_m,nearest_crack_m,'
                              'cluster_hold_code,n_geom_adv,KJ_local_probe_Pa_sqrtm,KJ_local_over_parent,'
                              'branch_B,branch_dB_last,branch_lambda_secondary,branch_K_secondary,'
                              'branch_J_secondary_signed,branch_J_secondary_effective,'
                              'branch_metric_ratio,branch_clock_ready,n_starved_zones,'
                              'branch_B_before_spawn,branch_dB_at_spawn,'
                              'branch_lambda_secondary_at_spawn,branch_K_secondary_at_spawn,'
                              'branch_J_secondary_signed_at_spawn,branch_J_secondary_effective_at_spawn,'
                              'branch_metric_ratio_at_spawn,stagnant_lag_m,stagnant_K_ratio,'
                              'stagnant_lambda,stagnant_count,retire_step,stagnant_retired,'
                              'coalesced,merge_target_front_id,merge_step,merge_x_m,merge_y_m,'
                              'fatigue_cycles,dB_block,dN_store_block,dN_mobile_block,dN_escape_block,'
                              'dN_peierls_block,dN_taylor_block,storage_fraction,'
                              'mu_emit_per_cycle,mu_escape_per_cycle,'
                              'pz_store_count,pz_mobile_count,pz_escape_count', comments='')
        # Preserve the full front-local process-zone profiles at the same
        # accepted states used for the bounded field-snapshot history.  These
        # are diagnostics for microstructure evolution, not a replacement for
        # the mechanical-field snapshots.
        if not bool(getattr(args, 'no_mpz_state_output', False)):
            mpz_snap_payload = []
            for ss in snaps:
                if ss.get('mpz_front_states'):
                    mpz_snap_payload.append({
                        'step': int(ss.get('step', 0)),
                        'KJ_Pa_sqrt_m': float(ss.get('KJ', 0.0)),
                        'a_tip_m': float(ss.get('a_tip', 0.0)),
                        'fronts': ss.get('mpz_front_states', []),
                    })
            final_state_fronts = fronts if deflect else [{'id': 0, 'parent': -1,
                                                          'xy': np.array([a_tip, 0.0]),
                                                          'direction': np.array([1.0, 0.0]),
                                                          'active': True, 'eng': eng}]
            final_payload = []
            for ff_state in final_state_fronts:
                ff_eng = ff_state.get('eng')
                if getattr(ff_eng, 'state_model', 'legacy_scalar') == 'moving_pz' and hasattr(ff_eng, 'export_process_zone_state'):
                    final_payload.append({
                        'front_id': int(ff_state.get('id', 0)),
                        'parent_id': int(ff_state.get('parent', -1)),
                        'xy_m': np.asarray(ff_state.get('xy', [a_tip, 0.0]), float).tolist(),
                        'direction': np.asarray(ff_state.get('direction', [1.0, 0.0]), float).tolist(),
                        'active': bool(ff_state.get('active', True)),
                        'inactive_reason': str(ff_state.get('inactive_reason', 'active')),
                        'coalesced': bool(ff_state.get('coalesced', False)),
                        'state': ff_eng.export_process_zone_state(),
                    })
            if final_payload:
                with open(os.path.join(args.out, f'mpz_state_snapshots_{tag}.json'), 'w') as fp:
                    json.dump({'schema': 'moving_process_zone_history_v1',
                               'temperature_K': float(T),
                               'snapshots': mpz_snap_payload,
                               'final_fronts': final_payload}, fp, indent=2)

        if not bool(getattr(args, 'no_plots', False)):
            snap_path = _render_field_snapshots(args.out, T, mesh, snaps,
                                                max_cols=args.snapshot_cols)
            if snap_path:
                print(f"  saved field snapshots to {snap_path}")
            diag_paths = _render_diagnostics(args.out, T, hist)
            if diag_paths:
                print(f"  saved {len(diag_paths)} diagnostic plots to {args.out}")

        defl_deg = 0.0; path_dy_mm = 0.0; branched = False
        n_primary_adv = 0; n_branch_adv = 0; n_fronts_final = 1
        W_emit_total = eng.W_emit; N_em_final = eng.N_em; max_reach = a_tip
        branch_len_mm = 0.0
        if deflect:
            n_fronts_final = len(fronts)
            n_primary_adv = int(round(fronts[0].get('n_geom_adv', fronts[0]['eng'].n_adv)))
            n_branch_adv = int(round(sum(f.get('n_geom_adv', f['eng'].n_adv) for f in fronts if f['id'] != 0)))
            W_emit_total = float(sum(f['eng'].W_emit for f in fronts))
            N_em_final = float(fronts[0]['eng'].N_em)
            max_reach = max(f['xy'][0] for f in fronts)
            try:
                P = np.array(fronts[0]['path'])
                np.savetxt(os.path.join(args.out, f'crack_path_{int(T)}K.csv'), P,
                           header='x_m,y_m', delimiter=',', comments='')
                if len(P) >= 2:
                    dvec = P[-1] - P[0]
                    defl_deg = float(np.degrees(np.arctan2(dvec[1], abs(dvec[0]) + 1e-30)))
                    path_dy_mm = float((P[:, 1].max() - P[:, 1].min()) * 1e3)
                for f in fronts:
                    if len(f['path']) >= 2:
                        np.savetxt(os.path.join(args.out, f"crack_path_front{f['id']}_{int(T)}K.csv"),
                                   np.array(f['path']), header='x_m,y_m',
                                   delimiter=',', comments='')
                daughters = [f for f in fronts if f['id'] != 0 and len(f['path']) >= 2
                             and float(np.linalg.norm(np.array(f['path'][-1])
                                                      - np.array(f['path'][0]))) > 1e-12]
                if daughters:
                    dlead = max(daughters, key=lambda f: f['xy'][0])
                    branched = True
                    branch_len_mm = float(dlead['xy'][0] - dlead['path'][0][0]) * 1e3
                    np.savetxt(os.path.join(args.out, f'crack_path_branch_{int(T)}K.csv'),
                               np.array(dlead['path']), header='x_m,y_m',
                               delimiter=',', comments='')
            except Exception:
                pass

        if Kc_first_step is not None and Kc_first_step < len(hist['sigma_back']):
            sb_init = hist['sigma_back'][Kc_first_step]
            re_init = hist['r_eff_over_r0'][Kc_first_step]
            nem_init = hist['N_em'][Kc_first_step]
        else:
            sb_init = (hist['sigma_back'][-1] if hist['sigma_back'] else 0.0)
            re_init = (hist['r_eff_over_r0'][-1] if hist['r_eff_over_r0'] else 1.0)
            nem_init = (hist['N_em'][-1] if hist['N_em'] else 0.0)
        summary.append({
            'T': T,
            'Kc_first_MPa_sqrt_m': None if Kc_first is None else Kc_first / 1e6,
            'a_final_mm': max_reach * 1e3,
            'n_advances': n_primary_adv + n_branch_adv,
            'n_advances_primary': n_primary_adv, 'n_advances_branch': n_branch_adv,
            'n_fronts': n_fronts_final,
            'n_active_fronts_final': int(sum(1 for f in fronts if f.get('active', True))) if deflect else 1,
            'n_coalesced': int(sum(1 for f in fronts if bool(f.get('coalesced', False)))) if deflect else 0,
            'n_stagnant_retired': int(sum(1 for f in fronts if str(f.get('inactive_reason','active')) == 'stagnant_branch')) if deflect else 0,
            'branch_length_dx_mm': branch_len_mm,
            'N_em_final': N_em_final, 'W_emit_J_per_m': W_emit_total,
            'sigma_back_init_GPa': sb_init / 1e9,
            'r_eff_over_r0_init': re_init,
            'N_em_init': nem_init,
            'hbar_tip_m': float(mesh.hbar_tip),
            'n_nodes': int(mesh.nn),
            'deflection_deg': defl_deg,
            'path_span_dy_mm': path_dy_mm,
            'branched': bool(branched),
            'mode': ('no-fracture' if Kc_first is None
                     else ('ductile' if W_emit_total > 0.1 * Kc_first ** 2 / mat.Eprime
                           else 'brittle')),
            'shelf': audit,
        })
        kc_str = ('none' if Kc_first is None else f"{Kc_first/1e6:.3f}")
        print(f"  == T={T:.0f}K: Kc_first={kc_str} MPa*sqrt(m), "
              f"advances={n_primary_adv + n_branch_adv} (prim {n_primary_adv}, "
              f"branch {n_branch_adv}), n_fronts={n_fronts_final}, "
              f"max_reach={max_reach*1e3:.3f} mm, mode={summary[-1]['mode']}")
    with open(os.path.join(args.out, 'summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    tvt = _render_toughness_vs_T(args.out, summary)
    if tvt:
        print(f"  saved {tvt}")
    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _dbtt_from_results(results):
    """Extract derived scalars from a 1D temperature sweep.

    DBTT is the onset of the contiguous HIGH-T ductile block: the lowest T such
    that that point and ALL higher-T points are ductile.  This is robust to a
    single low-T point grazing the ductility threshold (which would otherwise
    make 'lowest ductile T anywhere' report a spurious low DBTT).  If no such
    contiguous block exists, DBTT is None.
    """
    order = sorted(range(len(results)), key=lambda i: results[i]['T'])
    Ts = [results[i]['T'] for i in order]
    modes = [results[i]['mode'] for i in order]
    Kc = {results[i]['T']: results[i]['Kc_MPa_sqrt_m'] for i in order}
    # walk down from the highest T; the DBTT is where the all-ductile tail starts
    dbtt = None
    for k in range(len(Ts) - 1, -1, -1):
        if modes[k] == 'ductile':
            dbtt = Ts[k]
        else:
            break
    # only count it as a transition if there is a brittle point below it
    brittle_below = any(modes[k] == 'brittle' for k in range(len(Ts))
                        if dbtt is not None and Ts[k] < dbtt)
    if dbtt is not None and not brittle_below:
        # all-ductile (no transition in window) -> not a DBTT
        if all(m == 'ductile' for m in modes):
            dbtt = None
    brittle_T = [t for t, m in zip(Ts, modes) if m == 'brittle']
    ductile_T = [t for t, m in zip(Ts, modes) if m == 'ductile']
    kc_br = [Kc[t] for t in brittle_T if Kc[t] is not None]
    kc_du = [Kc[t] for t in ductile_T if Kc[t] is not None]
    return {
        'DBTT_K': dbtt,
        'Kc_brittle_shelf': float(np.mean(kc_br)) if kc_br else None,
        'Kc_ductile_shelf': float(np.mean(kc_du)) if kc_du else None,
        'Kc_at_lowest_T': Kc.get(min(Ts)) if Ts else None,
        'Kc_at_highest_T': Kc.get(max(Ts)) if Ts else None,
        'n_brittle': len(brittle_T), 'n_ductile': len(ductile_T),
        'n_nofracture': sum(1 for m in modes if m == 'no-fracture'),
    }


def run_sweep(args):
    """Sweep over entropy parameterizations.  For each condition in the grid,
    run the full 1D temperature sweep into its own subdirectory and record the
    derived DBTT / shelf scalars in a combined table.

    Grid axes (each a list; the Cartesian product is swept):
      --sweep-form           entropy stress form(s): gated, affine
      --sweep-gate-power     Hill exponent n (gate sharpness)
      --sweep-emit-S0-kB     emission entropy magnitude (drives the ductile onset)
      --sweep-emit-S0-gate-GPa  emission gate stress (shifts the DBTT)
      --sweep-cleave-S0-kB   cleavage entropy magnitude
    Axes left empty fall back to the calibrated default for that parameter.
    """
    import itertools
    os.makedirs(args.out, exist_ok=True)

    forms = args.sweep_form or [None]
    powers = args.sweep_gate_power or [None]
    emitS0 = args.sweep_emit_S0_kB or [None]
    emitSG = args.sweep_emit_S0_gate_GPa or [None]
    cleaveS0 = args.sweep_cleave_S0_kB or [None]
    eTc0 = args.sweep_emit_S_T_c0_kB or [None]
    eTc1 = args.sweep_emit_S_T_c1 or [None]
    eSsig = args.sweep_emit_S_sigma_max_kB or [None]

    grid = list(itertools.product(forms, powers, emitS0, emitSG, cleaveS0,
                                  eTc0, eTc1, eSsig))
    print("=" * 72)
    print(f"  ENTROPY SWEEP — {len(grid)} conditions x "
          f"{len(args.temperatures)} temperatures")
    print(f"  axes: form={forms} gate_power={powers} emit_S0={emitS0} "
          f"emit_S0_gate={emitSG} cleave_S0={cleaveS0}")
    print(f"        emit_S_T_c0={eTc0} emit_S_T_c1={eTc1} emit_S_sigma_max={eSsig}")
    print("=" * 72)

    table = []
    for idx, (form, power, eS0, eSG, cS0, c0, c1, ssig) in enumerate(grid):
        # build a per-condition tag and output dir
        parts = []
        if form is not None: parts.append(f"form-{form}")
        if power is not None: parts.append(f"n-{power:g}")
        if eS0 is not None: parts.append(f"eS0-{eS0:g}")
        if eSG is not None: parts.append(f"eSG-{eSG:g}")
        if cS0 is not None: parts.append(f"cS0-{cS0:g}")
        if c0 is not None: parts.append(f"eTc0-{c0:g}")
        if c1 is not None: parts.append(f"eTc1-{c1:g}")
        if ssig is not None: parts.append(f"eSsig-{ssig:g}")
        tag = "__".join(parts) if parts else "default"
        cond_dir = os.path.join(args.out, f"cond_{idx:03d}_{tag}")

        # clone args for this 1D run with the entropy overrides applied
        sub = copy.copy(args)
        sub.mode = '1d'
        sub.out = cond_dir
        sub.entropy_form = form
        sub.entropy_gate_power = power
        sub.emit_S0_kB = eS0
        sub.emit_sigma0_S = eSG
        sub.cleave_S0_kB = cS0
        sub.emit_S_T_c0_kB = c0
        sub.emit_S_T_c1 = c1
        sub.emit_S_sigma_max_kB = ssig

        print(f"\n[cond {idx+1}/{len(grid)}] {tag}")
        results = run_1d(sub)
        derived = _dbtt_from_results(results)
        row = {'cond': idx, 'tag': tag, 'dir': cond_dir,
               'entropy_form': form if form is not None else 'physical(default)',
               'gate_power': power, 'emit_S0_kB': eS0,
               'emit_S0_gate_GPa': eSG, 'cleave_S0_kB': cS0,
               'emit_S_T_c0_kB': c0, 'emit_S_T_c1': c1,
               'emit_S_sigma_max_kB': ssig, **derived}
        table.append(row)
        print(f"  -> DBTT={derived['DBTT_K']}  Kc_brittle={derived['Kc_brittle_shelf']}"
              f"  Kc_ductile={derived['Kc_ductile_shelf']}")

    # write combined table (CSV + JSON)
    with open(os.path.join(args.out, 'sweep_summary.json'), 'w') as fp:
        json.dump(table, fp, indent=2)
    cols = ['cond', 'tag', 'entropy_form', 'gate_power', 'emit_S0_kB',
            'emit_S0_gate_GPa', 'cleave_S0_kB', 'emit_S_T_c0_kB', 'emit_S_T_c1',
            'emit_S_sigma_max_kB', 'DBTT_K', 'Kc_brittle_shelf',
            'Kc_ductile_shelf', 'Kc_at_lowest_T', 'Kc_at_highest_T',
            'n_brittle', 'n_ductile', 'n_nofracture', 'dir']
    with open(os.path.join(args.out, 'sweep_summary.csv'), 'w') as fp:
        fp.write(','.join(cols) + '\n')
        for r in table:
            fp.write(','.join('' if r.get(c) is None else str(r.get(c))
                              for c in cols) + '\n')
    print(f"\n  Sweep complete: {len(grid)} conditions.")
    print(f"  Combined table: {os.path.join(args.out, 'sweep_summary.csv')}")
    print(f"  Per-condition results in: {args.out}/cond_*/")
    return table


def _build_parser():
    p = argparse.ArgumentParser(description='Sharp-front dual-hazard fracture')
    p.add_argument('--mode', choices=['1d', '2d', 'sweep', 'mesh-sweep'],
                   default='1d')
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[300, 400, 500, 600, 700, 800, 900])
    p.add_argument('--out', default='runs/sharp_front')

    # engine
    p.add_argument('--front-state-model', choices=['legacy_scalar', 'moving_pz'],
                   default='legacy_scalar', dest='front_state_model',
                   help='legacy_scalar reproduces the frozen v8 first-passage closure. '
                        'moving_pz uses finite source sites, a moving 1-D defect grid, '
                        'direct elastic K shielding, and shared monotonic/fatigue/dwell kinetics.')
    p.add_argument('--mpz-allow-sigma-cap', action='store_true',
                   help='retain --sigma-cap-GPa in moving_pz mode for a diagnostic regression. '
                        'By default the artificial tip-stress ceiling is disabled.')
    p.add_argument('--mpz-length-m', type=float, default=2e-6)
    p.add_argument('--mpz-n-bins', type=int, default=40)
    p.add_argument('--mpz-n-systems', type=int, default=2)
    p.add_argument('--mpz-source-sites-per-system', type=float, default=200.0)
    p.add_argument('--mpz-source-recovery-rate-s', type=float, default=0.0)
    p.add_argument('--mpz-source-refresh-length-m', type=float, default=2.5e-7)
    p.add_argument('--mpz-source-bin-count', type=int, default=2)
    p.add_argument('--mpz-shielding-factors', default='1 1',
                   help='space/comma-separated elastic orientation factors for the slip systems.')
    p.add_argument('--mpz-mobile-shield-fraction', type=float, default=0.0)
    p.add_argument('--mpz-shielding-core-m', type=float, default=2.5e-10)
    p.add_argument('--mpz-glide-nu0-s', type=float, default=1e11)
    p.add_argument('--mpz-glide-barrier-eV', type=float, default=0.80)
    p.add_argument('--mpz-glide-activation-volume-b3', type=float, default=8.0)
    p.add_argument('--mpz-glide-step-m', type=float, default=2.5e-10)
    p.add_argument('--mpz-glide-stress-fraction', type=float, default=0.45)
    p.add_argument('--mpz-trap-nu0-s', type=float, default=1e9)
    p.add_argument('--mpz-trap-barrier-eV', type=float, default=0.65)
    p.add_argument('--mpz-trap-activation-volume-b3', type=float, default=1.0)
    p.add_argument('--mpz-detrap-nu0-s', type=float, default=1e10)
    p.add_argument('--mpz-detrap-barrier-eV', type=float, default=1.20)
    p.add_argument('--mpz-detrap-activation-volume-b3', type=float, default=1.0)
    p.add_argument('--mpz-retained-recovery-nu0-s', type=float, default=1e9)
    p.add_argument('--mpz-retained-recovery-barrier-eV', type=float, default=1.50)
    p.add_argument('--mpz-retained-recovery-activation-volume-b3', type=float, default=0.0)
    p.add_argument('--mpz-mobile-recovery-rate-s', type=float, default=0.0)
    p.add_argument('--mpz-pair-annihilation-rate-per-count-s', type=float, default=0.0)
    p.add_argument('--mpz-blunting-length-m', type=float, default=5e-7)
    p.add_argument('--mpz-blunting-slip-fraction', type=float, default=1.0)
    p.add_argument('--mpz-max-transport-cfl', type=float, default=0.35)
    p.add_argument('--mpz-max-transport-substeps', type=int, default=2000)
    p.add_argument('--r-pz', type=float, default=1e-6, dest='r_pz')
    p.add_argument('--rho0', type=float, default=None, dest='rho0',
                   help='Initial background dislocation density [m^-2] (default 5e12). '
                        'Lower it (e.g. 1e9) in sources-only mode so the far field has few '
                        'carriers and stays elastic until tip-nucleated content arrives.')
    p.add_argument('--sigma-cap-GPa', type=float, default=30.0, dest='sigma_cap_GPa')
    p.add_argument('--multihit-m', type=float, default=3.0, dest='multihit_m')
    p.add_argument('--multihit-tau', type=float, default=1e-6, dest='multihit_tau')
    p.add_argument('--nu0-cleave', type=float, default=1e12, dest='nu0_cleave')
    p.add_argument('--nu0-emit', type=float, default=1e11, dest='nu0_emit')
    p.add_argument('--dN-cap', type=float, default=float('inf'), dest='dN_cap',
                   help='Optional cap on emitted-dislocation ledger increment per accepted step. Default inf disables this numerical limiter; use finite values only as a diagnostic stabilizer.')
    p.add_argument('--beta-back', type=float, default=1.0, dest='beta_back')
    p.add_argument('--c-blunt', type=float, default=1.0, dest='c_blunt')
    p.add_argument('--L-pz', type=float, default=1e-6, dest='L_pz')
    p.add_argument('--mesh-levels', type=float, nargs='+', dest='mesh_levels',
                   default=[8e-6, 4e-6, 2e-6, 1e-6, 5e-7, 2.5e-7],
                   help='(mesh-sweep) tip_h_fine values [m], coarse->fine. The '
                        'study refines the tip patch across L_pz and reports how '
                        'Kc / sigma_back / r_eff converge.')
    p.add_argument('--v-emb-b3', type=float, default=500.0, dest='v_emb_b3')
    p.add_argument('--wake-retain', type=float, default=0.3, dest='wake_retain')
    p.add_argument('--cleave-shield-chi', type=float, default=0.0, dest='chi_shield',
                   help='back-stress shielding of the cleavage hazard: '
                        'sigma_eff = sigma_tip - chi*sigma_back. 0=embrittlement-only '
                        '(as shipped); larger values toughen with emission (DBTT axis).')
    p.add_argument('--emb-sat-frac', type=float, default=1.0, dest='emb_sat_frac',
                   help='embrittlement saturation: dG_emb capped at this fraction of the '
                        'cleavage barrier. 1.0=uncapped (as shipped); <1 lets shielding '
                        'hold a tough upper shelf.')
    p.add_argument('--n-sat', type=float, default=float('inf'), dest='N_sat',
                   help='saturation density of the emitted-dislocation ledger [count]. '
                        'inf=off (as shipped, unbounded growth). Finite value gives a '
                        'PHYSICAL ceiling so shielding AND embrittlement saturate together.')
    p.add_argument('--recover-k', type=float, default=0.0, dest='recover_k',
                   help='linear dynamic-recovery (annihilation) rate of the ledger [1/s]. '
                        '0=off. Use instead of/with --n-sat to bound N_em physically.')
    p.add_argument('--v-rayleigh', type=float, default=float('inf'), dest='v_rayleigh',
                   help='deprecated compatibility option. v3 does not cap crack velocity; '
                        'use --adaptive-events to refine the numerical event sequence instead.')
    p.add_argument('--adaptive-events', action='store_true',
                   help='numerical adaptive load/time stepping for sharp-front 2D: '
                        'reduce the proposed load increment when the predicted renewal '
                        'clock increment is too large, then recompute FEM/J. This is '
                        'a timestep refinement, not a velocity cap.')
    p.add_argument('--adaptive-event-target', type=float, default=0.35,
                   help='target maximum predicted cleavage-clock increment per accepted '
                        '2D step when --adaptive-events is enabled. Smaller values '
                        'resolve event sequences more finely.')
    p.add_argument('--adaptive-min-frac', type=float, default=1e-8,
                   help='minimum fraction of the nominal dU/dt increment allowed by '
                        'adaptive stepping. If this is reached the step is accepted '
                        'and the crack may run unstably; no propagation cap is imposed.')
    p.add_argument('--adaptive-safety', type=float, default=0.7,
                   help='safety factor used when shrinking a rejected adaptive step.')
    p.add_argument('--adaptive-grow', type=float, default=4.0, dest='adaptive_grow',
                   help='max per-step growth of the adaptive load/time fraction '
                        '(warm-start). Each new step starts at min(1, grow*last_frac) '
                        'instead of restarting at 1.0, so the controller settles at a '
                        'frac where dB~target and the load keeps advancing instead of '
                        'deadlocking. This is numerical warm-starting, not a velocity cap.')
    p.add_argument('--bulk-kinetics-model',
                   choices=['emission_derived_peierls_taylor_multihit',
                            'legacy_additive_flow_stress'],
                   default='emission_derived_peierls_taylor_multihit',
                   dest='bulk_kinetics_model',
                   help='Production uses emission-derived EXP-floor Peierls/Taylor rates; legacy retains the older additive yield-stress ablation.')
    p.add_argument('--pt-taylor-corr-rho-c', type=float, default=1.0e14,
                   dest='pt_taylor_corr_rho_c')
    p.add_argument('--pt-taylor-renewal-time-s', type=float, default=1.0e-9,
                   dest='pt_taylor_renewal_time_s')
    p.add_argument('--pt-taylor-m-exponent', type=float, default=1.0,
                   dest='pt_taylor_m_exponent')
    p.add_argument('--pt-taylor-m-scale', type=float, default=1.0,
                   dest='pt_taylor_m_scale')
    p.add_argument('--pt-taylor-m-cap', type=float, default=float('inf'),
                   dest='pt_taylor_m_cap',
                   help='Finite obstacle count in one correlation domain; not a total-density cap. Every selected row is audited for high-density downturn.')
    p.add_argument('--pt-mobile-fraction', type=float, default=0.01,
                   dest='pt_mobile_fraction')
    p.add_argument('--pt-mobile-saturation-density-m2', type=float, default=1.0e14,
                   dest='pt_mobile_saturation_density_m2')
    p.add_argument('--pt-mobile-density-floor-m2', type=float, default=1.0e6,
                   dest='pt_mobile_density_floor_m2')
    p.add_argument('--pt-jump-fraction', type=float, default=1.0,
                   dest='pt_jump_fraction')
    p.add_argument('--pt-jump-length-min-m', type=float, default=2.5e-10,
                   dest='pt_jump_length_min_m')
    p.add_argument('--pt-taylor-phi-max', type=float, default=20.0,
                   dest='pt_taylor_phi_max')
    p.add_argument('--mpz-no-emission-derived-pt', action='store_false',
                   dest='mpz_use_emission_derived_pt', default=True,
                   help='Legacy ablation: use independent fixed glide/detrap barriers in moving-PZ transport.')

    p.add_argument('--bulk-mult-frac', type=float, default=1.0, dest='bulk_mult_frac',
                   help='Frank-Read bulk multiplication fraction (1=full, 0=sources-only). '
                        'Set 0 with --tip-source-rho-per-emit for the nucleation-source picture.')
    p.add_argument('--tip-source-rho-per-emit', type=float, default=0.0,
                   dest='tip_source_rho_per_emit',
                   help='Density [m^-2] deposited per tip-emitted dislocation, applied EVERY '
                        'step from the emission rate (continuous nucleation source). 0=off.')
    p.add_argument('--rho-transport-c', type=float, default=0.0, dest='rho_transport_c',
                   help='Mobility-scaled conservative density diffusivity coeff: '
                        'D=c*dot_ep*L_pz^2. 0=no transport.')
    p.add_argument('--exhaustion', action='store_true', dest='exhaustion',
                   help='Finite-content plasticity: mobile density is consumed as it '
                        'sweeps to sinks (d rho=-dgamma/(b*L_sink)); a fixed initial rho '
                        'mediates only gamma_max=rho*b*L_sink before the stress must rise.')
    p.add_argument('--glide-to-sink-m', type=float, default=1e-5, dest='glide_to_sink_m',
                   help='Mean glide path to a sink L_sink [m] (sets the strain budget).')
    p.add_argument('--mobile-rho-floor', type=float, default=1e6, dest='mobile_rho_floor',
                   help='rho evaluation/clip floor [m^-2]; lower (e.g. 1e8) lets content '
                        'actually exhaust below the legacy 1e6.')
    p.add_argument('--peierls-floor-MPa', type=float, default=0.0, dest='peierls_floor_MPa',
                   help='rho-independent athermal resistance floor on sigma_P [MPa]; keeps a '
                        'finite yield resistance when the kink-pair stress collapses at high T, '
                        'so elasticity carries stress above it while content exhausts.')
    p.add_argument('--da', type=float, default=2e-5)
    p.add_argument('--cleave-H0-eV', type=float, default=None, dest='cleave_H0_eV')
    p.add_argument('--cleave-S0-kB', type=float, default=None, dest='cleave_S0_kB')
    p.add_argument('--emit-H0-eV', type=float, default=None, dest='emit_H0_eV')
    p.add_argument('--emit-barrier-kind', choices=['classic', 'exp_floor'], default=None,
                   dest='emit_barrier_kind',
                   help='Local crack-tip emission barrier. exp_floor uses the direct bounded EXP-floor free-energy surface.')
    p.add_argument('--emit-G00-eV', type=float, default=None, dest='emit_G00_eV')
    p.add_argument('--emit-gT-eV-per-K', type=float, default=None, dest='emit_gT_eV_per_K')
    p.add_argument('--emit-sigc0-GPa', type=float, default=None, dest='emit_sigc0_GPa')
    p.add_argument('--emit-sT-GPa-per-K', type=float, default=None, dest='emit_sT_GPa_per_K')
    p.add_argument('--emit-exp-a', type=float, default=None, dest='emit_exp_a')
    p.add_argument('--emit-exp-n', type=float, default=None, dest='emit_exp_n')
    p.add_argument('--emit-floor-frac', type=float, default=None, dest='emit_floor_frac')
    p.add_argument('--emit-floor-min-eV', type=float, default=None, dest='emit_floor_min_eV')
    p.add_argument('--emit-floor-max-frac', type=float, default=None, dest='emit_floor_max_frac')
    p.add_argument('--emit-Tref-K', type=float, default=None, dest='emit_Tref_K')
    p.add_argument('--entropy-form', choices=['affine', 'gated', 'physical', 'meyer_neldel'], default=None,
                   dest='entropy_form',
                   help='Override entropy stress form on BOTH barriers '
                        '(affine=legacy -S0(1+s/s0); gated=Hill -S0 x/(1+x), '
                        'zero at s=0). Default: use each barrier preset.')
    p.add_argument('--cleave-S0-gate-GPa', type=float, default=None,
                   dest='cleave_sigma0_S', help='Override cleavage sigma0_S [GPa]')
    p.add_argument('--emit-S0-gate-GPa', type=float, default=None,
                   dest='emit_sigma0_S', help='Override emission sigma0_S [GPa]')
    p.add_argument('--emit-S0-kB', type=float, default=None, dest='emit_S0_kB',
                   help='Override emission entropy magnitude S0 [kB]')
    p.add_argument('--gate-power', type=float, default=None, dest='entropy_gate_power',
                   help='Override entropy Hill gate exponent n on BOTH barriers')
    # physical composite-entropy single-run overrides (emission channel)
    p.add_argument('--emit-S-T-c0-kB', type=float, default=None, dest='emit_S_T_c0_kB',
                   help='Emission baseline entropy S_T constant [kB] (physical form)')
    p.add_argument('--emit-S-T-c1', type=float, default=None, dest='emit_S_T_c1',
                   help='Emission baseline entropy S_T linear-in-T slope [kB/K]')
    p.add_argument('--emit-S-sigma-max-kB', type=float, default=None,
                   dest='emit_S_sigma_max_kB',
                   help='Emission Schoeck stress-term magnitude [kB] (physical form)')
    p.add_argument('--cleave-S-sigma-max-kB', type=float, default=None,
                   dest='cleave_S_sigma_max_kB',
                   help='Cleavage Schoeck stress-term magnitude [kB] (physical form)')
    p.add_argument('--cleave-barrier-kind', choices=['classic', 'exp_floor'], default=None,
                   dest='cleave_barrier_kind',
                   help='Cleavage free-energy surface. classic uses H-TS-sigma*v; exp_floor uses DeltaG*(sigma,T) directly.')
    p.add_argument('--cleave-G00-eV', type=float, default=None, dest='cleave_G00_eV')
    p.add_argument('--cleave-gT-eV-per-K', type=float, default=None, dest='cleave_gT_eV_per_K')
    p.add_argument('--cleave-sigc0-GPa', type=float, default=None, dest='cleave_sigc0_GPa')
    p.add_argument('--cleave-sT-GPa-per-K', type=float, default=None, dest='cleave_sT_GPa_per_K')
    p.add_argument('--cleave-exp-a', type=float, default=None, dest='cleave_exp_a')
    p.add_argument('--cleave-exp-n', type=float, default=None, dest='cleave_exp_n')
    p.add_argument('--cleave-floor-frac', type=float, default=None, dest='cleave_floor_frac')
    p.add_argument('--cleave-floor-min-eV', type=float, default=None, dest='cleave_floor_min_eV')
    p.add_argument('--cleave-floor-max-frac', type=float, default=None, dest='cleave_floor_max_frac')
    p.add_argument('--cleave-Tref-K', type=float, default=None, dest='cleave_Tref_K')
    p.add_argument('--cleave-exp-T-mode', choices=['linear', 'mu_scale'], default=None, dest='cleave_exp_T_mode',
                   help='Temperature law for exp_floor cleavage: linear or shear-modulus-like mu_scale.')
    p.add_argument('--cleave-mu-dlnmu-dT-per-K', type=float, default=None, dest='cleave_mu_dlnmu_dT_per_K')
    p.add_argument('--cleave-G0-mu-power', type=float, default=None, dest='cleave_G0_mu_power')
    p.add_argument('--cleave-sigc-mu-power', type=float, default=None, dest='cleave_sigc_mu_power')
    p.add_argument('--cleave-S-hs-kB', type=float, default=None, dest='cleave_S_hs_kB',
                   help='Optional high-stress entropy shift for exp_floor [kB].')
    p.add_argument('--cleave-sigma-S-GPa', type=float, default=None, dest='cleave_sigma_S_GPa')
    p.add_argument('--cleave-S-hs-power', type=float, default=None, dest='cleave_S_hs_power')
    p.add_argument('--cleave-S-hs-dT-per-K', type=float, default=None, dest='cleave_S_hs_dT_per_K')
    p.add_argument('--cleave-S-hs-Tref-K', type=float, default=None, dest='cleave_S_hs_Tref_K')
    p.add_argument('--cleave-monotone-stress', action=argparse.BooleanOptionalAction, default=None, dest='cleave_monotone_stress',
                   help='Apply monotone-stress envelope to classic cleavage barrier diagnostics/rates. Default uses barrier preset.')

    # sweep mode: each axis is a list; the Cartesian product is swept, and each
    # condition runs the full 1D temperature sweep into its own subdirectory.
    p.add_argument('--sweep-form', nargs='+', default=None,
                   choices=['gated', 'affine', 'physical', 'meyer_neldel'], dest='sweep_form',
                   help='Entropy stress form(s) to sweep')
    p.add_argument('--sweep-gate-power', nargs='+', type=float, default=None,
                   dest='sweep_gate_power', help='Hill gate exponent(s) n to sweep')
    p.add_argument('--sweep-emit-S0-kB', nargs='+', type=float, default=None,
                   dest='sweep_emit_S0_kB',
                   help='Emission entropy magnitude(s) S0 [kB] to sweep (drives ductile onset)')
    p.add_argument('--sweep-emit-S0-gate-GPa', nargs='+', type=float, default=None,
                   dest='sweep_emit_S0_gate_GPa',
                   help='Emission gate stress(es) sigma0_S [GPa] to sweep (shifts DBTT)')
    p.add_argument('--sweep-cleave-S0-kB', nargs='+', type=float, default=None,
                   dest='sweep_cleave_S0_kB',
                   help='Cleavage entropy magnitude(s) S0 [kB] to sweep')
    # physical composite-entropy sweep axes
    p.add_argument('--sweep-emit-S-T-c0-kB', nargs='+', type=float, default=None,
                   dest='sweep_emit_S_T_c0_kB',
                   help='Emission baseline S_T constant(s) [kB] (physical form)')
    p.add_argument('--sweep-emit-S-T-c1', nargs='+', type=float, default=None,
                   dest='sweep_emit_S_T_c1',
                   help='Emission baseline S_T linear slope(s) [kB/K] (physical form)')
    p.add_argument('--sweep-emit-S-sigma-max-kB', nargs='+', type=float, default=None,
                   dest='sweep_emit_S_sigma_max_kB',
                   help='Emission Schoeck stress-term magnitude(s) [kB] (physical form)')

    # v10 unified material/state selection
    p.add_argument('--material-class', choices=['ceramic', 'weakT', 'DBTT'], default=None,
                   help='Use the promoted unified MPZ material class.')
    p.add_argument('--material-manifest', default=None)
    p.add_argument('--mpz-length-um', type=float, default=100.0)
    p.add_argument('--mpz-source-bins', type=int, default=2)
    p.add_argument('--mpz-blunting-length-um', type=float, default=0.5)
    p.add_argument('--wake-length-um', type=float, default=100.0)
    p.add_argument('--wake-n-bins', type=int, default=0)
    p.add_argument('--wake-shielding', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wake-shield-projection', type=float, default=1.0)

    # 1d
    p.add_argument('--Kdot', type=float, default=0.02, help='MPa*sqrt(m)/s')
    p.add_argument('--Kmax', type=float, default=40.0, help='MPa*sqrt(m)')
    p.add_argument('--dt', type=float, default=1.0)
    p.add_argument('--n-advances', type=int, default=1, dest='n_advances')
    p.add_argument('--continue-after-init', action='store_true', dest='continue_after_init')
    p.add_argument('--ductile-N', type=float, default=50.0, dest='ductile_N')
    p.add_argument('--ductile-shield', type=float, default=0.3, dest='ductile_shield',
                   help='Ductile if back-stress shielding fraction sigma_back/sigma_tip '
                        'exceeds this (continuous criterion; default 0.3)')
    p.add_argument('--ductile-blunt', type=float, default=1.02, dest='ductile_blunt',
                   help='Ductile if tip blunting r_eff/r0 exceeds this '
                        '(continuous criterion; default 1.02)')

    # 2d
    p.add_argument('--nx', type=int, default=60)
    p.add_argument('--ny', type=int, default=120)
    p.add_argument('--plane-gate-global', action='store_true',
                   help='gate cleavage-plane choice against the macroscopic crack axis '
                        '(+x) instead of the local heading, enabling {100} variant '
                        'switching / zig-zag and asymmetric branch competition')
    p.add_argument('--min-global-forward', type=float, default=0.05,
                   dest='min_global_forward',
                   help='minimum dot(candidate_direction,+x) allowed for sharp-front propagation. '
                        'Rejects backward crack growth and near-vertical tensile-direction side '
                        'branches in the through-ligament geometry. Set <= -1 to disable.')
    p.add_argument('--allow-abs-directional-J', action='store_true',
                   dest='allow_abs_directional_J',
                   help='legacy/debug mode: use abs(J_signed) for directional crack driving. '
                        'Default uses only the positive signed-J part relative to the root-crack sign.')
    p.add_argument('--no-tip-remesh', action='store_true',
                   help='disable tip-following remeshing (debug; default ON for --crystal-aniso)')
    p.add_argument('--crystal-branch', action='store_true',
                   help='allow crack branching at co-critical {100} planes (energy-shared)')
    p.add_argument('--rJ-outer', type=float, default=None, dest='rJ_outer',
                   help='actual outer radius of the local J-integral contour [m]. '
                        'If omitted, use a mesh/process-zone/da based local value. '
                        'This is preferred over legacy --rJ for branched fronts.')
    p.add_argument('--rJ-cluster', type=float, default=None, dest='rJ_cluster',
                   help='legacy-domain length ell [m] for the outer group/cluster J '
                        'integral used by the primary crack and unresolved branch clusters. '
                        'Actual outer radius is about 8*rJ_cluster. Default preserves '
                        'the pre-local-J multifront driving scale.')
    p.add_argument('--j-decomposition', choices=['cluster', 'local'], default='cluster',
                   help='cluster: use an outer group J for parent/unresolved clusters '
                        'and local J only for resolved daughter tips. local: force all '
                        'resolved tips to use the small local J contour.')
    p.add_argument('--branch-resolve-length', type=float, default=0.0,
                   dest='branch_resolve_length',
                   help='length/separation a newborn branch must reach before it '
                        'gets an independent local J-integral [m]. 0 => use the '
                        'local J outer radius / several da_phys.')
    p.add_argument('--local-J-clearance-factor', type=float, default=1.0,
                   dest='local_J_clearance_factor',
                   help='minimum clearance between an independent local J contour and '
                        'other crack/wake segments, as a multiple of rJ_outer. '
                        'This is a contour-validity criterion, not a velocity cap.')
    p.add_argument('--min-J-active-elems', type=int, default=12,
                   dest='min_J_active_elems',
                   help='minimum active elements required in a probed local J domain '
                        'before an unresolved daughter is promoted to independent J.')
    p.add_argument('--local-J-handoff-min-K-ratio', type=float, default=0.25,
                   dest='local_J_handoff_min_K_ratio',
                   help='minimum local-J K / parent-cluster K ratio required before an '
                        'unresolved branch is promoted to independent local J. Set 0 '
                        'to disable. This is a handoff-validity criterion, not a '
                        'propagation cap.')
    p.add_argument('--max-fronts', type=int, default=32, dest='max_fronts',
                   help='maximum number of simultaneously tracked crack fronts in the '
                        'multi-front inventory. Numerical/memory guard on branch cascades, '
                        'not a physical branching cap; raise it if births are being clipped.')
    p.add_argument('--branch-spacing', type=float, default=10.0, dest='branch_spacing',
                   help='minimum propagation distance (in multiples of da_phys) a front '
                        'must travel since birth/its last branch before it may branch '
                        'again. Physical branch-spacing length; prevents a stationary '
                        'co-critical tip from re-branching every step. Set 0 to disable.')
    p.add_argument('--branch-fp-min-ratio', type=float, default=0.95,
                   dest='branch_fp_min_ratio',
                   help='minimum secondary/primary overdrive ratio required to accumulate '
                        'the branch-specific first-passage clock. Restrictive by default; '
                        'ratio-only branching is not sufficient for birth.')
    p.add_argument('--branch-clock-target', type=float, default=1.0,
                   dest='branch_clock_target',
                   help='first-passage threshold for the secondary branch clock. A branch '
                        'may be born only after this clock reaches the target and the parent fires.')
    p.add_argument('--branch-clock-angle-tol-deg', type=float, default=15.0,
                   dest='branch_clock_angle_tol_deg',
                   help='reset the branch clock if the secondary branch direction changes by more than this angle.')
    p.add_argument('--branch-secondary-min-K-ratio', type=float, default=0.85,
                   dest='branch_secondary_min_K_ratio',
                   help='absolute viability check: estimated secondary K must exceed this fraction of parent K.')
    p.add_argument('--branch-secondary-min-K-MPa', type=float, default=0.0,
                   dest='branch_secondary_min_K_MPa',
                   help='absolute viability check: estimated secondary K must exceed this MPa*sqrt(m) value. 0 disables.')
    p.add_argument('--branch-secondary-min-lambda', type=float, default=0.0,
                   dest='branch_secondary_min_lambda',
                   help='absolute viability check: secondary cleavage hazard must exceed this rate [1/s]. 0 disables.')
    p.add_argument('--branch-starved-suppression-radius', type=float, default=80e-6,
                   dest='branch_starved_suppression_radius',
                   help='radius around a mechanically starved daughter branch where new branch births are vetoed [m].')
    p.add_argument('--branch-starved-K-ratio', type=float, default=0.25,
                   dest='branch_starved_K_ratio',
                   help='record a resolved short daughter as starved when its K/parent-K falls below this value.')
    p.add_argument('--branch-starved-lambda', type=float, default=1e-20,
                   dest='branch_starved_lambda',
                   help='record a short daughter as starved when its cleavage hazard falls below this rate [1/s].')
    p.add_argument('--branch-starved-max-length-factor', type=float, default=4.0,
                   dest='branch_starved_max_length_factor',
                   help='only branches shorter than factor*branch_resolve_length are eligible for starved-zone recording.')
    p.add_argument('--coalesce-cracks', action=argparse.BooleanOptionalAction, default=True,
                   help='Retire an advancing front when its committed increment first intersects an existing crack path (default: enabled).')
    p.add_argument('--retire-stagnant-branches', action='store_true',
                   dest='retire_stagnant_branches',
                   help='deactivate resolved side-branch tips that are far behind the leading crack and persistently have weak K/lambda. The crack segment remains in the damage field; only the dormant tip is no longer evaluated.')
    p.add_argument('--branch-stagnant-lag', type=float, default=0.0,
                   dest='branch_stagnant_lag',
                   help='minimum x-lag behind the leading active tip before a resolved branch may be retired as stagnant [m]. 0 => max(4*branch_resolve_length,50*da).')
    p.add_argument('--branch-stagnant-K-ratio', type=float, default=0.20,
                   dest='branch_stagnant_K_ratio',
                   help='retire-side-branch criterion: K_branch/K_leading must remain below this value.')
    p.add_argument('--branch-stagnant-lambda', type=float, default=1e-30,
                   dest='branch_stagnant_lambda',
                   help='retire-side-branch criterion: branch cleavage hazard must remain below this rate [1/s].')
    p.add_argument('--branch-stagnant-steps', type=int, default=20,
                   dest='branch_stagnant_steps',
                   help='number of consecutive accepted steps satisfying stagnation criteria before retiring a branch tip.')
    p.add_argument('--branch-stagnant-no-fire-steps', type=int, default=80,
                   dest='branch_stagnant_no_fire_steps',
                   help='branch must have not fired for this many accepted steps before it can be retired as stagnant.')
    p.add_argument('--branch-stagnant-min-length', type=float, default=0.0,
                   dest='branch_stagnant_min_length',
                   help='minimum arclength for a side branch before stagnation retirement is allowed [m]. 0 => max(branch_resolve_length,5*da).')
    p.add_argument('--branch-ratio', type=float, default=None,
                   help='second plane branches if its opening stress >= ratio*winner. '
                        'Default is material-preset dependent: W=0.92, branchy=0.85')
    p.add_argument('--branch-share-mode', choices=['hazard', 'equal'], default='hazard',
                   help='ledger split at branch birth: hazard = split by relative overdrive; '
                        'equal = 50/50')
    p.add_argument('--branch-hazard-sharpness', type=float, default=2.0,
                   help='exponent mapping overdrive to branch-birth share. Larger makes '
                        'the stronger lobe dominate; 1-2 gives smooth competing statistics.')
    p.add_argument('--branch-energy-share', choices=['hazard-budget', 'none'],
                   default='hazard-budget',
                   help='when two active fronts fire in the same load step, hazard-budget '
                        'splits one continuous advance budget by local lambda_c. none lets '
                        'each front advance independently.')
    p.add_argument('--crystal-compete', action='store_true',
                   help='competing direction selection: crack advances along argmax of '
                        'overdrive sigma_nn(phi)/sqrt(gamma(phi)) over a continuum of '
                        'directions, with cubic cleavage-energy anisotropy gamma(phi). '
                        'Finite, tunable anisotropy instead of an infinitely sharp argmax.')
    p.add_argument('--crystal-material', choices=['w', 'branchy', 'bcc-branchy', 'model-branchy'],
                   default='w',
                   help='crystallographic material preset for 2D branching. w is the '
                        'near-isotropic tungsten baseline. branchy is an explicit second '
                        'model material class with stronger anisotropy and lower branch '
                        'co-criticality thresholds for branch-statistics sweeps.')
    p.add_argument('--crystal-include-110', action='store_true',
                   help='include secondary {110} cleavage traces in the candidate set. '
                        'Automatically enabled by --crystal-material branchy.')
    p.add_argument('--gamma-110-rel', type=float, default=None, dest='gamma_110_rel',
                   help='relative cleavage energy of secondary {110} traces; default W=1.3, branchy=1.15')
    p.add_argument('--cleave-gamma-aniso', type=float, default=None,
                   help='[--crystal-compete] cubic cleavage-energy anisotropy amplitude '
                        'delta = gamma_{110}/gamma_{100} - 1. 0=isotropic (follows mode-I), '
                        'large=locks to {100}. Default W=0.3; branchy=2.0.')
    p.add_argument('--branch-overdrive-ratio', type=float, default=None,
                   help='[--crystal-compete] a secondary overdrive maximum branches if it '
                        'is >= ratio*max (and angularly separated). Lower=branches more easily. '
                        'Default W=0.9; branchy=0.80.')
    p.add_argument('--crystal-aniso', action='store_true',
                   help='use cubic BCC anisotropic elasticity (else isotropic)')
    p.add_argument('--crystal-theta-deg', type=float, default=0.0,
                   help='in-plane crystal orientation [deg]')
    p.add_argument('--crystal-C11', type=float, default=None, help='cubic C11 [Pa] (default W)')
    p.add_argument('--crystal-C12', type=float, default=None, help='cubic C12 [Pa] (default W)')
    p.add_argument('--crystal-C44', type=float, default=None, help='cubic C44 [Pa] (default W; raise for A>1)')
    p.add_argument('--tip-h-fine', type=float, default=0.0, dest='tip_h_fine',
                   help='Adaptive mesh: fine element size at the crack tip [m]. '
                        '0 -> uniform nx*ny grid. Resolves the process zone (h<<L_pz) '
                        'at ~log node cost instead of uniform refinement.')
    p.add_argument('--tip-ratio', type=float, default=1.15, dest='tip_ratio',
                   help='Geometric coarsening ratio per element away from the tip '
                        '(graded mesh). Smaller = smoother grading, more nodes.')
    p.add_argument('--steps', type=int, default=120,
                   help='Max load steps (2D).')
    p.add_argument('--dU', type=float, default=5e-8,
                   help='Displacement increment per step [m] (2D).')
    p.add_argument('--n-stagger', type=int, default=2, dest='n_stagger',
                   help='Mech<->plastic stagger iterations per step. The coupling '
                        'is well-converged at 2 here (Kc unchanged at 10).')
    p.add_argument('--print-every', type=int, default=10, dest='print_every')
    p.add_argument('--save-snapshots', type=int, default=4, dest='save_snapshots',
                   help='Number of field-snapshot columns to capture over the run '
                        '(2D only). 0 or 1 saves only the final state.')
    p.add_argument('--snapshot-cols', type=int, default=4, dest='snapshot_cols',
                   help='Max columns in the rendered snapshot panel (2D only).')
    p.add_argument('--no-mpz-state-output', action='store_true',
                   help='Disable JSON export of moving process-zone profiles at accepted snapshot states and the final front inventory.')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip 2-D field/diagnostic plot rendering; CSV and summary outputs are still written. Useful for quick comparison sweeps.')
    p.add_argument('--stop-after-first-fire', action='store_true',
                   help='For diagnostic sweeps, stop a 2-D sharp-front run immediately after the first accepted crack advance. Production multifront runs should normally leave this off.')
    p.add_argument('--da-phys', type=float, default=None, dest='da_phys',
                   help='PHYSICAL crack advance per cleavage event [m] (2D). '
                        'Mesh-independent; default ~5*r_pz. Elements are killed '
                        'as the physical tip crosses them.')
    p.add_argument('--crack-backend', choices=['sharp_wake', 'edge_split_czm', 'adaptive_czm'],
                   default='sharp_wake', dest='crack_backend',
                   help='Crack geometry backend. sharp_wake reproduces the legacy stiffness-kill representation. edge_split_czm follows existing mesh edges. adaptive_czm locally relocates a tip-neighbor node onto the exact hazard-selected ray before the same topology split, preserving element history indexing while removing mesh-angle steering.')
    p.add_argument('--czm-penalty-normal', type=float, default=1.0e18, dest='czm_penalty_normal',
                   help='Normal cohesive penalty stiffness [Pa/m]. The Arrhenius hazard remains the failure criterion.')
    p.add_argument('--czm-penalty-tangent', type=float, default=1.0e18, dest='czm_penalty_tangent',
                   help='Tangential cohesive penalty stiffness [Pa/m].')
    p.add_argument('--czm-event-damage', type=float, default=1.0, dest='czm_event_damage',
                   help='Broken-link fraction assigned after one completed Arrhenius renewal. 1 gives abrupt discrete link failure, matching the current renewal event.')
    p.add_argument('--czm-max-angle-error-deg', type=float, default=35.0, dest='czm_max_angle_error_deg',
                   help='Maximum steering error allowed between the hazard-selected direction and the existing mesh edge used by the migration CZM backend.')
    p.add_argument('--czm-min-area-ratio', type=float, default=0.08, dest='czm_min_area_ratio',
                   help='adaptive_czm: minimum new/old signed-area magnitude ratio for every element incident to the steered node.')
    p.add_argument('--czm-min-triangle-quality', type=float, default=0.035, dest='czm_min_triangle_quality',
                   help='adaptive_czm: minimum accepted local triangle quality 4*sqrt(3)*A/sum(l_i^2).')
    p.add_argument('--czm-max-node-move-factor', type=float, default=1.75, dest='czm_max_node_move_factor',
                   help='adaptive_czm: maximum local node relocation divided by requested physical crack increment. Set 0 to disable this limit.')

    p.add_argument('--czm-max-hrefine-subsegments', type=int, default=512, dest='czm_max_hrefine_subsegments',
                   help='adaptive_czm: maximum exact-ray local h-refinement subsegments used to realize one physical crack event before vetoing. Increase for long-growth runs through heavily refined crack wakes.')
    p.add_argument('--target-crack-extension-um', type=float, default=float('inf'), dest='target_crack_extension_um',
                   help='Stop a 2-D run after the leading crack/front has advanced this much beyond the initial notch [micrometers]. Use for long-growth Paris/morphology runs; inf disables.')
    p.add_argument('--snapshot-by-crack-extension-um', type=float, default=0.0, dest='snapshot_by_crack_extension_um',
                   help='In addition to step/advance snapshots, save field snapshots whenever leading crack extension crosses this increment [micrometers]. 0 disables extension-triggered snapshots.')
    p.add_argument('--max-da-per-block-um', type=float, default=float('inf'), dest='max_da_per_block_um',
                   help='Audit guard for long-growth runs: print a warning if leading crack advance in one accepted cycle block exceeds this value [micrometers]. This does not reject the block; use smaller --target-dB to reduce expected jumps.')
    p.add_argument('--rJ', type=float, default=None, dest='rJ',
                   help='ABSOLUTE J-integral contour radius [m] (2D). '
                        'Mesh-independent; default ~10*L_pz. Must enclose the '
                        'plastic zone and exceed ~3 elements.')

    # 2-D fatigue-cycle adapter.  This keeps the v8 sharp-front/multifront
    # geometry and replaces monotonic dt-based hazard accumulation by the same
    # cycle-integrated Arrhenius controller used by the V1 tuning model.
    p.add_argument('--fatigue-cycles', action='store_true',
                   help='Enable cyclic-fatigue coupling inside --mode 2d: each active v8 front uses its local J-derived Kmax, integrates hazards over K(t), updates the front-local process-zone ledger, then uses the existing sharp-front cleavage renewal clock for advance/branching.')
    p.add_argument('--fatigue-hold-load', action='store_true',
                   help='For --mode 2d --fatigue-cycles comparison/tuning runs, ramp by --dU on the first accepted step and then hold the same displacement amplitude for the remaining fatigue blocks. This gives a near-constant local Kmax test analogous to V1.')
    p.add_argument('--R', type=float, default=0.1,
                   help='Fatigue load ratio Kmin/Kmax for --fatigue-cycles.')
    p.add_argument('--frequency-Hz', type=float, default=1.0e3, dest='frequency_Hz',
                   help='Cyclic frequency for --fatigue-cycles.')
    p.add_argument('--block-cycles', type=float, default=1.0e4, dest='block_cycles',
                   help='Requested fatigue cycles per accepted 2D load-amplitude step.')
    p.add_argument('--cycles-max', type=float, default=float('inf'), dest='cycles_max',
                   help='Physical fatigue-cycle horizon for --fatigue-cycles. In hazard_limited mode this caps a block by the remaining horizon rather than imposing an artificial fixed cycles-per-step cap.')
    p.add_argument('--max-block-cycles', type=float, default=1.0e6, dest='max_block_cycles')
    p.add_argument('--min-block-cycles', type=float, default=1.0e-6, dest='min_block_cycles',
                   help='Minimum numerical fatigue cycle block. Fractional cycles are allowed for high-hazard cases.')
    p.add_argument('--no-adaptive-cycles', action='store_true',
                   help='Disable adaptive fatigue-cycle block sizing; use --block-cycles exactly.')
    p.add_argument('--cycle-block-mode', choices=['requested_cap', 'hazard_limited'], default='requested_cap', dest='cycle_block_mode',
                   help='requested_cap: --block-cycles is a hard upper bound. hazard_limited: use --max-block-cycles as the upper bound and choose the largest block allowed by fracture/plasticity hazard increments.')
    p.add_argument('--target-dB', type=float, default=0.2,
                   help='Target cleavage-clock increment per fatigue block for adaptive cycle sizing.')
    p.add_argument('--target-dN-store', type=float, default=0.25, dest='target_dN_store',
                   help='Target retained process-zone ledger increment per fatigue block.')
    p.add_argument('--target-dN-emit', type=float, default=float('inf'), dest='target_dN_emit',
                   help='Optional target emitted-event increment per fatigue block.')
    p.add_argument('--target-dN-mobile', type=float, default=float('inf'), dest='target_dN_mobile',
                   help='Optional target mobile PZ increment per fatigue block.')
    p.add_argument('--target-dN-escape', type=float, default=float('inf'), dest='target_dN_escape',
                   help='Optional target Peierls/Taylor escape increment per fatigue block.')
    p.add_argument('--target-dN-peierls', type=float, default=float('inf'), dest='target_dN_peierls',
                   help='Optional target Peierls hazard-clock increment per fatigue block.')
    p.add_argument('--target-dN-taylor', type=float, default=float('inf'), dest='target_dN_taylor',
                   help='Optional target Taylor hazard-clock increment per fatigue block.')
    p.add_argument('--n-phase', type=int, default=96, dest='n_phase',
                   help='Quadrature points over one fatigue cycle.')
    p.add_argument('--no-closure-clip', action='store_true',
                   help='Do not clip negative K during fatigue cycling.')
    p.add_argument('--storage-model', choices=['escape_limited', 'all_retained', 'fixed_fraction'],
                   default='escape_limited', dest='storage_model',
                   help='Fatigue storage law for emitted events. Use all_retained/fixed_fraction as ablation controls to detect escape-throttle domination.')
    p.add_argument('--fixed-retained-fraction', type=float, default=1.0, dest='fixed_retained_fraction')
    p.add_argument('--pz-recovery-per-s', type=float, default=0.0, dest='pz_recovery_per_s',
                   help='Optional recovery rate for the fatigue process-zone ledger [1/s].')
    p.add_argument('--no-pz-spatial-state', action='store_true',
                   help='Disable the v8 spatial process-zone fields and recover the V1 scalar-ledger reduction.')
    p.add_argument('--pz-field-coupling', type=float, default=1.0, dest='pz_field_coupling',
                   help='0..1 projection of the spatial stored PZ field back into each front ledger before hazard integration. 1=full v8 spatial coupling.')
    p.add_argument('--pz-deposit-radius-factor', type=float, default=1.0, dest='pz_deposit_radius_factor',
                   help='Multiplier on L_pz for depositing retained/mobile/escaped PZ densities around a front.')
    p.add_argument('--pz-store-to-rho-scale', type=float, default=1.0, dest='pz_store_to_rho_scale',
                   help='Scale converting retained PZ count into the existing rho_gp Taylor/plasticity density field. 0 disables this coupling.')
    p.add_argument('--pz-mobile-recovery-per-s', type=float, default=0.0, dest='pz_mobile_recovery_per_s',
                   help='Recovery/escape rate for the explicit mobile PZ density field [1/s].')
    p.add_argument('--pz-field-transport-c', type=float, default=None, dest='pz_field_transport_c',
                   help='Optional transport coefficient for PZ fields. Default uses --rho-transport-c; 0 disables PZ-field transport.')
    p.add_argument('--no-cyclic-mechanics', action='store_true',
                   help='Disable full 2-D cyclic mechanics in fatigue mode and use the scalar K(t) mechanics surrogate only.')
    p.add_argument('--cyclic-mechanics-phases', type=int, default=0, dest='cyclic_mechanics_phases',
                   help='Number of FEM/plasticity phases per accepted fatigue block. 0=min(n_phase,24).')
    p.add_argument('--cyclic-mechanics-stagger', type=int, default=1, dest='cyclic_mechanics_stagger',
                   help='Mech/plastic stagger iterations at each cyclic phase.')
    p.add_argument('--exp-system', default='W[100]',
                   choices=['W[100]', 'Ta[111]', 'Al0.7CoCrFeNi-BCC', 'Al0.7CoCrFeNi-FCC', 'Cu'],
                   help='EXP-floor barrier family for fatigue emission/Peierls/Taylor hazards.')
    p.add_argument('--exp-a', type=float, default=None, help='Override EXP-floor shape a.')
    p.add_argument('--exp-n', type=float, default=None, help='Override EXP-floor shape n.')
    p.add_argument('--nu0-emit-pz', type=float, default=1.0e11)
    p.add_argument('--nu0-peierls', type=float, default=1.0e12)
    p.add_argument('--nu0-taylor', type=float, default=1.0e11)
    p.add_argument('--emit-energy-scale', type=float, default=1.0)
    p.add_argument('--emit-entropy-scale', type=float, default=1.0)
    p.add_argument('--emit-stress-scale', type=float, default=1.0)
    p.add_argument('--peierls-energy-scale', type=float, default=0.005)
    p.add_argument('--peierls-entropy-scale', type=float, default=0.005)
    p.add_argument('--peierls-stress-scale', type=float, default=1.0)
    p.add_argument('--taylor-energy-scale', type=float, default=0.02)
    p.add_argument('--taylor-entropy-scale', type=float, default=0.02)
    p.add_argument('--taylor-stress-scale', type=float, default=1.0)
    return p


def run_mesh_sweep(args):
    """Mesh-convergence study: hold dt and temperature fixed, refine the tip
    resolution (tip_h_fine) from coarse through sub-L_pz, and record how the
    initiation toughness and process-zone scalars (sigma_back, r_eff/r0 at
    initiation) converge.  Each refinement level runs a full 2D solve per
    temperature in its own subdir; a combined table + convergence plot summarize.

    The convergence target depends on temperature: at a brittle T the cleavage
    Kc should be nearly hbar-independent (the process zone is inactive); at a
    near-transition T sigma_back/r_eff are active and are the real test of
    whether resolving the process zone (hbar_tip < L_pz) changes the answer.
    """
    import copy as _copy
    levels = args.mesh_levels
    os.makedirs(args.out, exist_ok=True)
    rows = []
    print("=" * 72)
    print(f"  MESH-CONVERGENCE SWEEP — {len(levels)} tip resolutions x "
          f"{len(args.temperatures)} T")
    print(f"  tip_h_fine levels [m]: {levels}")
    print(f"  L_pz = {args.L_pz:.1e} m  (process zone resolved when hbar_tip < L_pz)")
    print("=" * 72)

    for i, hf in enumerate(levels):
        sub = _copy.copy(args)
        sub.mode = '2d'
        sub.tip_h_fine = hf
        sub.out = os.path.join(args.out, f"lvl_{i:02d}_hf-{hf:.0e}")
        sub.save_snapshots = max(0, args.save_snapshots)
        print(f"\n[level {i+1}/{len(levels)}] tip_h_fine = {hf:.2e} m")
        summary = run_2d(sub)
        for s in summary:
            rows.append({
                'level': i, 'tip_h_fine_m': hf,
                'hbar_tip_m': s.get('hbar_tip_m'), 'n_nodes': s.get('n_nodes'),
                'T': s['T'],
                'pz_resolved': bool(s.get('hbar_tip_m', 1.0) < args.L_pz),
                'Kc_first_MPa_sqrt_m': s['Kc_first_MPa_sqrt_m'],
                'sigma_back_init_GPa': s.get('sigma_back_init_GPa'),
                'r_eff_over_r0_init': s.get('r_eff_over_r0_init'),
                'N_em_init': s.get('N_em_init'),
                'mode': s['mode'],
            })

    # combined table
    import csv as _csv
    cols = ['level', 'tip_h_fine_m', 'hbar_tip_m', 'n_nodes', 'T',
            'pz_resolved', 'Kc_first_MPa_sqrt_m', 'sigma_back_init_GPa',
            'r_eff_over_r0_init', 'N_em_init', 'mode']
    with open(os.path.join(args.out, 'mesh_convergence.csv'), 'w', newline='') as fp:
        w = _csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out, 'mesh_convergence.json'), 'w') as fp:
        json.dump(rows, fp, indent=2)

    # per-temperature convergence report + plot
    _render_mesh_convergence(args.out, rows, args.L_pz)

    print("\n  Mesh sweep complete.")
    print(f"  Combined table: {os.path.join(args.out, 'mesh_convergence.csv')}")
    # console convergence summary
    for T in sorted(set(r['T'] for r in rows)):
        tr = [r for r in rows if r['T'] == T]
        tr.sort(key=lambda r: -r['tip_h_fine_m'])  # coarse -> fine
        print(f"\n  T = {T:.0f} K  (coarse -> fine):")
        print(f"    {'hbar_tip[um]':>12} {'nodes':>7} {'PZ?':>4} "
              f"{'Kc':>7} {'sig_back':>9} {'r_eff/r0':>9}")
        for r in tr:
            ht = (r['hbar_tip_m'] or 0) * 1e6
            kc = r['Kc_first_MPa_sqrt_m']
            kcs = 'none' if kc is None else f"{kc:.3f}"
            print(f"    {ht:12.3f} {r['n_nodes']:7d} "
                  f"{'yes' if r['pz_resolved'] else 'no':>4} {kcs:>7} "
                  f"{r['sigma_back_init_GPa']:9.4f} {r['r_eff_over_r0_init']:9.5f}")
    return rows


def _render_mesh_convergence(out_dir, rows, L_pz):
    """Plot Kc, sigma_back, r_eff/r0 vs tip resolution, one line per T."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    temps = sorted(set(r['T'] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for T in temps:
        tr = [r for r in rows if r['T'] == T]
        tr.sort(key=lambda r: r['hbar_tip_m'] or 0)
        ht = [(r['hbar_tip_m'] or 0) * 1e6 for r in tr]
        kc = [r['Kc_first_MPa_sqrt_m'] for r in tr]
        sb = [r['sigma_back_init_GPa'] for r in tr]
        re = [r['r_eff_over_r0_init'] for r in tr]
        lbl = f"{T:.0f} K"
        ax[0].plot(ht, kc, 'o-', label=lbl)
        ax[1].plot(ht, sb, 's-', label=lbl)
        ax[2].plot(ht, re, '^-', label=lbl)
    for a in ax:
        a.set_xscale('log')
        a.axvline(L_pz * 1e6, ls='--', color='gray', lw=1)
        a.set_xlabel(r'tip resolution $\bar h_{tip}$ [$\mu$m]')
        a.legend(fontsize=8)
        a.invert_xaxis()  # coarse (left) -> fine (right)
    ax[0].set_ylabel(r'$K_{c,\,first}$ [MPa$\sqrt{m}$]')
    ax[0].set_title('Initiation toughness')
    ax[1].set_ylabel(r'$\sigma_{back}$ at initiation [GPa]')
    ax[1].set_title('Back-stress (process zone)')
    ax[2].set_ylabel(r'$r_{eff}/r_0$ at initiation')
    ax[2].set_title('Tip blunting (process zone)')
    fig.suptitle('Mesh convergence (dashed line: $L_{pz}$; '
                 'right of it = process zone resolved)', y=1.02)
    fig.tight_layout()
    p = os.path.join(out_dir, 'mesh_convergence.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.mode == 'sweep':
        return run_sweep(args)
    if args.mode == 'mesh-sweep':
        return run_mesh_sweep(args)
    if args.mode == '1d':
        return run_1d(args)
    return run_2d(args)


if __name__ == '__main__':
    main()
