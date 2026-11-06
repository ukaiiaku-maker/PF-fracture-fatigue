"""Optional full-section plasticity for the scalar Ast cantilever.

The tip-only adapter confines plasticity to the native crack-tip MPZ strip.
The ``full_section`` option lets the same native v9.13 mechanisms operate at
every material point of the beam: persistent-site nucleation on the emission
surface, signed Peierls glide, Taylor capture/release and the persistent
sqrt(rho) backstress. Every rate is evaluated by the native state class's own
``_arrhenius_rate``/``_signed_rate`` with the candidate's own surfaces, so the
Arrhenius form and parameters are those of the tip model.

Mechanics are Euler-Bernoulli beam statics. The beam is divided into span
stations and through-height fibres. The plastic axial strain of a fibre is the
Schmid projection of its signed slip. Because the cantilever is statically
determinate, the fibre stress is the external bending stress plus the
self-equilibrated eigenstrain residual of its own section, and the load-point
plastic displacement is the unit-force virtual work of the eigenstrain:

    sigma = M(s)(c-y)/I - E[eps_p - P_s(eps_p)],
    du_pl = sum_s s ds dkappa_p(s),  dkappa_p = int (c-y) deps_p dA / I,

where P_s projects onto {1, c-y} over the currently uncracked fibres.

Declared conventions (not optimizer coordinates):
  * bulk carriers: by default no bulk nucleation (the emission surface is a
    crack-tip/surface process); the pre-existing native forest floor
    rho_forest_floor_m2 is the initial mobile population of every point.
    Because Orowan flow is b*rho_m*v, a factor-x error in this density
    shifts an inferred Peierls barrier by only kT*ln(x);
  * optional bulk nucleation (``bulk_nucleation=True``): persistent sites at
    rho_source0_m2**1.5 per volume, each activation creating the native line
    quantum c_line spanning the specimen front B;
  * the notch-ligament hinge station has length ``hinge_length_m`` (default:
    the initial ligament W-a0) and the cracked section; the other stations
    have the full height W;
  * omitted in the bulk: GND kernel stresses beyond the section residual,
    annihilation, transport between points and bulk contributions to tip
    shielding. Overlap with the 0.2 um tip strip is not subtracted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np

MODES = ("tip_only", "full_section")


class SectionSaturation(ArithmeticError):
    """Bulk flow at a trial force exceeded the resolvable substep budget."""


@dataclass(frozen=True)
class SectionPlasticityConfig:
    mode: str = "tip_only"
    bulk_nucleation: bool = False
    initial_mobile: str = "forest_floor"
    fibres: int = 48
    stations_outside_hinge: int = 12
    hinge_length_m: float | None = None
    specimen_length_m: float = 15e-6
    maximum_substep_stress_change_Pa: float = 20e6
    maximum_substeps: int = 2000
    root_iterations: int = 60

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"plasticity mode must be one of {MODES}")
        if self.fibres < 8 or self.stations_outside_hinge < 2:
            raise ValueError("section plasticity needs at least 8 fibres and 2 stations")
        if self.hinge_length_m is not None and self.hinge_length_m <= 0:
            raise ValueError("hinge length must be positive")
        if not (self.maximum_substep_stress_change_Pa > 0 and self.maximum_substeps > 0):
            raise ValueError("invalid section substep limits")
        if self.initial_mobile not in ("forest_floor", "zero"):
            raise ValueError("initial_mobile must be 'forest_floor' or 'zero'")
        if self.mode == "full_section" and not self.bulk_nucleation and self.initial_mobile == "zero":
            raise ValueError("bulk has no carriers without nucleation or an initial population")

    @property
    def active(self) -> bool:
        return self.mode == "full_section"

    def as_dict(self):
        out = asdict(self)
        out.update({
            "bulk_site_density_convention": ("rho_source0_m2**1.5" if self.bulk_nucleation
                                             else "none"),
            "bulk_line_per_activation_convention": ("activation_to_line_content * B"
                                                    if self.bulk_nucleation else "none"),
            "initial_mobile_density": ("rho_forest_floor_m2" if self.initial_mobile == "forest_floor"
                                       else "0"),
            "beam_mechanics": "Euler-Bernoulli, statically determinate eigenstrain",
            "omitted_bulk_terms": ["gnd_kernel_beyond_section_residual",
                                   "annihilation", "inter_point_transport",
                                   "bulk_tip_shielding"]})
        return out


class SectionPlasticity:
    """Bulk native-mechanism state over span stations and height fibres."""

    def __init__(self, candidate, physics, specimen, config: SectionPlasticityConfig,
                 native_state_type, *, peierls_surface=None):
        if not config.active:
            raise ValueError("section state requires mode='full_section'")
        if specimen.L_m >= config.specimen_length_m:
            raise ValueError("notch lever arm must lie inside the specimen length")
        self.p = candidate
        self.c = physics
        self.cfg = config
        self.native = native_state_type
        self.E = float(specimen.E_Pa)
        self.B = float(specimen.B_m)
        W = float(specimen.W_m)
        self.dy = W / config.fibres
        self.y = (np.arange(config.fibres) + .5) * self.dy
        hinge = (float(config.hinge_length_m) if config.hinge_length_m is not None
                 else W - float(specimen.a0_m))
        L = float(specimen.L_m)
        lo = max(L - .5 * hinge, 0.0)
        hi = min(L + .5 * hinge, float(config.specimen_length_m))
        n = config.stations_outside_hinge
        n_before = max(int(round(n * lo / (lo + config.specimen_length_m - hi))), 1) if lo > 0 else 0
        n_after = n - n_before if hi < config.specimen_length_m else 0
        edges = []
        if n_before:
            edges.append(np.linspace(0.0, lo, n_before + 1))
        edges.append(np.array([lo, hi]))
        if n_after:
            edges.append(np.linspace(hi, config.specimen_length_m, n_after + 1))
        starts = np.concatenate([e[:-1] for e in edges])
        ends = np.concatenate([e[1:] for e in edges])
        self.ds = ends - starts
        self.s = .5 * (starts + ends)
        self.hinge = np.isclose(starts, lo) & np.isclose(ends, hi)
        if np.count_nonzero(self.hinge) != 1:
            raise ValueError("ambiguous hinge station")
        self.hinge_length_m = hi - lo
        self.schmid = np.abs(np.asarray(physics.emission_schmid_factors, float))
        nsys = physics.n_systems
        shape = (nsys, self.s.size, self.y.size)
        self.mobile_m2 = np.full(shape, float(physics.rho_forest_floor_m2)
                                 if config.initial_mobile == "forest_floor" else 0.0)
        self.retained_m2 = np.zeros(shape)
        self.slip = np.zeros(shape)
        self.eps_p = np.zeros((self.s.size, self.y.size))
        rho_s = max(float(candidate.rho_source0_m2), 0.0)
        self.site_density_m3 = rho_s ** 1.5
        self.line_per_activation_m = (
            np.asarray(physics.activation_to_line_content_per_system, float) * self.B)
        self.kback = (max(float(physics.persistent_backstress_scale), 0.0) *
                      float(physics.G_Pa) * abs(float(physics.b_m)) /
                      max(abs(float(candidate.taylor.stress_fraction)), 1e-6))
        self.peierls_surface = (candidate.peierls.surface(candidate.emission)
                                if peierls_surface is None else peierls_surface)
        self.taylor_surface = candidate.taylor.surface(candidate.emission)
        self.upl_m = 0.0
        self.work_J = 0.0
        self.external_work_J = 0.0
        self.born_line_m = 0.0
        self.substeps = 0
        self.max_line_balance_residual = 0.0

    def clone(self) -> "SectionPlasticity":
        """Copy evolving fields; immutable material/geometry stay shared."""
        new = object.__new__(SectionPlasticity)
        new.__dict__.update(self.__dict__)
        for name in ("mobile_m2", "retained_m2", "slip", "eps_p"):
            setattr(new, name, getattr(self, name).copy())
        return new

    # Geometry -----------------------------------------------------------
    def weights(self, a_m: float) -> np.ndarray:
        """Uncracked fibre fractions; the crack only cuts the hinge station."""
        w = np.ones((self.s.size, self.y.size))
        top = self.y + .5 * self.dy
        cut = np.clip((top - float(a_m)) / self.dy, 0.0, 1.0)
        w[self.hinge] = cut
        return w

    def section(self, a_m: float):
        w = self.weights(a_m)
        area = np.sum(w, axis=1) * self.dy
        if np.any(area <= 0):
            raise ArithmeticError("section fully cracked")
        c = np.sum(w * self.y, axis=1) * self.dy / area
        arm = c[:, None] - self.y[None, :]
        inertia = np.sum(w * arm * arm, axis=1) * self.dy
        return w, area, arm, inertia

    def stress(self, force_N: float, a_m: float) -> np.ndarray:
        w, area, arm, inertia = self.section(a_m)
        external = (force_N * self.s / (self.B * inertia))[:, None] * arm
        return w * (external - self.E * self._free_eigenstrain(self.eps_p, w, area, arm, inertia))

    def _free_eigenstrain(self, eps, w, area, arm, inertia):
        mean = np.sum(w * eps, axis=1) * self.dy / area
        slope = np.sum(w * eps * arm, axis=1) * self.dy / inertia
        return eps - mean[:, None] - slope[:, None] * arm

    def plastic_displacement_increment(self, deps: np.ndarray, a_m: float) -> float:
        w, _, arm, inertia = self.section(a_m)
        dkappa = np.sum(w * arm * deps, axis=1) * self.dy / inertia
        return float(np.sum(self.s * self.ds * dkappa))

    # Native kinetics ------------------------------------------------------
    def _rates(self, sigma: np.ndarray, T_K: float):
        c, p, native = self.c, self.p, self.native
        tau = self.schmid[:, None, None] * sigma[None, :, :]
        total = self.mobile_m2 + self.retained_m2
        interaction = np.asarray(c.forest_interaction_matrix, float)
        forest = np.maximum(c.rho_forest_floor_m2 +
                            np.einsum("ij,jsy->isy", interaction, total), 1.0)
        spacing = 1.0 / (2.0 * np.sqrt(forest))
        jump = c.jump_fraction_of_forest_spacing * spacing
        peierls = native._signed_rate(self.peierls_surface,
                                      p.peierls.stress_fraction * tau, T_K,
                                      p.peierls.nu0_s)
        peierls_velocity = jump * peierls
        velocity = max(float(c.mobile_transport_velocity_scale), 0.0) * peierls_velocity
        phi = spacing / max(c.b_m, 1e-30)
        if np.isfinite(float(c.taylor_phi_max)):
            phi = np.minimum(phi, float(c.taylor_phi_max))
        taylor_single = np.maximum(native._signed_rate(
            self.taylor_surface, p.taylor.stress_fraction * np.abs(tau) * phi,
            T_K, p.taylor.nu0_s), 0.0)
        corr = p.taylor_corr_scale / (2.0 * sqrt(max(p.taylor_corr_rho_c_m2, 1e-300)))
        release = taylor_single / np.maximum(1.0 + 2.0 * corr * np.sqrt(forest), 1.0)
        mfp = c.mean_free_path_coefficient / np.sqrt(forest)
        encounter = (max(float(c.encounter_efficiency), 0.0) *
                     np.abs(peierls_velocity) / np.maximum(mfp, 1e-30))
        drive = np.abs(tau)
        return velocity, encounter, release, drive, total

    def _emission_rate(self, net_stress: np.ndarray, T_K: float) -> np.ndarray:
        rate = self.native._arrhenius_rate(
            self.p.emission.barrier_eV(np.maximum(net_stress, 0.0), T_K),
            T_K, self.c.emission_nu0_s)
        return np.where(net_stress > 0, rate, 0.0)

    def _activations(self, h, drive, rho, T_K):
        """Vectorized native backstress-limited root A = M h r(d-k sqrt(rho+qA))."""
        q = self.line_per_activation_m[:, None, None] * np.ones_like(drive)
        M = self.site_density_m3
        net0 = drive - self.kback * np.sqrt(np.maximum(rho, 0.0))
        upper = M * h * self._emission_rate(net0, T_K)
        blocking = np.maximum(((drive / self.kback) ** 2 - rho) / q, 0.0)
        upper = np.where(net0 > 0, np.minimum(upper, blocking), 0.0)
        lo = np.zeros_like(upper)
        hi = upper.copy()
        for _ in range(self.cfg.root_iterations):
            mid = .5 * (lo + hi)
            rate = self._emission_rate(drive - self.kback * np.sqrt(rho + q * mid), T_K)
            above = mid >= M * h * rate
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
            if np.all(hi - lo <= 1e-12 * np.maximum(hi, 1e-300)):
                break
        return hi * q

    def _substep(self, h, force_N, a_m, T_K):
        w = self.weights(a_m)
        sigma = self.stress(force_N, a_m)
        velocity, encounter, release, drive, total = self._rates(sigma, T_K)
        born = (self._activations(h, drive, total, T_K) * (w[None] > 0)
                if self.cfg.bulk_nucleation else np.zeros_like(total))
        m0, r0 = self.mobile_m2, self.retained_m2
        # Backward-Euler local capture/release with births in the mobile pool.
        # det=(1+he)(1+hr)-h^2 e r, written without cancellation.
        det = 1.0 + h * encounter + h * release
        rhs_m = m0 + born
        mobile = (rhs_m * (1.0 + h * release) + h * release * r0) / det
        retained = ((1.0 + h * encounter) * r0 + h * encounter * rhs_m) / det
        if np.any(mobile < -1e-9 * np.maximum(total + born, 1.0)) or not np.all(np.isfinite(mobile)):
            raise ArithmeticError("invalid bulk mobile/retained update")
        mobile = np.maximum(mobile, 0.0)
        retained = np.maximum(retained, 0.0)
        balance = np.max(np.abs(mobile + retained - total - born))
        self.max_line_balance_residual = max(self.max_line_balance_residual, float(balance))
        if balance > 1e-8 * max(float(np.max(total + born)), 1.0):
            raise ArithmeticError("bulk line-content balance")
        dgamma = abs(float(self.c.b_m)) * mobile * velocity * h * (w[None] > 0)
        deps = np.sum(self.schmid[:, None, None] * dgamma, axis=0)
        return sigma, born, mobile, retained, dgamma, deps

    def advance(self, dt_s: float, force_start_N: float, force_end_N: float,
                a_m: float, T_K: float, du_limit_m: float = float("inf")) -> float:
        """Advance with force linear in time; return the load-point du_pl.

        A trial force whose bulk flow already exceeds ``du_limit_m`` cannot be
        an equilibrium root, so integration stops with SectionSaturation.
        """
        if dt_s <= 0:
            return 0.0
        elapsed = 0.0
        h = float(dt_s)
        du = 0.0
        attempts = 0
        tol = self.cfg.maximum_substep_stress_change_Pa
        while elapsed < dt_s * (1 - 1e-12):
            attempts += 1
            if attempts > self.cfg.maximum_substeps:
                raise SectionSaturation("bulk flow exceeded the substep budget")
            h = min(h, dt_s - elapsed)
            fraction = (elapsed + .5 * h) / dt_s
            force = force_start_N + fraction * (force_end_N - force_start_N)
            sigma, born, mobile, retained, dgamma, deps = self._substep(h, force, a_m, T_K)
            if self.E * float(np.max(np.abs(deps))) > tol and h > dt_s * 1e-9:
                h *= .5
                continue
            ddu = self.plastic_displacement_increment(deps, a_m)
            volume = self.B * self.dy * self.ds[:, None]
            self.work_J += float(np.sum(sigma * deps * volume))
            self.external_work_J += force * ddu
            self.born_line_m += float(np.sum(born * volume[None]))
            self.mobile_m2, self.retained_m2 = mobile, retained
            self.slip += dgamma
            self.eps_p = self.eps_p + deps
            self.upl_m += ddu
            du += ddu
            elapsed += h
            self.substeps += 1
            if du > du_limit_m:
                raise SectionSaturation("bulk flow exceeds the available displacement")
            if self.E * float(np.max(np.abs(deps))) < .25 * tol:
                h *= 2.0
        return du

    def snapshot(self, force_N: float, a_m: float) -> dict:
        sigma = self.stress(force_N, a_m)
        h = int(np.flatnonzero(self.hinge)[0])
        return {"bulk_plastic_u_m": self.upl_m,
                "bulk_plastic_work_J": self.work_J,
                "bulk_external_plastic_work_J": self.external_work_J,
                "bulk_born_line_m": self.born_line_m,
                "bulk_substeps": self.substeps,
                "bulk_max_line_balance_residual_m2": self.max_line_balance_residual,
                "hinge_max_abs_eps_p": float(np.max(np.abs(self.eps_p[h]))),
                "hinge_max_density_m2": float(np.max(self.mobile_m2[:, h] + self.retained_m2[:, h])),
                "hinge_max_mobile_m2": float(np.max(self.mobile_m2[:, h])),
                "hinge_max_abs_stress_Pa": float(np.max(np.abs(sigma[h])))}
