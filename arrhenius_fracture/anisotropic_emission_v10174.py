"""v10.1.7.4 reduced anisotropic crack-tip emission.

This module adds a two-dimensional slip-trace-resolved emission closure to the
campaign-calibrated stochastic avalanche tip engine without changing the
validated crack-geometry transaction.  The FEM/J solution supplies the absolute
sharp-tip opening stress.  Damage-excluding finite-radius tensor probes supply
only the channel shape factors.

The implementation deliberately keeps the two promoted in-plane channels.  It
is not a full three-dimensional BCC crystal-plasticity model.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Callable

import numpy as np

from .campaign_calibrated_tip import _campaign_backstress
from .stochastic_avalanche_tip import StochasticAvalancheDiagnosticTipEngine


MODEL_ID = "v10.1.7.4_reduced_2d_slip_trace_anisotropic_emission"


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-30:
        return np.array([1.0, 0.0], dtype=float)
    return value / norm


@dataclass
class AnisotropicEmissionConfig:
    enabled: bool = True
    crystal_theta_deg: float = 45.0
    probe_radius_m: float = 10.0e-6
    sector_half_angle_deg: float = 25.0
    damage_cutoff: float = 0.85
    min_elements: int = 3
    schmid_reference: float = 0.5
    shared_forest_density: bool = True
    require_reliable_probe: bool = True

    def validate(self) -> "AnisotropicEmissionConfig":
        self.enabled = bool(self.enabled)
        self.crystal_theta_deg = float(self.crystal_theta_deg)
        self.probe_radius_m = max(float(self.probe_radius_m), 1.0e-12)
        self.sector_half_angle_deg = float(
            np.clip(float(self.sector_half_angle_deg), 1.0, 85.0)
        )
        self.damage_cutoff = float(np.clip(float(self.damage_cutoff), 0.0, 1.0))
        self.min_elements = max(int(self.min_elements), 1)
        self.schmid_reference = max(abs(float(self.schmid_reference)), 1.0e-12)
        self.shared_forest_density = bool(self.shared_forest_density)
        self.require_reliable_probe = bool(self.require_reliable_probe)
        return self


@dataclass
class _MechanicsObserver:
    mechanics_serial: int = 0
    drive_serial: int = 0
    mesh: Any = None
    damage: np.ndarray | None = None
    sigma_gp: np.ndarray | None = None
    latest_drive: dict[str, Any] | None = None
    reliable_drive_count: int = 0
    fallback_drive_count: int = 0
    failed_probe_count: int = 0

    def clear(self) -> None:
        self.mechanics_serial = 0
        self.drive_serial = 0
        self.mesh = None
        self.damage = None
        self.sigma_gp = None
        self.latest_drive = None
        self.reliable_drive_count = 0
        self.fallback_drive_count = 0
        self.failed_probe_count = 0


OBSERVER = _MechanicsObserver()


def _element_damage(mesh, damage) -> np.ndarray:
    value = np.asarray(damage, dtype=float).reshape(-1)
    elems = np.asarray(mesh.elems, dtype=int)
    if value.size == int(mesh.ne):
        return value.copy()
    if value.size == int(mesh.nn):
        return np.mean(value[elems], axis=1)
    raise ValueError(
        f"damage field has {value.size} entries; expected mesh.nn={mesh.nn} "
        f"or mesh.ne={mesh.ne}"
    )


def _element_centroids(mesh) -> np.ndarray:
    return np.asarray(mesh.nodes, dtype=float)[
        np.asarray(mesh.elems, dtype=int)
    ].mean(axis=1)


def infer_front_direction(mesh, damage, tip_xy, radius_m: float) -> np.ndarray:
    """Infer the local forward direction from the nearby damaged wake.

    The explicit front direction is not exposed by the protected v10.1 driver.
    The vector from the local damaged-wake centroid to the tip supplies the
    orientation, while a PCA axis supplies a noise-resistant line direction.
    """
    tip = np.asarray(tip_xy, dtype=float).reshape(2)
    centroids = _element_centroids(mesh)
    de = _element_damage(mesh, damage)
    distance = np.linalg.norm(centroids - tip[None, :], axis=1)
    selected = (de >= 0.5) & (distance <= max(4.0 * float(radius_m), 1.0e-12))
    points = centroids[selected]
    weights = np.maximum(de[selected], 1.0e-12)
    if points.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=float)

    mean = np.average(points, axis=0, weights=weights)
    wake_to_tip = tip - mean
    centered = points - mean[None, :]
    covariance = (centered * weights[:, None]).T @ centered
    if np.all(np.isfinite(covariance)) and np.linalg.norm(covariance) > 0.0:
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        if float(direction @ wake_to_tip) < 0.0:
            direction = -direction
    else:
        direction = wake_to_tip

    direction = _unit(direction)
    # The protected driver forbids global reversal. Preserve that convention.
    if direction[0] < 0.0:
        direction = -direction
    return direction


def probe_tensor_ahead(
    mesh,
    sigma_gp,
    damage,
    tip_xy,
    ray_direction,
    config: AnisotropicEmissionConfig,
) -> dict[str, Any]:
    """Area-weighted tensor probe in undamaged material ahead of one ray."""
    cfg = copy.deepcopy(config).validate()
    tip = np.asarray(tip_xy, dtype=float).reshape(2)
    ray = _unit(ray_direction)
    perpendicular = np.array([-ray[1], ray[0]], dtype=float)
    centroids = _element_centroids(mesh)
    offset = centroids - tip[None, :]
    longitudinal = offset @ ray
    transverse = offset @ perpendicular
    radius = np.linalg.norm(offset, axis=1)
    angle = np.degrees(
        np.arctan2(np.abs(transverse), np.maximum(longitudinal, 1.0e-30))
    )
    de = _element_damage(mesh, damage)

    selected = np.zeros(int(mesh.ne), dtype=bool)
    expansion_used = 0.0
    for expansion in (1.0, 1.5, 2.0, 3.0):
        radial_min = max(0.25 * cfg.probe_radius_m / expansion, 1.0e-12)
        radial_max = 1.75 * cfg.probe_radius_m * expansion
        angular_max = min(cfg.sector_half_angle_deg * expansion, 85.0)
        selected = (
            (longitudinal > 0.0)
            & (radius >= radial_min)
            & (radius <= radial_max)
            & (angle <= angular_max)
            & (de < cfg.damage_cutoff)
        )
        expansion_used = expansion
        if int(np.count_nonzero(selected)) >= cfg.min_elements:
            break

    if int(np.count_nonzero(selected)) < cfg.min_elements:
        admissible = np.flatnonzero((longitudinal > 0.0) & (de < cfg.damage_cutoff))
        if admissible.size:
            order = admissible[np.argsort(radius[admissible])]
            selected = np.zeros(int(mesh.ne), dtype=bool)
            selected[order[: cfg.min_elements]] = True

    indices = np.flatnonzero(selected)
    if indices.size < cfg.min_elements:
        return {
            "reliable": False,
            "n_elements": int(indices.size),
            "expansion": float(expansion_used),
            "ray_direction": ray.tolist(),
        }

    area = np.maximum(np.asarray(mesh.area_e, dtype=float)[indices], 1.0e-30)
    weights = area / float(np.sum(area))
    stress = np.asarray(sigma_gp, dtype=float)
    sxx = float(weights @ stress[0, indices])
    syy = float(weights @ stress[1, indices])
    sxy = float(weights @ stress[2, indices])
    tensor = np.array([[sxx, sxy], [sxy, syy]], dtype=float)
    return {
        "reliable": bool(np.all(np.isfinite(tensor))),
        "n_elements": int(indices.size),
        "expansion": float(expansion_used),
        "ray_direction": ray.tolist(),
        "tensor": tensor,
    }


def resolve_channel_drives(
    opening_tensor,
    channel_tensors,
    crystal_theta_deg: float,
    schmid_reference: float = 0.5,
) -> dict[str, Any]:
    """Resolve two BCC slip-trace channels without normalization or clipping."""
    from .crystal import bcc_slip_traces

    opening = np.asarray(opening_tensor, dtype=float).reshape(2, 2)
    eig = np.linalg.eigvalsh(opening)
    sigma1 = float(eig[-1])
    traces = bcc_slip_traces(float(crystal_theta_deg))
    if len(traces) != 2:
        raise RuntimeError(
            f"v10.1.7.4 requires exactly two reduced slip traces; got {len(traces)}"
        )
    tensors = [np.asarray(value, dtype=float).reshape(2, 2) for value in channel_tensors]
    if len(tensors) != len(traces):
        raise ValueError("one tensor is required for each slip-trace channel")

    # The opening normal is perpendicular to the local crack direction.  The
    # caller stores sigma_nn explicitly; sigma1 is the positive amplitude floor.
    sigma_nn = float(max(opening[1, 1], 0.0))
    sigma_amplitude = max(sigma1, sigma_nn, 1.0)
    reference = max(abs(float(schmid_reference)), 1.0e-12)

    names: list[str] = []
    signed: list[float] = []
    factors: list[float] = []
    directions: list[list[float]] = []
    normals: list[list[float]] = []
    for trace, tensor in zip(traces, tensors):
        t = _unit(trace["t"])
        n = _unit(trace["n"])
        tau = float(t @ tensor @ n)
        names.append(str(trace["name"]))
        signed.append(tau)
        factors.append(abs(tau) / (reference * sigma_amplitude))
        directions.append(t.tolist())
        normals.append(n.tolist())

    return {
        "channel_names": names,
        "trace_directions": directions,
        "trace_normals": normals,
        "tau_signed_Pa": signed,
        "drive_factors": factors,
        "sigma_amplitude_Pa": float(sigma_amplitude),
        "sigma1_probe_Pa": float(sigma1),
        "schmid_reference": float(reference),
        "factors_normalized": False,
        "factors_clipped": False,
    }


def build_front_drive(
    mesh,
    sigma_gp,
    damage,
    tip_xy,
    config: AnisotropicEmissionConfig,
) -> dict[str, Any]:
    """Build one front-local anisotropic emission drive from a mechanical state."""
    cfg = copy.deepcopy(config).validate()
    forward = infer_front_direction(mesh, damage, tip_xy, cfg.probe_radius_m)
    normal = np.array([-forward[1], forward[0]], dtype=float)
    opening_probe = probe_tensor_ahead(
        mesh, sigma_gp, damage, tip_xy, forward, cfg
    )
    if not opening_probe.get("reliable", False):
        raise RuntimeError("opening tensor probe is unreliable")

    opening_tensor = np.asarray(opening_probe["tensor"], dtype=float)
    sigma_nn = float(normal @ opening_tensor @ normal)

    from .crystal import bcc_slip_traces

    channel_probes: list[dict[str, Any]] = []
    channel_tensors: list[np.ndarray] = []
    for trace in bcc_slip_traces(cfg.crystal_theta_deg):
        ray = _unit(trace["t"])
        if float(ray @ forward) < 0.0:
            ray = -ray
        probe = probe_tensor_ahead(mesh, sigma_gp, damage, tip_xy, ray, cfg)
        channel_probes.append(probe)
        if not probe.get("reliable", False):
            raise RuntimeError(f"channel tensor probe failed for {trace['name']}")
        channel_tensors.append(np.asarray(probe["tensor"], dtype=float))

    drive = resolve_channel_drives(
        opening_tensor,
        channel_tensors,
        cfg.crystal_theta_deg,
        cfg.schmid_reference,
    )
    eig = np.linalg.eigvalsh(opening_tensor)
    sigma1 = float(eig[-1])
    sigma_amplitude = max(sigma1, max(sigma_nn, 0.0), 1.0)
    # Recompute factors with the crack-local opening amplitude rather than the
    # specimen-y component used by the pure tensor helper.
    drive["sigma_amplitude_Pa"] = float(sigma_amplitude)
    drive["sigma_nn_probe_Pa"] = float(sigma_nn)
    drive["sigma1_probe_Pa"] = float(sigma1)
    drive["drive_factors"] = [
        abs(float(tau)) / (cfg.schmid_reference * sigma_amplitude)
        for tau in drive["tau_signed_Pa"]
    ]
    drive.update({
        "model_id": MODEL_ID,
        "reliable": True,
        "front_direction": forward.tolist(),
        "front_normal": normal.tolist(),
        "tip_xy_m": np.asarray(tip_xy, dtype=float).reshape(2).tolist(),
        "opening_probe_elements": int(opening_probe["n_elements"]),
        "opening_probe_expansion": float(opening_probe["expansion"]),
        "channel_probe_elements": [
            int(probe["n_elements"]) for probe in channel_probes
        ],
        "channel_probe_expansion": [
            float(probe["expansion"]) for probe in channel_probes
        ],
        "mechanics_serial": int(OBSERVER.mechanics_serial),
    })
    return drive


def wrap_assemble_mechanics(original: Callable) -> Callable:
    """Capture the accepted FEM stress/damage state without altering mechanics."""
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            mesh = args[0]
            damage = args[4]
            sigma_gp = result[2]
            OBSERVER.mesh = mesh
            OBSERVER.damage = np.asarray(damage, dtype=float).copy()
            OBSERVER.sigma_gp = np.asarray(sigma_gp, dtype=float).copy()
            OBSERVER.mechanics_serial += 1
        except Exception:
            OBSERVER.failed_probe_count += 1
        return result

    wrapped.__name__ = getattr(original, "__name__", "assemble_mechanics")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    return wrapped


def wrap_near_tip_stress_tensor(
    original: Callable,
    config: AnisotropicEmissionConfig,
) -> Callable:
    """Capture channel drives while preserving the inherited cleavage tensor."""
    cfg = copy.deepcopy(config).validate()

    def wrapped(sigma_gp, mesh, tip_xy, radius):
        baseline = original(sigma_gp, mesh, tip_xy, radius)
        try:
            if OBSERVER.mesh is None:
                OBSERVER.mesh = mesh
                OBSERVER.sigma_gp = np.asarray(sigma_gp, dtype=float).copy()
            if OBSERVER.damage is None:
                raise RuntimeError("damage field has not been captured")
            drive = build_front_drive(
                OBSERVER.mesh,
                OBSERVER.sigma_gp,
                OBSERVER.damage,
                tip_xy,
                cfg,
            )
            OBSERVER.drive_serial += 1
            drive["drive_serial"] = int(OBSERVER.drive_serial)
            OBSERVER.latest_drive = drive
            OBSERVER.reliable_drive_count += 1
        except Exception as exc:
            OBSERVER.failed_probe_count += 1
            OBSERVER.drive_serial += 1
            OBSERVER.latest_drive = {
                "model_id": MODEL_ID,
                "reliable": False,
                "fallback_scalar": True,
                "drive_serial": int(OBSERVER.drive_serial),
                "mechanics_serial": int(OBSERVER.mechanics_serial),
                "tip_xy_m": np.asarray(tip_xy, dtype=float).reshape(2).tolist(),
                "drive_factors": [1.0, 1.0],
                "tau_signed_Pa": [0.0, 0.0],
                "channel_names": ["(110)", "(1-10)"],
                "failure": f"{type(exc).__name__}: {exc}",
            }
            OBSERVER.fallback_drive_count += 1
        return baseline

    wrapped.__name__ = getattr(original, "__name__", "near_tip_stress_tensor")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    return wrapped


def finite_source_emission_update(
    available_sites,
    rates_per_site,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounded source depletion, with no post-hazard directional factor."""
    available = np.maximum(np.asarray(available_sites, dtype=float), 0.0)
    rates = np.maximum(np.asarray(rates_per_site, dtype=float), 0.0)
    if available.shape != rates.shape:
        raise ValueError("available_sites and rates_per_site must have equal shapes")
    probability = 1.0 - np.exp(
        -np.minimum(rates * max(float(dt), 0.0), 700.0)
    )
    emitted = np.minimum(available * probability, available)
    return emitted, probability


def _drive_factors_for_state(state) -> np.ndarray:
    raw = np.asarray(
        getattr(state, "_anisotropic_drive_factors", np.ones(state.n_systems)),
        dtype=float,
    ).reshape(-1)
    if raw.size < state.n_systems:
        raw = np.pad(raw, (0, state.n_systems - raw.size), mode="edge")
    return np.maximum(raw[: state.n_systems], 0.0)


def _anisotropic_campaign_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    """Channel-wise finite-source emission.

    The anisotropic factor enters only through sigma_emit,s before the barrier.
    A nontrivial system_weights argument would apply a second directional factor
    after the hazard and is therefore rejected.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return 0.0
    if system_weights is not None:
        supplied = np.asarray(system_weights, dtype=float)
        if supplied.size and not np.allclose(supplied, 1.0):
            raise RuntimeError(
                "post-hazard system_weights are forbidden in anisotropic emission"
            )

    factors = _drive_factors_for_state(self)
    rho, tau_back, sigma_back = _campaign_backstress(self)
    sigma_opening = max(float(stress_Pa), 0.0)
    sigma_emit = np.maximum(factors * sigma_opening - sigma_back, 0.0)
    rates = np.maximum(
        np.asarray(
            [
                self.emission_rate_per_site(float(value), T_K)
                for value in sigma_emit
            ],
            dtype=float,
        ),
        0.0,
    )

    available0 = np.maximum(np.asarray(self.available_sites, dtype=float), 0.0)
    emitted_system, probability = finite_source_emission_update(
        available0, rates, dt
    )
    self.available_sites = np.maximum(available0 - emitted_system, 0.0)

    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    self.mobile[:, :nsrc] += emitted_system[:, None] / nsrc
    self.accumulated_slip[:, :nsrc] += emitted_system[:, None] / nsrc
    emitted = float(np.sum(emitted_system))
    self.emitted_total += emitted

    reference = np.maximum(np.asarray(self.site_capacity, dtype=float), 0.0)
    activity = np.divide(
        self.available_sites,
        reference,
        out=np.zeros_like(self.available_sites),
        where=reference > 0.0,
    )
    self.tip_source_activity = np.clip(activity, 0.0, 1.0)
    self.continuum_source_last_clear_rate_s = 0.0
    self.continuum_source_last_effective_multiplicity = float(
        np.sum(self.available_sites)
    )
    self.continuum_source_last_emission_rate_s = float(
        np.sum(rates * available0)
    )
    self.continuum_source_last_aggregate_hazard_s = float(
        np.sum(rates * available0)
    )
    self.continuum_source_last_throughput_bound = float(np.sum(available0))
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back))
    self.continuum_source_last_sigma_emit_effective_Pa = float(
        np.mean(sigma_emit)
    )
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(
        np.min(sigma_emit)
    )
    self.campaign_source_budget_remaining_total = float(
        np.sum(self.available_sites)
    )
    self.campaign_source_budget_consumed_total = float(
        np.sum(reference - self.available_sites)
    )

    self.anisotropic_last_drive_factors = factors.copy()
    self.anisotropic_last_sigma_opening_Pa = float(sigma_opening)
    self.anisotropic_last_rho_back_by_system_m2 = rho.copy()
    self.anisotropic_last_tau_back_by_system_Pa = tau_back.copy()
    self.anisotropic_last_sigma_back_by_system_Pa = sigma_back.copy()
    self.anisotropic_last_sigma_emit_by_system_Pa = sigma_emit.copy()
    self.anisotropic_last_lambda_emit_by_system_s = rates.copy()
    self.anisotropic_last_probability_by_system = probability.copy()
    self.anisotropic_last_dN_emit_by_system = emitted_system.copy()
    return emitted


def _anisotropic_evolve(
    self,
    dt_s: float,
    T_K: float,
    stress_Pa: float,
    b: float,
    system_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Channel-resolved emission and active-zone Peierls--Taylor transport."""
    dt = max(float(dt_s), 0.0)
    emitted = self._emit(dt, stress_Pa, T_K, system_weights)
    sigma_system = np.asarray(
        getattr(
            self,
            "anisotropic_last_sigma_emit_by_system_Pa",
            np.full(self.n_systems, max(float(stress_Pa), 0.0)),
        ),
        dtype=float,
    )

    if bool(getattr(self, "_anisotropic_shared_forest_density", True)):
        rho_shared = self.local_forest_density_m2(False)
        rho_by_system = [rho_shared for _ in range(self.n_systems)]
    else:
        width = max(float(self.cfg.blunting_length_m), float(self.dx), 1.0e-12)
        rho_by_system = [
            np.maximum(
                float(self.cfg.forest_density_floor_m2)
                + np.maximum(self.retained[s], 0.0)
                / max(self.dx * width, 1.0e-30),
                1.0,
            )
            for s in range(self.n_systems)
        ]

    trapped_total = 0.0
    released_total = 0.0
    escaped_total = 0.0
    peierls_max = 0.0
    taylor_max = 0.0
    encounter_max = 0.0
    m_max = 1.0
    rates_by_system: list[dict[str, np.ndarray]] = []

    # Exchange first, matching the inherited operator ordering.
    for system in range(self.n_systems):
        profile = self.local_stress_profile_Pa(float(sigma_system[system]))
        rates = self._transport_rates(
            profile, rho_by_system[system], T_K, b
        )
        rates_by_system.append(rates)
        mobile, retained, trapped, released = self._exchange(
            self.mobile[system : system + 1],
            self.retained[system : system + 1],
            rates["encounter"],
            rates["taylor"],
            dt,
        )
        self.mobile[system : system + 1] = mobile
        self.retained[system : system + 1] = retained
        trapped_total += float(trapped)
        released_total += float(released)
        peierls_max = max(peierls_max, float(np.max(rates["peierls"])))
        taylor_max = max(taylor_max, float(np.max(rates["taylor"])))
        encounter_max = max(encounter_max, float(np.max(rates["encounter"])))
        m_max = max(m_max, float(np.max(rates["m"])))

    # Recovery precedes transport, preserving the validated base sequence.
    fr = 1.0 - math.exp(
        -min(max(self.manifest.retained_recovery_rate_s, 0.0) * dt, 700.0)
    )
    fm = 1.0 - math.exp(
        -min(max(self.cfg.mobile_recovery_rate_s, 0.0) * dt, 700.0)
    )
    recovered_retained = self.retained * fr
    recovered_mobile = self.mobile * fm
    self.retained -= recovered_retained
    self.mobile -= recovered_mobile
    recovered = float(
        np.sum(recovered_retained) + np.sum(recovered_mobile)
    )

    velocity_by_system = np.zeros(self.n_systems, dtype=float)
    for system, rates in enumerate(rates_by_system):
        mobile_by_bin = np.maximum(self.mobile[system], 0.0)
        if float(np.sum(mobile_by_bin)) > 0.0:
            velocity = float(
                np.sum(rates["velocity"] * mobile_by_bin)
                / np.sum(mobile_by_bin)
            )
        else:
            nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
            velocity = float(np.mean(rates["velocity"][:nsrc]))
        velocity_by_system[system] = max(velocity, 0.0)
        moved, escaped = self._advect_forward(
            self.mobile[system : system + 1],
            velocity_by_system[system] * dt,
            self.dx,
        )
        self.mobile[system : system + 1] = moved
        escaped_total += float(escaped)

    self.escaped_total += escaped_total
    self.recovered_total += recovered
    self.time_s += dt
    wake = self._evolve_wake(dt, T_K, b)
    self.anisotropic_last_transport_velocity_by_system_m_s = (
        velocity_by_system.copy()
    )
    return {
        "dN_emit": float(emitted),
        "dN_trapped": float(trapped_total),
        "dN_released": float(released_total),
        "dN_recovered": float(recovered),
        "dN_escaped": float(escaped_total),
        "peierls_rate_s": float(peierls_max),
        "taylor_completion_rate_s": float(taylor_max),
        "encounter_rate_s": float(encounter_max),
        "taylor_m_eff": float(m_max),
        "available_site_fraction": float(self.available_site_fraction),
        "anisotropic_transport_active": 1.0,
        **wake,
    }


def install_anisotropic_campaign_emission(
    state,
    config: AnisotropicEmissionConfig,
) -> None:
    cfg = copy.deepcopy(config).validate()
    state._anisotropic_emission_config = cfg
    state._anisotropic_shared_forest_density = cfg.shared_forest_density
    state._anisotropic_drive_factors = np.ones(state.n_systems, dtype=float)
    state._anisotropic_tau_signed_Pa = np.zeros(state.n_systems, dtype=float)
    state._anisotropic_drive_reliable = False
    state._anisotropic_drive_serial = -1
    state.anisotropic_last_drive_factors = np.ones(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_sigma_opening_Pa = 0.0
    state.anisotropic_last_rho_back_by_system_m2 = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_tau_back_by_system_Pa = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_sigma_back_by_system_Pa = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_sigma_emit_by_system_Pa = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_lambda_emit_by_system_s = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_probability_by_system = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_dN_emit_by_system = np.zeros(
        state.n_systems, dtype=float
    )
    state.anisotropic_last_transport_velocity_by_system_m_s = np.zeros(
        state.n_systems, dtype=float
    )
    state._emit = MethodType(_anisotropic_campaign_emit, state)
    state.evolve = MethodType(_anisotropic_evolve, state)


class AnisotropicStochasticAvalancheTipEngine(
    StochasticAvalancheDiagnosticTipEngine
):
    """Stochastic avalanche engine with channel-resolved tensor emission."""

    anisotropic_emission_active = True
    _anisotropic_config_default = AnisotropicEmissionConfig()

    @classmethod
    def configure_anisotropic_emission(
        cls,
        config: AnisotropicEmissionConfig,
    ) -> None:
        cls._anisotropic_config_default = copy.deepcopy(config).validate()

    @classmethod
    def reset_audit(cls) -> None:
        super().reset_audit()
        OBSERVER.clear()

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["anisotropic_emission"] = {
            "model_id": MODEL_ID,
            **asdict(copy.deepcopy(cls._anisotropic_config_default).validate()),
            "two_reduced_slip_trace_channels": True,
            "full_3d_crystal_plasticity": False,
            "directional_factor_location": "inside_emission_stress_before_barrier",
            "post_hazard_directional_weighting": False,
            "opening_stress_unshielded": True,
            "cleavage_shielding_only": True,
            "taylor_backstress_emission_only": True,
            "observer_reliable_drive_count": int(
                OBSERVER.reliable_drive_count
            ),
            "observer_fallback_drive_count": int(
                OBSERVER.fallback_drive_count
            ),
            "observer_failed_probe_count": int(
                OBSERVER.failed_probe_count
            ),
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anisotropic_cfg = copy.deepcopy(
            type(self)._anisotropic_config_default
        ).validate()
        install_anisotropic_campaign_emission(self.mpz, self.anisotropic_cfg)
        self._anisotropic_drive: dict[str, Any] | None = None
        self._anisotropic_drive_serial = -1
        self._anisotropic_fallback_count = 0

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.anisotropic_cfg = copy.deepcopy(self.anisotropic_cfg)
        child._anisotropic_drive = copy.deepcopy(self._anisotropic_drive)
        child._anisotropic_drive_serial = int(self._anisotropic_drive_serial)
        child._anisotropic_fallback_count = int(
            self._anisotropic_fallback_count
        )
        install_anisotropic_campaign_emission(
            child.mpz, child.anisotropic_cfg
        )
        if child._anisotropic_drive is not None:
            child._install_current_drive_on_state()
        return child

    def _adopt_latest_drive(self) -> None:
        drive = OBSERVER.latest_drive
        if drive is None:
            return
        serial = int(drive.get("drive_serial", -1))
        if serial <= self._anisotropic_drive_serial:
            return
        self._anisotropic_drive = copy.deepcopy(drive)
        self._anisotropic_drive_serial = serial
        self._install_current_drive_on_state()

    def _install_current_drive_on_state(self) -> None:
        if self._anisotropic_drive is None:
            factors = np.ones(self.mpz.n_systems, dtype=float)
            tau = np.zeros(self.mpz.n_systems, dtype=float)
            reliable = False
        else:
            factors = np.asarray(
                self._anisotropic_drive.get(
                    "drive_factors", np.ones(self.mpz.n_systems)
                ),
                dtype=float,
            )
            tau = np.asarray(
                self._anisotropic_drive.get(
                    "tau_signed_Pa", np.zeros(self.mpz.n_systems)
                ),
                dtype=float,
            )
            reliable = bool(
                self._anisotropic_drive.get("reliable", False)
            )
        if factors.size < self.mpz.n_systems:
            factors = np.pad(
                factors,
                (0, self.mpz.n_systems - factors.size),
                mode="edge",
            )
        if tau.size < self.mpz.n_systems:
            tau = np.pad(
                tau,
                (0, self.mpz.n_systems - tau.size),
                mode="constant",
            )
        self.mpz._anisotropic_drive_factors = np.maximum(
            factors[: self.mpz.n_systems], 0.0
        )
        self.mpz._anisotropic_tau_signed_Pa = tau[: self.mpz.n_systems]
        self.mpz._anisotropic_drive_reliable = reliable
        self.mpz._anisotropic_drive_serial = int(
            self._anisotropic_drive_serial
        )

    def predict_clock_increment(self, K, T, dt):
        # In the protected driver the tensor probe immediately precedes this
        # per-front predictor, so this binds the correct drive to each engine.
        self._adopt_latest_drive()
        return super().predict_clock_increment(K, T, dt)

    def _plastic_half_step(self, dt: float, T: float, cleavage_stress: float):
        self._adopt_latest_drive()
        if self._anisotropic_drive is None:
            self._anisotropic_fallback_count += 1
        elif (
            self.anisotropic_cfg.require_reliable_probe
            and not bool(self._anisotropic_drive.get("reliable", False))
        ):
            raise RuntimeError(
                "anisotropic emission requires a reliable front tensor probe"
            )
        self._install_current_drive_on_state()
        return super()._plastic_half_step(dt, T, cleavage_stress)

    def _anisotropic_diagnostics(self) -> dict[str, Any]:
        drive = self._anisotropic_drive or {}
        return {
            "anisotropic_emission_model_id": MODEL_ID,
            "anisotropic_emission_enabled": bool(
                self.anisotropic_cfg.enabled
            ),
            "anisotropic_drive_reliable": bool(
                drive.get("reliable", False)
            ),
            "anisotropic_drive_serial": int(
                drive.get("drive_serial", -1)
            ),
            "anisotropic_mechanics_serial": int(
                drive.get("mechanics_serial", -1)
            ),
            "anisotropic_channel_names": list(
                drive.get("channel_names", ["(110)", "(1-10)"])
            ),
            "anisotropic_tau_signed_Pa": [
                float(value)
                for value in np.asarray(
                    getattr(
                        self.mpz,
                        "_anisotropic_tau_signed_Pa",
                        np.zeros(self.mpz.n_systems),
                    )
                )
            ],
            "anisotropic_drive_factors": [
                float(value)
                for value in np.asarray(
                    getattr(
                        self.mpz,
                        "anisotropic_last_drive_factors",
                        np.ones(self.mpz.n_systems),
                    )
                )
            ],
            "anisotropic_sigma_back_by_system_Pa": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_sigma_back_by_system_Pa
                )
            ],
            "anisotropic_sigma_emit_by_system_Pa": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_sigma_emit_by_system_Pa
                )
            ],
            "anisotropic_lambda_emit_by_system_s": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_lambda_emit_by_system_s
                )
            ],
            "anisotropic_emission_probability_by_system": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_probability_by_system
                )
            ],
            "anisotropic_dN_emit_by_system": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_dN_emit_by_system
                )
            ],
            "anisotropic_transport_velocity_by_system_m_s": [
                float(value)
                for value in np.asarray(
                    self.mpz.anisotropic_last_transport_velocity_by_system_m_s
                )
            ],
            "anisotropic_post_hazard_weighting_applied": False,
            "anisotropic_factors_normalized": False,
            "anisotropic_factors_clipped": False,
            "anisotropic_fallback_count": int(
                self._anisotropic_fallback_count
            ),
        }

    def step(self, K, T, dt):
        self._adopt_latest_drive()
        result = super().step(K, T, dt)
        diagnostics = self._anisotropic_diagnostics()
        result.update(diagnostics)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diagnostics)
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        self._adopt_latest_drive()
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(self._anisotropic_diagnostics())
        return result
