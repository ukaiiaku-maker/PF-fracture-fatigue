"""Authoritative production-owned state and kinetics for one explicit 2-D void."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

KB = 1.380649e-23
SCHEMA = "v12.production-voiding/5"


class VoidPhase(str, Enum):
    AVAILABLE_SITE = "AVAILABLE_SITE"
    EMBRYO = "EMBRYO"
    HEALED_SITE = "HEALED_SITE"
    STABLE_SUBGRID_VOID = "STABLE_SUBGRID_VOID"
    RESOLVED_VOID = "RESOLVED_VOID"
    CONNECTED_VOID = "CONNECTED_VOID"
    DOWNSTREAM_FRONT_ACTIVE = "DOWNSTREAM_FRONT_ACTIVE"
    MERGED_OR_CONSUMED = "MERGED_OR_CONSUMED"


@dataclass(frozen=True)
class VoidingConfig:
    enabled: bool = False
    required_birth_hits: int = 2
    attempt_frequency_s: float = 1.0e12
    birth_barrier_J: float = 1.2e-19
    stabilization_barrier_J: float = 0.8e-19
    healing_barrier_J: float = 1.0e-19
    activation_area_m2: float = 2.0e-20
    promotion_radius_m: float = 8.0e-6
    surface_reaction_barrier_J: float = 0.7e-19
    vacancy_transport_barrier_J: float = 0.9e-19
    radial_growth_scale_m: float = 1.0e-8
    hydrostatic_work_coefficient: float = 1.0
    normal_opening_work_coefficient: float = 1.0
    signed_shear_work_coefficient: float = 0.25
    plastic_accommodation_barrier_J: float = 1.0e-19
    shrinkage_mobility_m_per_J_s: float = 1.0e8
    schema: str = SCHEMA


@dataclass(frozen=True)
class HazardClock:
    accumulated: float
    threshold: float

    def crossing_time(self, rate_s: float) -> float:
        return math.inf if rate_s <= 0.0 else max(self.threshold - self.accumulated, 0.0) / rate_s


@dataclass(frozen=True)
class VoidSite:
    site_id: str
    center_m: tuple[float, float]
    phase: VoidPhase
    hits: int
    required_hits: int
    candidate_weight: float
    birth: HazardClock
    stabilization: HazardClock
    healing: HazardClock
    normal_xy: tuple[float, float] = (1.0, 0.0)


@dataclass(frozen=True)
class Cavity2D:
    cavity_id: str
    parent_site_id: str
    center_m: tuple[float, float]
    radius_m: float
    area_m2: float
    inventory_area_m2: float
    phase: VoidPhase
    geometry_generation: int = 0
    lineage: tuple[str, ...] = ()

    def __post_init__(self):
        expected = math.pi * float(self.radius_m) ** 2
        if not math.isclose(float(self.area_m2), expected, rel_tol=1.0e-12, abs_tol=1.0e-24):
            raise ValueError("production cavity area must use the 2-D pi*R^2 convention")


@dataclass(frozen=True)
class ProductionVoidState:
    sites: tuple[VoidSite, ...]
    cavities: tuple[Cavity2D, ...] = ()
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    event_history: tuple[Mapping[str, Any], ...] = ()
    length_ledgers: Mapping[str, float] = field(default_factory=lambda: {
        "fractured_ligament_length_m": 0.0,
        "ordinary_crack_fractured_length_m": 0.0,
        "preexisting_void_free_span_m": 0.0,
        "active_front_coordinate_advance_m": 0.0,
        "projected_fractured_length_m": 0.0,
        "projected_free_span_m": 0.0,
        "projected_front_advance_m": 0.0,
        "connected_free_surface_extent_m": 0.0,
    })
    schema: str = SCHEMA


def arrhenius_rates(config: VoidingConfig, *, temperature_K: float,
                    stress_tensor_Pa: np.ndarray,
                    normal_xy: tuple[float, float] = (1.0, 0.0)) -> dict[str, float]:
    """Compute rates from the local production tensor; no candidate weight here."""
    stress = np.asarray(stress_tensor_Pa, dtype=float).reshape(2, 2)
    eigenvalues = np.linalg.eigvalsh(stress)
    tensile = max(float(eigenvalues[-1]), 0.0)
    hydrostatic = float(np.trace(stress) / 2.0)
    deviator = stress - np.eye(2) * hydrostatic
    von_mises_2d = math.sqrt(max(1.5 * float(np.sum(deviator * deviator)), 0.0))
    normal = np.asarray(normal_xy, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-300)
    tangent = np.array((-normal[1], normal[0]))
    normal_opening = float(normal @ stress @ normal)
    signed_shear = float(tangent @ stress @ normal)
    thermal = KB * float(temperature_K)
    if thermal <= 0.0:
        raise ValueError("temperature must be positive")
    def rate(barrier, work=0.0):
        effective = max(float(barrier) - float(work), 0.0)
        return config.attempt_frequency_s * math.exp(-effective / thermal)
    birth_work = config.activation_area_m2 * (
        config.hydrostatic_work_coefficient * hydrostatic
        + config.normal_opening_work_coefficient * normal_opening
        + config.signed_shear_work_coefficient * signed_shear
    )
    tensile_work = config.activation_area_m2 * tensile
    surface = rate(config.surface_reaction_barrier_J, tensile_work)
    transport = rate(config.vacancy_transport_barrier_J, tensile_work)
    accommodation = rate(config.plastic_accommodation_barrier_J, config.activation_area_m2 * von_mises_2d)
    series = 0.0 if surface <= 0.0 or transport <= 0.0 else 1.0 / (1.0 / surface + 1.0 / transport)
    series = 0.0 if series <= 0.0 or accommodation <= 0.0 else 1.0 / (1.0 / series + 1.0 / accommodation)
    return {
        "birth_s": rate(config.birth_barrier_J, birth_work),
        "stabilization_s": rate(config.stabilization_barrier_J, tensile_work),
        "healing_s": rate(config.healing_barrier_J, -tensile_work),
        "local_max_principal_stress_Pa": tensile,
        "local_hydrostatic_stress_Pa": hydrostatic,
        "local_von_mises_stress_Pa": von_mises_2d,
        "local_normal_opening_stress_Pa": normal_opening,
        "local_signed_shear_stress_Pa": signed_shear,
        "birth_activation_work_J": birth_work,
        "surface_reaction_s": surface,
        "vacancy_transport_s": transport,
        "plastic_accommodation_s": accommodation,
        "series_limited_growth_s": series,
    }


def advance_site(state: ProductionVoidState, site_id: str, dt_s: float, *,
                 rates: Mapping[str, float]) -> tuple[ProductionVoidState, tuple[str, ...]]:
    """Localize all first passages and retain threshold/RNG ownership in state."""
    site = next(item for item in state.sites if item.site_id == site_id)
    remaining = float(dt_s)
    events = []
    current = site
    while remaining > max(1.0e-15 * dt_s, 1.0e-18):
        if current.phase == VoidPhase.AVAILABLE_SITE:
            birth_rate = max(float(rates["birth_s"]), 0.0) * current.candidate_weight
            crossing = current.birth.crossing_time(birth_rate)
            step = min(remaining, crossing)
            clock = replace(current.birth, accumulated=min(
                current.birth.threshold, current.birth.accumulated + birth_rate * step,
            ))
            current = replace(current, birth=clock)
            remaining -= step
            if crossing > step or not math.isfinite(crossing):
                break
            hits = current.hits + 1
            if hits < current.required_hits:
                rng = np.random.default_rng()
                rng.bit_generator.state = dict(state.rng_state)
                renewed = float(rng.exponential())
                state = replace(state, rng_state=rng.bit_generator.state)
                current = replace(current, hits=hits, birth=HazardClock(0.0, renewed))
                events.append("BIRTH_HIT")
                continue
            current = replace(current, hits=hits, phase=VoidPhase.EMBRYO)
            events.append("EMBRYO")
            continue
        if current.phase == VoidPhase.EMBRYO:
            ts = current.stabilization.crossing_time(float(rates["stabilization_s"]))
            th = current.healing.crossing_time(float(rates["healing_s"]))
            crossing = min(ts, th)
            step = min(remaining, crossing)
            current = replace(
                current,
                stabilization=replace(current.stabilization, accumulated=min(
                    current.stabilization.threshold,
                    current.stabilization.accumulated + float(rates["stabilization_s"]) * step,
                )),
                healing=replace(current.healing, accumulated=min(
                    current.healing.threshold,
                    current.healing.accumulated + float(rates["healing_s"]) * step,
                )),
            )
            remaining -= step
            if crossing > step or not math.isfinite(crossing):
                break
            phase = VoidPhase.HEALED_SITE if th <= ts else VoidPhase.STABLE_SUBGRID_VOID
            current = replace(current, phase=phase)
            events.append("HEALED" if phase == VoidPhase.HEALED_SITE else "STABILIZED")
            break
        break
    sites = tuple(current if item.site_id == site_id else item for item in state.sites)
    history = state.event_history + tuple({"event": event, "site_id": site_id} for event in events)
    return replace(state, sites=sites, event_history=history), tuple(events)


def create_subgrid_cavity(state: ProductionVoidState, site_id: str,
                          radius_m: float) -> ProductionVoidState:
    site = next(item for item in state.sites if item.site_id == site_id)
    if site.phase != VoidPhase.STABLE_SUBGRID_VOID:
        raise ValueError("site has not stabilized")
    if any(item.parent_site_id == site_id for item in state.cavities):
        raise ValueError("a stabilized site may own only one cavity")
    area = math.pi * float(radius_m) ** 2
    cavity = Cavity2D("void:" + site_id, site_id, site.center_m, radius_m, area, area,
                      VoidPhase.STABLE_SUBGRID_VOID, lineage=(site_id,))
    return replace(state, cavities=state.cavities + (cavity,))


def grow_cavity_2d(cavity: Cavity2D, delta_radius_m: float) -> Cavity2D:
    radius = float(cavity.radius_m) + float(delta_radius_m)
    if radius <= 0.0:
        raise ValueError("growth consumes the cavity")
    area = math.pi * radius ** 2
    inventory_increment = area - cavity.area_m2
    return replace(
        cavity, radius_m=radius, area_m2=area,
        inventory_area_m2=cavity.inventory_area_m2 + inventory_increment,
        geometry_generation=cavity.geometry_generation + 1,
    )


def grow_cavity_from_rate(cavity: Cavity2D, *, rates: Mapping[str, float],
                          dt_s: float, radial_growth_scale_m: float,
                          chemical_potential_drive_J: float = 1.0) -> Cavity2D:
    """Advance radius only through the measured series-limited kinetic rate."""
    drive = float(chemical_potential_drive_J)
    rate = max(float(rates["series_limited_growth_s"]), 0.0)
    dt = max(float(dt_s), 0.0)
    if drive > 0.0:
        delta = float(radial_growth_scale_m) * rate * dt
    elif drive < 0.0:
        delta = -min(cavity.radius_m * 0.5, abs(drive) * dt * 1.0e8)
    else:
        delta = 0.0
    return grow_cavity_2d(cavity, delta) if delta != 0.0 else cavity


def replace_cavity(state: ProductionVoidState, cavity: Cavity2D) -> ProductionVoidState:
    return replace(state, cavities=tuple(
        cavity if item.cavity_id == cavity.cavity_id else item for item in state.cavities
    ))


def promote_cavity(state: ProductionVoidState, cavity_id: str,
                   minimum_radius_m: float) -> ProductionVoidState:
    cavity = next(item for item in state.cavities if item.cavity_id == cavity_id)
    if cavity.phase != VoidPhase.STABLE_SUBGRID_VOID or cavity.radius_m < minimum_radius_m:
        raise ValueError("cavity is not eligible for geometric promotion")
    updated = replace(
        cavity, phase=VoidPhase.RESOLVED_VOID,
        geometry_generation=cavity.geometry_generation + 1,
        lineage=cavity.lineage + ("GEOMETRIC_PROMOTION",),
    )
    return replace(
        replace_cavity(state, updated),
        event_history=state.event_history + ({"event": "GEOMETRIC_PROMOTION", "cavity_id": cavity_id},),
    )


def fingerprint(state: ProductionVoidState | None) -> str:
    return hashlib.sha256(serialize(state).encode()).hexdigest()


def serialize(state: ProductionVoidState | None) -> str:
    def encode(value):
        if isinstance(value, Enum): return value.value
        if hasattr(value, "__dataclass_fields__"):
            return {name: encode(getattr(value, name)) for name in value.__dataclass_fields__}
        if isinstance(value, Mapping): return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [encode(v) for v in value]
        if isinstance(value, np.generic): return value.item()
        return value
    return json.dumps(encode(state), sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "Cavity2D", "HazardClock", "ProductionVoidState", "SCHEMA", "VoidPhase",
    "VoidSite", "VoidingConfig", "advance_site", "arrhenius_rates",
    "create_subgrid_cavity", "fingerprint", "grow_cavity_2d", "grow_cavity_from_rate",
    "promote_cavity", "replace_cavity", "serialize",
]
