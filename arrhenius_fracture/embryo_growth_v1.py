"""State-owned embryo-to-resolved-cavity growth by vacancies and loop emission.

The material values in this module are an immutable reference row, not a
calibration.  Vacancy supply is series-limited by bulk accommodation; emitted
loop area is an independent kinematic contribution and is never sent through
the vacancy bottleneck.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .canonical_kinetic_time_v1 import exact, packed, unpacked
from .vacancy_coupling_v1 import conservation_roundoff_budget_m2
from .vacancy_transport_phase_field_v2 import KB_J_PER_K, REFERENCE_UNCALIBRATED


SCHEMA = "unified-2d.embryo-growth/1"
SUBGRID = "ANALYTICAL_SUBGRID_CAVITY"
RESOLVED = "ACCEPTED_FEM_CAVITY_SURFACE"
RESIDUAL = "RESIDUAL_LOOP_BEARING_DEFECT"


class FrozenGrowthMapping(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("accepted embryo-growth mapping is immutable")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    def __reduce__(self):
        return (_frozen, (dict(self),))


def _frozen(values: Mapping[str, Any]) -> FrozenGrowthMapping:
    result = dict.__new__(FrozenGrowthMapping)
    dict.update(result, values)
    return result


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _canonical_float(value: float) -> float:
    """Remove nonphysical last-bit differences from composable flow maps."""
    return float(f"{float(value):.12g}")


@dataclass(frozen=True)
class CavityEmissionParameterRowV1:
    row_id: str = "v1.cavity-emission-reference-uncalibrated"
    attempt_frequency_s: float = 1.0e11
    zero_stress_barrier_J: float = 2.6e-19
    barrier_kind: str = "exp_floor"
    barrier_critical_stress_Pa: float = 3.0e9
    barrier_exp_a: float = 1.0
    barrier_exp_n: float = 1.0
    barrier_floor_fraction: float = 0.05
    barrier_floor_min_J: float = 1.0e-23
    surface_source_density_per_m: float = 2.0e8
    source_orientations_rad: tuple[float, ...] = (0.0, math.pi/3.0, 2.0*math.pi/3.0)
    burgers_m: float = 2.74e-10
    model_thickness_m: float = 1.0
    loop_swept_area_m2: float = 6.283185307179587e-9
    chi_loop: float = 0.25
    loop_projection: str = "REDUCED_CYLINDRICAL_2D"
    surface_energy_J_m2: float = 2.5
    minimum_stable_radius_m: float = 5.0e-10
    retained_loop_fraction: float = 0.8
    escaped_loop_fraction: float = 0.2
    depletion_per_loop: float = 0.02
    recovery_rate_s: float = 1.0e-3
    low_count_mu_max: float = 0.25
    continuum_mu_min: float = 25.0
    maximum_packet_area_fraction: float = 0.02
    calibration_status: str = REFERENCE_UNCALIBRATED
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        positive = (
            self.attempt_frequency_s, self.zero_stress_barrier_J,
            self.surface_source_density_per_m,
            self.barrier_critical_stress_Pa, self.barrier_exp_a,
            self.barrier_exp_n, self.barrier_floor_fraction,
            self.barrier_floor_min_J,
            self.burgers_m, self.model_thickness_m, self.loop_swept_area_m2,
            self.chi_loop, self.surface_energy_J_m2,
            self.minimum_stable_radius_m,
            self.low_count_mu_max, self.continuum_mu_min,
            self.maximum_packet_area_fraction,
        )
        if any(not math.isfinite(v) or v <= 0.0 for v in positive):
            raise ValueError("cavity-emission row requires finite positive values")
        if self.calibration_status != REFERENCE_UNCALIBRATED:
            raise ValueError("cavity-emission row must remain REFERENCE_UNCALIBRATED")
        if self.loop_projection not in {"FULL_3D_LOOP", "REDUCED_CYLINDRICAL_2D"}:
            raise ValueError("unsupported cavity-emission loop projection")
        if self.barrier_kind != "exp_floor":
            raise ValueError("cavity emission must use the qualified EXP-floor barrier class")
        if not 0.0 < self.barrier_floor_fraction < 1.0:
            raise ValueError("cavity-emission EXP floor fraction must lie in (0,1)")
        if not math.isclose(
            self.retained_loop_fraction+self.escaped_loop_fraction, 1.0,
            rel_tol=0.0, abs_tol=1.0e-12,
        ):
            raise ValueError("retained and escaped loop fractions must close")
        if not 0.0 < self.low_count_mu_max < self.continuum_mu_min:
            raise ValueError("hybrid count limits are not ordered")

    @property
    def fingerprint(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True)
class CavityEmissionSourceState:
    source_id: str
    orientation_rad: float
    available_fraction: float = 1.0
    integrated_action: float = 0.0
    next_exact_threshold: float = 1.0
    emitted_loop_count_effective: float = 0.0
    realized_loop_count: int = 0
    continuum_moment_loop_count: float = 0.0
    random_measure_seed: int = 1
    random_measure_cursor_count: int = 0
    boundary_position_m: tuple[float, float] | None = None
    local_normal: tuple[float, float] | None = None
    local_tangent: tuple[float, float] | None = None
    boundary_element_id: int | None = None
    integrated_action_exact: tuple[int, int] | None = None
    availability_reference_fraction: float = 1.0
    availability_reference_rate_s: float | None = None
    availability_elapsed_time_exact: tuple[int, int] = (0, 1)
    availability_reference_action_exact: tuple[int, int] = (0, 1)
    availability_flow_regime: str | None = None
    regime_epoch: int = 0
    regime_epoch_start_action: float = 0.0
    regime_epoch_start_moment_loop_count: float = 0.0

    def __post_init__(self) -> None:
        # This total is redundant accepted state.  Canonicalize it from the
        # two owning ledgers before validating so independent floating-point
        # rounding can never make an otherwise physical transition invalid.
        effective = float(self.realized_loop_count)+float(
            self.continuum_moment_loop_count)
        object.__setattr__(self, "emitted_loop_count_effective", effective)
        values = (self.available_fraction, self.integrated_action,
                  self.next_exact_threshold, self.emitted_loop_count_effective,
                  self.continuum_moment_loop_count,
                  self.regime_epoch_start_moment_loop_count)
        if any(not math.isfinite(v) for v in values):
            raise ValueError("cavity-emission source state must be finite")
        if not 0.0 <= self.available_fraction <= 1.0:
            raise ValueError("source availability must lie in [0,1]")
        if self.integrated_action < 0.0 or self.next_exact_threshold <= 0.0:
            raise ValueError("cavity-emission action state is invalid")
        if (self.realized_loop_count < 0 or self.random_measure_cursor_count < 0
                or self.random_measure_seed <= 0 or self.regime_epoch < 0):
            raise ValueError("random-measure state is invalid")
        if not math.isclose(
            self.emitted_loop_count_effective,
            self.realized_loop_count+self.continuum_moment_loop_count,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError("realized and continuum-moment loop counts do not close")
        if self.boundary_element_id is not None and self.boundary_element_id < 0:
            raise ValueError("source boundary element id must be nonnegative")
        for vector in (self.boundary_position_m, self.local_normal, self.local_tangent):
            if vector is not None and (len(vector) != 2 or
                    any(not math.isfinite(float(v)) for v in vector)):
                raise ValueError("source boundary frame must contain finite 2-D vectors")
        if (not 0.0 <= self.availability_reference_fraction <= 1.0
                or (self.availability_reference_rate_s is not None
                    and (not math.isfinite(self.availability_reference_rate_s)
                         or self.availability_reference_rate_s < 0.0))):
            raise ValueError("availability flow-map reference is invalid")

    def action_exact(self):
        return exact(self.integrated_action) if self.integrated_action_exact is None \
            else unpacked(self.integrated_action_exact)


@dataclass(frozen=True)
class CavityEmissionState:
    parameter_row: CavityEmissionParameterRowV1
    sources: tuple[CavityEmissionSourceState, ...]
    rng_state: Mapping[str, Any]
    enabled: bool = False
    cumulative_loop_count_effective: float = 0.0
    cumulative_emitted_burgers_m: float = 0.0
    cumulative_loop_swept_area_m2: float = 0.0
    cumulative_area_m2: float = 0.0
    cumulative_area_variance_m4: float = 0.0
    last_audit: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    cumulative_emitted_burgers_exact: tuple[int, int] | None = None
    cumulative_loop_swept_area_exact: tuple[int, int] | None = None
    cumulative_area_exact: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # Aggregate count ownership belongs to the accepted source states.
        # Never advance or round this redundant total independently.
        object.__setattr__(
            self, "cumulative_loop_count_effective",
            math.fsum(source.emitted_loop_count_effective
                      for source in self.sources))
        if self.cumulative_emitted_burgers_exact is None:
            object.__setattr__(self, "cumulative_emitted_burgers_exact",
                               packed(exact(self.cumulative_emitted_burgers_m)))
        if self.cumulative_loop_swept_area_exact is None:
            object.__setattr__(self, "cumulative_loop_swept_area_exact",
                               packed(exact(self.cumulative_loop_swept_area_m2)))
        if self.cumulative_area_exact is None:
            object.__setattr__(self, "cumulative_area_exact",
                               packed(exact(self.cumulative_area_m2)))
        object.__setattr__(self, "last_audit", _frozen(dict(self.last_audit)))

    @property
    def fingerprint(self) -> str:
        def encode(x: Any):
            if hasattr(x, "__dataclass_fields__"):
                encoded = {}
                for k in x.__dataclass_fields__:
                    value = getattr(x, k)
                    if (k == "availability_elapsed_time_exact"
                            and value is not None):
                        value = float(unpacked(value))
                    encoded[k] = encode(value)
                return encoded
            if isinstance(x, Mapping):
                return {str(k): encode(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)): return [encode(v) for v in x]
            if isinstance(x, np.generic): return x.item()
            if isinstance(x, float): return float(f"{x:.10g}")
            return x
        # ``last_audit`` is an observation of the most recent caller-selected
        # partition, not accepted physical state.  Including it makes one
        # interval and two half intervals fingerprint differently even when
        # the canonical compensator, random measure, population and all
        # physical ledgers are identical.
        physical = {k: getattr(self, k) for k in self.__dataclass_fields__
                    if k not in {"last_audit", "cumulative_area_variance_m4"}}
        return _sha(encode(physical))


@dataclass(frozen=True)
class EmbryoGrowthConfiguration:
    vacancy_clustering_birth_enabled: bool = True
    direct_interaction_birth_enabled: bool = True
    cavity_surface_loop_emission_enabled: bool = False
    cavity_emission_parameter_row: Any = None


@dataclass(frozen=True)
class TransportFluxCertificate:
    """State-bound proof that accepted transport supplied the vacancies."""
    owner_id: str
    owner_kind: str
    transport_before_fingerprint: str
    transport_after_fingerprint: str
    surface_registry_fingerprint: str
    capture_ledger_before_area_m2: float
    capture_ledger_after_area_m2: float
    raw_absorbed_area_m2: float
    field_inventory_before_area_m2: float
    field_inventory_after_area_m2: float
    field_inventory_decrement_area_m2: float
    surface_flux_integral_area_m2: float
    owner_inventory_increment_area_m2: float
    plastic_and_climb_source_area_m2: float
    other_sink_integrals_area_m2: float
    conservation_residual_area_m2: float
    duration_s: float

    def __post_init__(self) -> None:
        if self.owner_kind not in {"SUBGRID_SITE", "ACTIVE_CAVITY"}:
            raise ValueError("transport certificate owner kind is invalid")
        vals = (self.capture_ledger_before_area_m2,
                self.capture_ledger_after_area_m2,
                self.raw_absorbed_area_m2, self.field_inventory_before_area_m2,
                self.field_inventory_after_area_m2,
                self.field_inventory_decrement_area_m2,
                self.surface_flux_integral_area_m2, self.owner_inventory_increment_area_m2,
                self.plastic_and_climb_source_area_m2, self.other_sink_integrals_area_m2,
                self.conservation_residual_area_m2,
                self.duration_s)
        if any(not math.isfinite(float(v)) for v in vals) or self.duration_s <= 0.0:
            raise ValueError(
                "transport certificate values must be finite and duration positive: "
                +repr(vals))
        # ``raw_absorbed`` is a subtraction of long-lived cumulative ledgers.
        # Its admissible loss of significance is therefore determined by the
        # operands, not by a blanket relative tolerance on the small result.
        subtraction_budget = conservation_roundoff_budget_m2(
            self.capture_ledger_before_area_m2,
            self.capture_ledger_after_area_m2)
        direct_budget = conservation_roundoff_budget_m2(
            self.raw_absorbed_area_m2, self.surface_flux_integral_area_m2,
            self.owner_inventory_increment_area_m2)
        tol = subtraction_budget+direct_budget
        observed_difference = (self.capture_ledger_after_area_m2-
                               self.capture_ledger_before_area_m2)
        if abs(observed_difference-self.raw_absorbed_area_m2) > subtraction_budget:
            raise ValueError("capture ledger operands do not produce the certified increment")
        if self.raw_absorbed_area_m2 < -tol:
            raise ValueError("one-way transport certificate cannot contain negative capture")
        if not math.isclose(
                self.field_inventory_before_area_m2-self.field_inventory_after_area_m2,
                self.field_inventory_decrement_area_m2,
                rel_tol=1e-11, abs_tol=1e-30):
            raise ValueError("transport field inventory endpoints do not close")
        if abs(self.raw_absorbed_area_m2-self.surface_flux_integral_area_m2) > tol:
            raise ValueError(
                "transport surface flux does not close with captured inventory: "
                f"ledger={self.raw_absorbed_area_m2:.17e}, "
                f"flux={self.surface_flux_integral_area_m2:.17e}, tol={tol:.3e}")
        if abs(self.raw_absorbed_area_m2-self.owner_inventory_increment_area_m2) > tol:
            raise ValueError("owner inventory increment does not close with its flux")
        assembled = (self.field_inventory_decrement_area_m2
                     +self.plastic_and_climb_source_area_m2
                     -self.other_sink_integrals_area_m2
                     -self.raw_absorbed_area_m2)
        if abs(assembled-self.conservation_residual_area_m2) > max(tol, 1e-24):
            raise ValueError("independently assembled transport balance does not close")

    @property
    def fingerprint(self) -> str:
        return _sha(asdict(self))

    def verify(self, before: Any, after: Any) -> None:
        try:
            observed = transport_flux_certificate(
                before, after, owner_id=self.owner_id, owner_kind=self.owner_kind)
        except (KeyError, StopIteration, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("forged or stale transport flux certificate") from exc
        if self != observed:
            raise RuntimeError("forged or stale transport flux certificate")


def transport_flux_certificate(before: Any, after: Any, *, owner_id: str,
                               owner_kind: str) -> TransportFluxCertificate:
    """Construct a certificate solely from two accepted V2 transport states."""
    from .vacancy_transport_phase_field_v2 import VacancyTransportStateV2
    if not isinstance(before, VacancyTransportStateV2) or not isinstance(
            after, VacancyTransportStateV2):
        raise TypeError("scientific transport certificates require actual V2 states")
    if before.surface_registry_fingerprint != after.surface_registry_fingerprint:
        raise RuntimeError("surface registry changed across a transport-only growth interval")
    dt = float(after.transport_time_s-before.transport_time_s)
    if owner_kind == "ACTIVE_CAVITY":
        capture_before = float(before.ledgers["active_cavity_absorption_area_m2"])
        capture_after = float(after.ledgers["active_cavity_absorption_area_m2"])
        raw = float(capture_after-capture_before)
    elif owner_kind == "SUBGRID_SITE":
        b = next(x for x in before.site_states if x.site_id == owner_id)
        a = next(x for x in after.site_states if x.site_id == owner_id)
        capture_before = float(b.reserved_inventory_area_m2)
        capture_after = float(a.reserved_inventory_area_m2)
        raw = float(capture_after-capture_before)
    else:
        raise ValueError("unknown transport certificate owner kind")
    audit_dt = float(after.last_solver_audit.get("duration_s", dt))
    time_roundoff = 64*np.spacing(max(
        abs(float(after.transport_time_s)),
        abs(float(before.transport_time_s)), abs(dt), np.finfo(float).tiny))
    if not math.isclose(audit_dt, dt, rel_tol=2e-12,
                        abs_tol=max(np.finfo(float).tiny, time_roundoff)):
        raise RuntimeError(
            "transport audit does not describe the supplied state interval: "
            f"state_dt={dt!r} audit_dt="
            f"{after.last_solver_audit.get('duration_s')!r}")
    ratio = after.parameter_row.inventory_ratio
    field_before = ratio*math.fsum(float(a*c) for a, c in zip(
        before.transport_mesh.area_e, before.vacancy_fraction_gp))
    field_after = ratio*math.fsum(float(a*c) for a, c in zip(
        after.transport_mesh.area_e, after.vacancy_fraction_gp))
    field_decrement = field_before-field_after
    delta = {key: float(after.ledgers[key]-before.ledgers[key]) for key in after.ledgers}
    sources = delta["plastic_production_area_m2"]+delta["dislocation_climb_emission_area_m2"]
    total_sinks = math.fsum((
        delta["dislocation_climb_annihilation_area_m2"],
        delta["crack_surface_absorption_area_m2"],
        delta["external_surface_absorption_area_m2"],
        delta["active_cavity_absorption_area_m2"],
        delta["subgrid_capture_area_m2"],
    ))
    raw = max(raw, 0.0)
    if owner_kind == "ACTIVE_CAVITY":
        owner_inventory = delta["active_cavity_inventory_area_m2"]
        owner_flux = float(after.last_solver_audit[
            "active_cavity_absorption_increment_area_m2"])
    else:
        owner_inventory = raw
        index = next(i for i, x in enumerate(after.site_states) if x.site_id == owner_id)
        owner_flux = float(after.last_solver_audit["site_capture_fraction_area"][index])*ratio
    other_sinks = total_sinks-owner_flux
    assembled_residual = field_decrement+sources-other_sinks-owner_flux
    return TransportFluxCertificate(
        owner_id=str(owner_id), owner_kind=owner_kind,
        transport_before_fingerprint=before.fingerprint,
        transport_after_fingerprint=after.fingerprint,
        surface_registry_fingerprint=after.surface_registry_fingerprint,
        capture_ledger_before_area_m2=capture_before,
        capture_ledger_after_area_m2=capture_after,
        raw_absorbed_area_m2=raw,
        field_inventory_before_area_m2=field_before,
        field_inventory_after_area_m2=field_after,
        field_inventory_decrement_area_m2=field_decrement,
        surface_flux_integral_area_m2=owner_flux,
        owner_inventory_increment_area_m2=owner_inventory,
        plastic_and_climb_source_area_m2=sources,
        other_sink_integrals_area_m2=other_sinks,
        conservation_residual_area_m2=assembled_residual, duration_s=dt,
    )


@dataclass(frozen=True)
class EmbryoGrowthState:
    site_id: str
    radius_m: float
    area_m2: float
    initial_area_m2: float
    vacancy_derived_area_m2: float
    dislocation_derived_area_m2: float
    reversible_area_m2: float
    vacancy_absorbed_count_equivalent: float
    emitted_loop_count_effective: float
    cumulative_emitted_burgers_m: float
    cumulative_loop_swept_area_m2: float
    cavity_emission_state: CavityEmissionState
    vacancy_capture_state: Mapping[str, Any]
    stabilization_clock: Any
    healing_clock: Any
    representation: str
    lineage: tuple[str, ...]
    cavity_id: str | None = None
    deposited_plastic_work_J_per_m: float = 0.0
    deposited_dislocation_density_integral: float = 0.0
    captured_unaccommodated_vacancy_area_m2: float = 0.0
    vacancy_returned_or_annihilated_area_m2: float = 0.0
    vacancy_raw_captured_area_m2: float = 0.0
    direct_interaction_birth_hits: int = 0
    escaped_loop_count_effective: float = 0.0
    retained_loop_count_effective: float = 0.0
    stabilization_action_complete_waiting_for_radius: bool = False
    last_remesh_radius_m: float | None = None

    def __post_init__(self) -> None:
        scalars = (
            self.radius_m, self.area_m2, self.initial_area_m2,
            self.vacancy_derived_area_m2, self.dislocation_derived_area_m2,
            self.reversible_area_m2, self.vacancy_absorbed_count_equivalent,
            self.emitted_loop_count_effective, self.cumulative_emitted_burgers_m,
            self.cumulative_loop_swept_area_m2, self.deposited_plastic_work_J_per_m,
            self.deposited_dislocation_density_integral,
            self.captured_unaccommodated_vacancy_area_m2,
            self.vacancy_returned_or_annihilated_area_m2,
            self.vacancy_raw_captured_area_m2,
            self.escaped_loop_count_effective,
            self.retained_loop_count_effective,
        )
        if any(not math.isfinite(v) or v < 0.0 for v in scalars):
            raise ValueError("embryo growth inventories must be finite and nonnegative")
        last_remesh = (self.radius_m if self.last_remesh_radius_m is None
                       else float(self.last_remesh_radius_m))
        if (not math.isfinite(last_remesh) or last_remesh < 0.0
                or last_remesh > self.radius_m):
            raise ValueError(
                "last remesh radius must be finite and no larger than current radius")
        object.__setattr__(self, "last_remesh_radius_m", last_remesh)
        expected = (self.initial_area_m2+self.vacancy_derived_area_m2
                    +self.dislocation_derived_area_m2-self.reversible_area_m2)
        tol = max(1.0e-30, 256.0*np.finfo(float).eps*max(expected, self.area_m2, 1.0e-30))
        if abs(self.area_m2-expected) > tol:
            raise ValueError(
                "mechanism-resolved embryo area ledger does not close: "
                f"area={self.area_m2:.17e} expected={expected:.17e} tol={tol:.3e}")
        if self.direct_interaction_birth_hits < 0:
            raise ValueError("direct-interaction birth hits must be nonnegative")
        vacancy_sum = (self.vacancy_derived_area_m2
                       +self.captured_unaccommodated_vacancy_area_m2
                       +self.vacancy_returned_or_annihilated_area_m2)
        if abs(self.vacancy_raw_captured_area_m2-vacancy_sum) > max(
            1e-30, 64*np.finfo(float).eps*max(vacancy_sum, 1e-30)):
            raise ValueError("raw vacancy capture ledger does not close")
        if self.area_m2 > 0.0 and not math.isclose(
            self.area_m2, math.pi*self.radius_m**2, rel_tol=2.0e-12, abs_tol=tol,
        ):
            raise ValueError("embryo area and physical radius disagree")
        object.__setattr__(self, "vacancy_capture_state",
                           _frozen(dict(self.vacancy_capture_state)))

    @property
    def fingerprint(self) -> str:
        return _sha({
            "site_id": self.site_id, "radius_m": self.radius_m,
            "area_m2": self.area_m2, "initial_area_m2": self.initial_area_m2,
            "vacancy_derived_area_m2": self.vacancy_derived_area_m2,
            "dislocation_derived_area_m2": self.dislocation_derived_area_m2,
            "reversible_area_m2": self.reversible_area_m2,
            "vacancy_absorbed_count_equivalent": self.vacancy_absorbed_count_equivalent,
            "emitted_loop_count_effective": self.emitted_loop_count_effective,
            "cumulative_emitted_burgers_m": self.cumulative_emitted_burgers_m,
            "cumulative_loop_swept_area_m2": self.cumulative_loop_swept_area_m2,
            "emission": self.cavity_emission_state.fingerprint,
            "vacancy_capture_state": dict(self.vacancy_capture_state),
            "representation": self.representation, "lineage": self.lineage,
            "cavity_id": self.cavity_id,
            "deposited_plastic_work_J_per_m": self.deposited_plastic_work_J_per_m,
            "deposited_dislocation_density_integral": self.deposited_dislocation_density_integral,
            "captured_unaccommodated_vacancy_area_m2": self.captured_unaccommodated_vacancy_area_m2,
            "vacancy_returned_or_annihilated_area_m2": self.vacancy_returned_or_annihilated_area_m2,
            "vacancy_raw_captured_area_m2": self.vacancy_raw_captured_area_m2,
            "direct_interaction_birth_hits": self.direct_interaction_birth_hits,
            "escaped_loop_count_effective": self.escaped_loop_count_effective,
            "retained_loop_count_effective": self.retained_loop_count_effective,
            "stabilization_action_complete_waiting_for_radius": self.stabilization_action_complete_waiting_for_radius,
            "last_remesh_radius_m": self.last_remesh_radius_m,
        })


_POISSON_MEASURE_ROOT_POWER = 40
_POISSON_MEASURE_DEPTH = 92


def _counter_rng(seed: int, level: int, index: int, purpose: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"cavity-poisson-v1:{seed}:{level}:{index}:{purpose}".encode()).digest()
    return np.random.Generator(np.random.Philox(int.from_bytes(digest[:8], "little")))


def _root_measure_count(seed: int) -> int:
    return int(_counter_rng(seed, 0, 0, "root").poisson(2.0**_POISSON_MEASURE_ROOT_POWER))


def _split_measure_count(seed: int, level: int, index: int, parent_count: int) -> int:
    return int(_counter_rng(seed, level, index, "split").binomial(parent_count, 0.5))


def poisson_measure_count(seed: int, action: float) -> int:
    """Canonical unit-rate Poisson count on [0, action], in O(log range)."""
    x = float(action)
    root_hi = 2.0**_POISSON_MEASURE_ROOT_POWER
    if not 0.0 <= x <= root_hi:
        raise RuntimeError("cavity Poisson compensator exceeds the qualified counter-measure domain")
    count = _root_measure_count(seed)
    lo, hi, level, index, prefix = 0.0, root_hi, 0, 0, 0
    for _ in range(_POISSON_MEASURE_DEPTH):
        if x <= lo: return prefix
        if x >= hi: return prefix+count
        left = _split_measure_count(seed, level+1, 2*index, count)
        mid = 0.5*(lo+hi)
        if x < mid:
            count, hi, level, index = left, mid, level+1, 2*index
        else:
            prefix += left
            count, lo, level, index = count-left, mid, level+1, 2*index+1
    return prefix + (count if x >= 0.5*(lo+hi) else 0)


def poisson_measure_next_action(seed: int, action: float) -> float:
    """Locate the next canonical jump after action without enumerating earlier jumps."""
    rank = poisson_measure_count(seed, action)+1
    root_hi = 2.0**_POISSON_MEASURE_ROOT_POWER
    count = _root_measure_count(seed)
    if rank > count:
        raise RuntimeError("cavity Poisson measure root exhausted")
    lo, hi, level, index = 0.0, root_hi, 0, 0
    for _ in range(_POISSON_MEASURE_DEPTH):
        left = _split_measure_count(seed, level+1, 2*index, count)
        mid = 0.5*(lo+hi)
        if rank <= left:
            count, hi, level, index = left, mid, level+1, 2*index
        else:
            rank -= left
            count, lo, level, index = count-left, mid, level+1, 2*index+1
    return 0.5*(lo+hi)


def initialize_cavity_emission_state(
    *, row: CavityEmissionParameterRowV1 | None = None, seed: int = 84731,
    enabled: bool = False,
) -> CavityEmissionState:
    row = row or CavityEmissionParameterRowV1()
    rng = np.random.default_rng(int(seed))
    sources = []
    for i, theta in enumerate(row.source_orientations_rad):
        measure_seed = int(rng.integers(1, np.iinfo(np.int64).max))
        sources.append(CavityEmissionSourceState(
            source_id=f"cavity-source-{i}", orientation_rad=float(theta),
            random_measure_seed=measure_seed,
            next_exact_threshold=poisson_measure_next_action(measure_seed, 0.0)))
    sources = tuple(sources)
    return CavityEmissionState(row, sources, rng.bit_generator.state, enabled=enabled)


def initialize_embryo_growth_state(
    *, site_id: str, radius_m: float, initial_area_m2: float = 0.0,
    vacancy_derived_area_m2: float = 0.0,
    dislocation_derived_area_m2: float = 0.0, seed: int = 84731,
    stabilization_clock: Any = None, healing_clock: Any = None,
    representation: str = SUBGRID, cavity_id: str | None = None,
    row: CavityEmissionParameterRowV1 | None = None,
    emitted_loop_count_effective: float = 0.0,
    emission_enabled: bool = False,
    direct_interaction_birth_hits: int = 0,
) -> EmbryoGrowthState:
    area = initial_area_m2+vacancy_derived_area_m2+dislocation_derived_area_m2
    radius = math.sqrt(area/math.pi) if area > 0.0 else float(radius_m)
    if area == 0.0:
        area = math.pi*radius**2
        initial_area_m2 = area
    emission = initialize_cavity_emission_state(row=row, seed=seed, enabled=emission_enabled)
    loop_count = float(emitted_loop_count_effective)
    if loop_count > 0.0:
        emission_row = emission.parameter_row
        sources = list(emission.sources)
        sources[0] = replace(
            sources[0], emitted_loop_count_effective=loop_count,
            continuum_moment_loop_count=loop_count)
        emission = replace(
            emission, sources=tuple(sources),
            cumulative_loop_count_effective=loop_count,
            cumulative_emitted_burgers_m=loop_count*emission_row.burgers_m,
            cumulative_loop_swept_area_m2=loop_count*emission_row.loop_swept_area_m2,
            cumulative_area_m2=float(dislocation_derived_area_m2),
            cumulative_emitted_burgers_exact=packed(
                exact(loop_count)*exact(emission_row.burgers_m)),
            cumulative_loop_swept_area_exact=packed(
                exact(loop_count)*exact(emission_row.loop_swept_area_m2)),
            cumulative_area_exact=packed(exact(dislocation_derived_area_m2)),
        )
    return EmbryoGrowthState(
        site_id, radius, area, float(initial_area_m2),
        float(vacancy_derived_area_m2), float(dislocation_derived_area_m2), 0.0,
        0.0, loop_count, emission.cumulative_emitted_burgers_m,
        emission.cumulative_loop_swept_area_m2, emission,
        _frozen({"capture_mode": "CONSERVATIVE_LOCALIZED_KERNEL"}),
        stabilization_clock, healing_clock, representation, ("EMBRYO_BIRTH",), cavity_id,
        direct_interaction_birth_hits=int(direct_interaction_birth_hits),
        vacancy_raw_captured_area_m2=float(vacancy_derived_area_m2),
    )


def loop_area_kinematics(row: CavityEmissionParameterRowV1, radius_m: float) -> Mapping[str, Any]:
    radius = float(radius_m)
    full = row.burgers_m*row.loop_swept_area_m2/row.model_thickness_m
    reduced = row.chi_loop*2.0*math.pi*radius*row.burgers_m
    selected = full if row.loop_projection == "FULL_3D_LOOP" else reduced
    return MappingProxyType({
        "delta_volume_loop_m3": row.burgers_m*row.loop_swept_area_m2,
        "full_3d_projected_area_m2": full,
        "reduced_cylindrical_area_m2": reduced,
        "selected_area_m2": selected,
        "selected_projection": row.loop_projection,
        "chi_loop": row.chi_loop, "calibration_status": row.calibration_status,
    })


def cavity_surface_emission_rates(
    growth: EmbryoGrowthState, *, stress_tensor_Pa: np.ndarray,
    sigma_zz_Pa: float, temperature_K: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    row = growth.cavity_emission_state.parameter_row
    stress = np.asarray(stress_tensor_Pa, float)
    ns = len(growth.cavity_emission_state.sources)
    if stress.shape == (2, 2):
        stress = np.repeat(stress[None, :, :], ns, axis=0)
    if stress.shape != (ns, 2, 2):
        raise ValueError("cavity emission requires one local in-plane tensor per source")
    sigma_zz = np.asarray(sigma_zz_Pa, float)
    if sigma_zz.ndim == 0:
        sigma_zz = np.repeat(float(sigma_zz), ns)
    if sigma_zz.shape != (ns,):
        raise ValueError("cavity emission requires one sigma_zz value per source")
    if temperature_K <= 0.0 or not math.isfinite(temperature_K):
        raise ValueError("temperature must be finite and positive")
    capillary = row.surface_energy_J_m2/growth.radius_m
    rates = []
    drives = []
    circumference = 2.0*math.pi*growth.radius_m
    from .config import EV_TO_J, FractureBarrier
    barrier_model = FractureBarrier(
        barrier_kind="exp_floor", ef_G00_eV=row.zero_stress_barrier_J/EV_TO_J,
        ef_gT_eV_per_K=0.0, ef_sigc0_Pa=row.barrier_critical_stress_Pa,
        ef_sT_Pa_per_K=0.0, ef_a=row.barrier_exp_a, ef_n=row.barrier_exp_n,
        ef_Tref_K=float(temperature_K), ef_floor_frac=row.barrier_floor_fraction,
        ef_floor_min_eV=row.barrier_floor_min_J/EV_TO_J,
        ef_floor_max_frac=0.999999,
    )
    for source_index, source in enumerate(growth.cavity_emission_state.sources):
        local_stress = stress[source_index]
        n = np.array((math.cos(source.orientation_rad), math.sin(source.orientation_rad)))
        t = np.array((-n[1], n[0]))
        if growth.representation == SUBGRID:
            theta = source.orientation_rad
            # Kirsch surface hoop stress for a circular traction-free subgrid
            # cavity under the accepted remote in-plane tensor.
            normal = (float(local_stress[0, 0]+local_stress[1, 1])
                -2.0*float(local_stress[0, 0]-local_stress[1, 1])*math.cos(2.0*theta)
                -4.0*float(local_stress[0, 1])*math.sin(2.0*theta)-capillary)
            shear = 0.0
        else:
            # Promotion preserves the physical hoop/source drive.  A
            # traction-free surface has sigma_nn=sigma_nt=0; using those
            # components would make convergence switch emission off.  The
            # supplied tensor is a qualified solid-side boundary recovery.
            normal = float(t@local_stress@t)-capillary
            shear = 0.0
        resolved = max(normal+shear, 0.0)
        barrier = float(barrier_model.G_barrier(np.asarray([resolved]), temperature_K)[0])
        # Return the undepleted intensity lambda_0.  Availability belongs to
        # the state-owned random-time-change integrator and must not be folded
        # into the caller-visible rate a second time.
        base_sources = row.surface_source_density_per_m*circumference
        rates.append(base_sources*row.attempt_frequency_s
                     *math.exp(-barrier/(KB_J_PER_K*temperature_K)))
        drives.append((normal, shear, resolved, barrier))
    return np.asarray(rates), MappingProxyType({
        "stress_source": growth.representation,
        "accepted_sigma_zz_Pa": tuple(map(float, sigma_zz)),
        "sigma_zz_consumed_by_2d_emission_law": False,
        "capillary_pressure_Pa": capillary,
        "capillary_law": "gamma/R",
        "r_tip_consumed": False,
        "source_drives": tuple(drives),
        "resolved_drive_component": "SIGMA_TT_MINUS_GAMMA_OVER_R",
        "returned_rate_kind": "UNDEPLETED_BASE_INTENSITY_LAMBDA_0",
        "available_fractions": tuple(
            source.available_fraction for source in growth.cavity_emission_state.sources),
        "parameter_row_id": row.row_id,
        "parameter_row_fingerprint": row.fingerprint,
        "barrier_class": "shared config.FractureBarrier/exp_floor",
        "barrier_formula": "Gfloor+(G0-Gfloor)*exp(-a*(sigma/sigc)^n)",
    })


def _recovery_flow(availability: float, recovery_rate_s: float,
                   duration_s: float) -> tuple[float, float]:
    """Availability and its time integral under recovery alone."""
    a0, k, dt = float(availability), float(recovery_rate_s), float(duration_s)
    if k <= 0.0:
        return a0, a0*dt
    decay = math.exp(-k*dt)
    return 1.0-(1.0-a0)*decay, dt-(1.0-a0)*(1.0-decay)/k


def _continuum_population_flow(availability: float, base_rate_s: float,
                               row: CavityEmissionParameterRowV1,
                               duration_s: float) -> tuple[float, float]:
    """Exact mean-field flow and integrated compensator for fixed lambda_0."""
    a0, base, dt = float(availability), float(base_rate_s), float(duration_s)
    p = row.recovery_rate_s+row.depletion_per_loop*base
    if p <= 0.0:
        return a0, base*a0*dt
    aeq = row.recovery_rate_s/p
    decay = math.exp(-p*dt)
    after = aeq+(a0-aeq)*decay
    integral_a = aeq*dt+(a0-aeq)*(1.0-decay)/p
    return after, base*integral_a


def classify_cavity_emission_sources(
    state: CavityEmissionState, *, base_rates_s: Sequence[float], duration_s: float,
) -> tuple[tuple[str, ...], tuple[bool, ...], tuple[float, ...]]:
    """One regime/exclusivity decision shared by event and radius previews."""
    row = state.parameter_row
    regimes, discrete, actions = [], [], []
    for source, base in zip(state.sources, map(float, base_rates_s)):
        _, exposure = _recovery_flow(
            source.available_fraction, row.recovery_rate_s, duration_s)
        mu = base*exposure
        # Production regime ownership is set by the source-population
        # exposure, not by the caller timestep or a random next-jump gap.
        # Hysteresis makes each accepted epoch stable while still allowing
        # both transitions when the physical population scale changes.  The
        # former aggregate tau approximation is retained only as a
        # compatibility request and is executed by the exact path below.
        population_relaxation = (row.recovery_rate_s
                                 +row.depletion_per_loop*max(base, 0.0))
        population_exposure = (0.0 if base <= 0.0 else
            base/max(population_relaxation, np.finfo(float).tiny))
        prior = source.availability_flow_regime
        if prior == "LOW_COUNT_EXACT":
            regime = ("CONTINUUM" if
                      population_exposure >= row.continuum_mu_min
                      else "LOW_COUNT_EXACT")
        elif prior == "CONTINUUM":
            regime = ("LOW_COUNT_EXACT" if population_exposure <= row.low_count_mu_max
                      else "CONTINUUM")
        else:
            regime = ("LOW_COUNT_EXACT" if population_exposure < row.continuum_mu_min
                      else "CONTINUUM")
        if regime == "CONTINUUM":
            _, mu = _continuum_population_flow(
                source.available_fraction, base, row, duration_s)
        regimes.append(regime)
        # The representation mask covers the entire exact source process,
        # including an interval too short to reach its next jump.  Such a
        # source must not simultaneously contribute an expected continuous
        # radius increment.
        discrete.append(regime == "LOW_COUNT_EXACT")
        actions.append(mu)
    return tuple(regimes), tuple(discrete), tuple(actions)


def advance_cavity_emission(
    state: CavityEmissionState, *, radius_m: float, rates_s: Sequence[float],
    duration_s: float, forced_regime: str | None = None,
    forced_regimes_by_source: Mapping[str, str] | None = None,
) -> tuple[CavityEmissionState, Mapping[str, Any]]:
    row = state.parameter_row
    dt = float(duration_s)
    rates = np.asarray(rates_s, float)
    if rates.shape != (len(state.sources),) or np.any(rates < 0.0) or dt < 0.0:
        raise ValueError("invalid cavity-emission interval")
    sources = []
    counts = []
    regimes = []
    area = variance = swept = burgers = 0.0
    classified, _, _ = classify_cavity_emission_sources(
        state, base_rates_s=rates, duration_s=dt)
    action_increments = []
    accepted_count_increments_exact = []
    forced_by_source = ({} if forced_regimes_by_source is None
                        else dict(forced_regimes_by_source))
    if set(forced_by_source)-{item.source_id for item in state.sources}:
        raise ValueError("forced emission regime names an unknown source")
    for source, base_rate, automatic_regime in zip(state.sources, rates, classified):
        regime = forced_regime or forced_by_source.get(
            source.source_id, automatic_regime)
        if regime not in {"LOW_COUNT_EXACT", "TAU_LEAP", "CONTINUUM"}:
            raise ValueError("unknown cavity-emission integration regime")
        if regime == "TAU_LEAP":
            regime = "LOW_COUNT_EXACT"
        action_start = float(source.action_exact())
        availability = source.available_fraction
        if regime == "LOW_COUNT_EXACT":
            # Low-count exact mode resolves recovery and each realized jump.
            same_flow = (source.availability_flow_regime == "LOW_COUNT_EXACT"
                         and source.availability_reference_rate_s == float(base_rate))
            reference_availability = (source.availability_reference_fraction
                                      if same_flow else availability)
            reference_action = (float(unpacked(source.availability_reference_action_exact))
                                if same_flow else action_start)
            elapsed_exact = (unpacked(source.availability_elapsed_time_exact)
                             if same_flow else exact(0.0))
            remaining_time = dt
            action = action_start
            realized_increment = 0
            last_consumed_threshold = None
            while remaining_time > 0.0:
                threshold = poisson_measure_next_action(source.random_measure_seed, action)
                proposed_elapsed = elapsed_exact+exact(remaining_time)
                recovered, total_exposure = _recovery_flow(
                    reference_availability, row.recovery_rate_s,
                    float(proposed_elapsed))
                possible_action = reference_action+float(base_rate)*total_exposure
                possible = possible_action-action
                needed = threshold-action
                # Accepted packet boundaries canonicalize action to twelve
                # significant digits.  Use the same relative budget as the
                # preview/commit crossing contract so a localized named event
                # cannot be stranded just below its immutable threshold by
                # that representation rounding.
                if needed > possible+2e-11*max(
                        1.0, abs(possible), abs(needed), abs(threshold)):
                    availability = recovered; action = possible_action
                    elapsed_exact = proposed_elapsed
                    break
                lo, hi = float(elapsed_exact), float(proposed_elapsed)
                for _ in range(64):
                    mid = 0.5*(lo+hi)
                    _, partial = _recovery_flow(
                        reference_availability, row.recovery_rate_s, mid)
                    if reference_action+float(base_rate)*partial < threshold: lo = mid
                    else: hi = mid
                event_dt = hi-float(elapsed_exact)
                availability, partial = _recovery_flow(
                    reference_availability, row.recovery_rate_s, hi)
                action = math.nextafter(threshold, math.inf)
                last_consumed_threshold = threshold
                availability *= math.exp(-row.depletion_per_loop)
                realized_increment += 1
                remaining_time = max(0.0, remaining_time-event_dt)
                reference_availability = availability
                reference_action = action
                elapsed_exact = exact(0.0)
            moment_increment = 0.0
        elif regime == "TAU_LEAP":
            availability, exposure = _recovery_flow(
                availability, row.recovery_rate_s, dt)
            action = action_start+float(base_rate)*exposure
            start_count = poisson_measure_count(source.random_measure_seed, action_start)
            end_count = poisson_measure_count(source.random_measure_seed, action)
            realized_increment = end_count-start_count
            availability *= math.exp(-row.depletion_per_loop*realized_increment)
            moment_increment = 0.0
        else:
            same_flow = (source.availability_flow_regime == "CONTINUUM"
                         and source.availability_reference_rate_s == float(base_rate))
            if same_flow:
                reference_availability = source.availability_reference_fraction
                reference_action = float(unpacked(
                    source.availability_reference_action_exact))
                elapsed_exact = (unpacked(source.availability_elapsed_time_exact)
                                 +exact(dt))
            else:
                reference_availability = availability
                reference_action = action_start
                elapsed_exact = exact(dt)
            availability, total_action_increment = _continuum_population_flow(
                reference_availability, float(base_rate), row, float(elapsed_exact))
            action = reference_action+total_action_increment
            moment_increment = action-action_start
            realized_increment = 0
        action = _canonical_float(action)
        if (regime == "LOW_COUNT_EXACT" and realized_increment > 0
                and last_consumed_threshold is not None):
            # Significant-digit canonicalization must never move accepted
            # random time back below a threshold that this interval consumed.
            action = max(action, math.nextafter(
                float(last_consumed_threshold), math.inf))
        action_exact = exact(action)
        start_measure_count = poisson_measure_count(source.random_measure_seed, action_start)
        end_measure_count = poisson_measure_count(source.random_measure_seed, action)
        measure_increment = end_measure_count-start_measure_count
        mu = action-action_start
        count = float(realized_increment)+float(moment_increment)
        threshold = poisson_measure_next_action(source.random_measure_seed, action)
        q = float(loop_area_kinematics(row, radius_m)["selected_area_m2"])
        increment = q*count
        area += increment; variance += q*q*mu
        swept += row.loop_swept_area_m2*count; burgers += row.burgers_m*count
        availability = _canonical_float(min(1.0, max(0.0, availability)))
        regime_changed = (source.availability_flow_regime is not None and
                          source.availability_flow_regime != regime)
        new_realized = source.realized_loop_count+realized_increment
        if regime == "CONTINUUM":
            epoch_action = (action_start if regime_changed or
                            source.availability_flow_regime is None else
                            source.regime_epoch_start_action)
            epoch_moment = (source.continuum_moment_loop_count
                            if regime_changed or
                            source.availability_flow_regime is None else
                            source.regime_epoch_start_moment_loop_count)
            new_moment = _canonical_float(epoch_moment+action-epoch_action)
        else:
            new_moment = source.continuum_moment_loop_count
        new_effective = float(new_realized)+new_moment
        accepted_count_increment_exact = (
            exact(new_effective)-exact(source.emitted_loop_count_effective))
        accepted_count_increments_exact.append(accepted_count_increment_exact)
        sources.append(replace(
            source, available_fraction=availability, integrated_action=action,
            integrated_action_exact=packed(action_exact),
            next_exact_threshold=threshold,
            emitted_loop_count_effective=new_effective,
            realized_loop_count=new_realized,
            continuum_moment_loop_count=new_moment,
            random_measure_cursor_count=end_measure_count,
            availability_reference_fraction=(reference_availability
                if regime in {"CONTINUUM", "LOW_COUNT_EXACT"} else availability),
            availability_reference_rate_s=float(base_rate),
            availability_elapsed_time_exact=(packed(elapsed_exact)
                if regime in {"CONTINUUM", "LOW_COUNT_EXACT"}
                else packed(exact(0.0))),
            availability_reference_action_exact=(packed(exact(reference_action))
                if regime in {"CONTINUUM", "LOW_COUNT_EXACT"}
                else packed(exact(action))),
            availability_flow_regime=regime,
            regime_epoch=source.regime_epoch+int(regime_changed),
            regime_epoch_start_action=(action_start if regime_changed else
                                       source.regime_epoch_start_action),
            regime_epoch_start_moment_loop_count=(
                source.continuum_moment_loop_count if regime_changed else
                source.regime_epoch_start_moment_loop_count),
        ))
        counts.append(float(accepted_count_increment_exact)); regimes.append(regime); action_increments.append(mu)
    packet_count = max(1, int(math.ceil(area/max(
        row.maximum_packet_area_fraction*math.pi*radius_m**2, 1.0e-300))))
    audit = {
        "duration_s": dt, "expected_counts": tuple(action_increments),
        "accepted_counts": tuple(counts), "regimes": tuple(regimes),
        "random_measure_cursor_increments": tuple(
            b.random_measure_cursor_count-a.random_measure_cursor_count
            for a, b in zip(state.sources, sources)),
        "one_underlying_counting_process": True,
        "continuum_enumerated_thresholds": 0,
        "area_increment_m2": area, "area_variance_m4": variance,
        "packet_count": packet_count,
        "maximum_packet_area_fraction": row.maximum_packet_area_fraction,
    }
    accepted_count_increment_exact = sum(
        accepted_count_increments_exact, exact(0.0))
    burgers_exact = (unpacked(state.cumulative_emitted_burgers_exact)
                     +accepted_count_increment_exact*exact(row.burgers_m))
    swept_exact = (unpacked(state.cumulative_loop_swept_area_exact)
                   +accepted_count_increment_exact*exact(row.loop_swept_area_m2))
    area_exact = (unpacked(state.cumulative_area_exact)
                  +accepted_count_increment_exact*exact(float(
                      loop_area_kinematics(row, radius_m)["selected_area_m2"])))
    return replace(
        state, sources=tuple(sources),
        cumulative_loop_count_effective=math.fsum(
            source.emitted_loop_count_effective for source in sources),
        cumulative_emitted_burgers_m=float(burgers_exact),
        cumulative_loop_swept_area_m2=float(swept_exact),
        cumulative_area_m2=float(area_exact),
        cumulative_emitted_burgers_exact=packed(burgers_exact),
        cumulative_loop_swept_area_exact=packed(swept_exact),
        cumulative_area_exact=packed(area_exact),
        cumulative_area_variance_m4=float(
            f"{state.cumulative_area_variance_m4+variance:.8g}"),
        last_audit=_frozen(audit),
    ), MappingProxyType(audit)


def mechanism_resolved_growth_rates(
    *, vacancy_raw_area_rate_m2_s: float, bulk_accommodation_area_rate_m2_s: float,
    dislocation_area_rate_m2_s: float, relaxation_area_rate_m2_s: float = 0.0,
) -> Mapping[str, float]:
    raw = max(float(vacancy_raw_area_rate_m2_s), 0.0)
    accom = max(float(bulk_accommodation_area_rate_m2_s), 0.0)
    vacancy = 0.0 if raw <= 0.0 or accom <= 0.0 else 1.0/(1.0/raw+1.0/accom)
    loop = max(float(dislocation_area_rate_m2_s), 0.0)
    relax = max(float(relaxation_area_rate_m2_s), 0.0)
    return MappingProxyType({
        "vacancy_raw_area_rate_m2_s": raw,
        "vacancy_accommodated_area_rate_m2_s": vacancy,
        "bulk_accommodation_area_rate_m2_s": accom,
        "cavity_emitted_loops_excluded_from_bulk_accommodation": True,
        "dislocation_area_rate_m2_s": loop,
        "relaxation_area_rate_m2_s": relax,
        "total_area_rate_m2_s": vacancy+loop-relax,
    })


class CavityLoopSourceRecoveryQualificationError(RuntimeError):
    """State-bound failure with the complete recovery qualification record."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        super().__init__("resolved cavity loop source recovery failed qualification")
        self.diagnostic = dict(diagnostic)


def source_local_fem_tensors(live_state: Any, growth: EmbryoGrowthState
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Recover the accepted constitutive tensor at each physical source position."""
    from .vacancy_transport_phase_field_v2 import accepted_elastoplastic_stress_3d
    stress3d = accepted_elastoplastic_stress_3d(live_state)
    site = next(x for x in live_state.void_state.sites if x.site_id == growth.site_id)
    centers = np.asarray(live_state.mesh.nodes)[np.asarray(live_state.mesh.elems)].mean(axis=1)
    tensors, zz = [], []
    qualified = (growth.representation == RESOLVED
                 and growth.cavity_emission_state.enabled)
    recovered_by_source = {}
    if qualified:
        from .fem import assemble_mechanics
        from .piecewise_equilibrated_boundary_stress_v2 import (
            ACCEPTANCE_LIMITS, PiecewiseEquilibratedStressUnavailable,
            recover_piecewise_equilibrated_boundary_stress_v2,
        )
        from .topology_transaction_v11 import (
            complete_accepted_state_fingerprint, require_equilibrium_observables,
        )
        from .voiding_production_v5 import (
            _actual_cavity_boundary_edges, _bundle_from_state,
            fresh_sharp_front_engine,
        )
        cavity = next(x for x in live_state.void_state.cavities
                      if x.cavity_id == growth.cavity_id)
        boundary_edges = _actual_cavity_boundary_edges(live_state)
        boundary_nodes = np.unique(boundary_edges)
        _, _, sigma2d, *_ = assemble_mechanics(
            live_state.mesh, live_state.displacement, live_state.ep_gp,
            live_state.rho_gp, live_state.damage, live_state.elasticity_D,
            live_state.material, cohesive_network=live_state.cohesive_network)
        equilibrium = require_equilibrium_observables(live_state)
        width = float(np.ptp(np.asarray(live_state.mesh.nodes)[:, 0]))
        remote = 0.5*(abs(float(equilibrium["latest_top_reaction_N_per_m"]))
                     +abs(float(equilibrium["latest_bottom_reaction_N_per_m"])))/width
        engine = fresh_sharp_front_engine(
            live_state.material, _bundle_from_state(live_state))
        damage = getattr(live_state.mesh, "element_damage_gp", None)
        solid = None if damage is None else np.asarray(damage, float) < 0.5
        for source in growth.cavity_emission_state.sources:
            target = np.asarray(cavity.center_m)+growth.radius_m*np.array((
                math.cos(source.orientation_rad), math.sin(source.orientation_rad)))
            node = int(boundary_nodes[np.argmin(np.linalg.norm(
                np.asarray(live_state.mesh.nodes)[boundary_nodes]-target, axis=1))])
            try:
                recovered = recover_piecewise_equilibrated_boundary_stress_v2(
                    nodes=live_state.mesh.nodes, elements=live_state.mesh.elems,
                    stress_Pa=sigma2d, boundary_node=node,
                    cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
                    process_length_m=engine.f.L_pz, owned_boundary_edges=boundary_edges,
                    remote_traction_scale_Pa=remote,
                    state_fingerprint=complete_accepted_state_fingerprint(live_state),
                    solid_element_mask=solid)
            except PiecewiseEquilibratedStressUnavailable as exc:
                raise RuntimeError("resolved cavity loop source recovery unavailable") from exc
            checks = {
                "condition": {
                    "value": recovered["condition"],
                    "limit": recovered["condition_limit"],
                },
                "stress_fit_residual": {
                    "value": recovered["stress_fit_residual"],
                    "limit": recovered["stress_fit_residual_limit"],
                },
                "normalized_boundary_traction": {
                    "value": recovered["normalized_boundary_traction"],
                    "limit": recovered["constraint_limits"]["normalized_boundary_traction"],
                },
                "interface_traction_jump_residual": {
                    "value": recovered["interface_traction_jump_residual"],
                    "limit": recovered["constraint_limits"]["interface_traction_jump"],
                },
                "resultant_residual": {
                    "value": recovered["resultant_residual"],
                    "limit": recovered["constraint_limits"]["resultant"],
                },
                "moment_residual": {
                    "value": recovered["moment_residual"],
                    "limit": recovered["constraint_limits"]["moment"],
                },
            }
            failed = tuple(name for name, row in checks.items()
                           if float(row["value"]) > float(row["limit"]))
            if failed:
                raise CavityLoopSourceRecoveryQualificationError({
                    "schema": "resolved-cavity-loop-source-recovery-failure/1",
                    "source_id": source.source_id,
                    "source_orientation_rad": float(source.orientation_rad),
                    "source_target_m": target.tolist(),
                    "selected_boundary_node": node,
                    "cavity_id": cavity.cavity_id,
                    "cavity_center_m": list(map(float, cavity.center_m)),
                    "cavity_radius_m": float(cavity.radius_m),
                    "state_fingerprint": complete_accepted_state_fingerprint(live_state),
                    "mesh_node_count": int(len(live_state.mesh.nodes)),
                    "mesh_element_count": int(len(live_state.mesh.elems)),
                    "owned_boundary_edge_count": int(len(boundary_edges)),
                    "remote_traction_scale_Pa": float(remote),
                    "failed_checks": failed,
                    "checks": checks,
                    "recovery": recovered,
                })
            recovered_by_source[source.source_id] = recovered
    for source in growth.cavity_emission_state.sources:
        n = np.array((math.cos(source.orientation_rad), math.sin(source.orientation_rad)))
        xy = np.asarray(site.center_m)+(growth.radius_m*n if growth.representation == RESOLVED else 0.0)
        eid = int(np.argmin(np.linalg.norm(centers-xy, axis=1)))
        if qualified:
            tensors.append(recovered_by_source[source.source_id]["source_tensor_Pa"])
        else:
            tensors.append(((stress3d["sigma_xx_Pa"][eid], stress3d["sigma_xy_Pa"][eid]),
                            (stress3d["sigma_xy_Pa"][eid], stress3d["sigma_yy_Pa"][eid])))
        zz.append(stress3d["sigma_zz_Pa"][eid])
    return np.asarray(tensors, float), np.asarray(zz, float)


def consume_resolved_cavity_loop_source(
    live_state: Any, growth: EmbryoGrowthState, *, temperature_K: float,
    mechanism_requested: bool | None = None,
) -> Mapping[str, Any]:
    """Request the resolved production source without mutating accepted state.

    Scientific source-recovery failures are converted to a typed unavailable
    result.  Diagnostic calculations may still exist elsewhere, but this is
    the gate that decides whether production kinetics receives a usable rate.
    """
    from .topology_transaction_v11 import complete_accepted_state_fingerprint

    before = complete_accepted_state_fingerprint(live_state)
    configured = bool(
        growth.representation == RESOLVED
        and growth.cavity_emission_state.enabled
        and live_state.void_state.embryo_growth_configuration
            .cavity_surface_loop_emission_enabled)
    requested = configured if mechanism_requested is None else bool(mechanism_requested)
    reason = None
    rates = None
    source_audit = None
    if not requested:
        status = "MECHANISM_NOT_REQUESTED"
        reason = "resolved cavity-loop emission is not enabled by its accepted owner"
    else:
        try:
            requested_growth = (growth if growth.cavity_emission_state.enabled else
                replace(growth, cavity_emission_state=replace(
                    growth.cavity_emission_state, enabled=True)))
            tensors, sigma_zz = source_local_fem_tensors(live_state, requested_growth)
            rates, source_audit = cavity_surface_emission_rates(
                requested_growth, stress_tensor_Pa=tensors, sigma_zz_Pa=sigma_zz,
                temperature_K=temperature_K)
        except RuntimeError as exc:
            status = "UNAVAILABLE_NO_RATE"
            reason = str(exc)
        else:
            status = "RATE_SUPPLIED"
    after = complete_accepted_state_fingerprint(live_state)
    payload = {
        "schema": "unified-2d.resolved-cavity-loop-source-consumer/1",
        "mechanism_requested": requested,
        "mechanism_configured_on_accepted_state": configured,
        "source_status": status,
        "actual_reason": reason,
        "production_rate_s": (None if rates is None else tuple(map(float, rates))),
        "source_audit": (None if source_audit is None else dict(source_audit)),
        "input_state_fingerprint": before,
        "output_state_fingerprint": after,
        "accepted_state_unchanged": before == after,
    }
    payload["authentication_sha256"] = _sha(payload)
    return MappingProxyType(payload)


@dataclass(frozen=True)
class EmbryoGrowthProcessOwner:
    """The single production owner from embryo birth through cavity growth."""
    configuration: EmbryoGrowthConfiguration = field(default_factory=EmbryoGrowthConfiguration)

    def preview_next_transition(
        self, growth: EmbryoGrowthState, *, rates_s: Sequence[float],
        vacancy_raw_area_rate_m2_s: float, bulk_accommodation_area_rate_m2_s: float,
        stable_radius_m: float, promotion_radius_m: float,
        local_mesh_size_m: float, remesh_fraction: float,
        maximum_duration_s: float,
        additional_test_supply_area_rate_m2_s: float = 0.0,
    ) -> Mapping[str, Any]:
        rates = np.asarray(rates_s, float)
        candidates: list[tuple[float, str, str | None]] = []
        if self.configuration.cavity_surface_loop_emission_enabled:
            for source, rate in zip(growth.cavity_emission_state.sources, rates):
                if rate > 0.0:
                    t = (source.next_exact_threshold-float(source.action_exact()))/rate
                    if t >= -64*np.finfo(float).eps:
                        candidates.append((max(t, 0.0), "CAVITY_LOOP_EMISSION", source.source_id))
        q = float(loop_area_kinematics(
            growth.cavity_emission_state.parameter_row, growth.radius_m)["selected_area_m2"])
        area_rate = float(mechanism_resolved_growth_rates(
            vacancy_raw_area_rate_m2_s=vacancy_raw_area_rate_m2_s,
            bulk_accommodation_area_rate_m2_s=bulk_accommodation_area_rate_m2_s,
            dislocation_area_rate_m2_s=(float(np.sum(rates))*q+float(
                additional_test_supply_area_rate_m2_s)))["total_area_rate_m2_s"])
        radial_rate = area_rate/max(2.0*math.pi*growth.radius_m, 1e-300)
        for radius, kind in ((stable_radius_m, "R_STABLE"),
                             (promotion_radius_m, "R_PROMOTION")):
            if radius > growth.radius_m and area_rate > 0.0:
                candidates.append((math.pi*(radius*radius-growth.radius_m**2)/area_rate,
                                   kind, None))
            elif (kind == "R_PROMOTION" and growth.representation != RESOLVED
                  and radius > 0.0 and growth.radius_m >= radius):
                # A birth transaction may legitimately create an analytical
                # embryo already beyond the selected promotion radius.  That
                # is an immediately eligible topology event, not a state in
                # which promotion silently disappears from the scheduler.
                candidates.append((0.0, kind, None))
        if growth.representation == RESOLVED and radial_rate > 0.0:
            remesh_radius = (float(growth.last_remesh_radius_m)
                             +remesh_fraction*local_mesh_size_m)
            candidates.append((max(math.pi*(remesh_radius**2-growth.radius_m**2)
                                   /area_rate, 0.0),
                               "R_REMESH_CADENCE", None))
        candidates = [x for x in candidates if x[0] <= maximum_duration_s]
        if not candidates:
            return MappingProxyType({
                "eligible": False, "crossing_time_s": math.inf,
                "event_type": "NO_GROWTH_TRANSITION",
                # The absence of a radius transition does not mean the
                # continuous growth rate is zero.  The global scheduler uses
                # this rate to place the next mandatory growth-packet
                # boundary and must therefore receive it on every preview.
                "radial_rate_m_s": radial_rate,
                "rate_source": "OWNER_MECHANISM_RESOLVED_GROWTH",
            })
        time_s, event_type, source_id = min(candidates, key=lambda x: (x[0], x[1], x[2] or ""))
        return MappingProxyType({
            "eligible": True, "crossing_time_s": time_s, "event_type": event_type,
            "source_id": source_id, "radial_rate_m_s": radial_rate,
            "rate_source": "OWNER_MECHANISM_RESOLVED_GROWTH",
        })

    def advance_continuous_state(self, live_state: Any, **kwargs):
        return advance_live_embryo_growth(live_state, **kwargs)

    def commit_loop_event(self, live_state: Any, preview: Any, **kwargs):
        """Commit exactly the named low-count jump bound by ``preview``."""
        from .global_physical_scheduler_v1 import (
            PhysicalEventPreview, _accepted_fingerprint, _rng_fingerprint,
            _topology_fingerprint,
        )
        if not isinstance(preview, PhysicalEventPreview):
            raise TypeError("loop commit requires a PhysicalEventPreview")
        if (preview.event_type != "CAVITY_LOOP_EMISSION"
                or preview.owner_kind != "EmbryoGrowthProcessOwner"):
            raise RuntimeError("preview is not a cavity-loop event")
        site_id = str(kwargs.get("site_id", preview.owner_id))
        if preview.owner_id != site_id:
            raise RuntimeError("loop preview owner changed before commit")
        if preview.accepted_state_fingerprint != _accepted_fingerprint(live_state):
            raise RuntimeError("stale accepted-state loop preview")
        if preview.topology_fingerprint != _topology_fingerprint(live_state):
            raise RuntimeError("stale topology loop preview")
        if preview.rng_fingerprint != _rng_fingerprint(live_state):
            raise RuntimeError("stale RNG loop preview")
        growth_before = next(x for x in live_state.void_state.embryo_growth_states
                             if x.site_id == site_id)
        if preview.source_fingerprint != growth_before.cavity_emission_state.fingerprint:
            raise RuntimeError("stale growth-owner loop preview")
        if len(preview.candidate_ids) != 1 or len(preview.threshold_identity) != 1:
            raise RuntimeError("loop preview must bind exactly one source threshold")
        source_id = str(preview.candidate_ids[0])
        source_before = next(x for x in growth_before.cavity_emission_state.sources
                             if x.source_id == source_id)
        threshold_id, threshold = preview.threshold_identity[0]
        if threshold_id != source_id or not math.isclose(
                float(threshold), source_before.next_exact_threshold,
                rel_tol=0.0, abs_tol=8*np.spacing(source_before.next_exact_threshold)):
            raise RuntimeError("loop preview threshold identity changed")
        duration = float(kwargs.get("duration_s", preview.crossing_time_s))
        if not math.isclose(duration, preview.crossing_time_s,
                            rel_tol=2e-12, abs_tol=np.finfo(float).tiny):
            raise RuntimeError("loop commit duration differs from preview crossing")
        preview_rates = tuple(map(float, preview.commit_payload["rates_s"]))
        source_index = next(i for i, x in enumerate(growth_before.cavity_emission_state.sources)
                            if x.source_id == source_id)
        _, exposure = _recovery_flow(
            source_before.available_fraction,
            growth_before.cavity_emission_state.parameter_row.recovery_rate_s,
            duration)
        expected_action = preview_rates[source_index]*exposure
        remaining_action = (source_before.next_exact_threshold
                            -float(source_before.action_exact()))
        if not math.isclose(expected_action, remaining_action, rel_tol=2e-11,
                            abs_tol=max(1e-14, 8*np.spacing(remaining_action))):
            raise RuntimeError("loop preview rate/crossing contract changed")
        kwargs = dict(kwargs)
        preview_regimes = dict(kwargs.pop("forced_regimes_by_source", {}))
        preview_regimes[source_id] = "LOW_COUNT_EXACT"
        # Only the named winner is exact.  Losing sources retain the regimes
        # bound by the preview and advance once over the same physical time.
        kwargs.pop("forced_regime", None)
        kwargs.update(site_id=site_id, duration_s=duration,
                      forced_regimes_by_source=preview_regimes)
        commit_base = kwargs.pop("transport_committed_live_state", live_state)
        committed, audit = advance_live_embryo_growth(commit_base, **kwargs)
        growth_after = next(x for x in committed.void_state.embryo_growth_states
                            if x.site_id == site_id)
        increments = {
            b.source_id: a.realized_loop_count-b.realized_loop_count
            for b, a in zip(growth_before.cavity_emission_state.sources,
                            growth_after.cavity_emission_state.sources)
        }
        if increments.get(source_id) != 1 or any(
                value != 0 for key, value in increments.items() if key != source_id):
            raise RuntimeError(
                "atomic loop commit did not consume exactly one named source: "
                +repr({"source_id": source_id, "increments": increments,
                       "duration_s": duration, "preview_rates_s": preview_rates,
                       "source_action": float(source_before.action_exact()),
                       "stored_next_threshold": source_before.next_exact_threshold,
                       "derived_next_threshold": poisson_measure_next_action(
                           source_before.random_measure_seed,
                           float(source_before.action_exact())),
                       "growth_audit": dict(audit)}))
        return committed, MappingProxyType({**dict(audit),
            "committed_preview_source_id": source_id,
            "exact_named_source_increment": 1})

    def commit_radius_transition(self, growth: EmbryoGrowthState, *, event_type: str,
                                 cavity_id: str | None = None) -> EmbryoGrowthState:
        if event_type == "R_PROMOTION":
            if cavity_id is None:
                raise RuntimeError("promotion requires the committed cavity identity")
            return promote_representation(growth, cavity_id)
        if event_type == "R_STABLE":
            return replace(growth, lineage=growth.lineage+("STABLE_RADIUS_CROSSED",),
                           stabilization_action_complete_waiting_for_radius=False)
        if event_type == "R_REMESH_CADENCE":
            return replace(
                growth, lineage=growth.lineage+("RESOLVED_REMESH_CADENCE",),
                last_remesh_radius_m=growth.radius_m)
        raise ValueError("unknown growth-owned radius transition")

    def checkpoint_payload(self, growth: EmbryoGrowthState) -> Mapping[str, Any]:
        return MappingProxyType({"schema": SCHEMA, "growth_fingerprint": growth.fingerprint,
                                 "configuration": asdict(self.configuration)})

    def rollback_fingerprint(self, live_state: Any) -> str:
        from .topology_transaction_v11 import complete_accepted_state_fingerprint
        return complete_accepted_state_fingerprint(live_state)


def make_cavity_loop_event_preview(
    live_state: Any, growth: EmbryoGrowthState, *, temperature_K: float,
    maximum_duration_s: float,
) -> Any:
    """Expose the next low-count source threshold to the global scheduler."""
    from .global_physical_scheduler_v1 import (
        PhysicalEventPreview, _accepted_fingerprint, _rng_fingerprint,
        _topology_fingerprint,
    )
    tensors, zz = source_local_fem_tensors(live_state, growth)
    rates, rate_audit = cavity_surface_emission_rates(
        growth, stress_tensor_Pa=tensors, sigma_zz_Pa=zz,
        temperature_K=temperature_K)
    _, discrete_mask, maximum_actions = classify_cavity_emission_sources(
        growth.cavity_emission_state, base_rates_s=rates,
        duration_s=maximum_duration_s)
    candidates = []
    for source, rate, is_discrete, maximum_action in zip(
            growth.cavity_emission_state.sources, rates, discrete_mask, maximum_actions):
        remaining = source.next_exact_threshold-float(source.action_exact())
        crossing = math.inf
        if is_discrete and rate > 0.0 and remaining <= maximum_action+1e-12:
            lo, hi = 0.0, float(maximum_duration_s)
            for _ in range(64):
                mid = 0.5*(lo+hi)
                _, exposure = _recovery_flow(
                    source.available_fraction,
                    growth.cavity_emission_state.parameter_row.recovery_rate_s, mid)
                if float(rate)*exposure < remaining: lo = mid
                else: hi = mid
            crossing = hi
        if is_discrete and crossing <= maximum_duration_s:
            candidates.append((crossing, source, float(rate)))
    selected = min(candidates, key=lambda x: (x[0], x[1].source_id)) if candidates else None
    crossing = math.inf if selected is None else selected[0]
    source_id = "none" if selected is None else selected[1].source_id
    return PhysicalEventPreview(
        event_type="CAVITY_LOOP_EMISSION" if selected is not None else "NO_CAVITY_LOOP_EMISSION",
        owner_kind="EmbryoGrowthProcessOwner", owner_id=growth.site_id,
        crossing_time_s=crossing,
        accepted_state_fingerprint=_accepted_fingerprint(live_state),
        source_fingerprint=growth.cavity_emission_state.fingerprint,
        topology_fingerprint=_topology_fingerprint(live_state),
        event_ids=(() if selected is None else (f"cavity-loop:{growth.site_id}:{source_id}",)),
        candidate_ids=(() if selected is None else (source_id,)),
        threshold_identity=(() if selected is None else ((source_id, selected[1].next_exact_threshold),)),
        rng_fingerprint=_rng_fingerprint(live_state),
        commit_payload=MappingProxyType({
            "site_id": growth.site_id, "source_id": source_id,
            "rates_s": tuple(map(float, rates)), "rate_audit": dict(rate_audit),
        }),
    )


def make_growth_radius_event_preview(
    live_state: Any, growth: EmbryoGrowthState, *, temperature_K: float,
    accepted_equivalent_plastic_strain_rate_s: float,
    promotion_radius_m: float, local_mesh_size_m: float,
    remesh_fraction: float, maximum_duration_s: float,
    transport_trial_after: Any,
    functional_test_supply_area_rate_m2_s: float = 0.0,
) -> Any:
    """Preview radius boundaries from the owner's measured mechanism rates."""
    from .global_physical_scheduler_v1 import (
        PhysicalEventPreview, _accepted_fingerprint, _rng_fingerprint,
        _topology_fingerprint,
    )
    tensors, zz = source_local_fem_tensors(live_state, growth)
    rates, _ = cavity_surface_emission_rates(
        growth, stress_tensor_Pa=tensors, sigma_zz_Pa=zz,
        temperature_K=temperature_K)
    if not growth.cavity_emission_state.enabled:
        rates = np.zeros_like(rates)
    source_regimes, discrete_mask, action_increments = classify_cavity_emission_sources(
        growth.cavity_emission_state, base_rates_s=rates,
        duration_s=maximum_duration_s)
    exact_mask = np.asarray(discrete_mask, dtype=bool)
    # Exact sources are represented exclusively by CAVITY_LOOP_EMISSION
    # previews.  Their expected value may not also move a continuous radius
    # boundary in the same scheduler interval.
    continuous_rates = np.where(
        exact_mask, 0.0,
        np.asarray(action_increments)/max(float(maximum_duration_s), 1e-300))
    packet_continuum_rates = continuum_packet_rate_upper(
        growth.cavity_emission_state, rates, exact_mask)
    transport = live_state.vacancy_transport_state_v2
    owner_kind = "ACTIVE_CAVITY" if growth.representation == RESOLVED else "SUBGRID_SITE"
    owner_id = growth.cavity_id if owner_kind == "ACTIVE_CAVITY" else growth.site_id
    trial_certificate = transport_flux_certificate(
        transport, transport_trial_after, owner_id=owner_id, owner_kind=owner_kind)
    raw_rate = (trial_certificate.raw_absorbed_area_m2
                /trial_certificate.duration_s)
    bulk_rate = 2.0*math.pi*growth.radius_m**2*max(
        float(accepted_equivalent_plastic_strain_rate_s), 0.0)
    loop_area = float(loop_area_kinematics(
        growth.cavity_emission_state.parameter_row,
        growth.radius_m)["selected_area_m2"])
    # Packet synchronization must use the same state-owned continuum flow as
    # the radius preview.  ``rates`` is the undepleted lambda_0; using it here
    # after source availability has collapsed creates an endless sequence of
    # tiny safety boundaries that advance almost no physical growth.  Exact
    # sources remain owned exclusively by CAVITY_LOOP_EMISSION previews.
    packet_upper_area_rate = float(mechanism_resolved_growth_rates(
        vacancy_raw_area_rate_m2_s=raw_rate,
        bulk_accommodation_area_rate_m2_s=bulk_rate,
        dislocation_area_rate_m2_s=(float(np.sum(packet_continuum_rates))*loop_area
            +float(functional_test_supply_area_rate_m2_s)),
    )["total_area_rate_m2_s"])
    packet_upper_radial_rate = packet_upper_area_rate/max(
        2.0*math.pi*growth.radius_m, 1e-300)
    owner = EmbryoGrowthProcessOwner(EmbryoGrowthConfiguration(
        cavity_surface_loop_emission_enabled=False))
    stable_target = (
        growth.cavity_emission_state.parameter_row.minimum_stable_radius_m
        if "STABLE_RADIUS_CROSSED" not in growth.lineage else 0.0)
    result = owner.preview_next_transition(
        growth, rates_s=continuous_rates, vacancy_raw_area_rate_m2_s=raw_rate,
        bulk_accommodation_area_rate_m2_s=bulk_rate,
        stable_radius_m=stable_target,
        promotion_radius_m=promotion_radius_m, local_mesh_size_m=local_mesh_size_m,
        remesh_fraction=remesh_fraction, maximum_duration_s=maximum_duration_s,
        additional_test_supply_area_rate_m2_s=float(
            functional_test_supply_area_rate_m2_s))
    radius_types = {
        "R_STABLE": "GROWTH_STABLE_RADIUS", "R_PROMOTION": "GROWTH_PROMOTION",
        "R_REMESH_CADENCE": "GROWTH_REMESH_CADENCE",
    }
    kind = radius_types.get(result["event_type"])
    crossing = float(result["crossing_time_s"]) if kind is not None else math.inf
    target_radius = ({
        "GROWTH_STABLE_RADIUS": growth.cavity_emission_state.parameter_row.minimum_stable_radius_m,
        "GROWTH_PROMOTION": promotion_radius_m,
        "GROWTH_REMESH_CADENCE": (float(growth.last_remesh_radius_m)
                                   +remesh_fraction*local_mesh_size_m),
    }.get(kind, math.inf))
    return PhysicalEventPreview(
        event_type=kind or "NO_GROWTH_RADIUS_TRANSITION",
        owner_kind="EmbryoGrowthProcessOwner", owner_id=growth.site_id,
        crossing_time_s=crossing,
        accepted_state_fingerprint=_accepted_fingerprint(live_state),
        source_fingerprint=growth.fingerprint,
        topology_fingerprint=_topology_fingerprint(live_state),
        event_ids=(() if kind is None else (f"{kind.lower()}:{growth.site_id}",)),
        candidate_ids=(growth.site_id,),
        threshold_identity=(() if kind is None else ((kind, crossing),)),
        rng_fingerprint=_rng_fingerprint(live_state),
        commit_payload=MappingProxyType({
            **dict(result), "site_id": growth.site_id, "cavity_id": growth.cavity_id,
            "continuous_loop_source_ids": tuple(
                source.source_id for source, flag in zip(
                    growth.cavity_emission_state.sources, exact_mask) if not flag),
            "exact_loop_source_ids": tuple(
                source.source_id for source, flag in zip(
                    growth.cavity_emission_state.sources, exact_mask) if flag),
            "source_regimes": tuple(
                (source.source_id, regime) for source, regime in zip(
                    growth.cavity_emission_state.sources, source_regimes)),
            "low_count_sources_excluded_from_continuous_radius_rate": True,
            "transport_rate_source": "CURRENT_INTERVAL_ACCEPTED_TRANSPORT_TRIAL",
            "transport_trial_before_fingerprint": transport.fingerprint,
            "transport_trial_after_fingerprint": transport_trial_after.fingerprint,
            "transport_trial_certificate_fingerprint": trial_certificate.fingerprint,
            "packet_upper_radial_rate_m_s": packet_upper_radial_rate,
            "packet_emission_rate_kind": (
                "STATE_OWNED_INSTANTANEOUS_CONTINUUM_RATE_WITH_RECOVERY_EQUILIBRIUM_UPPER_BOUND"),
            "target_radius_m": float(target_radius),
            "radius_preview_requires_postintegration_verification": True,
        }),
    )


def continuum_packet_rate_upper(
    state: CavityEmissionState, base_rates_s: Any, exact_mask: Any,
) -> np.ndarray:
    """Upper-bound the next continuum packet using owned source availability."""
    rates = np.asarray(base_rates_s, dtype=float)
    mask = np.asarray(exact_mask, dtype=bool)
    if rates.shape != (len(state.sources),) or mask.shape != rates.shape:
        raise ValueError("packet-rate inputs must match the source population")
    recovery = state.parameter_row.recovery_rate_s
    depletion = state.parameter_row.depletion_per_loop
    availability_upper = np.asarray(tuple(
        max(source.available_fraction,
            recovery/max(recovery+depletion*float(rate), 1e-300))
        for source, rate in zip(state.sources, rates)
    ), dtype=float)
    return np.where(mask, 0.0, rates*availability_upper)


def advance_embryo_growth(
    growth: EmbryoGrowthState, *, duration_s: float,
    transport_certificate: TransportFluxCertificate,
    transport_before_state: Any, transport_after_state: Any,
    bulk_accommodation_area_rate_m2_s: float,
    stress_tensor_Pa: np.ndarray, sigma_zz_Pa: Any, temperature_K: float,
    relaxation_area_rate_m2_s: float = 0.0, forced_regime: str | None = None,
    direct_interaction_area_rate_m2_s: float = 0.0,
    forced_regimes_by_source: Mapping[str, str] | None = None,
    vacancy_formation_volume_m3: float = 1.0e-29,
    loop_emission_enabled: bool = True,
) -> tuple[EmbryoGrowthState, Mapping[str, Any]]:
    if not isinstance(transport_certificate, TransportFluxCertificate):
        raise TypeError("scientific embryo growth requires a typed transport flux certificate")
    if transport_certificate.owner_id not in {growth.site_id, growth.cavity_id}:
        raise RuntimeError("transport certificate is bound to a different growth owner")
    duration_roundoff = 64*np.spacing(max(
        abs(float(transport_after_state.transport_time_s)),
        abs(float(transport_before_state.transport_time_s)),
        abs(float(duration_s)), np.finfo(float).tiny))
    if not math.isclose(transport_certificate.duration_s, duration_s,
                        rel_tol=2e-12,
                        abs_tol=max(np.finfo(float).tiny, duration_roundoff)):
        raise RuntimeError("transport certificate duration does not match growth interval")
    transport_certificate.verify(transport_before_state, transport_after_state)
    vacancy_raw_area_rate_m2_s = transport_certificate.raw_absorbed_area_m2/duration_s
    if not math.isfinite(vacancy_formation_volume_m3) or vacancy_formation_volume_m3 <= 0.0:
        raise ValueError("vacancy formation volume must be finite and positive")
    rates, stress_audit = cavity_surface_emission_rates(
        growth, stress_tensor_Pa=stress_tensor_Pa, sigma_zz_Pa=sigma_zz_Pa,
        temperature_K=temperature_K,
    )
    if not loop_emission_enabled or not growth.cavity_emission_state.enabled:
        rates = np.zeros_like(rates)
    emission, emission_audit = advance_cavity_emission(
        growth.cavity_emission_state, radius_m=growth.radius_m, rates_s=rates,
        duration_s=duration_s, forced_regime=forced_regime,
        forced_regimes_by_source=forced_regimes_by_source,
    )
    area_rates = mechanism_resolved_growth_rates(
        vacancy_raw_area_rate_m2_s=vacancy_raw_area_rate_m2_s,
        bulk_accommodation_area_rate_m2_s=bulk_accommodation_area_rate_m2_s,
        dislocation_area_rate_m2_s=(
            emission_audit["area_increment_m2"]/max(duration_s, 1e-300)
            +max(float(direct_interaction_area_rate_m2_s), 0.0)),
        relaxation_area_rate_m2_s=relaxation_area_rate_m2_s,
    )
    vacancy_increment = area_rates["vacancy_accommodated_area_rate_m2_s"]*duration_s
    raw_increment = transport_certificate.raw_absorbed_area_m2
    unaccommodated_increment = max(raw_increment-vacancy_increment, 0.0)
    loop_increment = (emission_audit["area_increment_m2"]
                      +max(float(direct_interaction_area_rate_m2_s), 0.0)
                      *duration_s)
    relaxed = min(area_rates["relaxation_area_rate_m2_s"]*duration_s,
                  growth.area_m2+vacancy_increment+loop_increment)
    new_vacancy_area = growth.vacancy_derived_area_m2+vacancy_increment
    new_loop_area = growth.dislocation_derived_area_m2+loop_increment
    new_reversible_area = growth.reversible_area_m2+relaxed
    area = math.fsum((growth.initial_area_m2, new_vacancy_area,
                      new_loop_area, -new_reversible_area))
    capture = dict(growth.vacancy_capture_state)
    capture.update({
        "raw_absorbed_area_m2": float(capture.get("raw_absorbed_area_m2", 0.0))
            +max(float(vacancy_raw_area_rate_m2_s), 0.0)*duration_s,
        "accommodated_area_m2": float(capture.get("accommodated_area_m2", 0.0))
            +vacancy_increment,
    })
    updated = replace(
        growth, radius_m=math.sqrt(area/math.pi), area_m2=area,
        vacancy_derived_area_m2=new_vacancy_area,
        captured_unaccommodated_vacancy_area_m2=(
            growth.captured_unaccommodated_vacancy_area_m2+unaccommodated_increment),
        vacancy_raw_captured_area_m2=growth.vacancy_raw_captured_area_m2+raw_increment,
        dislocation_derived_area_m2=new_loop_area,
        reversible_area_m2=new_reversible_area,
        vacancy_absorbed_count_equivalent=(growth.vacancy_absorbed_count_equivalent
            +vacancy_increment*growth.cavity_emission_state.parameter_row.model_thickness_m
             /vacancy_formation_volume_m3),
        emitted_loop_count_effective=emission.cumulative_loop_count_effective,
        cumulative_emitted_burgers_m=emission.cumulative_emitted_burgers_m,
        cumulative_loop_swept_area_m2=emission.cumulative_loop_swept_area_m2,
        cavity_emission_state=emission, vacancy_capture_state=_frozen(capture),
        retained_loop_count_effective=(growth.retained_loop_count_effective
            +(emission.cumulative_loop_count_effective
              -growth.cavity_emission_state.cumulative_loop_count_effective)
             *emission.parameter_row.retained_loop_fraction),
        escaped_loop_count_effective=(growth.escaped_loop_count_effective
            +(emission.cumulative_loop_count_effective
              -growth.cavity_emission_state.cumulative_loop_count_effective)
             *emission.parameter_row.escaped_loop_fraction),
    )
    return updated, MappingProxyType({
        **dict(area_rates), **dict(stress_audit), "emission": dict(emission_audit),
        "transport_flux_certificate_fingerprint": transport_certificate.fingerprint,
        "captured_unaccommodated_vacancy_increment_m2": unaccommodated_increment,
        "radius_before_m": growth.radius_m, "radius_after_m": updated.radius_m,
        "area_identity_residual_m2": (updated.area_m2-updated.initial_area_m2
            -updated.vacancy_derived_area_m2-updated.dislocation_derived_area_m2
            +updated.reversible_area_m2),
    })


def deposit_emitted_loop_mechanics(
    live_state: Any, growth_before: EmbryoGrowthState, growth_after: EmbryoGrowthState,
    *, center_m: tuple[float, float], support_radius_m: float,
    stress_tensor_Pa: np.ndarray,
) -> tuple[Any, Mapping[str, Any]]:
    delta_loops = growth_after.emitted_loop_count_effective-growth_before.emitted_loop_count_effective
    if delta_loops <= 0.0:
        return live_state, MappingProxyType({"deposited_loop_count": 0.0})
    centers = np.asarray(live_state.mesh.nodes)[np.asarray(live_state.mesh.elems)].mean(axis=1)
    area_e = np.asarray(live_state.mesh.area_e, float)
    row = growth_after.cavity_emission_state.parameter_row
    ep = np.asarray(live_state.ep_gp, float).copy()
    rho = np.asarray(live_state.rho_gp, float).copy()
    tensors = np.asarray(stress_tensor_Pa, float)
    ns = len(growth_after.cavity_emission_state.sources)
    if tensors.shape == (2, 2):
        tensors = np.repeat(tensors[None, :, :], ns, axis=0)
    plastic_work = strain_integral = rho_integral = 0.0
    source_audits = []
    support_ids: set[int] = set()
    for sb, sa, tensor in zip(growth_before.cavity_emission_state.sources,
                              growth_after.cavity_emission_state.sources, tensors):
        count = sa.emitted_loop_count_effective-sb.emitted_loop_count_effective
        if count <= 0.0:
            continue
        n = np.array((math.cos(sa.orientation_rad), math.sin(sa.orientation_rad)))
        t = np.array((-n[1], n[0]))
        source_xy = np.asarray(center_m)+growth_before.radius_m*n
        adjacent = np.linalg.norm(centers-source_xy, axis=1) <= float(support_radius_m)
        if not np.any(adjacent):
            adjacent[int(np.argmin(np.linalg.norm(centers-source_xy, axis=1)))] = True
        weights = np.where(adjacent, area_e, 0.0); weights /= np.sum(weights)
        selected_area = float(loop_area_kinematics(row, growth_before.radius_m)["selected_area_m2"])
        local_strain_integral = count*selected_area/row.model_thickness_m
        # Symmetric source-oriented slip tensor, engineering shear in ep[2].
        P = 0.5*(np.outer(t, n)+np.outer(n, t))
        density = local_strain_integral*weights/np.maximum(area_e, 1e-300)
        ep[0] += P[0, 0]*density
        ep[1] += P[1, 1]*density
        if ep.shape[0] > 2:
            ep[2] += 2.0*P[0, 1]*density
        retained = count*row.retained_loop_fraction
        rho_increment = retained*2.0*math.pi*growth_before.radius_m/max(
            np.sum(area_e[adjacent]), 1e-300)
        rho[adjacent] += rho_increment
        work = max(float(np.sum(tensor*P)), 0.0)*local_strain_integral
        plastic_work += work; strain_integral += local_strain_integral
        rho_integral += rho_increment
        support_ids.update(map(int, np.flatnonzero(adjacent)))
        source_audits.append({
            "source_id": sa.source_id, "orientation_rad": sa.orientation_rad,
            "boundary_position_m": tuple(map(float, source_xy)),
            "local_normal": tuple(map(float, n)), "local_tangent": tuple(map(float, t)),
            "emitted_count": count, "retained_count": retained,
            "escaped_count": count*row.escaped_loop_fraction,
            "plastic_work_J_per_m": work,
        })
    energy = dict(live_state.energy_ledgers)
    energy["cavity_emission_plastic_work_J_per_m"] = (
        float(energy.get("cavity_emission_plastic_work_J_per_m", 0.0))+plastic_work
    )
    energy["cavity_emission_escaped_loop_wake_count"] = (
        float(energy.get("cavity_emission_escaped_loop_wake_count", 0.0))
        +delta_loops*row.escaped_loop_fraction)
    energy["cavity_emission_retained_backstress_population"] = (
        float(energy.get("cavity_emission_retained_backstress_population", 0.0))
        +delta_loops*row.retained_loop_fraction)
    return replace(live_state, ep_gp=ep, rho_gp=rho, energy_ledgers=energy), MappingProxyType({
        "deposited_loop_count": delta_loops, "support_element_ids": tuple(sorted(support_ids)),
        "source_resolved_deposition": tuple(source_audits),
        "plastic_strain_integral": strain_integral,
        "dislocation_density_increment_m2": rho_integral,
        "plastic_work_J_per_m": plastic_work,
        "crack_tip_mpz_updated": False,
    })


def promote_representation(growth: EmbryoGrowthState, cavity_id: str) -> EmbryoGrowthState:
    """Change representation only; all physical ownership remains identical."""
    return replace(
        growth, representation=RESOLVED, cavity_id=str(cavity_id),
        last_remesh_radius_m=growth.radius_m)


def advance_live_embryo_growth(
    live_state: Any, *, site_id: str, duration_s: float,
    transport_certificate: TransportFluxCertificate,
    transport_before_state: Any, transport_after_state: Any,
    bulk_accommodation_area_rate_m2_s: float, stress_tensor_Pa: np.ndarray,
    sigma_zz_Pa: Any, temperature_K: float, support_radius_m: float,
    local_mesh_size_m: float, remesh_fraction: float = 0.25,
    relaxation_area_rate_m2_s: float = 0.0,
    direct_interaction_area_rate_m2_s: float = 0.0,
    remesh_callback: Any = None, loop_emission_enabled: bool = True,
    forced_regime: str | None = None,
    forced_regimes_by_source: Mapping[str, str] | None = None,
    source_tensor_observer: Callable | None = None,
    equilibrium_callback: Callable | None = None, failure_injector: Callable | None = None,
    transport_packet_solver: Callable | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Atomically evolve the nested owner and deposit accepted plasticity.

    The transport owner must debit the raw absorbed inventory before this
    routine is called. Resolved growth that crosses the prospective remesh
    cadence fails closed unless the existing remesh callback is supplied.
    """
    voids = live_state.void_state
    before = next(item for item in voids.embryo_growth_states if item.site_id == site_id)
    if not isinstance(transport_certificate, TransportFluxCertificate):
        raise TypeError("scientific embryo growth requires a typed transport flux certificate")
    if transport_certificate.owner_id not in {site_id, before.cavity_id}:
        raise RuntimeError("typed transport certificate is not owned by this embryo")
    transport_certificate.verify(transport_before_state, transport_after_state)
    initial_rates, _ = cavity_surface_emission_rates(
        before, stress_tensor_Pa=stress_tensor_Pa, sigma_zz_Pa=sigma_zz_Pa,
        temperature_K=temperature_K)
    if not loop_emission_enabled or not before.cavity_emission_state.enabled:
        initial_rates = np.zeros_like(initial_rates)
    _, discrete_mask, action_increments = classify_cavity_emission_sources(
        before.cavity_emission_state, base_rates_s=initial_rates,
        duration_s=duration_s)
    # Keep the commit packet estimate identical to the scheduler preview:
    # exact sources are committed only by their first-passage owner, while
    # continuum sources contribute their state-owned (availability/recovery
    # limited) action rather than the undepleted attempt rate.
    continuum_action = float(np.sum(np.where(
        np.asarray(discrete_mask, dtype=bool), 0.0,
        np.asarray(action_increments, dtype=float))))
    expected_loop_area = continuum_action*float(
        loop_area_kinematics(before.cavity_emission_state.parameter_row,
                             before.radius_m)["selected_area_m2"])
    estimated_area = transport_certificate.raw_absorbed_area_m2+expected_loop_area
    area_cap = before.cavity_emission_state.parameter_row.maximum_packet_area_fraction*before.area_m2
    estimated_packet_count = max(1, int(math.ceil(estimated_area/max(area_cap, 1e-300))))
    if forced_regime == "LOW_COUNT_EXACT":
        estimated_packet_count = 1
    if estimated_packet_count > 1 and not callable(transport_packet_solver):
        raise RuntimeError(
            "multi-packet growth requires a current-domain transport packet solver")
    current = live_state
    if estimated_packet_count > 1:
        # The caller's full-interval state was only a prospective estimate.
        # Recompute transport from the accepted interval start on every
        # updated mechanical/domain state.
        current = replace(current, vacancy_transport_state_v2=transport_before_state)
    packet_audits = []
    remesh_any = False
    mechanically_null_packets = 0
    remaining = float(duration_s)
    trial_dt = remaining/estimated_packet_count
    total_inventory_debit = 0.0
    transport_recompute_count = 0
    while remaining > max(np.finfo(float).tiny, 8*np.spacing(duration_s)):
        packet_before = next(x for x in current.void_state.embryo_growth_states
                             if x.site_id == site_id)
        if callable(source_tensor_observer):
            tensors, zz = source_tensor_observer(current, packet_before)
        else:
            tensors, zz = stress_tensor_Pa, sigma_zz_Pa
        site = next(x for x in current.void_state.sites if x.site_id == site_id)
        centers = np.asarray(current.mesh.nodes)[np.asarray(current.mesh.elems)].mean(axis=1)
        framed_sources = []
        for source in packet_before.cavity_emission_state.sources:
            n = np.array((math.cos(source.orientation_rad), math.sin(source.orientation_rad)))
            t = np.array((-n[1], n[0]))
            xy = np.asarray(site.center_m)+packet_before.radius_m*n
            eid = int(np.argmin(np.linalg.norm(centers-xy, axis=1)))
            framed_sources.append(replace(
                source, boundary_position_m=tuple(map(float, xy)),
                local_normal=tuple(map(float, n)), local_tangent=tuple(map(float, t)),
                boundary_element_id=eid))
        packet_before = replace(packet_before, cavity_emission_state=replace(
            packet_before.cavity_emission_state, sources=tuple(framed_sources)))
        packet_dt = remaining if estimated_packet_count == 1 else min(trial_dt, remaining)
        if estimated_packet_count == 1:
            transport_live = current
            packet_transport_before = transport_before_state
            packet_transport_after = transport_after_state
            packet_certificate = transport_certificate
        else:
            solved = transport_packet_solver(current, packet_dt)
            if not isinstance(solved, tuple) or len(solved) != 3:
                raise RuntimeError(
                    "transport packet solver must return (live, before_state, after_state)")
            transport_live, packet_transport_before, packet_transport_after = solved
            owner_kind = ("ACTIVE_CAVITY" if packet_before.representation == RESOLVED
                          else "SUBGRID_SITE")
            owner_id = (packet_before.cavity_id if owner_kind == "ACTIVE_CAVITY"
                        else packet_before.site_id)
            packet_certificate = transport_flux_certificate(
                packet_transport_before, packet_transport_after,
                owner_id=owner_id, owner_kind=owner_kind)
            transport_recompute_count += 1
        packet_after, growth_audit = advance_embryo_growth(
            packet_before, duration_s=packet_dt,
            transport_certificate=packet_certificate,
            transport_before_state=packet_transport_before,
            transport_after_state=packet_transport_after,
            bulk_accommodation_area_rate_m2_s=bulk_accommodation_area_rate_m2_s,
            relaxation_area_rate_m2_s=relaxation_area_rate_m2_s,
            direct_interaction_area_rate_m2_s=direct_interaction_area_rate_m2_s,
            stress_tensor_Pa=tensors, sigma_zz_Pa=zz, temperature_K=temperature_K,
            loop_emission_enabled=loop_emission_enabled, forced_regime=forced_regime,
            forced_regimes_by_source=forced_regimes_by_source)
        packet_area_fraction = max(
            packet_after.area_m2-packet_before.area_m2, 0.0)/max(packet_before.area_m2, 1e-300)
        exact_indivisible = (forced_regime == "LOW_COUNT_EXACT" and
            sum(growth_audit["emission"]["accepted_counts"]) <= 1.0+1e-12)
        if (packet_area_fraction
                >packet_before.cavity_emission_state.parameter_row.maximum_packet_area_fraction
                  *(1.0+1e-10) and not exact_indivisible):
            if estimated_packet_count == 1:
                raise RuntimeError("physical growth packet exceeded the accepted area-fraction cap")
            trial_dt *= 0.5
            if trial_dt <= max(np.finfo(float).tiny, 8*np.spacing(duration_s)):
                raise RuntimeError("adaptive transport/growth packet bisection underflow")
            continue
        mechanical, deposit = deposit_emitted_loop_mechanics(
            transport_live, packet_before, packet_after, center_m=site.center_m,
            support_radius_m=support_radius_m, stress_tensor_Pa=tensors)
        packet_after = replace(
            packet_after,
            deposited_plastic_work_J_per_m=(packet_after.deposited_plastic_work_J_per_m
                +float(deposit.get("plastic_work_J_per_m", 0.0))),
            deposited_dislocation_density_integral=(
                packet_after.deposited_dislocation_density_integral
                +float(deposit.get("dislocation_density_increment_m2", 0.0))))
        cavities = list(mechanical.void_state.cavities)
        if packet_after.cavity_id is not None:
            ci = next(i for i, x in enumerate(cavities) if x.cavity_id == packet_after.cavity_id)
            cavity = cavities[ci]
            cavities[ci] = replace(
                cavity, radius_m=packet_after.radius_m, area_m2=packet_after.area_m2,
                inventory_area_m2=(cavity.inventory_area_m2
                    +packet_certificate.raw_absorbed_area_m2
                    +packet_after.dislocation_derived_area_m2
                    -packet_before.dislocation_derived_area_m2))
        owners = tuple(packet_after if x.site_id == site_id else x
                       for x in mechanical.void_state.embryo_growth_states)
        mechanical = replace(mechanical, void_state=replace(
            mechanical.void_state, embryo_growth_states=owners, cavities=tuple(cavities)))
        mechanics_changed = bool(
            float(deposit.get("deposited_loop_count", 0.0)) > 0.0
            or packet_after.radius_m != packet_before.radius_m
            or packet_after.area_m2 != packet_before.area_m2)
        if mechanics_changed:
            if failure_injector is not None:
                failure_injector("growth_packet_before_equilibrium", live_state)
            if equilibrium_callback is None:
                from .topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
                mechanical = equilibrate_fixed_load_with_production_fem(mechanical)
            else:
                mechanical = equilibrium_callback(mechanical)
        else:
            # Captured-but-unaccommodated inventory is stateful chemistry, not
            # a mechanics change.  Re-solving an identical FEM problem here
            # perturbs displacement at roundoff, invalidates the qualified
            # cavity-source binding, and freezes the next directional interval.
            mechanically_null_packets += 1
        remesh_required = (packet_before.representation == RESOLVED and
            abs(packet_after.radius_m-float(packet_before.last_remesh_radius_m))
            /max(float(local_mesh_size_m), 1e-300) >= float(remesh_fraction))
        if remesh_required:
            if not callable(remesh_callback):
                raise RuntimeError("resolved embryo growth reached remesh cadence without topology callback")
            mechanical = remesh_callback(mechanical, packet_before, packet_after)
            remesh_any = True
        if failure_injector is not None:
            failure_injector("growth_packet_committed", live_state)
        current = mechanical
        remaining = max(0.0, remaining-packet_dt)
        total_inventory_debit += packet_certificate.raw_absorbed_area_m2
        if estimated_packet_count > 1:
            trial_dt = min(2.0*trial_dt, remaining) if remaining > 0.0 else 0.0
        packet_audits.append({**dict(growth_audit), "mechanical_deposition": dict(deposit),
                              "applied_packet_area_fraction": packet_area_fraction,
                              "packet_duration_s": packet_dt,
                              "transport_before_fingerprint": packet_transport_before.fingerprint,
                              "transport_after_fingerprint": packet_transport_after.fingerprint,
                              "transport_recomputed_on_current_domain": estimated_packet_count > 1,
                              "indivisible_exact_event_exception": exact_indivisible,
                              "mechanics_changed": mechanics_changed,
                              "mechanical_equilibrium_performed": mechanics_changed})
    return current, MappingProxyType({
        "packet_count": len(packet_audits), "packets": tuple(packet_audits),
        "vacancy_inventory_debit_area_m2": total_inventory_debit,
        "transport_recompute_count": transport_recompute_count,
        "adaptive_packet_bisection": estimated_packet_count > 1,
        "remesh_required": remesh_any, "remesh_policy": "DELTA_R_OVER_H_LOCAL",
        "equilibrated_after_every_packet": mechanically_null_packets == 0,
        "equilibrated_after_every_mechanics_changing_packet": True,
        "mechanically_null_packets": mechanically_null_packets,
    })


def heal_embryo(growth: EmbryoGrowthState) -> tuple[EmbryoGrowthState, Mapping[str, Any]]:
    if growth.emitted_loop_count_effective > 0.0:
        return replace(growth, representation=RESIDUAL), MappingProxyType({
            "outcome": RESIDUAL, "pristine_available_site_allowed": False,
            "retained_emitted_loops": growth.emitted_loop_count_effective,
            "returned_vacancy_area_m2": 0.0,
        })
    returned = (growth.vacancy_derived_area_m2
                +growth.captured_unaccommodated_vacancy_area_m2)
    healed = replace(
        growth, radius_m=0.0, area_m2=0.0,
        last_remesh_radius_m=0.0,
        vacancy_derived_area_m2=0.0,
        captured_unaccommodated_vacancy_area_m2=0.0,
        vacancy_returned_or_annihilated_area_m2=(
            growth.vacancy_returned_or_annihilated_area_m2+returned),
        reversible_area_m2=(growth.initial_area_m2+growth.dislocation_derived_area_m2),
        representation="HEALED_AVAILABLE_SITE",
    )
    return healed, MappingProxyType({
        "outcome": "HEALED_AVAILABLE_SITE", "pristine_available_site_allowed": True,
        "retained_emitted_loops": 0.0, "returned_vacancy_area_m2": returned,
    })


__all__ = [
    "CavityEmissionParameterRowV1", "CavityEmissionSourceState",
    "CavityEmissionState", "EmbryoGrowthConfiguration", "EmbryoGrowthProcessOwner",
    "EmbryoGrowthState",
    "TransportFluxCertificate", "RESIDUAL", "RESOLVED",
    "SCHEMA", "SUBGRID", "advance_cavity_emission", "advance_embryo_growth",
    "advance_live_embryo_growth",
    "cavity_surface_emission_rates", "classify_cavity_emission_sources",
    "consume_resolved_cavity_loop_source",
    "deposit_emitted_loop_mechanics",
    "heal_embryo", "initialize_cavity_emission_state",
    "initialize_embryo_growth_state", "loop_area_kinematics",
    "make_cavity_loop_event_preview",
    "make_growth_radius_event_preview",
    "mechanism_resolved_growth_rates", "promote_representation",
    "CavityLoopSourceRecoveryQualificationError", "source_local_fem_tensors",
    "transport_flux_certificate",
]
