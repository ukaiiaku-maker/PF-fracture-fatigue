"""State-owned direct dislocation-interaction birth and growth mechanism.

The accepted bulk plasticity solve is the sole owner of plastic strain rate
and dislocation density.  This module only consumes those frozen fields.  Its
reference row is deliberately uncalibrated: the audited legacy MATLAB model
contained a dislocation-wall site class and a lumped density-enhanced vacancy
source, but no independently clocked direct-interaction mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .vacancy_transport_phase_field_v2 import EV_J, KB_J_PER_K, REFERENCE_UNCALIBRATED


SCHEMA = "unified-2d.dislocation-interaction-voiding/1"
MECHANISM_ID = "DISLOCATION_INTERACTION"


class FrozenInteractionMapping(dict):
    """Pickle-safe immutable mapping for accepted direct-mechanism audits."""
    def _immutable(self, *args, **kwargs):
        raise TypeError("accepted direct-interaction mapping is immutable")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __reduce__(self):
        return (_frozen_mapping, (dict(self),))


def _frozen_mapping(values):
    frozen = dict.__new__(FrozenInteractionMapping)
    dict.update(frozen, values)
    return frozen


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class DislocationInteractionParameterRowV1:
    row_id: str = "v1.direct-dislocation-interaction-reference-uncalibrated"
    interaction_geometry_factor_C_dd: float = 1.0e-20
    taylor_factor_M: float = 3.06
    unit_thickness_m: float = 1.0
    successful_direct_outcome_fraction: float = 0.10
    interaction_barrier_eV: float = 1.62
    stress_activation_volume_m3: float = 1.6e-29
    burgers_m: float = 2.74e-10
    embryo_reference_radius_m: float = 1.0e-9
    loop_swept_area_m2: float = 6.283185307179587e-9
    chi_loop: float = 1.0
    loop_projection: str = "REDUCED_CYLINDRICAL_2D"
    required_interaction_hits: int = 3
    interaction_attempt_law: str = "C_dd*rho_forest*M*abs(epdot_eq)/b"
    calibration_status: str = REFERENCE_UNCALIBRATED
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.calibration_status != REFERENCE_UNCALIBRATED:
            raise ValueError("direct interaction row must remain explicitly uncalibrated")
        positive = (
            self.interaction_geometry_factor_C_dd, self.taylor_factor_M,
            self.unit_thickness_m, self.interaction_barrier_eV,
            self.stress_activation_volume_m3, self.burgers_m,
            self.embryo_reference_radius_m, self.loop_swept_area_m2,
            self.chi_loop,
        )
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in positive):
            raise ValueError("direct interaction parameters must be finite and positive")
        fraction = float(self.successful_direct_outcome_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("direct interaction outcome fraction must lie in [0,1]")
        if int(self.required_interaction_hits) <= 0:
            raise ValueError("direct interaction requires at least one hit")
        if self.interaction_attempt_law != "C_dd*rho_forest*M*abs(epdot_eq)/b":
            raise ValueError("direct interaction attempt law is frozen for V1")
        if self.loop_projection not in {"FULL_3D_LOOP", "REDUCED_CYLINDRICAL_2D"}:
            raise ValueError("unsupported direct-interaction loop projection")

    @property
    def direct_free_volume_per_hit_area_m2(self) -> float:
        """Compatibility diagnostic derived from Burgers/loop geometry."""
        if self.loop_projection == "FULL_3D_LOOP":
            return self.burgers_m*self.loop_swept_area_m2/self.unit_thickness_m
        return self.chi_loop*2.0*math.pi*self.embryo_reference_radius_m*self.burgers_m

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class DirectInteractionSiteStateV1:
    site_id: str
    center_xy_m: tuple[float, float]
    support_radius_m: float
    interaction_action: float
    interaction_thresholds: tuple[float, ...]
    interaction_hit_count: int
    required_interaction_hits: int
    direct_free_volume_inventory_area_m2: float
    direct_free_volume_per_hit_area_m2: float
    interaction_barrier_eV: float
    interaction_rng_identity: str
    interaction_event_history: tuple[Mapping[str, Any], ...] = ()
    birth_inventory_transferred_area_m2: float = 0.0

    def __post_init__(self) -> None:
        if not self.site_id or not self.interaction_rng_identity:
            raise ValueError("direct interaction site requires identities")
        if not math.isfinite(self.support_radius_m) or self.support_radius_m <= 0.0:
            raise ValueError("direct interaction support radius must be positive")
        if self.required_interaction_hits <= 0:
            raise ValueError("direct interaction hit count must be positive")
        if len(self.interaction_thresholds) != self.required_interaction_hits:
            raise ValueError("direct interaction thresholds are incomplete")
        if any(not math.isfinite(v) or v <= 0.0 for v in self.interaction_thresholds):
            raise ValueError("direct interaction thresholds must be finite and positive")
        if any(b <= a for a, b in zip(self.interaction_thresholds,
                                      self.interaction_thresholds[1:])):
            raise ValueError("direct interaction thresholds must be cumulative and increasing")
        if not 0 <= self.interaction_hit_count <= self.required_interaction_hits:
            raise ValueError("direct interaction hit count is invalid")
        if not math.isfinite(self.interaction_action) or self.interaction_action < 0.0:
            raise ValueError("direct interaction action must be finite and nonnegative")
        if self.birth_inventory_transferred_area_m2 < 0.0:
            raise ValueError("direct birth transfer must be nonnegative")
        expected = (self.interaction_hit_count*self.direct_free_volume_per_hit_area_m2
                    -self.birth_inventory_transferred_area_m2)
        if not math.isclose(self.direct_free_volume_inventory_area_m2, expected,
                            rel_tol=1.0e-12, abs_tol=1.0e-30):
            raise ValueError("direct free-volume inventory does not close to committed hits")

    def next_threshold(self) -> float:
        if self.interaction_hit_count >= self.required_interaction_hits:
            return math.inf
        return float(self.interaction_thresholds[self.interaction_hit_count])


@dataclass(frozen=True)
class DislocationInteractionStateV1:
    parameter_row: DislocationInteractionParameterRowV1
    sites: tuple[DirectInteractionSiteStateV1, ...]
    enabled: bool = False
    total_direct_free_volume_inventory_area_m2: float = 0.0
    total_direct_free_volume_generated_area_m2: float = 0.0
    direct_free_volume_transferred_to_cavities_area_m2: float = 0.0
    direct_birth_inventory_transferred_to_embryos_area_m2: float = 0.0
    total_successful_interaction_hits: int = 0
    event_history: tuple[Mapping[str, Any], ...] = ()
    last_opportunity_audit: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("direct interaction state schema mismatch")
        generated = float(self.total_direct_free_volume_generated_area_m2)
        transferred = float(self.direct_free_volume_transferred_to_cavities_area_m2)
        if generated < 0.0 or transferred < 0.0 or transferred > generated+1.0e-30:
            raise ValueError("direct free-volume generation/transfer ledger is invalid")
        inventory = math.fsum(s.direct_free_volume_inventory_area_m2 for s in self.sites) \
            + generated-transferred
        hits = sum(s.interaction_hit_count for s in self.sites)
        if not math.isclose(inventory, self.total_direct_free_volume_inventory_area_m2,
                            rel_tol=1.0e-12, abs_tol=1.0e-30):
            raise ValueError("aggregate direct free-volume ledger is inconsistent")
        if hits != self.total_successful_interaction_hits:
            raise ValueError("aggregate direct interaction hit ledger is inconsistent")
        if self.direct_birth_inventory_transferred_to_embryos_area_m2 < 0.0:
            raise ValueError("direct birth transfer ledger is invalid")
        object.__setattr__(self, "last_opportunity_audit",
                           _frozen_mapping(dict(self.last_opportunity_audit)))

    @property
    def fingerprint(self) -> str:
        def encode(value: Any):
            if hasattr(value, "__dataclass_fields__"):
                return {name: encode(getattr(value, name)) for name in value.__dataclass_fields__}
            if isinstance(value, Mapping):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value
        return _fingerprint(encode(self))


def initialize_dislocation_interaction_state(
    production_sites: Sequence[Any], *, parameter_row: DislocationInteractionParameterRowV1 | None = None,
    stochastic: bool = False, seed: int = 49157, enabled: bool = False,
    support_radius_m: float = 5.0e-5,
) -> DislocationInteractionStateV1:
    row = parameter_row or DislocationInteractionParameterRowV1()
    rng = np.random.default_rng(int(seed))
    sites = []
    for index, site in enumerate(production_sites):
        increments = (rng.exponential(size=row.required_interaction_hits)
                      if stochastic else np.ones(row.required_interaction_hits))
        thresholds = tuple(float(v) for v in np.cumsum(increments))
        rng_identity = _fingerprint({
            "seed": int(seed), "site_id": site.site_id, "stream_index": index,
            "bit_generator": rng.bit_generator.__class__.__name__,
        })
        sites.append(DirectInteractionSiteStateV1(
            site_id=site.site_id, center_xy_m=tuple(map(float, site.center_m)),
            support_radius_m=float(support_radius_m), interaction_action=0.0,
            interaction_thresholds=thresholds, interaction_hit_count=0,
            required_interaction_hits=row.required_interaction_hits,
            direct_free_volume_inventory_area_m2=0.0,
            direct_free_volume_per_hit_area_m2=row.direct_free_volume_per_hit_area_m2,
            interaction_barrier_eV=row.interaction_barrier_eV,
            interaction_rng_identity=rng_identity,
        ))
    return DislocationInteractionStateV1(parameter_row=row, sites=tuple(sites), enabled=enabled)


def validate_interaction_outcome_partition(transport_row: Any,
                                           direct_row: DislocationInteractionParameterRowV1) -> Mapping[str, float | str]:
    """Bind the direct clock to the same prospective population partition."""
    mobile = float(transport_row.interaction_mobile_vacancy_fraction)
    direct = float(transport_row.interaction_direct_cavitation_fraction)
    benign = float(transport_row.interaction_benign_fraction)
    total = math.fsum((mobile, direct, benign))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("interaction outcome partition does not close")
    if not math.isclose(direct, direct_row.successful_direct_outcome_fraction,
                        rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("transport and direct owners claim different interaction populations")
    return MappingProxyType({
        "mobile_vacancy_fraction": mobile, "direct_cavitation_fraction": direct,
        "benign_hardening_recovery_fraction": benign, "sum": total,
        "calibration_status": REFERENCE_UNCALIBRATED,
    })


def accepted_interaction_opportunity(
    state: Any, site: DirectInteractionSiteStateV1, *, temperature_K: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray | None = None,
    accepted_forest_density_m2: np.ndarray | None = None,
    accepted_mobile_density_m2: np.ndarray | None = None,
    accepted_mean_stress_Pa: np.ndarray | None = None,
) -> Mapping[str, Any]:
    """Integrate ``C_dd rho_forest M |epdot|/b`` over site support volume."""
    owned = state.junction_process_state.get("accepted_bulk_plasticity_v104", {})
    constitutive = owned.get("constitutive", {})
    epdot = np.asarray(
        owned.get("dot_ep_gp_s") if accepted_equivalent_plastic_strain_rate_s is None
        else accepted_equivalent_plastic_strain_rate_s, dtype=float,
    )
    forest = np.asarray(
        state.rho_gp if accepted_forest_density_m2 is None else accepted_forest_density_m2,
        dtype=float,
    )
    mobile = np.asarray(
        constitutive.get("pt_rho_mobile_gp", np.zeros_like(forest))
        if accepted_mobile_density_m2 is None else accepted_mobile_density_m2,
        dtype=float,
    )
    if accepted_mean_stress_Pa is None:
        from .vacancy_transport_phase_field_v2 import accepted_elastoplastic_stress_3d
        mean_stress = np.asarray(accepted_elastoplastic_stress_3d(state)["mean_stress_Pa"], float)
    else:
        mean_stress = np.asarray(accepted_mean_stress_Pa, float)
    ne = state.mesh.ne
    if any(a.shape != (ne,) for a in (epdot, forest, mobile, mean_stress)):
        raise ValueError("accepted direct-interaction fields must be mechanical P0 arrays")
    if np.any(epdot < 0.0) or np.any(forest < 0.0) or np.any(mobile < 0.0):
        raise ValueError("accepted direct-interaction fields must be nonnegative")
    row = state.void_state.dislocation_interaction_state.parameter_row
    from .unified_fracture_material_v5 import require_bound_identity
    material_identity = dict(require_bound_identity(state))
    centers = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    support = np.linalg.norm(centers-np.asarray(site.center_xy_m), axis=1) <= site.support_radius_m
    if not np.any(support):
        raise RuntimeError("direct-interaction site support is unresolved")
    b = float(state.material.b)
    gamma = (row.interaction_geometry_factor_C_dd*forest*row.taylor_factor_M
             * np.abs(epdot)/b)
    volume = np.asarray(state.mesh.area_e)*row.unit_thickness_m
    opportunity = float(np.sum(gamma[support]*volume[support]))
    weights = volume[support]*gamma[support]
    denom = float(np.sum(weights))
    local_mean_stress = (float(np.sum(weights*mean_stress[support])/denom)
                         if denom > 0.0 else float(np.mean(mean_stress[support])))
    stress_work = max(local_mean_stress, 0.0)*row.stress_activation_volume_m3
    thermal = KB_J_PER_K*float(temperature_K)
    success = math.exp(-max(row.interaction_barrier_eV*EV_J-stress_work, 0.0)/thermal)
    successful_rate = opportunity*row.successful_direct_outcome_fraction*success
    support_area = float(np.sum(np.asarray(state.mesh.area_e)[support]))
    avg = lambda values: float(np.sum(np.asarray(values)[support]*np.asarray(state.mesh.area_e)[support])
                               / support_area)
    return MappingProxyType({
        "mechanism_id": MECHANISM_ID,
        "parameter_row_id": row.row_id,
        "parameter_row_fingerprint": row.fingerprint,
        "material_bundle_id": material_identity["material_bundle_id"],
        "material_bundle_sha256": material_identity["material_bundle_sha256"],
        "plasticity_fingerprint": material_identity["plasticity_fingerprint"],
        "site_id": site.site_id,
        "support_radius_m": site.support_radius_m,
        "support_area_m2": support_area,
        "support_volume_m3": support_area*row.unit_thickness_m,
        "rho_mobile_support_average_m2": avg(mobile),
        "rho_forest_support_average_m2": avg(forest),
        "accepted_epdot_eq_support_average_s": avg(epdot),
        "Gamma_dd_support_average_m-3_s-1": avg(gamma),
        "support_integrated_interaction_rate_s": opportunity,
        "temperature_K": float(temperature_K),
        "mean_stress_support_Pa": local_mean_stress,
        "stress_work_J": stress_work,
        "thermal_success_probability": success,
        "successful_direct_interaction_rate_s": successful_rate,
        "opportunity_formula": row.interaction_attempt_law,
        "plasticity_update_inside_mechanism": False,
        "calibration_status": REFERENCE_UNCALIBRATED,
    })


def preview_direct_interaction_hit(state: Any, *, maximum_duration_s: float,
                                   temperature_K: float, **accepted_fields: Any):
    """Return the earliest direct hit without changing the accepted state."""
    owner = state.void_state.dislocation_interaction_state
    if owner is None or not owner.enabled:
        return None, None
    maximum = float(maximum_duration_s)
    winner = None
    for site in owner.sites:
        production = next(p for p in state.void_state.sites if p.site_id == site.site_id)
        if site.interaction_hit_count >= site.required_interaction_hits or production.phase.value != "AVAILABLE_SITE":
            continue
        audit = accepted_interaction_opportunity(
            state, site, temperature_K=temperature_K, **accepted_fields,
        )
        rate = float(audit["successful_direct_interaction_rate_s"])
        crossing = (math.inf if rate <= 0.0 else
                    max(site.next_threshold()-site.interaction_action, 0.0)/rate)
        candidate = (crossing, site.site_id, audit)
        if winner is None or candidate[:2] < winner[:2]:
            winner = candidate
    if winner is None or not math.isfinite(winner[0]) or winner[0] > maximum:
        return None, None if winner is None else winner[2]
    return float(winner[0]), winner[2]


def advance_dislocation_interaction(
    state: DislocationInteractionStateV1, *, site_id: str, elapsed_time_s: float,
    opportunity_audit: Mapping[str, Any], commit_hit: bool = False,
) -> DislocationInteractionStateV1:
    """Advance one site on common time and optionally commit exactly one hit."""
    elapsed = float(elapsed_time_s)
    if elapsed < 0.0 or not math.isfinite(elapsed):
        raise ValueError("direct interaction elapsed time must be finite and nonnegative")
    sites = list(state.sites)
    index = next(i for i, item in enumerate(sites) if item.site_id == site_id)
    site = sites[index]
    rate = float(opportunity_audit["successful_direct_interaction_rate_s"])
    action = min(site.interaction_action+rate*elapsed, site.next_threshold())
    history = site.interaction_event_history + ({
        "event": "ACTION_ADVANCE", "elapsed_time_s": elapsed,
        "action_before": site.interaction_action, "action_after": action,
        "rate_s": rate, "opportunity": dict(opportunity_audit),
    },)
    hit_count = site.interaction_hit_count
    inventory = site.direct_free_volume_inventory_area_m2
    aggregate_history = state.event_history
    if commit_hit:
        tolerance = 32.0*np.finfo(float).eps*max(abs(site.next_threshold()), 1.0)
        if action+tolerance < site.next_threshold():
            raise RuntimeError("direct interaction hit committed before its threshold")
        if hit_count >= site.required_interaction_hits:
            raise RuntimeError("direct interaction site already completed")
        hit_count += 1
        inventory += site.direct_free_volume_per_hit_area_m2
        event = {
            "event": "DIRECT_INTERACTION_HIT", "site_id": site_id,
            "hit_count": hit_count, "action": action,
            "threshold": site.next_threshold(),
            "free_volume_increment_area_m2": site.direct_free_volume_per_hit_area_m2,
            "delta_volume_loop_m3": (
                state.parameter_row.burgers_m*state.parameter_row.loop_swept_area_m2
            ),
            "full_3d_projected_area_m2": (
                state.parameter_row.burgers_m*state.parameter_row.loop_swept_area_m2
                /state.parameter_row.unit_thickness_m
            ),
            "selected_loop_projection": state.parameter_row.loop_projection,
            "chi_loop": state.parameter_row.chi_loop,
            "calibration_status": state.parameter_row.calibration_status,
            "mechanism_id": MECHANISM_ID,
        }
        history += (event,); aggregate_history += (event,)
    sites[index] = replace(
        site, interaction_action=action, interaction_hit_count=hit_count,
        direct_free_volume_inventory_area_m2=inventory,
        interaction_event_history=history,
    )
    return replace(
        state, sites=tuple(sites),
        total_direct_free_volume_inventory_area_m2=math.fsum(
            item.direct_free_volume_inventory_area_m2 for item in sites
        ) + state.total_direct_free_volume_generated_area_m2 \
            - state.direct_free_volume_transferred_to_cavities_area_m2,
        total_successful_interaction_hits=sum(item.interaction_hit_count for item in sites),
        event_history=aggregate_history,
        last_opportunity_audit=dict(opportunity_audit),
    )


def record_direct_cavity_growth_transfer(
    state: DislocationInteractionStateV1, area_m2: float,
) -> DislocationInteractionStateV1:
    """Atomically generate and transfer direct free volume into a cavity."""
    area = float(area_m2)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("direct cavity-growth transfer must be finite and nonnegative")
    if area == 0.0:
        return state
    return replace(
        state,
        total_direct_free_volume_generated_area_m2=(
            state.total_direct_free_volume_generated_area_m2+area
        ),
        direct_free_volume_transferred_to_cavities_area_m2=(
            state.direct_free_volume_transferred_to_cavities_area_m2+area
        ),
        event_history=state.event_history+({
            "event": "DIRECT_FREE_VOLUME_TRANSFER_TO_CAVITY",
            "area_m2": area, "mechanism_id": MECHANISM_ID,
        },),
    )


def combined_cavity_growth_rate(
    *, vacancy_supply_rate_area_m2_s: float, direct_supply_rate_area_m2_s: float,
    radius_m: float, equivalent_plastic_strain_rate_s: float,
    plastic_accommodation_coefficient: float,
) -> Mapping[str, Any]:
    """Apply accommodation only to vacancy supply; add loop growth directly."""
    vacancy = float(vacancy_supply_rate_area_m2_s)
    direct = float(direct_supply_rate_area_m2_s)
    if vacancy < 0.0 or direct < 0.0:
        raise ValueError("cavity free-volume supplies must be nonnegative")
    from .vacancy_transport_phase_field_v2 import cavity_growth_rate_from_transport
    result = dict(cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=vacancy,
        radius_m=radius_m,
        equivalent_plastic_strain_rate_s=equivalent_plastic_strain_rate_s,
        plastic_accommodation_coefficient=plastic_accommodation_coefficient,
    ))
    vacancy_area_rate = 2.0*math.pi*float(radius_m)*float(result["Rdot_m_s"])
    total_area_rate = vacancy_area_rate+direct
    result.update({
        "vacancy_supply_rate_area_m2_s": vacancy,
        "direct_interaction_supply_rate_area_m2_s": direct,
        "total_supply_rate_area_m2_s": vacancy+direct,
        "vacancy_accommodated_rate_area_m2_s": vacancy_area_rate,
        "dislocation_direct_growth_rate_area_m2_s": direct,
        "total_growth_rate_area_m2_s": total_area_rate,
        "Rdot_m_s": total_area_rate/(2.0*math.pi*float(radius_m)),
        "growth_interpretation": "VACANCY_SERIES_LIMITED_PLUS_DIRECT_LOOP_AREA",
    })
    return MappingProxyType(result)


def direct_free_volume_supply_rate_area_m2_s(
    opportunity_audit: Mapping[str, Any], row: DislocationInteractionParameterRowV1,
) -> float:
    """Convert successful accepted interaction events to free-volume supply."""
    rate = float(opportunity_audit["successful_direct_interaction_rate_s"])
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("successful direct interaction rate is invalid")
    return rate*float(row.direct_free_volume_per_hit_area_m2)


def equivalent_population_birth_hazard(
    *, vacancy_clustering_action_rate_s: float,
    direct_interaction_action_rate_s: float,
) -> Mapping[str, float]:
    """Report the additive population hazard without merging owner clocks."""
    vacancy = float(vacancy_clustering_action_rate_s)
    direct = float(direct_interaction_action_rate_s)
    if any(not math.isfinite(v) or v < 0.0 for v in (vacancy, direct)):
        raise ValueError("birth action rates must be finite and nonnegative")
    return MappingProxyType({
        "h_vacancy_birth_s": vacancy,
        "h_dislocation_birth_s": direct,
        "h_birth_total_s": vacancy+direct,
    })


def make_physical_event_preview(
    state: Any, *, maximum_duration_s: float, temperature_K: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
):
    """Adapt the direct owner to the common scheduler preview schema."""
    from .global_physical_scheduler_v1 import (
        PhysicalEventPreview, _accepted_fingerprint, _rng_fingerprint,
        _sha, _topology_fingerprint,
    )
    direct = state.void_state.dislocation_interaction_state
    crossing, audit = preview_direct_interaction_hit(
        state, maximum_duration_s=maximum_duration_s,
        temperature_K=temperature_K,
        accepted_equivalent_plastic_strain_rate_s=(
            accepted_equivalent_plastic_strain_rate_s
        ),
        accepted_forest_density_m2=np.asarray(state.rho_gp),
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
    )
    site = None if crossing is None or audit is None else next(
        item for item in direct.sites if item.site_id == audit["site_id"]
    )
    return PhysicalEventPreview(
        event_type=("DISLOCATION_INTERACTION_HIT" if crossing is not None
                    else "NO_DISLOCATION_INTERACTION_HIT"),
        owner_kind="DislocationInteractionStateV1", owner_id="direct-interaction",
        crossing_time_s=math.inf if crossing is None else crossing,
        accepted_state_fingerprint=_accepted_fingerprint(state),
        source_fingerprint=_sha((direct.parameter_row.fingerprint,
                                 None if audit is None else dict(audit))),
        topology_fingerprint=_topology_fingerprint(state),
        event_ids=(() if site is None else
                   (f"direct-interaction-hit:{site.site_id}:{site.interaction_hit_count+1}",)),
        candidate_ids=() if site is None else (site.site_id,),
        threshold_identity=(() if site is None else ((
            site.site_id, site.next_threshold(), site.interaction_action,
            site.interaction_rng_identity,
        ),)),
        rng_fingerprint=_rng_fingerprint(state),
        commit_payload=MappingProxyType({
            "site_id": None if site is None else site.site_id,
            "opportunity_audit": None if audit is None else dict(audit),
        }),
    )


def transfer_direct_birth_inventory(
    state: DislocationInteractionStateV1, site_id: str,
) -> tuple[DislocationInteractionStateV1, float, int]:
    """Atomically remove a site's birth inventory from the direct source owner."""
    sites = list(state.sites)
    index = next(i for i, site in enumerate(sites) if site.site_id == site_id)
    site = sites[index]
    amount = float(site.direct_free_volume_inventory_area_m2)
    sites[index] = replace(
        site, direct_free_volume_inventory_area_m2=0.0,
        birth_inventory_transferred_area_m2=(site.birth_inventory_transferred_area_m2+amount),
    )
    return replace(
        state, sites=tuple(sites),
        total_direct_free_volume_inventory_area_m2=(
            state.total_direct_free_volume_inventory_area_m2-amount),
        direct_birth_inventory_transferred_to_embryos_area_m2=(
            state.direct_birth_inventory_transferred_to_embryos_area_m2+amount),
    ), amount, int(site.interaction_hit_count)


def advance_live_direct_owner(
    accepted_mechanical: Any, state: Any, *, elapsed_time_s: float,
    temperature_K: float, accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray, winning_site_id: str | None,
):
    """Advance every direct site once and commit at most the selected hit."""
    from .voiding_v5 import VoidPhase
    direct = state.void_state.dislocation_interaction_state
    if direct is None or not direct.enabled:
        return state
    evaluation = replace(accepted_mechanical, void_state=replace(
        state.void_state, dislocation_interaction_state=direct,
    ))
    for direct_site in direct.sites:
        production_site = next(item for item in state.void_state.sites
                               if item.site_id == direct_site.site_id)
        if (production_site.phase != VoidPhase.AVAILABLE_SITE
                or direct_site.interaction_hit_count >= direct_site.required_interaction_hits):
            continue
        audit = accepted_interaction_opportunity(
            evaluation, direct_site, temperature_K=temperature_K,
            accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
            accepted_forest_density_m2=np.asarray(accepted_mechanical.rho_gp),
            accepted_mean_stress_Pa=accepted_mean_stress_Pa,
        )
        direct = advance_dislocation_interaction(
            direct, site_id=direct_site.site_id, elapsed_time_s=elapsed_time_s,
            opportunity_audit=audit,
            commit_hit=(direct_site.site_id == winning_site_id),
        )
        evaluation = replace(evaluation, void_state=replace(
            evaluation.void_state, dislocation_interaction_state=direct,
        ))
    production_sites = list(state.void_state.sites)
    history = state.void_state.event_history
    if winning_site_id is not None:
        direct_site = next(item for item in direct.sites if item.site_id == winning_site_id)
        index = next(i for i, item in enumerate(production_sites)
                     if item.site_id == winning_site_id)
        production_site = production_sites[index]
        if production_site.phase != VoidPhase.AVAILABLE_SITE:
            raise RuntimeError("a second mechanism already created this embryo")
        completed = direct_site.interaction_hit_count >= direct_site.required_interaction_hits
        if completed:
            production_sites[index] = replace(production_site, phase=VoidPhase.EMBRYO)
            if not any(item.site_id == winning_site_id
                       for item in state.void_state.embryo_growth_states):
                transport_site = next(item for item in state.vacancy_transport_state_v2.site_states
                                      if item.site_id == winning_site_id)
                from .global_physical_scheduler_v1 import _ensure_vacancy_embryo_growth_owner
                staged_void = replace(state.void_state,
                                      sites=tuple(production_sites),
                                      dislocation_interaction_state=direct)
                staged_void, staged_transport = _ensure_vacancy_embryo_growth_owner(
                    staged_void, state.vacancy_transport_state_v2,
                    production_sites[index], transport_site)
                state = replace(state, void_state=staged_void,
                                vacancy_transport_state_v2=staged_transport)
                direct = staged_void.dislocation_interaction_state
        transport_site = next((item for item in state.vacancy_transport_state_v2.site_states
                               if item.site_id == winning_site_id), None)
        history = history + ({
            "event": "EMBRYO" if completed else "DIRECT_INTERACTION_HIT",
            "site_id": winning_site_id,
            "owner": "DislocationInteractionStateV1",
            "winning_mechanism": MECHANISM_ID,
            "vacancy_contribution_area_m2": (0.0 if transport_site is None else
                                                transport_site.reserved_inventory_area_m2),
            "direct_interaction_contribution_area_m2": (
                direct_site.direct_free_volume_inventory_area_m2
            ),
            "vacancy_action": None if transport_site is None else transport_site.birth_action,
            "vacancy_threshold": None if transport_site is None else transport_site.next_birth_threshold(),
            "direct_action": direct_site.interaction_action,
            "direct_threshold": direct_site.next_threshold(),
        },)
    return replace(state, void_state=replace(
        state.void_state, sites=tuple(production_sites), event_history=history,
        dislocation_interaction_state=direct,
    ))


__all__ = [
    "DirectInteractionSiteStateV1", "DislocationInteractionParameterRowV1",
    "DislocationInteractionStateV1", "MECHANISM_ID", "SCHEMA",
    "accepted_interaction_opportunity", "advance_dislocation_interaction",
    "advance_live_direct_owner",
    "combined_cavity_growth_rate", "initialize_dislocation_interaction_state",
    "direct_free_volume_supply_rate_area_m2_s",
    "equivalent_population_birth_hazard",
    "make_physical_event_preview", "preview_direct_interaction_hit",
    "record_direct_cavity_growth_transfer",
    "transfer_direct_birth_inventory",
    "validate_interaction_outcome_partition",
]
