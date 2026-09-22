"""State-owned plasticity/vacancy closure for the unified one-void model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq


SCHEMA = "unified-2d.plasticity-vacancy-coupling/1"
COMMON_VACANCY_ROW_ID = "v1.common-tungsten-vacancy-reference-uncalibrated/1"
KB_J_PER_K = 1.380649e-23
EV_J = 1.602176634e-19
ENGINEERING_SHEAR_VOIGT_CONVENTION = "ep_gp=(epsilon_xx,epsilon_yy,gamma_xy=2epsilon_xy)"
CONSERVATION_ABSOLUTE_FLOOR_M2 = 1.0e-24
DILUTE_VACANCY_FRACTION_LIMIT = 1.0e-2


@dataclass(frozen=True)
class VacancyParameterRow:
    row_id: str = COMMON_VACANCY_ROW_ID
    vacancy_formation_energy_eV: float = 3.2
    vacancy_formation_entropy_over_kB: float = 0.0
    vacancy_migration_energy_eV: float = 1.7
    vacancy_diffusivity_prefactor_m2_s: float = 1.0e-6
    plastic_vacancy_coefficient_per_strain: float = 3.0e-5
    dislocation_production_boost: float = 0.20
    dislocation_sink_efficiency: float = 1.0
    capture_factor: float = 1.0
    calibration_status: str = "REFERENCE_UNCALIBRATED"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        values = tuple(
            float(value) for name, value in asdict(self).items()
            if name not in ("row_id", "calibration_status", "schema")
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("vacancy reference values must be finite and nonnegative")
        if self.row_id != COMMON_VACANCY_ROW_ID or self.calibration_status != "REFERENCE_UNCALIBRATED":
            raise ValueError("the common vacancy row identity/status is frozen")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()


REFERENCE_VACANCY_ROW = VacancyParameterRow()


def equilibrium_vacancy_fraction(temperature_K: float, row: VacancyParameterRow = REFERENCE_VACANCY_ROW) -> float:
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    exponent = (
        float(row.vacancy_formation_entropy_over_kB)
        - float(row.vacancy_formation_energy_eV) * EV_J / (KB_J_PER_K * temperature)
    )
    return math.exp(exponent)


def vacancy_diffusivity_m2_s(temperature_K: float, row: VacancyParameterRow = REFERENCE_VACANCY_ROW) -> float:
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return float(row.vacancy_diffusivity_prefactor_m2_s) * math.exp(
        -float(row.vacancy_migration_energy_eV) * EV_J / (KB_J_PER_K * temperature)
    )


def equivalent_plastic_strain_rate_s(
    previous_ep_gp: np.ndarray, accepted_ep_gp: np.ndarray, duration_s: float,
) -> np.ndarray:
    """Equivalent rate from two accepted P0 plastic-strain fields."""
    duration = float(duration_s)
    previous = np.asarray(previous_ep_gp, dtype=float)
    accepted = np.asarray(accepted_ep_gp, dtype=float)
    if previous.shape != accepted.shape or previous.ndim != 2 or previous.shape[0] != 3:
        raise ValueError("plastic strain fields must have matching (3, ne) shape")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("accepted plastic interval must be finite and positive")
    increment = (accepted - previous) / duration
    exx, eyy, gamma_xy = increment
    ezz = -(exx + eyy)
    # FEM B matrices store engineering shear gamma_xy.  The tensor invariant
    # therefore contains 2*(gamma_xy/2)^2 = gamma_xy^2/2.
    rate = np.sqrt(np.maximum(
        2.0 / 3.0 * (exx**2 + eyy**2 + ezz**2 + 0.5 * gamma_xy**2), 0.0,
    ))
    return rate


def conservation_roundoff_budget_m2(*values: float) -> float:
    scale = max((abs(float(value)) for value in values), default=0.0)
    return max(CONSERVATION_ABSOLUTE_FLOOR_M2, 64.0 * np.finfo(float).eps * scale)


def vacancy_production_rate_s(
    equivalent_plastic_rate_s: np.ndarray | float,
    dislocation_density_m2: np.ndarray | float,
    *,
    rho_reference_m2: float,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> np.ndarray:
    rate = np.asarray(equivalent_plastic_rate_s, dtype=float)
    rho = np.asarray(dislocation_density_m2, dtype=float)
    reference = float(rho_reference_m2)
    if reference <= 0.0 or not math.isfinite(reference):
        raise ValueError("rho reference must be finite and positive")
    if np.any(~np.isfinite(rate)) or np.any(rate < 0.0) or np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("plastic rate and dislocation density must be finite and nonnegative")
    return float(row.plastic_vacancy_coefficient_per_strain) * rate * (
        1.0 + float(row.dislocation_production_boost) * np.sqrt(rho / reference)
    )


def vacancy_annihilation_rate_s(
    temperature_K: float,
    dislocation_density_m2: np.ndarray | float,
    *,
    process_zone_length_m: float,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> np.ndarray:
    rho = np.asarray(dislocation_density_m2, dtype=float)
    length = float(process_zone_length_m)
    if length <= 0.0 or not math.isfinite(length) or np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("sink geometry and density must be finite and nonnegative")
    return vacancy_diffusivity_m2_s(temperature_K, row) * (
        float(row.dislocation_sink_efficiency) * rho + 1.0 / length**2
    )


def exact_vacancy_interval(
    vacancy_fraction: np.ndarray,
    *,
    element_area_m2: np.ndarray,
    production_rate_s: np.ndarray | float,
    annihilation_rate_s: np.ndarray | float,
    equilibrium_fraction: float,
    duration_s: float,
) -> dict[str, Any]:
    """Exact constant-field solution and separately integrated source/sink terms."""
    c0 = np.asarray(vacancy_fraction, dtype=float)
    area = np.asarray(element_area_m2, dtype=float)
    production = np.broadcast_to(np.asarray(production_rate_s, dtype=float), c0.shape)
    sink = np.broadcast_to(np.asarray(annihilation_rate_s, dtype=float), c0.shape)
    equilibrium = float(equilibrium_fraction)
    duration = float(duration_s)
    if c0.ndim != 1 or area.shape != c0.shape:
        raise ValueError("vacancy field and element areas must have one value per element")
    if (
        np.any(~np.isfinite(c0)) or np.any(c0 < 0.0)
        or np.any(~np.isfinite(area)) or np.any(area <= 0.0)
        or np.any(~np.isfinite(production)) or np.any(production < 0.0)
        or np.any(~np.isfinite(sink)) or np.any(sink < 0.0)
        or not math.isfinite(equilibrium) or equilibrium < 0.0
        or not math.isfinite(duration) or duration < 0.0
    ):
        raise ValueError("vacancy interval inputs must be finite and nonnegative")
    active = sink > 0.0
    c1 = np.empty_like(c0)
    integral_c_dt = np.empty_like(c0)
    c_ss = np.empty_like(c0)
    c_ss[active] = equilibrium + production[active] / sink[active]
    decay = np.exp(-sink[active] * duration)
    c1[active] = c_ss[active] + (c0[active] - c_ss[active]) * decay
    integral_c_dt[active] = (
        c_ss[active] * duration
        + (c0[active] - c_ss[active]) * (1.0 - decay) / sink[active]
    )
    c1[~active] = c0[~active] + production[~active] * duration
    integral_c_dt[~active] = c0[~active] * duration + 0.5 * production[~active] * duration**2
    c1 = np.maximum(c1, 0.0)
    plastic_area = math.fsum(float(value) for value in area * production * duration)
    thermal_area = math.fsum(float(value) for value in area * sink * equilibrium * duration)
    raw_annihilated_area = math.fsum(float(value) for value in area * sink * integral_c_dt)
    inventory_change = math.fsum(float(value) for value in area * (c1 - c0))
    # The exact interval ODE owns this identity.  Book its algebraic sink
    # value and retain the independently integrated quadrature as a diagnostic
    # so summation roundoff cannot accumulate in accepted-state conservation.
    annihilated_area = plastic_area + thermal_area - inventory_change
    residual = math.fsum((plastic_area, thermal_area, -annihilated_area, -inventory_change))
    return {
        "vacancy_fraction_gp": c1,
        "vacancy_plastic_production_area_m2": plastic_area,
        "vacancy_thermal_exchange_area_m2": thermal_area,
        "vacancy_annihilation_area_m2": annihilated_area,
        "raw_integrated_annihilation_area_m2": raw_annihilated_area,
        "annihilation_quadrature_roundoff_m2": annihilated_area - raw_annihilated_area,
        "inventory_change_area_m2": inventory_change,
        "interval_conservation_residual_m2": float(residual),
    }


VACANCY_LEDGER_KEYS = (
    "vacancy_initial_area_m2",
    "vacancy_plastic_production_area_m2",
    "vacancy_thermal_exchange_area_m2",
    "vacancy_annihilation_area_m2",
    "vacancy_consumed_by_void_area_m2",
)


def initial_vacancy_ledgers(vacancy_fraction: np.ndarray, element_area_m2: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(vacancy_fraction, dtype=float)
    areas = np.asarray(element_area_m2, dtype=float)
    inventory = math.fsum(float(value * area) for value, area in zip(values, areas))
    return MappingProxyType({
        "vacancy_initial_area_m2": inventory,
        "vacancy_plastic_production_area_m2": 0.0,
        "vacancy_thermal_exchange_area_m2": 0.0,
        "vacancy_annihilation_area_m2": 0.0,
        "vacancy_consumed_by_void_area_m2": 0.0,
    })


def vacancy_conservation(state: Any) -> dict[str, float]:
    areas = np.asarray(state.mesh.area_e, dtype=float)
    values = np.asarray(state.vacancy_fraction_gp, dtype=float)
    field_inventory = math.fsum(float(value * area) for value, area in zip(values, areas))
    ledger = state.vacancy_ledgers
    lhs = (
        float(ledger["vacancy_initial_area_m2"])
        + float(ledger["vacancy_plastic_production_area_m2"])
        + float(ledger["vacancy_thermal_exchange_area_m2"])
    )
    rhs = (
        field_inventory
        + float(ledger["vacancy_consumed_by_void_area_m2"])
        + float(ledger["vacancy_annihilation_area_m2"])
    )
    return {
        "vacancy_field_inventory_area_m2": field_inventory,
        "vacancy_conservation_lhs_area_m2": lhs,
        "vacancy_conservation_rhs_area_m2": rhs,
        "vacancy_conservation_residual_m2": lhs - rhs,
    }


def advance_accepted_vacancy_state(
    previous_accepted_state: Any,
    accepted_state: Any,
    *,
    duration_s: float,
    temperature_K: float,
    process_zone_length_m: float,
    rho_reference_m2: float,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> tuple[Any, dict[str, Any]]:
    """Advance vacancies from accepted mechanical fields only."""
    if previous_accepted_state.mesh.elems.shape != accepted_state.mesh.elems.shape:
        raise ValueError("plastic vacancy production requires a common accepted mesh")
    plastic_rate = equivalent_plastic_strain_rate_s(
        previous_accepted_state.ep_gp, accepted_state.ep_gp, duration_s,
    )
    rho = np.asarray(accepted_state.rho_gp, dtype=float)
    production = vacancy_production_rate_s(
        plastic_rate, rho, rho_reference_m2=rho_reference_m2, row=row,
    )
    sink = vacancy_annihilation_rate_s(
        temperature_K, rho, process_zone_length_m=process_zone_length_m, row=row,
    )
    updated, audit = advance_vacancy_with_fixed_rates(
        previous_accepted_state, accepted_state,
        duration_s=duration_s, temperature_K=temperature_K,
        equivalent_plastic_rate_s=plastic_rate, production_rate_s=production,
        annihilation_rate_s=sink, row=row,
    )
    audit.update({
        "rho_reference_m2": float(rho_reference_m2),
        "process_zone_length_m": float(process_zone_length_m),
        "local_dislocation_density_m2": rho,
    })
    return updated, audit


def advance_vacancy_with_fixed_rates(
    previous_state: Any, accepted_state: Any, *, duration_s: float,
    temperature_K: float, equivalent_plastic_rate_s: np.ndarray,
    production_rate_s: np.ndarray, annihilation_rate_s: np.ndarray,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> tuple[Any, dict[str, Any]]:
    """Advance the vacancy field on one accepted piecewise-constant clock."""
    plastic_rate = np.asarray(equivalent_plastic_rate_s, dtype=float)
    production = np.asarray(production_rate_s, dtype=float)
    sink = np.asarray(annihilation_rate_s, dtype=float)
    if plastic_rate.shape != (accepted_state.mesh.ne,):
        raise ValueError("accepted plastic rate must own one value per element")
    equilibrium = equilibrium_vacancy_fraction(temperature_K, row)
    interval = exact_vacancy_interval(
        previous_state.vacancy_fraction_gp,
        element_area_m2=accepted_state.mesh.area_e,
        production_rate_s=production,
        annihilation_rate_s=sink,
        equilibrium_fraction=equilibrium,
        duration_s=duration_s,
    )
    ledgers = dict(previous_state.vacancy_ledgers)
    for key in (
        "vacancy_plastic_production_area_m2", "vacancy_thermal_exchange_area_m2",
        "vacancy_annihilation_area_m2",
    ):
        ledgers[key] = float(ledgers[key]) + float(interval[key])
    updated = replace(
        accepted_state,
        vacancy_fraction_gp=interval["vacancy_fraction_gp"],
        vacancy_ledgers=ledgers,
        vacancy_temperature_K=float(temperature_K),
    )
    ledger_roundoff_correction = vacancy_conservation(updated)[
        "vacancy_conservation_residual_m2"
    ]
    budget = conservation_roundoff_budget_m2(
        *updated.vacancy_ledgers.values(),
        vacancy_conservation(updated)["vacancy_field_inventory_area_m2"],
    )
    if abs(ledger_roundoff_correction) > budget:
        raise RuntimeError("vacancy ledger discrepancy exceeds numerical roundoff budget")
    if ledger_roundoff_correction != 0.0:
        ledgers = dict(updated.vacancy_ledgers)
        ledgers["vacancy_annihilation_area_m2"] = (
            float(ledgers["vacancy_annihilation_area_m2"]) + ledger_roundoff_correction
        )
        updated = replace(updated, vacancy_ledgers=ledgers)
    audit = {
        "schema": SCHEMA,
        "vacancy_row_id": row.row_id,
        "vacancy_row_fingerprint": row.fingerprint,
        "temperature_K": float(temperature_K),
        "equilibrium_vacancy_fraction": equilibrium,
        "vacancy_diffusivity_m2_s": vacancy_diffusivity_m2_s(temperature_K, row),
        "equivalent_plastic_strain_rate_s": plastic_rate,
        "vacancy_production_rate_s": production,
        "vacancy_annihilation_rate_s": sink,
        "accepted_ledger_roundoff_correction_m2": ledger_roundoff_correction,
        "conservation_roundoff_budget_m2": budget,
        **{key: value for key, value in interval.items() if key != "vacancy_fraction_gp"},
        **vacancy_conservation(updated),
    }
    return updated, audit


def analytic_vacancy_field_and_integral(
    vacancy_fraction: np.ndarray, *, production_rate_s: np.ndarray,
    annihilation_rate_s: np.ndarray, equilibrium_fraction: float, duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return c(t) and elementwise integral_0^t c(s) ds analytically."""
    c0 = np.asarray(vacancy_fraction, dtype=float)
    production = np.broadcast_to(np.asarray(production_rate_s, dtype=float), c0.shape)
    sink = np.broadcast_to(np.asarray(annihilation_rate_s, dtype=float), c0.shape)
    duration = float(duration_s)
    active = sink > 0.0
    field = np.empty_like(c0); integral = np.empty_like(c0)
    steady = np.empty_like(c0)
    steady[active] = float(equilibrium_fraction) + production[active] / sink[active]
    decay = np.exp(-sink[active] * duration)
    field[active] = steady[active] + (c0[active] - steady[active]) * decay
    integral[active] = (
        steady[active] * duration
        + (c0[active] - steady[active]) * (-np.expm1(-sink[active] * duration)) / sink[active]
    )
    field[~active] = c0[~active] + production[~active] * duration
    integral[~active] = c0[~active] * duration + 0.5 * production[~active] * duration**2
    return np.maximum(field, 0.0), integral


def integrated_capture_vacancy_time(
    state: Any, element_ids: np.ndarray, *, production_rate_s: np.ndarray,
    annihilation_rate_s: np.ndarray, equilibrium_fraction: float, duration_s: float,
) -> float:
    """Area-weighted integral of the local vacancy fraction on the common clock."""
    ids = np.asarray(element_ids, dtype=int)
    _, integral = analytic_vacancy_field_and_integral(
        state.vacancy_fraction_gp, production_rate_s=production_rate_s,
        annihilation_rate_s=annihilation_rate_s,
        equilibrium_fraction=equilibrium_fraction, duration_s=duration_s,
    )
    areas = np.asarray(state.mesh.area_e, dtype=float)[ids]
    return float(np.sum(areas * integral[ids]) / np.sum(areas))


def vacancy_weighted_first_passage_time(
    state: Any, element_ids: np.ndarray, *, production_rate_s: np.ndarray,
    annihilation_rate_s: np.ndarray, equilibrium_fraction: float,
    rate_per_vacancy_s: float, action_remaining: float, maximum_duration_s: float,
) -> float | None:
    """Root of A*integral(c_site dt)=remaining action on a fixed mechanics interval."""
    coefficient = float(rate_per_vacancy_s)
    remaining = float(action_remaining)
    limit = float(maximum_duration_s)
    if remaining <= 0.0:
        return 0.0
    if coefficient <= 0.0 or limit <= 0.0:
        return None
    def residual(time_s: float) -> float:
        return coefficient * integrated_capture_vacancy_time(
            state, element_ids, production_rate_s=production_rate_s,
            annihilation_rate_s=annihilation_rate_s,
            equilibrium_fraction=equilibrium_fraction, duration_s=time_s,
        ) - remaining
    if residual(limit) < 0.0:
        return None
    return float(brentq(residual, 0.0, limit, xtol=1.0e-14, rtol=4*np.finfo(float).eps))


def deterministic_capture_patch(state: Any, center_m: tuple[float, float], process_zone_length_m: float) -> np.ndarray:
    centroids = np.mean(np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)], axis=1)
    distance = np.linalg.norm(centroids - np.asarray(center_m, dtype=float), axis=1)
    selected = np.flatnonzero(distance <= float(process_zone_length_m))
    if not len(selected):
        selected = np.asarray((int(np.argmin(distance)),))
    return selected


def local_vacancy_fraction(state: Any, element_ids: np.ndarray) -> float:
    ids = np.asarray(element_ids, dtype=int)
    area = np.asarray(state.mesh.area_e, dtype=float)[ids]
    values = np.asarray(state.vacancy_fraction_gp, dtype=float)[ids]
    return float(np.sum(area * values) / np.sum(area))


def debit_local_vacancy_inventory(
    state: Any, *, requested_area_m2: float, element_ids: np.ndarray,
) -> tuple[Any, dict[str, float]]:
    requested = float(requested_area_m2)
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError("vacancy debit must be finite and nonnegative")
    ids = np.asarray(element_ids, dtype=int)
    field = np.asarray(state.vacancy_fraction_gp, dtype=float).copy()
    areas = np.asarray(state.mesh.area_e, dtype=float)
    available = math.fsum(float(areas[index] * field[index]) for index in ids)
    consumed = min(requested, available)
    remaining = max(available - consumed, 0.0)
    if available > 0.0:
        field[ids] *= remaining / available
    # Pin one element so the patch integral is the requested remaining
    # inventory under the same summation convention used by the audit.
    if len(ids) and remaining > 0.0:
        target = int(ids[int(np.argmax(areas[ids] * field[ids]))])
        other = math.fsum(
            float(areas[index] * field[index]) for index in ids if int(index) != target
        )
        field[target] = max((remaining - other) / float(areas[target]), 0.0)
    remaining_actual = math.fsum(float(areas[index] * field[index]) for index in ids)
    # The transaction owns the analytically requested debit.  The separately
    # reported representation residual measures its last-bit P0 realization.
    consumed_actual = consumed
    ledgers = dict(state.vacancy_ledgers)
    ledgers["vacancy_consumed_by_void_area_m2"] = (
        float(ledgers["vacancy_consumed_by_void_area_m2"]) + consumed_actual
    )
    updated = replace(state, vacancy_fraction_gp=field, vacancy_ledgers=ledgers)
    ledger_roundoff_correction = vacancy_conservation(updated)[
        "vacancy_conservation_residual_m2"
    ]
    correction_budget = conservation_roundoff_budget_m2(
        *updated.vacancy_ledgers.values(),
        vacancy_conservation(updated)["vacancy_field_inventory_area_m2"],
    )
    if abs(ledger_roundoff_correction) > correction_budget:
        raise RuntimeError("vacancy debit discrepancy exceeds numerical roundoff budget")
    if ledger_roundoff_correction != 0.0:
        ledgers = dict(updated.vacancy_ledgers)
        ledgers["vacancy_consumed_by_void_area_m2"] = (
            float(ledgers["vacancy_consumed_by_void_area_m2"]) + ledger_roundoff_correction
        )
        updated = replace(updated, vacancy_ledgers=ledgers)
    return updated, {
        "requested_vacancy_area_m2": requested,
        "available_capture_patch_vacancy_area_m2": available,
        "consumed_vacancy_area_m2": consumed_actual,
        "growth_inventory_limited": consumed_actual < requested,
        "p0_debit_representation_residual_m2": (
            available - remaining_actual - consumed_actual
        ),
        "accepted_ledger_roundoff_correction_m2": ledger_roundoff_correction,
        "conservation_roundoff_budget_m2": correction_budget,
        **vacancy_conservation(updated),
    }


def vacancy_attachment_frequency_s(
    temperature_K: float, vacancy_fraction_site: float, burgers_vector_m: float,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> float:
    concentration = float(vacancy_fraction_site)
    burgers = float(burgers_vector_m)
    if concentration < 0.0 or burgers <= 0.0:
        raise ValueError("attachment inputs must be physical")
    return float(row.capture_factor) * vacancy_diffusivity_m2_s(temperature_K, row) / burgers**2 * concentration


def vacancy_growth_rates_m_s(
    *,
    temperature_K: float,
    radius_m: float,
    vacancy_fraction_site: float,
    equivalent_plastic_rate_s: float,
    row: VacancyParameterRow = REFERENCE_VACANCY_ROW,
) -> dict[str, float | str]:
    radius = float(radius_m)
    plastic_rate = float(equivalent_plastic_rate_s)
    supersaturation = max(float(vacancy_fraction_site) - equilibrium_vacancy_fraction(temperature_K, row), 0.0)
    if radius <= 0.0 or plastic_rate < 0.0:
        raise ValueError("growth radius/rate inputs must be physical")
    diffusion = float(row.capture_factor) * vacancy_diffusivity_m2_s(temperature_K, row) / radius * supersaturation
    accommodation = radius * plastic_rate
    combined = (
        0.0 if diffusion <= 0.0 or accommodation <= 0.0
        else 1.0 / (1.0 / diffusion + 1.0 / accommodation)
    )
    controlling = (
        "BLOCKED_DIFFUSION" if diffusion <= 0.0 else
        "BLOCKED_PLASTIC_ACCOMMODATION" if accommodation <= 0.0 else
        "DIFFUSION" if diffusion <= accommodation else "PLASTIC_ACCOMMODATION"
    )
    return {
        "Rdot_diff_m_s": diffusion,
        "Rdot_plast_m_s": accommodation,
        "Rdot_m_s": combined,
        "controlling_branch": controlling,
    }


__all__ = [
    "COMMON_VACANCY_ROW_ID", "CONSERVATION_ABSOLUTE_FLOOR_M2",
    "DILUTE_VACANCY_FRACTION_LIMIT",
    "ENGINEERING_SHEAR_VOIGT_CONVENTION", "REFERENCE_VACANCY_ROW", "SCHEMA",
    "VACANCY_LEDGER_KEYS", "VacancyParameterRow", "advance_accepted_vacancy_state",
    "advance_vacancy_with_fixed_rates", "analytic_vacancy_field_and_integral",
    "conservation_roundoff_budget_m2", "debit_local_vacancy_inventory",
    "deterministic_capture_patch", "equilibrium_vacancy_fraction",
    "equivalent_plastic_strain_rate_s", "exact_vacancy_interval", "initial_vacancy_ledgers",
    "local_vacancy_fraction", "vacancy_annihilation_rate_s", "vacancy_attachment_frequency_s",
    "integrated_capture_vacancy_time", "vacancy_conservation", "vacancy_diffusivity_m2_s",
    "vacancy_growth_rates_m_s", "vacancy_production_rate_s",
    "vacancy_weighted_first_passage_time",
]
