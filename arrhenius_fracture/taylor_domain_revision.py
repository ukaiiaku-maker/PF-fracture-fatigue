"""Construct traceable Taylor-amplitude domain revisions.

The construction changes only ``taylor_H0_eV``.  Shared characteristic-stress
defects are reported separately because repairing those scales changes the
emission, Peierls, and Taylor mechanisms together.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Iterable

from .material_manifest import KB_EV_PER_K
from .taylor_release import (
    completion_rate, density_completion_order, signed_arrhenius_components,
)

T_MIN_K = 300.0
T_MAX_K = 1300.0
DEFAULT_MARGINS_EV = (0.10, 0.25, 0.50)
MINIMUM_RAW_STRESS_PA = 1.0
DEFAULT_BURGERS_M = 2.74e-10


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    if not rows or not fields:
        raise ValueError("parent registry must contain a header and rows")
    return fields, rows


def linear_extrema(reference: float, slope_per_K: float, reference_K: float,
                   minimum_K: float = T_MIN_K,
                   maximum_K: float = T_MAX_K) -> dict[str, float]:
    endpoints = {
        float(minimum_K): float(reference) + float(slope_per_K)
        * (float(minimum_K) - float(reference_K)),
        float(maximum_K): float(reference) + float(slope_per_K)
        * (float(maximum_K) - float(reference_K)),
    }
    minimum_at = min(endpoints, key=endpoints.get)
    maximum_at = max(endpoints, key=endpoints.get)
    return {
        "minimum": endpoints[minimum_at], "minimum_at_K": minimum_at,
        "maximum": endpoints[maximum_at], "maximum_at_K": maximum_at,
    }


def raw_taylor_extrema(row: dict[str, str]) -> dict[str, float]:
    slope = -float(row["taylor_activation_entropy_kB"]) * KB_EV_PER_K
    return linear_extrema(
        float(row["taylor_H0_eV"]), slope, float(row["Tref_K"])
    )


def required_taylor_reference(row: dict[str, str], margin_eV: float) -> dict[str, float]:
    margin = float(margin_eV)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("Taylor margin must be positive and finite")
    extrema = raw_taylor_extrema(row)
    parent_reference = float(row["taylor_H0_eV"])
    slope = -float(row["taylor_activation_entropy_kB"]) * KB_EV_PER_K
    if extrema["minimum"] >= margin:
        reference = parent_reference
    else:
        reference = margin - slope * (
            extrema["minimum_at_K"] - float(row["Tref_K"])
        )
        while linear_extrema(
            reference, slope, float(row["Tref_K"])
        )["minimum"] < margin:
            reference = math.nextafter(reference, math.inf)
    increase = reference - parent_reference
    revised_minimum = linear_extrema(
        reference, slope, float(row["Tref_K"])
    )["minimum"]
    return {
        "parent_reference_eV": parent_reference,
        "parent_minimum_eV": extrema["minimum"],
        "parent_minimum_at_K": extrema["minimum_at_K"],
        "required_increase_eV": increase,
        "revised_reference_eV": reference,
        "revised_minimum_eV": revised_minimum,
        "declared_margin_eV": margin,
    }


def margin_token(margin_eV: float) -> str:
    return f"{float(margin_eV):.2f}".replace(".", "p")


def construct_amplitude_rows(
    parent_rows: Iterable[dict[str, str]], margins_eV: Iterable[float]
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    variants: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    for parent in parent_rows:
        for margin in margins_eV:
            result = required_taylor_reference(parent, margin)
            token = margin_token(margin)
            row = dict(parent)
            row["option_key"] = f'{parent["option_key"]}__taylor_margin_{token}eV'
            row["candidate_id"] = f'{parent["candidate_id"]}__taylor_margin_{token}eV'
            row["role"] = f'prospective Taylor amplitude margin {float(margin):.2f} eV'
            row["validation_status"] = (
                "constructed amplitude-only domain candidate; not material calibrated; "
                "native qualification required"
            )
            row["taylor_H0_eV"] = repr(result["revised_reference_eV"])
            variants.append(row)
            changed = {
                key: {"parent": parent.get(key), "revised": row.get(key)}
                for key in row if row.get(key) != parent.get(key)
            }
            records.append({
                "parent_option_key": parent["option_key"],
                "variant_option_key": row["option_key"],
                "material_class": parent["material_class"],
                **result,
                "parameter_changes": changed,
                "physics_change": (
                    "taylor_H0_eV amplitude only"
                    if result["required_increase_eV"] > 0.0
                    else "none; parent already exceeds the declared margin"
                ),
                "calibration_status": "CONSTRUCTED_NOT_MATERIAL_CALIBRATED",
            })
    return variants, records


def raw_emission_stress_extrema(row: dict[str, str]) -> dict[str, float]:
    result = linear_extrema(
        float(row["emit_sigc0_GPa"]) * 1.0e9,
        float(row["emit_sT_GPa_per_K"]) * 1.0e9,
        float(row["Tref_K"]),
    )
    return {f"{key}_Pa" if key in {"minimum", "maximum"} else key: value
            for key, value in result.items()}


def minimal_intercept_stress_candidate(row: dict[str, str]) -> dict[str, object]:
    extrema = raw_emission_stress_extrema(row)
    increase_Pa = max(0.0, MINIMUM_RAW_STRESS_PA - extrema["minimum_Pa"])
    revised = dict(row)
    revised["option_key"] = f'{row["option_key"]}__shared_sigc_minimal_intercept'
    revised["candidate_id"] = f'{row["candidate_id"]}__shared_sigc_minimal_intercept'
    revised["emit_sigc0_GPa"] = repr(
        float(row["emit_sigc0_GPa"]) + increase_Pa / 1.0e9
    )
    return {
        "status": "PROSPECTIVE_UNSELECTED_SHARED_LAW_CHANGE",
        "construction": "raise emit_sigc0 only so raw emission sigc reaches 1 Pa at the continuous-domain minimum",
        "parent_option_key": row["option_key"],
        "minimum_before_Pa": extrema["minimum_Pa"],
        "minimum_at_K": extrema["minimum_at_K"],
        "intercept_increase_Pa": increase_Pa,
        "affected_mechanisms": [
            "emission characteristic stress",
            "Peierls characteristic stress inherited through peierls_stress_fraction",
            "Taylor characteristic stress inherited through taylor_stress_fraction",
        ],
        "changed_fields": ["option_key", "candidate_id", "emit_sigc0_GPa"],
        "row": revised,
    }


def assert_amplitude_only(parent: dict[str, str], variant: dict[str, str]) -> None:
    allowed = {"option_key", "candidate_id", "role", "validation_status", "taylor_H0_eV"}
    changed = {key for key in parent if parent.get(key) != variant.get(key)}
    if changed - allowed:
        raise AssertionError(f"non-amplitude parameter change: {sorted(changed)}")


def native_taylor_coordinates(
    row: dict[str, str], *, raw_density_m2: float, barrier_argument_Pa: float,
    burgers_m: float = DEFAULT_BURGERS_M,
) -> dict[str, float | str | bool]:
    """Map the Taylor surface argument back to every native stress coordinate."""
    raw_density = float(raw_density_m2)
    density = max(raw_density, float(row["rho_forest_floor_m2"]), 1.0)
    spacing = 1.0 / (2.0 * math.sqrt(density))
    amplification = spacing / max(abs(float(burgers_m)), 1.0e-30)
    projection = float(row["taylor_stress_fraction"])
    scale = projection * amplification
    if scale <= 0.0:
        raise ValueError("native Taylor projection and amplification must be positive")
    argument = float(barrier_argument_Pa)
    local = argument / scale
    return {
        "raw_forest_density_m2": raw_density,
        "consumed_forest_density_m2": density,
        "density_floor_was_active": density != raw_density,
        "forest_spacing_m": spacing,
        "burgers_m": abs(float(burgers_m)),
        "amplification_phi": amplification,
        "projection_fraction": projection,
        "local_specimen_stress_Pa": local,
        "projected_local_stress_before_amplification_Pa": projection * local,
        "barrier_argument_Pa": argument,
        "forward_coordinate_identity": "barrier_argument = projection_fraction * local_specimen_stress * amplification_phi",
        "fixed_density_chain_rule": "dG/d(local_specimen_stress) = dG/d(barrier_argument) * projection_fraction * amplification_phi",
    }


def inverse_elementary_forward_contour(
    surface, row: dict[str, str], *, temperature_K: float, rate_s: float,
    densities_m2: Iterable[float],
) -> dict[str, object]:
    """Invert the one-way surface rate and verify it in that same coordinate."""
    T_K = float(temperature_K)
    rate = float(rate_s)
    diagnostic = surface.diagnostic(0.0, T_K)
    G0 = float(diagnostic["executed_G0_eV"])
    floor = float(diagnostic["floor_eV"])
    sigc = float(diagnostic["executed_sigc_Pa"])
    target = -KB_EV_PER_K * T_K * math.log(rate / surface.attempt_frequency_s)
    base: dict[str, object] = {
        "rate_s": rate,
        "target_barrier_eV": target,
        "coordinate": "Taylor surface barrier argument; not specimen stress",
    }
    if not floor < target < G0 or surface.alpha <= 0.0 or surface.exponent <= 0.0:
        return {**base, "status": "NO_FINITE_CONTOUR"}
    ratio = (target - floor) / (G0 - floor)
    argument = sigc * ((-math.log(ratio) / surface.alpha) ** (1.0 / surface.exponent))
    forward = float(surface.rate(argument, T_K))
    return {
        **base,
        "status": "FINITE_CONTOUR",
        "barrier_argument_Pa": argument,
        "exact_forward_rate_s": forward,
        "relative_forward_error": abs(forward - rate) / rate,
        "native_coordinate_conversions": [
            native_taylor_coordinates(
                row, raw_density_m2=density, barrier_argument_Pa=argument
            )
            for density in densities_m2
        ],
    }


def inverse_net_completion_contour(
    surface, row: dict[str, str], *, temperature_K: float,
    completion_rate_s: float, raw_density_m2: float,
) -> dict[str, object]:
    """Invert the exact signed-net native Taylor completion mapping."""
    T_K = float(temperature_K)
    target = float(completion_rate_s)
    coordinates_at_zero = native_taylor_coordinates(
        row, raw_density_m2=raw_density_m2, barrier_argument_Pa=0.0
    )
    density = float(coordinates_at_zero["consumed_forest_density_m2"])
    order = float(density_completion_order(
        density,
        correlation_scale=float(row["taylor_corr_scale"]),
        correlation_density_m2=float(row["taylor_corr_rho_c_m2"]),
    ))

    def evaluated(argument: float) -> tuple[float, float, float, float]:
        parts = signed_arrhenius_components(surface, float(argument), T_K)
        forward = float(parts["forward_rate_s"])
        reverse = float(parts["reverse_rate_s"])
        net = float(parts["net_rate_s"])
        completed = float(completion_rate(net, order))
        return completed, forward, reverse, net

    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("completion-rate contour must be positive and finite")
    upper = max(float(surface.diagnostic(0.0, T_K)["executed_sigc_Pa"]), 1.0)
    upper_value = evaluated(upper)[0]
    for _ in range(80):
        if upper_value >= target:
            break
        upper *= 2.0
        upper_value = evaluated(upper)[0]
    else:
        return {
            "status": "NO_FINITE_CONTOUR",
            "target_completion_rate_s": target,
            "raw_forest_density_m2": float(raw_density_m2),
            "completion_order": order,
            "maximum_bracketed_completion_rate_s": upper_value,
        }
    lower = 0.0
    for _ in range(160):
        middle = 0.5 * (lower + upper)
        if evaluated(middle)[0] < target:
            lower = middle
        else:
            upper = middle
    argument = 0.5 * (lower + upper)
    completed, forward, reverse, net = evaluated(argument)
    coordinates = native_taylor_coordinates(
        row, raw_density_m2=raw_density_m2, barrier_argument_Pa=argument
    )
    return {
        "status": "FINITE_CONTOUR",
        "target_completion_rate_s": target,
        "completion_order": order,
        "target_net_elementary_rate_s": target * order,
        "forward_elementary_rate_s": forward,
        "reverse_elementary_rate_s": reverse,
        "exact_net_elementary_rate_s": net,
        "exact_forward_completion_rate_s": completed,
        "relative_forward_error": abs(completed - target) / target,
        **coordinates,
    }


__all__ = [
    "DEFAULT_BURGERS_M", "DEFAULT_MARGINS_EV", "MINIMUM_RAW_STRESS_PA",
    "T_MAX_K", "T_MIN_K",
    "assert_amplitude_only", "construct_amplitude_rows", "linear_extrema",
    "inverse_elementary_forward_contour", "inverse_net_completion_contour",
    "margin_token", "minimal_intercept_stress_candidate", "raw_emission_stress_extrema",
    "native_taylor_coordinates", "raw_taylor_extrema", "read_rows",
    "required_taylor_reference", "sha256_file",
]
