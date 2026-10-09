"""Prospective, result-independent V12 absolute-intensity criterion.

This module contains policy only.  It intentionally has no solver entry point so
importing or testing it cannot generate or inspect V12 intensity results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


CRITERION_ID = "v12_absolute_k_qualification_v1"
CONTOURS_M = ((240e-6, 260e-6), (250e-6, 270e-6), (260e-6, 280e-6))
MESH_LEVELS_M = (12.5e-6, 6.25e-6, 3.125e-6)
FIXED_KAPPAS = (1e-8,)
JOINT_LIMIT_P = 2
JOINT_LIMIT_H0_M = 25e-6
JOINT_LIMIT_KAPPA0 = 1e-6
WILLIAMS_ANNULI_M = ((75e-6, 175e-6), (87.5e-6, 175e-6), (75e-6, 162.5e-6))
WILLIAMS_TERMS = ("K_I", "K_II", "T_STRESS_SIGMA_XX")
WILLIAMS_WEIGHTING = "sqrt_element_area"

LIMITS = {
    "maximum_contour_spread": 0.05,
    "maximum_mode_II_to_mode_I": 0.05,
    "maximum_GK_to_energy_G_error": 0.05,
    "maximum_GK_to_compliance_G_error": 0.05,
    "maximum_KI_to_energy_K_error": 0.05,
    "maximum_Williams_KI_to_energy_K_error": 0.05,
    "maximum_Williams_radius_sensitivity": 0.05,
    "minimum_r_inner_over_h_tip": 8.0,
    "maximum_q_support_width_over_r_inner": 0.1,
    "minimum_root_exterior_patch_clearance_m": 15e-6,
    "minimum_active_angular_coverage_fraction": 0.70,
    "maximum_fit_condition_number": 1e8,
    "minimum_fit_samples": 24,
}


@dataclass(frozen=True)
class AbsoluteKClassification:
    standard_integral: str
    aggregate: str
    production_may_continue: bool


def classify_absolute_k(
    *, conforming_pass: bool, primal_pass: bool, corridor_v3_pass: bool,
    standard_pass: bool, energy_pass: bool, williams_pass: bool,
    production_consumes_absolute_k: bool, conforming_tip_patch_pass: bool = False,
) -> AbsoluteKClassification:
    """Apply the frozen Stage-I outcome tree without inventing a PASS."""
    prerequisites = conforming_pass and primal_pass and corridor_v3_pass
    if prerequisites and standard_pass:
        return AbsoluteKClassification("PASS", "PASS", True)
    if prerequisites and energy_pass and williams_pass:
        allowed = (not production_consumes_absolute_k) or conforming_tip_patch_pass
        return AbsoluteKClassification(
            "NOT_QUALIFIED_REQUIRES_INHOMOGENEITY_CORRECTION",
            "QUALIFIED_BY_ENERGY_AND_WILLIAMS_STANDARD_INTEGRAL_UNAVAILABLE",
            allowed,
        )
    if prerequisites and conforming_tip_patch_pass:
        return AbsoluteKClassification(
            "NOT_QUALIFIED", "CONFORMING_TIP_PATCH_REQUIRED", True
        )
    return AbsoluteKClassification("NOT_QUALIFIED", "NOT_QUALIFIED", False)


def all_limits_pass(checks: Mapping[str, bool], required: Sequence[str]) -> bool:
    """Evidence-derived conjunction; missing checks fail closed."""
    return all(checks.get(name) is True for name in required)


__all__ = [
    "AbsoluteKClassification", "CONTOURS_M", "CRITERION_ID", "FIXED_KAPPAS",
    "JOINT_LIMIT_H0_M", "JOINT_LIMIT_KAPPA0", "JOINT_LIMIT_P", "LIMITS",
    "MESH_LEVELS_M", "WILLIAMS_ANNULI_M", "WILLIAMS_TERMS",
    "WILLIAMS_WEIGHTING", "all_limits_pass", "classify_absolute_k",
]
