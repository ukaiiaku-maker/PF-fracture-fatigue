"""Shared adaptive cycle map for intrinsic reverse-glide v7.

This module is the single within-cycle authority for the next exact and
accelerated fatigue implementations.  The accelerator is allowed to accelerate
only *between* completed cycles; whenever a cycle is resolved, it must call
``advance_v7_cycle`` rather than a separate frozen-hazard or frozen-state law.

The module also records one bounded shear-stress decomposition audit.  It does
not add a constitutive parameter or alter the v7 state law.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np

from arrhenius_fracture import fatigue_v914 as base
from v914_adaptive_feedback_v6 import (
    AdaptiveFeedbackControls,
    error_passes,
    phase_sample,
    state_error,
    state_observables,
)
from v914_reverse_drive_utils import tensile_reference_signs


CYCLE_MAP_ID = "v9.14_v7_intrinsic_reverse_glide_adaptive_cycle_map_v1"


def intrinsic_phase_sample(state, loading, phase: float, depth: int) -> dict[str, float]:
    """Generic adaptive sample plus local applied/GND/effective shear split."""
    sample = dict(phase_sample(state, loading, phase, depth))
    K = float(loading.K_at_phase(float(phase)))
    rates = state.local_rates(K, loading.temperature_K)
    drive = np.asarray(state.emission_drive_factors(), dtype=float)
    signs = tensile_reference_signs(drive)

    tau_applied = np.asarray(rates["reversible_tau_transport_external_Pa"], dtype=float)
    tau_gnd = np.asarray(rates["tau_gnd_Pa"], dtype=float)
    tau_eff = np.asarray(rates["reversible_tau_transport_eff_Pa"], dtype=float)
    if tau_applied.shape != tau_gnd.shape or tau_eff.shape != tau_gnd.shape:
        raise RuntimeError("v7 shear decomposition arrays must have identical shape")

    projected_applied = tau_applied * signs[:, None]
    projected_gnd = tau_gnd * signs[:, None]
    projected_eff = tau_eff * signs[:, None]
    flat = int(np.argmin(projected_eff))
    system, bin_index = np.unravel_index(flat, projected_eff.shape)

    reconstruction = float(
        projected_eff[system, bin_index]
        - projected_applied[system, bin_index]
        - projected_gnd[system, bin_index]
    )
    scale = max(abs(float(projected_eff[system, bin_index])), 1.0)
    if abs(reconstruction) > 1.0e-11 * scale:
        raise RuntimeError("v7 local shear decomposition does not reconstruct tau_eff")

    sample.update(
        {
            "shear_system": int(system),
            "shear_bin": int(bin_index),
            "shear_x_um": float(1.0e6 * state.x[bin_index]),
            "shear_drive_factor": float(drive[system]),
            "tau_applied_projected_GPa": float(projected_applied[system, bin_index] / 1.0e9),
            "tau_gnd_projected_GPa": float(projected_gnd[system, bin_index] / 1.0e9),
            "tau_eff_projected_GPa": float(projected_eff[system, bin_index] / 1.0e9),
            "tau_reconstruction_error_Pa": reconstruction,
            "shear_mobile_density_m2": float(np.sum(state.mobile_m2[system, :, bin_index])),
            "shear_retained_density_m2": float(np.sum(state.retained_m2[system, :, bin_index])),
            "shear_signed_gnd_m2": float(state.signed_gnd_m2()[system, bin_index]),
            "shear_Kshield_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
        }
    )
    return sample


def _adaptive_cycle_with_sampler(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
    *,
    sample_fn: Callable[..., dict[str, float]],
    advance_phase_fn: Callable[..., float] = base._advance_phase,
) -> tuple[Any, float, dict[str, Any]]:
    """v6 step-doubling algorithm with a pluggable diagnostic sampler."""
    controls.validate()
    committed = copy.deepcopy(state)
    total_hazard = 0.0
    telemetry: dict[str, Any] = {
        "attempted_intervals": 0,
        "accepted_intervals": 0,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": math.inf,
        "maximum_accepted_error": {
            "shielding": 0.0,
            "mobile": 0.0,
            "retained": 0.0,
            "returned": 0.0,
            "tip_radius": 0.0,
            "hazard": 0.0,
        },
        "samples": [sample_fn(committed, loading, 0.0, 0)],
    }

    def advance_interval(start_state, p0: float, p1: float, depth: int):
        telemetry["attempted_intervals"] += 1
        telemetry["maximum_depth_reached"] = max(
            telemetry["maximum_depth_reached"], int(depth)
        )
        midpoint = 0.5 * (p0 + p1)

        coarse = copy.deepcopy(start_state)
        coarse_hazard = float(advance_phase_fn(coarse, loading, p0, p1))

        fine = copy.deepcopy(start_state)
        h0 = float(advance_phase_fn(fine, loading, p0, midpoint))
        midpoint_state = copy.deepcopy(fine)
        h1 = float(advance_phase_fn(fine, loading, midpoint, p1))
        fine_hazard = h0 + h1
        if (
            coarse_hazard < 0.0
            or fine_hazard < 0.0
            or not math.isfinite(coarse_hazard)
            or not math.isfinite(fine_hazard)
        ):
            raise FloatingPointError("invalid v7 adaptive cycle hazard")

        error = state_error(coarse, fine, coarse_hazard, fine_hazard, controls)
        if error_passes(error, controls):
            telemetry["accepted_intervals"] += 1
            width = float(p1 - p0)
            telemetry["minimum_accepted_phase_width"] = min(
                telemetry["minimum_accepted_phase_width"], width
            )
            for name, value in error.items():
                telemetry["maximum_accepted_error"][name] = max(
                    telemetry["maximum_accepted_error"][name], float(value)
                )
            telemetry["samples"].append(
                sample_fn(midpoint_state, loading, midpoint, depth + 1)
            )
            telemetry["samples"].append(sample_fn(fine, loading, p1, depth + 1))
            return fine, fine_hazard

        if depth >= controls.max_refinement_depth:
            raise RuntimeError(
                "v7 adaptive cycle refinement failed closed at "
                f"phase=[{p0:.16g},{p1:.16g}], depth={depth}, error={error}"
            )

        telemetry["refined_intervals"] += 1
        first_state, first_hazard = advance_interval(
            start_state, p0, midpoint, depth + 1
        )
        second_state, second_hazard = advance_interval(
            first_state, midpoint, p1, depth + 1
        )
        return second_state, first_hazard + second_hazard

    nbase = int(controls.base_phase_intervals)
    for index in range(nbase):
        p0 = index / nbase
        p1 = (index + 1) / nbase
        committed, increment = advance_interval(committed, p0, p1, 0)
        total_hazard += increment

    if math.isinf(telemetry["minimum_accepted_phase_width"]):
        telemetry["minimum_accepted_phase_width"] = math.nan

    by_phase: dict[float, dict[str, float]] = {}
    for sample in telemetry["samples"]:
        p = float(sample["phase"])
        previous = by_phase.get(p)
        if previous is None or sample["depth"] >= previous["depth"]:
            by_phase[p] = sample
    telemetry["samples"] = [by_phase[p] for p in sorted(by_phase)]
    return committed, total_hazard, telemetry


def advance_v7_cycle(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
) -> tuple[Any, float, dict[str, Any]]:
    """Authoritative deterministic within-cycle map for v7."""
    final_state, hazard, telemetry = _adaptive_cycle_with_sampler(
        state,
        loading,
        controls,
        sample_fn=intrinsic_phase_sample,
    )
    telemetry["cycle_map_id"] = CYCLE_MAP_ID
    return final_state, hazard, telemetry


def cycle_endpoint_summary(state, hazard: float, cycle_index: int) -> dict[str, float]:
    obs = state_observables(state)
    diagnostics = state.reversibility_diagnostics()
    return {
        "cycle_index": int(cycle_index),
        "hazard_action": float(hazard),
        "shielding_MPa_sqrt_m": float(obs["shielding_MPa_sqrt_m"]),
        "mobile_line_content": float(obs["mobile_line_content"]),
        "retained_line_content": float(obs["retained_line_content"]),
        "returned_source_slip": float(obs["returned_source_slip"]),
        "physical_return_fraction": float(obs["physical_return_fraction"]),
        "tip_radius_m": float(obs["tip_radius_m"]),
        "cumulative_source_slip": float(
            diagnostics.get("reversible_cumulative_source_slip_count", 0.0)
        ),
        "raw_return_fraction": float(
            diagnostics.get("reversible_raw_return_fraction_of_emitted", 0.0)
        ),
    }


def minimum_shear_sample(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("samples are required")
    return dict(min(samples, key=lambda row: float(row["tau_eff_projected_GPa"])))


__all__ = [
    "CYCLE_MAP_ID",
    "advance_v7_cycle",
    "cycle_endpoint_summary",
    "intrinsic_phase_sample",
    "minimum_shear_sample",
]
