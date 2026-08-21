#!/usr/bin/env python3
"""Exact-cycle causal audit of weak-T v7 self-passivation.

The constitutive trajectory is always the unmodified v7 trajectory.  Radius,
shielding, and one-hit counterfactuals are evaluated only on deep-copied saved
states and never feed back into transport, emission, or the physical hazard.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
V914 = Path(os.environ.get(
    "V914_ROOT",
    "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
))
for path in (str(ROOT / "scripts"), str(V914)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(V914))
sys.path.insert(0, str(ROOT / "scripts"))

from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics, KB_EV_PER_K
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_intrinsic_reverse_glide_v7 import (
    IntrinsicReverseGlideNoReturnState,
    IntrinsicReverseGlideState,
    intrinsic_signed_glide_stress_fields,
)
from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_v7_cycle_map import advance_v7_cycle


SCHEMA = "v914_v7_exact_cycle_causal_physics_audit_v1"


class DiagnosticIntrinsicReverseGlideState(IntrinsicReverseGlideState):
    """v7 state with non-feedback signed Orowan displacement quadrature."""

    def __init__(self, candidate, physics):
        super().__init__(candidate, physics)
        self.cumulative_signed_tip_plastic_displacement_proxy_m = 0.0
        self.cumulative_reverse_tip_plastic_displacement_proxy_m = 0.0

    def _plastic_power_snapshot(self, rates):
        result = dict(super()._plastic_power_snapshot(rates))
        velocity = np.asarray(rates["velocity_m_s"], dtype=float)
        mobile = np.sum(np.maximum(self.mobile_m2, 0.0), axis=1)
        gamma_dot = float(self.c.b_m) * mobile * velocity
        weights, _norm = self._tip_weights()
        projection = np.abs(np.asarray(self.emission_drive_factors(), dtype=float))[:, None]
        displacement_rate = float(
            np.sum(projection * gamma_dot * weights[None, :]) * float(self.dx)
        )
        result["signed_tip_plastic_displacement_rate_proxy_m_s"] = displacement_rate
        result["reverse_tip_plastic_displacement_rate_proxy_m_s"] = max(
            -displacement_rate, 0.0
        )
        return result

    def _accumulate_work_interval(self, start, end, duration_s):
        super()._accumulate_work_interval(start, end, duration_s)
        duration = max(float(duration_s), 0.0)
        signed_increment = 0.5 * duration * (
            float(start["signed_tip_plastic_displacement_rate_proxy_m_s"])
            + float(end["signed_tip_plastic_displacement_rate_proxy_m_s"])
        )
        reverse_increment = 0.5 * duration * (
            float(start["reverse_tip_plastic_displacement_rate_proxy_m_s"])
            + float(end["reverse_tip_plastic_displacement_rate_proxy_m_s"])
        )
        self.cumulative_signed_tip_plastic_displacement_proxy_m += signed_increment
        self.cumulative_reverse_tip_plastic_displacement_proxy_m += reverse_increment


class DiagnosticIntrinsicReverseGlideNoReturnState(
    DiagnosticIntrinsicReverseGlideState
):
    """Diagnostic v7 state using the production IRREV cancellation control."""

    MODEL_ID = IntrinsicReverseGlideNoReturnState.MODEL_ID

    def _cancel_returned_source_slip(
        self, system: int, q: int, returned_per_m: float
    ) -> float:
        return IntrinsicReverseGlideNoReturnState._cancel_returned_source_slip(
            self, system, q, returned_per_m
        )


def _common(path: Path) -> CommonPhysics:
    payload = json.loads(path.read_text())
    values = payload.get("common_physics", payload)
    names = {field.name for field in fields(CommonPhysics)}
    selected = {key: value for key, value in values.items() if key in names}
    for name in (
        "emission_signs", "emission_schmid_factors",
        "shielding_orientation_factors", "forest_interaction_matrix",
        "gnd_stress_projection_matrix", "activation_to_line_content_per_system",
        "emission_geometry_extension_m", "emission_geometry_factors",
    ):
        if name in selected:
            selected[name] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in selected[name]
            )
    result = CommonPhysics(**selected)
    result.validate()
    return result


def exp_floor_limit_eV(surface, temperature_K: float) -> float:
    """Analytical sigma->infinity limit of the active EXP-floor surface."""
    zero = float(surface.zero_stress_eV(temperature_K))
    return min(
        float(surface.floor_max_fraction) * zero,
        max(float(surface.floor_min_eV), float(surface.floor_fraction) * zero),
    )


def cleavage_components(
    state,
    K_MPa_sqrt_m: float,
    temperature_K: float,
    *,
    radius_m: float | None = None,
    shielding_MPa_sqrt_m: float | None = None,
) -> dict[str, float]:
    """Evaluate the unchanged cleavage law with diagnostic geometry overrides."""
    radius = float(state.tip_radius_m() if radius_m is None else radius_m)
    shielding = float(
        state.K_shield_MPa_sqrt_m()
        if shielding_MPa_sqrt_m is None else shielding_MPa_sqrt_m
    )
    effective_K = max(float(K_MPa_sqrt_m) - shielding, 0.0)
    sigma = effective_K * 1.0e6 / math.sqrt(
        2.0 * math.pi * max(radius, state.c.b_m, 1.0e-30)
    )
    barrier = float(state.p.cleavage.barrier_eV(sigma, temperature_K))
    raw = float(state.c.cleavage_nu0_s) * math.exp(
        -barrier / (KB_EV_PER_K * float(temperature_K))
    )
    if float(state.c.cleavage_hits) <= 1.0 + 1.0e-12:
        production = raw
    else:
        exposure = min(raw * float(state.c.cleavage_correlation_time_s), 1.0e12)
        production = float(
            gammainc(float(state.c.cleavage_hits), exposure)
            / float(state.c.cleavage_correlation_time_s)
        )
    equivalent = -KB_EV_PER_K * float(temperature_K) * math.log(
        max(production / float(state.c.cleavage_nu0_s), 1.0e-300)
    )
    return {
        "radius_m": radius,
        "shielding_MPa_sqrt_m": shielding,
        "effective_K_MPa_sqrt_m": effective_K,
        "cleavage_stress_Pa": sigma,
        "raw_arrhenius_barrier_eV": barrier,
        "production_equivalent_barrier_eV": equivalent,
        "raw_one_hit_rate_s": raw,
        "production_rate_s": production,
    }


def frozen_cycle_causal_audit(state, loading, phase_intervals: int = 4096) -> dict[str, float]:
    """Integrate causal rates over one waveform while holding saved state fixed."""
    variants = {
        "full": {},
        "initial_radius": {"radius_m": float(state.c.r0_m)},
        "zero_shield": {"shielding_MPa_sqrt_m": 0.0},
        "virgin_geometry": {
            "radius_m": float(state.c.r0_m),
            "shielding_MPa_sqrt_m": 0.0,
        },
    }
    action = {name: 0.0 for name in variants}
    raw_action = 0.0
    peak: dict[str, dict[str, float]] = {}
    dt = float(loading.period_s) / int(phase_intervals)
    previous = {
        name: cleavage_components(
            state, loading.K_at_phase(0.0), loading.temperature_K, **kwargs
        )
        for name, kwargs in variants.items()
    }
    raw_previous = previous["full"]["raw_one_hit_rate_s"]
    for index in range(1, int(phase_intervals) + 1):
        phase = index / int(phase_intervals)
        current = {
            name: cleavage_components(
                state, loading.K_at_phase(phase), loading.temperature_K, **kwargs
            )
            for name, kwargs in variants.items()
        }
        for name in variants:
            action[name] += 0.5 * dt * (
                previous[name]["production_rate_s"] + current[name]["production_rate_s"]
            )
        raw_current = current["full"]["raw_one_hit_rate_s"]
        raw_action += 0.5 * dt * (raw_previous + raw_current)
        previous = current
        raw_previous = raw_current
    Kmax = float(loading.deltaK_MPa_sqrt_m) / (1.0 - float(loading.R))
    for name, kwargs in variants.items():
        peak[name] = cleavage_components(
            state, Kmax, loading.temperature_K, **kwargs
        )
    full = max(action["full"], 0.0)
    return {
        "frozen_full_action": full,
        "frozen_initial_radius_action": action["initial_radius"],
        "frozen_zero_shield_action": action["zero_shield"],
        "frozen_virgin_geometry_action": action["virgin_geometry"],
        "frozen_raw_one_hit_action": raw_action,
        "H_full_over_H_initial_radius": full / max(action["initial_radius"], 1.0e-300),
        "H_full_over_H_zero_shield": full / max(action["zero_shield"], 1.0e-300),
        "H_full_over_H_virgin_geometry": full / max(action["virgin_geometry"], 1.0e-300),
        "H_three_hit_over_H_raw": full / max(raw_action, 1.0e-300),
        "minimum_raw_arrhenius_barrier_eV": peak["full"]["raw_arrhenius_barrier_eV"],
        "minimum_production_equivalent_barrier_eV": peak["full"]["production_equivalent_barrier_eV"],
        "peak_raw_one_hit_rate_s": peak["full"]["raw_one_hit_rate_s"],
        "peak_production_three_hit_rate_s": peak["full"]["production_rate_s"],
        "peak_cleavage_stress_GPa": peak["full"]["cleavage_stress_Pa"] / 1.0e9,
    }


def glide_stress_audit(state, loading) -> dict[str, float]:
    """Saved-state signed applied/local-GND glide-stress decomposition."""
    drive = np.asarray(state.emission_drive_factors(), dtype=float)
    signs = np.where(drive >= 0.0, 1.0, -1.0)
    rows = []
    for phase in np.linspace(0.0, 1.0, 257):
        K = float(loading.K_at_phase(float(phase)))
        rates = state.local_rates(K, loading.temperature_K)
        fields = intrinsic_signed_glide_stress_fields(
            K, state.tip_radius_m(), drive, np.asarray(rates["tau_gnd_Pa"], dtype=float)
        )
        applied = np.asarray(fields["tau_applied_Pa"]) * signs[:, None]
        internal = np.asarray(rates["tau_gnd_Pa"]) * signs[:, None]
        effective = np.asarray(fields["tau_effective_Pa"]) * signs[:, None]
        rows.append((applied, internal, effective))
    return {
        "tau_applied_projected_min_GPa": min(float(np.min(x[0])) for x in rows) / 1.0e9,
        "tau_applied_projected_max_GPa": max(float(np.max(x[0])) for x in rows) / 1.0e9,
        "tau_internal_projected_min_GPa": min(float(np.min(x[1])) for x in rows) / 1.0e9,
        "tau_internal_projected_max_GPa": max(float(np.max(x[1])) for x in rows) / 1.0e9,
        "tau_glide_projected_min_GPa": min(float(np.min(x[2])) for x in rows) / 1.0e9,
        "tau_glide_projected_max_GPa": max(float(np.max(x[2])) for x in rows) / 1.0e9,
        "reverse_glide_state_fraction": float(np.mean(np.concatenate([x[2].ravel() for x in rows]) < 0.0)),
    }


def cleavage_ceiling(candidate, physics, temperature_K: float, frequency_Hz: float) -> dict[str, Any]:
    floor = exp_floor_limit_eV(candidate.cleavage, temperature_K)
    raw = float(physics.cleavage_nu0_s) * math.exp(
        -floor / (KB_EV_PER_K * float(temperature_K))
    )
    exposure = min(raw * float(physics.cleavage_correlation_time_s), 1.0e12)
    production = (
        raw if float(physics.cleavage_hits) <= 1.0 + 1.0e-12
        else float(gammainc(float(physics.cleavage_hits), exposure)
                   / float(physics.cleavage_correlation_time_s))
    )
    action = production / float(frequency_Hz)
    waits = [1.0e3, 1.0e5, 1.0e7, 1.0e9, 1.0e12, 1.0e14]
    return {
        "temperature_K": float(temperature_K),
        "G_zero_stress_eV": float(candidate.cleavage.zero_stress_eV(temperature_K)),
        "G_floor_eV": floor,
        "nu0_s": float(physics.cleavage_nu0_s),
        "n_hits": float(physics.cleavage_hits),
        "tau_c_s": float(physics.cleavage_correlation_time_s),
        "Gamma_raw_max_s": raw,
        "raw_exposure_per_correlation_time": exposure,
        "Gamma_c_max_s": production,
        "maximum_action_per_cycle": action,
        "minimum_mean_wait_cycles": 1.0 / action,
        "waiting_time_comparison": [
            {
                "mean_wait_cycles": value,
                "required_action_per_cycle": 1.0 / value,
                "ceiling_to_required_ratio": action * value,
            }
            for value in waits
        ],
    }


def geometry_audit(candidate, physics, Kmax_values, temperature_K: float) -> dict[str, Any]:
    dx = float(physics.mpz_length_m) / int(physics.n_bins)
    strip_width = max(float(physics.blunting_length_m), dx, abs(float(physics.b_m)))
    centers = (np.arange(int(physics.n_bins), dtype=float) + 0.5) * dx
    source_bins = max(
        min(int(math.ceil(float(physics.source_zone_length_m) / dx)), int(physics.n_bins)),
        1,
    )
    source_weight = float(np.mean(np.exp(-centers[:source_bins] / strip_width)))
    unweighted_per_activation = (
        float(candidate.c_blunt) * abs(float(physics.b_m))
        * float(physics.blunting_slip_fraction)
    )
    conversions = np.asarray(physics.activation_to_line_content_per_system, dtype=float)
    weighted_per_activation = unweighted_per_activation * source_weight * conversions
    sigma_flow = float(candidate.emission.characteristic_stress_Pa(temperature_K))
    young = 2.0 * float(physics.G_Pa) * (1.0 + float(physics.nu))
    eprime = young / (1.0 - float(physics.nu) ** 2)
    references = []
    for K in Kmax_values:
        K_si = float(K) * 1.0e6
        references.append({
            "Kmax_MPa_sqrt_m": float(K),
            "rp_plane_stress_m": (K_si / sigma_flow) ** 2 / (2.0 * math.pi),
            "rp_plane_strain_m": (K_si / sigma_flow) ** 2 / (6.0 * math.pi),
            "CTOD_reference_m": K_si * K_si / (eprime * sigma_flow),
        })
    return {
        "mapping": "r_tip=r0+c_blunt*b*blunting_slip_fraction*weighted_source_line_count",
        "source_slip_count_units": "dimensionless line count",
        "one_activation_source_slip_count_before_tip_weight": conversions.tolist(),
        "one_activation_weighted_source_slip_count": (source_weight * conversions).tolist(),
        "one_activation_crack_tip_displacement_m": abs(float(physics.b_m)),
        "one_activation_unweighted_geometric_increment_m": (
            unweighted_per_activation * conversions
        ).tolist(),
        "one_activation_effective_CTOD_increment_m": weighted_per_activation.tolist(),
        "one_activation_tip_radius_increment_m": weighted_per_activation.tolist(),
        "source_zone_bin_count": source_bins,
        "source_zone_mean_tip_weight": source_weight,
        "dx_m": dx,
        "tip_weight_decay_length_m": strip_width,
        "c_blunt": float(candidate.c_blunt),
        "b_m": float(physics.b_m),
        "blunting_slip_fraction": float(physics.blunting_slip_fraction),
        "flow_stress_definition": "weak-T emission EXP characteristic stress at T (immutable model surface)",
        "sigma_flow_Pa": sigma_flow,
        "E_Pa": young,
        "Eprime_Pa": eprime,
        "dimensional_references": references,
    }


def write_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a compact visual of the causal decomposition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = sorted({(float(row["Kmax_MPa_sqrt_m"]), float(row["R"])) for row in rows})
    for Kmax, R in cases:
        case = sorted(
            (row for row in rows
             if float(row["Kmax_MPa_sqrt_m"]) == Kmax and float(row["R"]) == R),
            key=lambda row: int(row["cycle"]),
        )
        cycle = np.asarray([int(row["cycle"]) for row in case])
        label = f"Kmax={Kmax:g}, R={R:g}"
        axes[0, 0].plot(cycle, [1e6 * row["tip_radius_m"] for row in case], "o-", label=label)
        axes[0, 1].semilogy(
            cycle, [max(row["frozen_full_action"], 1e-300) for row in case], "o-", label=label
        )
        axes[1, 0].semilogy(
            cycle,
            [1.0 / max(row["H_full_over_H_initial_radius"], 1e-300) for row in case],
            "o-", label=label,
        )
        axes[1, 1].semilogy(
            cycle,
            [1.0 / max(row["H_three_hit_over_H_raw"], 1e-300) for row in case],
            "o-", label=label,
        )
    axes[0, 0].set(ylabel="tip radius (µm)", xlabel="cycle")
    axes[0, 1].set(ylabel="frozen-state full action/cycle", xlabel="cycle")
    axes[1, 0].set(ylabel="initial-radius / full action", xlabel="cycle")
    axes[1, 1].set(ylabel="raw one-hit / three-hit action", xlabel="cycle")
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Weak-T v7 exact-state causal cleavage suppression")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def state_row(state, loading, cycle: int, physical_cycle_hazard: float | None,
              previous: dict[str, float]) -> dict[str, Any]:
    cumulative_source = float(np.sum(state.cumulative_source_activations))
    gross_source = float(np.sum(state.accumulated_slip_m2) * state.cell_area_m2)
    gross_return = float(np.sum(state.returned_slip_m2) * state.cell_area_m2)
    physical_return = float(state.returned_source_slip_count())
    diagnostic = state.reversibility_diagnostics()
    row = {
        "cycle": int(cycle),
        "physical_cycle_hazard_action": physical_cycle_hazard,
        "cumulative_source_activations": cumulative_source,
        "source_emission_increment": cumulative_source - previous.get("source", cumulative_source),
        "gross_source_slip_line_content": gross_source,
        "gross_source_slip_increment": gross_source - previous.get("gross_source", gross_source),
        "gross_returned_slip_line_content": gross_return,
        "gross_returned_slip_increment": gross_return - previous.get("gross_return", gross_return),
        "physical_source_return_count": physical_return,
        "physical_source_return_increment": physical_return - previous.get("physical_return", physical_return),
        "net_source_slip_count": float(state.local_accumulated_slip_count()),
        "tip_radius_m": float(state.tip_radius_m()),
        "mobile_line_content": float(np.sum(state.mobile_m2) * state.cell_area_m2),
        "retained_line_content": float(np.sum(state.retained_m2) * state.cell_area_m2),
        "Kshield_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
        "reverse_mobile_exposure_fraction": float(diagnostic.get(
            "reversible_reverse_mobile_exposure_fraction", 0.0
        )),
        "signed_tip_plastic_displacement_proxy_m": float(
            state.cumulative_signed_tip_plastic_displacement_proxy_m
        ),
        "reverse_tip_plastic_displacement_proxy_m": float(
            state.cumulative_reverse_tip_plastic_displacement_proxy_m
        ),
    }
    row.update(frozen_cycle_causal_audit(state, loading))
    row.update(glide_stress_audit(state, loading))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--candidate", default="v913_zeroD_sobol_0129902")
    parser.add_argument("--Kmax", default="12,30,62.60,70.77")
    parser.add_argument("--R-values", default="0.1")
    parser.add_argument("--return-control", choices=("REV", "IRREV"), default="REV")
    parser.add_argument(
        "--configuration-label",
        choices=("HISTORICAL", "CAL-TRANS-IRREV", "CAL-TRANS-REV", "LEGACY1D-V7"),
        required=True,
    )
    parser.add_argument("--matched-R-Kmax", type=float, default=30.0)
    parser.add_argument("--matched-R", type=float, default=-0.95)
    parser.add_argument("--sample-cycles", default="0,1,2,4,8,12")
    parser.add_argument("--frequency-Hz", type=float, default=1000.0)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--base-phase-intervals", type=int, default=32)
    parser.add_argument("--coupled-substeps", type=int, default=2)
    parser.add_argument("--max-refinement-depth", type=int, default=22)
    parser.add_argument("--reuse-rows-json", type=Path, default=None,
                        help="Reuse completed exact rows and regenerate only analytical metadata.")
    parser.add_argument("--expected-head", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if args.expected_head and head != args.expected_head:
        raise SystemExit(f"HEAD mismatch: expected {args.expected_head}, found {head}")
    if args.expected_head and dirty:
        raise SystemExit("authoritative launch requires a clean worktree")
    # Fail before expensive exact-cycle work if the audited output location
    # cannot be created (for example, a full production volume).
    args.out.mkdir(parents=True, exist_ok=True)

    registry_rows = list(csv.DictReader(args.registry.open()))
    registry_row = next(row for row in registry_rows if row["candidate_id"] == args.candidate)
    candidate = candidate_from_registry_row(registry_row)
    common = replace(_common(args.physics), n_bins=int(args.n_bins))
    physics = physics_for_row(common, registry_row)
    state_class = (
        DiagnosticIntrinsicReverseGlideState
        if args.return_control == "REV"
        else DiagnosticIntrinsicReverseGlideNoReturnState
    )
    state_class.coupled_operator_substeps = int(args.coupled_substeps)
    controls = AdaptiveFeedbackControls(
        state_rtol=0.0025, tip_radius_rtol=0.001, hazard_rtol=0.01,
        base_phase_intervals=int(args.base_phase_intervals),
        max_refinement_depth=int(args.max_refinement_depth),
    )
    sample_cycles = sorted(set(int(x) for x in args.sample_cycles.split(",")))
    if not sample_cycles or sample_cycles[0] != 0:
        raise ValueError("sample cycles must include cycle zero")
    K_values = [float(x) for x in args.Kmax.split(",")]
    cases = [(K, float(R)) for R in args.R_values.split(",") for K in K_values]
    matched = (float(args.matched_R_Kmax), float(args.matched_R))
    if matched not in cases:
        cases.append(matched)

    all_rows = []
    if args.reuse_rows_json is not None:
        reused = json.loads(args.reuse_rows_json.read_text())
        if reused.get("candidate_id") != args.candidate:
            raise ValueError("reused causal-audit candidate mismatch")
        all_rows = list(reused["rows"])
    for Kmax, R in ([] if args.reuse_rows_json is not None else cases):
        loading = SignedFatigueLoading(
            Kmax * (1.0 - R), R=R, frequency_Hz=args.frequency_Hz,
            temperature_K=args.temperature_K,
            phase_steps=controls.base_phase_intervals,
        )
        state = state_class(candidate, physics)
        previous = {}
        hazards: dict[int, float] = {}
        for cycle in range(0, sample_cycles[-1] + 1):
            if cycle > 0:
                state, hazard, _telemetry = advance_v7_cycle(state, loading, controls)
                hazards[cycle] = float(hazard)
            if cycle in sample_cycles:
                row = state_row(state, loading, cycle, hazards.get(cycle), previous)
                row.update({"Kmax_MPa_sqrt_m": Kmax, "R": R})
                all_rows.append(row)
                previous = {
                    "source": row["cumulative_source_activations"],
                    "gross_source": row["gross_source_slip_line_content"],
                    "gross_return": row["gross_returned_slip_line_content"],
                    "physical_return": row["physical_source_return_count"],
                }
                print(
                    f"Kmax={Kmax:g} R={R:g} cycle={cycle} "
                    f"r={row['tip_radius_m']*1e6:.6g}um "
                    f"Hfull={row['frozen_full_action']:.6g}", flush=True
                )

    with (args.out / "causal_physics_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    payload = {
        "schema": SCHEMA,
        "candidate_id": args.candidate,
        "configuration_label": args.configuration_label,
        "return_control": args.return_control,
        "state_model_id": state_class.MODEL_ID,
        "trajectory_authority": "unmodified_exact_v7_adaptive_cycle_map",
        "counterfactuals_feed_back_to_state": False,
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "physics_sha256": hashlib.sha256(args.physics.read_bytes()).hexdigest(),
        "resolution": {
            "n_bins": args.n_bins,
            "base_phase_intervals": args.base_phase_intervals,
            "coupled_substeps": args.coupled_substeps,
            "max_refinement_depth": args.max_refinement_depth,
        },
        "cleavage_ceiling": cleavage_ceiling(
            candidate, physics, args.temperature_K, args.frequency_Hz
        ),
        "source_slip_tip_radius_geometry": geometry_audit(
            candidate, physics, K_values, args.temperature_K
        ),
        "rows": all_rows,
    }
    (args.out / "causal_physics_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_summary_plot(all_rows, args.out / "causal_physics_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
