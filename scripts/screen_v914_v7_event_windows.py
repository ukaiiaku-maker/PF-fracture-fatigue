#!/usr/bin/env python3
"""Deterministically bracket useful weak-T first-passage cycle windows.

This is a loading-amplitude screen, not a fatigue-growth law.  Each amplitude
starts from the same virgin state, resolves the authoritative v7 cycle map
through its initial transient, and estimates the developed cycle action from
the geometric mean of the final exact cycles.  Log-amplitude bisection targets
specified expected waiting-cycle windows without consuming stochastic RNG.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
import math
import os
from pathlib import Path
import sys

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
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_intrinsic_reverse_glide_v7 import IntrinsicReverseGlideState
from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_v7_cycle_map import advance_v7_cycle


SCHEMA = "v914_v7_deterministic_event_window_screen_v1"


def _safe_screen_evaluation(evaluate_fn, *, Kmax: float, R: float) -> dict:
    try:
        result = dict(evaluate_fn())
        result["status"] = "resolved"
        return result
    except (RuntimeError, FloatingPointError) as exc:
        return {
            "Kmax_MPa_sqrt_m": float(Kmax),
            "deltaK_MPa_sqrt_m": float(Kmax) * (1.0 - float(R)),
            "status": "numerical_rejection",
            "failure": str(exc),
            "expected_waiting_cycles": math.nan,
        }


def _common(path: Path) -> CommonPhysics:
    values = json.loads(path.read_text())
    values = values.get("common_physics", values)
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


def _evaluate(candidate, physics, *, Kmax, R, frequency, temperature, controls, cycles):
    deltaK = float(Kmax) * (1.0 - float(R))
    loading = SignedFatigueLoading(
        deltaK, R=R, frequency_Hz=frequency,
        temperature_K=temperature, phase_steps=controls.base_phase_intervals,
    )
    state = IntrinsicReverseGlideState(candidate, physics)
    hazards = []
    for _ in range(int(cycles)):
        state, hazard, _telemetry = advance_v7_cycle(state, loading, controls)
        hazards.append(max(float(hazard), 1.0e-300))
    tail = hazards[max(len(hazards) // 2, 0):]
    developed = math.exp(sum(math.log(value) for value in tail) / len(tail))
    diagnostics = state.reversibility_diagnostics()
    return {
        "Kmax_MPa_sqrt_m": float(Kmax),
        "deltaK_MPa_sqrt_m": deltaK,
        "resolved_transient_cycles": int(cycles),
        "first_cycle_hazard_action": hazards[0],
        "final_cycle_hazard_action": hazards[-1],
        "developed_cycle_hazard_action": developed,
        "expected_waiting_cycles": 1.0 / developed,
        "tip_radius_m": float(state.tip_radius_m()),
        "K_shield_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
        "mobile_density_sum_m2": float(state.mobile_m2.sum()),
        "retained_density_sum_m2": float(state.retained_m2.sum()),
        "net_source_slip_count": float(state.local_accumulated_slip_count()),
        "physical_return_fraction": float(
            diagnostics.get("reversible_physical_return_fraction_of_emitted", 0.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--candidate", default="v913_zeroD_sobol_0129902")
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--frequency-Hz", type=float, default=1000.0)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--Kmax-min", type=float, default=12.0)
    parser.add_argument("--Kmax-max", type=float, default=30.0)
    parser.add_argument("--transient-cycles", type=int, default=16)
    parser.add_argument("--maximum-amplitudes", type=int, default=10)
    parser.add_argument("--target-windows", default="1e3,1e5,1e7,1e9,1e12")
    parser.add_argument("--seed-screen-json", type=Path, default=None)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--base-phase-intervals", type=int, default=32)
    parser.add_argument("--coupled-substeps", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.registry.open()))
    row = next(item for item in rows if item["candidate_id"] == args.candidate)
    candidate = candidate_from_registry_row(row)
    common = replace(_common(args.physics), n_bins=int(args.n_bins))
    physics = physics_for_row(common, row)
    IntrinsicReverseGlideState.coupled_operator_substeps = int(args.coupled_substeps)
    controls = AdaptiveFeedbackControls(
        state_rtol=0.0025, tip_radius_rtol=0.001, hazard_rtol=0.01,
        base_phase_intervals=int(args.base_phase_intervals), max_refinement_depth=18,
    )
    targets = sorted(float(item) for item in args.target_windows.split(","))
    evaluated: dict[float, dict] = {}
    if args.seed_screen_json is not None:
        prior = json.loads(args.seed_screen_json.read_text())
        for item in prior.get("rows", []):
            restored = dict(item)
            restored.setdefault("status", "resolved")
            evaluated[float(item["Kmax_MPa_sqrt_m"])] = restored

    def evaluate(value: float):
        key = float(value)
        if key not in evaluated:
            row_result = _safe_screen_evaluation(
                lambda: _evaluate(
                    candidate, physics, Kmax=key, R=args.R,
                    frequency=args.frequency_Hz, temperature=args.temperature_K,
                    controls=controls, cycles=args.transient_cycles,
                ), Kmax=key, R=args.R,
            )
            evaluated[key] = row_result
            print(f"Kmax={key:.8g} status={row_result['status']} " + (
                f"developed_Hcyc={row_result['developed_cycle_hazard_action']:.6g} "
                f"expected_N={row_result['expected_waiting_cycles']:.6g}"
                if row_result["status"] == "resolved" else row_result["failure"]
            ), flush=True)
        return evaluated[key]

    evaluate(args.Kmax_min)
    evaluate(args.Kmax_max)
    # Greedy log-space bisection: refine the target with the largest current
    # log-cycle error that is bracketed by evaluated amplitudes.
    while len(evaluated) < int(args.maximum_amplitudes):
        ordered = sorted(evaluated.values(), key=lambda item: item["Kmax_MPa_sqrt_m"])
        rejected_pairs = [
            (left, right) for left, right in zip(ordered[:-1], ordered[1:])
            if (left["status"] == "resolved") != (right["status"] == "resolved")
        ]
        if rejected_pairs:
            left, right = max(
                rejected_pairs,
                key=lambda pair: math.log(pair[1]["Kmax_MPa_sqrt_m"] / pair[0]["Kmax_MPa_sqrt_m"]),
            )
            midpoint = math.sqrt(
                left["Kmax_MPa_sqrt_m"] * right["Kmax_MPa_sqrt_m"]
            )
            evaluate(midpoint)
            continue
        resolved = [item for item in ordered if item["status"] == "resolved"]
        best = None
        for target in targets:
            for left, right in zip(resolved[:-1], resolved[1:]):
                a = math.log10(left["expected_waiting_cycles"])
                b = math.log10(right["expected_waiting_cycles"])
                t = math.log10(target)
                if (a - t) * (b - t) <= 0.0:
                    score = abs(a - b)
                    if best is None or score > best[0]:
                        best = (score, left, right)
        if best is None:
            # No target bracket yet: bisect the widest amplitude interval.
            left, right = max(
                zip(resolved[:-1], resolved[1:]),
                key=lambda pair: math.log(pair[1]["Kmax_MPa_sqrt_m"] / pair[0]["Kmax_MPa_sqrt_m"]),
            )
        else:
            _score, left, right = best
        midpoint = math.sqrt(
            left["Kmax_MPa_sqrt_m"] * right["Kmax_MPa_sqrt_m"]
        )
        if midpoint in evaluated:
            break
        evaluate(midpoint)

    output = sorted(evaluated.values(), key=lambda item: item["Kmax_MPa_sqrt_m"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "event_window_screen.csv").open("w", newline="") as stream:
        fieldnames = []
        for item in output:
            for key in item:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(output)
    payload = {
        "schema": SCHEMA, "candidate_id": args.candidate, "R": args.R,
        "frequency_Hz": args.frequency_Hz, "temperature_K": args.temperature_K,
        "targets_expected_cycles": targets, "development_resolution": {
            "n_bins": args.n_bins, "base_phase_intervals": args.base_phase_intervals,
            "coupled_substeps": args.coupled_substeps,
        }, "rows": output,
    }
    (args.out / "event_window_screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
