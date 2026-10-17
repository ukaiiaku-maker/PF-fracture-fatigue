#!/usr/bin/env python3
"""Finite-state 2-D cohesive trigger with and without EXP_floor feedback.

The plastic state is evolved by the retained native engine.  Its established
reduced mechanical shielding operator is then consumed by the spatial
interface as a signed closure traction.  The control has the identical saved
state and geometry, but disables that one coupling term.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (  # noqa: E402
    BILINEAR, SMOOTH_EXPONENTIAL, COHESIVE_ATHERMAL_MECHANICAL,
    COHESIVE_KINETIC_CZM,
    mechanical_completion,
)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY  # noqa: E402
from arrhenius_fracture.fatigue_v1 import FatigueWaveform  # noqa: E402
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (  # noqa: E402
    restore_active_state, serialize_active_state,
)
from arrhenius_fracture.retained_material_cohesive_r10 import (  # noqa: E402
    build_cohesive_current_engine, load_cohesive_binding,
)
from scripts.qualify_r10_cohesive_2d import equilibrium, specimen  # noqa: E402


def atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def evolve(engine, Kmax_MPa_sqrt_m: float, temperature_K: float,
           cycles: int, phases: int) -> list[dict]:
    wave = FatigueWaveform(Kmax=Kmax_MPa_sqrt_m * 1e6, R=0.1, frequency_Hz=1000.)
    dt = wave.period_s / phases
    history = []
    for cycle in range(cycles):
        for phase in range(phases):
            angle = 2 * math.pi * (phase + 0.5) / phases
            K = float(wave.K_phase(angle))
            if hasattr(engine.mpz, "_reversible_transport_K_signed_Pa_sqrt_m"):
                engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = K
                engine.mpz._reversible_tip_radius_m = float(engine.r_eff())
            sigma = max(float(engine.sigma_tip(max(K, 0.))), 0.)
            engine._plastic_half_step(0.5 * dt, temperature_K, sigma)
            sigma = max(float(engine.sigma_tip(max(K, 0.))), 0.)
            engine._plastic_half_step(0.5 * dt, temperature_K, sigma)
            engine.t = float(engine.t) + dt
        history.append({
            "cycle": cycle + 1,
            "physical_time_s": float(engine.t),
            "mobile_count": float(engine.mpz.mobile_count),
            "retained_count": float(engine.mpz.retained_count),
            "K_shield_Pa_sqrt_m": float(engine.K_shield()),
            "tip_radius_m": float(engine.r_eff()),
            "source_backstress_Pa": float(getattr(
                engine.mpz, "continuum_source_last_sigma_back_Pa", 0.)),
        })
    return history


def solve_path(thermal, temperature_K: float, shape: str,
               closure_Pa: float, step_delta0: float,
               maximum_delta0: float) -> dict:
    mesh, network, elem = specimen(thermal)
    elem.temperature_K = temperature_K
    elem.cohesive_shape = shape
    network.execution_identity = COHESIVE_ATHERMAL_MECHANICAL
    network.plastic_closure_Pa = closure_Pa
    u = None
    accepted = {"factor_delta0": 0., "u": None, "diagnostics": None}
    failed = None
    trace = []
    for factor in np.arange(0., maximum_delta0 + 0.5 * step_delta0, step_delta0):
        try:
            u, diag = equilibrium(mesh, network, float(factor * thermal.delta0_m), u)
            complete = bool(mechanical_completion(
                shape, max(float(diag["normal_jump_m"]), 0.), thermal,
                temperature_K)["complete"])
            row = {"factor_delta0": float(factor), **diag,
                   "mechanical_completion": complete}
            trace.append(row)
            if complete:
                failed = {"factor_delta0": float(factor), "kind": "MECHANICAL_COMPLETION"}
                break
            accepted = {"factor_delta0": float(factor), "u": u.copy(),
                        "diagnostics": diag}
        except RuntimeError as exc:
            failed = {"factor_delta0": float(factor), "kind": "FE_BRANCH_LOSS",
                      "error": str(exc)}
            break
    if failed is None:
        return {"trigger": "RAMP_LIMIT_CENSOR", "bracket_delta0": None,
                "trace": trace}

    lo = float(accepted["factor_delta0"])
    hi = float(failed["factor_delta0"])
    # Path-following bisection: every trial starts from the last accepted lower
    # state, so a failed upper solve is never reused as an equilibrium state.
    ulo = accepted["u"]
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        try:
            umid, diag = equilibrium(mesh, network, mid * thermal.delta0_m, ulo)
            complete = bool(mechanical_completion(
                shape, max(float(diag["normal_jump_m"]), 0.), thermal,
                temperature_K)["complete"])
        except RuntimeError:
            complete = True
            umid = None
        if complete:
            hi = mid
        else:
            lo = mid
            ulo = umid
    return {"trigger": failed["kind"], "bracket_delta0": [lo, hi],
            "bracket_width_delta0": hi - lo,
            "last_supported_diagnostics": accepted["diagnostics"], "trace": trace}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--shape", choices=(SMOOTH_EXPONENTIAL, BILINEAR),
                   default=SMOOTH_EXPONENTIAL)
    p.add_argument("--temperature-K", type=float, default=300.)
    p.add_argument("--Kmax", type=float, required=True)
    p.add_argument("--cycles", type=int, default=32)
    p.add_argument("--phases", type=int, default=8)
    p.add_argument("--step-delta0", type=float, default=0.05)
    p.add_argument("--maximum-delta0", type=float, default=20.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    with args.registry.open(newline="") as stream:
        row = next(r for r in csv.DictReader(stream)
                   if r["candidate_id"] == args.candidate_id)
    binding = load_cohesive_binding(
        args.registry, args.candidate_id, execution_identity=COHESIVE_KINETIC_CZM)
    engine = build_cohesive_current_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    initial = serialize_active_state(engine)
    history = evolve(engine, args.Kmax, args.temperature_K, args.cycles, args.phases)
    developed = serialize_active_state(engine)

    # Prove that the saved developed state is exactly restorable before it is
    # used to generate the coupling observable.
    replay = build_cohesive_current_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    restore_active_state(replay, developed)
    replay_state = serialize_active_state(replay)
    exact_restore = bool(np.array_equal(developed.vector, replay_state.vector))
    if not exact_restore:
        raise RuntimeError("developed active-state restore was not exact")

    r_eff = float(replay.r_eff())
    K_shield = float(replay.K_shield())
    # Transfer the state effect already owned by the retained reduced mechanics,
    # including both shielding and blunting.  Using the A0-to-current stress
    # discrepancy avoids inventing a line-count-to-eigenstrain conversion.
    reference = build_cohesive_current_engine(
        binding, kernel_family_path=R4_KERNEL_FAMILY)
    applied_K = args.Kmax * 1e6
    sigma_A0 = float(reference.sigma_tip(applied_K))
    sigma_current = float(replay.sigma_tip(applied_K))
    sigma_closure = sigma_A0 - sigma_current
    thermal = binding.manifest.cleavage.thermal
    cases = []
    # Positive normal cohesive traction is the closing/resisting sign in the
    # interface residual (it is the derivative of stored surface energy).
    for label, closure in (("FULL_STATE_FEEDBACK", sigma_closure),
                           ("IDENTICAL_STATE_FEEDBACK_DISABLED", 0.)):
        result = solve_path(thermal, args.temperature_K, args.shape, closure,
                            args.step_delta0, args.maximum_delta0)
        result.update({"case": label, "plastic_closure_traction_Pa": closure,
                       "same_saved_state_sha256": None})
        cases.append(result)

    np.savez_compressed(args.out / "active_states.npz",
                        initial_vector=initial.vector,
                        developed_vector=developed.vector,
                        replay_vector=replay_state.vector)
    summary = {
        "schema": "r10.opening-finite-state-2d-feedback/1",
        "candidate_id": args.candidate_id,
        "same_complete_material_sha256": row["complete_row_sha256"],
        "paired_plasticity_sha256": row["parent_complete_row_sha256"],
        "temperature_K": args.temperature_K,
        "Kmax_MPa_sqrt_m": args.Kmax,
        "cycles": args.cycles,
        "phases_per_cycle": args.phases,
        "shape": args.shape,
        "state_restore_bit_exact": exact_restore,
        "state_vector_size": int(developed.vector.size),
        "state_vector_l2_change": float(np.linalg.norm(developed.vector - initial.vector)),
        "K_shield_Pa_sqrt_m": K_shield,
        "tip_radius_m": r_eff,
        "sigma_tip_A0_Pa": sigma_A0,
        "sigma_tip_current_Pa": sigma_current,
        "closure_operator": "sigma_closure=sigma_tip(A0,K)-sigma_tip(current_state,K)",
        "closure_magnitude_Pa": sigma_closure,
        "feedback_scope": "RETAINED_REDUCED_MECHANICAL_OPERATOR; no line-count-to-strain inference",
        "old_2d_evidence_disposition": "ONE_WAY_STATE_EVOLUTION_ONLY; superseded for feedback claims",
        "history": history,
        "cases": cases,
    }
    atomic(args.out / "feedback_qualification.json", summary)
    print(json.dumps({"candidate_id": args.candidate_id,
                      "closure_magnitude_Pa": sigma_closure,
                      "state_restore_bit_exact": exact_restore,
                      "triggers": [{"case": c["case"], "trigger": c["trigger"],
                                    "bracket_delta0": c["bracket_delta0"]}
                                   for c in cases]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
