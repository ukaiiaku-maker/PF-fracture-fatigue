"""PX3.5 section 5: localize the frequency transition via a prospectively
frozen log-frequency bisection between 100 Hz and 1000 Hz (per external
review's exact decision, since the analytically-preselected 100 Hz showed
no live measurable effect while 10000 Hz merely repeated the 1000 Hz
baseline -- neither satisfies mission section 7.8's rule 2).

Deterministic procedure (frozen BEFORE the first of these trajectories
runs, per the review's explicit instruction):

  f1 = sqrt(100 * 1000) = 316.227766 Hz

  Accept f if BOTH:
    |S_h(f)| >= 0.01
    |S_h(f) - S_h(1000 Hz)| >= 0.01

  If f1 is below the first gate (no measurable effect): move UP toward
  1000 Hz -> f2 = sqrt(f1 * 1000) = 562.341325 Hz.
  If f1 clears the first gate but is too close to the 1kHz plateau (fails
  the second gate): move DOWN toward 100 Hz -> f2 = sqrt(100 * f1)
  = 177.827941 Hz.

  At most 3 new reversible finite/zero pairs total. Select the FIRST
  frequency (in evaluation order) satisfying both gates. If none does,
  record FREQUENCY_TRANSITION_BRACKETED_BUT_NOT_LOCALIZED and omit D3.

Each new pair reuses the exact PX3 screen conditions (R=-0.5, hold=0,
Kmax=18 MPa sqrt(m), seed=1720, COMPETING_REVERSIBLE, 12 accepted events
or 60 um) via the same qualified causal-pilot-v2 event loop PX3 itself
used -- this is a protocol AMENDMENT (a new, prospectively bounded set of
conditions), not a new mechanism, so no other machinery changes.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import os
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"px3_5_freq_bisect_{os.getpid()}_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)
from part_x_run_one_job import _load_row_payload, _resolve_screen_rebonding_cfg  # noqa: E402

SEED = 1720
R = -0.5
KMAX_PA_SQRT_M = 18.0e6
MAX_ACCEPTED_EVENTS = 12
MAX_PROJECTED_EXTENSION_M = 60.0e-6
MEASURABLE_GATE = 0.01
MAX_NEW_PAIRS = 3


def _load_S_h_1000Hz_baseline() -> float:
    """Read the ALREADY-COMMITTED PX3 screen's own 7.1_R_panel R=-0.5
    (f=1000 Hz, hold=0) pair's S_h_all directly from px3_screen_pair_
    analysis.json, rather than hardcoding a copied digit string that could
    silently drift out of sync with that file."""
    pairs = json.loads((ARTIFACTS_DIR / "px3_screen_pair_analysis.json").read_text())["pairs"]
    matches = [p for p in pairs if p["protocol"] == "7.1_R_panel" and p["R"] == -0.5]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one 7.1_R_panel R=-0.5 pair, found {len(matches)}")
    return float(matches[0]["S_h_all"])


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def _resolve_reversible_configs(Eprime_Pa: float):
    row_payload = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]["COMPETING_REVERSIBLE"]
    finite_job = {"cohesion": "finite", "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0}
    zero_job = {"cohesion": "zero", "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 0.0}
    return (
        _resolve_screen_rebonding_cfg(finite_job, row_payload, Eprime_Pa),
        _resolve_screen_rebonding_cfg(zero_job, row_payload, Eprime_Pa),
    )


def run_pair(frequency_Hz: float, finite_cfg, zero_cfg) -> dict:
    trajectories = {}
    for label, cfg in [("finite", finite_cfg), ("zero", zero_cfg)]:
        result_dir = RUN_ROOT / f"px3_5_freq_bisection_f{frequency_Hz:.6f}Hz_{label}"
        if result_dir.exists():
            raise RuntimeError(f"refusing to overwrite existing frequency-bisection result: {result_dir}")
        Engine.configure_hazard(mode="exponential", seed=SEED)
        Engine.reset_audit()
        trajectory = pilot.run_trajectory(
            name=f"freq_bisect_{frequency_Hz:.6f}Hz_{label}", build_engine=_build_engine,
            make_controller=_make_controller, waveform_cls=FatigueWaveform,
            rebonding_cfg=cfg, R=R, reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=KMAX_PA_SQRT_M, frequency_Hz=frequency_Hz, n_phase=80, T_K_=300.0,
            max_accepted_events=MAX_ACCEPTED_EVENTS, max_projected_extension_m=MAX_PROJECTED_EXTENSION_M,
            hazard_rng_seed=SEED, minimum_load_hold_s=0.0,
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "result.json").write_text(json.dumps(
            {"schema": "v10230_part_x_px3_5_frequency_bisection_leg_v1", "frequency_Hz": frequency_Hz,
             "cohesion": label, "trajectory": trajectory}, indent=2, default=str,
        ))
        trajectories[label] = trajectory

    g_finite = trajectories["finite"]["cumulative_extension_m"] / trajectories["finite"]["cumulative_cycles"]
    g_zero = trajectories["zero"]["cumulative_extension_m"] / trajectories["zero"]["cumulative_cycles"]
    S_h = math.log10(g_finite / g_zero)
    return {
        "frequency_Hz": frequency_Hz, "g_finite": g_finite, "g_zero": g_zero, "S_h": S_h,
        "n_events_finite": trajectories["finite"]["n_accepted_events"],
        "n_events_zero": trajectories["zero"]["n_accepted_events"],
        "censored_finite": trajectories["finite"]["censored"], "censored_zero": trajectories["zero"]["censored"],
    }


def main() -> None:
    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    finite_cfg, zero_cfg = _resolve_reversible_configs(Eprime_Pa)
    S_H_1000HZ_BASELINE = _load_S_h_1000Hz_baseline()

    evaluated = []
    f = math.sqrt(100.0 * 1000.0)
    localized = None
    while len(evaluated) < MAX_NEW_PAIRS:
        result = run_pair(f, finite_cfg, zero_cfg)
        evaluated.append(result)
        measurable = abs(result["S_h"]) >= MEASURABLE_GATE
        distinct_from_1000 = abs(result["S_h"] - S_H_1000HZ_BASELINE) >= MEASURABLE_GATE
        result["measurable_effect_gate"] = measurable
        result["distinct_from_1000Hz_gate"] = distinct_from_1000
        print(f"f={f:.6f} Hz  S_h={result['S_h']:.6f}  measurable={measurable}  distinct_from_1000Hz={distinct_from_1000}")
        if measurable and distinct_from_1000:
            localized = result
            break
        if not measurable:
            f = math.sqrt(f * 1000.0)  # move up toward the 1kHz plateau
        else:
            f = math.sqrt(100.0 * f)  # too close to baseline -- move down toward 100Hz

    summary = {
        "schema": "v10230_part_x_px3_5_frequency_bisection_v1",
        "gate_measurable": MEASURABLE_GATE,
        "S_h_1000Hz_baseline": S_H_1000HZ_BASELINE,
        "evaluated_pairs": evaluated,
        "localized": localized,
        "classification": (
            f"FREQUENCY_TRANSITION_LOCALIZED_AT_{localized['frequency_Hz']:.6f}HZ" if localized is not None
            else "FREQUENCY_TRANSITION_BRACKETED_BUT_NOT_LOCALIZED"
        ),
    }
    out_path = ARTIFACTS_DIR / "px3_5_frequency_bisection.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"classification: {summary['classification']}")


if __name__ == "__main__":
    main()
