"""V2 Section 9: RB1 (CONTACT_PROXY_ONLY) compression-sampling preflight.

Drives the REAL A_NATIVE production engine (arrhenius_fracture.
a_native_engine_v10230.build_a_native_engine -- not the DBTT test fixture)
through several accepted cleavage events at the mission's reference
protocol (T=300K, R=-0.95, Kmax=18 MPa*sqrt(m), f=1000 Hz, mpz_n_bins=80),
using RB1 (CONTACT_PROXY_ONLY, contact diagnostics only -- patch_Q is the
exact zero generator, so this cannot perturb event timing at all; see
crack_rebonding_causal_pilot_v10230/V2-B's strict-parity fix).

For every post-first-event interval, independently (outside the engine,
using only the physical cumulative time each block reports -- FatigueWaveform
.K_phase is a pure function of continuous phase = 2*pi*f*t with no resets,
confirmed in fatigue_v1.py) computes the SIGNED K(t) waveform the
event-created patch actually sees between its creation and the next event,
and checks whether a COMPLETE negative-K (compressive) excursion occurs
(K crosses to negative and back to non-negative strictly inside the
interval, not just touching the interval boundary while still negative).

Usage:
    <pinned interpreter> scripts/run_v2_rb1_compression_preflight.py \\
        --out artifacts/crack_rebonding_causal_pilot_v2/preflight_protocol_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.a_native_engine_v10230 import (  # noqa: E402
    build_a_native_engine,
    load_a_native_provenance,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
)
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

REQUIRED_PYTHON = (
    "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
)

SEED = 1720
T_K = 300.0
KMAX_Pa_sqrt_m = 18.0e6
F_HZ = 1000.0
R_REF = -0.95
N_PHASE = 80
MAX_EVENTS = 8
MAX_BLOCKS_PER_EVENT = 20000


def rb1_config() -> CrackRebondingControls:
    return CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CONTACT_PROXY_ONLY,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY,
        feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
        wake_length_m=5.0e-4,
        wake_weight_length_m=5.0e-7,
        fresh_surface_clean_fraction=1.0,
        chemistry_factor=1.0,
        restored_work_of_separation_J_m2=0.0,
        rebond_K_geometry_factor=1.0,
        bond_barrier_eV=0.2,
        rupture_barrier_eV=1.0,
        bond_attempt_frequency_s=1.0e10,
        rupture_attempt_frequency_s=1.0e10,
    ).validate()


def signed_K(t_s: np.ndarray) -> np.ndarray:
    """K_signed(t) for the reference protocol, unclipped (contact proxy),
    matching FatigueWaveform.K_phase's own convention (phase=0 -> K=Kmax,
    phase = 2*pi*f*t, no resets)."""
    Kmin = R_REF * KMAX_Pa_sqrt_m
    Kmean = 0.5 * (KMAX_Pa_sqrt_m + Kmin)
    Kamp = 0.5 * (KMAX_Pa_sqrt_m - Kmin)
    phase = 2.0 * np.pi * F_HZ * t_s
    return Kmean + Kamp * np.cos(phase)


def interval_compression_analysis(t_creation_s: float, t_next_event_s: float, n_samples: int = 4000) -> dict:
    if t_next_event_s <= t_creation_s:
        return {
            "elapsed_time_s": 0.0,
            "elapsed_cycles": 0.0,
            "negative_contact_duration_s": 0.0,
            "complete_negative_excursion": False,
        }
    t = np.linspace(t_creation_s, t_next_event_s, n_samples)
    K = signed_K(t)
    neg = K < 0.0
    dt = t[1] - t[0]
    negative_duration_s = float(np.count_nonzero(neg)) * dt

    complete = False
    if neg.any() and not neg[0] and not neg[-1]:
        # At least one negative run strictly interior to the sampled window
        # (bounded by non-negative K on both sides) -- a genuinely completed
        # compressive excursion, not one truncated by the interval boundary.
        edges = np.diff(neg.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        complete = starts.size > 0 and ends.size > 0

    return {
        "elapsed_time_s": float(t_next_event_s - t_creation_s),
        "elapsed_cycles": float((t_next_event_s - t_creation_s) * F_HZ),
        "negative_contact_duration_s": negative_duration_s,
        "complete_negative_excursion": bool(complete),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if sys.executable != REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {REQUIRED_PYTHON!r}, got {sys.executable!r}")

    provenance = load_a_native_provenance()

    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    engine, manifest_audit = build_a_native_engine(rb1_config())
    ctrl = FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=N_PHASE, block_cycles=1000.0, max_block_cycles=1.0e6),
        None, None, None,
    )
    waveform = FatigueWaveform(Kmax=KMAX_Pa_sqrt_m, R=R_REF, frequency_Hz=F_HZ)

    events: list[dict] = []
    cumulative_time_s = 0.0
    for event_index in range(MAX_EVENTS):
        fired_result = None
        for _ in range(MAX_BLOCKS_PER_EVENT):
            result = engine.cycle_step_waveform(ctrl, waveform, T_K)
            cumulative_time_s += float(result.get("kinetic_dt_consumed_s", 0.0))
            if result.get("fired"):
                fired_result = result
                break
        if fired_result is None:
            events.append({"event_index": event_index, "censored": True})
            break

        pending = engine._energy_gate_pending
        committed_length = pending["proposal_m"]
        gate = {
            "energy_admissible_event_length_m": committed_length,
            "arrest_reason": "preflight_commit",
            "hazard_resistance_J_per_m2": 1.0,
            "orientation_gamma_relative": 1.0,
        }
        result_ref = pending["descriptor"].get("energy_gate_result_ref")
        engine.commit_energy_gated_event(committed_length, gate, result_ref)

        events.append({
            "event_index": event_index,
            "cumulative_time_s": cumulative_time_s,
            "accepted_length_m": float(committed_length),
            "censored": False,
        })

    intervals = []
    for i in range(1, len(events) - 1 if events and events[-1].get("censored") else len(events)):
        prev_e, this_e = events[i - 1], events[i]
        if prev_e.get("censored") or this_e.get("censored"):
            continue
        analysis = interval_compression_analysis(
            prev_e["cumulative_time_s"], this_e["cumulative_time_s"]
        )
        analysis.update({
            "creation_event_index": prev_e["event_index"],
            "next_event_index": this_e["event_index"],
            "creation_time_s": prev_e["cumulative_time_s"],
            "next_event_time_s": this_e["cumulative_time_s"],
        })
        intervals.append(analysis)

    n_complete = sum(1 for iv in intervals if iv["complete_negative_excursion"])
    usable = n_complete >= 2

    payload = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_preflight_protocol_selection_v1",
        "seed": SEED,
        "protocol": {
            "T_K": T_K, "R": R_REF, "Kmax_Pa_sqrt_m": KMAX_Pa_sqrt_m,
            "frequency_Hz": F_HZ, "n_phase": N_PHASE,
        },
        "manifest_audit": manifest_audit,
        "a_native_reconstruction_classification": provenance["reconstruction_classification"],
        "events": events,
        "post_first_event_intervals": intervals,
        "n_intervals_with_complete_negative_excursion": n_complete,
        "reference_protocol_usable": usable,
        "status": (
            "REFERENCE_PROTOCOL_SAMPLES_COMPRESSION"
            if usable
            else "REFERENCE_1KHZ_PROTOCOL_CANNOT_SAMPLE_COMPRESSION_REBONDING"
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"status={payload['status']}  n_complete_negative_excursions={n_complete}")
    for iv in intervals:
        print(
            f"  interval {iv['creation_event_index']}->{iv['next_event_index']}: "
            f"elapsed_cycles={iv['elapsed_cycles']:.4f} "
            f"neg_dwell_s={iv['negative_contact_duration_s']:.3e} "
            f"complete={iv['complete_negative_excursion']}"
        )
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
