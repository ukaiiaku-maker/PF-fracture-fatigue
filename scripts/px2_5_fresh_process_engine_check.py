"""PX2.5: fresh-process engine reproducibility check.

Constructs the real A_NATIVE production engine using the explicit
configure_hazard + reset_audit pattern every PX3+ scientific launch
script must use, runs a short trajectory, and prints a JSON fingerprint.
Invoked as its own subprocess by
tests/test_v10_2_30_crack_rebonding_part_x_engine_reproducibility.py to
prove reproducibility across truly independent process invocations,
optionally after constructing --pollute-n unrelated engines first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)


def _construct_unrelated_engines(n: int) -> None:
    for _ in range(n):
        e, _ = build_a_native_engine()
        cfg = CrackRebondingControls(
            enabled=True, model_level=RebondModelLevel.PASSIVATION_GATED_REBOND,
            contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY, feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
            wake_length_m=5.0e-4, wake_weight_length_m=5.0e-7, restored_work_of_separation_J_m2=1.8225,
            rebond_K_geometry_factor=1.0, bond_barrier_eV=0.39, rupture_barrier_eV=0.386,
            depassivation_barrier_eV=0.40, repassivation_barrier_eV=0.40,
            initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
        ).validate()
        install_crack_rebonding(e, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pollute-n", type=int, default=0)
    args = parser.parse_args()

    if args.pollute_n > 0:
        _construct_unrelated_engines(args.pollute_n)

    Engine.configure_hazard(mode="exponential", seed=args.seed)
    Engine.reset_audit()
    engine, audit = build_a_native_engine()
    ctrl = FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=80, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)

    thresholds = [float(engine.hazard_threshold_action)]
    events = []
    for i in range(40):
        r = engine.cycle_step_waveform(ctrl, wave, 300.0)
        if r.get("fired"):
            events.append([i, r["cycles_consumed"], r["B"]])
            thresholds.append(float(engine.hazard_threshold_action))
        if len(events) >= 3:
            break

    print(json.dumps({
        "engine_id": engine._engine_id,
        "mro": [c.__name__ for c in type(engine).__mro__],
        "material_row_sha256": audit["source_row_sha256"],
        "thresholds": thresholds,
        "events": events,
        "final_B": float(engine.B),
        "final_mobile_count": float(engine.mpz.mobile_count),
        "final_retained_count": float(engine.mpz.retained_count),
        "final_t": float(engine.t),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
