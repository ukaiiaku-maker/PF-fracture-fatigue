"""PX3.5 section 6: run the COMPETING_PERSISTENT finite/zero pair at the
frequency-bisection's localized transition condition (316.227766 Hz), so
D6's persistent-vs-reversible distinction gate can be evaluated at the
SAME condition D3 will actually use for developed confirmation -- per
external review's explicit instruction that "D6 must use the ultimately
selected frequency-transition condition."
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"px3_5_persistent_{os.getpid()}_")

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


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def main() -> None:
    freq_data = json.loads((ARTIFACTS_DIR / "px3_5_frequency_bisection.json").read_text())
    if freq_data["localized"] is None:
        print("No localized frequency transition -- nothing to run for D6.")
        return
    frequency_Hz = freq_data["localized"]["frequency_Hz"]

    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)

    row_payload = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]["COMPETING_PERSISTENT"]
    trajectories = {}
    for cohesion in ("finite", "zero"):
        job = {
            "cohesion": cohesion, "chemistry_factor": 1.0,
            "K_rebond_max_target_Pa_sqrt_m": 900000.0 if cohesion == "finite" else 0.0,
        }
        cfg = _resolve_screen_rebonding_cfg(job, row_payload, Eprime_Pa)
        result_dir = RUN_ROOT / f"px3_5_persistent_at_transition_f{frequency_Hz:.6f}Hz_{cohesion}"
        if result_dir.exists():
            raise RuntimeError(f"refusing to overwrite existing result: {result_dir}")
        Engine.configure_hazard(mode="exponential", seed=SEED)
        Engine.reset_audit()
        trajectory = pilot.run_trajectory(
            name=f"persistent_transition_{cohesion}", build_engine=_build_engine, make_controller=_make_controller,
            waveform_cls=FatigueWaveform, rebonding_cfg=cfg, R=R, reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=KMAX_PA_SQRT_M, frequency_Hz=frequency_Hz, n_phase=80, T_K_=300.0,
            max_accepted_events=12, max_projected_extension_m=60.0e-6,
            hazard_rng_seed=SEED, minimum_load_hold_s=0.0,
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "result.json").write_text(json.dumps(
            {"schema": "v10230_part_x_px3_5_persistent_at_transition_v1", "frequency_Hz": frequency_Hz,
             "cohesion": cohesion, "trajectory": trajectory}, indent=2, default=str,
        ))
        trajectories[cohesion] = trajectory
        print(f"{cohesion}: n_events={trajectory['n_accepted_events']} censored={trajectory['censored']} "
              f"cumulative_extension_m={trajectory['cumulative_extension_m']:.6e} "
              f"cumulative_cycles={trajectory['cumulative_cycles']:.6e}")

    import math
    g_f = trajectories["finite"]["cumulative_extension_m"] / trajectories["finite"]["cumulative_cycles"]
    g_z = trajectories["zero"]["cumulative_extension_m"] / trajectories["zero"]["cumulative_cycles"]
    S_h = math.log10(g_f / g_z)
    summary = {"schema": "v10230_part_x_px3_5_persistent_at_transition_summary_v1",
               "frequency_Hz": frequency_Hz, "g_finite": g_f, "g_zero": g_z, "S_h": S_h}
    out_path = RUN_ROOT / "px3_5_persistent_at_transition_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"S_h at persistent transition condition: {S_h:.6f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
