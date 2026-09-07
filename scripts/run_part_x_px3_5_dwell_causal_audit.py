"""PX3.5 section 4: dwell-induced rate-sign-reversal causal audit.

After fixing the real bug in persistent_site_coupled_hazard_v10229.py's
_phase_statistics (a cursor-rotated K array duration-weighted by the WRONG,
unrotated dt array -- see the fix's own inline comments and tests/
test_v10_2_30_crack_rebonding_part_x_px3_5_dwell_hazard_fix.py), this
reruns the hold>0 dwell-panel trajectories fresh under the corrected code,
plus a prescribed-static K_b=0.9 MPa sqrt(m) control at the same hold, to
decompose whatever dwell effect remains (per the review's explicit audit
procedure) into:

  - dynamic COMPETING_REVERSIBLE finite cohesion (rerun, bug fixed)
  - prescribed post-first-event static K_b=0.9 MPa sqrt(m) (new leg --
    first live-production use of PX1.4's static_shield_phase_resolved_
    action-adjacent mechanism-control ablation via run_trajectory's
    existing static_shield_control hook, which was already wired into the
    SAME production _phase_statistics this fix touches)

The zero-cohesion leg is NOT rerun -- it was never touched by the bug
(hazard_coupled=False for zero cohesion, always used the unrotated
schedule) and the mission's own instruction is not to rerun/alter/delete
already-completed trajectories unless a targeted diagnostic gap requires
it. Its already-completed result (hold=0.002, zero cohesion) is reused
verbatim as the common baseline for all three legs.

Writes fresh, virgin result paths under runs/crack_rebonding_part_x_v1/
(never touching the original 28 PX3 result directories).
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"px3_5_dwell_audit_{os.getpid()}_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from part_x_run_one_job import _load_row_payload, _resolve_screen_rebonding_cfg  # noqa: E402

SEED = 1720
R = -0.5
FREQUENCY_HZ = 1000.0
KMAX_PA_SQRT_M = 18.0e6
HOLD_S_VALUES = [0.0005, 0.002]
K_B_STATIC_PA_SQRT_M = 900000.0  # 0.9 MPa sqrt(m), the SAT_EXISTING/K_rebond_max baseline
MAX_ACCEPTED_EVENTS = 12
MAX_PROJECTED_EXTENSION_M = 60.0e-6


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def _resolve_reversible_finite_cfg(Eprime_Pa: float):
    registry = json.loads((REPO_ROOT / "artifacts/crack_rebonding_part_x_v1/kinetic_regime_registry.json").read_text())
    row_payload = registry["rows"]["COMPETING_REVERSIBLE"]
    job = {"cohesion": "finite", "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": K_B_STATIC_PA_SQRT_M}
    return _resolve_screen_rebonding_cfg(job, row_payload, Eprime_Pa)


def run_leg(name: str, hold_s: float, *, rebonding_cfg, static_shield_control) -> dict:
    result_dir = RUN_ROOT / f"px3_5_dwell_audit_{name}_hold{hold_s}"
    if result_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing audit leg result: {result_dir}")

    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    trajectory = pilot.run_trajectory(
        name=name, build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=rebonding_cfg, R=R,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=KMAX_PA_SQRT_M, frequency_Hz=FREQUENCY_HZ,
        n_phase=80, T_K_=300.0,
        max_accepted_events=MAX_ACCEPTED_EVENTS, max_projected_extension_m=MAX_PROJECTED_EXTENSION_M,
        hazard_rng_seed=SEED, minimum_load_hold_s=hold_s,
        static_shield_control=static_shield_control,
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "v10230_part_x_px3_5_dwell_audit_leg_result_v1",
        "leg_name": name, "hold_s": hold_s,
        "rebonding_cfg_hash": rebonding_cfg.config_hash() if rebonding_cfg is not None else None,
        "static_shield_control": static_shield_control,
        "trajectory": trajectory,
    }
    (result_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"{name} (hold={hold_s}s): n_events={trajectory['n_accepted_events']} "
          f"censored={trajectory['censored']} ({trajectory['censor_reason']}) "
          f"cumulative_extension_m={trajectory['cumulative_extension_m']:.6e} "
          f"cumulative_cycles={trajectory['cumulative_cycles']:.6e} "
          f"wall_s={trajectory['wall_seconds']:.1f}")
    return result


def main() -> None:
    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    bare_engine, _bare_audit = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    finite_cfg = _resolve_reversible_finite_cfg(Eprime_Pa)

    results = {}
    for hold_s in HOLD_S_VALUES:
        results[f"dynamic_finite_hold{hold_s}"] = run_leg(
            "dynamic_finite", hold_s, rebonding_cfg=finite_cfg, static_shield_control=None,
        )
        results[f"prescribed_static_hold{hold_s}"] = run_leg(
            "prescribed_static", hold_s, rebonding_cfg=None,
            static_shield_control={
                "enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_PA_SQRT_M, "first_event_fired": False,
            },
        )

    summary_path = RUN_ROOT / "px3_5_dwell_audit_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
