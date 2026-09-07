"""PX3.6 section 4: re-qualify D5's passivation chemistry_factor selection
against the LITERAL live cycle-mean p_B/p_P rule (mission section 7.8 rule
4), not the pre_event_max_pB/max_pB_post_commit event-extrema proxy PX3.5
used as a stand-in.

Adds a minimal, additive `state_sampler` hook to run_trajectory (default
None, zero effect on every existing caller) that is called once per block
with (engine, cycle_step_waveform's own result dict) -- this script uses it
to accumulate a genuine duration-weighted mean of p_P/p_C/p_B across all
active rebonding patches (length-weighted the same way representative_
cycle_K_rebond weights patches elsewhere), duration-weighted by each
block's kinetic_dt_consumed_s, over the WHOLE trajectory.

Only accumulated once at least one patch is active (mission's initial_
precrack_wake_mode=NO_INITIAL_ACTIVE_WAKE means there is no wake -- and
therefore no meaningful bonded/clean/passivated FRACTION of it -- before
the first crack-advance event); the "cycle-mean p_B/p_P" rule 4 asks for
is naturally a property of the wake that exists, not diluted by the
pre-wake phase of the trajectory.

Reruns all three chemistry_factor finite-cohesion trajectories (zero-
cohesion is unaffected -- rule 4 is about the finite trajectory's own
mean state, not a finite/zero comparison) under the clean, committed
c720d59 producer, into fresh virgin result paths.
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"px3_6_passivation_{os.getpid()}_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa, wake_weight
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)
from part_x_run_one_job import _load_row_payload, _resolve_screen_rebonding_cfg  # noqa: E402

SEED = 1720
R = -0.5
FREQUENCY_HZ = 1000.0
KMAX_PA_SQRT_M = 18.0e6


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


class _CycleMeanStateSampler:
    """Accumulates a duration-weighted mean of (p_P, p_C, p_B) across all
    active rebonding patches, length-weighted the same way representative_
    cycle_K_rebond weights patches (wake_weight(s_j_m) * patch.length_m),
    over every block from the first block with >=1 active patch onward."""

    def __init__(self, wake_length_m: float, wake_weight_length_m: float) -> None:
        self.L_h = wake_length_m
        self.L_w = wake_weight_length_m
        self.weighted_sum_p_P = 0.0
        self.weighted_sum_p_C = 0.0
        self.weighted_sum_p_B = 0.0
        self.total_weight_time = 0.0
        self.n_blocks_with_active_patches = 0

    def __call__(self, engine, result: dict) -> None:
        rebonding_state = getattr(engine, "_rebonding_state", None)
        if rebonding_state is None:
            return
        active_patches = [p for p in rebonding_state.active if not p.retired]
        if not active_patches:
            return
        dt = float(result.get("kinetic_dt_consumed_s", 0.0))
        if dt <= 0.0:
            return
        total_w = sum(wake_weight(p.s_j_m, self.L_h, self.L_w) * p.length_m for p in active_patches)
        if total_w <= 0.0:
            return
        mean_p_P = sum(wake_weight(p.s_j_m, self.L_h, self.L_w) * p.length_m * p.state_vector()[0] for p in active_patches) / total_w
        mean_p_C = sum(wake_weight(p.s_j_m, self.L_h, self.L_w) * p.length_m * p.state_vector()[1] for p in active_patches) / total_w
        mean_p_B = sum(wake_weight(p.s_j_m, self.L_h, self.L_w) * p.length_m * p.state_vector()[2] for p in active_patches) / total_w
        self.weighted_sum_p_P += mean_p_P * dt
        self.weighted_sum_p_C += mean_p_C * dt
        self.weighted_sum_p_B += mean_p_B * dt
        self.total_weight_time += dt
        self.n_blocks_with_active_patches += 1

    def result(self) -> dict:
        if self.total_weight_time <= 0.0:
            return {"cycle_mean_p_P": None, "cycle_mean_p_C": None, "cycle_mean_p_B": None, "n_blocks_with_active_patches": 0}
        return {
            "cycle_mean_p_P": self.weighted_sum_p_P / self.total_weight_time,
            "cycle_mean_p_C": self.weighted_sum_p_C / self.total_weight_time,
            "cycle_mean_p_B": self.weighted_sum_p_B / self.total_weight_time,
            "n_blocks_with_active_patches": self.n_blocks_with_active_patches,
            "total_weighted_time_s": self.total_weight_time,
        }


def main() -> None:
    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()
    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)

    row_payload = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]["PASSIVATION_LIMITED"]
    results = {}
    for chem in (1.0, 0.3, 0.1):
        job = {"cohesion": "finite", "chemistry_factor": chem, "K_rebond_max_target_Pa_sqrt_m": 900000.0}
        cfg = _resolve_screen_rebonding_cfg(job, row_payload, Eprime_Pa)
        sampler = _CycleMeanStateSampler(cfg.wake_length_m, cfg.wake_weight_length_m)

        result_dir = RUN_ROOT / f"px3_6_passivation_requalification_chem{chem}"
        if result_dir.exists():
            raise RuntimeError(f"refusing to overwrite existing result: {result_dir}")
        Engine.configure_hazard(mode="exponential", seed=SEED)
        Engine.reset_audit()
        trajectory = pilot.run_trajectory(
            name=f"passivation_requal_chem{chem}", build_engine=_build_engine, make_controller=_make_controller,
            waveform_cls=FatigueWaveform, rebonding_cfg=cfg, R=R, reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=KMAX_PA_SQRT_M, frequency_Hz=FREQUENCY_HZ, n_phase=80, T_K_=300.0,
            max_accepted_events=12, max_projected_extension_m=60.0e-6,
            hazard_rng_seed=SEED, minimum_load_hold_s=0.0, state_sampler=sampler,
        )
        cycle_mean = sampler.result()
        result_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "result.json").write_text(json.dumps(
            {"schema": "v10230_part_x_px3_6_passivation_requalification_v1", "chemistry_factor": chem,
             "trajectory": trajectory, "cycle_mean_state": cycle_mean}, indent=2, default=str,
        ))
        results[chem] = {"trajectory": trajectory, "cycle_mean_state": cycle_mean}
        print(f"chem={chem}: n_events={trajectory['n_accepted_events']} censored={trajectory['censored']} "
              f"cycle_mean_p_B={cycle_mean['cycle_mean_p_B']} cycle_mean_p_P={cycle_mean['cycle_mean_p_P']}")

    out_path = ARTIFACTS_DIR / "px3_6_passivation_requalification.json"
    out_path.write_text(json.dumps(
        {"schema": "v10230_part_x_px3_6_passivation_requalification_summary_v1", "results": results}, indent=2, default=str,
    ))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
