"""Section B: run the developed-response confirmation trajectory matrix.

Runs six (or twelve, for Stage 2) FRESH, virgin trajectories -- Kmax in
{15, 18, 21} MPa*sqrt(m) x {reversible zero cohesion, reversible finite
cohesion} -- using the real A_NATIVE production engine's own
cycle_step_waveform/commit_energy_gated_event path exclusively (via
arrhenius_fracture.crack_rebonding_causal_pilot_v2_v10230.run_trajectory,
already qualified and reused verbatim from the causal-pilot lineage).

Unlike the minimal slope screen, NO complete-compression-excursion
admission gate is imposed here: contact exposure is treated as an
outcome/diagnostic, not an admissibility criterion, per the already
accepted reasoning in the slope-exposure-continuation branch and per
Section B of the developed-confirmation protocol
(artifacts/crack_rebonding_developed_confirmation/
developed_confirmation_frozen_protocol.json).

Every trajectory is virgin: no restart/resume is used or supported by
run_trajectory, so every invocation here starts from a fresh engine via
reset_engine_registry. Budgets are the widened ones frozen in Section A
(max_accepted_events=30, max_projected_extension_m=150um,
max_wall_seconds=7200s) needed to reach the reused ~18-event/100um
developed/stationarity target -- NOT the prior campaign's 7-8 event/30um
budget. A trajectory that exhausts its budget before reaching the
developed target is reported censored=true and is never coerced into a
da/dN=0 reading.

Usage:
    <pinned interpreter> scripts/run_developed_confirmation_stage.py \\
        --seed 1720 --out runs/crack_rebonding_developed_confirmation/stage1_seed1720

    <pinned interpreter> scripts/run_developed_confirmation_stage.py \\
        --seed 1001723 --out runs/crack_rebonding_developed_confirmation/stage2_seed1001723
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
DEV_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"

KMAX_GRID_Pa_sqrt_m = [15.0e6, 18.0e6, 21.0e6]
COHESION_KEYS = ["zero", "finite"]
CONFIG_KEY_BY_COHESION = {"zero": "RB2_reversible_zero", "finite": "RB2_reversible_finite"}

# Widened budgets frozen in Section A (developed_confirmation_frozen_protocol.json
# -> "budgets"). NOT the prior campaign's 7-8 event / 30um budget.
MAX_ACCEPTED_EVENTS = 30
MAX_PROJECTED_EXTENSION_m = 1.5e-4
MAX_BLOCKS_PER_EVENT = 20000
MAX_WALL_SECONDS_PER_TRAJECTORY = 7200.0
MIN_ACCEPTED_EVENTS_FOR_UNCENSORED = 3


def _check_interpreter() -> None:
    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(
            f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}"
        )


def _make_controller(n_phase: int) -> FatigueCycleHazardController:
    return FatigueCycleHazardController(
        FatigueControllerConfig(
            n_phase=n_phase, block_cycles=pilot.BLOCK_CYCLES, max_block_cycles=pilot.MAX_BLOCK_CYCLES,
        ),
        None, None, None,
    )


def _config_from_frozen(frozen: dict, key: str) -> CrackRebondingControls:
    payload = frozen["configs"][key]
    kwargs = dict(payload)
    kwargs["model_level"] = RebondModelLevel(kwargs["model_level"])
    kwargs["contact_model"] = ContactModel(kwargs["contact_model"])
    kwargs["feedback_mode"] = FeedbackMode(kwargs["feedback_mode"])
    kwargs["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(kwargs["initial_precrack_wake_mode"])
    cfg = CrackRebondingControls(**kwargs)
    expected_hash = frozen["config_hashes"][key]
    actual_hash = cfg.config_hash()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"reloaded config {key!r} hash {actual_hash} does not match the frozen "
            f"configuration's own hash {expected_hash} -- refusing to run against a "
            "silently different config"
        )
    return cfg


def _trajectory_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=[1720, 1001723])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--only", default=None,
        help="optional comma-separated subset, e.g. '15:zero,15:finite' to run only those points",
    )
    args = parser.parse_args(argv)
    _check_interpreter()

    seed = args.seed
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    DEV_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())

    only = None
    if args.only:
        only = set()
        for token in args.only.split(","):
            k_str, coh = token.split(":")
            only.add((int(k_str) * 1_000_000.0, coh))

    trajectories: dict[str, dict] = {}
    manifest_rows: list[dict] = []

    for Kmax_Pa_sqrt_m in KMAX_GRID_Pa_sqrt_m:
        Kmax_MPa = int(round(Kmax_Pa_sqrt_m / 1.0e6))
        for cohesion in COHESION_KEYS:
            if only is not None and (Kmax_Pa_sqrt_m, cohesion) not in only:
                continue
            config_key = CONFIG_KEY_BY_COHESION[cohesion]
            rebonding_cfg = _config_from_frozen(frozen, config_key)
            name = _trajectory_name(seed, Kmax_MPa, cohesion)

            traj_path = run_root / f"trajectory_{name}.json"
            if traj_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite existing result path {traj_path} -- "
                    "no restart/resume authorized; every trajectory must start "
                    "from a virgin result path (delete or choose a fresh --out "
                    "if this is intentional)"
                )

            print(
                f"Running {name} (cohesion={cohesion}, Kmax={Kmax_Pa_sqrt_m:.3e}, "
                f"seed={seed}, config_hash={rebonding_cfg.config_hash()[:12]}...)"
            )
            print("  no complete-compression-excursion admission gate imposed "
                  "(contact exposure is an outcome/diagnostic here, not a gate)")

            Engine.configure_hazard(mode="exponential", seed=seed)
            Engine.reset_audit()

            def build_engine(cfg):
                return build_a_native_engine(cfg)

            t0 = time.monotonic()
            result = pilot.run_trajectory(
                name=name,
                build_engine=build_engine,
                make_controller=_make_controller,
                waveform_cls=FatigueWaveform,
                rebonding_cfg=rebonding_cfg,
                R=pilot.R_REF,
                reset_engine_registry=Engine.reset_audit,
                Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
                hazard_rng_seed=seed,
                max_accepted_events=MAX_ACCEPTED_EVENTS,
                max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
                min_accepted_events_for_uncensored=MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
                max_blocks_per_event=MAX_BLOCKS_PER_EVENT,
                max_wall_seconds=MAX_WALL_SECONDS_PER_TRAJECTORY,
            )
            elapsed = time.monotonic() - t0

            n_complete = sum(
                1 for iv in result["post_first_event_intervals"]
                if iv["complete_negative_excursion"]
            )
            n_partial = sum(
                1 for iv in result["post_first_event_intervals"]
                if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
            )
            print(
                f"  {name}: {result['n_accepted_events']} events, "
                f"final_extension_um={1.0e6 * result['cumulative_extension_m']:.3f}, "
                f"censored={result['censored']} ({result['censor_reason']}), "
                f"exposure(complete/partial)={n_complete}/{n_partial}, "
                f"wall={elapsed:.1f}s"
            )

            traj_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            trajectories[name] = result
            manifest_rows.append({
                "name": name, "seed": seed, "Kmax_Pa_sqrt_m": Kmax_Pa_sqrt_m,
                "cohesion": cohesion, "config_key": config_key,
                "config_hash": rebonding_cfg.config_hash(),
                "n_accepted_events": result["n_accepted_events"],
                "final_extension_um": 1.0e6 * result["cumulative_extension_m"],
                "censored": result["censored"], "censor_reason": result["censor_reason"],
                "wall_seconds": elapsed,
            })

            (run_root / "trajectories.json").write_text(
                json.dumps(trajectories, indent=2, sort_keys=True) + "\n"
            )
            (run_root / "stage_manifest.json").write_text(
                json.dumps(
                    {"seed": seed, "rows": manifest_rows,
                     "parent_frozen_configuration_sha256": frozen["frozen_configuration_sha256"]},
                    indent=2, sort_keys=True,
                ) + "\n"
            )

    print(f"wrote {run_root / 'trajectories.json'} ({len(trajectories)} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
