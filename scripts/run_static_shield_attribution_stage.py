"""Section C: run the static-shield attribution trajectory matrix.

Runs three (or six, for Stage 2) FRESH, virgin PRESCRIBED_POST_FIRST_
EVENT_COHESIVE_SHIELD trajectories -- Kmax in {15, 18, 21} MPa*sqrt(m) at
a given seed -- using the real A_NATIVE production engine with NO
CrackRebondingControls installed at all (rebonding_cfg=None). K_b enters
only through arrhenius_fracture.persistent_site_coupled_hazard_v10229.py's
new, default-off static-shield branch in _phase_statistics -- the
CONFIRMED correct injection point for the real production engine
hierarchy (docs/v10_2_30_crack_rebonding_equation_lineage.md's
"injection-point correction history": CoupledPersistentSiteCyclicTipEngine.
cycle_step_waveform bypasses persistent_site_cyclic_v10229.py's own
preview_cycle_waveform/cycle_step_waveform entirely, delegating to
integrate_state_coupled_waveform/_phase_statistics instead -- confirmed
empirically here too: an initial attempt at the (dead-code, for this
engine) persistent_site_cyclic_v10229.py location showed zero waiting-
time sensitivity even at K_b comparable to Kmax itself, while this
location shows a consistent, physically sensible ~34% waiting-time
increase at K_b=900000 Pa sqrt(m) and correct censoring/saturation at
more extreme K_b values). K_b: 0.0 before the first accepted event,
900000.0 Pa sqrt(m) (0.9 MPa sqrt(m), the already-frozen K_rebond_max)
from the first accepted event onward. This is a mechanism-control
ablation, not a physical rebonding model -- no P/C/B Markov kinetics, no
wake-state, no contact-gated formation.

Reuses (never reruns) the existing zero-cohesion and dynamic-rebonding
seed=1720 trajectories from artifacts/crack_rebonding_developed_
confirmation/event_ledger.json for the S_dynamic/S_zero comparison; this
script only produces the three (or six) new static-shield trajectories.

Records the future-run provenance fields the prior review round
requested: Kmax/T_K/frequency_Hz per trajectory (now returned natively by
run_trajectory), full git HEAD at launch, physical producer-file hashes,
an explicit no-resume marker, an RNG-state/stream identifier, and the
frozen-configuration hash.

Usage:
    <pinned interpreter> scripts/run_static_shield_attribution_stage.py \\
        --seed 1720 --out runs/crack_rebonding_static_shield_attribution/stage1_seed1720
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
STATIC_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_static_shield_attribution"

KMAX_GRID_Pa_sqrt_m = [15.0e6, 18.0e6, 21.0e6]
K_B_STATIC_Pa_sqrt_m = 900000.0

# Identical widened budgets as the developed-confirmation campaign, for a
# directly comparable developed/true-terminal window analysis.
MAX_ACCEPTED_EVENTS = 30
MAX_PROJECTED_EXTENSION_m = 1.5e-4
MAX_BLOCKS_PER_EVENT = 20000
MAX_WALL_SECONDS_PER_TRAJECTORY = 7200.0
MIN_ACCEPTED_EVENTS_FOR_UNCENSORED = 3

PRODUCER_FILES = [
    "scripts/run_static_shield_attribution_stage.py",
    "arrhenius_fracture/persistent_site_coupled_hazard_v10229.py",
    "arrhenius_fracture/crack_rebonding_causal_pilot_v2_v10230.py",
]


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


def _traj_name(seed: int, Kmax_MPa: int) -> str:
    stage = "S1" if seed == 1720 else "S2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_static"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=[1720, 1001723])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _check_interpreter()

    seed = args.seed
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    STATIC_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    git_head_at_launch = _git_head()
    producer_file_hashes = {p: _sha256_file(REPO_ROOT / p) for p in PRODUCER_FILES}

    trajectories: dict[str, dict] = {}
    manifest_rows: list[dict] = []

    for Kmax_Pa_sqrt_m in KMAX_GRID_Pa_sqrt_m:
        Kmax_MPa = int(round(Kmax_Pa_sqrt_m / 1.0e6))
        name = _traj_name(seed, Kmax_MPa)

        traj_path = run_root / f"trajectory_{name}.json"
        if traj_path.exists():
            raise RuntimeError(
                f"refusing to overwrite existing result path {traj_path} -- "
                "no restart/resume authorized; every trajectory must start "
                "from a virgin result path"
            )

        print(f"Running {name} (static shield, Kmax={Kmax_Pa_sqrt_m:.3e}, seed={seed}, "
              f"K_b_static={K_B_STATIC_Pa_sqrt_m:.1f} Pa sqrt(m))")

        Engine.configure_hazard(mode="exponential", seed=seed)
        Engine.reset_audit()

        def build_engine(cfg):
            return build_a_native_engine(cfg)

        static_shield_control = {
            "enabled": True,
            "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m,
            "first_event_fired": False,
        }

        t0 = time.monotonic()
        result = pilot.run_trajectory(
            name=name,
            build_engine=build_engine,
            make_controller=_make_controller,
            waveform_cls=FatigueWaveform,
            rebonding_cfg=None,
            R=pilot.R_REF,
            reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
            hazard_rng_seed=seed,
            max_accepted_events=MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
            min_accepted_events_for_uncensored=MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
            max_blocks_per_event=MAX_BLOCKS_PER_EVENT,
            max_wall_seconds=MAX_WALL_SECONDS_PER_TRAJECTORY,
            static_shield_control=static_shield_control,
        )
        elapsed = time.monotonic() - t0

        k_b_sequence = [e["K_b_applied_Pa_sqrt_m"] for e in result["events"]]
        n_zero_before = sum(1 for v in k_b_sequence if v == 0.0)
        n_static_after = sum(1 for v in k_b_sequence if v == K_B_STATIC_Pa_sqrt_m)
        print(f"  {name}: {result['n_accepted_events']} events, "
              f"final_extension_um={1.0e6 * result['cumulative_extension_m']:.3f}, "
              f"censored={result['censored']} ({result['censor_reason']}), "
              f"K_b sequence: {n_zero_before} events at 0.0, {n_static_after} at "
              f"{K_B_STATIC_Pa_sqrt_m:.0f} (expect exactly 1 zero, rest static), "
              f"wall={elapsed:.1f}s")
        if n_zero_before != 1 or n_static_after != result["n_accepted_events"] - 1:
            raise RuntimeError(
                f"{name}: K_b step-function sequence is not exactly "
                f"[0.0 once, then {K_B_STATIC_Pa_sqrt_m} for the rest] -- "
                f"got {n_zero_before} zero events and {n_static_after} static "
                f"events out of {result['n_accepted_events']}; refusing to "
                "archive a trajectory whose ablation control did not apply "
                "as specified"
            )

        rng_stream_identifier = f"exponential_seed_{seed}_engine_hazard_rng"
        result_extended = dict(result)
        result_extended.update({
            "no_resume_no_restart_marker": True,
            "rng_state_or_stream_identifier": rng_stream_identifier,
            "frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
            "git_head_at_launch": git_head_at_launch,
            "producer_file_sha256_at_launch": producer_file_hashes,
        })

        (run_root / f"trajectory_{name}.json").write_text(
            json.dumps(result_extended, indent=2, sort_keys=True) + "\n"
        )
        trajectories[name] = result_extended
        manifest_rows.append({
            "name": name, "seed": seed, "Kmax_Pa_sqrt_m": Kmax_Pa_sqrt_m,
            "T_K": result["T_K"], "frequency_Hz": result["frequency_Hz"],
            "n_accepted_events": result["n_accepted_events"],
            "final_extension_um": 1.0e6 * result["cumulative_extension_m"],
            "censored": result["censored"], "censor_reason": result["censor_reason"],
            "wall_seconds": elapsed, "git_head_at_launch": git_head_at_launch,
            "no_resume_no_restart_marker": True,
            "rng_state_or_stream_identifier": rng_stream_identifier,
        })

        (run_root / "trajectories.json").write_text(
            json.dumps(trajectories, indent=2, sort_keys=True) + "\n"
        )
        (run_root / "stage_manifest.json").write_text(
            json.dumps(
                {
                    "seed": seed, "rows": manifest_rows,
                    "parent_frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
                    "git_head_at_launch": git_head_at_launch,
                    "producer_file_sha256_at_launch": producer_file_hashes,
                },
                indent=2, sort_keys=True,
            ) + "\n"
        )

    print(f"wrote {run_root / 'trajectories.json'} ({len(trajectories)} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
