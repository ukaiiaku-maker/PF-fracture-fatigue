"""EXPOSURE_UNCONDITIONED_COMPLETION: run the one missing trajectory
(Kmax=21 MPa*sqrt(m), seed=1001723, RB2 reversible, finite cohesion).

This is NOT a retroactive pass of the minimal slope screen's exposure
gate. For a slope study, reduced compression exposure at high Kmax is
potentially part of the physical load dependence being measured (faster
opening renewals -> less bond formation -> weaker cohesive contribution),
so admission is not conditioned on the >=2-complete-excursion rule that
correctly applied to the earlier causal-pilot qualification question.

Reloads every setting verbatim from the already-frozen minimal slope
screen configuration (artifacts/crack_rebonding_minimal_slope_screen_v1/
frozen_predictions.json's referenced parent frozen_configuration.json) --
same A_NATIVE row, T, R, f, mpz_n_bins, n_phase, bond/rupture barriers,
attempt frequencies, activation volumes, restored_work_of_separation_J_m2,
rebond_K_geometry_factor (hence K_rebond_max), event target, and numerical
controls as every other reversible-finite trajectory in this campaign.
Only Kmax and seed are set, matching the already-run zero-cohesion twin
(C2R_K21MPa_seed1001723) exactly.

Usage:
    <pinned interpreter> scripts/run_v2_slope_exposure_completion.py \\
        --out runs/crack_rebonding_slope_exposure_continuation
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
SLOPE_SCREEN_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"
OUT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_slope_exposure_continuation"

KMAX_Pa_sqrt_m = 21.0e6
SEED = 1001723
TRAJECTORY_NAME = "C3R_K21MPa_seed1001723_EXPOSURE_UNCONDITIONED"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _check_interpreter()

    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    frozen_predictions = json.loads((SLOPE_SCREEN_ARTIFACTS / "frozen_predictions.json").read_text())
    if frozen_predictions["parent_frozen_configuration_sha256"] != frozen["frozen_configuration_sha256"]:
        raise RuntimeError("slope screen's frozen_predictions.json does not match the parent frozen config")

    rb2_rev_finite = _config_from_frozen(frozen, "RB2_reversible_finite")

    # Cross-check against the already-run zero-cohesion twin at this exact
    # (Kmax, seed) point -- confirms we are completing the SAME pair the
    # exposure gate stopped, not a different one. (Not cross-checking the
    # twin's own stored "seed" field: run_v2_minimal_slope_screen.py was
    # run before the hazard_rng_seed reporting fix below, so that
    # trajectory's ledger entry has the same pre-existing "seed" ==
    # pilot.SEED mislabeling this fix addresses going forward -- its
    # trajectory NAME and the file it lives under are the authoritative
    # identifiers instead, both unambiguous here.)
    screen_ledger = json.loads((SLOPE_SCREEN_ARTIFACTS / "event_ledger.json").read_text())
    zero_twin = screen_ledger["trajectories"]["C2R_K21MPa_seed1001723"]
    if zero_twin["R"] != pilot.R_REF:
        raise RuntimeError("zero-cohesion twin's R does not match this completion's own settings")

    print(f"Running {TRAJECTORY_NAME} (finite cohesion, Kmax={KMAX_Pa_sqrt_m:.3e}, seed={SEED})...")
    print("EXPOSURE_UNCONDITIONED_COMPLETION -- not gated on the zero-cohesion twin's exposure count.")

    Engine.configure_hazard(mode="exponential", seed=SEED)
    Engine.reset_audit()

    def build_engine(rebonding_cfg):
        return build_a_native_engine(rebonding_cfg)

    t0 = time.monotonic()
    result = pilot.run_trajectory(
        name=TRAJECTORY_NAME, build_engine=build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=rb2_rev_finite, R=pilot.R_REF,
        reset_engine_registry=Engine.reset_audit, Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m,
        hazard_rng_seed=SEED,
    )
    elapsed = time.monotonic() - t0
    print(f"  {TRAJECTORY_NAME}: {result['n_accepted_events']} events, "
          f"censored={result['censored']} ({result['censor_reason']}), wall={elapsed:.1f}s")

    n_complete = sum(
        1 for iv in result["post_first_event_intervals"] if iv["complete_negative_excursion"]
    )
    n_partial = sum(
        1 for iv in result["post_first_event_intervals"]
        if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
    )
    print(f"  compression exposure (informational, not a gate): "
          f"{n_complete} complete, {n_partial} partial, of "
          f"{len(result['post_first_event_intervals'])} intervals")

    (run_root / f"trajectory_{TRAJECTORY_NAME}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    trajectories = {TRAJECTORY_NAME: result}
    (run_root / "trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n")

    completion_record = {
        "schema": "v10.2.30_crack_rebonding_slope_exposure_completion_v1",
        "classification": "EXPOSURE_UNCONDITIONED_COMPLETION",
        "Kmax_Pa_sqrt_m": KMAX_Pa_sqrt_m,
        "seed": SEED,
        "config_hash_reused": frozen["config_hashes"]["RB2_reversible_finite"],
        "parent_frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
        "n_accepted_events": result["n_accepted_events"],
        "censored": result["censored"],
        "n_intervals_complete_excursion": n_complete,
        "n_intervals_partial_excursion": n_partial,
        "note": (
            "this trajectory was run WITHOUT the >=2-complete-excursion exposure "
            "gate applied to admit finite-cohesion twins in the minimal slope "
            "screen -- reduced compression exposure at high Kmax is treated as "
            "part of the physical effect under study, not a validity failure"
        ),
    }
    (OUT_ARTIFACTS / "completion_record.json").write_text(
        json.dumps(completion_record, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {OUT_ARTIFACTS / 'completion_record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
