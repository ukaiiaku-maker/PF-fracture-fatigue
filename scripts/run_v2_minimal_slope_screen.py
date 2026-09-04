"""Minimal local Paris-slope screen: 8 new trajectories (reversible regime
only, per the passed regime-equivalence gate), Kmax in {15, 21} MPa*sqrt(m),
seeds {1720, 1001723}, zero/finite cohesion.

Reuses the Kmax=18 reversible results from the parent branch's tracked
ledgers verbatim (fingerprint-checked, not re-run). Runs the ZERO-cohesion
member of each (Kmax, seed) pair first and requires >= 2 post-first-event
intervals with a complete negative-K excursion before launching that pair's
finite-cohesion twin. If Kmax=21 fails this exposure gate for either seed,
STOPS without launching that Kmax's finite runs and without substituting a
different load.

Every kinetics/cohesion parameter (bond/rupture barriers, attempt
frequencies, activation volumes, restored_work_of_separation_J_m2,
rebond_K_geometry_factor -> K_rebond_max) is reused byte-identical from the
parent's frozen_configuration.json at every Kmax -- only the waveform's own
Kmax changes. Pi_K = K_rebond_max/Kmax is therefore NOT held at 0.05; it
varies naturally (see frozen_predictions.json).

Usage:
    <pinned interpreter> scripts/run_v2_minimal_slope_screen.py \\
        --out runs/crack_rebonding_minimal_slope_screen_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    SCREEN_KMAX_GRID_Pa_sqrt_m,
)
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

PARENT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
OUT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"

NEW_KMAX_VALUES_Pa_sqrt_m = (15.0e6, 21.0e6)
SEEDS = (1720, 1001723)
MIN_COMPRESSION_INTERVALS_REQUIRED = 2


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
            f"reloaded config {key!r} hash {actual_hash} does not match the parent "
            f"branch's committed frozen_configuration.json hash {expected_hash}"
        )
    return cfg


def _reused_kmax18(seed: int) -> dict:
    """Loads the already-qualified Kmax=18 reversible zero/finite results
    verbatim from the parent branch's tracked ledgers -- not re-run."""
    if seed == 1720:
        ledger = json.loads((PARENT_ARTIFACTS_DIR / "event_ledger.json").read_text())
        zero, finite = ledger["trajectories"]["C2R"], ledger["trajectories"]["C3R"]
    elif seed == 1001723:
        ledger = json.loads((PARENT_ARTIFACTS_DIR / "second_seed_event_ledger.json").read_text())
        zero, finite = ledger["trajectories"]["S2_C2R"], ledger["trajectories"]["S2_C3R"]
    else:
        raise ValueError(seed)
    return {"zero": zero, "finite": finite, "source_ledger_frozen_configuration_sha256": ledger[
        "frozen_configuration_sha256"
    ]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _check_interpreter()

    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((PARENT_ARTIFACTS_DIR / "frozen_configuration.json").read_text())
    rb2_rev_zero = _config_from_frozen(frozen, "RB2_reversible_zero")
    rb2_rev_finite = _config_from_frozen(frozen, "RB2_reversible_finite")

    for seed in SEEDS:
        reused = _reused_kmax18(seed)
        if reused["source_ledger_frozen_configuration_sha256"] != frozen["frozen_configuration_sha256"]:
            raise RuntimeError(f"seed {seed}'s Kmax=18 source ledger used a different frozen config")

    results: dict[str, dict] = {}
    exposure_report: dict[str, dict] = {}
    stopped_early: str | None = None

    def build_engine(rebonding_cfg):
        return build_a_native_engine(rebonding_cfg)

    for Kmax in NEW_KMAX_VALUES_Pa_sqrt_m:
        Kmax_label = f"{Kmax/1e6:.0f}MPa"
        for seed in SEEDS:
            zero_name = f"C2R_K{Kmax_label}_seed{seed}"
            print(f"Running {zero_name} (zero cohesion, Kmax={Kmax:.3e}, seed={seed})...")
            Engine.configure_hazard(mode="exponential", seed=seed)
            Engine.reset_audit()
            t0 = time.monotonic()
            zero_result = pilot.run_trajectory(
                name=zero_name, build_engine=build_engine, make_controller=_make_controller,
                waveform_cls=FatigueWaveform, rebonding_cfg=rb2_rev_zero, R=pilot.R_REF,
                reset_engine_registry=Engine.reset_audit, Kmax_Pa_sqrt_m=Kmax,
            )
            print(f"  {zero_name}: {zero_result['n_accepted_events']} events, "
                  f"censored={zero_result['censored']}, wall={time.monotonic()-t0:.1f}s")
            results[zero_name] = zero_result
            (run_root / f"trajectory_{zero_name}.json").write_text(
                json.dumps(zero_result, indent=2, sort_keys=True) + "\n"
            )

            n_complete = sum(
                1 for iv in zero_result["post_first_event_intervals"]
                if iv["complete_negative_excursion"]
            )
            exposure_report[f"K{Kmax_label}_seed{seed}"] = {
                "Kmax_Pa_sqrt_m": Kmax, "seed": seed,
                "n_intervals_with_complete_negative_excursion": n_complete,
                "n_accepted_events": zero_result["n_accepted_events"],
                "censored": zero_result["censored"],
                "exposure_gate_pass": n_complete >= MIN_COMPRESSION_INTERVALS_REQUIRED,
            }
            if n_complete < MIN_COMPRESSION_INTERVALS_REQUIRED:
                print(
                    f"  EXPOSURE GATE FAILED for Kmax={Kmax:.3e}, seed={seed}: "
                    f"only {n_complete} compression-containing intervals "
                    f"(need >= {MIN_COMPRESSION_INTERVALS_REQUIRED}). Stopping -- "
                    "not substituting a different load."
                )
                stopped_early = f"K{Kmax_label}_seed{seed}"
                break

            finite_name = f"C3R_K{Kmax_label}_seed{seed}"
            print(f"Running {finite_name} (finite cohesion, Kmax={Kmax:.3e}, seed={seed})...")
            Engine.configure_hazard(mode="exponential", seed=seed)
            Engine.reset_audit()
            t0 = time.monotonic()
            finite_result = pilot.run_trajectory(
                name=finite_name, build_engine=build_engine, make_controller=_make_controller,
                waveform_cls=FatigueWaveform, rebonding_cfg=rb2_rev_finite, R=pilot.R_REF,
                reset_engine_registry=Engine.reset_audit, Kmax_Pa_sqrt_m=Kmax,
            )
            print(f"  {finite_name}: {finite_result['n_accepted_events']} events, "
                  f"censored={finite_result['censored']}, wall={time.monotonic()-t0:.1f}s")
            results[finite_name] = finite_result
            (run_root / f"trajectory_{finite_name}.json").write_text(
                json.dumps(finite_result, indent=2, sort_keys=True) + "\n"
            )
        if stopped_early is not None:
            break

    (run_root / "trajectories.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    exposure_path = run_root / "exposure_gate_report.json"
    exposure_payload = {
        "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_exposure_gate_v1",
        "min_compression_intervals_required": MIN_COMPRESSION_INTERVALS_REQUIRED,
        "results": exposure_report,
        "stopped_early_at": stopped_early,
    }
    exposure_path.write_text(json.dumps(exposure_payload, indent=2, sort_keys=True) + "\n")
    (OUT_ARTIFACTS_DIR / "exposure_gate_report.json").write_text(
        json.dumps(exposure_payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {exposure_path}")

    reuse_record = {
        seed: {
            "zero_cumulative_time_s": _reused_kmax18(seed)["zero"]["cumulative_time_s"],
            "finite_cumulative_time_s": _reused_kmax18(seed)["finite"]["cumulative_time_s"],
        }
        for seed in SEEDS
    }
    (run_root / "reused_kmax18_record.json").write_text(
        json.dumps(reuse_record, indent=2, sort_keys=True) + "\n"
    )

    if stopped_early is not None:
        print(f"STOPPED EARLY at {stopped_early}: exposure gate failed, no substitution made.")
        return 1

    print("All 8 new trajectories completed; exposure gate passed at every point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
