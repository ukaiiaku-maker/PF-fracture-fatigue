"""Bounded second-seed replication of the zero/finite-cohesion comparison
(crack-rebonding causal pilot v2), seed=1001723.

Runs ONLY the four RB2 trajectories the replication was scoped to:
    RB2 reversible zero cohesion
    RB2 reversible finite cohesion
    RB2 persistent zero cohesion
    RB2 persistent finite cohesion

Reuses the IDENTICAL frozen configs from the committed
artifacts/crack_rebonding_causal_pilot_v2/frozen_configuration.json
(reconstructed from their exact serialized fields, not recalibrated) --
same A_NATIVE row, Kmax=18 MPa*sqrt(m), T=300K, R=-0.95, f=1000 Hz,
mpz_n_bins=80, n_phase=80, cohesive scale, kinetics, event target
(MAX_ACCEPTED_EVENTS), and numerical controls. Only the hazard RNG seed
changes (1720 -> 1001723).

Does NOT touch the frozen 0.05-decade expansion threshold, does NOT launch
C0/C1/C4/C5 (already qualified once; this replication only concerns the
zero/finite-cohesion waiting-time comparison), and does NOT proceed to any
multi-K matrix regardless of outcome.

Usage:
    <pinned interpreter> scripts/run_v2_second_seed_replication.py \\
        --out runs/crack_rebonding_causal_pilot_v2_seed1001723
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
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
SEED_REPLICATION = 1001723
TRAJECTORY_KEYS = (
    "RB2_reversible_zero", "RB2_reversible_finite",
    "RB2_persistent_zero", "RB2_persistent_finite",
)


def _check_interpreter() -> None:
    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(
            "wrong interpreter: this replication must run under the frozen "
            f"required interpreter {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}"
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
            f"reloaded config {key!r} hash {actual_hash} does not match the "
            f"committed frozen_configuration.json hash {expected_hash} -- "
            "refusing to replicate against a silently different config"
        )
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="run root directory")
    parser.add_argument(
        "--frozen-configuration", default=str(ARTIFACTS_DIR / "frozen_configuration.json"),
        help="path to the committed frozen_configuration.json to replicate against",
    )
    args = parser.parse_args(argv)

    _check_interpreter()

    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(Path(args.frozen_configuration).read_text())
    if frozen["seed"] != pilot.SEED:
        raise RuntimeError(
            f"frozen_configuration.json's own seed is {frozen['seed']!r}, expected the "
            f"original pilot seed {pilot.SEED!r} -- refusing to replicate against a "
            "configuration frozen under a different original seed"
        )
    common = frozen["common_settings"]
    if not (
        common["T_K"] == pilot.T_K and common["Kmax_Pa_sqrt_m"] == pilot.KMAX_Pa_sqrt_m
        and common["frequency_Hz"] == pilot.F_HZ and common["mpz_n_bins"] == pilot.MPZ_N_BINS
        and common["n_phase"] == pilot.N_PHASE
    ):
        raise RuntimeError("frozen_configuration.json's protocol does not match the mission's frozen values")

    rb_configs = {key: _config_from_frozen(frozen, key) for key in TRAJECTORY_KEYS}
    R = pilot.R_REF

    Engine.configure_hazard(mode="exponential", seed=SEED_REPLICATION)

    def build_engine(rebonding_cfg):
        return build_a_native_engine(rebonding_cfg)

    trajectory_names = {
        "RB2_reversible_zero": "S2_C2R", "RB2_reversible_finite": "S2_C3R",
        "RB2_persistent_zero": "S2_C2P", "RB2_persistent_finite": "S2_C3P",
    }

    results: dict[str, dict] = {}
    for key in TRAJECTORY_KEYS:
        name = trajectory_names[key]
        cfg = rb_configs[key]
        print(f"Running {name} ({key}, seed={SEED_REPLICATION})...")
        t0 = time.monotonic()
        result = pilot.run_trajectory(
            name=name, build_engine=build_engine, make_controller=_make_controller,
            waveform_cls=FatigueWaveform, rebonding_cfg=cfg, R=R,
            reset_engine_registry=Engine.reset_audit,
        )
        elapsed = time.monotonic() - t0
        results[name] = result
        print(f"  {name}: {result['n_accepted_events']} accepted events, "
              f"censored={result['censored']} ({result['censor_reason']}), wall={elapsed:.1f}s")
        (run_root / f"trajectory_{name}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    (run_root / "trajectories.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    # Zero/finite-cohesion waiting-time comparison, interval- and
    # cumulative-basis, under the SAME format as the primary pilot's
    # interval_causal_analysis.csv (frozen 0.05-decade threshold reused
    # verbatim, not re-derived or adjusted after seeing this result).
    import importlib
    analyze = importlib.import_module("analyze_v10_2_30_crack_rebonding_causal_pilot_v2")

    rows_reversible = analyze._matched_delay_rows(
        results["S2_C2R"], results["S2_C3R"], "reversible"
    )
    rows_persistent = analyze._matched_delay_rows(
        results["S2_C2P"], results["S2_C3P"], "persistent"
    )
    all_rows = rows_reversible + rows_persistent
    compression_rows = [r for r in all_rows if r["contains_complete_negative_excursion"]]
    n_unique_compression_intervals = len({r["interval_group_id"] for r in compression_rows})
    max_ratio_compression = max(
        (r["log10_ratio_abs_decade"] for r in compression_rows), default=0.0
    )

    def _cumulative_waiting_time(res: dict) -> float:
        return float(res["cumulative_time_s"])

    cumulative_comparison = {
        "reversible": {
            "t_zero_cumulative_s": _cumulative_waiting_time(results["S2_C2R"]),
            "t_finite_cumulative_s": _cumulative_waiting_time(results["S2_C3R"]),
        },
        "persistent": {
            "t_zero_cumulative_s": _cumulative_waiting_time(results["S2_C2P"]),
            "t_finite_cumulative_s": _cumulative_waiting_time(results["S2_C3P"]),
        },
    }
    for label, d in cumulative_comparison.items():
        d["delta_cumulative_s"] = d["t_finite_cumulative_s"] - d["t_zero_cumulative_s"]
        d["log10_ratio_abs_decade_cumulative"] = (
            abs(__import__("math").log10(d["t_finite_cumulative_s"] / d["t_zero_cumulative_s"]))
            if d["t_zero_cumulative_s"] > 0.0 and d["t_finite_cumulative_s"] > 0.0
            else float("inf")
        )

    replication = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_second_seed_replication_v1",
        "seed": SEED_REPLICATION,
        "original_pilot_seed": pilot.SEED,
        "frozen_configuration_sha256_replicated_against": frozen["frozen_configuration_sha256"],
        "config_hashes_replicated": {key: frozen["config_hashes"][key] for key in TRAJECTORY_KEYS},
        "trajectory_summary": {
            name: {
                "n_accepted_events": res["n_accepted_events"], "censored": res["censored"],
                "censor_reason": res["censor_reason"], "uncensored": res["uncensored"],
                "cumulative_time_s": res["cumulative_time_s"],
            }
            for name, res in results.items()
        },
        "interval_rows": all_rows,
        "n_compression_containing_rows": len(compression_rows),
        "n_unique_compression_containing_intervals": n_unique_compression_intervals,
        "max_log10_ratio_abs_decade_in_compression_containing_intervals": max_ratio_compression,
        "expansion_threshold_log10_decade": pilot.EXPANSION_THRESHOLD_LOG10_DECADE,
        "expansion_threshold_exceeded": max_ratio_compression >= pilot.EXPANSION_THRESHOLD_LOG10_DECADE,
        "cumulative_waiting_time_comparison": cumulative_comparison,
        "note": (
            "threshold reused verbatim from the original pilot (0.05 decades); "
            "not re-derived or adjusted after seeing this replication's result"
        ),
    }

    replication_path = run_root / "second_seed_replication.json"
    replication_path.write_text(json.dumps(replication, indent=2, sort_keys=True) + "\n")
    (ARTIFACTS_DIR / "second_seed_replication.json").write_text(
        json.dumps(replication, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {replication_path}")
    print(f"max_log10_ratio_abs_decade_in_compression_containing_intervals="
          f"{max_ratio_compression:.6f}  expansion_threshold_exceeded="
          f"{replication['expansion_threshold_exceeded']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
