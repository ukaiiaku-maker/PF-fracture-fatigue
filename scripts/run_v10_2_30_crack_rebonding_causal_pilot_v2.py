"""Run the corrected crack-rebonding causal pilot (v10.2.30, V2).

Builds C0/C1/C2R/C3R/C2P/C3P/C4/C5, freezes and hashes the complete pilot
configuration BEFORE any trajectory runs, drives eight fresh, unresumed
trajectories in-process against the real A_NATIVE production engine
(arrhenius_fracture.a_native_engine_v10230.build_a_native_engine -- never
through the mesh-dependent CLI backend), and writes every artifact under
the given --out run root plus the tracked artifacts/ bundle.

Usage (REQUIRED interpreter, per the frozen configuration's own contract):
    /opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python \\
        scripts/run_v10_2_30_crack_rebonding_causal_pilot_v2.py \\
        --out runs/crack_rebonding_causal_pilot_v2
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
from arrhenius_fracture.a_native_engine_v10230 import (  # noqa: E402
    build_a_native_engine,
    load_a_native_provenance,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"


def _check_interpreter() -> None:
    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(
            "wrong interpreter: this pilot must run under the frozen "
            f"required interpreter {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}"
        )


def _make_controller(n_phase: int) -> FatigueCycleHazardController:
    return FatigueCycleHazardController(
        FatigueControllerConfig(
            n_phase=n_phase, block_cycles=pilot.BLOCK_CYCLES, max_block_cycles=pilot.MAX_BLOCK_CYCLES,
        ),
        None, None, None,
    )


def _configs_for_key(key: str, frozen: dict) -> object | None:
    payload = frozen["configs"][key]
    if payload is None:
        return None
    from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
        ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
    )

    kwargs = dict(payload)
    kwargs["model_level"] = RebondModelLevel(kwargs["model_level"])
    kwargs["contact_model"] = ContactModel(kwargs["contact_model"])
    kwargs["feedback_mode"] = FeedbackMode(kwargs["feedback_mode"])
    kwargs["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(kwargs["initial_precrack_wake_mode"])
    return CrackRebondingControls(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="run root directory")
    args = parser.parse_args(argv)

    _check_interpreter()

    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    Engine.configure_hazard(mode="exponential", seed=pilot.SEED)
    Engine.reset_audit()

    print("Building reference (bare) A_NATIVE engine for material constants...")
    bare_engine, bare_audit = build_a_native_engine(None)
    reference_contact_radius_m = max(bare_engine.r_eff(), 1.0e-9)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    print(f"  r_eff={reference_contact_radius_m:.6e} m  G={bare_engine.G:.6e} Pa  "
          f"nu={bare_engine.nu:.4f}  E'={Eprime_Pa:.6e} Pa")

    provenance = load_a_native_provenance()

    print("Freezing and hashing pilot configuration (before any trajectory runs)...")
    frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=reference_contact_radius_m,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    frozen_path = run_root / "frozen_configuration.json"
    frozen_path.write_text(json.dumps(pilot._jsonable(frozen), indent=2, sort_keys=True) + "\n")
    (ARTIFACTS_DIR / "frozen_configuration.json").write_text(
        json.dumps(pilot._jsonable(frozen), indent=2, sort_keys=True) + "\n"
    )
    print(f"  wrote {frozen_path}  sha256={frozen['frozen_configuration_sha256']}")

    rb_configs = {name: _configs_for_key(name, frozen) for name in frozen["configs"]}
    for name, cfg in rb_configs.items():
        expected = frozen["config_hashes"][name]
        actual = cfg.config_hash() if cfg is not None else None
        if actual != expected:
            raise RuntimeError(f"config {name} hash mismatch after reload: expected {expected}, got {actual}")

    def build_engine(rebonding_cfg):
        return build_a_native_engine(rebonding_cfg)

    results: dict[str, dict] = {}
    for spec in frozen["trajectories"]:
        name = spec["name"]
        R = spec["R"]
        rebonding_key = spec["rebonding"]
        cfg = rb_configs[rebonding_key] if rebonding_key != "RB0" else None
        print(f"Running trajectory {name} (R={R}, rebonding={rebonding_key})...")
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
        traj_path = run_root / f"trajectory_{name}.json"
        traj_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    trajectories_path = run_root / "trajectories.json"
    trajectories_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {trajectories_path}")

    summary = {
        name: {
            "R": res["R"], "rebonding_model_level": res["rebonding_model_level"],
            "n_accepted_events": res["n_accepted_events"], "censored": res["censored"],
            "censor_reason": res["censor_reason"], "uncensored": res["uncensored"],
            "cumulative_extension_m": res["cumulative_extension_m"],
            "cumulative_time_s": res["cumulative_time_s"],
            "n_intervals_with_complete_negative_excursion": sum(
                1 for iv in res["post_first_event_intervals"] if iv["complete_negative_excursion"]
            ),
        }
        for name, res in results.items()
    }
    summary_path = run_root / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (ARTIFACTS_DIR / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
