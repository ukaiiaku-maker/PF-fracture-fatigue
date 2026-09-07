"""Run the bounded crack-rebonding causal pilot (v10.2.30, post-S8).

Builds RB0/RB1/RB2-reversible/RB2-persistent configurations, freezes and
hashes the complete pilot configuration BEFORE any trajectory runs, drives
six fresh, unresumed trajectories (P0..P5) in-process against the real
production engine (never through the mesh-dependent CLI backend -- see
crack_rebonding_causal_pilot_v10230.run_trajectory's docstring), performs
the censor-aware causal analysis against the mission's eight hard gates, and
writes every artifact under the given --out run root.

Usage (REQUIRED interpreter, per the frozen configuration's own contract):
    /opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python \\
        scripts/run_v10_2_30_crack_rebonding_causal_pilot.py \\
        --out runs/crack_rebonding_causal_pilot_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _crack_rebonding_engine_fixture as fx  # noqa: E402

from arrhenius_fracture import crack_rebonding_causal_pilot_v10230 as pilot  # noqa: E402
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402
from arrhenius_fracture.fatigue_v1 import (  # noqa: E402
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)


def _check_interpreter() -> None:
    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(
            "wrong interpreter: this pilot must run under the frozen "
            f"required interpreter {pilot.REQUIRED_PYTHON!r}, got "
            f"{sys.executable!r}"
        )


def _make_controller(n_phase: int) -> FatigueCycleHazardController:
    return FatigueCycleHazardController(
        FatigueControllerConfig(
            n_phase=n_phase,
            block_cycles=pilot.BLOCK_CYCLES,
            max_block_cycles=pilot.MAX_BLOCK_CYCLES,
        ),
        None,
        None,
        None,
    )


def _configs_for_key(key: str, frozen: dict) -> object | None:
    payload = frozen["configs"][key]
    if payload is None:
        return None
    from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
        ContactModel,
        CrackRebondingControls,
        FeedbackMode,
        InitialPrecrackWakeMode,
        RebondModelLevel,
    )

    kwargs = dict(payload)
    kwargs["model_level"] = RebondModelLevel(kwargs["model_level"])
    kwargs["contact_model"] = ContactModel(kwargs["contact_model"])
    kwargs["feedback_mode"] = FeedbackMode(kwargs["feedback_mode"])
    kwargs["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(
        kwargs["initial_precrack_wake_mode"]
    )
    return CrackRebondingControls(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="run root directory")
    args = parser.parse_args(argv)

    _check_interpreter()

    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)

    # Hazard stochasticity: exponential mode, frozen seed. Class-level
    # default -- set once, before any engine in this process is built.
    Engine.configure_hazard(mode="exponential", seed=pilot.SEED)

    print("Building reference (bare) engine for material constants...")
    Engine.reset_audit()
    bare = fx.build_real_engine(None)
    reference_contact_radius_m = max(bare.r_eff(), 1.0e-9)
    Eprime_Pa = reduced_modulus_Pa(bare.G, bare.nu)
    print(f"  r_eff={reference_contact_radius_m:.6e} m  G={bare.G:.6e} Pa  "
          f"nu={bare.nu:.4f}  E'={Eprime_Pa:.6e} Pa")

    print("Freezing and hashing pilot configuration (before any trajectory runs)...")
    frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa,
        reference_contact_radius_m=reference_contact_radius_m,
        engine_G_Pa=bare.G,
        engine_nu=bare.nu,
    )
    frozen_path = run_root / "frozen_configuration.json"
    frozen_path.write_text(json.dumps(pilot._jsonable(frozen), indent=2, sort_keys=True) + "\n")
    print(f"  wrote {frozen_path}  sha256={frozen['frozen_configuration_sha256']}")

    rb_configs = {name: _configs_for_key(name, frozen) for name in frozen["configs"]}
    # Confirm the reloaded configs hash identically to what was frozen --
    # the trajectories below must run against exactly the frozen configs,
    # not a silently-redone calibration.
    for name, cfg in rb_configs.items():
        expected = frozen["config_hashes"][name]
        actual = cfg.config_hash() if cfg is not None else None
        if actual != expected:
            raise RuntimeError(
                f"config {name} hash mismatch after reload: expected {expected}, got {actual}"
            )

    trajectory_cfg_key = {
        "RB0": None,
        "RB1": "RB1",
        "RB2_reversible": "RB2_reversible",
        "RB2_persistent": "RB2_persistent",
    }

    results: dict[str, dict] = {}
    for spec in frozen["trajectories"]:
        name = spec["name"]
        R = spec["R"]
        rebonding_key = spec["rebonding"]
        cfg = rb_configs[rebonding_key] if rebonding_key != "RB0" else None
        print(f"Running trajectory {name} (R={R}, rebonding={rebonding_key})...")
        t0 = time.monotonic()
        result = pilot.run_trajectory(
            name=name,
            build_engine=fx.build_real_engine,
            make_controller=_make_controller,
            waveform_cls=FatigueWaveform,
            rebonding_cfg=cfg,
            R=R,
            reset_engine_registry=Engine.reset_audit,
        )
        elapsed = time.monotonic() - t0
        results[name] = result
        print(
            f"  {name}: {result['n_accepted_events']} accepted events, "
            f"censored={result['censored']} ({result['censor_reason']}), "
            f"wall={elapsed:.1f}s"
        )
        traj_path = run_root / f"trajectory_{name}.json"
        traj_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    trajectories_path = run_root / "trajectories.json"
    trajectories_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {trajectories_path}")

    print("Running censor-aware causal analysis (hard gates 1-8)...")
    analysis = pilot.causal_analysis(results)
    analysis_path = run_root / "causal_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"wrote {analysis_path}  overall_pass={analysis['overall_pass']}")

    return 0 if analysis["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
