"""Part X: run exactly one scientific job in its own fresh Python process.

This is the ONE and ONLY entry point the controller launches as a
subprocess -- fresh-process-per-job is not an incidental implementation
detail, it is the reproducibility guarantee PX2.5 established (see
tests/test_v10_2_30_crack_rebonding_part_x_engine_reproducibility.py):
Engine.configure_hazard(seed=...) + Engine.reset_audit() on the actual
leaf production class, called here before any construction, give exact
reproducibility regardless of arbitrary prior process history.

Currently implements a BOUNDED PREFLIGHT trajectory (a handful of blocks,
not the real 12-event/60um screen budget) -- sufficient to qualify the
controller's process-per-job/quarantine/registry machinery end to end.
The real PX3 screen budget (12 accepted events or 60 um, explicit
phase-resolved integration) extends this same entry point; the
controller/orchestration layer does not change.

A real concurrency bug was caught by the controller's own preflight
qualification (max 3 workers, run concurrently): ``a_native_engine_
v10230.build_a_native_manifest`` writes its compatibility manifest to a
FIXED, non-process-unique path
(``tempfile.gettempdir()/v10230_a_native_v2_manifest/selected_material_
manifest_A_NATIVE.csv``), so two concurrent workers race to write/rename
the same file and one gets ``FileNotFoundError`` on the atomic rename.
Fixed here, not in the shared production module (avoiding any change to
code other studies/tests depend on): each job process points
``tempfile.tempdir`` at its own PID-qualified directory before importing
anything that constructs the engine, giving every concurrent worker a
private temp namespace with zero risk of collision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"part_x_job_{os.getpid()}_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PREFLIGHT_MAX_BLOCKS = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True, help="Path to this job's canonical job-spec JSON")
    parser.add_argument("--result-dir", required=True, help="Virgin output directory for this job's results")
    parser.add_argument("--preflight", action="store_true", help="Run the bounded preflight budget instead of the real screen budget")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)  # must be virgin -- fails loudly on any collision

    status_path = result_dir / "job_status.json"
    status_path.write_text(json.dumps({"status": "PREPHYSICS_STARTED"}, indent=2))

    try:
        job = json.loads(Path(args.job_json).read_text())

        from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
        from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls
        from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
        from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
            CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
        )
        from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding

        # Reconstruct the exact rebonding config from its recorded hash's
        # source row in the kinetic_regime_registry, or (for the zero-
        # cohesion control) skip rebonding installation entirely.
        registry = json.loads(
            (REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "kinetic_regime_registry.json").read_text()
        )
        row_payload = registry["rows"][job["row_name"]]
        if row_payload["config_hash"] != job["config_hash"]:
            raise RuntimeError(
                f"config_hash mismatch: job requests {job['config_hash']}, "
                f"registry row {job['row_name']} has {row_payload['config_hash']} -- refusing to launch"
            )

        from dataclasses import replace as _dc_replace
        from arrhenius_fracture.crack_rebonding_kinetics_v10230 import RebondModelLevel, ContactModel, FeedbackMode, InitialPrecrackWakeMode

        cfg_dict = dict(row_payload["config"])
        cfg_dict["model_level"] = RebondModelLevel(cfg_dict["model_level"])
        cfg_dict["contact_model"] = ContactModel(cfg_dict["contact_model"])
        cfg_dict["feedback_mode"] = FeedbackMode(cfg_dict["feedback_mode"])
        cfg_dict["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(cfg_dict["initial_precrack_wake_mode"])
        rebonding_cfg = CrackRebondingControls(**cfg_dict)
        if job["cohesion"] == "zero":
            rebonding_cfg = _dc_replace(rebonding_cfg, restored_work_of_separation_J_m2=0.0)
        rebonding_cfg = rebonding_cfg.validate()

        actual_config_hash = rebonding_cfg.config_hash()

        # Fresh-process reproducibility pattern (PX2.5-qualified).
        Engine.configure_hazard(mode="exponential", seed=int(job["seed"]))
        Engine.reset_audit()
        engine, audit = build_a_native_engine()
        install_crack_rebonding(engine, rebonding_cfg)

        ctrl = FatigueCycleHazardController(
            FatigueControllerConfig(n_phase=80, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
        )
        wave = FatigueWaveform(
            Kmax=float(job["Kmax_Pa_sqrt_m"]), R=float(job["R"]), frequency_Hz=float(job["frequency_Hz"]),
            minimum_load_hold_s=float(job["minimum_load_hold_s"]),
        )

        status_path.write_text(json.dumps({"status": "PHYSICS_RUNNING"}, indent=2))

        max_blocks = PREFLIGHT_MAX_BLOCKS if args.preflight else None
        if max_blocks is None:
            raise NotImplementedError(
                "The real PX3 screen budget (12 accepted events or 60 um, explicit "
                "phase-resolved integration) is implemented as a PX3 extension of this "
                "entry point, not yet wired -- only --preflight is currently supported."
            )

        events = []
        blocks_run = 0
        for i in range(max_blocks):
            r = engine.cycle_step_waveform(ctrl, wave, 300.0)
            blocks_run += 1
            if r.get("fired"):
                events.append({"block": i, "cycles_consumed": r["cycles_consumed"], "B": r["B"]})

        result = {
            "schema": "v10230_part_x_job_result_v1",
            "job": job,
            "actual_config_hash": actual_config_hash,
            "engine_mro": [c.__name__ for c in type(engine).__mro__],
            "engine_id": engine._engine_id,
            "material_row_sha256": audit["source_row_sha256"],
            "blocks_run": blocks_run,
            "events": events,
            "final_B": float(engine.B),
            "final_t_s": float(engine.t),
            "preflight": bool(args.preflight),
        }
        (result_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        status_path.write_text(json.dumps({"status": "COMPLETE_PREFLIGHT" if args.preflight else "COMPLETE"}, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001 -- top-level job boundary, must not crash the controller
        (result_dir / "prephysics_failure.json").write_text(json.dumps({
            "status": "PREPHYSICS_LAUNCH_FAILURE",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2))
        status_path.write_text(json.dumps({"status": "PREPHYSICS_LAUNCH_FAILURE"}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
