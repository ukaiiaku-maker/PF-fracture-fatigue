"""Part X: run exactly one scientific job in its own fresh Python process.

This is the ONE and ONLY entry point the controller launches as a
subprocess -- fresh-process-per-job is not an incidental implementation
detail, it is the reproducibility guarantee PX2.5 established (see
tests/test_v10_2_30_crack_rebonding_part_x_engine_reproducibility.py):
Engine.configure_hazard(seed=...) + Engine.reset_audit() on the actual
leaf production class, called here before any construction, give exact
reproducibility regardless of arbitrary prior process history.

Two independent budgets, selected by --preflight:

- ``--preflight``: a bounded handful of blocks (PREFLIGHT_MAX_BLOCKS),
  unconditionally recorded and total-block-counted, sufficient to qualify
  the controller's process-per-job/quarantine/registry machinery end to
  end (PX2.5). Never a physical datum. This code path and its result
  schema are UNCHANGED from PX2.5 -- do not couple it to the real screen
  budget below.

- real budget (this flag absent): the PX3 screen budget from mission
  section 7's common screen settings -- 12 accepted events or 60 um,
  whichever comes first, driven by the same qualified, PX2.5-extended
  event loop the crack-rebonding causal pilot v2 uses
  (``crack_rebonding_causal_pilot_v2_v10230.run_trajectory``), never a
  second reimplementation of event acceptance/extension bookkeeping.
  Only ``engine.cycle_step_waveform``/``commit_energy_gated_event`` are
  ever called (never the mesh-dependent ``--fatigue-cycles`` CLI
  backend's accelerated/DMD swap), so this is unconditionally
  V10230_FATIGUE_INTEGRATOR_MODE=explicit by construction -- DMD/
  Poincare/projective acceleration is simply never reachable from this
  code path. This branch additionally resolves each job's
  ``chemistry_factor`` and ``K_rebond_max_target_Pa_sqrt_m`` screen-panel
  overrides against the row's frozen base config (PX2's kinetic_regime_
  registry.json only records each ROW's baseline config; sections 7.5/7.6
  vary chemistry_factor/restored_work_of_separation_J_m2 per job on top of
  that baseline -- see build_part_x_kinetic_regime_registry.py's
  canonical_job_key/_screen_job, whose K_target->G_max mapping
  (G = (K_target / rebond_K_geometry_factor)**2 / Eprime_Pa) is
  reproduced here exactly).

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
from dataclasses import replace as _dc_replace
from pathlib import Path

tempfile.tempdir = tempfile.mkdtemp(prefix=f"part_x_job_{os.getpid()}_")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PREFLIGHT_MAX_BLOCKS = 5

# Mission section 7 "Common screen settings".
SCREEN_MAX_ACCEPTED_EVENTS = 12
SCREEN_MAX_PROJECTED_EXTENSION_m = 60.0e-6
SCREEN_N_PHASE = 80
SCREEN_T_K = 300.0


def _load_row_payload(job: dict) -> dict:
    registry = json.loads(
        (REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "kinetic_regime_registry.json").read_text()
    )
    row_payload = registry["rows"][job["row_name"]]
    if row_payload["config_hash"] != job["config_hash"]:
        raise RuntimeError(
            f"config_hash mismatch: job requests {job['config_hash']}, "
            f"registry row {job['row_name']} has {row_payload['config_hash']} -- refusing to launch"
        )
    return row_payload


def _base_rebonding_cfg(row_payload: dict):
    from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
        ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
    )

    cfg_dict = dict(row_payload["config"])
    cfg_dict["model_level"] = RebondModelLevel(cfg_dict["model_level"])
    cfg_dict["contact_model"] = ContactModel(cfg_dict["contact_model"])
    cfg_dict["feedback_mode"] = FeedbackMode(cfg_dict["feedback_mode"])
    cfg_dict["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(cfg_dict["initial_precrack_wake_mode"])
    return CrackRebondingControls(**cfg_dict)


def _resolve_screen_rebonding_cfg(job: dict, row_payload: dict, Eprime_Pa: float):
    """Apply the job's chemistry_factor / K_rebond_max_target_Pa_sqrt_m
    screen-panel overrides on top of the row's frozen baseline config,
    reproducing build_part_x_kinetic_regime_registry.py's own
    K_target -> restored_work_of_separation_J_m2 mapping exactly (that
    script is the single source of truth for this formula -- see its
    ``G_max_target = (K_b_target / cfg.rebond_K_geometry_factor) ** 2 /
    Eprime_Pa``)."""
    base_cfg = _base_rebonding_cfg(row_payload)
    chemistry_factor = float(job["chemistry_factor"])
    K_target = float(job["K_rebond_max_target_Pa_sqrt_m"])
    if job["cohesion"] == "zero":
        G = 0.0
    else:
        G = (K_target / base_cfg.rebond_K_geometry_factor) ** 2 / Eprime_Pa
    return _dc_replace(
        base_cfg, chemistry_factor=chemistry_factor, restored_work_of_separation_J_m2=G,
    ).validate()


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
        from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
        from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
            CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
        )
        from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding

        row_payload = _load_row_payload(job)
        seed = int(job["seed"])

        if args.preflight:
            from dataclasses import replace as _dc_replace  # noqa: F811 -- local shadow, harmless

            rebonding_cfg = _base_rebonding_cfg(row_payload)
            if job["cohesion"] == "zero":
                rebonding_cfg = _dc_replace(rebonding_cfg, restored_work_of_separation_J_m2=0.0)
            rebonding_cfg = rebonding_cfg.validate()
            actual_config_hash = rebonding_cfg.config_hash()

            Engine.configure_hazard(mode="exponential", seed=seed)
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

            events = []
            blocks_run = 0
            for i in range(PREFLIGHT_MAX_BLOCKS):
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
                "preflight": True,
            }
            (result_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            status_path.write_text(json.dumps({"status": "COMPLETE_PREFLIGHT"}, indent=2))
            return 0

        # --- Real PX3 screen budget: 12 accepted events or 60 um, whichever
        # first, driven by the qualified causal-pilot-v2 event loop. ---
        from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
        from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa

        Engine.configure_hazard(mode="exponential", seed=seed)
        Engine.reset_audit()
        bare_engine, bare_audit = build_a_native_engine(None)
        Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)

        rebonding_cfg = _resolve_screen_rebonding_cfg(job, row_payload, Eprime_Pa)
        actual_config_hash = rebonding_cfg.config_hash()

        def _make_controller(n_phase: int):
            return FatigueCycleHazardController(
                FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6),
                None, None, None,
            )

        def _build_engine(cfg):
            return build_a_native_engine(cfg)

        status_path.write_text(json.dumps({"status": "PHYSICS_RUNNING"}, indent=2))

        trajectory = pilot.run_trajectory(
            name=job["canonical_job_key"][:16],
            build_engine=_build_engine,
            make_controller=_make_controller,
            waveform_cls=FatigueWaveform,
            rebonding_cfg=rebonding_cfg,
            R=float(job["R"]),
            reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=float(job["Kmax_Pa_sqrt_m"]),
            frequency_Hz=float(job["frequency_Hz"]),
            n_phase=SCREEN_N_PHASE,
            T_K_=SCREEN_T_K,
            max_accepted_events=SCREEN_MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=SCREEN_MAX_PROJECTED_EXTENSION_m,
            hazard_rng_seed=seed,
            minimum_load_hold_s=float(job["minimum_load_hold_s"]),
        )

        result = {
            "schema": "v10230_part_x_screen_job_result_v1",
            "job": job,
            "actual_config_hash": actual_config_hash,
            "Eprime_Pa": Eprime_Pa,
            "engine_mro": [c.__name__ for c in type(bare_engine).__mro__],
            "material_row_sha256": bare_audit["source_row_sha256"],
            "trajectory": trajectory,
            "preflight": False,
        }
        (result_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        status_path.write_text(json.dumps({"status": "COMPLETE"}, indent=2))
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
