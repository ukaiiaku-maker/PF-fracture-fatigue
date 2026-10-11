"""R3 contracts for the qualified 900 K persistent-source comparison.

This module deliberately does not call the archived spatial rows a reduced
prediction.  It resolves the actual consumers on each side, exposes the two
different local-strength sources, and labels deterministic and sampled renewal
strategies separately even though both translate the MPZ continuously between
outer geometry commits.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .reduced_persistent_reference_r2 import (
    DEFAULT_KERNEL_FAMILY, advance_signed, build_restored_engine, runtime_receipt,
)
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .persistent_site_source_v10221 import _source_geometry
from .stochastic_avalanche_tip import clear_pending_geometry_events
from .reduced_shared_state_v1023 import SharedReducedConfig
from .signed_burgers_shared_v1025 import _active_K, _wake_K


MODEL_ID = "r3.matched-persistent-execution/1"
SPATIAL_EVIDENCE = (
    Path(__file__).resolve().parents[1] / "integration_evidence"
    / "option_completion_v2" / "event_online_qualified"
)
SPATIAL_SOURCE_COMMIT = "b6e878d39c97b687f4b3c7b28f8795ec6f295d1f"
SPATIAL_SOURCE_TREE = "a77b8a587d6583797ab82df4340ae8a8b81d7b13"
SPATIAL_COHESIVE_STRENGTH_PA = 30.0e9
REDUCED_EVENT_LAW_ID = "UNIT_MARK_FIXED_RENEWAL_WITH_CONTINUOUS_MPZ_TRANSLATION"
SPATIAL_EVENT_LAW_ID = REDUCED_EVENT_LAW_ID
AVAILABLE_STOCHASTIC_EVENT_LAW_ID = "EXPONENTIAL_MARK_THRESHOLD_CORRELATED_RENEWAL"


def matched_config() -> SharedReducedConfig:
    """Numerics and geometry serialized by the qualified 900 K spatial case."""
    return SharedReducedConfig(
        checkpoint_da_um=5.0,
        r0_m=1.0e-6,
        mpz_length_um=50.0,
        mpz_n_bins=80,
        wake_length_um=100.0,
        mobile_shield_fraction=0.0,
        crystal_theta_deg=30.0,
        # The autonomous reduced provider owns one declared tensor shape.  The
        # recorded-operator replay below replaces it stepwise with each archived
        # FEM probe, without copying any constitutive state.
        drive_factors=(0.32209902346877634, 0.5140167394041307),
        max_action_substep=0.02,
        max_translation_substep_m=1.0e-7,
        max_internal_steps=100000,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spatial_identity() -> dict[str, Any]:
    root = SPATIAL_EVIDENCE
    attestation = json.loads((root / "shared_runtime_attestation.post.json").read_text())
    selection = json.loads((root / "v10_2_22_parameter_selection.json").read_text())
    run_args = json.loads((root / "run_args.json").read_text())
    source = attestation["source"]
    if source["commit"] != SPATIAL_SOURCE_COMMIT or source["tree"] != SPATIAL_SOURCE_TREE:
        raise RuntimeError("qualified spatial source identity changed")
    return {
        "evidence_directory": str(root.resolve()),
        "source_commit": source["commit"],
        "source_tree_including_dirty_capture": source["tree"],
        "source_dirty_at_execution": bool(source["dirty"]),
        "executor": attestation["executor"],
        "material": {
            "path": str((root / "selected_material_manifest_v10_2_22.csv").resolve()),
            "sha256": _sha256(root / "selected_material_manifest_v10_2_22.csv"),
            "candidate_id": selection["candidate_id"],
        },
        "selection": {
            "path": str((root / "v10_2_22_parameter_selection.json").resolve()),
            "sha256": _sha256(root / "v10_2_22_parameter_selection.json"),
        },
        "kernel": {
            "path": str(DEFAULT_KERNEL_FAMILY.resolve()),
            "sha256": _sha256(DEFAULT_KERNEL_FAMILY),
        },
        "loading": {
            "temperature_K": 900.0,
            "control": "top_displacement_increment",
            "dU_m": float(run_args["dU"]),
            "dt_s": float(run_args["dt"]),
            "orientation_deg": float(run_args["crystal_theta_deg"]),
            "steps": int(run_args["steps"]),
        },
        "event_strategy": spatial_event_contract(),
    }


def spatial_rows() -> list[dict[str, float]]:
    with (SPATIAL_EVIDENCE / "steps_0900K.csv").open(newline="") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def spatial_tensor_drives() -> list[dict[str, Any]]:
    payload = json.loads(
        (SPATIAL_EVIDENCE / "anisotropic_emission_audit_v10174.json").read_text()
    )
    return [
        {
            "drive_serial": int(row["anisotropic_drive_serial"]),
            "drive_factors": [float(value) for value in row["anisotropic_drive_factors"]],
            "tau_signed_Pa": [float(value) for value in row["anisotropic_tau_signed_Pa"]],
            "reliable": bool(row["anisotropic_drive_reliable"]),
        }
        for row in payload["records"]
    ]


def spatial_event_contract() -> dict[str, Any]:
    audit = json.loads(
        (SPATIAL_EVIDENCE / "anisotropic_emission_audit_v10174.json").read_text()
    )
    wrapper = json.loads(
        (SPATIAL_EVIDENCE / "v10_2_17_final_signed_stochastic_stack.json").read_text()
    )
    hazard = audit["stochastic_hazard"]
    reward = audit["stochastic_avalanche"]
    selected_sampled = hazard["mode"] == "exponential"
    return {
        "law_id": (AVAILABLE_STOCHASTIC_EVENT_LAW_ID
                   if selected_sampled else REDUCED_EVENT_LAW_ID),
        "observed_engine_hazard_mode": hazard["mode"],
        "observed_engine_reward_mode": reward["mode"],
        "wrapper_declared_hazard_mode": wrapper["cleavage_hazard_mode"],
        "wrapper_declared_reward_mode": wrapper["event_length_mode"],
        "wrapper_matches_post_install_consumer": bool(
            wrapper["cleavage_hazard_mode"] == hazard["mode"]
            and wrapper["event_length_mode"] == reward["mode"]
        ),
        "selected_case_uses_sampled_marks": selected_sampled,
        "continuous_fractional_progress": True,
        "continuous_mpz_translation": True,
        "geometry_commit_at_unit_checkpoint": not selected_sampled,
        "available_but_not_selected_stochastic_law_id": (
            AVAILABLE_STOCHASTIC_EVENT_LAW_ID
        ),
    }


def build_matched_reduced_engine(*, spatial_strength_contract: bool = True):
    return build_restored_engine(
        matched_config(),
        material_path=SPATIAL_EVIDENCE / "selected_material_manifest_v10_2_22.csv",
        selection_path=SPATIAL_EVIDENCE / "v10_2_22_parameter_selection.json",
        local_strength_Pa=(SPATIAL_COHESIVE_STRENGTH_PA
                           if spatial_strength_contract else None),
    )


def _callable(value: Any) -> str:
    function = getattr(value, "__func__", value)
    return f"{function.__module__}.{function.__qualname__}"


def event_contract(engine: Any) -> dict[str, Any]:
    hazard = engine.hazard_cfg
    avalanche = engine.avalanche_cfg
    sampled = str(hazard.mode) == "exponential"
    return {
        "law_id": SPATIAL_EVENT_LAW_ID if sampled else REDUCED_EVENT_LAW_ID,
        "integrator": _callable(engine._integrate_coupled),
        "hazard_mark_mode": str(hazard.mode),
        "current_mark_action": float(engine.hazard_threshold_action),
        "reward_mode": str(avalanche.mode),
        "base_checkpoint_m": float(engine.avalanche_base_checkpoint_m),
        "current_proposed_reward_m": float(engine.avalanche_event_advance_m),
        "continuous_fractional_progress": True,
        "continuous_mpz_translation": True,
        "outer_geometry_only_at_mark_crossing": True,
        "checkpoint_sync_translates_mpz_again": False,
        "mark_consumed_once_by": _callable(engine._integrate_coupled),
        "pending_geometry_queue_is_outer_commit_request_only": True,
        "stochastic_mark_crossing": sampled,
    }


def stress_contract(engine: Any, applied_K_Pa_sqrt_m: float) -> dict[str, Any]:
    supplied = float(engine.sigma_tip(float(applied_K_Pa_sqrt_m)))
    active = float(_active_K(engine.mpz))
    wake = float(_wake_K(engine.mpz))
    effective = float(engine._effective_K_tip_Pa_sqrt_m)
    raw = float(engine._opening_sigma_uncapped_Pa)
    strength = float(engine.f.sigma_cap)
    factors = np.asarray(
        getattr(engine.mpz, "_anisotropic_drive_factors", np.ones(engine.mpz.n_systems)),
        dtype=float,
    )
    backstress = np.asarray(
        getattr(engine.mpz, "persistent_site_last_sigma_back_Pa",
                np.zeros(engine.mpz.n_systems)), dtype=float,
    )
    emission_opening = supplied * factors
    return {
        "applied_K_Pa_sqrt_m": float(applied_K_Pa_sqrt_m),
        "effective_K_Pa_sqrt_m": effective,
        "active_shield_K_Pa_sqrt_m": active,
        "wake_shield_K_Pa_sqrt_m": wake,
        "raw_local_opening_stress_Pa": raw,
        "local_strength_Pa": strength,
        "local_strength_units": "Pa",
        "local_strength_source": getattr(
            engine, "_r2_local_strength_source", "front_config.sigma_cap"
        ),
        "normalized_atlas_coordinate": float(
            engine._signed_last_state_coordinates["opening_strength_fraction"]
        ),
        "local_strength_limiter_active": bool(raw > strength),
        "cleavage_stress_supplied_Pa": supplied,
        "source_opening_stress_by_channel_Pa": emission_opening.tolist(),
        "source_backstress_by_channel_Pa": backstress.tolist(),
        "source_effective_stress_by_channel_Pa": np.maximum(
            emission_opening - backstress, 0.0
        ).tolist(),
        "transport_signed_direction_by_channel": np.asarray(
            getattr(engine.mpz, "_anisotropic_tau_signed_Pa",
                    np.zeros(engine.mpz.n_systems)), dtype=float
        ).tolist(),
        "active_overrides": {
            "stress_override": None,
            "lambda_override": None,
            "mobile_shield_fraction": float(engine.mpz.cfg.mobile_shield_fraction),
            "wake_shielding": bool(engine.mpz.cfg.wake_shielding),
        },
        "sigma_tip_consumer": _callable(engine.sigma_tip),
    }


def archived_spatial_stress_contract(row: Mapping[str, float]) -> dict[str, Any]:
    applied = float(row["KJ_Pa_sqrtm"])
    supplied = float(row["sigma_tip_Pa"])
    effective = applied - float(row["mpz_K_shield_Pa_sqrt_m"])
    # The selected 30 GPa ceiling is inactive in all three archived rows, so
    # the returned stress is also the actual raw stress at the evolved radius.
    raw = supplied
    radius = (effective / max(raw, 1.0)) ** 2 / (2.0 * math.pi)
    return {
        "applied_K_Pa_sqrt_m": applied,
        "effective_K_Pa_sqrt_m": effective,
        "active_shield_K_Pa_sqrt_m": float(row["mpz_K_shield_Pa_sqrt_m"]),
        "wake_shield_K_Pa_sqrt_m": 0.0,
        "raw_local_opening_stress_Pa": raw,
        "local_strength_Pa": SPATIAL_COHESIVE_STRENGTH_PA,
        "local_strength_units": "Pa",
        "local_strength_source": "run_args.sigma_cap_GPa",
        "normalized_atlas_coordinate": min(max(supplied / SPATIAL_COHESIVE_STRENGTH_PA, 0.0), 1.0),
        "local_strength_limiter_active": bool(raw > SPATIAL_COHESIVE_STRENGTH_PA),
        "cleavage_stress_supplied_Pa": supplied,
        "analytical_kinetic_radius_m": radius,
        "stress_override": None,
        "historical_runtime_note": (
            "The archived value is the selected engine's sigma_tip output; no "
            "FEM stress_override appears in the qualified call path."
        ),
    }


def run_recorded_operator_replay(engine: Any) -> list[dict[str, Any]]:
    """Apply only the archived K/time schedule; no future state is copied."""
    result = []
    rows = spatial_rows()
    drives = spatial_tensor_drives()
    for index, (row, drive) in enumerate(zip(rows, drives, strict=True), start=1):
        engine.mpz._anisotropic_drive_factors = np.asarray(
            drive["drive_factors"], dtype=float
        )
        engine.mpz._anisotropic_tau_signed_Pa = np.asarray(
            drive["tau_signed_Pa"], dtype=float
        )
        engine.mpz._anisotropic_drive_reliable = bool(drive["reliable"])
        engine.mpz._anisotropic_drive_serial = int(drive["drive_serial"])
        before = stress_contract(engine, row["KJ_Pa_sqrtm"])
        step = advance_signed(
            engine, K_Pa_sqrt_m=row["KJ_Pa_sqrtm"],
            temperature_K=900.0, dt_s=row["dt_cur_s"],
        )
        result.append({
            "step": index,
            "physical_time_s": sum(item["dt_cur_s"] for item in rows[:index]),
            "recorded_spatial": dict(row),
            "recorded_tensor_drive": drive,
            "reduced_before": before,
            "reduced_after": stress_contract(engine, row["KJ_Pa_sqrtm"]),
            "reduced_step": step,
            "event_contract": event_contract(engine),
        })
    return result


def _state_row(engine: Any) -> dict[str, Any]:
    diagnostics = engine.mpz.diagnostics()
    source = _source_geometry(engine.mpz)
    return {
        "time_s": float(engine.t),
        "physical_crack_extension_m": float(engine.checkpoint_advance_total_m),
        "continuous_micro_advance_m": float(engine.micro_advance_total_m),
        "checkpoint_count": int(engine.n_adv),
        "fractional_checkpoint_progress": float(engine.B),
        "tip_radius_m": float(engine.r_eff()),
        "front_width_m": float(source["front_width_m"]),
        "source_area_m2": float(source.get("source_area_m2", 0.0)),
        "source_multiplicity": float(source.get("multiplicity_per_system", 0.0)),
        "gross_source_slip": float(engine.mpz.cumulative_gross_source_activity),
        "gross_return_slip": float(engine.mpz.cumulative_gross_return_activity),
        "net_source_slip": float(engine.mpz.local_slip_count()),
        "active_shield_K_Pa_sqrt_m": float(_active_K(engine.mpz)),
        "wake_shield_K_Pa_sqrt_m": float(_wake_K(engine.mpz)),
        "diagnostics": diagnostics,
    }


def _restore_geometry(engine: Any, checkpoint: Path) -> None:
    payload = json.loads((checkpoint / "high_cycle_live_checkpoint.json").read_text())
    geometry = payload["geometry_signature"]
    engine.n_adv = int(geometry[0])
    engine.a_adv = float(geometry[1])
    engine.micro_advance_total_m = float(geometry[2])
    engine.checkpoint_advance_total_m = float(geometry[3])
    engine.mpz.advance_total_m = float(geometry[4])


def _same_active_state(left: Any, right: Any) -> bool:
    return bool(
        np.array_equal(left.vector, right.vector)
        and left.fields == right.fields
        and left.geometry_signature == right.geometry_signature
    )


def run_matched_campaign(output_directory: str | Path) -> dict[str, Any]:
    """Run recorded replay plus an independent load/unload/reload history."""
    root = Path(output_directory).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    clear_pending_geometry_events()

    replay_engine = build_matched_reduced_engine()
    virgin_contract = stress_contract(
        replay_engine, spatial_rows()[0]["KJ_Pa_sqrtm"]
    )
    replay = run_recorded_operator_replay(replay_engine)
    active_contract = stress_contract(
        replay_engine, spatial_rows()[-1]["KJ_Pa_sqrtm"]
    )
    checkpoint = root / "recorded_replay_checkpoint"
    previous = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint)
    try:
        write_checkpoint(
            replay_engine, temperature_K=900.0,
            reason="r3_matched_recorded_replay",
            metadata={"model": MODEL_ID, "stage": "post-recorded-replay"},
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = previous
    before_restart = serialize_active_state(replay_engine)
    restarted = build_matched_reduced_engine()
    _restore_geometry(restarted, checkpoint)
    restore_checkpoint(restarted, checkpoint)
    restart_exact = _same_active_state(before_restart, serialize_active_state(restarted))
    restart_contract = stress_contract(
        restarted, spatial_rows()[-1]["KJ_Pa_sqrtm"]
    )

    autonomous = build_matched_reduced_engine()
    rows = spatial_rows()
    # External load only: the reduced implementation receives no recorded
    # tensor state, event time, or future constitutive value.
    load_history = [
        ("load_10um", rows[0]["KJ_Pa_sqrtm"]),
        ("load_20um", rows[1]["KJ_Pa_sqrtm"]),
        ("load_30um", rows[2]["KJ_Pa_sqrtm"]),
        ("unload_20um", rows[1]["KJ_Pa_sqrtm"]),
        ("reload_30um", rows[2]["KJ_Pa_sqrtm"]),
    ]
    autonomous_rows = []
    for stage, applied_K in load_history:
        before = _state_row(autonomous)
        contract_before = stress_contract(autonomous, applied_K)
        step = advance_signed(
            autonomous, K_Pa_sqrt_m=applied_K,
            temperature_K=900.0, dt_s=rows[0]["dt_cur_s"],
        )
        autonomous_rows.append({
            "stage": stage,
            "applied_K_Pa_sqrt_m": applied_K,
            "duration_s": rows[0]["dt_cur_s"],
            "before": before,
            "stress_contract_before": contract_before,
            "step": step,
            "after": _state_row(autonomous),
        })

    replay_comparison = []
    for item in replay:
        spatial = item["recorded_spatial"]
        reduced = item["reduced_step"]
        replay_comparison.append({
            "step": item["step"],
            "spatial_sigma_tip_Pa": spatial["sigma_tip_Pa"],
            "reduced_sigma_tip_Pa": reduced["sigma_tip"],
            "relative_sigma_difference": abs(
                reduced["sigma_tip"] - spatial["sigma_tip_Pa"]
            ) / max(abs(spatial["sigma_tip_Pa"]), 1.0),
            "spatial_N_em": spatial["N_em"],
            "reduced_N_em": reduced["N_em"],
            "spatial_fired": bool(spatial["n_fire"] > 0.0),
            "reduced_fired": bool(reduced["fired"]),
            "post_event_taylor_consumer_divergence_retained": item["step"] >= 3,
        })

    payload = {
        "schema": MODEL_ID,
        "classification": "MATCHED_REPLAY_PLUS_AUTONOMOUS_REDUCED_HISTORY",
        "spatial_identity": spatial_identity(),
        "event_contract": event_contract(replay_engine),
        "spatial_event_contract": spatial_event_contract(),
        "stress_contracts": {
            "virgin_reduced": virgin_contract,
            "active_reduced": active_contract,
            "after_restart_reduced": restart_contract,
            "archived_spatial": [
                archived_spatial_stress_contract(row) for row in rows
            ],
        },
        "recorded_operator_replay": replay,
        "recorded_replay_comparison": replay_comparison,
        "restart": {
            "exact": restart_exact,
            "checkpoint": str(checkpoint),
            "post_restart_runtime_receipt": runtime_receipt(
                restarted, stage="post-r3-restart"
            ),
        },
        "autonomous_reduced": {
            "external_history_source": "archived displacement-to-K loading history only",
            "future_spatial_state_injected": False,
            "recorded_tensor_drive_injected": False,
            "temperature_K": 900.0,
            "rows": autonomous_rows,
            "unload_reload_exercised": True,
            "terminal": _state_row(autonomous),
        },
        "comparison_limits": [
            "the archived spatial reference has three monotonic rows only",
            "unload/reload is executed by the autonomous reduced provider, not relabelled as spatial evidence",
            "step-three source population diverges under the restored named Taylor consumer and is retained",
        ],
        "passed": bool(
            restart_exact
            and [item["reduced_step"]["fired"] for item in replay]
                == [False, True, True]
            and all(item["relative_sigma_difference"] < 3.0e-4
                    for item in replay_comparison[:2])
            and autonomous_rows[-1]["stage"] == "reload_30um"
        ),
    }
    (root / "matched_campaign.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


__all__ = [
    "AVAILABLE_STOCHASTIC_EVENT_LAW_ID", "MODEL_ID", "REDUCED_EVENT_LAW_ID",
    "SPATIAL_EVENT_LAW_ID",
    "SPATIAL_EVIDENCE", "archived_spatial_stress_contract",
    "build_matched_reduced_engine", "event_contract", "matched_config",
    "run_recorded_operator_replay", "spatial_identity", "spatial_rows",
    "run_matched_campaign", "spatial_event_contract", "spatial_tensor_drives",
    "stress_contract",
]
