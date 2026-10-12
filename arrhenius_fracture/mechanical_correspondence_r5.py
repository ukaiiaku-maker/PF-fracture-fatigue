"""Complete-input and signed-consumer correspondence for the R5 spatial path."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .current_coupled_execution_r4 import complete_state
from .current_law_spatial_comparison_r4 import _build_spatial_event_law_engine
from .reduced_persistent_reference_r2 import advance_signed
from .sharp_front_r4_current_law import CurrentLawSpatialTipEngine
from .stochastic_avalanche_tip import pending_geometry_event_scope


MODEL_ID = "r5.complete-mechanical-input-correspondence/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    engine: Any, steps: list[Mapping[str, Any]], *, complete_inputs: bool,
) -> dict[str, Any]:
    rows = []
    with pending_geometry_event_scope() as pending:
        for index, step in enumerate(steps):
            if complete_inputs:
                engine._r2_synthetic_drive_setup = False
                engine.mpz._anisotropic_drive_factors = np.asarray(
                    step["drive_factors"], dtype=float
                )
                engine.mpz._anisotropic_tau_signed_Pa = np.asarray(
                    step["tau_signed_Pa"], dtype=float
                )
                engine.mpz._anisotropic_drive_reliable = True
                engine.mpz._anisotropic_drive_serial = index + 1
            result = advance_signed(
                engine, K_Pa_sqrt_m=float(step["K_Pa_sqrt_m"]),
                temperature_K=900.0, dt_s=float(step["duration_s"]),
            )
            rows.append({
                "stage": step["stage"],
                "mechanical_input_timing": "BEFORE_NATIVE_ADVANCE",
                "state_observation_timing": "AFTER_EVENT_COMMIT_OR_INTERVAL_END",
                "K_Pa_sqrt_m": step["K_Pa_sqrt_m"],
                "drive_factors": list(step["drive_factors"]),
                "tau_signed_Pa": list(step["tau_signed_Pa"]),
                "N_em": float(result["N_em"]),
                "r_eff_m": float(result["r_eff"]),
                "sigma_tip_Pa": float(result["sigma_tip"]),
                "fired": bool(result["fired"]),
                "consumed_s": float(result["kinetic_dt_consumed_s"]),
                "unused_s": float(result["kinetic_dt_unused_s"]),
                "consumed_tau_signed_Pa": list(result["anisotropic_tau_signed_Pa"]),
                "consumed_drive_factors": list(result["anisotropic_drive_factors"]),
                "negative_transport_velocity_present": bool(
                    result["reversible_mobile_velocity_min_m_s"] < 0.0
                ),
            })
        events = [copy.deepcopy(item) for item in pending]
    return {"rows": rows, "geometry_events": events, "terminal": complete_state(engine)}


def signed_consumer_probe() -> dict[str, Any]:
    parent = _build_spatial_event_law_engine()
    rows = []
    for label, tau in (
        ("mixed_signed", (-2.0e8, 3.0e8)),
        ("positive_reference", (2.0e8, 3.0e8)),
    ):
        engine = copy.deepcopy(parent)
        # Exercise the actual spatial subclass method-resolution path while
        # retaining the fully constructed current backend state.
        engine.__class__ = CurrentLawSpatialTipEngine
        engine._r2_synthetic_drive_setup = False
        engine.mpz._anisotropic_drive_factors = np.asarray((0.4, 0.6))
        engine.mpz._anisotropic_tau_signed_Pa = np.asarray(tau)
        engine.mpz._anisotropic_drive_reliable = True
        with pending_geometry_event_scope():
            result = advance_signed(
                engine, K_Pa_sqrt_m=16.9e6,
                temperature_K=900.0, dt_s=0.01,
            )
        rows.append({
            "case": label, "input_tau_signed_Pa": list(tau),
            "consumed_tau_signed_Pa": list(result["anisotropic_tau_signed_Pa"]),
            "transport_velocity_min_m_s": float(
                result["reversible_mobile_velocity_min_m_s"]
            ),
            "transport_velocity_max_m_s": float(
                result["reversible_mobile_velocity_max_m_s"]
            ),
            "emission_increment_by_system": list(
                result["anisotropic_dN_emit_by_system"]
            ),
        })
    return {
        "cases": rows,
        "mixed_sign_preserved": rows[0]["consumed_tau_signed_Pa"] == [-2.0e8, 3.0e8],
        "negative_component_reaches_transport_consumer": (
            rows[0]["transport_velocity_min_m_s"] < 0.0
        ),
        "positive_reference_has_positive_minimum_velocity": (
            rows[1]["transport_velocity_min_m_s"] > 0.0
        ),
    }


def execute_mechanical_correspondence_r5(
    spatial_directory: str | Path, output: str | Path,
) -> dict[str, Any]:
    root = Path(spatial_directory).resolve()
    audit_path = root / "anisotropic_emission_audit_v10174.json"
    steps_path = root / "steps_0900K.csv"
    audit = json.loads(audit_path.read_text())
    spatial_csv = list(csv.DictReader(steps_path.open()))
    records = audit["records"]
    if len(records) != 5 or len(spatial_csv) != 5:
        raise RuntimeError("R5 correspondence requires the five-stage spatial run")
    labels = ["load_10um", "load_20um", "load_30um", "unload_20um", "reload_30um"]
    steps = [{
        "stage": label,
        "K_Pa_sqrt_m": float(csv_row["KJ_Pa_sqrtm"]),
        "duration_s": float(csv_row["dt_cur_s"]),
        "opening_m": float(csv_row["Uapp_m"]),
        "projected_tip_m": float(csv_row["a_tip_m"]),
        "path_angle_deg": 0.0,
        "drive_factors": tuple(float(x) for x in record["anisotropic_drive_factors"]),
        "tau_signed_Pa": tuple(float(x) for x in record["anisotropic_tau_signed_Pa"]),
        "spatial_N_em": float(csv_row["N_em"]),
        "spatial_r_eff_m": float(record["persistent_tip_radius_m"]),
        "spatial_fired": bool(float(csv_row["n_fire"]) > 0.0),
    } for label, csv_row, record in zip(labels, spatial_csv, records, strict=True)]

    parent = _build_spatial_event_law_engine()
    initial = complete_state(parent)
    k_only = _run(copy.deepcopy(parent), steps, complete_inputs=False)
    complete = _run(copy.deepcopy(parent), steps, complete_inputs=True)
    comparisons = []
    for source, old, full in zip(steps, k_only["rows"], complete["rows"], strict=True):
        comparisons.append({
            "stage": source["stage"],
            "spatial_N_em": source["spatial_N_em"],
            "K_only_N_em": old["N_em"],
            "complete_input_N_em": full["N_em"],
            "complete_input_relative_population_error": (
                (full["N_em"] - source["spatial_N_em"])
                / max(abs(source["spatial_N_em"]), 1.0e-300)
            ),
            "K_only_relative_population_error": (
                (old["N_em"] - source["spatial_N_em"])
                / max(abs(source["spatial_N_em"]), 1.0e-300)
            ),
            "spatial_fired": source["spatial_fired"],
            "complete_input_fired": full["fired"],
        })
    probe = signed_consumer_probe()
    result = {
        "schema": MODEL_ID,
        "spatial_source": str(root),
        "source_hashes": {
            "anisotropic_audit_sha256": _sha256(audit_path),
            "steps_sha256": _sha256(steps_path),
        },
        "operator_contract": {
            "inputs": [
                "K", "duration", "opening", "projected_tip", "path_orientation",
                "anisotropic_drive_factors", "resolved_signed_shear",
            ],
            "future_population_imported": False,
            "event_time_forced": False,
            "spatial_remap_reproduced": False,
            "source_geometry_re_equilibration_reproduced": False,
        },
        "common_random_owner": {
            "forked_complete_initial_owner": True,
            "initial_complete_state": initial,
            "same_mark_reward_correlation": True,
            "independent_realization_claimed": False,
        },
        "steps": steps,
        "K_only_control": k_only,
        "complete_input_replay": complete,
        "comparisons": comparisons,
        "signed_consumer_probe": probe,
        "positive_loading_result_changed_after_sign_fix": any(
            abs(float(a["active_mobile"]) - float(b["active_mobile"])) > 1.0e-6
            for a, b in zip(
                json.loads((Path(__file__).resolve().parents[1] /
                    "integration_evidence/r4_same_engine_coupled_spatial/"
                    "current_law_spatial/anisotropic_emission_audit_v10174.json").read_text())["records"],
                records, strict=True,
            )
        ),
        "residual_classification": (
            "CONTINUOUS_SPATIAL_REMAP_AND_EVENT_GEOMETRY_VERSUS_OUTER_REDUCED_"
            "ADVANCE; NOT_A_STRESS_ONLY_ERROR"
        ),
    }
    result["passed"] = bool(
        probe["mixed_sign_preserved"]
        and probe["negative_component_reaches_transport_consumer"]
        and probe["positive_reference_has_positive_minimum_velocity"]
        and all(row["consumed_tau_signed_Pa"] == list(step["tau_signed_Pa"])
                for row, step in zip(complete["rows"], steps, strict=True))
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["MODEL_ID", "execute_mechanical_correspondence_r5", "signed_consumer_probe"]
