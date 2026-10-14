#!/usr/bin/env python3
"""Compare Taylor completion laws in the native operator on a full-PF state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_transport_exchange import joint_transport_exchange
from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.taylor_release import historical_legacy_implicit_1s
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState

SOURCE_ROOT = (
    ROOT / "integration_evidence" / "f1_native_taylor_peierls_qualification_2276745b"
)
SOURCE_CHECKPOINT = SOURCE_ROOT / "refinement_step46_checkpoint"
SOURCE_RUN = SOURCE_ROOT / "initial_to_step48"
B_M = 2.74e-10
T_K = 950.0
REFERENCE_INTERVAL_S = 8.4
TARGET_MAX_ACTIVITY = 1.0e6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _callable_sha(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _verified_generation() -> tuple[dict, Path]:
    manifest_path = SOURCE_CHECKPOINT / "run_state_checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    directory = SOURCE_CHECKPOINT / "run_state_generations" / manifest["generation"]
    for name, expected in manifest["files"].items():
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"source checkpoint hash mismatch for {name}")
    return manifest, directory


def _step46_stress() -> float:
    with (SOURCE_RUN / "steps_0950K.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if int(float(item["step"])) == 46)
    return float(row["sigma_tip_Pa"])


def _load_state() -> tuple[UnifiedMPZState, dict]:
    manifest, directory = _verified_generation()
    run_args = json.loads((SOURCE_RUN / "run_args.json").read_text())
    material_path = SOURCE_RUN / "selected_material_manifest_v10_2_22.csv"
    material = MaterialManifest.from_csv(material_path)
    with material_path.open(newline="") as handle:
        material_row = next(csv.DictReader(handle))
    state = UnifiedMPZState(
        material,
        MPZConfig(
            length_m=float(run_args["mpz_length_m"]),
            n_bins=int(run_args["mpz_n_bins"]),
            n_systems=int(run_args["mpz_n_systems"]),
            source_bin_count=int(run_args["mpz_source_bin_count"]),
            blunting_length_m=float(run_args["mpz_blunting_length_m"]),
            forest_density_floor_m2=float(material_row["rho_forest_floor_m2"]),
            jump_fraction=float(run_args["pt_jump_fraction"]),
            mobile_recovery_rate_s=float(run_args["mpz_mobile_recovery_rate_s"]),
            mobile_shield_fraction=float(run_args["mpz_mobile_shield_fraction"]),
            wake_length_m=float(run_args["wake_length_um"]) * 1.0e-6,
            wake_n_bins=int(run_args["wake_n_bins"]),
            wake_shielding=bool(run_args["wake_shielding"]),
        ),
    )
    kinetic = json.loads((directory / "kinetic.json").read_text())
    with np.load(directory / "state.npz", allow_pickle=False) as data:
        vector = data["kinetic_active_vector"].copy()
    signed_fields = {
        "mobile_positive", "mobile_negative", "retained_positive", "retained_negative",
        "accumulated_slip_positive", "accumulated_slip_negative",
        "wake_mobile_positive", "wake_mobile_negative",
        "wake_retained_positive", "wake_retained_negative",
        "wake_slip_positive", "wake_slip_negative",
    }
    for field in kinetic["fields"]:
        name = field["name"]
        if name not in signed_fields:
            continue
        start, stop = int(field["start"]), int(field["stop"])
        values = vector[start:stop].reshape(field["shape"])
        setattr(state, name, values.copy())
    state.mobile = state.mobile_positive + state.mobile_negative
    state.retained = state.retained_positive + state.retained_negative
    state.accumulated_slip = (
        state.accumulated_slip_positive + state.accumulated_slip_negative
    )
    return state, {
        "checkpoint_manifest": str((SOURCE_CHECKPOINT / "run_state_checkpoint.json").relative_to(ROOT)),
        "checkpoint_manifest_sha256": _sha(SOURCE_CHECKPOINT / "run_state_checkpoint.json"),
        "checkpoint_generation": manifest["generation"],
        "material_manifest": str(material_path.relative_to(ROOT)),
        "material_manifest_sha256": _sha(material_path),
        "run_args_sha256": _sha(SOURCE_RUN / "run_args.json"),
    }


def _result_payload(result) -> dict:
    return {
        "mobile_total": float(np.sum(result.mobile)),
        "retained_total": float(np.sum(result.retained)),
        "escaped": result.escaped,
        "recovered": result.recovered,
        "net_retained_gain": result.net_retained_gain,
        "net_retained_loss": result.net_retained_loss,
        "gross_trapping": result.gross_trapping,
        "gross_release": result.gross_release,
        "mobile_distance_population_m": result.mobile_distance_population_m,
        "conservation_error": result.conservation_error,
        "complete_numerical_correction": result.complete_numerical_correction,
        "stiff_solver_method": result.stiff_solver_method,
        "error_controlled_stiff_solve": result.error_controlled_stiff_solve,
    }


def build(output: Path) -> dict:
    if output.exists():
        raise SystemExit(f"refusing to overwrite consequence evidence: {output}")
    output.mkdir(parents=True)
    state, source = _load_state()
    stress_tip = _step46_stress()
    stress = state.local_stress_profile_Pa(stress_tip)
    rho = state.local_forest_density_m2(False)
    rates = state._transport_rates(stress, rho, T_K, B_M)
    historical_release = historical_legacy_implicit_1s(
        np.maximum(rates["taylor_single"], 0.0), rates["m"], rate_cap_s=1.0e12
    )
    hop = np.maximum(rates["velocity"], 0.0) / state.dx
    fixed_recovery = (
        max(float(state.cfg.mobile_recovery_rate_s), 0.0)
        + max(float(state.manifest.retained_recovery_rate_s), 0.0)
    )
    current_activity_rate = float(np.max(
        rates["encounter"] + rates["taylor"] + hop + fixed_recovery
    ))
    historical_activity_rate = float(np.max(
        rates["encounter"] + historical_release + hop + fixed_recovery
    ))
    maximum_activity_rate = max(current_activity_rate, historical_activity_rate)
    diagnostic_interval_s = min(
        REFERENCE_INTERVAL_S, TARGET_MAX_ACTIVITY / maximum_activity_rate
    )
    mobile = np.concatenate((state.mobile_positive, state.mobile_negative), axis=0)
    retained = np.concatenate((state.retained_positive, state.retained_negative), axis=0)
    common = {
        "velocity_m_s": np.maximum(rates["velocity"], 0.0),
        "dt_s": diagnostic_interval_s, "dx_m": state.dx,
        "mobile_recovery_rate_s": max(float(state.cfg.mobile_recovery_rate_s), 0.0),
        "retained_recovery_rate_s": max(
            float(state.manifest.retained_recovery_rate_s), 0.0
        ),
    }
    current = joint_transport_exchange(
        mobile, retained, rates["encounter"], rates["taylor"], **common
    )
    historical = joint_transport_exchange(
        mobile, retained, rates["encounter"], historical_release, **common
    )
    delta_mobile = current.mobile - historical.mobile
    delta_retained = current.retained - historical.retained
    initial_total = float(np.sum(mobile) + np.sum(retained))
    payload = {
        "schema": "taylor-completion-native-operator-consequence/1",
        "status": "PASS",
        "source_head": _git_head(),
        "source": source,
        "state": {
            "classification": "accepted full-PF active step-46 spatial checkpoint",
            "temperature_K": T_K,
            "reference_native_interval_s": REFERENCE_INTERVAL_S,
            "diagnostic_interval_s": diagnostic_interval_s,
            "diagnostic_interval_policy": (
                "min(reference native interval, target activity / maximum exact "
                "current-or-historical generator activity rate)"
            ),
            "target_max_activity": TARGET_MAX_ACTIVITY,
            "current_max_activity": current_activity_rate * diagnostic_interval_s,
            "historical_max_activity": historical_activity_rate * diagnostic_interval_s,
            "tip_stress_Pa": stress_tip, "bins": state.n_bins,
            "systems": state.n_systems, "dx_m": state.dx,
            "initial_mobile_total": float(np.sum(mobile)),
            "initial_retained_total": float(np.sum(retained)),
            "initial_population_total": initial_total,
            "forest_density_min_m2": float(np.min(rho)),
            "forest_density_max_m2": float(np.max(rho)),
            "local_stress_min_Pa": float(np.min(stress)),
            "local_stress_max_Pa": float(np.max(stress)),
        },
        "identical_inputs": {
            "native_operator": "arrhenius_fracture.coupled_transport_exchange.joint_transport_exchange",
            "native_operator_body_sha256": _callable_sha(joint_transport_exchange),
            "elementary_signed_rates": "current exact UnifiedMPZState._transport_rates",
            "density_order": "current unshifted exact order in both arms",
            "phi_policy": "current uncapped exact phi in both arms",
            "jump_and_velocity": "current native spatial fields in both arms",
            "encounter_field": "current native field in both arms",
            "mobile_source_rate": None,
            "source_exclusion_reason": "isolate completion effect on already-active accepted populations",
        },
        "changed_input_only": {
            "current": "arrhenius_fracture.taylor_release.completion_rate",
            "current_body_sha256": _callable_sha(
                UnifiedMPZState._transport_rates.__globals__["completion_rate"]
            ),
            "historical": "arrhenius_fracture.taylor_release.historical_legacy_implicit_1s",
            "historical_body_sha256": _callable_sha(historical_legacy_implicit_1s),
            "historical_implicit_window_s": 1.0,
            "historical_rate_cap_s": 1.0e12,
        },
        "rate_fields": {
            "current_completion_min_s": float(np.min(rates["taylor"])),
            "current_completion_max_s": float(np.max(rates["taylor"])),
            "historical_completion_min_s": float(np.min(historical_release)),
            "historical_completion_max_s": float(np.max(historical_release)),
            "current_to_historical_max_ratio": float(
                np.max(rates["taylor"] / np.maximum(historical_release, 1.0e-300))
            ),
            "order_min": float(np.min(rates["m"])),
            "order_max": float(np.max(rates["m"])),
            "phi_min": float(np.min(rates["jump"] / B_M)),
            "phi_max": float(np.max(rates["jump"] / B_M)),
            "velocity_min_m_s": float(np.min(rates["velocity"])),
            "velocity_max_m_s": float(np.max(rates["velocity"])),
        },
        "current_terminal": _result_payload(current),
        "historical_terminal": _result_payload(historical),
        "active_consequence": {
            "mobile_l1_difference": float(np.sum(np.abs(delta_mobile))),
            "mobile_max_abs_difference": float(np.max(np.abs(delta_mobile))),
            "retained_l1_difference": float(np.sum(np.abs(delta_retained))),
            "retained_max_abs_difference": float(np.max(np.abs(delta_retained))),
            "escaped_difference_current_minus_historical": current.escaped - historical.escaped,
            "gross_release_difference_current_minus_historical": (
                current.gross_release - historical.gross_release
            ),
        },
        "scope": (
            "one activity-bounded frozen-coefficient native coupled exchange/transport "
            "micro-interval on an accepted original full-PF spatial state; the bound "
            "avoids the native long-horizon BDF branch whose historical arm accumulated "
            "unbounded adaptive history, without changing the operator or rates; no "
            "mechanics re-equilibration, source injection, production selection, or "
            "effect-size claim"
        ),
        "failed_predecessors": [
            "taylor_completion_native_consequence_d4cdd51e: parent-only overlay did not cross worker boundary",
            "taylor_completion_native_consequence_e42ca85f: worker-bound full re-equilibration stopped after active 9.4 GiB repeated global factorization; receipt also overwritten by child process",
            "uncommitted full-state 8.4 s native-operator validation: current arm completed but historical arm was stopped after 16 minutes and 3.5 GiB while repeatedly factoring in solve_ivp BDF",
        ],
    }
    consequence = payload["active_consequence"]
    if not all(np.isfinite(value) for value in consequence.values()):
        payload["status"] = "FAIL_NONFINITE_CONSEQUENCE"
    elif consequence["mobile_l1_difference"] <= 0.0 and consequence["retained_l1_difference"] <= 0.0:
        payload["status"] = "FAIL_NO_ACTIVE_CONSEQUENCE"
    if abs(current.conservation_error) > 1.0e-12 * max(initial_total, 1.0):
        payload["status"] = "FAIL_CURRENT_CONSERVATION"
    if abs(historical.conservation_error) > 1.0e-12 * max(initial_total, 1.0):
        payload["status"] = "FAIL_HISTORICAL_CONSERVATION"
    if current.stiff_solver_method != "affine_bounded_exponential":
        payload["status"] = "FAIL_CURRENT_SOLVER_PATH"
    if historical.stiff_solver_method != "affine_bounded_exponential":
        payload["status"] = "FAIL_HISTORICAL_SOLVER_PATH"
    correction_limit = 1.0e-8 * max(initial_total, 1.0e-30)
    if current.complete_numerical_correction > correction_limit:
        payload["status"] = "FAIL_CURRENT_NUMERICAL_CORRECTION"
    if historical.complete_numerical_correction > correction_limit:
        payload["status"] = "FAIL_HISTORICAL_NUMERICAL_CORRECTION"
    path = output / "native_operator_consequence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "sha256_manifest.json").write_text(json.dumps({
        path.name: _sha(path)
    }, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "PASS":
        raise SystemExit(payload["status"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.output.resolve())
    print(json.dumps({
        "status": payload["status"],
        "mobile_l1_difference": payload["active_consequence"]["mobile_l1_difference"],
        "retained_l1_difference": payload["active_consequence"]["retained_l1_difference"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
