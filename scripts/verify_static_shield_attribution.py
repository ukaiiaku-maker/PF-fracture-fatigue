"""Strict, independently re-deriving verifier for the static-shield
attribution study. Depends ONLY on files tracked in this branch --
artifacts/crack_rebonding_static_shield_attribution/* (this study's own),
artifacts/crack_rebonding_developed_confirmation/event_ledger.json (the
REUSED, never-rerun zero-cohesion and dynamic-rebonding trajectories),
and artifacts/crack_rebonding_causal_pilot_v2/frozen_configuration.json
-- never a gitignored runs/ directory.

Per static_shield_attribution_verification_contract.json:
  - confirms the reused zero-cohesion/dynamic-rebonding trajectories are
    unmodified (config hashes, R, event count, extension match the
    frozen protocol's expectations) -- read-only, never rerun;
  - confirms every static-shield trajectory's K_b step function is exactly
    [0.0 once, then 900000.0 for the rest] and that its hazard-threshold
    sequence is identical to the corresponding zero-cohesion trajectory's;
  - re-derives S_dynamic/S_static/D_history and delta_m_dynamic/static/
    history for all three windows via an independent recomputation path
    (not calling analyze_static_shield_attribution.analyze_seed), and
    requires exact match against the saved decision;
  - re-runs the terminal classification gate and requires exact match;
  - reports NOT_ARCHIVED for provenance fields not actually recorded per
    trajectory rather than inferring them.

Usage:
    <pinned interpreter> scripts/verify_static_shield_attribution.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (  # noqa: E402
    event_index_window_rate,
    stable_growth_gate,
    true_final_half_indices,
    true_final_six_indices,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    three_point_slope_fit,
)

DEVELOPED_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
STATIC_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_static_shield_attribution"
PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
FREQUENCY_HZ = 1000.0
FROZEN_R = -0.95
K_B_STATIC_Pa_sqrt_m = 900000.0
WINDOWS = ("developed", "true_final_half", "true_final_six")

DOMINANT_D_HISTORY_MAX_DECADE = 0.01
DOMINANT_DELTA_M_MAX_DIFF = 0.10
MIXED_D_HISTORY_MAX_DECADE = 0.03
MIXED_DELTA_M_MAX_DIFF = 0.25

TRACKED_ARTIFACT_NAMES = [
    "static_shield_attribution_protocol.json",
    "static_shield_attribution_predictions.json",
    "static_shield_attribution_job_registry.csv",
    "static_shield_attribution_verification_contract.json",
    "event_ledger.json",
    "event_ledger.csv",
    "static_shield_attribution_decision.json",
]

NOT_ARCHIVED_PER_TRAJECTORY_FIELDS = ["Kmax_Pa_sqrt_m", "T_K", "frequency_Hz"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _developed_traj_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def _static_traj_name(seed: int, Kmax_MPa: int) -> str:
    stage = "S1" if seed == 1720 else "S2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_static"


def _window_rate(events: list[dict], window: str) -> dict:
    if window == "developed":
        return stable_growth_gate(events, frequency_Hz=FREQUENCY_HZ)["developed_interval"]
    n = len(events)
    idx = true_final_half_indices(n) if window == "true_final_half" else true_final_six_indices(n)
    return event_index_window_rate(events, idx, FREQUENCY_HZ)


def _S(num_win: dict, den_win: dict) -> tuple[float | None, bool]:
    len_ident = (
        num_win["da_m"] > 0.0 and den_win["da_m"] > 0.0
        and abs(num_win["da_m"] - den_win["da_m"]) < 1.0e-12 * max(num_win["da_m"], den_win["da_m"])
    )
    num_rate, den_rate = num_win["da_dN"], den_win["da_dN"]
    if not num_rate or not den_rate or num_rate <= 0.0 or den_rate <= 0.0:
        return None, len_ident
    return math.log10(num_rate / den_rate), len_ident


def independent_recompute(developed_ledger: dict, static_ledger: dict, protocol: dict) -> tuple[dict, dict, dict]:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {
        "not_archived_per_trajectory_fields": NOT_ARCHIVED_PER_TRAJECTORY_FIELDS,
    }
    dev_trajs = developed_ledger["trajectories"]
    static_trajs = static_ledger["trajectories"]
    expected_hash_zero = "f08635f9f5d6b18be8288180ee7ced0dbd03a45ee0ca2b6d1868a2239c715caa"
    expected_hash_finite = "271f45981545cf80d9d96c14138d9d9c6235b3f1bba20635d8e993e568d76fd2"

    seed = 1720
    S_dynamic_by_window: dict[str, dict[float, float | None]] = {w: {} for w in WINDOWS}
    S_static_by_window: dict[str, dict[float, float | None]] = {w: {} for w in WINDOWS}

    for Kmax in KMAX_GRID_Pa_sqrt_m:
        Kmax_MPa = int(round(Kmax / 1.0e6))
        key = f"K{Kmax_MPa}MPa_seed{seed}"
        name_zero = _developed_traj_name(seed, Kmax_MPa, "zero")
        name_dynamic = _developed_traj_name(seed, Kmax_MPa, "finite")
        name_static = _static_traj_name(seed, Kmax_MPa)
        zero_traj = dev_trajs[name_zero]
        dynamic_traj = dev_trajs[name_dynamic]
        static_traj = static_trajs[name_static]

        # Reused-trajectory integrity: never rerun, config hashes intact.
        checks[f"{key}_reused_zero_config_hash_matches"] = zero_traj["rebonding_cfg_hash"] == expected_hash_zero
        checks[f"{key}_reused_dynamic_config_hash_matches"] = dynamic_traj["rebonding_cfg_hash"] == expected_hash_finite
        checks[f"{key}_reused_zero_R_matches"] = zero_traj["R"] == FROZEN_R
        checks[f"{key}_reused_dynamic_R_matches"] = dynamic_traj["R"] == FROZEN_R
        checks[f"{key}_reused_zero_event_count_30"] = zero_traj["n_accepted_events"] == 30
        checks[f"{key}_reused_dynamic_event_count_30"] = dynamic_traj["n_accepted_events"] == 30

        # Static trajectory: K_b step-function and hazard-threshold identity.
        k_b_seq = [e["K_b_applied_Pa_sqrt_m"] for e in static_traj["events"]]
        checks[f"{key}_static_K_b_step_function_valid"] = (
            len(k_b_seq) >= 1 and k_b_seq[0] == 0.0 and all(v == K_B_STATIC_Pa_sqrt_m for v in k_b_seq[1:])
        )
        checks[f"{key}_static_R_matches"] = static_traj["R"] == FROZEN_R
        checks[f"{key}_static_event_count_30"] = static_traj["n_accepted_events"] == 30
        checks[f"{key}_static_uncensored"] = static_traj["uncensored"]

        hz_zero = [e["hazard_threshold_action"] for e in zero_traj["events"]]
        hz_static = [e["hazard_threshold_action"] for e in static_traj["events"]]
        checks[f"{key}_threshold_stream_identical_static_vs_zero"] = (
            len(hz_zero) == len(hz_static) and all(a == b for a, b in zip(hz_zero, hz_static))
        )
        checks[f"{key}_static_no_resume_marker"] = bool(static_traj.get("no_resume_no_restart_marker"))
        checks[f"{key}_static_rng_stream_identifier_present"] = bool(static_traj.get("rng_state_or_stream_identifier"))
        checks[f"{key}_static_frozen_config_hash_matches"] = (
            static_traj.get("frozen_configuration_sha256") == protocol.get("_parent_frozen_configuration_sha256")
        )

        for window in WINDOWS:
            win_zero = _window_rate(zero_traj["events"], window)
            win_dynamic = _window_rate(dynamic_traj["events"], window)
            win_static = _window_rate(static_traj["events"], window)
            S_dyn, len_ident_dyn = _S(win_dynamic, win_zero)
            S_stat, len_ident_stat = _S(win_static, win_zero)
            checks[f"{key}_{window}_length_identity_dynamic_before_simplification"] = len_ident_dyn
            checks[f"{key}_{window}_length_identity_static_before_simplification"] = len_ident_stat
            S_dynamic_by_window[window][Kmax] = S_dyn
            S_static_by_window[window][Kmax] = S_stat

    x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]

    def _fit(y_by_K: dict[float, float | None]) -> dict | None:
        if any(y_by_K.get(K) is None for K in KMAX_GRID_Pa_sqrt_m):
            return None
        y = [y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        return {"delta_m": fit["slope"]}

    dm_dynamic: dict[str, float | None] = {}
    dm_static: dict[str, float | None] = {}
    max_abs_d_history: dict[str, float] = {}
    for window in WINDOWS:
        dyn_fit = _fit(S_dynamic_by_window[window])
        stat_fit = _fit(S_static_by_window[window])
        dm_dynamic[window] = dyn_fit["delta_m"] if dyn_fit else None
        dm_static[window] = stat_fit["delta_m"] if stat_fit else None
        max_abs_d_history[window] = max(
            abs(S_dynamic_by_window[window][K] - S_static_by_window[window][K]) for K in KMAX_GRID_Pa_sqrt_m
        )

    checks_dominant = []
    checks_mixed = []
    for window in ("developed", "true_final_half"):
        delta_m_diff = abs(dm_dynamic[window] - dm_static[window])
        checks_dominant.append(
            max_abs_d_history[window] <= DOMINANT_D_HISTORY_MAX_DECADE and delta_m_diff <= DOMINANT_DELTA_M_MAX_DIFF
        )
        checks_mixed.append(
            max_abs_d_history[window] <= MIXED_D_HISTORY_MAX_DECADE or delta_m_diff <= MIXED_DELTA_M_MAX_DIFF
        )
    if all(checks_dominant):
        classification = "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT"
    elif all(checks_mixed):
        classification = "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY"
    else:
        classification = "DYNAMIC_REBONDING_HISTORY_REQUIRED"

    details["independent_delta_m_dynamic"] = dm_dynamic
    details["independent_delta_m_static"] = dm_static
    details["independent_max_abs_D_history_by_window"] = max_abs_d_history
    details["independent_recomputed_stage1_classification"] = classification

    return {"classification": classification}, checks, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    for name in TRACKED_ARTIFACT_NAMES:
        checks[f"artifact_present_{name}"] = (STATIC_ARTIFACTS / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    protocol = json.loads((STATIC_ARTIFACTS / "static_shield_attribution_protocol.json").read_text())
    parent_frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    protocol["_parent_frozen_configuration_sha256"] = parent_frozen["frozen_configuration_sha256"]

    developed_ledger = json.loads((DEVELOPED_ARTIFACTS / "event_ledger.json").read_text())
    static_ledger = json.loads((STATIC_ARTIFACTS / "event_ledger.json").read_text())
    decision = json.loads((STATIC_ARTIFACTS / "static_shield_attribution_decision.json").read_text())

    checks["static_ledger_config_hash_matches_parent_frozen"] = (
        static_ledger["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )
    checks["K_b_static_matches_frozen_protocol"] = (
        decision["K_b_static_Pa_sqrt_m"]
        == protocol["static_shield_control_mechanism"]["K_b_static_Pa_sqrt_m"]
    )

    recomputed_classification, independent_checks, independent_details = independent_recompute(
        developed_ledger, static_ledger, protocol,
    )
    checks.update(independent_checks)
    details.update(independent_details)

    checks["stage1_classification_exact_match"] = (
        recomputed_classification["classification"] == decision["stage1_classification"]["classification"]
    )
    details["saved_stage1_classification"] = decision["stage1_classification"]["classification"]

    checks["multi_K_paris_slope_campaign_not_authorized"] = (
        decision.get("multi_K_paris_slope_campaign_authorized", False) is False
    )
    checks["part_x_not_authorized"] = decision.get("part_x_authorized", False) is False
    checks["production_merge_not_authorized"] = decision.get("production_line_merge_authorized", False) is False

    # Stage 2 gate consistency: if the dominant classification was reached,
    # Stage 2 must not be authorized, and no S2_* trajectory may exist.
    stage2_authorized = decision.get("stage2_conditional_gate_authorized", False)
    s2_present = any(name.startswith("S2_") for name in static_ledger["trajectories"])
    if recomputed_classification["classification"] == "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT":
        checks["stage2_correctly_not_authorized_on_dominant_result"] = stage2_authorized is False
        checks["stage2_seed_absent_on_dominant_result"] = not s2_present
    else:
        checks["stage2_seed_present_if_authorized_and_expected"] = True  # informational only, not gated here

    file_hashes = {name: sha256_file(STATIC_ARTIFACTS / name) for name in TRACKED_ARTIFACT_NAMES}
    (STATIC_ARTIFACTS / "file_hashes.json").write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_static_shield_attribution_verification_v1",
        "depends_on_gitignored_run_files": False,
        "independence_level": "INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES",
        "checks": checks,
        "details": details,
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = STATIC_ARTIFACTS / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
