"""Strict verifier for the exposure-unconditioned slope-exposure
continuation (v2, hardened per review). Depends ONLY on tracked artifacts
(this worktree's own, the minimal slope screen's, and the parent causal
pilot's -- all git-tracked) -- never on any gitignored runs/... file.

v2 adds, per review:
  - reruns classify_slope_effect_v2 and requires EXACT equality with the
    saved classification_analysis (not just membership in an allowed set);
  - requires accepted_lengths_identical True for all six zero/finite pairs;
  - requires the hazard_threshold_action sequence to match EXACTLY, event
    index by event index, between each zero/finite pair (the common-
    random-numbers design this whole campaign relies on);
  - requires every admitted event's bulk_action to have been certified
    (all_bulk_action_qualified) for all six pairs;
  - re-confirms config/material hashes for the reused Kmax=18 point and
    the one new completion trajectory.

Usage:
    <pinned interpreter> scripts/verify_v2_slope_exposure_continuation.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import (  # noqa: E402
    build_a_native_engine,
    load_a_native_provenance,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    three_point_slope_fit,
)
from arrhenius_fracture.crack_rebonding_slope_exposure_continuation_v10230 import (  # noqa: E402
    classify_slope_effect_v2,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
SLOPE_SCREEN_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"
CONTINUATION_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_slope_exposure_continuation"

CONTINUATION_TRACKED_ARTIFACT_NAMES = [
    "partial_analysis.json",
    "completion_record.json",
    "event_ledger.json",
    "event_ledger.csv",
    "slope_exposure_continuation_decision.json",
]
SEEDS = (1720, 1001723)
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(analyze) -> dict[tuple[float, int], tuple[dict, dict]]:
    parent_1720 = json.loads((PARENT_PILOT_ARTIFACTS / "event_ledger.json").read_text())
    parent_1001723 = json.loads((PARENT_PILOT_ARTIFACTS / "second_seed_event_ledger.json").read_text())
    screen_ledger = json.loads((SLOPE_SCREEN_ARTIFACTS / "event_ledger.json").read_text())
    continuation_ledger = json.loads((CONTINUATION_ARTIFACTS / "event_ledger.json").read_text())
    completion_finite = continuation_ledger["trajectories"][
        "C3R_K21MPa_seed1001723_EXPOSURE_UNCONDITIONED"
    ]
    completion_zero = screen_ledger["trajectories"]["C2R_K21MPa_seed1001723"]
    return {
        (15.0e6, 1720): (screen_ledger["trajectories"]["C2R_K15MPa_seed1720"],
                          screen_ledger["trajectories"]["C3R_K15MPa_seed1720"]),
        (15.0e6, 1001723): (screen_ledger["trajectories"]["C2R_K15MPa_seed1001723"],
                             screen_ledger["trajectories"]["C3R_K15MPa_seed1001723"]),
        (18.0e6, 1720): (parent_1720["trajectories"]["C2R"], parent_1720["trajectories"]["C3R"]),
        (18.0e6, 1001723): (parent_1001723["trajectories"]["S2_C2R"], parent_1001723["trajectories"]["S2_C3R"]),
        (21.0e6, 1720): (screen_ledger["trajectories"]["C2R_K21MPa_seed1720"],
                          screen_ledger["trajectories"]["C3R_K21MPa_seed1720"]),
        (21.0e6, 1001723): (completion_zero, completion_finite),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    for name in CONTINUATION_TRACKED_ARTIFACT_NAMES:
        checks[f"artifact_present_{name}"] = (CONTINUATION_ARTIFACTS / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    parent_frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    completion_record = json.loads((CONTINUATION_ARTIFACTS / "completion_record.json").read_text())
    checks["completion_matches_parent_frozen_config"] = (
        completion_record["parent_frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )
    checks["completion_config_hash_matches"] = (
        completion_record["config_hash_reused"] == parent_frozen["config_hashes"]["RB2_reversible_finite"]
    )
    checks["completion_is_exposure_unconditioned_labeled"] = (
        completion_record["classification"] == "EXPOSURE_UNCONDITIONED_COMPLETION"
    )
    checks["completion_Kmax_is_21"] = completion_record["Kmax_Pa_sqrt_m"] == 21.0e6
    checks["completion_seed_is_1001723"] = completion_record["seed"] == 1001723

    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    r_eff = max(bare_engine.r_eff(), 1.0e-9)
    provenance = load_a_native_provenance()
    recomputed_frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=r_eff,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    checks["parent_frozen_configuration_reproducible"] = (
        recomputed_frozen["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )

    analyze = importlib.import_module("analyze_v2_slope_exposure_continuation")
    pairs = _pairs(analyze)

    # Hard checks, for EVERY one of the six zero/finite pairs: accepted
    # event-length identity, hazard-threshold sequence identity (event
    # ordinal by event ordinal -- the common-random-numbers design), and
    # certified bulk-action status on every admitted event.
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
        zero_events = {e["event_index"]: e for e in zero_res["events"]}
        finite_events = {e["event_index"]: e for e in finite_res["events"]}
        common_idx = sorted(set(zero_events) & set(finite_events))

        checks[f"{key}_accepted_lengths_identical"] = all(
            abs(zero_events[i]["accepted_length_m"] - finite_events[i]["accepted_length_m"]) < 1.0e-15
            for i in common_idx
        )
        checks[f"{key}_hazard_threshold_sequence_identical"] = all(
            zero_events[i]["hazard_threshold_action"] == finite_events[i]["hazard_threshold_action"]
            for i in common_idx
        )
        checks[f"{key}_event_ordinals_match"] = (
            set(zero_events) == set(finite_events) == set(range(zero_res["n_accepted_events"]))
        )
        checks[f"{key}_zero_all_bulk_action_qualified"] = all(
            e["all_bulk_action_qualified"] for e in zero_res["events"]
        )
        checks[f"{key}_finite_all_bulk_action_qualified"] = all(
            e["all_bulk_action_qualified"] for e in finite_res["events"]
        )

    # Re-derive S_h (all-window) at all six points and the per-seed
    # 3-point slope fit, and compare against the saved decision.
    S_h_by_Kmax: dict[float, dict[int, float]] = {K: {} for K in KMAX_GRID_Pa_sqrt_m}
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        rs = analyze._rate_shift_windowed(zero_res, finite_res, [0, 1, 2, 3, 4, 5, 6])
        S_h_by_Kmax[Kmax][seed] = rs["S_h_decade"]
        key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
        checks[f"{key}_accepted_lengths_identical_windowed"] = rs["accepted_lengths_identical"]

    saved_decision = json.loads(
        (CONTINUATION_ARTIFACTS / "slope_exposure_continuation_decision.json").read_text()
    )
    for Kmax in KMAX_GRID_Pa_sqrt_m:
        for seed in SEEDS:
            key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
            saved_S_h = saved_decision["per_pair"][key]["windows"]["all"]["S_h_decade"]
            checks[f"S_h_{key}_reproducible"] = abs(S_h_by_Kmax[Kmax][seed] - saved_S_h) < 1.0e-9

    delta_m_by_seed = {}
    for seed in SEEDS:
        x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]
        y = [S_h_by_Kmax[K][seed] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        delta_m_by_seed[seed] = fit["slope"]
        checks[f"delta_m_seed_{seed}_reproducible"] = (
            abs(fit["slope"] - saved_decision["per_seed_by_window"][str(seed)]["all"]["delta_m_least_squares"])
            < 1.0e-9
        )

    # STRICT classification re-derivation: exact equality with the saved
    # classification_analysis dict, not membership in an allowed set.
    exposure_at_Kmax_hi = {
        seed: saved_decision["per_pair"][f"K21MPa_seed{seed}"]["finite_cohesion_diagnostics"]
        for seed in SEEDS
    }
    recomputed_classification = classify_slope_effect_v2(
        S_h_by_Kmax=S_h_by_Kmax, delta_m_by_seed=delta_m_by_seed,
        exposure_by_seed_at_Kmax=exposure_at_Kmax_hi,
    )
    saved_classification_analysis = saved_decision["classification_analysis_all_window"]
    # Round-trip the freshly recomputed dict through JSON before comparing,
    # so both sides go through the same string-key/float-repr normalization
    # (dict keys here are ints in-process but strings once loaded back from
    # the saved JSON -- this is a representation artifact, not a value
    # difference, and comparing raw Python dicts without normalizing would
    # produce a false mismatch on every seed-keyed sub-dict).
    recomputed_normalized = json.loads(json.dumps(recomputed_classification, sort_keys=True))
    saved_normalized = json.loads(json.dumps(saved_classification_analysis, sort_keys=True))
    checks["classification_analysis_exact_match"] = recomputed_normalized == saved_normalized
    checks["overall_classification_exact_match"] = (
        recomputed_classification["classification"] == saved_decision["overall_classification"]
    )
    details["recomputed_classification"] = recomputed_classification["classification"]
    details["saved_classification"] = saved_decision["overall_classification"]

    checks["multi_K_paris_slope_campaign_not_authorized"] = (
        saved_decision.get("multi_K_paris_slope_campaign_authorized", False) is False
    )
    checks["part_x_not_authorized"] = saved_decision.get("part_x_authorized", False) is False
    checks["production_merge_not_authorized"] = (
        saved_decision.get("production_line_merge_authorized", False) is False
    )

    file_hashes = {name: sha256_file(CONTINUATION_ARTIFACTS / name) for name in CONTINUATION_TRACKED_ARTIFACT_NAMES}
    (CONTINUATION_ARTIFACTS / "file_hashes.json").write_text(
        json.dumps(file_hashes, indent=2, sort_keys=True) + "\n"
    )

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_slope_exposure_continuation_verification_v2",
        "depends_on_gitignored_run_files": False,
        "checks": checks,
        "details": details,
        "saved_overall_classification": saved_decision["overall_classification"],
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = CONTINUATION_ARTIFACTS / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
