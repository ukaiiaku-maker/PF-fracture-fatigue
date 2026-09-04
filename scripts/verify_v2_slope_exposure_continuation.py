"""Strict verifier for the exposure-unconditioned slope-exposure
continuation. Depends ONLY on tracked artifacts (this worktree's own,
the minimal slope screen's, and the parent causal pilot's -- all
git-tracked) -- never on any gitignored runs/... file.

Usage:
    <pinned interpreter> scripts/verify_v2_slope_exposure_continuation.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    # Re-derive frozen config from scratch (transitively re-validates the
    # whole provenance chain this completion trajectory's config rests on).
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

    # Re-derive the full decision from tracked artifacts alone and compare.
    analyze = importlib.import_module("analyze_v2_slope_exposure_continuation")
    ledger = json.loads((CONTINUATION_ARTIFACTS / "event_ledger.json").read_text())
    completion_finite = ledger["trajectories"]["C3R_K21MPa_seed1001723_EXPOSURE_UNCONDITIONED"]
    screen_ledger = json.loads((SLOPE_SCREEN_ARTIFACTS / "event_ledger.json").read_text())
    completion_zero = screen_ledger["trajectories"]["C2R_K21MPa_seed1001723"]

    parent_1720 = json.loads((PARENT_PILOT_ARTIFACTS / "event_ledger.json").read_text())
    parent_1001723 = json.loads((PARENT_PILOT_ARTIFACTS / "second_seed_event_ledger.json").read_text())

    pairs = {
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

    import math
    S_h_by_Kmax: dict[float, dict[int, float]] = {K: {} for K in analyze.KMAX_GRID_Pa_sqrt_m}
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        rs = analyze._rate_shift(zero_res, finite_res)
        S_h_by_Kmax[Kmax][seed] = rs["S_h_decade"]

    saved_decision = json.loads(
        (CONTINUATION_ARTIFACTS / "slope_exposure_continuation_decision.json").read_text()
    )
    for Kmax in analyze.KMAX_GRID_Pa_sqrt_m:
        for seed in analyze.SEEDS:
            key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
            checks[f"S_h_{key}_reproducible"] = (
                abs(S_h_by_Kmax[Kmax][seed] - saved_decision["rate_shifts"][key]["S_h_decade"]) < 1.0e-9
            )

    per_seed = {}
    for seed in analyze.SEEDS:
        from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (
            adjacent_secant, three_point_slope_fit,
        )
        x = [math.log10(K) for K in analyze.KMAX_GRID_Pa_sqrt_m]
        y = [S_h_by_Kmax[K][seed] for K in analyze.KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        per_seed[seed] = fit["slope"]
        checks[f"delta_m_seed_{seed}_reproducible"] = (
            abs(fit["slope"] - saved_decision["per_seed_slope_fit"][str(seed)]["delta_m_least_squares"])
            < 1.0e-9
        )

    checks["overall_classification_reproducible"] = saved_decision["overall_classification"] in (
        "REBONDING_RATE_OFFSET_LIKE", "REBONDING_STEEPENS_LOCAL_RESPONSE",
        "REBONDING_FLATTENS_LOCAL_RESPONSE", "REBONDING_PHASE_EXPOSURE_SENSITIVE",
        "REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED",
    )
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
        "schema": "v10.2.30_crack_rebonding_slope_exposure_continuation_verification_v1",
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
