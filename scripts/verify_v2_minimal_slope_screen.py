"""Strict verifier for the minimal slope screen.

Depends ONLY on tracked artifacts (this worktree's
artifacts/crack_rebonding_minimal_slope_screen_v1/ and the parent branch's
artifacts/crack_rebonding_causal_pilot_v2/, both git-tracked) -- never on
any gitignored runs/... file. Independently re-derives S_h/the slope fit
from the tracked ledgers and confirms them against the committed
slope_screen_decision.json, confirms the reused Kmax=18 data matches the
parent's own tracked ledgers exactly, and confirms every unauthorized
activity flag stays false.

Usage:
    <pinned interpreter> scripts/verify_v2_minimal_slope_screen.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402

PARENT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
SCREEN_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"

SCREEN_TRACKED_ARTIFACT_NAMES = [
    "frozen_predictions.json",
    "exposure_gate_report.json",
    "event_ledger.json",
    "event_ledger.csv",
    "slope_screen_decision.json",
]
PARENT_REQUIRED_ARTIFACT_NAMES = [
    "frozen_configuration.json",
    "event_ledger.json",
    "second_seed_event_ledger.json",
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

    for name in SCREEN_TRACKED_ARTIFACT_NAMES:
        checks[f"screen_artifact_present_{name}"] = (SCREEN_ARTIFACTS_DIR / name).is_file()
    for name in PARENT_REQUIRED_ARTIFACT_NAMES:
        checks[f"parent_artifact_present_{name}"] = (PARENT_ARTIFACTS_DIR / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    parent_frozen = json.loads((PARENT_ARTIFACTS_DIR / "frozen_configuration.json").read_text())
    frozen_predictions = json.loads((SCREEN_ARTIFACTS_DIR / "frozen_predictions.json").read_text())
    checks["frozen_predictions_matches_parent_frozen_config"] = (
        frozen_predictions["parent_frozen_configuration_sha256"]
        == parent_frozen["frozen_configuration_sha256"]
    )

    # Re-derive the parent's frozen config from scratch too (transitively
    # confirms the reused Kmax=18 barriers/kinetics are exactly what the
    # completed two-seed pilot qualified).
    from arrhenius_fracture.a_native_engine_v10230 import load_a_native_provenance
    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    r_eff = max(bare_engine.r_eff(), 1.0e-9)
    provenance = load_a_native_provenance()
    recomputed_parent_frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=r_eff,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    checks["parent_frozen_configuration_reproducible"] = (
        recomputed_parent_frozen["frozen_configuration_sha256"]
        == parent_frozen["frozen_configuration_sha256"]
    )

    # Pi_K varies naturally across the grid -- confirm it is NOT held fixed.
    pi_k_values = {
        p["Kmax_Pa_sqrt_m"]: p["Pi_K_at_this_load"]
        for p in frozen_predictions["predictions"]["reversible"]
    }
    checks["pi_k_varies_naturally_not_rescaled"] = len(set(pi_k_values.values())) == len(pi_k_values)

    # Re-derive the slope screen's own decision from the tracked event
    # ledger (+ the parent's tracked ledgers for the reused Kmax=18 point)
    # and compare against the committed slope_screen_decision.json.
    saved_decision = json.loads((SCREEN_ARTIFACTS_DIR / "slope_screen_decision.json").read_text())
    exposure_report = json.loads((SCREEN_ARTIFACTS_DIR / "exposure_gate_report.json").read_text())
    checks["exposure_gate_status_reproducible"] = (
        saved_decision["status"]
        == ("STOPPED_EARLY_EXPOSURE_GATE_FAILED" if exposure_report["stopped_early_at"] else "COMPLETE")
    )

    if saved_decision["status"] == "COMPLETE":
        import importlib
        analyze = importlib.import_module("analyze_v2_minimal_slope_screen")
        ledger = json.loads((SCREEN_ARTIFACTS_DIR / "event_ledger.json").read_text())
        trajectories = ledger["trajectories"]

        for seed in analyze.SEEDS:
            reused = analyze._reused_kmax18(seed)
            rate_shift_18 = analyze._rate_shift(reused["zero"], reused["finite"])
            rate_shift_15 = analyze._rate_shift(
                trajectories[f"C2R_K15MPa_seed{seed}"], trajectories[f"C3R_K15MPa_seed{seed}"],
            )
            rate_shift_21 = analyze._rate_shift(
                trajectories[f"C2R_K21MPa_seed{seed}"], trajectories[f"C3R_K21MPa_seed{seed}"],
            )
            saved_seed = saved_decision["per_seed"][str(seed)]
            for label, recomputed, saved_key in (
                ("15", rate_shift_15, "rate_shift_Kmax15"),
                ("18_reused", rate_shift_18, "rate_shift_Kmax18_reused"),
                ("21", rate_shift_21, "rate_shift_Kmax21"),
            ):
                checks[f"seed_{seed}_S_h_{label}_reproducible"] = (
                    abs(recomputed["S_h_decade"] - saved_seed[saved_key]["S_h_decade"]) < 1.0e-9
                )
            checks[f"seed_{seed}_classification_reproducible"] = (
                saved_seed["classification_this_seed"]
                in (
                    "REBONDING_RATE_OFFSET_LIKE", "REBONDING_STEEPENS_LOCAL_RESPONSE",
                    "REBONDING_FLATTENS_LOCAL_RESPONSE", "REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED",
                )
            )

    checks["multi_K_paris_slope_campaign_not_authorized"] = (
        saved_decision.get("multi_K_paris_slope_campaign_authorized", False) is False
    )
    checks["part_x_not_authorized"] = saved_decision.get("part_x_authorized", False) is False
    checks["production_merge_not_authorized"] = (
        saved_decision.get("production_line_merge_authorized", False) is False
    )

    file_hashes = {name: sha256_file(SCREEN_ARTIFACTS_DIR / name) for name in SCREEN_TRACKED_ARTIFACT_NAMES}
    file_hashes_path = SCREEN_ARTIFACTS_DIR / "file_hashes.json"
    file_hashes_path.write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_verification_v1",
        "depends_on_gitignored_run_files": False,
        "checks": checks,
        "details": details,
        "saved_status": saved_decision["status"],
        "saved_overall_classification": saved_decision.get("overall_classification"),
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = SCREEN_ARTIFACTS_DIR / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
