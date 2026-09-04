"""Strict, independently re-deriving verifier for the developed-response
confirmation campaign. Depends ONLY on files tracked in this branch
(artifacts/crack_rebonding_developed_confirmation/*, artifacts/
crack_rebonding_causal_pilot_v2/frozen_configuration.json and
A_native_provenance.json) -- never on any gitignored runs/ directory, so
it must still pass with every runs/ output deleted.

Per developed_confirmation_verification_contract.json:
  - rebuilds the bare A_NATIVE engine and reproduces complete_row_sha256
    from scratch;
  - reloads RB2_reversible_zero/finite from the parent frozen
    configuration and reproduces their config_hash() exactly;
  - re-derives, from the tracked ledger ONLY, the developed/stationarity
    gate (stable_growth_gate, reused verbatim) for every trajectory and
    compares stable_growth_provisional against the saved decision;
  - checks hazard_threshold_action sequence identity and
    all_bulk_action_qualified certification for every zero/finite pair;
  - re-derives S_h/delta_m/secant fits and the Section F classification,
    requiring EXACT dict equality (via JSON round-trip normalization)
    against the saved decision, not membership in an allowed set;
  - if Stage 2 was not authorized, confirms no seed-1001723 trajectory
    exists anywhere in the tracked ledger.

Usage:
    <pinned interpreter> scripts/verify_developed_confirmation.py
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
from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (  # noqa: E402
    STABILITY_DEFINITION_STRING,
    classify_developed_confirmation,
    effective_horizon_censored,
    stable_growth_gate,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
DEV_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
FREQUENCY_HZ = 1000.0
MAX_ACCEPTED_EVENTS = 30
MAX_PROJECTED_EXTENSION_m = 1.5e-4

TRACKED_ARTIFACT_NAMES = [
    "developed_confirmation_frozen_protocol.json",
    "developed_confirmation_job_registry.csv",
    "developed_confirmation_predictions.json",
    "developed_confirmation_verification_contract.json",
    "event_ledger.json",
    "event_ledger.csv",
    "developed_confirmation_decision.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_from_frozen(frozen: dict, key: str) -> CrackRebondingControls:
    payload = dict(frozen["configs"][key])
    payload["model_level"] = RebondModelLevel(payload["model_level"])
    payload["contact_model"] = ContactModel(payload["contact_model"])
    payload["feedback_mode"] = FeedbackMode(payload["feedback_mode"])
    payload["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(payload["initial_precrack_wake_mode"])
    return CrackRebondingControls(**payload)


def _traj_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    for name in TRACKED_ARTIFACT_NAMES:
        checks[f"artifact_present_{name}"] = (DEV_ARTIFACTS / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    protocol = json.loads((DEV_ARTIFACTS / "developed_confirmation_frozen_protocol.json").read_text())
    parent_frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    ledger = json.loads((DEV_ARTIFACTS / "event_ledger.json").read_text())
    decision = json.loads((DEV_ARTIFACTS / "developed_confirmation_decision.json").read_text())

    checks["ledger_config_hash_matches_parent_frozen"] = (
        ledger["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )

    # Independently rebuild the bare A_NATIVE engine and reproduce
    # complete_row_sha256 from scratch.
    provenance = load_a_native_provenance()
    checks["complete_row_sha256_matches_frozen_protocol"] = (
        provenance["complete_row_sha256"]
        == protocol["material_and_provenance"]["complete_active_material_row_sha256"]
    )

    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    r_eff = max(bare_engine.r_eff(), 1.0e-9)
    recomputed_frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=r_eff,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    checks["parent_frozen_configuration_reproducible"] = (
        recomputed_frozen["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )

    for key in ("RB2_reversible_zero", "RB2_reversible_finite"):
        cfg = _config_from_frozen(parent_frozen, key)
        checks[f"{key}_config_hash_reproducible"] = cfg.config_hash() == parent_frozen["config_hashes"][key]
    checks["dmd_poincare_acceleration_disabled"] = (
        parent_frozen["common_settings"]["dmd_poincare_acceleration_enabled"] is False
    )

    checks["stability_definition_string_unaltered"] = (
        protocol["developed_growth_and_stationarity_gate"]["stability_definition_string_verbatim"]
        == STABILITY_DEFINITION_STRING
    )

    seeds_present = set()
    for seed in (1720, 1001723):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            Kmax_MPa = int(round(Kmax / 1.0e6))
            for cohesion in ("zero", "finite"):
                name = _traj_name(seed, Kmax_MPa, cohesion)
                if name in ledger["trajectories"]:
                    seeds_present.add(seed)

    checks["stage_2_seed_absent_unless_authorized"] = True
    stage2_authorized = decision.get("stage_2_conditional_gate", {}).get("authorized", False)
    if not stage2_authorized:
        checks["stage_2_seed_absent_unless_authorized"] = 1001723 not in seeds_present
        if 1001723 in seeds_present:
            details["stage2_gate_authorized"] = False
            details["stage2_trajectories_found_anyway"] = True

    # Re-derive the developed/stationarity gate, hazard-threshold pairing,
    # and bulk-action certification for every present pair -- from the
    # tracked ledger only.
    for seed in sorted(seeds_present):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            Kmax_MPa = int(round(Kmax / 1.0e6))
            name_zero = _traj_name(seed, Kmax_MPa, "zero")
            name_finite = _traj_name(seed, Kmax_MPa, "finite")
            if name_zero not in ledger["trajectories"] or name_finite not in ledger["trajectories"]:
                continue
            zero_traj = ledger["trajectories"][name_zero]
            finite_traj = ledger["trajectories"][name_finite]
            key = f"K{Kmax_MPa}MPa_seed{seed}"

            gate_zero = stable_growth_gate(zero_traj["events"], frequency_Hz=FREQUENCY_HZ)
            gate_finite = stable_growth_gate(finite_traj["events"], frequency_Hz=FREQUENCY_HZ)

            saved_pair = decision[f"seed_{seed}_analysis"]["per_pair"][key]
            checks[f"{key}_gate_zero_stable_reproducible"] = (
                gate_zero["stable_growth_provisional"]
                == saved_pair["gate_zero"]["stable_growth_provisional"]
            )
            checks[f"{key}_gate_finite_stable_reproducible"] = (
                gate_finite["stable_growth_provisional"]
                == saved_pair["gate_finite"]["stable_growth_provisional"]
            )

            hz_zero = [e["hazard_threshold_action"] for e in zero_traj["events"]]
            hz_finite = [e["hazard_threshold_action"] for e in finite_traj["events"]]
            checks[f"{key}_hazard_threshold_sequence_identical"] = (
                len(hz_zero) == len(hz_finite) and all(a == b for a, b in zip(hz_zero, hz_finite))
            )
            checks[f"{key}_zero_all_bulk_action_qualified"] = all(
                e["all_bulk_action_qualified"] for e in zero_traj["events"]
            )
            checks[f"{key}_finite_all_bulk_action_qualified"] = all(
                e["all_bulk_action_qualified"] for e in finite_traj["events"]
            )

            horizon_zero = effective_horizon_censored(
                zero_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
                max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
            )
            horizon_finite = effective_horizon_censored(
                finite_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
                max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
            )
            checks[f"{key}_never_coerced_zero_rate_if_horizon_censored"] = not (
                (horizon_zero["stopped_by_budget_exhaustion"] and not gate_zero["stable_growth_provisional"]
                 and gate_zero["developed_interval"]["da_dN"] is None)
                and saved_pair["gate_zero"].get("developed_da_dN_m_per_cycle") == 0.0
            ) and not (
                (horizon_finite["stopped_by_budget_exhaustion"] and not gate_finite["stable_growth_provisional"]
                 and gate_finite["developed_interval"]["da_dN"] is None)
                and saved_pair["gate_finite"].get("developed_da_dN_m_per_cycle") == 0.0
            )

    # STRICT classification re-derivation via the analysis module itself
    # (same code path the campaign used to produce the saved decision) --
    # exact dict equality after JSON round-trip normalization.
    analyze = importlib.import_module("analyze_developed_confirmation")
    seed1720 = analyze.analyze_seed(ledger, 1720)
    seed1001723 = analyze.analyze_seed(ledger, 1001723)
    both_seeds_pass = (
        seed1720 is not None and seed1001723 is not None
        and seed1720["all_uncensored"] and seed1720["all_pairs_pass_developed_gate"]
        and seed1001723["all_uncensored"] and seed1001723["all_pairs_pass_developed_gate"]
    )
    delta_m_developed_by_seed = None
    S_h_developed_by_Kmax_by_seed = None
    if both_seeds_pass:
        delta_m_developed_by_seed = {
            1720: seed1720["fits_by_window"]["developed"]["delta_m_least_squares"],
            1001723: seed1001723["fits_by_window"]["developed"]["delta_m_least_squares"],
        }
        S_h_developed_by_Kmax_by_seed = {
            K: {1720: seed1720["S_h_developed_by_Kmax"][K], 1001723: seed1001723["S_h_developed_by_Kmax"][K]}
            for K in KMAX_GRID_Pa_sqrt_m
        }
    recomputed_classification = classify_developed_confirmation(
        both_seeds_pass_gate_uncensored=both_seeds_pass,
        delta_m_developed_by_seed=delta_m_developed_by_seed,
        S_h_developed_by_Kmax_by_seed=S_h_developed_by_Kmax_by_seed,
    )
    recomputed_normalized = json.loads(json.dumps(recomputed_classification, sort_keys=True))
    saved_normalized = json.loads(json.dumps(decision["terminal_classification"], sort_keys=True))
    checks["classification_exact_match"] = recomputed_normalized == saved_normalized
    details["recomputed_classification"] = recomputed_classification["classification"]
    details["saved_classification"] = decision["terminal_classification"]["classification"]

    checks["multi_K_paris_slope_campaign_not_authorized"] = (
        decision.get("multi_K_paris_slope_campaign_authorized", False) is False
    )
    checks["part_x_not_authorized"] = decision.get("part_x_authorized", False) is False
    checks["production_merge_not_authorized"] = (
        decision.get("production_line_merge_authorized", False) is False
    )

    file_hashes = {name: sha256_file(DEV_ARTIFACTS / name) for name in TRACKED_ARTIFACT_NAMES}
    (DEV_ARTIFACTS / "file_hashes.json").write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_developed_confirmation_verification_v1",
        "depends_on_gitignored_run_files": False,
        "checks": checks,
        "details": details,
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = DEV_ARTIFACTS / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
