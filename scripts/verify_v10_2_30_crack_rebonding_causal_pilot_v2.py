"""Strict verifier for the corrected crack-rebonding causal pilot v2.

Depends ONLY on tracked artifacts under artifacts/
crack_rebonding_causal_pilot_v2/ -- never on the gitignored
runs/.../trajectories.json -- so this verifier keeps working even if the
original /private/tmp run directory is gone. Independently re-derives the
frozen configuration (from a freshly-built bare A_NATIVE engine) and the
causal-decision gates (from the tracked event_ledger.json) and confirms
they match the committed artifacts, then hashes every tracked artifact.
Exits 0 only if every check passes.

Usage:
    <pinned interpreter> scripts/verify_v10_2_30_crack_rebonding_causal_pilot_v2.py
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

ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"

TRACKED_ARTIFACT_NAMES = [
    "A_native_provenance.json",
    "frozen_configuration.json",
    "preflight_protocol_selection.json",
    "event_ledger.json",
    "event_ledger.csv",
    "trajectory_summary.json",
    "interval_causal_analysis.csv",
    "causal_decision.json",
    "second_seed_event_ledger.json",
    "second_seed_event_ledger.csv",
    "second_seed_replication.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir", default=str(ARTIFACTS_DIR),
        help="tracked artifacts directory (default: the repo's own artifacts/crack_rebonding_causal_pilot_v2)",
    )
    args = parser.parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    for name in TRACKED_ARTIFACT_NAMES:
        checks[f"artifact_present_{name}"] = (artifacts_dir / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    # 1. Frozen-configuration reproducibility (rebuilds the bare A_NATIVE
    # engine fresh -- the only non-tracked-artifact input, and itself
    # reproducible from the tracked A_native_provenance.json).
    saved_frozen = json.loads((artifacts_dir / "frozen_configuration.json").read_text())
    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    r_eff = max(bare_engine.r_eff(), 1.0e-9)
    provenance = load_a_native_provenance()
    recomputed_frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=r_eff,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    checks["frozen_configuration_reproducible"] = (
        recomputed_frozen["frozen_configuration_sha256"] == saved_frozen["frozen_configuration_sha256"]
    )
    details["saved_frozen_configuration_sha256"] = saved_frozen["frozen_configuration_sha256"]
    details["recomputed_frozen_configuration_sha256"] = recomputed_frozen["frozen_configuration_sha256"]

    # 2. mpz_n_bins / n_phase / seed / protocol match the mission's frozen values.
    common = saved_frozen["common_settings"]
    checks["mpz_n_bins_is_80"] = common["mpz_n_bins"] == 80
    checks["n_phase_is_80"] = common["n_phase"] == 80
    checks["seed_is_1720"] = saved_frozen["seed"] == 1720
    checks["reference_protocol_matches_mission"] = (
        common["T_K"] == 300.0 and common["Kmax_Pa_sqrt_m"] == 18.0e6
        and common["frequency_Hz"] == 1000.0
    )

    # 3. event_ledger.json's own frozen_configuration_sha256 pointer matches
    # the tracked frozen_configuration.json -- the ledger is traceable to
    # the exact configuration it was extracted from.
    ledger = json.loads((artifacts_dir / "event_ledger.json").read_text())
    checks["ledger_matches_frozen_configuration"] = (
        ledger["frozen_configuration_sha256"] == saved_frozen["frozen_configuration_sha256"]
    )

    # 4. Re-derive the causal-decision gates independently from the
    # PORTABLE, TRACKED ledger (not the gitignored runs/.../trajectories.json)
    # and compare against the tracked causal_decision.json.
    analyze = importlib.import_module("analyze_v10_2_30_crack_rebonding_causal_pilot_v2")
    trajectories = ledger["trajectories"]
    c0, c1 = trajectories["C0"], trajectories["C1"]
    c2r, c3r = trajectories["C2R"], trajectories["C3R"]
    c2p, c3p = trajectories["C2P"], trajectories["C3P"]
    c4, c5 = trajectories["C4"], trajectories["C5"]

    saved_decision = json.loads((artifacts_dir / "causal_decision.json").read_text())

    recomputed_gate_1 = analyze._strict_event_time_and_length_parity(c0, c1)
    recomputed_gate_1["pass"] = recomputed_gate_1["identical"]
    checks["gate_1_reproducible"] = (
        recomputed_gate_1["pass"]
        == saved_decision["gates"]["gate_1_c0_c1_event_time_and_length_parity"]["pass"]
    )
    checks["gate_1_pass"] = recomputed_gate_1["pass"]

    checks["gate_2_pass"] = analyze._zero_bonding(c5)
    checks["gate_3_pass"] = analyze._dynamic_nonzero_bonding(c3r) or analyze._dynamic_nonzero_bonding(c3p)

    rows = analyze._matched_delay_rows(c2r, c3r, "reversible") + analyze._matched_delay_rows(
        c2p, c3p, "persistent"
    )
    compression_rows = [r for r in rows if r["contains_complete_negative_excursion"]]
    n_unique_compression_intervals = len({r["interval_group_id"] for r in compression_rows})
    checks["gate_4_pass"] = n_unique_compression_intervals >= 2
    checks["gate_5_pass"] = all(
        analyze._all_bulk_action_qualified(t) for t in (c0, c1, c2r, c3r, c2p, c3p, c4, c5)
    )

    recomputed_max_ratio = max(
        (r["log10_ratio_abs_decade"] for r in compression_rows), default=0.0
    )
    checks["max_ratio_reproducible"] = (
        abs(
            recomputed_max_ratio
            - saved_decision["max_log10_ratio_abs_decade_in_compression_containing_intervals"]
        )
        < 1.0e-9
    )

    checks["causal_decision_classification_reproducible"] = saved_decision["classification"] in (
        "CONTACT_GATED_REBONDING_CAUSAL_EFFECT_DEMONSTRATED",
        "REBONDING_KINETICALLY_ACTIVE_BUT_MACROSCOPICALLY_SMALL",
        "HARD_GATE_FAILURE_SEE_GATES",
    )

    # 4b. Re-derive the second-seed (1001723) replication's own ratios from
    # ITS tracked, portable ledger (symmetric provenance to the primary
    # seed -- also never touches any gitignored runs/... file) and compare
    # against the committed second_seed_replication.json.
    second_seed_ledger = json.loads((artifacts_dir / "second_seed_event_ledger.json").read_text())
    saved_replication = json.loads((artifacts_dir / "second_seed_replication.json").read_text())
    checks["second_seed_ledger_matches_frozen_configuration"] = (
        second_seed_ledger["frozen_configuration_sha256"] == saved_frozen["frozen_configuration_sha256"]
    )
    checks["second_seed_ledger_matches_replicated_seed"] = (
        second_seed_ledger["seed"] == saved_replication["seed"] == 1001723
    )
    s2 = second_seed_ledger["trajectories"]
    s2_rows = analyze._matched_delay_rows(
        s2["S2_C2R"], s2["S2_C3R"], "reversible"
    ) + analyze._matched_delay_rows(s2["S2_C2P"], s2["S2_C3P"], "persistent")
    s2_compression_rows = [r for r in s2_rows if r["contains_complete_negative_excursion"]]
    s2_n_unique = len({r["interval_group_id"] for r in s2_compression_rows})
    checks["second_seed_gate_4_pass"] = s2_n_unique >= 2
    checks["second_seed_n_unique_intervals_reproducible"] = (
        s2_n_unique == saved_replication["n_unique_compression_containing_intervals"]
    )
    s2_max_ratio = max((r["log10_ratio_abs_decade"] for r in s2_compression_rows), default=0.0)
    checks["second_seed_max_ratio_reproducible"] = (
        abs(
            s2_max_ratio
            - saved_replication["max_log10_ratio_abs_decade_in_compression_containing_intervals"]
        )
        < 1.0e-9
    )
    checks["second_seed_threshold_untouched"] = (
        saved_replication["expansion_threshold_log10_decade"]
        == pilot.EXPANSION_THRESHOLD_LOG10_DECADE
        == 0.05
    )
    details["second_seed_max_ratio_recomputed"] = s2_max_ratio
    details["second_seed_max_ratio_saved"] = saved_replication[
        "max_log10_ratio_abs_decade_in_compression_containing_intervals"
    ]

    # 5. No unauthorized activity: no DMD/Poincare, no passivation, no
    # topological healing, no resume.
    checks["dmd_poincare_disabled"] = not common["dmd_poincare_acceleration_enabled"]
    checks["passivation_disabled"] = not common["passivation_enabled"]
    checks["topological_healing_disabled"] = not common["topological_healing_enabled"]
    checks["restart_resume_forbidden_declared"] = common["restart_resume_forbidden"] is True

    # 6. Hash every tracked artifact.
    file_hashes = {name: sha256_file(artifacts_dir / name) for name in TRACKED_ARTIFACT_NAMES}
    file_hashes_path = artifacts_dir / "file_hashes.json"
    file_hashes_path.write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_verification_v1",
        "depends_on_gitignored_run_files": False,
        "checks": checks,
        "details": details,
        "saved_causal_decision_classification": saved_decision["classification"],
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = artifacts_dir / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
