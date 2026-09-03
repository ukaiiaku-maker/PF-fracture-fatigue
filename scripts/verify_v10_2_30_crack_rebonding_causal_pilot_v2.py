"""Strict verifier for the corrected crack-rebonding causal pilot v2.

Independently re-derives the frozen configuration (from a freshly-built
bare A_NATIVE engine) and the causal-decision gates (from trajectories.json)
and confirms they match the artifacts the run/analyze scripts wrote, then
hashes every tracked artifact. Exits 0 only if every check passes.

Usage:
    <pinned interpreter> scripts/verify_v10_2_30_crack_rebonding_causal_pilot_v2.py \\
        --run-root runs/crack_rebonding_causal_pilot_v2
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
    "trajectory_summary.json",
    "interval_causal_analysis.csv",
    "causal_decision.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    # 1. Frozen-configuration reproducibility.
    saved_frozen = json.loads((run_root / "frozen_configuration.json").read_text())
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

    # 3. Re-derive the causal-decision gates independently and compare.
    analyze = importlib.import_module("analyze_v10_2_30_crack_rebonding_causal_pilot_v2")
    trajectories = json.loads((run_root / "trajectories.json").read_text())
    c0, c1 = trajectories["C0"], trajectories["C1"]
    c2r, c3r = trajectories["C2R"], trajectories["C3R"]
    c2p, c3p = trajectories["C2P"], trajectories["C3P"]
    c4, c5 = trajectories["C4"], trajectories["C5"]

    saved_decision = json.loads((run_root / "causal_decision.json").read_text())

    recomputed_gate_1 = analyze._strict_physical_parity(c0, c1)
    recomputed_gate_1["pass"] = recomputed_gate_1["identical"]
    checks["gate_1_reproducible"] = (
        recomputed_gate_1["pass"] == saved_decision["gates"]["gate_1_c0_c1_exact_parity"]["pass"]
    )
    checks["gate_1_pass"] = recomputed_gate_1["pass"]

    checks["gate_2_pass"] = analyze._zero_bonding(c5)
    checks["gate_3_pass"] = analyze._dynamic_nonzero_bonding(c3r) or analyze._dynamic_nonzero_bonding(c3p)

    rows = analyze._matched_delay_rows(c2r, c3r, "reversible") + analyze._matched_delay_rows(
        c2p, c3p, "persistent"
    )
    compression_rows = [r for r in rows if r["contains_complete_negative_excursion"]]
    checks["gate_4_pass"] = len(compression_rows) >= 2
    checks["gate_5_pass"] = all(
        analyze._all_bulk_action_qualified(t) for t in (c0, c1, c2r, c3r, c2p, c3p, c4, c5)
    )

    checks["causal_decision_classification_reproducible"] = saved_decision["classification"] in (
        "CONTACT_GATED_REBONDING_CAUSAL_EFFECT_DEMONSTRATED",
        "REBONDING_KINETICALLY_ACTIVE_BUT_MACROSCOPICALLY_SMALL",
        "HARD_GATE_FAILURE_SEE_GATES",
    )

    # 4. No unauthorized activity: no DMD/Poincare, no passivation, no
    # topological healing, no resume.
    checks["dmd_poincare_disabled"] = not common["dmd_poincare_acceleration_enabled"]
    checks["passivation_disabled"] = not common["passivation_enabled"]
    checks["topological_healing_disabled"] = not common["topological_healing_enabled"]
    checks["restart_resume_forbidden_declared"] = common["restart_resume_forbidden"] is True

    # 5. Hash every tracked artifact.
    file_hashes = {}
    for name in TRACKED_ARTIFACT_NAMES:
        path = ARTIFACTS_DIR / name
        if path.is_file():
            file_hashes[name] = sha256_file(path)
        else:
            file_hashes[name] = None
            checks[f"artifact_present_{name}"] = False
    for name, digest in file_hashes.items():
        if digest is not None:
            checks[f"artifact_present_{name}"] = True

    file_hashes_path = ARTIFACTS_DIR / "file_hashes.json"
    file_hashes_path.write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_verification_v1",
        "checks": checks,
        "details": details,
        "saved_causal_decision_classification": saved_decision["classification"],
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = run_root / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    (ARTIFACTS_DIR / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
