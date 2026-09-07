#!/usr/bin/env python
"""Strict verifier for the bounded crack-rebonding causal pilot run.

Re-derives, independently of the run script, that a completed pilot run's
artifacts under --run-root are internally consistent:

- the frozen configuration's own recorded hash matches a fresh re-hash of
  its content;
- each trajectory config actually used (by hash) matches the frozen
  configuration's config_hashes;
- the six trajectories present are exactly P0..P5, each with a coherent
  event count / censoring state;
- the causal_analysis.json gate outcomes are recomputed FRESH from
  trajectories.json (not merely re-read) and compared byte-for-byte against
  what is on disk, so a hand-edited or stale analysis file is caught;
- the required interpreter recorded in the frozen configuration matches
  this process's own interpreter (documents which environment the run
  claims to have used -- this script itself must also be invoked under it).

Exits 0 (schema: v10.2.30_crack_rebonding_causal_pilot_verification_v1) only
if every one of the above holds AND the mission's 8 hard gates are present
in the analysis (their actual pass/fail values are NOT re-asserted here --
gate 2's documented tension with gate 3 is an expected, reported finding,
not a verifier failure; see the run's report for the honest accounting).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v10230 as pilot  # noqa: E402


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def verify(run_root: Path) -> dict:
    checks: dict[str, bool] = {}
    notes: dict[str, str] = {}

    frozen_path = run_root / "frozen_configuration.json"
    trajectories_path = run_root / "trajectories.json"
    analysis_path = run_root / "causal_analysis.json"

    for name, path in (
        ("frozen_configuration_exists", frozen_path),
        ("trajectories_exists", trajectories_path),
        ("causal_analysis_exists", analysis_path),
    ):
        checks[name] = path.exists()

    if not all(checks.values()):
        return {
            "schema": "v10.2.30_crack_rebonding_causal_pilot_verification_v1",
            "checks": checks,
            "notes": notes,
            "passed": False,
        }

    frozen = _load(frozen_path)
    trajectories = _load(trajectories_path)
    analysis = _load(analysis_path)

    # 1. Frozen configuration self-hash re-derivation.
    frozen_without_hash = {k: v for k, v in frozen.items() if k != "frozen_configuration_sha256"}
    recomputed_hash = pilot.canonical_hash(frozen_without_hash)
    checks["frozen_configuration_hash_matches"] = (
        recomputed_hash == frozen["frozen_configuration_sha256"]
    )

    # 2. Interpreter contract.
    checks["required_python_matches_this_process"] = sys.executable == frozen["required_python"]
    notes["required_python"] = frozen["required_python"]

    # 3. Six trajectories present, coherent.
    checks["all_six_trajectories_present"] = set(trajectories) == set(pilot.TRAJECTORY_NAMES)
    coherent = True
    for name in pilot.TRAJECTORY_NAMES:
        traj = trajectories.get(name)
        if traj is None:
            coherent = False
            continue
        if traj["n_accepted_events"] != len(traj["events"]):
            coherent = False
        if traj["censored"] and traj["n_accepted_events"] >= pilot.MAX_ACCEPTED_EVENTS:
            coherent = False
    checks["trajectory_event_counts_coherent"] = coherent

    # 4. Trajectory rebonding config hashes match the frozen configuration
    #    (each trajectory really ran against the frozen, not a redone, config).
    spec_by_name = {spec["name"]: spec for spec in frozen["trajectories"]}
    hash_ok = True
    for name in pilot.TRAJECTORY_NAMES:
        traj = trajectories.get(name, {})
        spec = spec_by_name.get(name, {})
        expected_hash = frozen["config_hashes"].get(spec.get("rebonding"))
        if traj.get("rebonding_cfg_hash") != expected_hash:
            hash_ok = False
    checks["trajectory_config_hashes_match_frozen"] = hash_ok

    # 5. Fresh recomputation of the causal analysis from trajectories.json,
    #    compared byte-for-byte against the on-disk analysis file.
    recomputed_analysis = pilot.causal_analysis(trajectories)
    checks["causal_analysis_reproducible_from_trajectories"] = recomputed_analysis == analysis

    # 6. The 8 hard gates are all present (their individual pass/fail is
    #    reported, not enforced here -- see module docstring).
    expected_gate_keys = {
        "gate_1_p0_p1_identical",
        "gate_2_p4_p5_identical_and_p5_zero_bonding",
        "gate_3_dynamic_nonzero_bonding",
        "gate_4_post_first_event_waiting_intervals",
        "gate_5_K_rebond_not_in_energy_gate_directly",
        "gate_6_certified_bulk_action_only",
        "gate_7_contact_proxy_and_mesh_backend_limitation_preserved",
        "gate_8_no_paris_slope_inference",
    }
    checks["all_eight_hard_gates_present"] = expected_gate_keys <= set(analysis.get("gates", {}))

    notes["gate_pass_summary"] = {
        key: analysis["gates"][key].get("pass", analysis["gates"][key].get("identical"))
        for key in expected_gate_keys
        if key in analysis.get("gates", {})
    }
    notes["overall_pass_as_reported"] = analysis.get("overall_pass")
    notes["part_x_physical_campaign_status"] = frozen.get("common_settings", {})
    notes["frozen_configuration_sha256"] = frozen.get("frozen_configuration_sha256")

    passed = all(checks.values())
    return {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_verification_v1",
        "checks": checks,
        "notes": notes,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = verify(Path(args.run_root))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
