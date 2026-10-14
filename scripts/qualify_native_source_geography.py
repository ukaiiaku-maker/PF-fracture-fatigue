#!/usr/bin/env python3
"""Qualify pure-bulk and tip-plus-bulk native geography with restart."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import OutputProfile, PRESETS
from scripts.analyze_f1_current_restart import compare


GEOGRAPHIES = {
    "bulk_uniform": "pf_monotonic_bulk",
    "tip_plus_bulk": "pf_monotonic_tip_plus_bulk",
}


def native_args(kernel_family: Path, steps: int) -> list[str]:
    return [
        "--signed-kernel-family", str(kernel_family.resolve()),
        "--mode", "2d",
        "--parameter-option", "v913_paper_dbtt01_0202500_persistent_sites",
        "--temperatures", "900",
        "--steps", str(int(steps)),
        "--dU", "1e-5",
        "--dt", "8.4",
        "--n-stagger", "1",
        "--da-phys", "5e-6",
        "--target-crack-extension-um", "10",
        "--front-state-model", "moving_pz",
        "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity",
        "--bulk-plasticity-mode", "full_field",
        "--directional-j-mode", "root_signed",
        "--active-shielding",
        "--signed-active-shielding",
        "--mobile-shield-fraction", "0",
        "--no-wake-shielding",
        "--crystal-aniso",
        "--crystal-compete",
        "--crystal-theta-deg", "30",
        "--crystal-material", "w",
        "--j-decomposition", "cluster",
        "--max-fronts", "1",
        "--crack-backend", "sharp_wake",
        "--print-every", "1",
        "--nx", "12",
        "--ny", "24",
    ]


def _binding(path: Path) -> dict:
    payload = json.loads(
        (path / "shared_runtime_attestation.post.json").read_text()
    )
    binding = payload.get("executed_physics_binding")
    if not isinstance(binding, dict) or not binding.get(
        "all_expected_bindings_match", False
    ):
        raise RuntimeError(f"missing executed geography binding in {path}")
    return binding


def qualify(root: Path, kernel_family: Path) -> dict:
    if not kernel_family.is_file():
        raise FileNotFoundError(kernel_family)
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for geography, preset in GEOGRAPHIES.items():
        options = replace(PRESETS[preset], output=OutputProfile.COMPACT_SMOKE)
        initial = root / geography / "initial_step1"
        restarted = root / geography / "restart_to_step2"
        uninterrupted = root / geography / "uninterrupted_to_step2"
        execute_shared(
            options, native_args(kernel_family, 1), initial,
            source_policy="qualified",
        )
        execute_shared(
            options, native_args(kernel_family, 2), restarted,
            source_policy="qualified", restart_from=initial / "checkpoint",
        )
        execute_shared(
            options, native_args(kernel_family, 2), uninterrupted,
            source_policy="qualified",
        )
        comparison = compare(
            uninterrupted / "checkpoint", restarted / "checkpoint"
        )
        comparison_path = root / geography / "restart_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
        bindings = {
            label: _binding(path)
            for label, path in (
                ("initial", initial), ("restarted", restarted),
                ("uninterrupted", uninterrupted),
            )
        }
        binding_geographies = {
            item["source_geography"] for item in bindings.values()
        }
        passed = (
            comparison["status"] == "PASS"
            and binding_geographies == {geography}
        )
        results[geography] = {
            "status": "PASS" if passed else "FAIL",
            "options_sha256": options.fingerprint,
            "restart_family": (
                "v10_4_pure_bulk_combined_outer"
                if geography == "bulk_uniform"
                else "v10_4_tip_plus_bulk_combined_outer"
            ),
            "executed_bindings": bindings,
            "restart_comparison": str(comparison_path),
            "complete_terminal_state_equal": comparison[
                "complete_terminal_state_equal"
            ],
            "state_arrays_compared": len(comparison["state_arrays"]),
        }
    report = {
        "schema": "native-source-geography-qualification/1",
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in results.values())
            else "FAIL"
        ),
        "classification": "compact readiness qualification",
        "production_execution_authorized": False,
        "kernel_family": str(kernel_family.resolve()),
        "results": results,
        "scope": (
            "native geography selection, fail-closed post-run binding, and "
            "two-step restart only; not a source-geography sensitivity campaign"
        ),
    }
    (root / "qualification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-family", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify(args.output.resolve(), args.kernel_family.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
