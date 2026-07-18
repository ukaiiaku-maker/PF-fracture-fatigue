#!/usr/bin/env python3
"""Validate monotonic/fatigue output metadata for shared uncapped shielding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required audit file: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"audit payload is not an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="one monotonic or fatigue case directory")
    args = parser.parse_args()

    modes = _load(args.root / "v10_1_driver_modes.json")
    source = _load(args.root / "v10_1_1_source_model.json")
    kinetic = _load(args.root / "kinetic_tip_cell_audit_v101.json")
    campaign = kinetic.get("campaign_calibration", {})

    checks = {
        "manifest_cap_disabled": modes.get("manifest_K_shield_cap_enabled") is False,
        "legacy_reference_only": modes.get("legacy_manifest_K_shield_cap_reference_only") is True,
        "shared_core": modes.get("shared_monotonic_and_fatigue_core") is True,
        "unbounded_cleavage_shielding": source.get("cleavage_shielding_bound")
        == "none; signed raw elastic dislocation field",
        "class_audit_cap_disabled": campaign.get("shielding_cap_from_manifest") is False,
        "class_audit_population_limited": campaign.get("shielding_saturation")
        == "population_dynamics_only",
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"root": str(args.root), "checks": checks, "pass": not failed}, indent=2))
    if failed:
        raise SystemExit("shared uncapped shielding audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
