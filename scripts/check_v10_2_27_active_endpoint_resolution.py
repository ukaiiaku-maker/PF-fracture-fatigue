#!/usr/bin/env python3
"""Verify exact active-bin-zero perturbations are resolved by captured FEM meshes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration
from arrhenius_fracture.physical_fem_snapshot_v10212 import load_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    snapshot_root = args.snapshot_root.expanduser().resolve()
    configuration = load_configuration(args.mechanical_config)
    first_center = (
        configuration.process_zone_length_m / (2.0 * configuration.process_zone_bins)
    )
    rows = []
    for metadata_path in sorted(snapshot_root.glob("*/snapshot.json")):
        state_root = metadata_path.parent
        data = load_snapshot(state_root)
        hbar_tip = float(data["mesh"].hbar_tip)
        burgers = float(data["mat"].b)
        placement_resolution = max(2.0 * hbar_tip, 10.0 * burgers, 1.0e-12)
        required_distance = 2.0 * placement_resolution
        passed = bool(
            math.isfinite(hbar_tip)
            and hbar_tip > 0.0
            and first_center + 1.0e-15 >= required_distance
        )
        rows.append(
            {
                "state_id": state_root.name,
                "first_active_bin_center_m": first_center,
                "hbar_tip_m": hbar_tip,
                "burgers_m": burgers,
                "placement_resolution_m": placement_resolution,
                "minimum_resolvable_active_station_m": required_distance,
                "passed": passed,
            }
        )
    if len(rows) < 2:
        raise SystemExit("active-endpoint resolution audit requires at least two snapshots")

    payload = {
        "schema": "v10.2.27_active_endpoint_resolution_audit_v1",
        "mechanical_configuration_fingerprint": configuration.fingerprint(),
        "active_station_policy_id": configuration.active_station_policy_id,
        "first_active_bin_center_m": first_center,
        "state_count": len(rows),
        "all_states_passed": all(row["passed"] for row in rows),
        "states": rows,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else snapshot_root / "active_endpoint_resolution_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_states_passed"]:
        worst = max(rows, key=lambda row: row["minimum_resolvable_active_station_m"])
        raise SystemExit(
            "captured FEM mesh cannot resolve active bin zero: "
            f"first_center={first_center:.9g} m, "
            f"required={worst['minimum_resolvable_active_station_m']:.9g} m. "
            "Reduce tip_h_fine_m and recalculate the configuration."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
