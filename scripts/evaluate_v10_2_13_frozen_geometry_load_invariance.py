#!/usr/bin/env python3
"""Evaluate signed shielding response at several loads on one frozen FEM geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    evaluate_frozen_geometry_load_invariance,
)


def _floats(text: str):
    return [float(value) for value in str(text).replace(",", " ").split()]


def _persist_provenance(payload: dict, snapshot: Path, outroot: Path) -> dict:
    source = snapshot.expanduser().resolve()
    source_metadata = json.loads((source / "snapshot.json").read_text())
    for case in payload.get("generated_load_cases", []):
        audit_path = Path(case["audit"])
        audit = json.loads(audit_path.read_text())
        scale = float(case["load_scale"])
        audit.update(
            {
                "snapshot": str(source),
                "source_snapshot_schema": source_metadata.get("schema"),
                "frozen_geometry_parent_state_id": payload["parent_state_id"],
                "frozen_geometry_load_scale": scale,
                "scaled_Uy_top_m": float(source_metadata["Uy_top_m"]) * scale,
                "scaled_Uy_bot_m": float(source_metadata["Uy_bot_m"]) * scale,
                "transient_scaled_snapshot_path_persisted": False,
                "scaled_state_reconstructed_from_source_snapshot": True,
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2))
        case["station_report"] = audit
    payload["source_snapshot"] = str(source)
    payload["transient_scaled_snapshot_paths_persisted"] = False
    payload["scaled_states_reconstructible_from_source_snapshot"] = True
    (outroot / "frozen_geometry_load_invariance.json").write_text(
        json.dumps(payload, indent=2)
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--load-scales", default="0.5 1.0 1.5")
    parser.add_argument("--magnitudes", default="0.25 0.5")
    parser.add_argument("--ribbon-width-m", type=float)
    parser.add_argument("--minimum-station-spacing-m", type=float)
    parser.add_argument("--linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--load-invariance-tolerance", type=float, default=0.05)
    parser.add_argument("--significance-floor-fraction", type=float, default=1.0e-3)
    args = parser.parse_args()
    payload = evaluate_frozen_geometry_load_invariance(
        args.snapshot,
        outroot=args.outroot,
        load_scales=_floats(args.load_scales),
        perturbation_magnitudes=_floats(args.magnitudes),
        ribbon_width_m=args.ribbon_width_m,
        minimum_station_spacing_m=args.minimum_station_spacing_m,
        linearity_tolerance=args.linearity_tolerance,
        load_invariance_tolerance=args.load_invariance_tolerance,
        significance_floor_fraction=args.significance_floor_fraction,
    )
    payload = _persist_provenance(payload, args.snapshot, args.outroot)
    print(
        json.dumps(
            {
                "outroot": str(args.outroot),
                "state_id": payload["parent_state_id"],
                "cumulative_crack_path_extension_m": payload[
                    "cumulative_crack_path_extension_m"
                ],
                "load_scales": payload["load_scales"],
                "maximum_relative_load_variation": payload["checks"][
                    "maximum_relative_load_variation"
                ],
                "load_invariance_passed": payload["load_invariance_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
