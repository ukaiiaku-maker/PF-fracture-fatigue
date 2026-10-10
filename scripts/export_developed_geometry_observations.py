#!/usr/bin/env python3
"""Attach native event geometry arclength to an existing observation history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from arrhenius_fracture.fracture_observation_history import (
    ObservationHistory, with_geometry_event_arclength,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--geometry-events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    source = ObservationHistory.load(args.observations)
    events = json.loads(args.geometry_events.read_text())
    recovered = with_geometry_event_arclength(source, events)
    recovered.write_jsonl(args.out)
    values = [row.path_arclength_m for row in recovered.rows]
    payload = {
        "schema": "developed-growth-geometry-export/1",
        "observation_source": str(args.observations),
        "observation_source_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "geometry_event_source": str(args.geometry_events),
        "geometry_event_source_sha256": hashlib.sha256(args.geometry_events.read_bytes()).hexdigest(),
        "geometry_event_count": len(events),
        "output": str(args.out),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "observation_rows": len(recovered.rows),
        "path_arclength_finite_rows": len(values),
        "terminal_path_arclength_m": max(values),
        "terminal_projected_extension_m": max(row.stage_extension_m for row in recovered.rows),
        "association": "single-front chronological native event endpoints; committed rows consume event_multiplicity",
    }
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
