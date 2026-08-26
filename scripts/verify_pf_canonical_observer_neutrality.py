#!/usr/bin/env python3
"""Compare identical sparse-observer and observer-off PF smoke trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


TRAJECTORY_FILES = (
    "steps_1100K.csv", "fronts_1100K.csv", "crack_path_1100K.csv",
    "stochastic_avalanche_geometry_events.json", "summary.json",
    "sharp_wake_advance_log.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packed_json(path: Path):
    return json.loads(subprocess.check_output(["zstd", "-dc", str(path)]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--off-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = []; all_equal = True; profile_gate = True
    sparse_cases = sorted(path for path in args.sparse_root.iterdir() if path.is_dir())
    for sparse in sparse_cases:
        off = args.off_root / sparse.name
        for name in TRAJECTORY_FILES:
            left, right = sparse / name, off / name
            equal = left.is_file() and right.is_file() and sha256(left) == sha256(right)
            rows.append({"case_id": sparse.name, "artifact": name,
                         "sparse_sha256": sha256(left) if left.is_file() else None,
                         "observer_off_sha256": sha256(right) if right.is_file() else None,
                         "byte_exact": equal})
            all_equal = all_equal and equal
        packed = sparse / "anisotropic_emission_audit_v10174.json.zst"
        records = load_packed_json(packed)
        if isinstance(records, dict):
            records = records.get("records", records.get("audit_records", []))
        profile_records = [record for record in records if "taylor_peierls_state_profile_schema" in record]
        event_records = [record for record in records if bool(record.get("fired", False))]
        event_only = bool(profile_records) and len(profile_records) == len(event_records) and all(
            bool(record.get("fired", False)) for record in profile_records)
        profile_gate = profile_gate and event_only
        rows.append({"case_id": sparse.name, "artifact": "SPARSE_PROFILE_EVENT_BOUNDARY_GATE",
                     "sparse_sha256": len(profile_records), "observer_off_sha256": len(event_records),
                     "byte_exact": event_only})
    import csv
    table = args.out / "pf_observer_trajectory_neutrality.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {
        "schema": "pf_canonical_observer_neutrality_v1", "case_count": len(sparse_cases),
        "trajectory_files_byte_exact": all_equal, "profiles_only_at_event_boundaries": profile_gate,
        "observer_feedback": False, "status": "PASS" if all_equal and profile_gate else "FAIL",
        "table_sha256": sha256(table),
    }
    (args.out / "pf_observer_trajectory_neutrality.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
