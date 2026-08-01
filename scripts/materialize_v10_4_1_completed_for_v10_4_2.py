#!/usr/bin/env python3
"""Materialize completed v10.4.1 fracture cases into a v10.4.2 root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.reuse_v1041_v1042 import materialize_completed_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-commit", required=True)
    args = parser.parse_args()
    payload = materialize_completed_cases(
        args.source_root,
        args.destination_root,
        source_commit=args.source_commit,
        target_commit=args.target_commit,
    )
    print(json.dumps({
        "manifest": str(
            args.destination_root.expanduser().resolve()
            / "v10_4_2_materialized_reuse_manifest.json"
        ),
        "materialized_case_count": payload["materialized_case_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
