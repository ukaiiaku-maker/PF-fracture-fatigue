#!/usr/bin/env python3
"""Add the v10.1.6.1 refresh-scale compatibility alias to existing cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def repair(root: Path) -> tuple[int, int]:
    scanned = 0
    changed = 0
    for path in sorted(root.glob("**/v10_1_driver_modes.json")):
        scanned += 1
        payload = json.loads(path.read_text())
        canonical = payload.get("campaign_refresh_length_scale")
        if canonical is None:
            continue
        if payload.get("campaign_refresh_scale") == canonical:
            continue
        payload["campaign_refresh_scale"] = canonical
        payload["matrix_audit_key_compatibility"] = "v10.1.6.1"
        path.write_text(json.dumps(payload, indent=2))
        changed += 1
    return scanned, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scanned, changed = repair(args.root)
    print(f"matrix audit alias repair: scanned={scanned} changed={changed} root={args.root}")


if __name__ == "__main__":
    main()
