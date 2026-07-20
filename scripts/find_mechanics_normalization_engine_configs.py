#!/usr/bin/env python3
"""Locate serialized engine configs usable for mechanics normalization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _positive(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()

    matches = []
    seen = set()
    for supplied in args.roots:
        root = supplied.expanduser().resolve()
        paths = [root] if root.is_file() else root.rglob("*.json") if root.is_dir() else []
        for path in paths:
            path = path.resolve()
            if path in seen or path.suffix.lower() != ".json":
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            front = payload.get("front_config")
            mpz = payload.get("mpz_config")
            tip = payload.get("tip_config")
            if not all(isinstance(item, dict) for item in (front, mpz, tip)):
                continue
            if not _positive(payload.get("b_m")):
                continue
            if not _positive(front.get("L_pz")):
                continue
            if not _positive(tip.get("packet_length_m")):
                continue
            try:
                n_systems = int(mpz.get("n_systems", 2))
            except (TypeError, ValueError):
                continue
            if n_systems < 1:
                continue
            matches.append(
                {
                    "path": path,
                    "b_m": float(payload["b_m"]),
                    "L_pz_um": float(front["L_pz"]) * 1.0e6,
                    "packet_length_um": float(tip["packet_length_m"]) * 1.0e6,
                    "n_systems": n_systems,
                    "transport_mode": payload.get("transport_mode"),
                }
            )

    matches.sort(key=lambda row: str(row["path"]))
    print("MECHANICS_NORMALIZATION_ENGINE_CONFIGS")
    if not matches:
        print("NONE")
        return 1
    for row in matches:
        print(
            f"{row['path']}\t"
            f"L_pz_um={row['L_pz_um']:.12g}\t"
            f"packet_length_um={row['packet_length_um']:.12g}\t"
            f"b_m={row['b_m']:.12g}\t"
            f"n_systems={row['n_systems']}\t"
            f"transport_mode={row['transport_mode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
