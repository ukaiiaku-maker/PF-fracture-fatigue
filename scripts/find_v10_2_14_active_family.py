#!/usr/bin/env python3
"""Locate v10.2.14 active-only signed kernel-family artifacts.

The search is content-based rather than filename-based. Valid results must load
through the production family class and must explicitly authorize production
parameterization. Authorized v10.2.13 extension-only families are reported as
near matches but are never accepted automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.signed_kernel_family_v10213 import SCHEMA as V10213_SCHEMA
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as V10214_SCHEMA,
)


def _json_files(root: Path):
    if root.is_file():
        if root.suffix.lower() == ".json":
            yield root
        return
    if not root.is_dir():
        return
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        type=Path,
        nargs="+",
        help="Files or directories to search recursively.",
    )
    args = parser.parse_args()

    valid: list[Path] = []
    unauthorized: list[tuple[Path, str]] = []
    near_v10213: list[Path] = []
    seen: set[Path] = set()

    for supplied in args.roots:
        root = supplied.expanduser().resolve()
        for path in _json_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            schema = payload.get("schema")
            if schema == V10213_SCHEMA:
                if payload.get("production_parameterization_allowed") is True:
                    near_v10213.append(resolved)
                continue
            if schema != V10214_SCHEMA:
                continue
            try:
                family = ActiveOnlySigned2DShieldingKernelFamily.from_json(resolved)
            except Exception as exc:  # report malformed candidate without aborting search
                unauthorized.append((resolved, f"invalid: {exc}"))
                continue
            if family.metadata.get("production_parameterization_allowed") is not True:
                unauthorized.append(
                    (resolved, "production_parameterization_allowed is not true")
                )
                continue
            valid.append(resolved)

    print("VALID_V10_2_14_ACTIVE_ONLY_PRODUCTION_FAMILIES")
    for path in valid:
        print(path)
    if not valid:
        print("NONE")

    if unauthorized:
        print("\nV10_2_14_NEAR_MATCHES_NOT_AUTHORIZED")
        for path, reason in unauthorized:
            print(f"{path}\t{reason}")

    if near_v10213:
        print("\nAUTHORIZED_V10_2_13_EXTENSION_ONLY_NEAR_MATCHES")
        for path in near_v10213:
            print(path)
        print(
            "These are not accepted by Stage 3 because they have not been "
            "promoted to the v10.2.14 active-only schema."
        )

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
