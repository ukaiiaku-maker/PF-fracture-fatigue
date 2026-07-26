#!/usr/bin/env python3
"""Validate the bootstrap signed kernel used for accepted production capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SCHEMA = "v10.2.14_active_only_real_signed_2d_shielding_atlas"
REQUIRED_TRUTHS = {
    "production_parameterization_allowed": True,
    "campaign_parameterization_allowed": True,
    "active_kernel_mechanically_measured": True,
    "candidate_independent": True,
    "same_kernel_family_for_monotonic_and_fatigue": True,
    "frozen_geometry_load_invariance_passed": True,
    "normalization_is_mechanically_derived": True,
    "positive_and_negative_perturbations": True,
    "multi_amplitude_validation_passed": True,
    "wake_shielding_supported": False,
    "wake_kernel_forced_zero": True,
    "wake_kernel_mechanically_measured": False,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--required-path-extension-um", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    family = args.family.expanduser().resolve()
    required_um = float(args.required_path_extension_um)
    if not family.is_file():
        raise FileNotFoundError(family)
    if not math.isfinite(required_um) or required_um <= 0.0:
        raise SystemExit("required bootstrap path extension must be positive and finite")

    payload = json.loads(family.read_text())
    failures = []
    if payload.get("schema") != EXPECTED_SCHEMA:
        failures.append("family_schema")
    for key, expected in REQUIRED_TRUTHS.items():
        if payload.get(key) is not expected:
            failures.append(key)

    extensions = []
    state_ids = []
    for index, state in enumerate(payload.get("states", [])):
        state_ids.append(str(state.get("state_id", f"state_{index}")))
        try:
            extension = float(state["crack_extension_m"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"state_{index}_crack_extension_m")
            continue
        if not math.isfinite(extension) or extension < 0.0:
            failures.append(f"state_{index}_crack_extension_m")
            continue
        extensions.append(extension)
    if len(set(extensions)) < 2:
        failures.append("state_extension_count")
        minimum_um = math.nan
        maximum_um = math.nan
    else:
        minimum_um = 1.0e6 * min(extensions)
        maximum_um = 1.0e6 * max(extensions)
        tolerance_um = max(1.0e-6, 1.0e-9 * max(required_um, maximum_um, 1.0))
        if minimum_um > tolerance_um:
            failures.append("zero_extension_coverage")
        if maximum_um + tolerance_um < required_um:
            failures.append("required_path_extension_coverage")

    result = {
        "schema": "v10.2.27_capture_seed_family_audit_v2",
        "family": str(family),
        "family_sha256": hashlib.sha256(family.read_bytes()).hexdigest(),
        "family_schema": payload.get("schema"),
        "required_authorization_gates": REQUIRED_TRUTHS,
        "observed_authorization_gates": {
            key: payload.get(key) for key in REQUIRED_TRUTHS
        },
        "state_ids": state_ids,
        "state_count": len(extensions),
        "minimum_path_extension_um": minimum_um,
        "maximum_path_extension_um": maximum_um,
        "required_path_extension_um": required_um,
        "passed": not failures,
        "failures": sorted(set(failures)),
        "policy": (
            "bootstrap family is an explicit production-authorized shielding operator "
            "used only to evolve accepted v10.2.27 trajectory states; it does not "
            "define the target measurement mesh or target mechanical fingerprint"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else family.parent / "capture_seed_family_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    if failures:
        raise SystemExit(
            "capture bootstrap family failed validation: "
            + ",".join(result["failures"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
