#!/usr/bin/env python3
"""Fail closed when a signed-kernel atlas cannot cover a requested crack-growth run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "v10.2.14_active_only_real_signed_2d_shielding_atlas"


def clipped_exponential_mean(minimum_factor: float, maximum_factor: float) -> float:
    a = max(float(minimum_factor), 0.0)
    b = max(float(maximum_factor), a)
    return max(a + math.exp(-a) - math.exp(-b), 1.0e-300)


def state_extensions(payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for index, state in enumerate(payload.get("states", [])):
        try:
            value = float(state["crack_extension_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"kernel state {index} lacks a numeric crack_extension_m"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"kernel state {index} has invalid crack_extension_m={value!r}"
            )
        values.append(value)
    if len(set(values)) < 2:
        raise ValueError("signed-kernel family requires at least two path-extension states")
    return sorted(set(values))


def coverage_report(
    family: Path,
    *,
    target_extension_um: float,
    theta_deg: float,
    da_phys_um: float,
    event_minimum_factor: float,
    event_maximum_factor: float,
    margin_events: float,
) -> dict[str, Any]:
    source = family.expanduser().resolve()
    payload = json.loads(source.read_text())
    schema = payload.get("schema")
    if schema != EXPECTED_SCHEMA:
        raise ValueError(
            f"expected signed-kernel schema {EXPECTED_SCHEMA!r}; got {schema!r}"
        )

    extensions = state_extensions(payload)
    cosine = abs(math.cos(math.radians(float(theta_deg))))
    if cosine <= 1.0e-12:
        raise ValueError("theta places the projected-extension direction at zero cosine")

    target_path_um = float(target_extension_um) / cosine
    normalized_maximum_factor = float(event_maximum_factor) / clipped_exponential_mean(
        event_minimum_factor, event_maximum_factor
    )
    maximum_event_um = float(da_phys_um) * normalized_maximum_factor
    safety_margin_um = float(margin_events) * maximum_event_um
    required_um = target_path_um + safety_margin_um
    atlas_min_um = 1.0e6 * extensions[0]
    atlas_max_um = 1.0e6 * extensions[-1]
    tolerance_um = max(1.0e-6, 1.0e-9 * max(atlas_max_um, required_um, 1.0))
    passed = atlas_max_um + tolerance_um >= required_um

    return {
        "schema": "v10.2.27_signed_kernel_coverage_audit_v1",
        "family": str(source),
        "family_schema": schema,
        "state_count": len(extensions),
        "atlas_min_crack_path_extension_um": atlas_min_um,
        "atlas_max_crack_path_extension_um": atlas_max_um,
        "target_projected_crack_extension_um": float(target_extension_um),
        "fixed_crack_tilt_deg": float(theta_deg),
        "minimum_path_extension_to_target_um": target_path_um,
        "base_checkpoint_um": float(da_phys_um),
        "event_minimum_factor": float(event_minimum_factor),
        "event_maximum_factor": float(event_maximum_factor),
        "normalized_maximum_event_factor": normalized_maximum_factor,
        "maximum_event_advance_um": maximum_event_um,
        "margin_events": float(margin_events),
        "required_atlas_max_crack_path_extension_um": required_um,
        "coverage_margin_um": atlas_max_um - required_um,
        "pass": passed,
        "policy": (
            "strict measured-envelope coverage; no clipping, endpoint hold, or "
            "extrapolation of the signed FEM kernel family"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--da-phys-um", type=float, default=5.0)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = coverage_report(
        args.family,
        target_extension_um=args.target_extension_um,
        theta_deg=args.theta_deg,
        da_phys_um=args.da_phys_um,
        event_minimum_factor=args.event_minimum_factor,
        event_maximum_factor=args.event_maximum_factor,
        margin_events=args.margin_events,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    if not report["pass"]:
        raise SystemExit(
            "ERROR: signed-kernel family does not cover the requested campaign; "
            f"atlas_max={report['atlas_max_crack_path_extension_um']:.6g} um, "
            f"required={report['required_atlas_max_crack_path_extension_um']:.6g} um"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
