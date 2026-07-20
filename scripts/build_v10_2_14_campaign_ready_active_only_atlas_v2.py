#!/usr/bin/env python3
"""Build the Stage 3 active-only atlas using exact endpoint projection.

The completed v10.2.14 active-only responses measure the first and last active
MPZ stations exactly and use piecewise-linear interpolation between them. With
only two measured active stations per curve, leave-one-out cross-validation is
mathematically unavailable; that legacy diagnostic must not block the Stage 3
campaign. This wrapper keeps every other mechanics gate and replaces only that
inapplicable diagnostic with an explicit exact-endpoint projection gate.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "build_v10_2_14_campaign_ready_active_only_atlas.py"
SPEC = importlib.util.spec_from_file_location("v10214_campaign_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load campaign builder from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

ENDPOINT_GATE = "exact_endpoint_piecewise_linear_projection_ready"
BASE.MECHANICAL_GATES = tuple(
    name
    for name in BASE.MECHANICAL_GATES
    if name != "spatial_projection_cross_validation_passed"
) + (ENDPOINT_GATE,)
_ORIGINAL_PROMOTE = BASE._promote


def exact_endpoint_projection_ready(payload: dict) -> tuple[bool, dict]:
    projection = payload.get("measured_station_projection", {})
    checks = projection.get("projection_checks", [])
    active_checks = [
        row
        for row in checks
        if isinstance(row, dict) and str(row.get("region", "")).lower() == "active"
    ]
    details = {
        "piecewise_linear_spatial_projection": bool(
            projection.get("piecewise_linear_spatial_projection") is True
        ),
        "active_kernel_mechanically_measured": bool(
            projection.get("active_kernel_mechanically_measured") is True
        ),
        "wake_kernel_mechanically_measured": bool(
            projection.get("wake_kernel_mechanically_measured") is True
        ),
        "wake_shielding_supported": bool(
            projection.get("wake_shielding_supported") is True
        ),
        "active_projection_check_count": len(active_checks),
        "all_active_curves_have_exact_endpoint_coverage": False,
    }
    endpoint_coverage = bool(active_checks)
    for row in active_checks:
        bins = row.get("measured_bins", [])
        full_count = int(row.get("full_grid_count", 0) or 0)
        valid = (
            isinstance(bins, list)
            and len(bins) >= 2
            and full_count >= 2
            and int(bins[0]) == 0
            and int(bins[-1]) == full_count - 1
        )
        endpoint_coverage = endpoint_coverage and valid
    details["all_active_curves_have_exact_endpoint_coverage"] = endpoint_coverage
    ready = bool(
        details["piecewise_linear_spatial_projection"]
        and details["active_kernel_mechanically_measured"]
        and not details["wake_kernel_mechanically_measured"]
        and not details["wake_shielding_supported"]
        and endpoint_coverage
        and projection.get("projected_schema_matches_measured_schema") is True
        and projection.get("subelement_rows_claimed_as_direct_fem") is False
    )
    details["projected_schema_matches_measured_schema"] = bool(
        projection.get("projected_schema_matches_measured_schema") is True
    )
    details["subelement_rows_not_claimed_as_direct_fem"] = bool(
        projection.get("subelement_rows_claimed_as_direct_fem") is False
    )
    details["ready"] = ready
    return ready, details


def _promote(source_out: Path, out: Path, metadata: dict) -> None:
    payload = json.loads(source_out.read_text())
    ready, details = exact_endpoint_projection_ready(payload)
    gates = dict(payload.get("real_atlas_authorization_gates", {}))
    gates[ENDPOINT_GATE] = ready
    payload["real_atlas_authorization_gates"] = gates
    payload["exact_endpoint_projection_assessment"] = details
    payload["spatial_cross_validation_not_required_for_two_endpoint_active_curves"] = True
    source_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not ready:
        raise SystemExit(
            "completed active-only mechanics lack exact endpoint projection coverage: "
            + json.dumps(details, sort_keys=True)
        )
    _ORIGINAL_PROMOTE(source_out, out, metadata)


BASE._promote = _promote

if __name__ == "__main__":
    BASE.main()
