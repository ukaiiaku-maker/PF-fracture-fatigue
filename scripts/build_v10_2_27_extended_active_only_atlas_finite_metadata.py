#!/usr/bin/env python3
"""Run the extended atlas assembler and canonicalize inapplicable diagnostics.

The inherited v10.2.14 active-only atlas intentionally measures the exact first
and last MPZ stations. Leave-one-out spatial cross-validation is therefore
mathematically unavailable and is replaced by the audited exact-endpoint
piecewise-linear projection gate. The legacy projection payload represents that
unavailable diagnostic with positive infinity, which cannot enter the canonical
v10.2.27 physics fingerprint.

Only explicitly inapplicable review-diagnostic paths may be replaced with JSON
null. Kernel coefficients, coordinates, normalization values, and all other
non-finite values remain fatal.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas.py"

_ALLOWED_PATH_PATTERNS: tuple[tuple[object, ...], ...] = (
    ("input_states", "*", "maximum_relative_load_variation"),
    ("input_states", "*", "maximum_within_load_relative_spread"),
    (
        "measured_station_projection",
        "maximum_relative_spatial_cross_validation_error",
    ),
    (
        "measured_station_projection",
        "projection_checks",
        "*",
        "maximum_relative_cross_validation_error",
    ),
    (
        "measured_station_projection",
        "projection_checks",
        "*",
        "mode_I_leave_one_out",
        "maximum_relative_error",
    ),
    (
        "measured_station_projection",
        "projection_checks",
        "*",
        "mode_II_leave_one_out",
        "maximum_relative_error",
    ),
)


def _path_allowed(path: tuple[object, ...]) -> bool:
    for pattern in _ALLOWED_PATH_PATTERNS:
        if len(pattern) != len(path):
            continue
        if all(expected == "*" or expected == actual for expected, actual in zip(pattern, path)):
            return True
    return False


def _validate_exact_endpoint_unavailable_cross_validation(payload: dict[str, Any]) -> None:
    if payload.get("spatial_cross_validation_not_required_for_two_endpoint_active_curves") is not True:
        raise ValueError(
            "non-finite spatial diagnostics require the explicit two-endpoint waiver"
        )
    assessment = payload.get("exact_endpoint_projection_assessment", {})
    if not isinstance(assessment, dict) or assessment.get("ready") is not True:
        raise ValueError(
            "non-finite spatial diagnostics require a passed exact-endpoint projection gate"
        )
    gates = payload.get("real_atlas_authorization_gates", {})
    if (
        not isinstance(gates, dict)
        or gates.get("exact_endpoint_piecewise_linear_projection_ready") is not True
    ):
        raise ValueError("exact-endpoint authorization gate is missing or false")

    projection = payload.get("measured_station_projection", {})
    checks = projection.get("projection_checks", []) if isinstance(projection, dict) else []
    if not isinstance(checks, list) or not checks:
        raise ValueError("exact-endpoint projection checks are missing")
    for row in checks:
        if not isinstance(row, dict):
            raise ValueError("invalid exact-endpoint projection check")
        bins = row.get("measured_bins", [])
        full_count = int(row.get("full_grid_count", 0) or 0)
        if (
            row.get("cross_validation_available") is not False
            or not isinstance(bins, list)
            or len(bins) != 2
            or full_count < 2
            or int(bins[0]) != 0
            or int(bins[-1]) != full_count - 1
        ):
            raise ValueError(
                "spatial diagnostic is not the audited unavailable two-endpoint case"
            )
        for mode_name in ("mode_I_leave_one_out", "mode_II_leave_one_out"):
            mode = row.get(mode_name, {})
            if (
                not isinstance(mode, dict)
                or mode.get("available") is not False
                or "at least three measured stations" not in str(mode.get("reason", ""))
            ):
                raise ValueError(
                    f"{mode_name} is not explicitly unavailable for two endpoints"
                )


def _sanitize(
    value: Any,
    *,
    path: tuple[object, ...] = (),
) -> tuple[Any, int]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        count = 0
        for child_key, child_value in value.items():
            child_path = path + (str(child_key),)
            sanitized, child_count = _sanitize(child_value, path=child_path)
            output[str(child_key)] = sanitized
            count += child_count
        return output, count
    if isinstance(value, list):
        output = []
        count = 0
        for index, child_value in enumerate(value):
            sanitized, child_count = _sanitize(
                child_value,
                path=path + (index,),
            )
            output.append(sanitized)
            count += child_count
        return output, count
    if isinstance(value, float) and not math.isfinite(value):
        if not _path_allowed(path):
            raise ValueError(
                "non-finite family value outside permitted review diagnostics: "
                f"path={path!r}, value={value!r}"
            )
        return None, 1
    return value, 0


def _option_path(name: str) -> Path:
    prefix = name + "="
    arguments = sys.argv[1:]
    for index, token in enumerate(arguments):
        if token.startswith(prefix):
            return Path(token[len(prefix):]).expanduser().resolve()
        if token == name:
            if index + 1 >= len(arguments):
                break
            return Path(arguments[index + 1]).expanduser().resolve()
    raise SystemExit(f"missing required option {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    out = _option_path("--out")
    try:
        runpy.run_path(str(ORIGINAL), run_name="__main__")
    except SystemExit as exc:
        code = 0 if exc.code is None else int(exc.code)
        if code != 0:
            raise

    payload = json.loads(out.read_text())
    _validate_exact_endpoint_unavailable_cross_validation(payload)
    sanitized, replacements = _sanitize(payload)
    out.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    audit = out.with_name(f"{out.stem}_extended_assembly_audit.json")
    if audit.is_file():
        audit_payload = json.loads(audit.read_text())
        audit_payload["out_sha256"] = _sha256(out)
        audit_payload["nonfinite_review_diagnostics_replaced_with_null"] = replacements
        audit_payload["exact_endpoint_unavailable_cross_validation_preserved"] = True
        audit.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )

    print(
        json.dumps(
            {
                "schema": "v10.2.28_finite_extended_atlas_metadata_v2",
                "family": str(out),
                "family_sha256": _sha256(out),
                "nonfinite_review_diagnostics_replaced_with_null": replacements,
                "exact_endpoint_unavailable_cross_validation_preserved": True,
                "kernel_coefficients_modified": False,
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
