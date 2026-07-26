#!/usr/bin/env python3
"""Run the extended atlas assembler and canonicalize non-finite review metadata.

Only two dimensionless validation diagnostics are permitted to become non-finite
when every mechanically significant response is below the significance floor.
They are provenance fields, not kernel coefficients.  Replace only those values
with JSON null before the family enters canonical hashing; all other non-finite
values remain fatal.
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
_ALLOWED_NONFINITE_KEYS = {
    "maximum_relative_load_variation",
    "maximum_within_load_relative_spread",
}


def _sanitize(value: Any, *, key: str | None = None) -> tuple[Any, int]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        count = 0
        for child_key, child_value in value.items():
            sanitized, child_count = _sanitize(child_value, key=str(child_key))
            output[str(child_key)] = sanitized
            count += child_count
        return output, count
    if isinstance(value, list):
        output = []
        count = 0
        for child_value in value:
            sanitized, child_count = _sanitize(child_value, key=key)
            output.append(sanitized)
            count += child_count
        return output, count
    if isinstance(value, float) and not math.isfinite(value):
        if key not in _ALLOWED_NONFINITE_KEYS:
            raise ValueError(
                "non-finite family value outside permitted review diagnostics: "
                f"key={key!r}, value={value!r}"
            )
        return None, 1
    return value, 0


def _option_path(name: str) -> Path:
    prefix = name + "="
    for index, token in enumerate(sys.argv[1:]):
        if token.startswith(prefix):
            return Path(token[len(prefix):]).expanduser().resolve()
        if token == name:
            arguments = sys.argv[1:]
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
    sanitized, replacements = _sanitize(payload)
    out.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    audit = out.with_name(f"{out.stem}_extended_assembly_audit.json")
    if audit.is_file():
        audit_payload = json.loads(audit.read_text())
        audit_payload["out_sha256"] = _sha256(out)
        audit_payload["nonfinite_review_diagnostics_replaced_with_null"] = replacements
        audit.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )

    print(
        json.dumps(
            {
                "schema": "v10.2.28_finite_extended_atlas_metadata_v1",
                "family": str(out),
                "family_sha256": _sha256(out),
                "nonfinite_review_diagnostics_replaced_with_null": replacements,
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
