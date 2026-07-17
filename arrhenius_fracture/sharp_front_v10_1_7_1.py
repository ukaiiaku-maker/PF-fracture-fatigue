"""v10.1.7.1 final three-class temperature-sweep entry point.

Constitutive physics is inherited unchanged from v10.1.7.  This wrapper only
labels final production runs and preserves the developed-state diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_1_7 as _diagnostic


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _rewrite_production_audits(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    mode_path = root / "v10_1_driver_modes.json"
    if not mode_path.exists():
        return
    payload = json.loads(mode_path.read_text())
    payload.update({
        "schema": "v10.1.7.1_final_production_temperature_sweep",
        "final_production_sweep": True,
        "production_classes": ["ceramic", "weakT", "DBTT"],
        "production_temperature_range_K": [300, 1100],
        "production_temperature_increment_K": 100,
        "production_target_crack_extension_um": 500,
        "constitutive_change_from_v10_1_7": False,
        "geometry_source_feedback": False,
        "forward_spatial_source_field": False,
    })
    mode_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.7.1 final production sweep: v10.1.7 constitutive physics unchanged; "
        "developed-state diagnostics retained"
    )
    result = _diagnostic.main(args)
    _rewrite_production_audits(args)
    return result


if __name__ == "__main__":
    main()
