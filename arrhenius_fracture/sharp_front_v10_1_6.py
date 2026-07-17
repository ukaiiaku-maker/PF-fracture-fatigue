"""v10.1.6.1 entry point for emergent temperature-response validation.

This wrapper intentionally changes no constitutive physics. Source capacity,
source refresh, back-stress scaling, shielding limits, and blunting remain
independent of temperature at runtime. Temperature dependence enters only
through the promoted cleavage, emission, Peierls, Taylor, and recovery kinetics
used by v10.1.5.

The maintenance revision also writes ``campaign_refresh_scale`` as a compatibility
alias for the canonical ``campaign_refresh_length_scale`` driver-audit field.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_1_5 as _campaign


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _write_refresh_scale_alias(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    path = Path(out) / "v10_1_driver_modes.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    canonical = payload.get("campaign_refresh_length_scale")
    if canonical is None:
        return
    payload["campaign_refresh_scale"] = canonical
    payload["matrix_audit_key_compatibility"] = "v10.1.6.1"
    path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.6.1 temperature emergence: runtime source parameters are "
        "temperature-independent; T enters through Arrhenius kinetics only"
    )
    result = _campaign.main(args)
    _write_refresh_scale_alias(args)
    return result


if __name__ == "__main__":
    main()
