"""v10.1.7 DBTT developed-state diagnostic entry point.

Constitutive physics is inherited unchanged from v10.1.6.1.  This process only
routes the campaign-calibrated source through a diagnostic subclass that records
cumulative source use, refresh, transport/storage bookkeeping, and population
residence integrals.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import continuum_source_tip
from .developed_state_diagnostic_tip import (
    DIAGNOSTIC_SCHEMA,
    DevelopedStateDiagnosticTipEngine,
)
from . import sharp_front_v10_1_5 as _campaign


DevelopedStateDiagnosticTipEngine.configure_campaign(
    _campaign.BACKSTRESS_SCALE,
    _campaign.REFRESH_SCALE,
)
continuum_source_tip.ContinuumSourceKineticTipEngine = DevelopedStateDiagnosticTipEngine
_campaign._protected.ContinuumSourceKineticTipEngine = DevelopedStateDiagnosticTipEngine


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _rewrite_diagnostic_audits(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    mode_path = root / "v10_1_driver_modes.json"
    if mode_path.exists():
        payload = json.loads(mode_path.read_text())
        canonical = payload.get("campaign_refresh_length_scale")
        if canonical is not None:
            payload["campaign_refresh_scale"] = canonical
        payload.update({
            "schema": "v10.1.7_dbtt_developed_state_driver_modes",
            "developed_state_diagnostics": True,
            "developed_state_diagnostic_schema": DIAGNOSTIC_SCHEMA,
            "constitutive_change_from_v10_1_6_1": False,
        })
        mode_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.7 DBTT developed-state diagnostics: constitutive physics unchanged; "
        "recording emission, refresh, retention, residence, back stress, and shielding"
    )
    result = _campaign.main(args)
    _rewrite_diagnostic_audits(args)
    return result


if __name__ == "__main__":
    main()
