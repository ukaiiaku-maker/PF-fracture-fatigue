"""v10.1.7.2 deterministic-versus-stochastic hazard pilot entry point.

The deterministic mode preserves v10.1.7.1 exactly.  The optional stochastic
mode changes only the integrated cleavage-hazard threshold from one to an
independent unit-mean exponential variate for each completed renewal.  No noise
is added to K, barriers, source capacity, shielding, or material parameters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from .stochastic_hazard_tip import (
    HAZARD_SCHEMA,
    StochasticHazardDiagnosticTipEngine,
)
from . import sharp_front_v10_1_5 as _campaign


HAZARD_MODE = os.environ.get("CLEAVAGE_HAZARD_MODE", "deterministic").strip().lower()
HAZARD_SEED = int(os.environ.get("CLEAVAGE_HAZARD_SEED", "0"))
HAZARD_MIN_THRESHOLD = float(
    os.environ.get("CLEAVAGE_HAZARD_MIN_THRESHOLD", "1e-12")
)

StochasticHazardDiagnosticTipEngine.configure_campaign(
    _campaign.BACKSTRESS_SCALE,
    _campaign.REFRESH_SCALE,
)
StochasticHazardDiagnosticTipEngine.configure_hazard(
    HAZARD_MODE,
    HAZARD_SEED,
    HAZARD_MIN_THRESHOLD,
)
continuum_source_tip.ContinuumSourceKineticTipEngine = (
    StochasticHazardDiagnosticTipEngine
)
_campaign._protected.ContinuumSourceKineticTipEngine = (
    StochasticHazardDiagnosticTipEngine
)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _rewrite_hazard_audits(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    mode_path = root / "v10_1_driver_modes.json"
    if not mode_path.exists():
        return
    payload = json.loads(mode_path.read_text())
    canonical = payload.get("campaign_refresh_length_scale")
    if canonical is not None:
        payload["campaign_refresh_scale"] = canonical
    payload.update({
        "schema": "v10.1.7.2_stochastic_hazard_pilot",
        "developed_state_diagnostics": True,
        "stochastic_hazard_schema": HAZARD_SCHEMA,
        "cleavage_hazard_mode": HAZARD_MODE,
        "cleavage_hazard_seed": HAZARD_SEED,
        "cleavage_hazard_distribution": (
            "delta_at_one" if HAZARD_MODE == "deterministic"
            else "exponential_unit_mean"
        ),
        "constitutive_change_from_v10_1_7_1": False,
        "noise_added_to_K": False,
        "noise_added_to_barriers": False,
        "geometry_source_feedback": False,
        "forward_spatial_source_field": False,
    })
    mode_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.7.2 hazard pilot: mode="
        f"{HAZARD_MODE} seed={HAZARD_SEED}; source/backstress/shielding physics unchanged"
    )
    result = _campaign.main(args)
    _rewrite_hazard_audits(args)
    return result


if __name__ == "__main__":
    main()
