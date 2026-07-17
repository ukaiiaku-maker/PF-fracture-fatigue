"""Protected v10.1.5 campaign-calibrated tip-source entry point."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from .campaign_calibrated_tip import CampaignCalibratedTipEngine, SOURCE_MODEL


def _scale(name: str, default: float = 1.0) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0.0:
        raise SystemExit(f"{name} must be positive")
    return value


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


BACKSTRESS_SCALE = _scale("CAMPAIGN_BACKSTRESS_SCALE", 1.0)
REFRESH_SCALE = _scale("CAMPAIGN_REFRESH_SCALE", 1.0)
CampaignCalibratedTipEngine.configure_campaign(BACKSTRESS_SCALE, REFRESH_SCALE)

# The protected v10.1 parser already routes --tip-source-model continuum through
# this public symbol. Patch before importing the entry-point module so only this
# process uses the campaign-calibrated implementation.
continuum_source_tip.ContinuumSourceKineticTipEngine = CampaignCalibratedTipEngine

from . import sharp_front_v10_1 as _protected  # noqa: E402

_protected.ContinuumSourceKineticTipEngine = CampaignCalibratedTipEngine


def _rewrite_mode_audits(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    mode_path = root / "v10_1_driver_modes.json"
    if mode_path.exists():
        payload = json.loads(mode_path.read_text())
        payload.update({
            "schema": "v10.1.5_campaign_calibrated_driver_modes",
            "tip_source_model": "campaign_calibrated",
            "tip_source_model_internal": SOURCE_MODEL,
            "source_sites_per_system_role": "promoted_initial_continuum_tip_budget",
            "source_recovery_time": "none_while_stationary",
            "source_recovery_geometry": "crack_advance_over_promoted_refresh_length",
            "campaign_backstress_scale": BACKSTRESS_SCALE,
            "campaign_refresh_length_scale": REFRESH_SCALE,
            "manifest_K_shield_cap_enabled": True,
        })
        mode_path.write_text(json.dumps(payload, indent=2))

    source_path = root / "v10_1_1_source_model.json"
    if source_path.exists():
        payload = json.loads(source_path.read_text())
        payload.update({
            "schema": "v10.1.5_campaign_calibrated_source_model",
            "tip_source_model": "campaign_calibrated",
            "tip_source_state": "bounded continuum capacity per crystallographic system",
            "initial_capacity": "manifest.source_sites_per_system",
            "stationary_time_reactivation": False,
            "activity_recovery_time": "none",
            "activity_recovery_geometry": "manifest.source_refresh_length_m times campaign scale",
            "emission_feedback": "local Taylor back stress from mobile plus retained density",
            "cleavage_shielding_bound": "manifest.max_K_shield_MPa_sqrt_m",
            "campaign_backstress_scale": BACKSTRESS_SCALE,
            "campaign_refresh_length_scale": REFRESH_SCALE,
            "new_dimensional_parameters": 0,
        })
        source_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.5 campaign calibration: "
        f"backstress_scale={BACKSTRESS_SCALE:g}, "
        f"refresh_scale={REFRESH_SCALE:g}, "
        "temporal_source_recycling=0"
    )
    result = _protected.main(args)
    _rewrite_mode_audits(args)
    return result


if __name__ == "__main__":
    main()
