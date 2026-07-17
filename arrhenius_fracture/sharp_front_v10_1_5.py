"""Protected v10.1.5 campaign-calibrated tip-source entry point."""
from __future__ import annotations

import os
import sys

from . import continuum_source_tip
from .campaign_calibrated_tip import CampaignCalibratedTipEngine


def _scale(name: str, default: float = 1.0) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0.0:
        raise SystemExit(f"{name} must be positive")
    return value


BACKSTRESS_SCALE = _scale("CAMPAIGN_BACKSTRESS_SCALE", 1.0)
REFRESH_SCALE = _scale("CAMPAIGN_REFRESH_SCALE", 1.0)
CampaignCalibratedTipEngine.configure_campaign(BACKSTRESS_SCALE, REFRESH_SCALE)

# The protected v10.1 parser already routes --tip-source-model continuum through
# this public symbol. Patch before importing the entry-point module so only this
# process uses the campaign-calibrated implementation.
continuum_source_tip.ContinuumSourceKineticTipEngine = CampaignCalibratedTipEngine

from . import sharp_front_v10_1 as _protected  # noqa: E402

_protected.ContinuumSourceKineticTipEngine = CampaignCalibratedTipEngine


def main(argv=None):
    print(
        "  v10.1.5 campaign calibration: "
        f"backstress_scale={BACKSTRESS_SCALE:g}, "
        f"refresh_scale={REFRESH_SCALE:g}, "
        "temporal_source_recycling=0"
    )
    return _protected.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
