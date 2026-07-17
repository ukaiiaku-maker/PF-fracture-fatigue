"""v10.1.9 bounded tip-geometry/source-capacity feedback entry point."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from .geometry_source_feedback_tip import (
    GEOMETRY_SCHEMA,
    GeometrySourceFeedbackTipEngine,
)
from . import sharp_front_v10_1_5 as _campaign


def _nonnegative_env(name: str, default: float = 0.0) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < 0.0:
        raise SystemExit(f"{name} must be non-negative")
    return value


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


GEOMETRY_SOURCE_GAIN = _nonnegative_env("TIP_GEOMETRY_SOURCE_GAIN", 0.0)
GeometrySourceFeedbackTipEngine.configure_campaign(
    _campaign.BACKSTRESS_SCALE,
    _campaign.REFRESH_SCALE,
)
GeometrySourceFeedbackTipEngine.configure_geometry_source_feedback(
    GEOMETRY_SOURCE_GAIN
)
continuum_source_tip.ContinuumSourceKineticTipEngine = GeometrySourceFeedbackTipEngine
_campaign._protected.ContinuumSourceKineticTipEngine = GeometrySourceFeedbackTipEngine


def _rewrite_mode_audits(args: list[str]) -> None:
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
            "schema": "v10.1.9_geometry_source_feedback_driver_modes",
            "developed_state_diagnostics": True,
            "geometry_source_feedback": True,
            "geometry_source_schema": GEOMETRY_SCHEMA,
            "geometry_source_gain": GEOMETRY_SOURCE_GAIN,
            "geometry_reference": "effective radius at first crack advance",
            "first_passage_feedback_disabled": True,
            "temperature_dependent_geometry_parameter": False,
            "wake_primary_toughening_state": False,
        })
        mode_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.9 geometry source feedback: "
        f"gain={GEOMETRY_SOURCE_GAIN:g}, first-passage feedback disabled, "
        "campaign budget and Arrhenius barriers preserved"
    )
    result = _campaign.main(args)
    _rewrite_mode_audits(args)
    return result


if __name__ == "__main__":
    main()
