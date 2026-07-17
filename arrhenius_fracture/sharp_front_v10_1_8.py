"""v10.1.8 protected entry point for forward interaction-zone memory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from .forward_interaction_zone_tip import (
    FORWARD_SCHEMA,
    SOURCE_MODEL,
    ForwardInteractionZoneTipEngine,
)
from . import sharp_front_v10_1_5 as _campaign


def _positive_scale(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0.0:
        raise SystemExit(f"{name} must be positive")
    return value


def _nonnegative_scale(name: str, default: float) -> float:
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


INTERACTION_LENGTH_SCALE = _positive_scale(
    "FORWARD_INTERACTION_LENGTH_SCALE", 1.0
)
RETENTION_SCALE = _nonnegative_scale("FORWARD_RETENTION_SCALE", 1.0)

ForwardInteractionZoneTipEngine.configure_campaign(
    _campaign.BACKSTRESS_SCALE,
    1.0,
)
ForwardInteractionZoneTipEngine.configure_forward_zone(
    INTERACTION_LENGTH_SCALE,
    RETENTION_SCALE,
)
continuum_source_tip.ContinuumSourceKineticTipEngine = ForwardInteractionZoneTipEngine
_campaign._protected.ContinuumSourceKineticTipEngine = ForwardInteractionZoneTipEngine


def _rewrite_forward_zone_audits(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)

    mode_path = root / "v10_1_driver_modes.json"
    if mode_path.exists():
        payload = json.loads(mode_path.read_text())
        payload.update({
            "schema": "v10.1.8_forward_interaction_zone_driver_modes",
            "tip_source_model": "forward_interaction_zone",
            "tip_source_model_internal": SOURCE_MODEL,
            "forward_interaction_schema": FORWARD_SCHEMA,
            "forward_interaction_length_scale": INTERACTION_LENGTH_SCALE,
            "forward_retention_scale": RETENTION_SCALE,
            "forward_total_virgin_source_content": "manifest.source_sites_per_system",
            "forward_source_inflow_boundary": "virgin capacity at far edge",
            "forward_source_outflow_boundary": "material crossing xi=0",
            "scalar_uniform_source_refresh": False,
            "wake_primary_toughening_state": False,
            "temperature_dependent_runtime_source_count": False,
        })
        mode_path.write_text(json.dumps(payload, indent=2))

    source_path = root / "v10_1_1_source_model.json"
    if source_path.exists():
        payload = json.loads(source_path.read_text())
        payload.update({
            "schema": "v10.1.8_forward_interaction_zone_source_model",
            "tip_source_model": "forward_interaction_zone",
            "spatial_state": "available source capacity per system and forward MPZ bin",
            "total_virgin_capacity": "manifest.source_sites_per_system per system",
            "interaction_length_reference": "manifest.source_refresh_length_m",
            "interaction_length_scale": INTERACTION_LENGTH_SCALE,
            "retention_scale": RETENTION_SCALE,
            "source_recovery": "far-boundary virgin-material inflow only",
            "local_emission": "Arrhenius rate at local forward stress minus local Taylor back stress",
            "wake_shielding_required": False,
            "temperature_dependent_runtime_source_count": False,
        })
        source_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.1.8 forward interaction-zone memory: "
        f"length_scale={INTERACTION_LENGTH_SCALE:g}, "
        f"retention_scale={RETENTION_SCALE:g}, "
        "virgin source content fixed by manifest, far-edge inflow only"
    )
    result = _campaign.main(args)
    _rewrite_forward_zone_audits(args)
    return result


if __name__ == "__main__":
    main()
