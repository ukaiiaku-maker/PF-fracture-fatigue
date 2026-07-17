"""v10.1.7.3 threshold-correlated stochastic avalanche-length pilot.

This version retains the v10.1.7.1 material/source/back-stress/shielding model and
the v10.1.7.2 stochastic integrated-hazard threshold.  It adds an opt-in event
reward: the stochastic threshold sets the total crack advance associated with
that renewal, while a segmented sharp-wake wrapper realizes the event geometry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from . import sharp_front as _sharp_front_base
from .stochastic_avalanche_backend import build_avalanche_backend
from .stochastic_avalanche_tip import (
    AVALANCHE_SCHEMA,
    StochasticAvalancheDiagnosticTipEngine,
)
from . import sharp_front_v10_1_5 as _campaign


HAZARD_MODE = os.environ.get("CLEAVAGE_HAZARD_MODE", "deterministic").strip().lower()
HAZARD_SEED = int(os.environ.get("CLEAVAGE_HAZARD_SEED", "0"))
HAZARD_MIN_THRESHOLD = float(
    os.environ.get("CLEAVAGE_HAZARD_MIN_THRESHOLD", "1e-12")
)
EVENT_LENGTH_MODE = os.environ.get(
    "CLEAVAGE_EVENT_LENGTH_MODE", "fixed"
).strip().lower()
EVENT_MIN_FACTOR = float(os.environ.get("CLEAVAGE_EVENT_MIN_FACTOR", "0.2"))
EVENT_MAX_FACTOR = float(os.environ.get("CLEAVAGE_EVENT_MAX_FACTOR", "4.0"))
EVENT_SUBSEGMENT_FRACTION = float(
    os.environ.get("CLEAVAGE_EVENT_SUBSEGMENT_FRACTION", "0.1")
)

StochasticAvalancheDiagnosticTipEngine.configure_campaign(
    _campaign.BACKSTRESS_SCALE,
    _campaign.REFRESH_SCALE,
)
StochasticAvalancheDiagnosticTipEngine.configure_hazard(
    HAZARD_MODE,
    HAZARD_SEED,
    HAZARD_MIN_THRESHOLD,
)
StochasticAvalancheDiagnosticTipEngine.configure_avalanche(
    EVENT_LENGTH_MODE,
    EVENT_MIN_FACTOR,
    EVENT_MAX_FACTOR,
    EVENT_SUBSEGMENT_FRACTION,
)
continuum_source_tip.ContinuumSourceKineticTipEngine = (
    StochasticAvalancheDiagnosticTipEngine
)
_campaign._protected.ContinuumSourceKineticTipEngine = (
    StochasticAvalancheDiagnosticTipEngine
)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _rewrite_audits(args: list[str]) -> None:
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
        "schema": "v10.1.7.3_stochastic_avalanche_length_pilot",
        "developed_state_diagnostics": True,
        "stochastic_avalanche_schema": AVALANCHE_SCHEMA,
        "cleavage_hazard_mode": HAZARD_MODE,
        "cleavage_hazard_seed": HAZARD_SEED,
        "cleavage_event_length_mode": EVENT_LENGTH_MODE,
        "cleavage_event_min_factor": EVENT_MIN_FACTOR,
        "cleavage_event_max_factor": EVENT_MAX_FACTOR,
        "cleavage_event_subsegment_fraction": EVENT_SUBSEGMENT_FRACTION,
        "mean_event_length_preserved": True,
        "geometry_subsegments_re_equilibrated": False,
        "constitutive_material_change_from_v10_1_7_1": False,
        "stochastic_geometry_reward_change_from_v10_1_7_2": True,
        "noise_added_to_K": False,
        "noise_added_to_barriers": False,
        "geometry_source_feedback": False,
        "forward_spatial_source_field": False,
    })
    mode_path.write_text(json.dumps(payload, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_builder = _sharp_front_base.build_crack_backend

    def _builder(local_args, geom):
        return build_avalanche_backend(
            local_args,
            geom,
            original_builder,
            default_subsegment_fraction=EVENT_SUBSEGMENT_FRACTION,
        )

    _sharp_front_base.build_crack_backend = _builder
    try:
        print(
            "  v10.1.7.3 avalanche pilot: "
            f"hazard={HAZARD_MODE} seed={HAZARD_SEED} "
            f"event_length={EVENT_LENGTH_MODE} bounds="
            f"[{EVENT_MIN_FACTOR:g},{EVENT_MAX_FACTOR:g}] "
            f"subsegment_fraction={EVENT_SUBSEGMENT_FRACTION:g}"
        )
        result = _campaign.main(args)
        _rewrite_audits(args)
        return result
    finally:
        _sharp_front_base.build_crack_backend = original_builder


if __name__ == "__main__":
    main()
