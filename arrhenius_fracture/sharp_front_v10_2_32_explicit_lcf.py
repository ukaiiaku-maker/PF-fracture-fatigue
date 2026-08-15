"""Fixed-DeltaK explicit-cycle LCF entry for A--D and canonical rows."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_30_energy_gated_fatigue as _energy
from . import sharp_front_v10_2_30_fixed_deltaK as _fixed
from . import sharp_front_v10_2_31_endurance_knee as _mapped
from . import persistent_site_forward_selector_v10230 as _selector
from .persistent_site_explicit_cycle_v1032 import (
    ExplicitCycleHazardEnergyGatedPersistentSiteCyclicTipEngine,
    MODEL_ID,
    select_explicit_cycle_block,
)


ENTRY_ID = "v10.2.32_fixed_deltaK_explicit_cycle_lcf_v1"
_PRODUCTION_ENERGY_MAIN = _energy.main


def _pop_mode(args: list[str]) -> str:
    for index, token in enumerate(list(args)):
        if token == "--cycle-integration-mode":
            if index + 1 >= len(args):
                raise SystemExit("--cycle-integration-mode requires explicit")
            value = args[index + 1]
            del args[index:index + 2]
            return value
        if token.startswith("--cycle-integration-mode="):
            value = token.split("=", 1)[1]
            args.remove(token)
            return value
    return "explicit"


def _option(args: list[str], name: str) -> str:
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return ""


def _explicit_energy_main(args):
    original_engine = _energy.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    original_selector = _selector.select_nonlinear_block
    _energy.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine = (
        ExplicitCycleHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    _selector.select_nonlinear_block = select_explicit_cycle_block
    try:
        return _PRODUCTION_ENERGY_MAIN(args)
    finally:
        _selector.select_nonlinear_block = original_selector
        _energy.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine = original_engine


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    mode = _pop_mode(args).strip().lower()
    if mode != "explicit":
        raise SystemExit("v10.2.32 explicit LCF entry accepts only explicit mode")
    option = _option(args, "--parameter-option")
    original_fixed_energy = _fixed._energy.main
    original_mapped_production = _mapped._PRODUCTION_MAIN
    if option.startswith("v914_endurance_knee_"):
        _mapped._PRODUCTION_MAIN = _explicit_energy_main
        routed = _mapped.main
    else:
        routed = _explicit_energy_main
    _fixed._energy.main = routed
    try:
        result = _fixed.main(args)
    finally:
        _fixed._energy.main = original_fixed_energy
        _mapped._PRODUCTION_MAIN = original_mapped_production
    out = _option(args, "--out")
    if out:
        payload = {
            "schema": ENTRY_ID,
            "integration_model_id": MODEL_ID,
            "cycle_integration_mode": "explicit",
            "multi_cycle_projection": False,
            "same_cycle_post_event_continuation": True,
            "energy_gate_event_load": "cycle_Kmax",
            "constitutive_physics_changed": False,
            "parameter_refit": False,
        }
        (Path(out) / "v10_2_32_explicit_cycle_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return result


if __name__ == "__main__":
    main()
