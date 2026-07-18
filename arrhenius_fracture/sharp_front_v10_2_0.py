"""v10.2.0 entry point: stochastic anisotropic moving-MPZ fatigue.

This entry point reuses the existing v8/v10 2-D cycle-block, cyclic-mechanics,
spatial-field, J-integral, and geometry machinery. It installs a narrow adapter
so fatigue front updates execute through the current moving process-zone engine
rather than the obsolete scalar ``FrontEngine`` ledger.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CLEAVAGE_HAZARD_MODE", "exponential")
os.environ.setdefault("CLEAVAGE_HAZARD_SEED", "1720")
os.environ.setdefault("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
os.environ.setdefault("CLEAVAGE_EVENT_MIN_FACTOR", "0.5")
os.environ.setdefault("CLEAVAGE_EVENT_MAX_FACTOR", "4.0")
os.environ.setdefault("CLEAVAGE_EVENT_SUBSEGMENT_FRACTION", "0.1")
os.environ.setdefault("ANISOTROPIC_USE_AVALANCHE_BACKEND", "1")
os.environ.setdefault("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")

from .fatigue_reintegration_v1020 import (  # noqa: E402
    MODEL_ID,
    install_v1020_fatigue_dispatch,
)
from . import sharp_front_v10_1_7_5 as _transport  # noqa: E402


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _rewrite_audit(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_ID,
        "fatigue_cycles_enabled": "--fatigue-cycles" in args,
        "fatigue_front_dispatch": "native_moving_mpz_cycle_step_waveform",
        "legacy_scalar_front_cycle_step_used_for_v10": False,
        "paris_law_used": False,
        "direct_da_dN_used": False,
        "cleavage_hazard_mode": os.environ["CLEAVAGE_HAZARD_MODE"],
        "cleavage_hazard_seed": int(os.environ["CLEAVAGE_HAZARD_SEED"]),
        "event_length_mode": os.environ["CLEAVAGE_EVENT_LENGTH_MODE"],
        "stochastic_response_default": True,
        "anisotropic_avalanche_backend": os.environ[
            "ANISOTROPIC_USE_AVALANCHE_BACKEND"
        ],
        "anisotropic_transport_mode": os.environ[
            "ANISOTROPIC_TRANSPORT_MODE"
        ],
        "cycle_block_controller": "legacy_adaptive_controller_current_mpz_dispatch",
        "cyclic_mechanics_available": True,
        "cyclic_mechanics_validation_status": "requires_dedicated_stage_2_validation",
    }
    (root / "v10_2_0_fatigue_reintegration.json").write_text(
        json.dumps(payload, indent=2)
    )

    mode_path = root / "v10_1_driver_modes.json"
    if mode_path.exists():
        mode = json.loads(mode_path.read_text())
        mode.update(payload)
        mode["schema"] = "v10.2.0_fatigue_reintegration"
        mode_path.write_text(json.dumps(mode, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--fatigue-cycles" not in args:
        raise SystemExit(
            "v10.2.0 is the fatigue entry point; pass --fatigue-cycles explicitly"
        )
    if "--mode" not in args and not any(token.startswith("--mode=") for token in args):
        args[0:0] = ["--mode", "2d"]

    print(
        "  v10.2.0 fatigue reintegration: "
        f"hazard={os.environ['CLEAVAGE_HAZARD_MODE']} "
        f"event_length={os.environ['CLEAVAGE_EVENT_LENGTH_MODE']} "
        f"seed={os.environ['CLEAVAGE_HAZARD_SEED']} "
        "front_dispatch=native_moving_mpz"
    )
    with install_v1020_fatigue_dispatch():
        result = _transport.main(args)
    _rewrite_audit(args)
    return result


if __name__ == "__main__":
    main()
