"""v10.1.7.5 deterministic reduced-candidate transfer gate.

This diagnostic entry point preserves the v10.1.7.4 tensor-resolved anisotropic
emission implementation while exposing two zero-valued ablations that the
production defaults intentionally do not permit:

* ``V10175_BACKSTRESS_SCALE=0`` disables the crack-tip Taylor back stress;
* ``V10175_FOREST_DENSITY_FLOOR_M2=0`` removes the imposed MPZ forest-density
  floor while preserving emission-generated mobile/retained populations and
  their Peierls--Taylor transport.

The gate is deterministic, fixed-event, single-front, tip-plasticity-only, and
uses the original non-avalanche geometry transaction.  No production default or
constitutive parameter is changed outside this process.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from . import sharp_front_v10_1_7_4 as _base
from . import unified_mpz as _unified_mpz


SCHEMA = "v10.1.7.5_reduced_candidate_transfer_gate"
BACKSTRESS_ENV = "V10175_BACKSTRESS_SCALE"
FOREST_ENV = "V10175_FOREST_DENSITY_FLOOR_M2"


def _option_value(args: Iterable[str], name: str, default: str | None = None) -> str | None:
    values = list(args)
    prefix = name + "="
    for index, token in enumerate(values):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(values):
            return values[index + 1]
    return default


def _env_nonnegative(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value < 0.0:
        raise SystemExit(f"{name} must be nonnegative")
    return value


def _env_false(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return not bool(default)
    return raw.strip().lower() in {"0", "false", "no", "off"}


def _require_transfer_scope(args: list[str]) -> None:
    bulk = _option_value(args, "--bulk-plasticity-mode", "tip_only")
    if str(bulk).strip().lower() != "tip_only":
        raise SystemExit("v10.1.7.5 requires --bulk-plasticity-mode tip_only")
    max_fronts = int(_option_value(args, "--max-fronts", "32"))
    if max_fronts != 1 or "--crystal-branch" in args:
        raise SystemExit(
            "v10.1.7.5 is a single-front transfer gate; use --max-fronts 1 "
            "and do not enable --crystal-branch"
        )
    if "--no-wake-shielding" not in args:
        raise SystemExit("v10.1.7.5 requires --no-wake-shielding")
    if not _env_false("ANISOTROPIC_USE_AVALANCHE_BACKEND", default=False):
        raise SystemExit(
            "v10.1.7.5 requires ANISOTROPIC_USE_AVALANCHE_BACKEND=0"
        )
    if os.environ.get("CLEAVAGE_HAZARD_MODE", "deterministic").strip().lower() != "deterministic":
        raise SystemExit("v10.1.7.5 requires CLEAVAGE_HAZARD_MODE=deterministic")
    if os.environ.get("CLEAVAGE_EVENT_LENGTH_MODE", "fixed").strip().lower() != "fixed":
        raise SystemExit("v10.1.7.5 requires CLEAVAGE_EVENT_LENGTH_MODE=fixed")


def _write_transfer_audit(
    args: list[str],
    *,
    backstress_scale: float,
    forest_floor_override_m2: float | None,
) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "deterministic_hazard": True,
        "fixed_event_length": True,
        "anisotropic_avalanche_backend": False,
        "single_front": True,
        "bulk_plasticity_mode": "tip_only",
        "wake_shielding": False,
        "backstress_scale": float(backstress_scale),
        "forest_density_floor_override_m2": (
            None
            if forest_floor_override_m2 is None
            else float(forest_floor_override_m2)
        ),
        "background_field_off": forest_floor_override_m2 == 0.0,
        "production_defaults_modified": False,
    }
    (root / "v10_1_7_5_transfer_gate.json").write_text(
        json.dumps(payload, indent=2)
    )

    mode_path = root / "v10_1_driver_modes.json"
    if mode_path.exists():
        mode = json.loads(mode_path.read_text())
        mode.update(
            {
                "transfer_gate_schema": SCHEMA,
                "transfer_gate_backstress_scale": float(backstress_scale),
                "transfer_gate_forest_density_floor_override_m2": (
                    None
                    if forest_floor_override_m2 is None
                    else float(forest_floor_override_m2)
                ),
                "transfer_gate_background_field_off": forest_floor_override_m2 == 0.0,
                "transfer_gate_deterministic_fixed_event": True,
            }
        )
        mode_path.write_text(json.dumps(mode, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    _require_transfer_scope(args)

    original_backstress = float(_base._campaign.BACKSTRESS_SCALE)
    backstress = _env_nonnegative(BACKSTRESS_ENV, original_backstress)
    assert backstress is not None
    forest_floor = _env_nonnegative(FOREST_ENV, None)

    original_init = _unified_mpz.UnifiedMPZState.__init__

    def transfer_init(self, manifest, cfg):
        local_cfg = copy.deepcopy(cfg)
        if forest_floor is not None:
            local_cfg.forest_density_floor_m2 = float(forest_floor)
        original_init(self, manifest, local_cfg)
        self._v10175_forest_density_floor_override_m2 = forest_floor

    _base._campaign.BACKSTRESS_SCALE = float(backstress)
    _unified_mpz.UnifiedMPZState.__init__ = transfer_init
    try:
        print(
            "  v10.1.7.5 transfer gate: "
            f"backstress_scale={backstress:g}, "
            "forest_floor_m2="
            + ("manifest/default" if forest_floor is None else f"{forest_floor:g}")
            + ", deterministic=1, avalanche_backend=0"
        )
        result = _base.main(args)
        _write_transfer_audit(
            args,
            backstress_scale=float(backstress),
            forest_floor_override_m2=forest_floor,
        )
        return result
    finally:
        _base._campaign.BACKSTRESS_SCALE = original_backstress
        _unified_mpz.UnifiedMPZState.__init__ = original_init


if __name__ == "__main__":
    main()
