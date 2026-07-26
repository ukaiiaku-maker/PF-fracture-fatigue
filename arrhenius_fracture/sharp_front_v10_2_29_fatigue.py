"""Hazard-based cyclic overlay on the validated v10.2.28 four-class model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _stage3
from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_28 as _entry
from .fatigue_driver_cycle_accounting_v10229 import (
    install_consumed_cycle_accounting,
    restore_consumed_cycle_accounting,
)
from .persistent_site_cyclic_v10229 import PersistentSiteCyclicTipEngine

MODEL_ID = "v10.2.29_hazard_cyclic_fatigue_long_growth"
PersistentSiteStateResolvedTipEngine = PersistentSiteCyclicTipEngine


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _remove_toggle(args: list[str], name: str) -> bool:
    found = False
    kept: list[str] = []
    for token in args:
        if token == name:
            found = True
        else:
            kept.append(token)
    args[:] = kept
    return found


def _force_toggle(args: list[str], positive: str, negative: str, enabled: bool) -> None:
    args[:] = [token for token in args if token not in {positive, negative}]
    args.append(positive if enabled else negative)


def _require_zero_value(args: list[str], option: str) -> None:
    value = _stage3._option_value(args, option)
    if value is not None and abs(float(value)) > 1.0e-30:
        raise SystemExit(f"v10.2.29 requires {option}=0; got {value!r}")


def _prepare_fatigue_args(args: list[str]) -> None:
    # The current paper mechanics have an elastic surrounding bulk and one signed
    # moving tip process zone. Disable the older duplicate full-field cyclic and
    # scalar/spatial fatigue-PZ adapters rather than evolving two independent states.
    _force_toggle(args, "--no-cyclic-mechanics", "--cyclic-mechanics", True)
    _force_toggle(args, "--no-pz-spatial-state", "--pz-spatial-state", True)
    for option in (
        "--pz-recovery-per-s",
        "--pz-mobile-recovery-per-s",
        "--recover-k",
    ):
        _require_zero_value(args, option)


def _fatigue_capable_stage3_validity(original, args: list[str]) -> int:
    """Run the original validity envelope with only the fatigue flag hidden."""
    fatigue = _remove_toggle(args, "--fatigue-cycles")
    try:
        seed = original(args)
    finally:
        if fatigue:
            args.append("--fatigue-cycles")
    return seed


def _write_fatigue_audit(args: list[str]) -> None:
    out = _stage3._option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_ID,
        "base_entry": "arrhenius_fracture.sharp_front_v10_2_28",
        "parameter_registry": str(_paper.DEFAULT_REGISTRY),
        "canonical_options": dict(_paper.VALID_OPTIONS),
        "parameter_refit": False,
        "persistent_site_source": True,
        "finite_source_inventory": False,
        "source_depletion": False,
        "source_refresh": False,
        "explicit_recovery": False,
        "engine_native_cycle_predictor": True,
        "legacy_fatigue_barrier_predictor_used": False,
        "duplicate_spatial_fatigue_state": False,
        "full_field_cyclic_mechanics": False,
        "consumed_cycle_accounting": True,
        "one_outer_geometry_event_per_fem_state": True,
    }
    (root / "v10_2_29_fatigue_model_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--fatigue-cycles"):
        # Exact monotonic path: no v10.2.29 monkeypatches or state substitutions.
        return _entry.main(args)

    _prepare_fatigue_args(args)
    original_validity = _stage3._force_stage3_validity_envelope
    original_engine = _paper.PersistentSiteStateResolvedTipEngine
    original_model_id = _entry.MODEL_ID
    _stage3._force_stage3_validity_envelope = lambda a: _fatigue_capable_stage3_validity(
        original_validity, a
    )
    _paper.PersistentSiteStateResolvedTipEngine = PersistentSiteStateResolvedTipEngine
    _entry.MODEL_ID = MODEL_ID
    install_consumed_cycle_accounting()
    try:
        result = _entry.main(args)
        _write_fatigue_audit(args)
        return result
    finally:
        restore_consumed_cycle_accounting()
        _entry.MODEL_ID = original_model_id
        _paper.PersistentSiteStateResolvedTipEngine = original_engine
        _stage3._force_stage3_validity_envelope = original_validity


if __name__ == "__main__":
    main()
