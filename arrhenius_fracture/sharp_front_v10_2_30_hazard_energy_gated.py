"""Hazard-derived energy-gated monotonic and cyclic sharp-front entry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _stage3
from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_28 as _entry
from .fatigue_controller_delegate_v10229 import (
    install_engine_native_cycle_preview,
    restore_engine_native_cycle_preview,
)
from .fatigue_driver_cycle_accounting_v10229 import (
    install_consumed_cycle_accounting,
    restore_consumed_cycle_accounting,
)
from .hazard_energy_backend_audit_v10230 import (
    audit_payload as backend_audit_payload,
    install_hazard_energy_backend_audit,
    restore_hazard_energy_backend_audit,
)
from .hazard_energy_differential_engine_v10230 import (
    DifferentialHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from .hazard_energy_observer_v10230 import (
    audit_payload as observer_audit_payload,
    install_observer,
    restore_observer,
)

MODEL_ID = "v10.2.30_hazard_energy_gated_extension"
PersistentSiteStateResolvedTipEngine = (
    DifferentialHazardEnergyGatedPersistentSiteCyclicTipEngine
)


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
        raise SystemExit(f"v10.2.30 requires {option}=0; got {value!r}")


def _prepare_fatigue_args(args: list[str]) -> None:
    _force_toggle(args, "--no-cyclic-mechanics", "--cyclic-mechanics", True)
    _force_toggle(args, "--no-pz-spatial-state", "--pz-spatial-state", True)
    for option in (
        "--pz-recovery-per-s",
        "--pz-mobile-recovery-per-s",
        "--recover-k",
    ):
        _require_zero_value(args, option)


def _fatigue_capable_stage3_validity(original, args: list[str]) -> int:
    fatigue = _remove_toggle(args, "--fatigue-cycles")
    try:
        seed = original(args)
    finally:
        if fatigue:
            args.append("--fatigue-cycles")
    return seed


def _require_single_front(args: list[str]) -> None:
    fronts = _stage3._option_value(args, "--max-fronts", "1")
    if int(fronts) != 1 or _has_option(args, "--crystal-branch"):
        raise SystemExit(
            "v10.2.30 currently supports the validated single-front production "
            "contract only; use --max-fronts 1 and disable branching"
        )
    if _has_option(args, "--allow-abs-directional-J"):
        raise SystemExit(
            "v10.2.30 requires root-signed positive directional J; "
            "--allow-abs-directional-J is forbidden"
        )


def _write_model_audit(args: list[str], fatigue: bool) -> None:
    out = _stage3._option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_ID,
        "base_fatigue_model": "v10.2.29_hazard_cyclic_fatigue_long_growth",
        "base_monotonic_model": "v10.2.28_direct_prescribed_geometry_kernel",
        "loading_mode": "cyclic" if fatigue else "monotonic",
        "single_front_only": True,
        "event_initiation": "Arrhenius_first_passage_only",
        "absolute_athermal_Gc": False,
        "hazard_dissipation_density": (
            "Gamma_haz=gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2"
        ),
        "anisotropic_hazard_scaling": (
            "sigma_hazard=sigma_physical/sqrt(gamma_rel)"
        ),
        "fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2",
        "positive_signed_J_required": True,
        "absolute_directional_J_forbidden": True,
        "event_length_energy_gated_before_MPZ_translation": True,
        "gate_resolution": "every_internal_Strang_microstep",
        "persistent_site_source": True,
        "finite_source_inventory": False,
        "source_depletion": False,
        "source_refresh": False,
        "explicit_recovery": False,
        "engine_native_cycle_predictor": bool(fatigue),
        "legacy_fatigue_barrier_predictor_used": False,
        "observer": observer_audit_payload(),
        "geometry_audit": backend_audit_payload(),
        "engine": PersistentSiteStateResolvedTipEngine.audit_payload(),
    }
    (root / "v10_2_30_hazard_energy_gate_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    fatigue = _has_option(args, "--fatigue-cycles")
    _require_single_front(args)
    if fatigue:
        _prepare_fatigue_args(args)

    original_validity = _stage3._force_stage3_validity_envelope
    original_engine = _paper.PersistentSiteStateResolvedTipEngine
    original_model_id = _entry.MODEL_ID
    _paper.PersistentSiteStateResolvedTipEngine = PersistentSiteStateResolvedTipEngine
    _entry.MODEL_ID = MODEL_ID
    install_observer()
    install_hazard_energy_backend_audit()

    if fatigue:
        _stage3._force_stage3_validity_envelope = (
            lambda a: _fatigue_capable_stage3_validity(original_validity, a)
        )
        install_engine_native_cycle_preview()
        install_consumed_cycle_accounting()

    try:
        result = _entry.main(args)
        _write_model_audit(args, fatigue)
        return result
    finally:
        if fatigue:
            restore_consumed_cycle_accounting()
            restore_engine_native_cycle_preview()
            _stage3._force_stage3_validity_envelope = original_validity
        restore_hazard_energy_backend_audit()
        restore_observer()
        _entry.MODEL_ID = original_model_id
        _paper.PersistentSiteStateResolvedTipEngine = original_engine


if __name__ == "__main__":
    main()
