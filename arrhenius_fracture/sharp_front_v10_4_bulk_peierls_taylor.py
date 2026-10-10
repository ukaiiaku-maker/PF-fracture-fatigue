"""v10.4 experimental full-field bulk Peierls--Taylor coupling.

This entry preserves the validated v10.2.30 single-front hazard-energy gate and
persistent-site moving MPZ. It changes only the surrounding FEM bulk from the
transactional tip-only no-op to the exact selected-row Arrhenius
Peierls--Taylor constitutive update.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_1 as _v101
from . import sharp_front_v10_2_17 as _stage3
from . import sharp_front_v10_2_30_hazard_energy_gated as _base
from .bulk_plasticity_manifest_v104 import (
    BulkManifestParameters,
    BulkPlasticityCoupling,
    MODEL_ID as BULK_MODEL_ID,
)

MODEL_ID = "v10.4.0_bulk_peierls_taylor_coupled_hazard_energy_gate"


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _remove_toggle(args: list[str], name: str) -> None:
    args[:] = [token for token in args if token != name]


def _force_value(args: list[str], name: str, value: object) -> None:
    _stage3._set_value_option(args, name, value)


def _prepare_v104_args(args: list[str]) -> None:
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.4.0 bulk coupling is monotonic-only")
    if _has_option(args, "--exhaustion"):
        raise SystemExit("v10.4.0 does not permit finite bulk-content exhaustion")
    _force_value(args, "--bulk-plasticity-mode", "full_field")
    _force_value(args, "--bulk-mult-frac", 1.0)
    _force_value(args, "--tip-source-rho-per-emit", 0.0)
    _force_value(args, "--rho-transport-c", 0.0)


def _bulk_capable_stage3_validity(original, args: list[str]) -> int:
    supplied = _stage3._option_value(args, "--bulk-plasticity-mode", "full_field")
    if str(supplied).strip().lower() != "full_field":
        raise SystemExit("v10.4.0 requires --bulk-plasticity-mode full_field")
    _force_value(args, "--bulk-plasticity-mode", "tip_only")
    seed = original(args)
    _force_value(args, "--bulk-plasticity-mode", "full_field")
    return seed


def _rewrite_driver_audit(args: list[str]) -> None:
    out = _v101._option_value(args, "--out")
    if not out:
        return
    path = Path(out) / "v10_1_driver_modes.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "bulk_plasticity_mode": "full_field",
            "legacy_full_field_enabled": False,
            "manifest_mapped_full_field_enabled": True,
            "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
            "bulk_thermodynamic_projection": "local_time_cone",
            "tip_and_bulk_source_populations_distinct": True,
            "direct_tip_to_bulk_density_transfer": False,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _full_field_v101_main(argv=None):
    args, bulk_mode, j_mode, kinetics_mode, tip_cfg, source_model = (
        _v101._prepare_args_v1011(sys.argv[1:] if argv is None else argv)
    )
    if bulk_mode != "full_field":
        raise SystemExit("v10.4.0 internal driver requires full_field bulk plasticity")
    manifest_path = _v101._option_value(args, "--material-manifest")
    if not manifest_path:
        raise SystemExit("v10.4.0 requires the exact selected material manifest")

    parameters = BulkManifestParameters.from_csv(manifest_path)
    coupling = BulkPlasticityCoupling(parameters)

    _force_value(args, "--rho0", parameters.rho0_m2)
    _force_value(args, "--bulk-mult-frac", 1.0)
    _force_value(args, "--tip-source-rho-per-emit", 0.0)
    _force_value(args, "--rho-transport-c", 0.0)
    _remove_toggle(args, "--exhaustion")

    engine_cls = (
        _v101.ContinuumSourceKineticTipEngine
        if source_model == "continuum"
        else _v101.KineticMovingTipFrontEngine
    )
    original_update = _v101.plasticity.update_plasticity
    original_diag = _v101.UnifiedMPZState.diagnostics
    original_advance = _v101.UnifiedMPZState.advance
    original_engine = _v101.sharp_front.UnifiedMPZFrontEngine
    had_shield_cfg = hasattr(_v101.UnifiedMPZState, "_v101_shield_cfg")
    old_shield_cfg = getattr(_v101.UnifiedMPZState, "_v101_shield_cfg", None)
    try:
        _v101.plasticity.update_plasticity = coupling.wrap(original_update)
        _v101.UnifiedMPZState.diagnostics = _v101._diagnostics_with_csv_aliases
        if kinetics_mode == "moving_velocity":
            _v101.UnifiedMPZState.advance = _v101.fractional_moving_frame_advance
            _v101.UnifiedMPZState._v101_shield_cfg = tip_cfg
            engine_cls.configure_default(tip_cfg)
            engine_cls.reset_audit()
            _v101.sharp_front.UnifiedMPZFrontEngine = engine_cls
        wake_mode = _v101._resolved_wake_shielding(args)
        print(
            "  v10.4.0 driving modes: "
            f"bulk_plasticity={bulk_mode} directional_J={j_mode} "
            f"tip_kinetics={kinetics_mode} tip_source_model={source_model} "
            f"tip_plasticity={int(tip_cfg.plasticity_enabled)} "
            f"active_shielding={int(tip_cfg.active_shielding)} "
            f"wake_shielding={int(wake_mode)} "
            f"bulk_row={parameters.option_key}"
        )
        result = _v101.sharp_front.main(args)
        _v101._write_mode_audit(
            args, bulk_mode, j_mode, kinetics_mode, tip_cfg, source_model, engine_cls
        )
        _rewrite_driver_audit(args)
        out = _v101._option_value(args, "--out")
        if out:
            coupling.write_audit(out)
        return result
    finally:
        _v101.plasticity.update_plasticity = original_update
        _v101.UnifiedMPZState.diagnostics = original_diag
        _v101.UnifiedMPZState.advance = original_advance
        _v101.sharp_front.UnifiedMPZFrontEngine = original_engine
        if had_shield_cfg:
            _v101.UnifiedMPZState._v101_shield_cfg = old_shield_cfg
        elif hasattr(_v101.UnifiedMPZState, "_v101_shield_cfg"):
            delattr(_v101.UnifiedMPZState, "_v101_shield_cfg")


def _write_model_audit(args: list[str]) -> None:
    out = _stage3._option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    base_path = root / "v10_2_30_hazard_energy_gate_audit.json"
    base = json.loads(base_path.read_text()) if base_path.is_file() else {}
    payload = {
        "schema": MODEL_ID,
        "experimental_model": True,
        "validated_reference": "v10.2.30_single_front_tip_only",
        "base_hazard_energy_audit": str(base_path),
        "base_event_initiation": base.get("event_initiation"),
        "base_absolute_athermal_Gc": base.get("absolute_athermal_Gc"),
        "base_gate_resolution": base.get("gate_resolution"),
        "single_front_only": True,
        "loading_mode": "monotonic",
        "bulk_plasticity_mode": "full_field",
        "bulk_model": BULK_MODEL_ID,
        "bulk_sources": "homogeneous_persistent_background",
        "bulk_initial_density_from_exact_selected_row": True,
        "bulk_peierls_taylor_parameters_from_exact_selected_row": True,
        "bulk_tip_mechanical_interaction": "shared_FEM_stress_and_directional_J",
        "direct_tip_bulk_population_exchange": False,
        "tip_source_rho_per_emit": 0.0,
        "bulk_density_transport": False,
        "bulk_finite_inventory": False,
        "v10_2_30_code_path_modified": False,
    }
    (root / "v10_4_bulk_coupled_model_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare_v104_args(args)

    original_validity = _stage3._force_stage3_validity_envelope
    original_v101_main = _v101.main
    _stage3._force_stage3_validity_envelope = (
        lambda local_args: _bulk_capable_stage3_validity(
            original_validity, local_args
        )
    )
    _v101.main = _full_field_v101_main
    try:
        result = _base.main(args)
        _write_model_audit(args)
        return result
    finally:
        _v101.main = original_v101_main
        _stage3._force_stage3_validity_envelope = original_validity


if __name__ == "__main__":
    main()
