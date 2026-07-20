"""v10.2.19 full-field bulk-plasticity DBTT candidate screen.

Relative to v10.2.18 this entry changes one controlled model choice:
``bulk_plasticity_mode`` is ``full_field`` rather than ``tip_only``.  Tip
plasticity, the moving MPZ, signed active shielding, stochastic cleavage and
all mechanics controls are preserved.

The surrounding FEM does not use an independently fitted plastic law.  The
exact selected v9.11.1 emission, Peierls and Taylor surfaces are installed on
the production emission-derived Peierls--Taylor update before the 2-D driver
constructs its material state.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from . import plasticity as _plasticity
from . import sharp_front as _sharp
from . import sharp_front_v10_1 as _protected
from . import sharp_front_v10_1_5 as _campaign
from . import sharp_front_v10_2_17 as _stage3
from . import sharp_front_v10_2_18 as _screen
from .emission_derived_plasticity import install_manifest_bulk_kinetics
from .material_manifest import MaterialManifest
from .parameter_registry_v9111 import SelectedResponseOption


MODEL_ID = "v10.2.19_dbtt_full_field_bulk_screen"
BASE_ENTRY = "arrhenius_fracture.sharp_front_v10_2_17"
PARAMETER_ENTRY = "arrhenius_fracture.sharp_front_v10_2_18"
BULK_MODE = "full_field"

_ORIGINAL_FORCE_ENVELOPE = _stage3._force_stage3_validity_envelope
_ORIGINAL_MAKE_CONFIG = _sharp.make_emergent_config
_ACTIVE_SELECTION: SelectedResponseOption | None = None
_ACTIVE_MANIFEST: MaterialManifest | None = None
_ACTIVE_MANIFEST_PATH: Path | None = None
_ACTIVE_BULK_MAPPING: dict[str, Any] = {}


def _prepare_full_field_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    global _ACTIVE_SELECTION, _ACTIVE_MANIFEST, _ACTIVE_MANIFEST_PATH
    selected, manifest_path, audit_path = _screen._prepare_dbtt_screen_option(args)
    _ACTIVE_SELECTION = selected
    _ACTIVE_MANIFEST_PATH = Path(manifest_path).expanduser().resolve()
    _ACTIVE_MANIFEST = MaterialManifest.from_csv(_ACTIVE_MANIFEST_PATH)
    if _ACTIVE_MANIFEST.candidate_id != selected.candidate_id:
        raise RuntimeError(
            "selected DBTT candidate and compatibility manifest disagree: "
            f"{selected.candidate_id!r} != {_ACTIVE_MANIFEST.candidate_id!r}"
        )
    return selected, manifest_path, audit_path


def _force_full_field_envelope(args: list[str]) -> int:
    requested = _stage3._option_value(args, "--bulk-plasticity-mode", BULK_MODE)
    if str(requested).strip().lower() != BULK_MODE:
        raise SystemExit(
            f"v10.2.19 requires --bulk-plasticity-mode {BULK_MODE}; got {requested!r}"
        )
    _stage3._remove_value_option(args, "--bulk-plasticity-mode")
    seed = _ORIGINAL_FORCE_ENVELOPE(args)
    _stage3._set_value_option(args, "--bulk-plasticity-mode", BULK_MODE)
    return seed


def _make_manifest_bulk_config():
    global _ACTIVE_BULK_MAPPING
    if _ACTIVE_SELECTION is None or _ACTIVE_MANIFEST is None:
        raise RuntimeError("v10.2.19 material selection was not prepared before FEM configuration")
    cfg = _ORIGINAL_MAKE_CONFIG()
    _ACTIVE_BULK_MAPPING = install_manifest_bulk_kinetics(
        cfg.dislocations,
        _ACTIVE_MANIFEST,
        _ACTIVE_SELECTION.row,
    )
    return cfg


def _option_value(args: list[str], name: str) -> str | None:
    return _protected._option_value(args, name)


def _rewrite_full_field_mode_audits(
    out: Path,
    update_audit: dict[str, Any],
) -> None:
    if not _ACTIVE_BULK_MAPPING:
        raise RuntimeError("full-field run finished without installing exact bulk kinetics")
    calls = max(int(update_audit.get("calls", 0)), 1)
    update_audit = dict(update_audit)
    update_audit["fraction_calls_with_nonzero_accepted_strain"] = (
        float(update_audit.get("calls_with_nonzero_accepted_strain", 0)) / calls
    )
    update_audit["fraction_calls_with_increment_limiter"] = (
        float(update_audit.get("calls_with_increment_limiter", 0)) / calls
    )
    common = {
        "bulk_plasticity_mode": BULK_MODE,
        "full_field_bulk_enabled": True,
        "tip_plasticity_enabled": True,
        "bulk_update_is_noop": False,
        "bulk_manifest_mapped": True,
        "bulk_independent_parameter_fit": False,
        "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
        "bulk_and_tip_share_selected_arrhenius_surfaces": True,
    }
    for name in ("v10_0_1_driver_modes.json", "v10_1_driver_modes.json"):
        path = out / name
        payload = json.loads(path.read_text()) if path.is_file() else {}
        payload.update(common)
        if name == "v10_0_1_driver_modes.json":
            payload["legacy_full_field_enabled"] = True
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    payload = {
        "schema": MODEL_ID,
        "base_entry": BASE_ENTRY,
        "parameter_entry": PARAMETER_ENTRY,
        "selected_option": None if _ACTIVE_SELECTION is None else _ACTIVE_SELECTION.option_key,
        "candidate_id": None if _ACTIVE_SELECTION is None else _ACTIVE_SELECTION.candidate_id,
        "selected_manifest": None if _ACTIVE_MANIFEST_PATH is None else str(_ACTIVE_MANIFEST_PATH),
        **common,
        "bulk_mapping": dict(_ACTIVE_BULK_MAPPING),
        "bulk_update_audit": update_audit,
    }
    (out / "v10_2_19_full_field_bulk_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _full_field_protected_main(argv=None):
    args, bulk_mode, j_mode, kinetics_mode, tip_cfg, source_model = (
        _protected._prepare_args_v1011(sys.argv[1:] if argv is None else argv)
    )
    if bulk_mode != BULK_MODE:
        raise SystemExit(f"v10.2.19 expected bulk mode {BULK_MODE}; got {bulk_mode!r}")
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit("v10.2.19 requires a selected material manifest")
    if _ACTIVE_MANIFEST is None:
        raise RuntimeError("selected manifest was not prepared")

    engine_cls = (
        _protected.ContinuumSourceKineticTipEngine
        if source_model == "continuum"
        else _protected.KineticMovingTipFrontEngine
    )
    original_update = _plasticity.update_plasticity
    original_diag = _protected.UnifiedMPZState.diagnostics
    original_advance = _protected.UnifiedMPZState.advance
    original_engine = _protected.sharp_front.UnifiedMPZFrontEngine
    had_shield_cfg = hasattr(_protected.UnifiedMPZState, "_v101_shield_cfg")
    old_shield_cfg = getattr(_protected.UnifiedMPZState, "_v101_shield_cfg", None)

    audit: dict[str, Any] = {
        "calls": 0,
        "bulk_pt_active_calls": 0,
        "calls_with_nonzero_accepted_strain": 0,
        "calls_with_increment_limiter": 0,
        "limited_gauss_point_count": 0,
        "accepted_equivalent_strain_sum": 0.0,
        "accepted_equivalent_strain_max": 0.0,
        "accepted_plastic_work_density_sum_Pa": 0.0,
        "rho_change_abs_sum_m2": 0.0,
        "rho_change_abs_max_m2": 0.0,
    }

    def audited_update(*call_args, **call_kwargs):
        rho_before = np.asarray(call_args[1], dtype=float).copy()
        result = original_update(*call_args, **call_kwargs)
        audit["calls"] += 1
        rho_after = np.asarray(result[1], dtype=float)
        drho = np.abs(rho_after - rho_before)
        audit["rho_change_abs_sum_m2"] += float(np.sum(drho))
        audit["rho_change_abs_max_m2"] = max(
            float(audit["rho_change_abs_max_m2"]),
            float(np.max(drho)) if drho.size else 0.0,
        )
        if len(result) >= 4 and isinstance(result[3], dict):
            info = result[3]
            if bool(info.get("bulk_pt_active", False)):
                audit["bulk_pt_active_calls"] += 1
            dep = np.asarray(info.get("dep_eq_accepted_gp", 0.0), dtype=float)
            work = np.asarray(info.get("dWp_accepted_gp", 0.0), dtype=float)
            limited = np.asarray(info.get("dep_eq_limited_gp", 0.0), dtype=float)
            dep_sum = float(np.sum(np.maximum(dep, 0.0)))
            dep_max = float(np.max(np.maximum(dep, 0.0))) if dep.size else 0.0
            n_limited = int(np.count_nonzero(limited > 0.5))
            if dep_max > 0.0:
                audit["calls_with_nonzero_accepted_strain"] += 1
            if n_limited > 0:
                audit["calls_with_increment_limiter"] += 1
                audit["limited_gauss_point_count"] += n_limited
            audit["accepted_equivalent_strain_sum"] += dep_sum
            audit["accepted_equivalent_strain_max"] = max(
                float(audit["accepted_equivalent_strain_max"]), dep_max
            )
            audit["accepted_plastic_work_density_sum_Pa"] += float(
                np.sum(np.maximum(work, 0.0))
            )
        return result

    try:
        _plasticity.update_plasticity = audited_update
        _protected.UnifiedMPZState.diagnostics = _protected._diagnostics_with_csv_aliases
        if kinetics_mode == "moving_velocity":
            _protected.UnifiedMPZState.advance = _protected.fractional_moving_frame_advance
            _protected.UnifiedMPZState._v101_shield_cfg = tip_cfg
            engine_cls.configure_default(tip_cfg)
            engine_cls.reset_audit()
            _protected.sharp_front.UnifiedMPZFrontEngine = engine_cls

        wake_mode = _protected._resolved_wake_shielding(args)
        print(
            "  v10.2.19 driving modes: "
            f"bulk_plasticity={bulk_mode}, directional_J={j_mode}, "
            f"tip_kinetics={kinetics_mode}, tip_source_model={source_model}, "
            f"tip_plasticity={int(tip_cfg.plasticity_enabled)}, "
            f"active_shielding={int(tip_cfg.active_shielding)}, "
            f"wake_shielding={int(wake_mode)}, "
            "bulk_mapping=exact_selected_manifest"
        )
        result = _protected.sharp_front.main(args)
        _protected._write_mode_audit(
            args, bulk_mode, j_mode, kinetics_mode, tip_cfg, source_model, engine_cls
        )
        out_value = _option_value(args, "--out")
        if out_value:
            _rewrite_full_field_mode_audits(Path(out_value), audit)
        return result
    finally:
        _plasticity.update_plasticity = original_update
        _protected.UnifiedMPZState.diagnostics = original_diag
        _protected.UnifiedMPZState.advance = original_advance
        _protected.sharp_front.UnifiedMPZFrontEngine = original_engine
        if had_shield_cfg:
            _protected.UnifiedMPZState._v101_shield_cfg = old_shield_cfg
        elif hasattr(_protected.UnifiedMPZState, "_v101_shield_cfg"):
            delattr(_protected.UnifiedMPZState, "_v101_shield_cfg")


def main(argv=None):
    global _ACTIVE_SELECTION, _ACTIVE_MANIFEST, _ACTIVE_MANIFEST_PATH, _ACTIVE_BULK_MAPPING
    args = list(sys.argv[1:] if argv is None else argv)
    _ACTIVE_SELECTION = None
    _ACTIVE_MANIFEST = None
    _ACTIVE_MANIFEST_PATH = None
    _ACTIVE_BULK_MAPPING = {}

    original_prepare = _stage3._prepare_parameter_option
    original_force = _stage3._force_stage3_validity_envelope
    original_make = _sharp.make_emergent_config
    original_protected_main = _campaign._protected.main
    try:
        _stage3._prepare_parameter_option = _prepare_full_field_option
        _stage3._force_stage3_validity_envelope = _force_full_field_envelope
        _sharp.make_emergent_config = _make_manifest_bulk_config
        _campaign._protected.main = _full_field_protected_main
        return _stage3.main(args)
    finally:
        _stage3._prepare_parameter_option = original_prepare
        _stage3._force_stage3_validity_envelope = original_force
        _sharp.make_emergent_config = original_make
        _campaign._protected.main = original_protected_main


if __name__ == "__main__":
    main()
