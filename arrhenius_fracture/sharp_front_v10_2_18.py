"""Alternate-DBTT parameter screen on the unchanged v10.2.17 signed 2-D stack.

This wrapper changes only which exact v9.11.1 DBTT registry row is selected.  All
mechanics, signed-kernel, moving-MPZ, source, transport, anisotropic-emission,
stochastic-hazard, and crack-geometry behavior is delegated to v10.2.17.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _base
from .parameter_registry_v9111 import (
    SelectedResponseOption,
    default_registry_path,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)

MODEL_ID = "v10.2.18_dbtt_candidate_short_screen"
BASE_ENTRY = "arrhenius_fracture.sharp_front_v10_2_17"
DBTT_SCREEN_OPTIONS = (
    "dbtt_primary",
    "dbtt_broad_shielding",
    "dbtt_intrinsic_control",
    "dbtt_moderate_shielding_reference",
)


def _prepare_dbtt_screen_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _base._option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.18 requires --out")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_value = _base._pop_value(
        args,
        "--parameter-registry",
        str(default_registry_path()),
    )
    option_key = _base._pop_value(args, "--parameter-option")
    if option_key not in DBTT_SCREEN_OPTIONS:
        allowed = ", ".join(DBTT_SCREEN_OPTIONS)
        raise SystemExit(f"v10.2.18 DBTT screen option must be one of: {allowed}")

    selected = select_option(option_key, registry_value, canonical_stage3_only=False)
    if selected.material_class.strip().lower() != "dbtt":
        raise SystemExit(
            f"v10.2.18 accepts only DBTT rows; {option_key!r} is "
            f"{selected.material_class!r}"
        )

    _base._remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected, output_root / "selected_material_manifest_v9_11_1.csv"
    )
    _base._set_value_option(args, "--material-manifest", manifest_path)
    _base._set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _base._set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)
    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_18_dbtt_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "base_entry": BASE_ENTRY,
            "parameter_overlay_only": True,
            "alternate_dbtt_screen": True,
            "material_parameter_refit_in_2d": False,
            "mechanics_changed": False,
            "source_model_changed": False,
            "signed_kernel_changed": False,
            "stochastic_law_changed": False,
        },
    )
    return selected, manifest_path, audit_path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_prepare = _base._prepare_parameter_option
    _base._prepare_parameter_option = _prepare_dbtt_screen_option
    try:
        result = _base.main(args)
    finally:
        _base._prepare_parameter_option = original_prepare

    out = _base._option_value(args, "--out")
    option = _base._option_value(args, "--parameter-option")
    if out:
        payload = {
            "schema": MODEL_ID,
            "base_entry": BASE_ENTRY,
            "selected_option": option,
            "candidate_screen_only": True,
            "existing_v10_2_17_physics_preserved": True,
        }
        (Path(out) / "v10_2_18_dbtt_candidate_screen.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return result


if __name__ == "__main__":
    main()
