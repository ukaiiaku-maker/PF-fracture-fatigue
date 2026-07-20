"""v10.2.15 Stage 3 parameter overlay for the final accepted 2-D model.

This entry changes only the selected material manifest and the option-specific
MPZ length/bin count. It executes ``sharp_front_v10_1_7_5`` directly and does
not replace the tip engine, source lifecycle, transport operator, shielding
law, crack geometry, or mechanics observer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from . import sharp_front_v10_1_7_5 as _final_2d
from . import zero_event_summary_v10215 as _zero_event_summary  # noqa: F401
from .parameter_registry_v9111 import (
    CANONICAL_STAGE3_OPTIONS,
    SelectedResponseOption,
    default_registry_path,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)

MODEL_ID = "v10.2.15_stage3_existing_2d_parameter_overlay"
FINAL_2D_ENTRY = "arrhenius_fracture.sharp_front_v10_1_7_5"


def _pop_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return default


def _option_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _remove_value_option(args: list[str], name: str) -> None:
    prefix = name + "="
    kept: list[str] = []
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(prefix):
            continue
        kept.append(token)
    args[:] = kept


def _set_value_option(args: list[str], name: str, value: Any) -> None:
    _remove_value_option(args, name)
    args.extend([name, str(value)])


def _set_toggle(args: list[str], positive: str, negative: str, enabled: bool) -> None:
    args[:] = [token for token in args if token not in {positive, negative}]
    args.append(positive if enabled else negative)


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _require_env_value(name: str, expected: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower().replace("-", "_")
    normalized_expected = expected.strip().lower().replace("-", "_")
    if value != normalized_expected:
        raise SystemExit(f"v10.2.15 Stage 3 requires {name}={expected}; got {value!r}")
    return value


def _force_stage3_validity_envelope(args: list[str]) -> None:
    mode = _option_value(args, "--mode", "2d")
    if mode != "2d":
        raise SystemExit("v10.2.15 Stage 3 supports only --mode 2d")
    _set_value_option(args, "--mode", "2d")
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.2.15 Stage 3 is monotonic; fatigue options are not permitted")
    if "--crystal-branch" in args:
        raise SystemExit("v10.2.15 Stage 3 requires branching disabled")
    supplied_fronts = _option_value(args, "--max-fronts")
    if supplied_fronts is not None and int(supplied_fronts) != 1:
        raise SystemExit("v10.2.15 Stage 3 requires --max-fronts 1")
    _set_value_option(args, "--max-fronts", 1)
    if "--wake-shielding" in args:
        raise SystemExit("Stage 3 requires the requested no-wake-shielding configuration")
    _set_toggle(args, "--wake-shielding", "--no-wake-shielding", False)
    mobile_fraction = _option_value(args, "--mobile-shield-fraction")
    if mobile_fraction is not None and abs(float(mobile_fraction)) > 1.0e-15:
        raise SystemExit("v9.11.1 Stage 3 requires --mobile-shield-fraction 0")
    _set_value_option(args, "--mobile-shield-fraction", 0.0)
    controls = {
        "--bulk-plasticity-mode": "tip_only",
        "--directional-j-mode": "root_signed",
        "--tip-kinetics-mode": "moving_velocity",
        "--tip-source-model": "continuum",
        "--front-state-model": "moving_pz",
    }
    for name, expected in controls.items():
        supplied = _option_value(args, name)
        if supplied is not None and supplied.strip().lower() != expected:
            raise SystemExit(f"v10.2.15 Stage 3 requires {name} {expected}")
        _set_value_option(args, name, expected)
    if "--crystal-aniso" not in args:
        raise SystemExit("v10.2.15 Stage 3 requires --crystal-aniso")
    if "--crystal-compete" not in args:
        raise SystemExit("v10.2.15 Stage 3 requires --crystal-compete")
    _require_env_value("CLEAVAGE_HAZARD_MODE", "deterministic", "deterministic")
    _require_env_value("CLEAVAGE_EVENT_LENGTH_MODE", "fixed", "fixed")
    _require_env_value("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar", "validated_scalar")


def _prepare_parameter_option(args: list[str]) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.15 requires --out so the selected registry row is persistent")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry_value = _pop_value(
        args,
        "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(default_registry_path())),
    )
    option_key = _pop_value(args, "--parameter-option", os.environ.get("PARAMETER_OPTION"))
    if not option_key:
        allowed = ", ".join(CANONICAL_STAGE3_OPTIONS)
        raise SystemExit(f"v10.2.15 requires --parameter-option; Stage 3 options: {allowed}")
    selected = select_option(option_key, registry_value, canonical_stage3_only=True)
    _remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected, output_root / "selected_material_manifest_v9_11_1.csv"
    )
    _set_value_option(args, "--material-manifest", manifest_path)
    _set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)
    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_15_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "final_2d_entry": FINAL_2D_ENTRY,
            "parameter_overlay_only": True,
            "tip_engine_replaced": False,
            "source_lifecycle_replaced": False,
            "transport_operator_replaced": False,
            "shielding_law_replaced": False,
            "geometry_backend_replaced": False,
            "material_parameter_refit_in_2d": False,
        },
    )
    return selected, manifest_path, audit_path


def _write_overlay_audit(
    args: list[str],
    selected: SelectedResponseOption,
    manifest_path: Path,
    selection_audit_path: Path,
) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    payload = {
        "schema": MODEL_ID,
        "final_2d_entry": FINAL_2D_ENTRY,
        "parameter_overlay_only": True,
        "selected_option": selected.audit_payload(),
        "selected_material_manifest": str(manifest_path),
        "parameter_selection_audit": str(selection_audit_path),
        "physics_stack": {
            "campaign_calibrated_source_budget": "preserved",
            "tensor_resolved_anisotropic_emission": "preserved",
            "validated_scalar_peierls_taylor_transport": "preserved",
            "source_depletion_and_crack_advance_refresh": "preserved",
            "local_mobile_retained_backstress": "preserved",
            "sharp_wake_geometry_backend": "preserved",
            "tip_engine_substitution": False,
            "signed_atlas_substitution": False,
        },
    }
    root = Path(out)
    (root / "v10_2_15_existing_2d_parameter_overlay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    selected, manifest_path, selection_audit_path = _prepare_parameter_option(args)
    _force_stage3_validity_envelope(args)
    print(
        "  v10.2.15 parameter overlay only: "
        f"entry={FINAL_2D_ENTRY} option={selected.option_key} "
        f"candidate={selected.candidate_id} "
        f"mpz={selected.mpz_length_um:g}um/{selected.mpz_n_bins}bins"
    )
    result = _final_2d.main(args)
    _write_overlay_audit(args, selected, manifest_path, selection_audit_path)
    return result


if __name__ == "__main__":
    main()
