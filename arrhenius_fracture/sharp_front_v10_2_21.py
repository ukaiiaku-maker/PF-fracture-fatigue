"""Top-ranked v9.12 DBTT/peak transfer with persistent crack-tip sources.

The frozen v10.2.18 signed 2-D mechanics are preserved.  This entry changes the
selected material row and replaces only the finite source-budget closure with
v10.2.21 persistent areal nucleation sites, backstress-limited emission, dynamic
blunting, density-limited along-front influence width, and moving-frame
resharpening.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _base
from .parameter_registry_v9111 import (
    SelectedResponseOption,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)
from .persistent_site_source_v10221 import (
    MODEL_ID as SOURCE_MODEL_ID,
    PersistentSiteConfig,
    PersistentSiteStateResolvedTipEngine,
)


MODEL_ID = "v10.2.21_v912_top1_persistent_sites_blunting"
OPTION_KEY = "v912_top1_peak_persistent_sites"
CANDIDATE_ID = "v912_targeted_local_peak_013476_0368"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_21_v912_top1_persistent_site_registry.csv"
)


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v10.2.21 registry field {key!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"v10.2.21 registry field {key!r} is not finite")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None else raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None else raw)


def _prepare_top1_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _base._option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.21 requires --out")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_value = _base._pop_value(
        args,
        "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(DEFAULT_REGISTRY)),
    )
    option_key = _base._pop_value(
        args,
        "--parameter-option",
        os.environ.get("PARAMETER_OPTION", OPTION_KEY),
    )
    if option_key != OPTION_KEY:
        raise SystemExit(f"v10.2.21 requires --parameter-option {OPTION_KEY}")

    selected = select_option(option_key, registry_value, canonical_stage3_only=False)
    if selected.candidate_id != CANDIDATE_ID:
        raise SystemExit(
            f"v10.2.21 candidate fingerprint mismatch: expected {CANDIDATE_ID!r}; "
            f"got {selected.candidate_id!r}"
        )
    if selected.material_class.strip().lower() != "dbtt":
        raise SystemExit("v10.2.21 top-ranked row must have material_class=DBTT")

    row = selected.row
    zero_fields = (
        "source_recovery_rate_s",
        "retained_recovery_rate_s",
        "recovery_nu0_s",
        "legacy_source_sites_active",
        "legacy_source_refresh_active",
        "explicit_recovery_active",
    )
    for key in zero_fields:
        if abs(_number(row, key)) > 1.0e-30:
            raise SystemExit(f"v10.2.21 requires {key}=0; got {row[key]!r}")

    cfg = PersistentSiteConfig(
        rho_site0_m2=_number(row, "rho_source0_m2"),
        reference_source_area_m2=1.0e-12
        * _env_float("PERSISTENT_SOURCE_REFERENCE_AREA_UM2", _number(row, "reference_source_area_um2")),
        reference_front_width_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_REFERENCE_WIDTH_UM", _number(row, "reference_front_width_um")),
        reference_density_m2=_env_float(
            "PERSISTENT_SOURCE_REFERENCE_DENSITY_M2",
            _number(row, "rho_forest_floor_m2"),
        ),
        source_zone_length_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_ZONE_LENGTH_UM", _number(row, "source_zone_length_um")),
        minimum_front_width_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_MIN_WIDTH_UM", 0.0),
        maximum_front_width_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_MAX_WIDTH_UM", selected.mpz_length_um),
        implicit_tolerance=_env_float("PERSISTENT_SOURCE_IMPLICIT_TOL", 1.0e-10),
        implicit_max_iterations=_env_int("PERSISTENT_SOURCE_IMPLICIT_MAX_ITER", 96),
    ).validate()
    PersistentSiteStateResolvedTipEngine.configure_persistent_sites(cfg)

    _base._remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected,
        output_root / "selected_material_manifest_v10_2_21.csv",
    )
    _base._set_value_option(args, "--material-manifest", manifest_path)
    _base._set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _base._set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)

    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_21_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "source_model_id": SOURCE_MODEL_ID,
            "base_entry": "arrhenius_fracture.sharp_front_v10_2_17",
            "mechanics_changed": False,
            "signed_kernel_changed": False,
            "stochastic_cleavage_law_changed": False,
            "source_closure_changed": True,
            "finite_source_inventory": False,
            "legacy_source_sites_per_system_active": False,
            "legacy_source_refresh_length_active": False,
            "explicit_recovery_active": False,
            "persistent_site_config": cfg.__dict__,
            "rho_source0_semantics": "persistent_areal_nucleation_site_density_m2",
            "reference_area_semantics": "v9.12 calibrated in-plane source area per reduced slip system",
        },
    )
    return selected, manifest_path, audit_path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_prepare = _base._prepare_parameter_option
    original_engine = _base.StateResolvedSignedBurgersTipEngine
    _base._prepare_parameter_option = _prepare_top1_option
    _base.StateResolvedSignedBurgersTipEngine = PersistentSiteStateResolvedTipEngine
    try:
        result = _base.main(args)
    finally:
        _base._prepare_parameter_option = original_prepare
        _base.StateResolvedSignedBurgersTipEngine = original_engine

    out = _base._option_value(args, "--out")
    if out:
        payload = {
            "schema": MODEL_ID,
            "base_commit_model": "v10.2.18/v10.2.17 signed 2-D stack",
            "selected_option": OPTION_KEY,
            "selected_candidate": CANDIDATE_ID,
            "persistent_sites": True,
            "finite_source_inventory": False,
            "source_refresh": False,
            "backstress_limited_emission": True,
            "dynamic_tip_blunting": True,
            "moving_frame_resharpening": True,
        }
        (Path(out) / "v10_2_21_persistent_site_model.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return result


if __name__ == "__main__":
    main()
