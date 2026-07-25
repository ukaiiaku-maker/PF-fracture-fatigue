#!/usr/bin/env python3
"""Install the canonical final four-class v10.2.27 paper registry.

The source registry is the exact full-precision four-row transfer committed by the
v9.13 weak-T/ceramic selection handoff.  This installer preserves the legacy
output filenames consumed by the audited v10.2.27 entry point while validating
that no mechanics or persistent-site closure coordinate changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "arrhenius_fracture" / "data" / "materials"
SOURCE_REGISTRY = MATERIALS / "v10_2_27_v913_four_class_paper_registry.csv"
SOURCE_SELECTION = MATERIALS / "v10_2_27_v913_four_class_paper_selection.json"
DEFAULT_OUTPUT_REGISTRY = MATERIALS / "v10_2_27_paper_four_class_registry.csv"
DEFAULT_OUTPUT_SELECTION = MATERIALS / "v10_2_27_paper_four_class_selection.json"

CANONICAL_OPTIONS = (
    ("v913_paper_peak01_0242980_persistent_sites", "v913_zeroD_sobol_0242980", "peak"),
    ("v913_paper_dbtt01_0202500_persistent_sites", "v913_zeroD_sobol_0202500", "DBTT"),
    ("v913_paper_weakT01_0129902_persistent_sites", "v913_zeroD_sobol_0129902", "weakT"),
    ("v913_paper_ceramic01_0077080_persistent_sites", "v913_zeroD_sobol_0077080", "ceramic"),
)
ACTIVE_PARAMETER_FIELDS = (
    "Tref_K",
    "cleave_G00_eV",
    "cleave_gT_eV_per_K",
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "cleave_exp_a",
    "cleave_exp_n",
    "cleave_floor_frac",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_nu0_s",
    "rho_source0_m2",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "c_blunt",
)
ZERO_FIELDS = (
    "source_recovery_rate_s",
    "retained_recovery_rate_s",
    "source_refresh_length_um",
    "recovery_nu0_s",
    "recovery_H0_eV",
    "recovery_activation_entropy_kB",
    "legacy_source_sites_active",
    "legacy_source_refresh_active",
    "explicit_recovery_active",
)
REJECTED_OPTIONS = {
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"registry has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _validate_source() -> tuple[str, dict[str, Any]]:
    if not SOURCE_REGISTRY.is_file() or not SOURCE_SELECTION.is_file():
        raise FileNotFoundError(
            "missing canonical v10.2.27 source registry or selection record"
        )

    registry_text = SOURCE_REGISTRY.read_text()
    header, rows = _read_registry(SOURCE_REGISTRY)
    selection = json.loads(SOURCE_SELECTION.read_text())

    expected_options = [item[0] for item in CANONICAL_OPTIONS]
    expected_candidates = [item[1] for item in CANONICAL_OPTIONS]
    expected_classes = [item[2] for item in CANONICAL_OPTIONS]

    if len(rows) != 4:
        raise ValueError(f"canonical registry must contain exactly four rows; got {len(rows)}")
    if [row.get("option_key") for row in rows] != expected_options:
        raise ValueError("canonical registry option order mismatch")
    if [row.get("candidate_id") for row in rows] != expected_candidates:
        raise ValueError("canonical registry candidate order mismatch")
    if [row.get("material_class") for row in rows] != expected_classes:
        raise ValueError("canonical registry class labels must be peak, DBTT, weakT, ceramic")
    if len({row["option_key"] for row in rows}) != 4:
        raise ValueError("canonical registry option keys are not unique")
    if len({row["candidate_id"] for row in rows}) != 4:
        raise ValueError("canonical registry candidate IDs are not unique")
    if REJECTED_OPTIONS.intersection(row["option_key"] for row in rows):
        raise ValueError("rejected old weakT/ceramic option entered canonical registry")

    missing_active = [field for field in ACTIVE_PARAMETER_FIELDS if field not in header]
    if missing_active:
        raise ValueError(f"canonical registry is missing active fields: {missing_active}")
    for row in rows:
        for field in ACTIVE_PARAMETER_FIELDS:
            value = row.get(field, "")
            if value == "":
                raise ValueError(f"{row['option_key']} has empty active field {field}")
        for field in ZERO_FIELDS:
            value = float(row.get(field, "nan"))
            if value != 0.0:
                raise ValueError(
                    f"{row['option_key']} violates fixed closure: {field}={value}"
                )

    if selection.get("canonical_option_order") != expected_options:
        raise ValueError("canonical selection option order mismatch")
    primary = selection.get("primary_candidates", [])
    if [item.get("option_key") for item in primary] != expected_options:
        raise ValueError("canonical selection primary candidate order mismatch")
    fixed = selection.get("fixed_closure", {})
    required_fixed = {
        "persistent_sites": True,
        "finite_source_inventory": False,
        "source_depletion_on_emission": False,
        "source_refresh_on_crack_advance": False,
        "explicit_recovery": False,
        "dynamic_tip_radius": True,
        "dynamic_front_width": True,
    }
    for key, value in required_fixed.items():
        if fixed.get(key) is not value:
            raise ValueError(f"canonical selection fixed closure mismatch: {key}")

    expected_hash = selection.get("source_registry_sha256")
    actual_hash = _sha256_bytes(registry_text.encode())
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            f"canonical source registry SHA mismatch: {actual_hash} != {expected_hash}"
        )
    return registry_text, selection


def build_payloads() -> tuple[str, str]:
    registry_text, source_selection = _validate_source()
    expected_options = [item[0] for item in CANONICAL_OPTIONS]
    registry_sha = _sha256_bytes(registry_text.encode())

    selection_payload: dict[str, Any] = {
        "schema": "v10.2.27_paper_four_class_selection_v2",
        "purpose": (
            "Final exact four-class parameter overlay for the 30 degree, 1000 um, "
            "stochastic PF/sharp-front paper campaign."
        ),
        "canonical_option_order": expected_options,
        "primary_candidates": source_selection["primary_candidates"],
        "weakT_ceramic_backup_candidates": source_selection.get(
            "weakT_ceramic_backup_candidates", []
        ),
        "source_active_parameter_fingerprint_sha256": source_selection.get(
            "source_active_parameter_fingerprint_sha256"
        ),
        "installed_registry_sha256": registry_sha,
        "source_files": {
            str(SOURCE_REGISTRY.relative_to(ROOT)): _sha256_file(SOURCE_REGISTRY),
            str(SOURCE_SELECTION.relative_to(ROOT)): _sha256_file(SOURCE_SELECTION),
        },
        "class_labels": ["peak", "DBTT", "weakT", "ceramic"],
        "physics_contract": {
            "base_model": "v10.2.22 audited persistent-site sharp-front model",
            "parameter_transfer_only": True,
            "mechanics_changed": False,
            "stochastic_cleavage_law_changed": False,
            "persistent_sites": True,
            "finite_source_inventory": False,
            "source_depletion_on_emission": False,
            "source_refresh_on_crack_advance": False,
            "explicit_recovery": False,
            "dynamic_tip_radius": True,
            "physical_front_width": True,
            "front_width_grid_independent": True,
        },
        "rejected_replaced_options": sorted(REJECTED_OPTIONS),
        "transfer_policy": source_selection.get(
            "transfer_policy",
            "Exact source-row transfer only; no fitting, transformation, or rounding.",
        ),
    }
    selection_text = json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
    return registry_text, selection_text


def _write_or_check(path: Path, content: str, check_only: bool) -> None:
    if check_only:
        if not path.is_file():
            raise FileNotFoundError(f"generated file is missing: {path}")
        if path.read_text() != content:
            raise RuntimeError(f"generated file is stale or modified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-registry", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    parser.add_argument("--output-selection", type=Path, default=DEFAULT_OUTPUT_SELECTION)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    registry_text, selection_text = build_payloads()
    registry = args.output_registry.expanduser().resolve()
    selection = args.output_selection.expanduser().resolve()
    _write_or_check(registry, registry_text, args.check_only)
    _write_or_check(selection, selection_text, args.check_only)

    print(
        json.dumps(
            {
                "check_only": args.check_only,
                "registry": str(registry),
                "registry_sha256": _sha256_bytes(registry_text.encode()),
                "selection": str(selection),
                "selection_sha256": _sha256_bytes(selection_text.encode()),
                "options": [item[0] for item in CANONICAL_OPTIONS],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
