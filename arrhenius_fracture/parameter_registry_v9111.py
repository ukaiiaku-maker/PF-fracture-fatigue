"""Exact v9.11.1 MPZ response-option registry selection.

The production sharp-front driver consumes one selected registry row through the
legacy one-row :class:`MaterialManifest` CSV interface. This module performs
that mechanical conversion without fitting, rounding, or substituting class
defaults. Fields that are not represented by the legacy manifest are validated
against the v10.2.15 production contract before the run is allowed to proceed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REGISTRY_SCHEMA = "MPZ_v9_11_1_parameter_registry"
CANONICAL_STAGE3_OPTIONS = (
    "ceramic_primary",
    "weakT_primary",
    "dbtt_primary",
    "peak_primary",
)
CANONICAL_CANDIDATES = {
    "ceramic_primary": "ceramic_restart02_candidate00",
    "weakT_primary": "weakT_restart00_candidate00",
    "dbtt_primary": "DBTT_restart04_candidate03",
    "peak_primary": "DBTT_restart05_candidate61",
}

_REQUIRED_FIELDS = {
    "option_key", "candidate_id", "material_class", "Tref_K",
    "n_slip_channels", "rho_forest_floor_m2", "peierls_stress_fraction",
    "taylor_stress_fraction", "mobile_shield_fraction",
    "source_recovery_rate_s", "L_pz_um_recommended", "n_bins_recommended",
    "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n",
    "cleave_floor_frac", "emit_G00_eV", "emit_gT_eV_per_K",
    "emit_sigc0_GPa", "emit_sT_GPa_per_K", "emit_exp_a", "emit_exp_n",
    "emit_floor_frac", "peierls_H0_eV", "peierls_activation_entropy_kB",
    "peierls_exp_a", "peierls_exp_n", "taylor_H0_eV",
    "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
    "taylor_corr_rho_c_m2", "taylor_corr_scale", "source_sites_per_system",
    "encounter_efficiency", "retained_recovery_rate_s",
    "source_refresh_length_um", "c_blunt", "peierls_nu0_s", "taylor_nu0_s",
}


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "materials" / "MPZ_v9_11_1_parameter_registry.csv"


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"registry field {key!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"registry field {key!r} is not finite")
    return value


@dataclass(frozen=True)
class SelectedResponseOption:
    option_key: str
    candidate_id: str
    material_class: str
    role: str
    mechanism_summary: str
    validation_status: str
    mpz_length_um: float
    mpz_n_bins: int
    row: dict[str, str]
    registry_path: str
    registry_sha256: str

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA,
            "option_key": self.option_key,
            "candidate_id": self.candidate_id,
            "material_class": self.material_class,
            "role": self.role,
            "mechanism_summary": self.mechanism_summary,
            "validation_status": self.validation_status,
            "mpz_length_um": self.mpz_length_um,
            "mpz_n_bins": self.mpz_n_bins,
            "registry_path": self.registry_path,
            "registry_sha256": self.registry_sha256,
            "exact_registry_row": dict(self.row),
        }


def read_registry(path: str | Path | None = None) -> list[dict[str, str]]:
    source = Path(path or default_registry_path()).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = set(reader.fieldnames or ())
    missing = sorted(_REQUIRED_FIELDS - fields)
    if missing:
        raise ValueError(f"v9.11.1 registry is missing required fields: {missing}")
    if not rows:
        raise ValueError("v9.11.1 registry contains no rows")
    keys = [row.get("option_key", "") for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("v9.11.1 registry contains duplicate option_key values")
    return rows


def _validate_current_spatial_contract(row: dict[str, str]) -> None:
    """Fail closed when a row requires controls not wired into v10.2.15."""
    exact_or_close = {
        "Tref_K": 481.33,
        "n_slip_channels": 2.0,
        "rho_forest_floor_m2": 5.0e12,
        "peierls_stress_fraction": 1.0 / math.sqrt(3.0),
        "taylor_stress_fraction": 1.0 / math.sqrt(3.0),
        "mobile_shield_fraction": 0.0,
        "source_recovery_rate_s": 0.0,
    }
    for key, expected in exact_or_close.items():
        value = _number(row, key)
        if not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(
                f"registry option {row.get('option_key')!r} requires {key}={value!r}; "
                f"v10.2.15 is wired and validated for {expected!r}"
            )
    n_bins = _number(row, "n_bins_recommended")
    if not float(n_bins).is_integer() or n_bins < 4:
        raise ValueError("n_bins_recommended must be an integer >= 4")
    if _number(row, "L_pz_um_recommended") <= 0.0:
        raise ValueError("L_pz_um_recommended must be positive")


def select_option(
    option_key: str,
    registry_path: str | Path | None = None,
    *,
    canonical_stage3_only: bool = False,
) -> SelectedResponseOption:
    key = str(option_key).strip()
    if not key:
        raise ValueError("parameter option must be nonempty")
    if canonical_stage3_only and key not in CANONICAL_STAGE3_OPTIONS:
        allowed = ", ".join(CANONICAL_STAGE3_OPTIONS)
        raise ValueError(f"Stage 3 option must be one of: {allowed}")
    source = Path(registry_path or default_registry_path()).expanduser().resolve()
    matches = [row for row in read_registry(source) if row["option_key"].strip() == key]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one registry row for {key!r}; found {len(matches)}")
    row = matches[0]
    expected = CANONICAL_CANDIDATES.get(key)
    if expected is not None and row["candidate_id"].strip() != expected:
        raise ValueError(
            f"candidate fingerprint mismatch for {key}: expected {expected!r}, "
            f"got {row['candidate_id']!r}"
        )
    _validate_current_spatial_contract(row)
    return SelectedResponseOption(
        option_key=key,
        candidate_id=row["candidate_id"].strip(),
        material_class=row["material_class"].strip(),
        role=row.get("role", "").strip(),
        mechanism_summary=row.get("mechanism_summary", "").strip(),
        validation_status=row.get("validation_status", "").strip(),
        mpz_length_um=_number(row, "L_pz_um_recommended"),
        mpz_n_bins=int(round(_number(row, "n_bins_recommended"))),
        row=dict(row),
        registry_path=str(source),
        registry_sha256=sha256_file(source),
    )


def write_compatibility_manifest(selected: SelectedResponseOption, destination: str | Path) -> Path:
    """Write one exact selected row for the existing MaterialManifest loader."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(selected.row)
    row["target_class"] = selected.option_key
    row["max_K_shield_MPa_sqrt_m"] = "0.0"
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(target)
    return target


def write_selection_audit(
    selected: SelectedResponseOption,
    destination: str | Path,
    *,
    compatibility_manifest: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = selected.audit_payload()
    if compatibility_manifest is not None:
        manifest_path = Path(compatibility_manifest).expanduser().resolve()
        payload["compatibility_manifest_path"] = str(manifest_path)
        payload["compatibility_manifest_sha256"] = sha256_file(manifest_path)
        payload["compatibility_max_K_shield_MPa_sqrt_m"] = 0.0
        payload["constitutive_K_shield_cap_applied"] = False
    if extra:
        payload.update(extra)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def stage3_case_rows(
    options: Iterable[str] = CANONICAL_STAGE3_OPTIONS,
    temperatures: Iterable[float] = tuple(range(300, 1201, 100)),
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for option in options:
        selected = select_option(option, registry_path, canonical_stage3_only=True)
        for temperature in temperatures:
            rows.append({
                "option_key": selected.option_key,
                "candidate_id": selected.candidate_id,
                "temperature_K": float(temperature),
                "mpz_length_um": selected.mpz_length_um,
                "mpz_n_bins": selected.mpz_n_bins,
            })
    return rows


__all__ = [
    "REGISTRY_SCHEMA", "CANONICAL_STAGE3_OPTIONS", "CANONICAL_CANDIDATES",
    "SelectedResponseOption", "default_registry_path", "read_registry",
    "select_option", "write_compatibility_manifest", "write_selection_audit",
    "stage3_case_rows", "sha256_file",
]
