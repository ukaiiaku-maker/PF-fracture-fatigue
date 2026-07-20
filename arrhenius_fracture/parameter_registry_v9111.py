"""Select one exact v9.11.1 parameter row for the existing v10.1.7.5 2-D model."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

CANONICAL_OPTIONS = (
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


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "materials" / "MPZ_v9_11_1_parameter_registry.csv"


@dataclass(frozen=True)
class SelectedOption:
    option_key: str
    candidate_id: str
    material_class: str
    mpz_length_um: float
    mpz_n_bins: int
    row: dict[str, str]
    registry_path: str


def select_option(option_key: str, registry_path: str | Path | None = None) -> SelectedOption:
    key = str(option_key).strip()
    if key not in CANONICAL_OPTIONS:
        raise ValueError(f"parameter option must be one of {CANONICAL_OPTIONS}; got {key!r}")
    source = Path(registry_path or default_registry_path()).expanduser().resolve()
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("option_key", "").strip() == key]
    if len(matches) != 1:
        raise ValueError(f"expected one registry row for {key!r}; found {len(matches)}")
    row = dict(matches[0])
    expected = CANONICAL_CANDIDATES[key]
    if row.get("candidate_id", "").strip() != expected:
        raise ValueError(
            f"candidate mismatch for {key}: expected {expected!r}, got {row.get('candidate_id')!r}"
        )
    return SelectedOption(
        option_key=key,
        candidate_id=expected,
        material_class=row["material_class"].strip(),
        mpz_length_um=float(row["L_pz_um_recommended"]),
        mpz_n_bins=int(round(float(row["n_bins_recommended"]))),
        row=row,
        registry_path=str(source),
    )


def write_material_manifest(selected: SelectedOption, destination: str | Path) -> Path:
    """Write the selected row in the unchanged legacy MaterialManifest format.

    A zero historical cap means uncapped raw shielding in CampaignCalibratedTipEngine.
    It does not turn shielding off.
    """
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(selected.row)
    row["target_class"] = selected.option_key
    row["max_K_shield_MPa_sqrt_m"] = "0.0"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return target


def write_selection_audit(selected: SelectedOption, destination: str | Path, manifest: Path) -> Path:
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v10.1.7.6_four_option_parameter_overlay",
        "option_key": selected.option_key,
        "candidate_id": selected.candidate_id,
        "material_class": selected.material_class,
        "mpz_length_um": selected.mpz_length_um,
        "mpz_n_bins": selected.mpz_n_bins,
        "registry_path": selected.registry_path,
        "selected_material_manifest": str(manifest),
        "material_parameter_refit": False,
        "existing_2d_physics_modified": False,
        "historical_shielding_cap_MPa_sqrt_m": 0.0,
        "historical_zero_cap_semantics": "uncapped_raw_shielding",
        "exact_registry_row": selected.row,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


__all__ = [
    "CANONICAL_OPTIONS",
    "CANONICAL_CANDIDATES",
    "SelectedOption",
    "default_registry_path",
    "select_option",
    "write_material_manifest",
    "write_selection_audit",
]
