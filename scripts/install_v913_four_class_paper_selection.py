#!/usr/bin/env python3
"""Install the final v9.13 four-class paper parameter registry.

The installer preserves the accepted v10.2.25 peak and classical-DBTT primary
rows, verifies the hash-checked weak-T/ceramic primary handoff, installs exactly
one strict primary row for each of those classes, and records the second-best
strict rows as backups in the selection manifest.  It changes no mechanics or
source-closure settings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "arrhenius_fracture" / "data" / "materials"
DEFAULT_BASE_REGISTRY = MATERIALS / "v10_2_25_v913_paper_campaign_registry.csv"
DEFAULT_BASE_SELECTION = MATERIALS / "v10_2_25_v913_paper_campaign_selection.json"
DEFAULT_OUT_REGISTRY = MATERIALS / "v10_2_27_v913_four_class_paper_registry.csv"
DEFAULT_OUT_SELECTION = MATERIALS / "v10_2_27_v913_four_class_paper_selection.json"

ACTIVE_FIELDS = (
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

METADATA_FIELDS = {
    "option_key",
    "candidate_id",
    "material_class",
    "role",
    "mechanism_summary",
    "validation_status",
}

BASE_PRIMARY = {
    "peak": {
        "option_key": "v913_paper_peak01_0242980_persistent_sites",
        "candidate_id": "v913_zeroD_sobol_0242980",
    },
    "DBTT": {
        "option_key": "v913_paper_dbtt01_0202500_persistent_sites",
        "candidate_id": "v913_zeroD_sobol_0202500",
    },
}
HANDOFF_CLASSES = ("weakT_FCC_like", "ceramic_like")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--base-registry", type=Path, default=DEFAULT_BASE_REGISTRY)
    parser.add_argument("--base-selection", type=Path, default=DEFAULT_BASE_SELECTION)
    parser.add_argument("--out-registry", type=Path, default=DEFAULT_OUT_REGISTRY)
    parser.add_argument("--out-selection", type=Path, default=DEFAULT_OUT_SELECTION)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def active_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    if not candidate_id:
        raise RuntimeError("row lacks candidate_id")
    payload: dict[str, Any] = {"candidate_id": candidate_id}
    for field in ACTIVE_FIELDS:
        if field not in row or row[field] in (None, ""):
            raise RuntimeError(f"candidate {candidate_id} lacks active field {field}")
        value = float(row[field])
        if not math.isfinite(value):
            raise RuntimeError(f"candidate {candidate_id} has nonfinite {field}={row[field]!r}")
        payload[field] = value
    return payload


def active_fingerprint(rows: list[Mapping[str, Any]]) -> str:
    payload = [
        active_payload(row)
        for row in sorted(rows, key=lambda item: str(item["candidate_id"]))
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_base_registry(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]]]:
    fields = list(rows[0])
    required = set(ACTIVE_FIELDS) | METADATA_FIELDS
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"base registry lacks fields: {missing}")
    fixed_fields = [field for field in fields if field not in required]
    template = rows[0]
    for row in rows[1:]:
        differing = [field for field in fixed_fields if row[field] != template[field]]
        if differing:
            raise RuntimeError(
                "v10.2.25 registry does not have one common fixed contract; "
                f"differing fields={differing}"
            )
    by_option = {str(row["option_key"]): row for row in rows}
    selected: dict[str, dict[str, str]] = {}
    for paper_class, expected in BASE_PRIMARY.items():
        option = expected["option_key"]
        if option not in by_option:
            raise RuntimeError(f"base registry lacks required {paper_class} option {option}")
        row = by_option[option]
        if str(row["candidate_id"]) != expected["candidate_id"]:
            raise RuntimeError(
                f"base {paper_class} candidate mismatch: expected={expected['candidate_id']}, "
                f"observed={row['candidate_id']}"
            )
        selected[paper_class] = row
    return fields, template, selected


def validate_handoff(
    rows: list[dict[str, str]], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if manifest.get("schema") != "v9.13_weakT_ceramic_100um_final_selection_v1":
        raise RuntimeError(f"unexpected handoff schema: {manifest.get('schema')}")
    if len(rows) != 2:
        raise RuntimeError(f"primary handoff must contain exactly two rows, found {len(rows)}")
    by_class = {str(row.get("paper_material_class", "")): row for row in rows}
    if set(by_class) != set(HANDOFF_CLASSES):
        raise RuntimeError(
            f"handoff classes mismatch: expected={sorted(HANDOFF_CLASSES)}, "
            f"observed={sorted(by_class)}"
        )
    metadata = {
        str(row.get("paper_material_class", "")): row
        for row in manifest.get("primary_candidates", [])
    }
    if set(metadata) != set(HANDOFF_CLASSES):
        raise RuntimeError("manifest must contain one primary candidate for each handoff class")
    for material_class in HANDOFF_CLASSES:
        row = by_class[material_class]
        meta = metadata[material_class]
        for key in ("candidate_id", "option_key"):
            if str(row.get(key)) != str(meta.get(key)):
                raise RuntimeError(
                    f"{material_class} {key} mismatch between CSV and manifest"
                )
        if int(float(row.get("final_class_rank", 0))) != 1:
            raise RuntimeError(f"{material_class} handoff row is not rank 1")
        if not as_bool(row.get("oneD_strict_gate_passed")):
            raise RuntimeError(f"{material_class} primary did not pass the strict 1-D gate")
        active_payload(row)
    return by_class, metadata


def install_rows(
    base_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    fields, template, base_selected = validate_base_registry(base_rows)
    handoff_by_class, metadata_by_class = validate_handoff(handoff_rows, manifest)
    installed: list[dict[str, Any]] = []
    selected_metadata: list[dict[str, Any]] = []

    for paper_class in ("peak", "DBTT"):
        row = dict(base_selected[paper_class])
        installed.append(row)
        selected_metadata.append(
            {
                "paper_material_class": paper_class,
                "option_key": row["option_key"],
                "candidate_id": row["candidate_id"],
                "selection_role": "paper primary",
                "source": "v10.2.25 accepted 2-D transfer",
                "mechanism_summary": row["mechanism_summary"],
            }
        )

    for material_class in HANDOFF_CLASSES:
        source = handoff_by_class[material_class]
        metadata = metadata_by_class[material_class]
        row: dict[str, Any] = dict(template)
        row.update(
            {
                "option_key": str(source["option_key"]),
                "candidate_id": str(source["candidate_id"]),
                "material_class": "weakT" if material_class == "weakT_FCC_like" else "ceramic",
                "role": "paper primary weak-temperature/FCC-like" if material_class == "weakT_FCC_like" else "paper primary ceramic-like",
                "mechanism_summary": (
                    "Strict 100 um 1-D weak-temperature/FCC-like primary selected by minimum class score."
                    if material_class == "weakT_FCC_like"
                    else "Strict 100 um 1-D ceramic-like primary selected by minimum class score."
                ),
                "validation_status": (
                    "v10.2.27 exact active-parameter transfer from the final v9.13 "
                    "five-temperature 100 um 1-D selection; 2-D validation required."
                ),
            }
        )
        for field in ACTIVE_FIELDS:
            row[field] = float(source[field])
        for field in (
            "source_recovery_rate_s",
            "retained_recovery_rate_s",
            "source_refresh_length_um",
            "recovery_nu0_s",
            "recovery_H0_eV",
            "recovery_activation_entropy_kB",
            "legacy_source_sites_active",
            "legacy_source_refresh_active",
            "explicit_recovery_active",
        ):
            if float(row[field]) != 0.0:
                raise RuntimeError(f"fixed persistent-site closure requires {field}=0")
        installed.append(row)
        selected_metadata.append(
            {
                **metadata,
                "material_class_2d": row["material_class"],
                "role_2d": row["role"],
                "mechanism_summary": row["mechanism_summary"],
                "source": "v9.13 strict five-temperature 100 um 1-D final selection",
            }
        )

    if len(installed) != 4 or len({row["option_key"] for row in installed}) != 4:
        raise RuntimeError("four-class registry must contain four unique options")
    return fields, installed, selected_metadata


def main() -> int:
    args = parse_args()
    paths = [
        args.handoff_csv.expanduser().resolve(),
        args.handoff_json.expanduser().resolve(),
        args.base_registry.expanduser().resolve(),
        args.base_selection.expanduser().resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    handoff_csv, handoff_json, base_registry, base_selection = paths
    manifest = json.loads(handoff_json.read_text())

    observed_csv_hash = sha256_path(handoff_csv)
    if observed_csv_hash != str(manifest.get("primary_handoff_csv_sha256", "")):
        raise RuntimeError("primary handoff CSV SHA-256 mismatch")
    handoff_rows = read_csv(handoff_csv)
    observed_fingerprint = active_fingerprint(handoff_rows)
    if observed_fingerprint != str(
        manifest.get("primary_active_parameter_fingerprint_sha256", "")
    ):
        raise RuntimeError("primary active-parameter fingerprint mismatch")

    base_rows = read_csv(base_registry)
    fields, installed, primary_metadata = install_rows(base_rows, handoff_rows, manifest)
    out_registry = args.out_registry.expanduser().resolve()
    out_selection = args.out_selection.expanduser().resolve()
    write_csv(out_registry, installed, fields)

    selection = {
        "schema": "v10.2.27_v913_four_class_paper_selection_v1",
        "candidate_count": 4,
        "paper_class_order": ["peak", "DBTT", "weakT_FCC_like", "ceramic_like"],
        "primary_candidates": primary_metadata,
        "weakT_ceramic_backup_candidates": manifest.get("backup_candidates", []),
        "source_peak_dbtt_registry": str(base_registry),
        "source_peak_dbtt_registry_sha256": sha256_path(base_registry),
        "source_peak_dbtt_selection": str(base_selection),
        "source_peak_dbtt_selection_sha256": sha256_path(base_selection),
        "source_weakT_ceramic_handoff": str(handoff_csv),
        "source_weakT_ceramic_handoff_sha256": observed_csv_hash,
        "source_weakT_ceramic_manifest": str(handoff_json),
        "source_active_parameter_fingerprint_sha256": observed_fingerprint,
        "installed_registry": str(out_registry),
        "installed_registry_sha256": sha256_path(out_registry),
        "fixed_closure": manifest.get("fixed_closure", {}),
        "transfer_policy": manifest.get("transfer_policy"),
        "mechanics_changed": False,
        "stochastic_cleavage_law_changed": False,
    }
    out_selection.parent.mkdir(parents=True, exist_ok=True)
    out_selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print("V10227_FOUR_CLASS_PAPER_SELECTION_INSTALLED")
    for row in installed:
        print(
            f"option={row['option_key']} candidate={row['candidate_id']} "
            f"material_class={row['material_class']}"
        )
    print(f"registry={out_registry} selection={out_selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
