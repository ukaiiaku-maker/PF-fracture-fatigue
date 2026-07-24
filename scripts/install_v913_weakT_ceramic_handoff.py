#!/usr/bin/env python3
"""Install the exact v9.13 weak-T/FCC-like and ceramic-like handoff.

The source handoff is produced by
``export_v913_weakT_ceramic_paper_handoff.py`` in Arrhenius_FEM_CZM_MPZ.  This
installer verifies its SHA-256 and active-parameter fingerprint, combines each
active row with the already audited v10.2.25 fixed persistent-site contract, and
writes the v10.2.26 2-D registry and selection record.
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
DEFAULT_TEMPLATE = (
    ROOT
    / "arrhenius_fracture"
    / "data"
    / "materials"
    / "v10_2_25_v913_paper_campaign_registry.csv"
)
DEFAULT_OUT_REGISTRY = (
    ROOT
    / "arrhenius_fracture"
    / "data"
    / "materials"
    / "v10_2_26_v913_weakT_ceramic_registry.csv"
)
DEFAULT_OUT_SELECTION = (
    ROOT
    / "arrhenius_fracture"
    / "data"
    / "materials"
    / "v10_2_26_v913_weakT_ceramic_selection.json"
)

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

EXPECTED = {
    "weakT_FCC_like": {
        "candidate_id": "v913_zeroD_sobol_0257068",
        "option_key": "v913_paper_weakT01_0257068_persistent_sites",
        "material_class": "weakT",
        "role": "paper primary weak-temperature/FCC-like",
    },
    "ceramic_like": {
        "candidate_id": "v913_zeroD_sobol_0189364",
        "option_key": "v913_paper_ceramic01_0189364_persistent_sites",
        "material_class": "ceramic",
        "role": "paper primary ceramic-like",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--template-registry", type=Path, default=DEFAULT_TEMPLATE)
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
        raise RuntimeError("handoff row lacks candidate_id")
    values: dict[str, Any] = {"candidate_id": candidate_id}
    for field in ACTIVE_FIELDS:
        if field not in row or row[field] in (None, ""):
            raise RuntimeError(f"candidate {candidate_id} lacks active field {field}")
        value = float(row[field])
        if not math.isfinite(value):
            raise RuntimeError(f"candidate {candidate_id} has nonfinite {field}={row[field]!r}")
        values[field] = value
    return values


def active_fingerprint(rows: list[Mapping[str, Any]]) -> str:
    payload = [active_payload(row) for row in sorted(rows, key=lambda item: str(item["candidate_id"]))]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metric(metadata: dict[str, Any], name: str) -> float:
    value = metadata.get("oneD_metrics", {}).get(name)
    return float(value) if value is not None else float("nan")


def _summary(material_class: str, metadata: dict[str, Any]) -> str:
    span = _metric(metadata, "K50_temperature_span_MPa_sqrt_m")
    high_loss = _metric(metadata, "high_temperature_toughness_loss_MPa_sqrt_m")
    total_rise = _metric(metadata, "median_R_rise_first_to_50_MPa_sqrt_m")
    late_rise = _metric(metadata, "median_R_rise_25_to_50_MPa_sqrt_m")
    if material_class == "weakT_FCC_like":
        return (
            "Weak-temperature/FCC-like candidate with "
            f"1-D K50 span {span:.6g} MPa sqrt(m), median first-to-50um rise "
            f"{total_rise:.6g} MPa sqrt(m), and 25-to-50um change "
            f"{late_rise:.6g} MPa sqrt(m)."
        )
    return (
        "Ceramic-like candidate with "
        f"1-D K50 span {span:.6g} MPa sqrt(m), high-temperature toughness loss "
        f"{high_loss:.6g} MPa sqrt(m), median first-to-50um rise "
        f"{total_rise:.6g} MPa sqrt(m), and 25-to-50um change "
        f"{late_rise:.6g} MPa sqrt(m)."
    )


def validate_template(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, str]]:
    fields = list(rows[0])
    missing = sorted(set(ACTIVE_FIELDS) | METADATA_FIELDS - set(fields))
    if missing:
        raise RuntimeError(f"template registry lacks fields: {missing}")
    fixed_fields = [field for field in fields if field not in set(ACTIVE_FIELDS) | METADATA_FIELDS]
    template = rows[0]
    for row in rows[1:]:
        differing = [field for field in fixed_fields if row[field] != template[field]]
        if differing:
            raise RuntimeError(
                "v10.2.25 template does not have a common fixed contract; "
                f"differing fields={differing}"
            )
    return fields, template


def build_registry(
    handoff_rows: list[dict[str, str]],
    manifest: dict[str, Any],
    template_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    fields, template = validate_template(template_rows)
    by_class = {str(row["paper_material_class"]): row for row in handoff_rows}
    metadata_by_class = {
        str(row["paper_material_class"]): row for row in manifest.get("selected", [])
    }
    if set(by_class) != set(EXPECTED):
        raise RuntimeError(
            f"handoff classes mismatch: expected={sorted(EXPECTED)}, observed={sorted(by_class)}"
        )

    target_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for material_class, expected in EXPECTED.items():
        source = by_class[material_class]
        metadata = metadata_by_class.get(material_class)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"manifest lacks selected metadata for {material_class}")
        for key in ("candidate_id", "option_key"):
            if str(source[key]) != expected[key] or str(metadata[key]) != expected[key]:
                raise RuntimeError(
                    f"{material_class} {key} mismatch: expected={expected[key]}, "
                    f"csv={source[key]}, manifest={metadata[key]}"
                )

        row: dict[str, Any] = dict(template)
        row.update(
            {
                "option_key": expected["option_key"],
                "candidate_id": expected["candidate_id"],
                "material_class": expected["material_class"],
                "role": expected["role"],
                "mechanism_summary": _summary(material_class, metadata),
                "validation_status": (
                    "v10.2.26 exact active-parameter transfer from the hash-checked "
                    "v9.13 384-candidate 1-D selection; persistent sites, no finite "
                    "source inventory, no source refresh, and no explicit recovery."
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
        target_rows.append(row)
        selected.append(
            {
                **metadata,
                "material_class_2d": expected["material_class"],
                "role_2d": expected["role"],
                "mechanism_summary": row["mechanism_summary"],
            }
        )
    return fields, target_rows, selected


def main() -> int:
    args = parse_args()
    handoff_csv = args.handoff_csv.expanduser().resolve()
    handoff_json = args.handoff_json.expanduser().resolve()
    template_path = args.template_registry.expanduser().resolve()
    for path in (handoff_csv, handoff_json, template_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(handoff_json.read_text())
    if manifest.get("schema") != "v9.13_weakT_ceramic_paper_handoff_v1":
        raise RuntimeError(f"unexpected handoff schema: {manifest.get('schema')}")
    observed_csv_hash = sha256_path(handoff_csv)
    expected_csv_hash = str(manifest.get("handoff_csv_sha256", ""))
    if observed_csv_hash != expected_csv_hash:
        raise RuntimeError(
            f"handoff CSV SHA-256 mismatch: expected={expected_csv_hash}, observed={observed_csv_hash}"
        )

    handoff_rows = read_csv(handoff_csv)
    observed_fingerprint = active_fingerprint(handoff_rows)
    expected_fingerprint = str(manifest.get("active_parameter_fingerprint_sha256", ""))
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "active-parameter fingerprint mismatch: "
            f"expected={expected_fingerprint}, observed={observed_fingerprint}"
        )

    template_rows = read_csv(template_path)
    fields, target_rows, selected = build_registry(handoff_rows, manifest, template_rows)
    out_registry = args.out_registry.expanduser().resolve()
    out_selection = args.out_selection.expanduser().resolve()
    write_csv(out_registry, target_rows, fields)

    selection = {
        "schema": "v10.2.26_v913_weakT_ceramic_paper_selection_v1",
        "source_handoff_schema": manifest["schema"],
        "source_handoff_csv": str(handoff_csv),
        "source_handoff_csv_sha256": observed_csv_hash,
        "source_handoff_manifest": str(handoff_json),
        "source_active_parameter_fingerprint_sha256": observed_fingerprint,
        "template_registry": str(template_path),
        "template_registry_sha256": sha256_path(template_path),
        "installed_registry": str(out_registry),
        "installed_registry_sha256": sha256_path(out_registry),
        "candidate_count": len(target_rows),
        "primary_candidates": selected,
        "fixed_closure": manifest.get("fixed_closure", {}),
        "transfer_policy": manifest.get("transfer_policy"),
    }
    out_selection.parent.mkdir(parents=True, exist_ok=True)
    out_selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    print(
        "V10226_WEAKT_CERAMIC_INSTALLED "
        f"rows={len(target_rows)} fingerprint={observed_fingerprint} "
        f"registry={out_registry} selection={out_selection}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
