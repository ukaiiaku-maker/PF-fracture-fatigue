#!/usr/bin/env python3
"""Generate the six-entry opt-in V2 naming and evidence layer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.canonical_v2_registry_v10230 import canonical_hash, load_rows


CATALOG = ROOT / "v2_named_parameterization_registry.json"
SUMMARY = ROOT / "v2_named_parameterization_registry.csv"
DOCUMENT = ROOT / "V2_PARAMETERIZATION_CATALOG.md"
PROVENANCE = ROOT / "v2_named_parameterization_provenance.json"
CARDS = ROOT / "v2_parameterization_evidence_cards"
PARENT = "c1b3f08d956242844ee1206bf10576f08f8a2859"
SOURCE_COMMIT = "6d56fd936699d244004d5e66c35d3718d8fcc357"
TRANSFER_COMMIT = "7bb4a341b1c5370c826a3a6bc0253c042fd2e6fc"
SUPPORTED_LOADERS = [
    "analytical", "monotonic_reduced_1D", "fatigue_reduced_1D",
    "PF_sharp_front", "FEM_CZM", "checkpoint_serialization",
]
ABSENT = ["Peak_V2", "weakT_V2", "ceramic_V2"]
UNCHANGED_PARENT_FILES = {
    "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv": "4ba723c80abcfdd7101cee0afaa9b4104eebf2a8e847892528128664829c7760",
    "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json": "3267ac05595cee0800a81a6300e80cb0e71b216067c35cc7a50d82ac4c70e4a2",
    "arrhenius_fracture/sharp_front_v10_2_27.py": "ce1663e68be0695fb6616e886af8a1135ec6f774ff612956972ec9e813bae95a",
    "arrhenius_fracture/parameter_registry_v9111.py": "6946759c5786344a7111b9e25ce8b2513e5335d78fc590ba860fc730bb7d83a7",
    "arrhenius_fracture/persistent_site_high_cycle_checkpoint_v10230.py": "0af1a7964012047ade46e822906669f7778b699b36b68b8738cbe8d0c00c0d05",
}
LEGACY_ROW_HASHES = {
    "Peak": "83245aa3a01450f08d13820f6a042d1c01518748fbf790b6082879a9ed6fdeb1",
    "DBTT": "6d2d454e3c79e2171b8547c895a0ee2d42fa4897af90354efe906e7b27d044d4",
    "weakT": "cfc86642687cac968223e4d3ac202c9ff14648c37982d3f188326f0ff25da1e5",
    "ceramic": "d6abae369475fa80722bc3232603815a73f5ebc19433339dd73a2a84c0a1c3db",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specifications() -> list[dict[str, Any]]:
    return [
        {"display_name": "DBTT V2", "registry_alias": "DBTT_V2", "source_candidate_id": "P25_TJBSV2_S_002987", "parent_background": "P25", "status": "NAMED_V2_PROSPECTIVE_PARAMETERIZATION", "temperature_response_classification": "DBTT_LIKE_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "1D_STRONG_RISING_RESISTANCE", "fatigue_status": "THREE_FINITE_EXACT_ROW_SHORT_GROWTH_POINTS_CURVED", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Reduced 1-D rising resistance through 100 um with exact restart parity and three finite curved exact-row 300 K short-growth points.", "missing_evidence": "Developed post-initiation fatigue and spatial R-curve transfer."},
        {"display_name": "DBTT V2-P40", "registry_alias": "DBTT_V2_P40", "source_candidate_id": "P40_TJBSV2_S_038503", "parent_background": "P40", "status": "NAMED_V2_ALTERNATE_PARAMETERIZATION", "temperature_response_classification": "DBTT_LIKE_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "UNRESOLVED_AT_80_MPa_sqrt_m_CENSOR", "fatigue_status": "NOT_TESTED_EXACT_ROW", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Confirmed coupled DBTT-like reduced production response.", "missing_evidence": "Rising resistance beyond the 80 MPa sqrt(m) censor, exact-row fatigue, and spatial transfer."},
        {"display_name": "weak-T V2-P25", "registry_alias": "weakT_V2_P25", "source_candidate_id": "P25_TJBSV2_S_026704", "parent_background": "P25", "status": "NAMED_V2_ALTERNATE_PARAMETERIZATION", "temperature_response_classification": "WEAK_T_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "1D_EFFECTIVE_RESISTANCE_FLAT", "fatigue_status": "NOT_TESTED_EXACT_ROW", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Confirmed coupled weak-T reduced production response.", "missing_evidence": "Exact-row fatigue and spatial transfer."},
        {"display_name": "weak-T V2-P40", "registry_alias": "weakT_V2_P40", "source_candidate_id": "P40_TJBSV2_S_038278", "parent_background": "P40", "status": "NAMED_V2_ALTERNATE_PARAMETERIZATION", "temperature_response_classification": "WEAK_T_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "1D_EFFECTIVE_RESISTANCE_FLAT", "fatigue_status": "NOT_TESTED_EXACT_ROW", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Confirmed coupled weak-T reduced production response.", "missing_evidence": "Exact-row fatigue and spatial transfer."},
        {"display_name": "ceramic-like V2-P25", "registry_alias": "ceramic_V2_P25", "source_candidate_id": "P25_TJBSV2_S_004127", "parent_background": "P25", "status": "NAMED_V2_ALTERNATE_PARAMETERIZATION", "temperature_response_classification": "CERAMIC_LIKE_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "1D_EFFECTIVE_RESISTANCE_FLAT", "fatigue_status": "NOT_TESTED_EXACT_ROW", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Confirmed coupled ceramic-like reduced production response.", "missing_evidence": "Exact-row fatigue and spatial transfer."},
        {"display_name": "ceramic-like V2-P40", "registry_alias": "ceramic_V2_P40", "source_candidate_id": "P40_TJBSV2_S_031027", "parent_background": "P40", "status": "NAMED_V2_ALTERNATE_PARAMETERIZATION", "temperature_response_classification": "CERAMIC_LIKE_CONFIRMED_REDUCED_PRODUCTION", "rising_resistance_status": "1D_EFFECTIVE_RESISTANCE_FLAT", "fatigue_status": "NOT_TESTED_EXACT_ROW", "developed_fatigue_status": "UNRESOLVED", "spatial_transfer_status": "NOT_TESTED", "principal_evidence": "Confirmed coupled ceramic-like reduced production response.", "missing_evidence": "Exact-row fatigue and spatial transfer."},
    ]


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_rows()
    entries = []
    for spec in specifications():
        row = rows[spec["source_candidate_id"]]
        renewal = {
            "cleavage_hits": float(row["physics__cleavage_hits"]),
            "cleavage_correlation_time_s": float(row["physics__cleavage_correlation_time_s"]),
        }
        entries.append({
            **{key: value for key, value in spec.items() if key not in {"principal_evidence", "missing_evidence"}},
            "generation": "CORRECTED_JOINT_THERMODYNAMIC_V2",
            "source_commit": SOURCE_COMMIT,
            "transfer_commit": TRANSFER_COMMIT,
            "complete_bound_row_sha256": row["complete_bound_row_sha256"],
            "thermodynamic_surface_fingerprint": canonical_hash({"opening": row["opening_surface_sha256"], "emission": row["emission_surface_sha256"]}),
            "renewal_fingerprint": canonical_hash(renewal),
            "entropy_convention": row["entropy_representation"],
            "response_evidence": {"principal": spec["principal_evidence"], "missing": spec["missing_evidence"]},
            "limitations": ["NOT_EXPERIMENTALLY_CALIBRATED", "NOT_ASTM_KIC", "NOT_CONVENTIONAL_R_CURVE", "DEVELOPED_FATIGUE_RATE_PENDING", "SPATIAL_R_CURVE_TRANSFER_PENDING", "NOT_DEFAULT_CANONICAL_PARAMETERIZATION"],
            "cross_code_loaders": SUPPORTED_LOADERS,
        })
    catalog = {
        "schema": "v10.2.30_opt_in_named_v2_parameterizations_v1",
        "catalog_version": "2.0.1",
        "accepted_parent_commit": PARENT,
        "generation": "CORRECTED_JOINT_THERMODYNAMIC_V2",
        "policy": "Separate additive V2 overlay; no established parameterization is represented or intercepted.",
        "named_aliases": [item["registry_alias"] for item in entries],
        "reserved_absent_aliases": ABSENT,
        "exact_only_controls": ["P25_TJBSV2_S_064036", "P40_TJBSV2_S_043821"],
        "entries": entries,
    }
    source_rows = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_rows.csv"
    source_manifest = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_manifest.json"
    provenance = {
        "schema": "v10.2.30_opt_in_named_v2_provenance_v1",
        "accepted_parent_commit": PARENT,
        "corrected_source_rows_sha256": sha(source_rows),
        "corrected_source_manifest_sha256": sha(source_manifest),
        "unchanged_parent_files_sha256": UNCHANGED_PARENT_FILES,
        "unchanged_established_row_hashes": LEGACY_ROW_HASHES,
        "established_rows_copied_into_v2_catalog": False,
        "established_aliases_added": [],
        "default_selection_changed": False,
        "physics_changed": False,
        "simulations_run": 0,
    }
    return catalog, provenance


def render(catalog: dict[str, Any], provenance: dict[str, Any]) -> dict[Path, str]:
    output: dict[Path, str] = {
        CATALOG: json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        PROVENANCE: json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    }
    fields = ["display_name", "registry_alias", "source_candidate_id", "parent_background", "status", "complete_bound_row_sha256", "temperature_response_classification", "rising_resistance_status", "fatigue_status", "developed_fatigue_status", "spatial_transfer_status", "limitations"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entry in catalog["entries"]:
        writer.writerow({key: json.dumps(entry[key]) if key == "limitations" else entry[key] for key in fields})
    output[SUMMARY] = stream.getvalue()
    cards = []
    for entry in catalog["entries"]:
        card = CARDS / f"{entry['registry_alias']}.md"
        cards.append(f"- [{entry['display_name']}](v2_parameterization_evidence_cards/{card.name})")
        output[card] = (
            f"# {entry['display_name']}\n\n"
            f"- **Alias:** `{entry['registry_alias']}`\n"
            f"- **Exact source:** `{entry['source_candidate_id']}`\n"
            f"- **Parent:** `{entry['parent_background']}`\n"
            f"- **Status:** `{entry['status']}`\n"
            f"- **Response:** `{entry['temperature_response_classification']}`\n"
            f"- **Principal evidence:** {entry['response_evidence']['principal']}\n"
            f"- **Missing evidence:** {entry['response_evidence']['missing']}\n"
            f"- **Recommended use:** explicit prospective V2 comparison through the alias or exact candidate ID\n"
            f"- **Prohibited claims:** experimental calibration, ASTM KIC, conventional or spatial R-curve qualification, canonical replacement\n"
            f"- **Source commit:** `{entry['source_commit']}`\n"
            f"- **Complete row hash:** `{entry['complete_bound_row_sha256']}`\n"
        )
    output[DOCUMENT] = (
        "# Opt-in corrected V2 parameterizations\n\n"
        "This is a separate additive naming layer over the immutable corrected-search candidate source. "
        "It contains only six new V2 aliases and does not represent, wrap, rename, document, or intercept any established parameterization. "
        "Existing lookups and default selection remain outside this module and unchanged.\n\n"
        "`CORRECTED_JOINT_THERMODYNAMIC_V2` denotes the corrected negative-emission-entropy coupled opening/emission/transport search generation. "
        "Every alias loads its complete full-precision source row without defaults or rounding.\n\n"
        "No `Peak_V2` exists because no Peak-T candidate passed the frozen F1B/F2R search. "
        "Preferred `weakT_V2` and `ceramic_V2` aliases remain unassigned. The two accessible controls are callable only by exact candidate ID.\n\n"
        "## V2 evidence cards\n\n" + "\n".join(cards) + "\n\n"
        "No V2 row is experimentally calibrated. `DBTT_V2` is prospective, has reduced 1-D rising-resistance and finite short-growth evidence, is not an ASTM or conventional R-curve, and has not earned canonical production promotion.\n"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog, provenance = build()
    outputs = render(catalog, provenance)
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if not path.is_file() or path.read_text() != text]
        if stale:
            raise RuntimeError(f"generated V2 catalog outputs are missing or stale: {stale}")
    else:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print(json.dumps({"status": "PASS", "named_v2_aliases": 6, "exact_only_controls": 2, "established_rows_touched": 0, "simulations_run": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
