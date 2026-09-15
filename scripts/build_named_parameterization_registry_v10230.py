#!/usr/bin/env python3
"""Build the additive named Legacy and corrected V2 parameterization catalog."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.canonical_v2_registry_v10230 import canonical_hash, load_rows


LEGACY_SOURCE = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
JSON_OUT = ROOT / "named_parameterization_registry.json"
CSV_OUT = ROOT / "named_parameterization_registry.csv"
DOC_OUT = ROOT / "PARAMETERIZATION_CATALOG.md"
CARDS = ROOT / "parameterization_evidence_cards"
SUPPORTED_LOADERS = [
    "analytical", "monotonic_reduced_1D", "fatigue_reduced_1D",
    "PF_sharp_front", "FEM_CZM", "checkpoint_serialization",
]
LEGACY_COMMIT = "503943fe7d83c06096c61443cd896e07458d7e24"
CORRECTED_SEARCH_COMMIT = "6d56fd936699d244004d5e66c35d3718d8fcc357"
TRANSFER_COMMIT = "7bb4a341b1c5370c826a3a6bc0253c042fd2e6fc"
V23_COMMIT = "c1b3f08d956242844ee1206bf10576f08f8a2859"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_rows() -> dict[str, dict[str, str]]:
    with LEGACY_SOURCE.open(newline="") as stream:
        return {row["candidate_id"]: row for row in csv.DictReader(stream)}


def fingerprints(row: dict[str, str], generation: str) -> tuple[str, str, str]:
    if generation == "LEGACY":
        thermo = canonical_hash({key: row[key] for key in row if key.startswith(("cleave_", "emit_"))})
        renewal = {
            "scope": "CANONICAL_ENGINE_CONTRACT_NOT_ROW_BOUND",
            "cleavage_hits": 3.0,
            "cleavage_correlation_time_s": 1.0e-6,
        }
        return thermo, canonical_hash(renewal), "LEGACY_BARRIER_PARAMETERIZATION"
    thermo = canonical_hash({
        "opening": row["opening_surface_sha256"],
        "emission": row["emission_surface_sha256"],
    })
    renewal = {
        "scope": "COMPLETE_ROW_BOUND",
        "cleavage_hits": float(row["physics__cleavage_hits"]),
        "cleavage_correlation_time_s": float(row["physics__cleavage_correlation_time_s"]),
    }
    return thermo, canonical_hash(renewal), row["entropy_representation"]


def entry(spec: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    generation = spec["generation"]
    thermo, renewal, entropy = fingerprints(row, generation)
    bound_hash = canonical_hash(row) if generation == "LEGACY" else row["complete_bound_row_sha256"]
    return {
        **spec,
        "callable": True,
        "complete_bound_row_sha256": bound_hash,
        "row_hash_origin": (
            "CATALOG_CANONICAL_JSON_OF_EXACT_LEGACY_CSV_ROW"
            if generation == "LEGACY" else "SEALED_CORRECTED_V2_COMPLETE_BOUND_ROW"
        ),
        "thermodynamic_surface_fingerprint": thermo,
        "renewal_fingerprint": renewal,
        "entropy_convention": entropy,
        "cross_code_loaders": SUPPORTED_LOADERS,
        "full_precision_row": row,
    }


def specs() -> list[dict[str, Any]]:
    legacy = [
        ("Peak", "Peak", "Peak_Legacy", "v913_zeroD_sobol_0242980", "peak"),
        ("DBTT", "DBTT", "DBTT_Legacy", "v913_zeroD_sobol_0202500", "DBTT"),
        ("weak-T", "weakT", "weakT_Legacy", "v913_zeroD_sobol_0129902", "weak-T"),
        ("ceramic-like", "ceramic", "ceramic_Legacy", "v913_zeroD_sobol_0077080", "ceramic-like"),
    ]
    result: list[dict[str, Any]] = []
    for display, alias, explicit, candidate, response in legacy:
        result.append({
            "display_name": display,
            "registry_alias": alias,
            "aliases": [alias, explicit],
            "source_candidate_id": candidate,
            "generation": "LEGACY",
            "parent_background": "ESTABLISHED_CANONICAL_FOUR_CLASS",
            "status": "LEGACY_CANONICAL_PARAMETERIZATION",
            "canonical_default": True,
            "legacy_predecessor_alias": None,
            "source_commit": LEGACY_COMMIT,
            "transfer_source_commit": LEGACY_COMMIT,
            "temperature_response_classification": f"LEGACY_{response.upper().replace('-', '_')}",
            "rising_resistance_status": "ESTABLISHED_CAMPAIGN_PROVENANCE",
            "fatigue_status": "ESTABLISHED_CANONICAL_CAMPAIGN_ROW",
            "developed_fatigue_status": "NOT_SEPARATELY_CATALOGED",
            "spatial_transfer_status": "COMPLETED_CANONICAL_PF_CAMPAIGN",
            "response_evidence": {
                "principal": "Existing four-class canonical PF campaign: 288 conditions and 432,710 accepted states",
                "missing": "No V2 corrected joint-search qualification is implied",
                "recommended_uses": ["legacy campaign reproduction", "publication comparison", "default four-class studies"],
                "prohibited_claims": ["experimental calibration", "equivalence to a corrected V2 row"],
            },
            "limitations": ["NOT_EXPERIMENTALLY_CALIBRATED", "LEGACY_GENERATION"],
        })
    v2 = [
        ("DBTT V2", "DBTT_V2", "P25_TJBSV2_S_002987", "P25", "NAMED_V2_PROSPECTIVE_PARAMETERIZATION",
         "DBTT_LIKE_CONFIRMED_REDUCED_PRODUCTION", "1D_STRONG_RISING_RESISTANCE",
         "THREE_FINITE_EXACT_ROW_SHORT_GROWTH_POINTS_CURVED", "UNRESOLVED", "NOT_TESTED", "DBTT"),
        ("DBTT V2-P40", "DBTT_V2_P40", "P40_TJBSV2_S_038503", "P40", "NAMED_V2_ALTERNATE_PARAMETERIZATION",
         "DBTT_LIKE_CONFIRMED_REDUCED_PRODUCTION", "UNRESOLVED_AT_80_MPa_sqrt_m_CENSOR",
         "NOT_TESTED_EXACT_ROW", "UNRESOLVED", "NOT_TESTED", "DBTT"),
        ("weak-T V2-P25", "weakT_V2_P25", "P25_TJBSV2_S_026704", "P25", "NAMED_V2_ALTERNATE_PARAMETERIZATION",
         "WEAK_T_CONFIRMED_REDUCED_PRODUCTION", "1D_EFFECTIVE_RESISTANCE_FLAT",
         "NOT_TESTED_EXACT_ROW", "UNRESOLVED", "NOT_TESTED", "weakT"),
        ("weak-T V2-P40", "weakT_V2_P40", "P40_TJBSV2_S_038278", "P40", "NAMED_V2_ALTERNATE_PARAMETERIZATION",
         "WEAK_T_CONFIRMED_REDUCED_PRODUCTION", "1D_EFFECTIVE_RESISTANCE_FLAT",
         "NOT_TESTED_EXACT_ROW", "UNRESOLVED", "NOT_TESTED", "weakT"),
        ("ceramic-like V2-P25", "ceramic_V2_P25", "P25_TJBSV2_S_004127", "P25", "NAMED_V2_ALTERNATE_PARAMETERIZATION",
         "CERAMIC_LIKE_CONFIRMED_REDUCED_PRODUCTION", "1D_EFFECTIVE_RESISTANCE_FLAT",
         "NOT_TESTED_EXACT_ROW", "UNRESOLVED", "NOT_TESTED", "ceramic"),
        ("ceramic-like V2-P40", "ceramic_V2_P40", "P40_TJBSV2_S_031027", "P40", "NAMED_V2_ALTERNATE_PARAMETERIZATION",
         "CERAMIC_LIKE_CONFIRMED_REDUCED_PRODUCTION", "1D_EFFECTIVE_RESISTANCE_FLAT",
         "NOT_TESTED_EXACT_ROW", "UNRESOLVED", "NOT_TESTED", "ceramic"),
    ]
    for display, alias, candidate, parent, status, coupled, rising, fatigue, developed, spatial, predecessor in v2:
        result.append({
            "display_name": display,
            "registry_alias": alias,
            "aliases": [alias],
            "source_candidate_id": candidate,
            "generation": "CORRECTED_JOINT_THERMODYNAMIC_V2",
            "parent_background": parent,
            "status": status,
            "canonical_default": False,
            "legacy_predecessor_alias": predecessor,
            "source_commit": CORRECTED_SEARCH_COMMIT,
            "transfer_source_commit": TRANSFER_COMMIT,
            "temperature_response_classification": coupled,
            "rising_resistance_status": rising,
            "fatigue_status": fatigue,
            "developed_fatigue_status": developed,
            "spatial_transfer_status": spatial,
            "response_evidence": {
                "coupled_temperature_response": coupled,
                "one_d_rising_resistance": rising,
                "one_d_fatigue": fatigue,
                "developed_fatigue_rate": developed,
                "spatial_r_curve_transfer": spatial,
                "principal": (
                    "Reduced 1-D K_reinit rose from 35.674 to 53.966 MPa sqrt(m) through 50 um and to 63.603 MPa sqrt(m) through 100 um; opening-only control was flat; restart was exact; three exact-row short-growth points were finite and curved"
                    if alias == "DBTT_V2" else (
                        "Corrected coupled DBTT-like response; reload-separated rising resistance remained unresolved at the 80 MPa sqrt(m) censor"
                        if alias == "DBTT_V2_P40" else "Confirmed corrected coupled response classification; reduced 1-D effective resistance was flat"
                    )
                ),
                "missing": (
                    "Developed fatigue rate and spatial R-curve transfer remain incomplete"
                    if alias == "DBTT_V2" else "Exact-row fatigue and spatial R-curve transfer have not been tested"
                ),
                "recommended_uses": ["explicit prospective comparison", "cross-code parameter study"],
                "prohibited_claims": ["experimental calibration", "ASTM KIC", "conventional R-curve", "canonical replacement"],
            },
            "limitations": [
                "NOT_EXPERIMENTALLY_CALIBRATED", "NOT_ASTM_KIC", "NOT_CONVENTIONAL_R_CURVE",
                "DEVELOPED_FATIGUE_RATE_PENDING", "SPATIAL_TRANSFER_PENDING",
            ],
        })
    return result


def build() -> dict[str, Any]:
    legacy = legacy_rows()
    v2 = load_rows()
    entries = []
    for spec in specs():
        sources = legacy if spec["generation"] == "LEGACY" else v2
        entries.append(entry(spec, sources[spec["source_candidate_id"]]))
    alias_index = {alias: item["registry_alias"] for item in entries for alias in item["aliases"]}
    return {
        "schema": "v10.2.30_named_parameterization_registry_v1",
        "catalog_version": "2.0.0",
        "accepted_parent_commit": V23_COMMIT,
        "generations": {
            "LEGACY": "Established four-class rows used by the completed canonical PF campaign.",
            "CORRECTED_JOINT_THERMODYNAMIC_V2": (
                "Corrected negative-emission-entropy coupled opening/emission/transport search generation. "
                "The V2 label is independent of historical internal filenames containing v2."
            ),
        },
        "status_definitions": {
            "LEGACY_CANONICAL_PARAMETERIZATION": "Established canonical campaign row.",
            "NAMED_V2_PROSPECTIVE_PARAMETERIZATION": "Primary named V2 row with prospective evidence.",
            "NAMED_V2_ALTERNATE_PARAMETERIZATION": "Named V2 family alternate without preferred-primary status.",
            "ARCHIVAL_RESEARCH_CANDIDATE": "Retained for research provenance and excluded from active named selection.",
        },
        "default_four_aliases": ["Peak", "DBTT", "weakT", "ceramic"],
        "alias_index": alias_index,
        "absent_reserved_aliases": ["Peak_V2", "weakT_V2", "ceramic_V2"],
        "policy": "V2 is a prospective successor generation under evaluation.",
        "legacy_source_sha256": file_sha(LEGACY_SOURCE),
        "entries": entries,
    }


def write_outputs(payload: dict[str, Any]) -> None:
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = [
        "display_name", "registry_alias", "aliases", "source_candidate_id", "generation",
        "parent_background", "status", "canonical_default", "legacy_predecessor_alias",
        "source_commit", "complete_bound_row_sha256", "thermodynamic_surface_fingerprint",
        "renewal_fingerprint", "entropy_convention", "temperature_response_classification",
        "rising_resistance_status", "fatigue_status", "developed_fatigue_status",
        "spatial_transfer_status", "limitations",
    ]
    with CSV_OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in payload["entries"]:
            writer.writerow({key: json.dumps(item.get(key)) if key in {"aliases", "limitations"} else item.get(key) for key in fields})
    CARDS.mkdir(exist_ok=True)
    expected_cards = set()
    for item in payload["entries"]:
        path = CARDS / f"{item['registry_alias']}.md"
        expected_cards.add(path)
        ev = item["response_evidence"]
        path.write_text(
            f"# {item['display_name']}\n\n"
            f"- **Aliases:** {', '.join(f'`{alias}`' for alias in item['aliases'])}\n"
            f"- **Exact source:** `{item['source_candidate_id']}`\n"
            f"- **Generation:** `{item['generation']}`\n"
            f"- **Response class:** `{item['temperature_response_classification']}`\n"
            f"- **Status:** `{item['status']}`\n"
            f"- **Principal evidence:** {ev['principal']}\n"
            f"- **Missing evidence:** {ev['missing']}\n"
            f"- **Recommended uses:** {', '.join(ev['recommended_uses'])}\n"
            f"- **Prohibited claims:** {', '.join(ev['prohibited_claims'])}\n"
            f"- **Predecessor/comparison row:** `{item.get('legacy_predecessor_alias') or 'none'}`\n"
            f"- **Source commit:** `{item['source_commit']}`\n"
            f"- **Complete row hash:** `{item['complete_bound_row_sha256']}`\n"
        )
    for stale in CARDS.glob("*.md"):
        if stale not in expected_cards:
            stale.unlink()
    rows = {item["registry_alias"]: item for item in payload["entries"]}
    comparisons = [("DBTT", "DBTT_V2 / DBTT_V2_P40"), ("weak-T", "weakT_V2_P25 / weakT_V2_P40"), ("ceramic-like", "ceramic_V2_P25 / ceramic_V2_P40")]
    comparison_lines = [f"| {family} | `{legacy}` | `{v2names}` | Explicit opt-in; preferred unsuffixed V2 alias is assigned only for DBTT |" for family, v2names in comparisons for legacy in ({"DBTT":"DBTT","weak-T":"weakT","ceramic-like":"ceramic"}[family],)]
    cards = "\n".join(f"- [{item['display_name']}](parameterization_evidence_cards/{item['registry_alias']}.md)" for item in payload["entries"])
    DOC_OUT.write_text(
        "# Named response parameterizations, catalog V2.0.0\n\n"
        "`LEGACY` identifies the established four-class parameter set used by the completed canonical PF campaign. "
        "`CORRECTED_JOINT_THERMODYNAMIC_V2` identifies the corrected negative-emission-entropy, coupled opening/emission/transport search generation. "
        "The displayed V2 designation is a scientific parameterization generation and is independent of historical internal filenames such as `oneD_v2_focused_*`.\n\n"
        "The unsuffixed aliases `Peak`, `DBTT`, `weakT`, and `ceramic` retain their established resolution. "
        "V2 is a prospective successor generation under evaluation. It does not supersede Legacy and is available only through explicit aliases. "
        "No `Peak_V2` alias is assigned because no Peak-T candidate passed F1B or F2R in the corrected frozen search domain. "
        "No preferred `weakT_V2` or `ceramic_V2` alias is assigned while their P25 and P40 alternates remain under evaluation.\n\n"
        "This catalog changes naming, lookup, and provenance only. It changes no physics, runs no simulation, and leaves the canonical source registry byte-identical.\n\n"
        "## Legacy versus V2\n\n"
        "| Family | Legacy alias | V2 aliases | Selection policy |\n|---|---|---|---|\n"
        + "\n".join(comparison_lines) + "\n\n"
        "## Evidence cards\n\n" + cards + "\n\n"
        "The JSON registry is authoritative for naming metadata and embeds exact source-row strings. "
        "At load time, every embedded row and hash is checked against its sealed source registry.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build()
    if args.check:
        expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if not JSON_OUT.is_file() or JSON_OUT.read_text() != expected:
            raise RuntimeError("named parameterization JSON is missing or stale")
        before = {path: path.read_bytes() for path in (JSON_OUT, CSV_OUT, DOC_OUT, *sorted(CARDS.glob("*.md")))}
        write_outputs(payload)
        after = {path: path.read_bytes() for path in before}
        if before != after:
            raise RuntimeError("named parameterization generated outputs are not deterministic")
    else:
        write_outputs(payload)
    print(json.dumps({"status": "PASS", "entries": len(payload["entries"]), "aliases": len(payload["alias_index"]), "simulations_run": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
