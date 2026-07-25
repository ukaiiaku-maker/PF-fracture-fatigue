#!/usr/bin/env python3
"""Build the exact four-class v10.2.27 registry from audited source registries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "arrhenius_fracture" / "data" / "materials"

PEAK_DBTT_REGISTRY = MATERIALS / "v10_2_25_v913_paper_campaign_registry.csv"
PEAK_DBTT_SELECTION = MATERIALS / "v10_2_25_v913_paper_campaign_selection.json"
WEAKT_CERAMIC_REGISTRY = MATERIALS / "v10_2_26_v913_weakT_ceramic_registry.csv"
WEAKT_CERAMIC_SELECTION = MATERIALS / "v10_2_26_v913_weakT_ceramic_selection.json"

DEFAULT_OUTPUT_REGISTRY = MATERIALS / "v10_2_27_paper_four_class_registry.csv"
DEFAULT_OUTPUT_SELECTION = MATERIALS / "v10_2_27_paper_four_class_selection.json"

CANONICAL_OPTIONS = (
    ("v913_paper_peak01_0242980_persistent_sites", "v913_zeroD_sobol_0242980"),
    ("v913_paper_dbtt01_0202500_persistent_sites", "v913_zeroD_sobol_0202500"),
    ("v913_paper_weakT01_0257068_persistent_sites", "v913_zeroD_sobol_0257068"),
    ("v913_paper_ceramic01_0189364_persistent_sites", "v913_zeroD_sobol_0189364"),
)

FORBIDDEN_OPTIONS = {
    "weakT_primary",
    "weakT_restart00_candidate00",
    "ceramic_primary",
    "ceramic_restart02_candidate00",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_registry(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"registry has no header: {path}")
        rows = {str(row["option_key"]): dict(row) for row in reader}
    return list(reader.fieldnames), rows


def _canonical_row_sha256(row: dict[str, str]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _selection_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(payload.get("primary_candidates", []))
    candidates.extend(payload.get("secondary_candidates", []))
    return {
        str(candidate["option_key"]): dict(candidate)
        for candidate in candidates
        if candidate.get("option_key")
    }


def _normalized_candidate(
    *,
    order: int,
    option: str,
    candidate_id: str,
    source_selection: Path,
    metadata: dict[str, Any],
    row_sha256: str,
) -> dict[str, Any]:
    if "peak01" in option:
        response_class = "peak_like"
        paper_role = "primary_peak"
        interpretation = metadata.get("interpretation")
    elif "dbtt01" in option:
        response_class = "classic_dbtt_upper_shelf"
        paper_role = "primary_dbtt"
        interpretation = metadata.get("interpretation")
    elif "weakT01" in option:
        response_class = "weak_temperature_fcc_like"
        paper_role = "primary_weak_temperature_fcc_like"
        interpretation = metadata.get("mechanism_summary")
    elif "ceramic01" in option:
        response_class = "ceramic_like"
        paper_role = "primary_ceramic_like"
        interpretation = metadata.get("mechanism_summary")
    else:
        raise ValueError(f"unrecognized canonical option: {option}")

    return {
        "paper_order": order,
        "option_key": option,
        "candidate_id": candidate_id,
        "response_class": response_class,
        "paper_role": paper_role,
        "interpretation": str(interpretation or ""),
        "active_row_sha256": row_sha256,
        "source_selection_record": str(source_selection.relative_to(ROOT)),
        "source_selection_metadata": metadata,
    }


def build_payloads() -> tuple[str, str]:
    required = (
        PEAK_DBTT_REGISTRY,
        PEAK_DBTT_SELECTION,
        WEAKT_CERAMIC_REGISTRY,
        WEAKT_CERAMIC_SELECTION,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audited source files: {missing}")

    header_a, rows_a = _read_registry(PEAK_DBTT_REGISTRY)
    header_b, rows_b = _read_registry(WEAKT_CERAMIC_REGISTRY)
    if header_a != header_b:
        raise ValueError("v10.2.25 and v10.2.26 registry headers differ")

    selection_a_payload = json.loads(PEAK_DBTT_SELECTION.read_text())
    selection_b_payload = json.loads(WEAKT_CERAMIC_SELECTION.read_text())
    selection_a = _selection_lookup(selection_a_payload)
    selection_b = _selection_lookup(selection_b_payload)

    combined_rows: list[dict[str, str]] = []
    normalized: list[dict[str, Any]] = []
    for order, (option, expected_candidate) in enumerate(CANONICAL_OPTIONS, start=1):
        row_source = rows_a if option in rows_a else rows_b
        metadata_source = selection_a if option in selection_a else selection_b
        source_selection = (
            PEAK_DBTT_SELECTION if option in selection_a else WEAKT_CERAMIC_SELECTION
        )
        if option not in row_source:
            raise KeyError(f"canonical option missing from source registry: {option}")
        if option not in metadata_source:
            raise KeyError(f"canonical option missing from selection metadata: {option}")
        row = dict(row_source[option])
        candidate = str(row.get("candidate_id", ""))
        if candidate != expected_candidate:
            raise ValueError(
                f"candidate mismatch for {option}: {candidate!r} != {expected_candidate!r}"
            )
        if option in FORBIDDEN_OPTIONS or candidate in FORBIDDEN_OPTIONS:
            raise ValueError(f"forbidden historical option entered registry: {option}")
        combined_rows.append(row)
        normalized.append(
            _normalized_candidate(
                order=order,
                option=option,
                candidate_id=candidate,
                source_selection=source_selection,
                metadata=metadata_source[option],
                row_sha256=_canonical_row_sha256(row),
            )
        )

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header_a, lineterminator="\n")
    writer.writeheader()
    writer.writerows(combined_rows)
    registry_text = stream.getvalue()

    selection_payload: dict[str, Any] = {
        "schema": "v10.2.27_paper_four_class_selection_v1",
        "purpose": (
            "Exact four-class parameter overlay for the 30 degree, long-crack, "
            "stochastic PF/sharp-front paper campaign."
        ),
        "physics_contract": {
            "base_model": "v10.2.22 audited persistent-site sharp-front model",
            "parameter_transfer_only": True,
            "mechanics_changed": False,
            "stochastic_cleavage": True,
            "persistent_sites": True,
            "finite_source_inventory": False,
            "source_depletion_on_emission": False,
            "source_refresh_on_crack_advance": False,
            "explicit_recovery": False,
            "physical_front_width": True,
            "front_width_grid_independent": True,
        },
        "canonical_option_order": [option for option, _ in CANONICAL_OPTIONS],
        "primary_candidates": normalized,
        "source_files": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in required
        },
        "installed_registry_sha256": _sha256_bytes(registry_text.encode()),
        "transfer_policy": (
            "Exact source-row transfer only; no fitting, transformation, rounding, "
            "or substitution of inactive legacy source/recovery coordinates."
        ),
        "forbidden_historical_options": sorted(FORBIDDEN_OPTIONS),
    }
    selection_text = json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
    return registry_text, selection_text


def _write_or_check(path: Path, content: str, check_only: bool) -> None:
    if check_only:
        if not path.is_file():
            raise FileNotFoundError(f"generated file is missing: {path}")
        existing = path.read_text()
        if existing != content:
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
                "options": [option for option, _ in CANONICAL_OPTIONS],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
