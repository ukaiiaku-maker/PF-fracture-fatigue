#!/usr/bin/env python3
"""Project current selected rows into exact one-row V10.4 bulk manifests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_manifest_v104 import (  # noqa: E402
    BulkManifestParameters, _REQUIRED,
)

SOURCE = (
    ROOT / "arrhenius_fracture/data/materials/v10_2_32_taylor_domain/"
    "v10_2_32_taylor_domain_registry.csv"
)
OUTPUT = ROOT / "arrhenius_fracture/data/materials/v10_4_current_bulk"
REPORT = OUTPUT / "qualification_report.json"


def build() -> dict:
    with SOURCE.open(newline="") as stream:
        selected = [
            row for row in csv.DictReader(stream)
            if row["option_key"].endswith("taylor_margin_0p10eV")
        ]
    if {row["material_class"] for row in selected} != {
        "peak", "DBTT", "weakT", "ceramic",
    }:
        raise RuntimeError("current bulk projection requires exactly four material classes")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    products = []
    for row in selected:
        family = row["material_class"]
        path = OUTPUT / f"{family}_selected_bulk.csv"
        projected = {key: row[key] for key in _REQUIRED}
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(_REQUIRED))
            writer.writeheader()
            writer.writerow(projected)
        parsed = BulkManifestParameters.from_csv(path)
        if parsed.option_key != row["option_key"]:
            raise RuntimeError(f"{family}: projected option identity changed")
        products.append({
            "material_class": family,
            "option_key": parsed.option_key,
            "candidate_id": parsed.candidate_id,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "field_count": len(_REQUIRED),
            "native_constructor_valid": True,
        })
    report = {
        "schema": "v10.4-current-bulk-manifest-projection/1",
        "status": "PASS",
        "source_registry": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "mapping": "exact field projection; no fitted or default coefficient",
        "production_execution_authorized": False,
        "products": sorted(products, key=lambda item: item["material_class"]),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    report = build()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
