#!/usr/bin/env python3
"""Build and evaluate explicit, unselected DBTT qualification alternatives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.parameter_registry_v9111 import (
    select_option, write_compatibility_manifest,
)
from arrhenius_fracture.taylor_domain_revision import (
    T_MAX_K, T_MIN_K, construct_amplitude_rows, raw_emission_stress_extrema,
    raw_taylor_extrema, read_rows,
)
from arrhenius_fracture.taylor_factorization import raw_scales


PARENT_REGISTRY = ROOT / (
    "arrhenius_fracture/data/materials/"
    "v10_2_27_v913_four_class_paper_registry.csv"
)
STRESS_CANDIDATES = ROOT / (
    "arrhenius_fracture/data/materials/v10_2_32_taylor_domain/"
    "v10_2_32_shared_stress_scale_candidates.json"
)
TEMPERATURES_K = (300.0, 600.0, 800.0, 900.0, 950.0, 1000.0,
                  1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _variant(base: dict[str, str], identity: str) -> dict[str, str]:
    variants, records = construct_amplitude_rows([base], (0.10,))
    row = variants[0]
    row["option_key"] = identity
    row["candidate_id"] = identity + "__candidate"
    row["role"] = "explicit DBTT numerical qualification alternative"
    row["validation_status"] = (
        "constructed qualification option; not calibrated; not a production default"
    )
    row["qualification_parent_option_key"] = base["option_key"]
    row["qualification_taylor_construction"] = json.dumps(
        records[0], sort_keys=True, separators=(",", ":")
    )
    return row


def construct_rows() -> list[dict[str, str]]:
    _parent_fields, parents = read_rows(PARENT_REGISTRY)
    parent_dbtt = next(row for row in parents if row["material_class"] == "DBTT")
    candidates = json.loads(STRESS_CANDIDATES.read_text())
    minimal = dict(candidates["minimal_intercept_candidates"][0]["row"])
    approved = dict(candidates["existing_approved_alternative"]["row"])

    rows = [
        _variant(minimal, "dbtt_paper_minimal_sigc_taylor_0p10eV_qualification"),
        _variant(approved, "dbtt_primary_taylor_0p10eV_qualification"),
    ]
    # Demonstrate that the minimal alternative is the paper lineage and the
    # approved alternative remains a distinct historical parameter family.
    if rows[0]["qualification_parent_option_key"] == rows[1]["qualification_parent_option_key"]:
        raise RuntimeError("DBTT qualification alternatives lost distinct ancestry")
    if not rows[0]["qualification_parent_option_key"].startswith(
        parent_dbtt["option_key"] + "__shared_sigc_minimal_intercept"
    ):
        raise RuntimeError("minimal-stress DBTT candidate lost paper-parent identity")
    return rows


def _domain(row: dict[str, str]) -> dict:
    raw = []
    for temperature in TEMPERATURES_K:
        values = raw_scales(row, temperature)
        raw.append({
            "temperature_K": temperature,
            "raw_scales": values,
            "admissible": all(value > 0.0 for value in values.values()),
        })
    emission = raw_emission_stress_extrema(row)
    taylor = raw_taylor_extrema(row)
    return {
        "continuous_domain_K": [T_MIN_K, T_MAX_K],
        "emission_stress_extrema": emission,
        "taylor_barrier_extrema_eV": taylor,
        "all_declared_points_admissible": all(item["admissible"] for item in raw),
        "declared_points": raw,
    }


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = construct_rows()
    constructor_rows = []
    for index, row in enumerate(rows):
        registry = output / (
            "dbtt_paper_minimal_qualification_registry.csv" if index == 0 else
            "dbtt_primary_qualification_registry.csv"
        )
        with registry.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        selected = select_option(row["option_key"], registry)
        manifest_path = output / f"constructor_manifest_{index}.csv"
        write_compatibility_manifest(selected, manifest_path)
        manifest = MaterialManifest.from_csv(manifest_path)
        constructor_rows.append({
            "option_key": row["option_key"],
            "candidate_id": row["candidate_id"],
            "parent_option_key": row["qualification_parent_option_key"],
            "registry_sha256": selected.registry_sha256,
            "registry": _display(registry),
            "manifest_sha256": _sha(manifest_path),
            "constructor_candidate_id": manifest.candidate_id,
            "domain": _domain(row),
            "source_lifecycle_provenance": (
                "current_persistent_paper_lineage"
                if index == 0 else
                "older_approved_finite_source_refresh_and_recovery_lineage"
            ),
        })
    report = {
        "schema": "dbtt-explicit-qualification-options/1",
        "status": "PASS" if all(
            row["domain"]["all_declared_points_admissible"]
            and row["domain"]["emission_stress_extrema"]["minimum_Pa"] > 0.0
            and row["domain"]["taylor_barrier_extrema_eV"]["minimum"] >= 0.10
            for row in constructor_rows
        ) else "FAIL",
        "production_selection": None,
        "physical_calibration_claimed": False,
        "registries": [
            {"path": row["registry"], "sha256": row["registry_sha256"]}
            for row in constructor_rows
        ],
        "source_candidates": _display(STRESS_CANDIDATES),
        "source_candidates_sha256": _sha(STRESS_CANDIDATES),
        "options": constructor_rows,
        "selection_boundary": (
            "Both rows are complete numerical qualification inputs. The paper-lineage "
            "minimal-stress row changes the shared emission/Peierls/Taylor stress scale; "
            "the earlier dbtt_primary row changes the complete material/source family. "
            "Neither is selected for production."
        ),
    }
    (output / "qualification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
