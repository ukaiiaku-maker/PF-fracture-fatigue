#!/usr/bin/env python3
"""Build the compact, deterministic common-framework handoff.

This builder performs constitutive analysis and writes plans only.  It never
launches a PF or FEM/CZM specimen calculation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_landscape import (
    ExponentialLandscape, PolynomialLandscape, TriangularBilinearLandscape,
    evaluate_traction_controlled_barrier, numerical_work_J_m2,
)
from arrhenius_fracture.prospective_campaign import (
    TEMPERATURES_K, build_cases, campaign_summary, load_material_rows,
)


DEFAULT_OUT = ROOT / "common_framework_release"


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cohesive_rows() -> list[dict]:
    tc, gamma, area = 4.0e9, 20.0, 2.5e-18
    shapes = {
        "triangular_bilinear": TriangularBilinearLandscape(tc, gamma),
        "exponential": ExponentialLandscape(tc, gamma),
        "specified_polynomial": PolynomialLandscape(tc, gamma),
    }
    rows = []
    for name, shape in shapes.items():
        for traction_fraction in (0.10, 0.25, 0.50, 0.75, 0.90):
            value = evaluate_traction_controlled_barrier(shape, traction_fraction * tc, area)
            rows.append({
                "shape": name, "peak_traction_Pa": tc,
                "separation_work_J_m2": gamma,
                "quadrature_work_J_m2": numerical_work_J_m2(shape),
                "traction_fraction": traction_fraction,
                "barrier_J": value.barrier_J,
                "activation_volume_m3": value.activation_volume_m3,
                "minimum_opening_m": value.minimum_opening_m,
                "saddle_opening_m": value.saddle_opening_m,
                "activation_area_m2": area,
                "ensemble": value.ensemble,
            })
    return rows


def _source_identities(companion: Path | None) -> dict:
    payload = {
        "schema": "common-fracture-source-identities/1",
        "pf": {"path": str(ROOT), "branch": _git("branch", "--show-current"), "generation_parent_commit": _git("rev-parse", "HEAD")},
        "independent_fem_czm": None,
        "manuscript_package": {"status": "frozen_intermediate_record", "modified_by_builder": False},
        "stateful_kitagawa": {"included": False},
    }
    if companion is not None:
        companion = companion.resolve()
        payload["independent_fem_czm"] = {
            "path": str(companion),
            "branch": _git("branch", "--show-current", cwd=companion),
            "commit": _git("rev-parse", "HEAD", cwd=companion),
            "identity": "original independent adaptive-CZM finite-element realization",
        }
    return payload


def _commands() -> str:
    return """# Common-framework commands

These commands assume the repository root and the qualified Python environment.
They do not authorize the large prospective campaign.

## Portable code qualification

```bash
python scripts/run_common_framework_qualification.py portable
```

## Complete framework qualification

```bash
python scripts/run_common_framework_qualification.py full \\
  --external-manifest common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json
```

The full command exits nonzero with `INCOMPLETE` when any mandatory external
artifact is absent or has the wrong content digest. It never turns that state
into an optional skip.

## Rebuild this compact handoff

```bash
python scripts/build_common_framework_release.py \\
  --companion ../fem-czm-current-pf-companion
```

## Inspect the prospective plan (no solver execution)

```bash
python scripts/run_prospective_common_campaign.py --status
python scripts/run_prospective_common_campaign.py --block P1 --status
```

## Query an accepted continuous history

```bash
python scripts/query_k_at_crack_length.py HISTORY.csv \\
  --target-um 1000 --coordinate projected_extension --stage equilibrated
```

Production launch remains disabled until the release gates in
`FRAMEWORK_STATUS.json` are resolved and a separate author-approved execution
stage names the frozen framework commit and output root.
"""


def build(out: Path, companion: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(ROOT)
    rows = [case.to_dict() for case in cases]
    with (out / "PROSPECTIVE_CAMPAIGN.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    csv_rows = [{**row, "native_args": json.dumps(row["native_args"])} for row in rows]
    _write_csv(out / "PROSPECTIVE_CAMPAIGN.csv", csv_rows)
    _write_json(out / "CAMPAIGN_SUMMARY.json", campaign_summary(cases))

    materials = []
    for row in load_material_rows(ROOT).values():
        for temperature in TEMPERATURES_K:
            materials.append({"family": row.family, "parameter_option": row.option_key,
                              "temperature_K": temperature, **row.domain(temperature)})
    _write_csv(out / "MATERIAL_DOMAIN_AUDIT.csv", materials)
    _write_json(out / "MATERIAL_DOMAIN_AUDIT.json", materials)

    cohesive = _cohesive_rows()
    _write_csv(out / "COHESIVE_BENCHMARK.csv", cohesive)
    _write_json(out / "COHESIVE_BENCHMARK.json", {
        "schema": "matched-peak-work-cohesive-landscape/1",
        "scope": "traction-controlled constitutive benchmark; not a specimen prediction",
        "rows": cohesive,
    })
    _write_json(out / "SOURCE_IDENTITIES.json", _source_identities(companion))
    _write_json(out / "FRAMEWORK_STATUS.json", {
        "schema": "common-fracture-framework-status/1",
        "framework_code": "IMPLEMENTED_PENDING_COMPLETE_QUALIFICATION",
        "large_production_campaign_authorized": False,
        "paper_package": "FROZEN_INTERMEDIATE_RECORD",
        "completed": [
            "orthogonal shared option resolver and native execution plans",
            "cohesive potential/barrier constitutive interface and exact bilinear Kocks benchmark",
            "candidate-class normalization contract that never charges cavity free span as solid",
            "declarative P1-P8 matrix with raw constitutive-domain gates",
        ],
        "mandatory_open_gates": [
            "recursive physical branch birth and daughter continuation qualification",
            "production PF/FEM-CZM mesh and observer-support convergence",
            "complete external-evidence verification from independent storage",
        ],
        "author_decision": "P7 finite-ligament kinetic interpretation: local incremental front or collective finite ligament",
        "invalid_domain": "DBTT 1050-1300 K raw emission stress scale is nonpositive; 1000 K requires boundary qualification",
    })
    for name in ("COMMON_FRAMEWORK_RELEASE.md", "COMMON_FRAMEWORK_SCIENTIFIC_SPECIFICATION.md", "CAVITY_CANDIDATE_AUTHOR_DECISION.md"):
        shutil.copyfile(ROOT / "docs" / name, out / name)
    (out / "COMMANDS.md").write_text(_commands())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--companion", type=Path)
    args = parser.parse_args()
    build(args.out.resolve(), args.companion)


if __name__ == "__main__":
    main()
