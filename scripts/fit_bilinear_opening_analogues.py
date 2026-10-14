#!/usr/bin/env python3
"""Fit four shared opening surfaces to finite-domain bilinear analogues."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bilinear_opening_fit import fit_temperature_law
from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.parameter_registry_v9111 import select_option, write_compatibility_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(registry: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite bilinear fit registry: {output}")
    output.mkdir(parents=True)
    with registry.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = [row for row in rows if row["option_key"].endswith("__taylor_margin_0p10eV")]
    if len(selected_rows) != 4:
        raise RuntimeError("expected one primary readiness row for each of four opening families")
    fits = []
    with tempfile.TemporaryDirectory(prefix="bilinear-opening-fit-") as scratch:
        for row in selected_rows:
            selected = select_option(row["option_key"], registry)
            manifest_path = Path(scratch) / f"{row['material_class']}.csv"
            write_compatibility_manifest(selected, manifest_path)
            surface = MaterialManifest.from_csv(manifest_path).cleavage
            fit = fit_temperature_law(surface)
            fit.update({
                "material_class": row["material_class"],
                "parent_option_key": row["option_key"],
                "parent_candidate_id": row["candidate_id"],
                "opening_surface": surface.__dict__,
                "shared_across_taylor_margin_variants": True,
            })
            fits.append(fit)
    payload = {
        "schema": "v10.2.32-bilinear-opening-fit-registry/1",
        "status": "PASS_IDENTIFIABLE_FIT_PHYSICAL_SUPPORT_UNRESOLVED",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_registry": str(registry.relative_to(ROOT)),
        "source_registry_sha256": _sha(registry),
        "objective": "sum w*((G_BL-G_EXP)/(kBT))^2 + beta*sum w*(q_span*(V_BL-V_EXP)/(kBT))^2",
        "beta": 1.0,
        "weighting": "uniform normalized weights within every temperature slice",
        "physical_support_policy": "report identifiable B0=A*Gamma and peak traction; do not assign A, Gamma, delta_f, delta_p or stiffness without authoritative support",
        "fits": fits,
    }
    path = output / "v10_2_32_bilinear_opening_fit_registry.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "sha256_manifest.json").write_text(json.dumps({
        path.name: _sha(path)
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "families": len(fits)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.registry.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
