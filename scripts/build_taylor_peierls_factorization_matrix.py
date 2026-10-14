#!/usr/bin/env python3
"""Build compact, same-state Taylor--Peierls factorization evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.prospective_campaign import REGISTRY, TEMPERATURES_K
from arrhenius_fracture.taylor_factorization import evaluate_same_state, select_temperature_anchors


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> dict:
    registry = ROOT / REGISTRY
    rows = list(csv.DictReader(registry.open(newline="")))
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    records = []
    domains = []
    for row in rows:
        anchors, domain = select_temperature_anchors(
            row, tuple(float(T) for T in TEMPERATURES_K)
        )
        domains.append({
            "family": row["material_class"], "option_key": row["option_key"],
            "selected_admissible_anchors_K": anchors, "declared_grid_domain": domain,
        })
        rho_c = float(row["taylor_corr_rho_c_m2"])
        densities = (0.01 * rho_c, rho_c, 100.0 * rho_c)
        for T in anchors:
            raw = next(item["raw_scales"] for item in domain if item["temperature_K"] == T)
            # A frozen local stress at half the positive raw emission scale keeps
            # all switch variants at precisely the same material state.
            magnitude = 0.5 * raw["emission_sigc_Pa"]
            for sign in (-1.0, 0.0, 1.0):
                for rho in densities:
                    records.append(evaluate_same_state(
                        row, temperature_K=T,
                        signed_local_stress_Pa=sign * magnitude,
                        raw_density_m2=rho,
                    ))

    payload = {
        "schema": "taylor-peierls-factorization-matrix/1",
        "status": "DIAGNOSTIC_ONLY_NOT_RUNTIME_SELECTABLE",
        "source_head": git_head,
        "registry": {"path": str(Path(REGISTRY)), "sha256": _sha(registry)},
        "declared_temperature_grid_K": list(TEMPERATURES_K),
        "state_design": {
            "temperature": "lowest, middle, highest admissible point of each selected row on the declared campaign grid",
            "unsupported_states": "reported from raw scales and never evaluated through executed clamps",
            "stress": "minus, zero, plus one-half of the row's raw emission characteristic stress",
            "raw_density": "0.01*rho_c, rho_c, 100*rho_c",
            "consumed_density": "max(raw_density, selected rho_forest_floor)",
            "burgers_m": 2.74e-10,
        },
        "provenance": {
            "historical_moving_tip": "git 6b200bde^ arrhenius_fracture/unified_mpz.py plus quarantined historical_legacy_implicit_1s",
            "historical_bulk": "git 6b200bde^ arrhenius_fracture/emission_derived_plasticity.py and config.py",
            "current": "arrhenius_fracture.taylor_release and UnifiedMPZState._transport_rates",
            "warning": "The two historical bundles are kept separate because they were distinct native consumers.",
        },
        "log_policy": "Only unsigned elementary forward/reverse rates are logged; signed net rates are never logged.",
        "domain_summary": domains,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=False)
    matrix = output / "factorization_matrix.json"
    matrix.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary_fields = [
        "family", "temperature_K", "signed_local_stress_Pa", "raw_density_m2",
        "consumed_forest_density_m2", "variant", "order", "completion_rate_s",
        "completion_time_s", "jump_length_m", "peierls_velocity_m_s",
    ]
    with (output / "factorization_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for record in records:
            for variant in record["isolated_variants"]:
                writer.writerow({
                    "family": record["family"],
                    "temperature_K": record["temperature_K"],
                    "signed_local_stress_Pa": record["signed_local_stress_Pa"],
                    "raw_density_m2": record["raw_density_m2"],
                    "consumed_forest_density_m2": record["consumed_forest_density_m2"],
                    "variant": variant["name"], "order": variant["order"],
                    "completion_rate_s": variant["completion_rate_s"],
                    "completion_time_s": variant["completion_time_s"],
                    "jump_length_m": variant["jump_length_m"],
                    "peierls_velocity_m_s": variant["peierls_velocity_m_s"],
                })
    readme = output / "README.md"
    readme.write_text(
        "# Taylor--Peierls same-state factorization\n\n"
        "This is diagnostic-only readiness evidence. It does not expose the historical law "
        "to production. The JSON contains exact current elementary components, isolated "
        "completion/order/phi/rate-cap/jump-floor variants, and separate exact historical "
        "moving-tip and bulk bundles. Unsupported raw parent temperatures are reported but "
        "not evaluated. `phi=ell_f/b`, with `ell_f=1/(2 sqrt(rho_f))`; the already-local "
        "stress enters once, so the diagnostic does not add a second crack-tip concentration.\n"
    )
    files = [matrix, output / "factorization_summary.csv", readme]
    manifest = {p.name: _sha(p) for p in files}
    (output / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.output)
    print(json.dumps({
        "status": payload["status"], "records": len(payload["records"]),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
