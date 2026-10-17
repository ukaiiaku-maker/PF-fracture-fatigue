#!/usr/bin/env python3
"""Materialize the retained-allocation rows for the first moderate-window batch."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_r10_global_mechanism_search import (  # noqa: E402
    FAMILY, canonical_hash, generate_row, phase_action, read_sources,
)

BASE = ROOT / "integration_evidence/r10_global_mechanism_search"
INPUTS = ROOT / "integration_evidence/r10_joint_search_amendment/inputs"
OUT = BASE / "moderate_window_batch"
SELECTION = {
    "R10A_GLOBAL_002081": (1.9522535292003416, 6.238890993326063),
    "R10A_GLOBAL_011750": (6.115160984474933, 19.937861678523163),
    "R10A_GLOBAL_014670": (3.5605846091763644, 11.60892470577081),
}
TEMPERATURE = {"OPENING_DIRECTED": 300.0, "MIXED_DIRECTED": 600.0,
               "PLASTICITY_DIRECTED": 900.0}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    allocation = {r["candidate_id"]: r for r in csv.DictReader(
        (BASE / "state_reduction_allocation.csv").open(newline=""))}
    sources = read_sources(INPUTS / "original_four_registry.csv",
                           INPUTS / "fatigue_parent_registry.csv")
    points = qmc.Sobol(d=22, scramble=True, seed=10162026).random_base2(17)
    registry_rows, descriptor_rows, plan = [], [], {"schema": "r10.moderate-window-load-plan/1", "candidates": {}}
    for candidate, (low, high) in SELECTION.items():
        index = int(candidate.rsplit("_", 1)[1])
        row = generate_row(index, points[index], sources)
        descriptor = dict(allocation[candidate])
        mechanism = descriptor["mechanism_acquisition_stratum"]
        temperature = TEMPERATURE[mechanism]
        loads = [low, math.sqrt(low * high), high]
        refs = [5e-6 * phase_action(row, "cleave", load, temperature) for load in loads]
        slope = math.log(refs[-1] / refs[0]) / math.log(loads[-1] / loads[0])
        descriptor.update({
            "planned_temperature_K": temperature,
            "planned_Kmax_MPa_sqrt_m": ";".join(map(str, loads)),
            "planned_A0_rates_m_per_cycle": ";".join(map(str, refs)),
            "planned_endpoint_A0_slope": slope,
            "planned_rate_decades": math.log10(refs[-1] / refs[0]),
            "planned_K_ratio": loads[-1] / loads[0],
            "selection_basis": "RETAINED_4096_ALLOCATION_BROAD_MODERATE_RECURRENT_A0_WINDOW",
        })
        descriptor_rows.append(descriptor)
        registry_rows.append({
            "candidate_id": candidate, "family": FAMILY,
            "complete_row_sha256": canonical_hash(row), "cleavage_hits": 3.0,
            "cleavage_tau_s": 1e-6,
            "mechanism_acquisition_stratum": mechanism,
            "thermal_acquisition_stratum": descriptor["thermal_acquisition_stratum"],
            "complete_parameter_vector": json.dumps(row, sort_keys=True, separators=(",", ":")),
        })
        plan["candidates"][candidate] = {
            "temperature_K": temperature, "Kmax_MPa_sqrt_m": loads,
            "A0_physical_m_per_cycle": refs,
        }
    write_csv(OUT / "registry.csv", registry_rows)
    write_csv(OUT / "descriptors.csv", descriptor_rows)
    (OUT / "load_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    decision = {
        "schema": "r10.moderate-window-acquisition/1", "family": FAMILY,
        "status": "PREPARED_FOR_NATIVE_EXECUTION", "candidate_count": len(registry_rows),
        "selection_invariants": {
            "endpoint_slope_range": [3.4, 4.6], "minimum_rate_decades": 2.0,
            "minimum_K_ratio": 2.72134, "maximum_upper_A0_rate_m_per_cycle": 2e-5,
            "minimum_lower_A0_rate_m_per_cycle": 1e-10,
            "no_A0_mechanism_gate": True, "thermal_labels_are_proxies": True,
        },
        "files": {},
    }
    for name in ("registry.csv", "descriptors.csv", "load_plan.json"):
        path = OUT / name
        decision["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "bytes": path.stat().st_size}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
