#!/usr/bin/env python3
"""Reconcile R10 temperature coordinates with the source EXP_floor law.

This is an audit, not a fitter.  It records the portable source identity,
checks the four runtime surfaces, and quarantines rows made by independently
choosing temperature endpoints.  A row is eligible for the next continuation
only when its temperature dependence is produced by the original affine
free-energy coordinates and established cross-mechanism mappings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_thermal_accessibility_iteration import exp_surface, transport_surface


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, candidate_id: str) -> tuple[dict, dict]:
    with path.open(newline="") as stream:
        matches = [x for x in csv.DictReader(stream) if x["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"expected one {candidate_id!r} in {path}")
    return matches[0], json.loads(matches[0]["complete_parameter_vector"])


def source_w100(path: Path) -> dict:
    raw = json.loads(path.read_text())
    matches = [x for x in raw["models"] if x["system"] == "W[100]"]
    if len(matches) != 1 or matches[0]["model_type"] != "EXP_floor":
        raise ValueError("portable source has no unique W[100] EXP_floor model")
    return matches[0]


def surfaces(row: dict) -> dict:
    emission = exp_surface(row, "emit", 1.0e11)
    return {
        "opening": exp_surface(row, "cleave", 1.0e12),
        "emission": emission,
        "peierls": transport_surface(row, "peierls"),
        "taylor": transport_surface(row, "taylor"),
    }


def surface_receipt(surface, temperatures=(300.0, 750.0, 1200.0)) -> dict:
    cases = []
    for temperature in temperatures:
        dT = temperature - float(surface.Tref_K)
        G0 = float(surface.G00_eV + surface.gT_eV_per_K * dT)
        sigc = float(surface.sigc0_Pa + surface.sT_Pa_per_K * dT)
        probes = []
        for stress_GPa in (0.0, 3.0, 9.0):
            h = 1.0e-3
            plus = float(np.asarray(surface.values_eV(stress_GPa * 1.0e9, temperature + h)))
            minus = float(np.asarray(surface.values_eV(stress_GPa * 1.0e9, temperature - h)))
            entropy_kB = -(plus - minus) / (2.0 * h) / 8.617333262145e-5
            probes.append({"stress_GPa": stress_GPa, "activation_entropy_kB": entropy_kB})
        cases.append({
            "temperature_K": temperature,
            "raw_G0_eV": G0,
            "raw_sigc_GPa": sigc / 1.0e9,
            "raw_domain_valid": G0 > 0.0 and sigc > 0.0,
            "entropy_probes": probes,
        })
    return {
        "G00_eV": float(surface.G00_eV),
        "gT_eV_per_K": float(surface.gT_eV_per_K),
        "sigc0_GPa": float(surface.sigc0_Pa) / 1.0e9,
        "sT_GPa_per_K": float(surface.sT_Pa_per_K) / 1.0e9,
        "alpha": float(surface.alpha),
        "exponent": float(surface.exponent),
        "floor_fraction": float(surface.floor_fraction),
        "cases": cases,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-export", type=Path, required=True)
    p.add_argument("--parent-registry", type=Path, required=True)
    p.add_argument("--parent-id", required=True)
    p.add_argument("--active-registry", type=Path, required=True)
    p.add_argument("--active-id", required=True)
    p.add_argument("--thermal-refinement-registry", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    source = source_w100(a.source_export)
    parent_record, parent = record(a.parent_registry, a.parent_id)
    active_record, active = record(a.active_registry, a.active_id)
    thermal_rows = []
    for registry in a.thermal_refinement_registry:
        with registry.open(newline="") as stream:
            thermal_rows.extend(list(csv.DictReader(stream)))

    parent_surfaces = surfaces(parent)
    active_surfaces = surfaces(active)
    transport_coupling = {}
    for name in ("peierls", "taylor"):
        transport_coupling[name] = {
            "parent_sigc0_over_emission": (
                parent_surfaces[name].sigc0_Pa / parent_surfaces["emission"].sigc0_Pa
            ),
            "parent_sT_over_emission": (
                parent_surfaces[name].sT_Pa_per_K / parent_surfaces["emission"].sT_Pa_per_K
            ),
            "active_sigc0_over_emission": (
                active_surfaces[name].sigc0_Pa / active_surfaces["emission"].sigc0_Pa
            ),
            "active_sT_over_emission": (
                active_surfaces[name].sT_Pa_per_K / active_surfaces["emission"].sT_Pa_per_K
            ),
        }

    source_params = source["parameters"]
    report = {
        "schema": "r10.constitutive-temperature-lineage/1",
        "decision": "RECONCILED_BEFORE_FURTHER_THERMAL_GENERATION",
        "portable_source": {
            "path": str(a.source_export.resolve()),
            "sha256": sha256(a.source_export),
            "system": source["system"],
            "model_type": source["model_type"],
            "formula": source_params["formula"],
            "G0_formula": source_params["G0_formula"],
            "sigc_formula": source_params["sigc_formula"],
            "Gfloor_formula": source_params["Gfloor_formula"],
            "parameters": {k: source_params[k] for k in (
                "Tref_K", "Tmin_K", "Tmax_K", "G00_eV", "gT_eV_per_K",
                "sigc0_Pa", "sT_Pa_per_K", "a", "n", "Gfloor_fraction",
                "Gfloor_min_eV", "Gfloor_max_fraction",
            )},
        },
        "runtime_constitutive_mapping": {
            "rate": "nu0*exp(-DeltaG_EXP_floor(stress,T)/(kB*T))",
            "activation_entropy": "S_star(stress,T)=-partial DeltaG_EXP_floor/partial T",
            "opening_and_emission": "direct ExpFloorBarrier affine G0(T), sigc(T), and source floor law",
            "peierls_and_taylor": (
                "nonzero activation_entropy_kB sets gT=-S*kB; sigc0 and sT are inherited "
                "together from emission through TransportBarrier.as_surface"
            ),
            "separate_entropy_prefactor": False,
        },
        "evaluated_rows": {
            "retained_parent": {
                "candidate_id": a.parent_id,
                "complete_row_sha256": parent_record["complete_row_sha256"],
                "surfaces": {k: surface_receipt(v) for k, v in parent_surfaces.items()},
            },
            "historical_active_endpoint_row": {
                "candidate_id": a.active_id,
                "complete_row_sha256": active_record["complete_row_sha256"],
                "surfaces": {k: surface_receipt(v) for k, v in active_surfaces.items()},
            },
        },
        "cross_mechanism_coupling": transport_coupling,
        "historical_row_disposition": {
            "active_acquisition": {
                "status": "PRESERVED_NON_PROMOTABLE",
                "reason": "emit_sigc(300 K) and emit_sigc(1200 K) were independently selected",
                "candidate_id": a.active_id,
            },
            "thermal_refinement": {
                "status": "PRESERVED_NON_PROMOTABLE",
                "reason": "opening cold/hot coordinates were independently selected",
                "candidate_ids": [x["candidate_id"] for x in thermal_rows],
                "registries": [str(x.resolve()) for x in a.thermal_refinement_registry],
            },
        },
        "authorized_continuation_contract": {
            "coordinate": "coherent multiplicative emission characteristic-stress scale",
            "required_transform": [
                "emit_sigc0_offspring = scale * emit_sigc0_parent",
                "emit_sT_offspring = scale * emit_sT_parent",
            ],
            "inherited_transform": "Peierls and Taylor receive the same sigc(T) through emission",
            "must_remain_exactly_unchanged": [
                "opening complete EXP_floor surface",
                "emission G00 and gT (including activation entropy)",
                "emission alpha, exponent, floor, and attempt frequency",
                "Peierls H0, activation entropy, alpha, exponent, and attempt frequency",
                "Taylor H0, activation entropy, alpha, exponent, and attempt frequency",
                "all state-evolution, source, geometry, mechanics, event, and reward fields",
            ],
            "independent_temperature_endpoint_optimization": False,
        },
    }
    if any(abs(float(parent[k])) == 0.0 for k in (
        "peierls_activation_entropy_kB", "taylor_activation_entropy_kB",
    )):
        raise RuntimeError("retained transport entropy was zeroed")
    if any(abs(v[x] - 1.0) > 1.0e-12 for v in transport_coupling.values()
           for x in v):
        raise RuntimeError("transport stress-scale inheritance is inconsistent")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "source_sha256": report["portable_source"]["sha256"],
        "historical_thermal_rows_quarantined": len(thermal_rows) + 1,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
