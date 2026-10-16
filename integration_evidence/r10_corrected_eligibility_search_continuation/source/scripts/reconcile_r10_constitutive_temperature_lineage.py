#!/usr/bin/env python3
"""Reconcile the complete R10 ancestry with the source EXP_floor law.

This is an audit, not a fitter.  It records the portable source identity,
checks the four runtime surfaces (including the final stress-scaled row), and
distinguishes physical-source records, assistant proposals, endpoint
constructions, and runtime capabilities.  It deliberately does not infer
physical authorization from the presence of a field in software.
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


def record_by(path: Path, key: str, value: str) -> tuple[dict, dict]:
    with path.open(newline="") as stream:
        matches = [x for x in csv.DictReader(stream) if x[key] == value]
    if len(matches) != 1:
        raise KeyError(f"expected one {key}={value!r} in {path}")
    raw = matches[0]
    vector = json.loads(raw["complete_parameter_vector"]) if raw.get(
        "complete_parameter_vector"
    ) else dict(raw)
    return raw, vector


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
        evaluated_G0 = max(G0, 1.0e-12)
        evaluated_sigc = max(sigc, 1.0)
        raw_floor = max(surface.floor_min_eV, surface.floor_fraction * evaluated_G0)
        evaluated_floor = min(surface.floor_max_fraction * evaluated_G0, raw_floor)
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
            "active_guards": {
                "G0_min": G0 <= 1.0e-12,
                "sigc_min": sigc <= 1.0,
                "floor_min": surface.floor_fraction * evaluated_G0 < surface.floor_min_eV,
                "floor_max_fraction": raw_floor > surface.floor_max_fraction * evaluated_G0,
            },
            "evaluated_floor_eV": float(evaluated_floor),
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


PROCESS_FIELDS = {
    "opening": ("cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
                "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n",
                "cleave_floor_frac"),
    "emission": ("emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa",
                 "emit_sT_GPa_per_K", "emit_exp_a", "emit_exp_n",
                 "emit_floor_frac"),
    "peierls": ("peierls_H0_eV", "peierls_activation_entropy_kB",
                 "peierls_exp_a", "peierls_exp_n", "peierls_nu0_s"),
    "taylor": ("taylor_H0_eV", "taylor_activation_entropy_kB",
                "taylor_exp_a", "taylor_exp_n", "taylor_nu0_s"),
}


def same(a, b) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=1.0e-13, atol=1.0e-15))
    except (TypeError, ValueError):
        return a == b


def lineage_rows(stages: list[dict]) -> list[dict]:
    """Attribute the value actually inherited by the final row."""
    rows = []
    for process, fields in PROCESS_FIELDS.items():
        for field in fields:
            last = 0
            for index in range(1, len(stages)):
                if not same(stages[index - 1]["row"].get(field), stages[index]["row"].get(field)):
                    last = index
            source = stages[last]
            rows.append({
                "process": process,
                "field": field,
                "final_value": stages[-1]["row"].get(field),
                "last_changed_stage": source["stage"],
                "last_changed_candidate_id": source["candidate_id"],
                "provision_kind": source["provision_kind"],
                "physical_authorization": source["physical_authorization"],
                "source_scope": source["source_scope"],
            })
    # Transport characteristic stress is a runtime coupling rather than a CSV
    # transport field.  Make it explicit in the source-to-field table.
    for process in ("peierls", "taylor"):
        for field in ("effective_sigc0_GPa", "effective_sT_GPa_per_K"):
            rows.append({
                "process": process,
                "field": field,
                "final_value": stages[-1]["row"][
                    "emit_sigc0_GPa" if "sigc0" in field else "emit_sT_GPa_per_K"
                ],
                "last_changed_stage": stages[-1]["stage"],
                "last_changed_candidate_id": stages[-1]["candidate_id"],
                "provision_kind": "runtime inherited coupling after coherent stress scaling",
                "physical_authorization": "transform coherent; upstream parent remains non-promotable",
                "source_scope": "TransportBarrier.as_surface inherits emission sigc0 and sT",
            })
    return rows


def transform_audit(parent: dict, final: dict) -> dict:
    parent_surfaces, final_surfaces = surfaces(parent), surfaces(final)
    scale = float(final["emit_sigc0_GPa"]) / float(parent["emit_sigc0_GPa"])
    expected_sT = scale * float(parent["emit_sT_GPa_per_K"])
    samples = []
    max_G_error = 0.0
    max_S_error = 0.0
    for process in ("emission", "peierls", "taylor"):
        old, new = parent_surfaces[process], final_surfaces[process]
        for temperature in (300.0, 750.0, 1200.0):
            for stress_GPa in (0.0, 3.0, 9.0, 30.0):
                h = 1.0e-3
                sigma = stress_GPa * 1.0e9
                G_new = float(np.asarray(new.values_eV(sigma, temperature)))
                G_parent = float(np.asarray(old.values_eV(sigma / scale, temperature)))
                S_new = -(
                    float(np.asarray(new.values_eV(sigma, temperature + h)))
                    - float(np.asarray(new.values_eV(sigma, temperature - h)))
                ) / (2.0 * h) / 8.617333262145e-5
                S_parent = -(
                    float(np.asarray(old.values_eV(sigma / scale, temperature + h)))
                    - float(np.asarray(old.values_eV(sigma / scale, temperature - h)))
                ) / (2.0 * h) / 8.617333262145e-5
                max_G_error = max(max_G_error, abs(G_new - G_parent))
                max_S_error = max(max_S_error, abs(S_new - S_parent))
                samples.append({
                    "process": process, "temperature_K": temperature,
                    "physical_stress_GPa": stress_GPa,
                    "parent_mapped_stress_GPa": stress_GPa / scale,
                    "G_new_eV": G_new, "G_parent_mapped_eV": G_parent,
                    "S_new_kB": S_new, "S_parent_mapped_kB": S_parent,
                    "same_stress_entropy_change_kB": S_new - (-(
                        float(np.asarray(old.values_eV(sigma, temperature + h)))
                        - float(np.asarray(old.values_eV(sigma, temperature - h)))
                    ) / (2.0 * h) / 8.617333262145e-5),
                })
    return {
        "scale": scale,
        "sT_scale_error_GPa_per_K": float(final["emit_sT_GPa_per_K"]) - expected_sT,
        "identity": "G_new(chi,T)=G_parent(chi/c,T); S_new(chi,T)=S_parent(chi/c,T)",
        "max_abs_G_identity_error_eV": max_G_error,
        "max_abs_S_identity_error_kB": max_S_error,
        "named_energy_and_entropy_coefficients_preserved": all(same(parent[k], final[k]) for k in (
            "emit_G00_eV", "emit_gT_eV_per_K", "peierls_H0_eV",
            "peierls_activation_entropy_kB", "taylor_H0_eV",
            "taylor_activation_entropy_kB",
        )),
        "derived_entropy_at_same_physical_stress_preserved": all(
            abs(x["same_stress_entropy_change_kB"]) < 1.0e-10 for x in samples
        ),
        "samples": samples,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-export", type=Path, required=True)
    p.add_argument("--parent-registry", type=Path, required=True)
    p.add_argument("--parent-id", required=True)
    p.add_argument("--active-registry", type=Path, required=True)
    p.add_argument("--active-id", required=True)
    p.add_argument("--thermal-refinement-registry", type=Path, action="append", required=True)
    p.add_argument("--original-registry", type=Path, required=True)
    p.add_argument("--original-option-key", required=True)
    p.add_argument("--projection-registry", type=Path, required=True)
    p.add_argument("--projection-id", required=True)
    p.add_argument("--level-registry", type=Path, required=True)
    p.add_argument("--level-id", required=True)
    p.add_argument("--final-registry", type=Path, required=True)
    p.add_argument("--final-id", required=True)
    p.add_argument("--lineage-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    source = source_w100(a.source_export)
    parent_record, parent = record(a.parent_registry, a.parent_id)
    active_record, active = record(a.active_registry, a.active_id)
    original_record, original = record_by(
        a.original_registry, "option_key", a.original_option_key
    )
    projection_record, projection = record(a.projection_registry, a.projection_id)
    level_record, level = record(a.level_registry, a.level_id)
    final_record, final = record(a.final_registry, a.final_id)
    thermal_rows = []
    for registry in a.thermal_refinement_registry:
        with registry.open(newline="") as stream:
            thermal_rows.extend(list(csv.DictReader(stream)))

    parent_surfaces = surfaces(parent)
    active_surfaces = surfaces(active)
    final_surfaces = surfaces(final)
    stages = [
        {
            "stage": "retained_original_four",
            "candidate_id": original.get("candidate_id", a.original_option_key),
            "record_sha256": hashlib.sha256(json.dumps(
                original, sort_keys=True, separators=(",", ":"), default=str,
            ).encode()).hexdigest(),
            "row": original,
            "provision_kind": "retained physical-source registry row",
            "physical_authorization": "eligible retained original-family parent",
            "source_scope": "complete v9.13 four-class material record; distinct from W[100] export",
        },
        {
            "stage": "strict_projection_frontier",
            "candidate_id": a.projection_id,
            "record_sha256": projection_record["complete_row_sha256"],
            "row": projection,
            "provision_kind": "assistant-authored inverse projection and 1200 K anchor",
            "physical_authorization": "not established by portable physical source",
            "source_scope": "opening fields only; helper/search proposal",
        },
        {
            "stage": "joint_level_shift",
            "candidate_id": a.level_id,
            "record_sha256": level_record["complete_row_sha256"],
            "row": level,
            "provision_kind": "assistant-authored 300 K level-shift proposal",
            "physical_authorization": "not independently established",
            "source_scope": "opening G00 and floor fraction",
        },
        {
            "stage": "P3_P4_endpoint_construction",
            "candidate_id": a.parent_id,
            "record_sha256": parent_record["complete_row_sha256"],
            "row": parent,
            "provision_kind": "assistant-authored independent 300/1200 K endpoint construction",
            "physical_authorization": "excluded from promotion under controlling constitutive specification",
            "source_scope": "opening and emission affine coordinates; Peierls/Taylor energy-entropy endpoints",
        },
        {
            "stage": "coherent_emission_stress_scale",
            "candidate_id": a.final_id,
            "record_sha256": final_record["complete_row_sha256"],
            "row": final,
            "provision_kind": "coherent multiplicative sigc(T) search transform",
            "physical_authorization": "internally coherent but does not cure inherited endpoint ancestry",
            "source_scope": "emission sigc0/sT and runtime-inherited Peierls/Taylor stress scale",
        },
    ]
    lineage = lineage_rows(stages)
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
        "schema": "r10.constitutive-temperature-lineage/2",
        "decision": "ANCESTRY_RECONCILED_ENDPOINT_PARENT_NON_PROMOTABLE",
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
            "authorization_scope": (
                "W[100] formula and fitted parameter record over 298-673 K only; "
                "not authorization of later four-process endpoint constructions to 1200 K"
            ),
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
            "final_f038_row": {
                "candidate_id": a.final_id,
                "complete_row_sha256": final_record["complete_row_sha256"],
                "surfaces": {k: surface_receipt(v) for k, v in final_surfaces.items()},
            },
        },
        "complete_ancestry": [
            {k: stage[k] for k in ("stage", "candidate_id", "record_sha256",
                                    "provision_kind", "physical_authorization", "source_scope")}
            for stage in stages
        ],
        "source_to_field_table": lineage,
        "final_stress_scale_transform_audit": transform_audit(parent, final),
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
        "selection_contract": {
            "eligible_parents": "ancestry-qualified original/refined/retained-bank complete rows",
            "permitted_directions": "physical parameters authorized by each source family with inherited couplings",
            "excluded_directions": "independently selected cold/hot endpoints without source authorization",
            "not_implied": "no P1/P2-only restriction and no opening-only restriction",
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
    a.lineage_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.lineage_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lineage[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(lineage)
    print(json.dumps({
        "decision": report["decision"],
        "source_sha256": report["portable_source"]["sha256"],
        "historical_thermal_rows_quarantined": len(thermal_rows) + 1,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
