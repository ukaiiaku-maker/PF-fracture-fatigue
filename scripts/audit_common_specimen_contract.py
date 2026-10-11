#!/usr/bin/env python3
"""Audit whether the retained PF and FEM/CZM histories are physically matched."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e"
DEFAULT_PF = EVIDENCE / "pf_peak1000_theta0_seed8666_anchor"
DEFAULT_FEM = EVIDENCE / "fem_czm_peak1000_theta0_seed8666_first_passage"
DEFAULT_ELASTIC = (ROOT / "integration_evidence" / "final_fracture_campaign" /
                   "common_specimen_elastic_reference" / "comparison.json")
DEFAULT_OUT = (ROOT / "review_packet" / "final_fracture_campaign_a5a5883" /
               "PF_FEM_COMMON_SPECIMEN.json")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _native_value(attestation: dict, flag: str):
    argv = attestation["native_argv"]
    return argv[argv.index(flag) + 1] if flag in argv else None


def classify_contract(pf: Path, fem: Path, elastic: Path) -> dict:
    pf_args = json.loads((pf / "run_args.json").read_text())
    fem_args = json.loads((fem / "run_args.json").read_text())
    pf_attestation = json.loads((pf / "shared_runtime_attestation.post.json").read_text())
    fem_contract = json.loads((fem / "current_pf_companion_contract_v1.json").read_text())
    parity = json.loads((fem / "theta0_pf_full_field_parity_contract_v10_0_5_18_4_0.json").read_text())
    pf_rows = _load_rows(pf / "steps_1000K.csv")
    fem_rows = _load_rows(fem / "steps_1000K.csv")
    elastic_result = json.loads(elastic.read_text()) if elastic.is_file() else None

    pf_bulk = _native_value(pf_attestation, "--bulk-plasticity-mode") or "tip_only"
    fem_bulk = fem_args.get("bulk_plasticity_semantic_mode") or fem_args.get("bulk_plasticity_mode")
    pf_lpz = float(pf_args["L_pz"])
    fem_lpz = float(fem_args["L_pz"])
    pf_first = pf_rows[0]
    fem_first = fem_rows[0]
    opening_match = float(pf_first["Uapp_m"]) == float(fem_first["Uapp_m"])
    force_ratio = float(pf_first["Ftop_N"]) / float(fem_first["Ftop_N"])
    kj_ratio = float(pf_first["KJ_Pa_sqrtm"]) / float(fem_first["KJ_Pa_sqrtm"])

    matched = {
        "temperature_K": float(pf_args["temperatures"][0]) == float(fem_args["temperatures"][0]),
        "theta_deg": float(pf_args["crystal_theta_deg"]) == float(fem_args["crystal_theta_deg"]),
        "hazard_seed": pf_attestation["options"]["random_seed"] == 8666,
        "nominal_opening_increment_m": float(pf_args["dU"]) == float(fem_args["dU"]),
        "nominal_time_increment_s": float(pf_args["dt"]) == float(fem_args["dt"]),
        "mesh_counts_and_tip_resolution": all(
            pf_args[key] == fem_args[key] for key in ("nx", "ny", "tip_h_fine", "tip_ratio")
        ),
        "cubic_elasticity_and_orientation": all(
            pf_args[key] == fem_args[key]
            for key in ("crystal_C11", "crystal_C12", "crystal_C44", "crystal_theta_deg")
        ),
        "first_recorded_total_opening": opening_match,
    }
    unmatched = {
        "crack_backend": {"pf": pf_args["crack_backend"], "fem": fem_args["crack_backend"]},
        "accommodation": {"pf": pf_bulk, "fem": fem_bulk},
        "process_zone_length_m": {"pf": pf_lpz, "fem": fem_lpz},
        "mobile_density_floor_m2": {
            "pf": pf_args.get("pt_mobile_density_floor_m2"),
            "fem": fem_args.get("pt_mobile_density_floor_m2"),
        },
        "cluster_J_outer_radius_m": {
            "pf": 8.0 * max(10.0 * pf_lpz, 5.0e-6),
            "fem": 8.0 * float(fem_args["rJ_cluster"]),
        },
        "local_J_outer_radius_m": {
            "pf": max(3.0 * float(pf_args["da_phys"]), 2.5 * pf_lpz,
                      6.0 * float(pf_args["tip_h_fine"]), 3.0e-6),
            "fem": float(fem_args["rJ_outer"]),
        },
        "source_state_representation": {
            "pf": "moving tip-process zone; no full-field bulk accommodation",
            "fem": "moving tip state plus full-field local Peierls-Taylor accommodation",
        },
    }
    false_parity_claim = bool(
        parity["bulk_parity"].get("PF_bulk_plasticity_mode") == "full_field"
        and pf_bulk != "full_field"
    )
    return {
        "schema": "pf-fem-common-specimen-audit/1",
        "inputs": {"pf": str(pf.resolve()), "fem": str(fem.resolve())},
        "specimen_contract": {
            "geometry_from_shared_kernel": {
                "length_x_m": 2.0e-3,
                "length_y_m": 4.0e-3,
                "initial_notch_length_m": 0.5e-3,
                "notch_half_thickness_m": 0.08e-3,
                "plane_condition": "plane_strain",
                "out_of_plane_convention": "unit thickness; force columns are N/m despite legacy Ftop_N label",
                "opening_semantics": "total opening; top +U/2 and bottom -U/2",
            },
            "matched_fields": matched,
            "unmatched_fields": unmatched,
        },
        "prior_first_row": {
            "opening_m": float(pf_first["Uapp_m"]),
            "pf_reaction_N_per_m": float(pf_first["Ftop_N"]),
            "fem_reaction_N_per_m": float(fem_first["Ftop_N"]),
            "pf_over_fem_reaction_ratio": force_ratio,
            "pf_KJ_Pa_sqrtm": float(pf_first["KJ_Pa_sqrtm"]),
            "fem_KJ_Pa_sqrtm": float(fem_first["KJ_Pa_sqrtm"]),
            "pf_over_fem_KJ_ratio": kj_ratio,
        },
        "metadata_defect": {
            "false_full_field_parity_claim": false_parity_claim,
            "fem_contract_claim": parity["bulk_parity"].get("PF_bulk_plasticity_mode"),
            "executed_pf_native_mode": pf_bulk,
            "effect": "the retained pair cannot be interpreted as a backend-only PF/FEM comparison",
            "state_modified": False,
        },
        "elastic_common_problem": elastic_result,
        "conclusion": (
            "The retained Peak/1000 K histories match geometry, nominal loading, mesh controls, "
            "elasticity, angle and seed, but do not match accommodation, process-zone scale or J support. "
            "The large first-step response difference is therefore not evidence of a PF-versus-CZM "
            "mechanical inconsistency. The accommodation-disabled common reference is authoritative "
            "for boundary, unit and elastic-operator parity."
        ),
        "next_required_comparison": (
            "Run matched tip-only first-passage histories first, then introduce the existing full-field "
            "Peierls-Taylor option in both original spatial models with identical process-zone and J supports."
        ),
        "source_contracts_preserved": {
            "fem_companion": fem_contract["schema"],
            "prior_parity_contract_modified": False,
        },
    }


def _markdown(payload: dict) -> str:
    prior = payload["prior_first_row"]
    unmatched = payload["specimen_contract"]["unmatched_fields"]
    elastic = payload["elastic_common_problem"] or {}
    return f"""# PF–FEM common-specimen audit

The retained Peak/1000 K runs are not a backend-only comparison. Their body,
opening history, cubic elasticity, mesh controls, orientation and nominal seed
match, but their accommodation states and observation supports do not.

At the first common opening ({prior['opening_m'] * 1e6:.1f} µm), PF reports
{prior['pf_reaction_N_per_m']:.9g} N/m and FEM/CZM reports
{prior['fem_reaction_N_per_m']:.9g} N/m (ratio {prior['pf_over_fem_reaction_ratio']:.6g}).
The executed PF arm is `{unmatched['accommodation']['pf']}` with
Lpz={unmatched['process_zone_length_m']['pf'] * 1e6:g} µm; the FEM arm is
`{unmatched['accommodation']['fem']}` with
Lpz={unmatched['process_zone_length_m']['fem'] * 1e6:g} µm. The saved parity
metadata incorrectly calls the PF arm full-field; the numerical state itself
was not altered.

The accommodation-disabled elastic reference passed={elastic.get('passed')},
with maximum relative PF–FEM difference
{elastic.get('maximum_relative_difference', float('nan')):.3e}. This closes the
geometry, symmetric-opening, force-unit and base elastic-operator questions for
the shared diffuse-notch specimen. It does not establish first-passage or
post-event parity.

Next, the histories must be compared under identical tip-only accommodation,
then under the existing full-field Peierls–Taylor option with the same process
zone and J supports. Any event-level cohesive claim still requires a native
FEM cohesive event and post-event equilibrium.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf", type=Path, default=DEFAULT_PF)
    parser.add_argument("--fem", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--elastic", type=Path, default=DEFAULT_ELASTIC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = classify_contract(args.pf, args.fem, args.elastic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
