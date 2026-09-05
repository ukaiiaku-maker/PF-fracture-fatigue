#!/usr/bin/env python3
"""Generate and directly evaluate the combined crack/void static matrix."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.evidence_ontology_v5 import canonical_hash, validate_evidence_rows
from arrhenius_fracture.voiding_production_v5 import deterministic_trajectory


def main(argv=None):
    out = Path(argv[0] if argv else "artifacts/voiding_v5/static")
    out.mkdir(parents=True, exist_ok=True)
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [clean(item) for item in value]
        return value
    specifications = []
    for segments in (32, 64, 128):
        specifications.append((f"kirsch_{segments}", dict(boundary_segments=segments, crack_enabled=False)))
    specifications.append(("crack_only_matched", dict(cavity_enabled=False)))
    for center_x in (7.1e-4, 7.3e-4, 7.5e-4):
        specifications.append((f"far_void_{center_x:.2e}", dict(cavity_center_m=(center_x, 0.0))))
    specifications.append(("centered", dict()))
    for offset in (-2.5e-5, 2.5e-5):
        specifications.append((f"offset_{offset:+.2e}", dict(cavity_center_m=(7.0e-4, offset))))
    for radius in (3.5e-5, 5.0e-5, 6.5e-5):
        for tip_layer in (2, 3, 4):
            specifications.append((f"radius_{radius:.2e}_ligament_layer_{tip_layer}", dict(cavity_radius_m=radius, tip_layer=tip_layer)))
    for tip_layer in (2, 3, 4):
        specifications.append((f"virtual_crack_extension_{tip_layer}", dict(tip_layer=tip_layer)))
    for radius in (4.5e-5, 5.0e-5, 5.5e-5):
        specifications.append((f"virtual_cavity_growth_{radius:.2e}", dict(cavity_radius_m=radius)))
    for layers in (10, 12, 16):
        specifications.append((f"mesh_refinement_{layers}", dict(radial_layers=layers)))

    rows = []
    for case, kwargs in specifications:
        try:
            result = solve_crack_void_case(**kwargs)
        except Exception as error:
            rows.append({"case": case, "executed_operation": "production_static_fem_solve",
                         "configuration": kwargs, "observables": {}, "support_audit": None,
                         "error": f"{type(error).__name__}: {error}", "passed": False})
            continue
        obs = result["observables"]
        finite = all(
            isinstance(obs[name], (int, float)) and obs[name] == obs[name]
            for name in ("reaction_top_N_per_m", "compliance_m2_per_N", "stored_energy_J_per_m",
                         "free_residual_norm_N_per_m")
        )
        support_ok = not result["configuration"]["crack_enabled"] or (
            obs["v12_support_certified"] is True and obs["v12_support_elements"] > 0
            and result["configuration"]["centroid_band_fallback"] is False
        )
        dimensionless = {
            "radius_over_width": float(result["configuration"]["cavity_radius_m"] / 1.0e-3),
            "ligament_over_radius": float(
                (result["configuration"]["cavity_center_m"][0]
                 - result["observables"]["crack_graph_length_m"]
                 - result["configuration"]["cavity_radius_m"])
                / max(result["configuration"]["cavity_radius_m"], 1.0e-300)
            ),
        }
        rows.append({"case": case, "executed_operation": "production_static_fem_solve",
                     "configuration": result["configuration"], "dimensionless": dimensionless,
                     "observables": obs,
                     "support_audit": result["support_audit"], "passed": finite and support_ok})

    _, transferred = deterministic_trajectory(stop_before_ligament=True)
    promotion = next(row for row in transferred if row["operation"] == "geometric_promotion")
    growth = next(row for row in transferred if row["operation"] == "resolved_growth")
    rows.append({
        "case": "mesh_and_nonzero_field_transfer_convergence",
        "executed_operation": "body_fitted_remesh_project_equilibrate",
        "configuration": {"source": "production_voiding_v5"},
        "observables": {
            "promotion": promotion, "resolved_growth": growth,
            "reaction_relative_change": abs(growth["reaction_N_per_m"] - promotion["reaction_N_per_m"]) /
                max(abs(growth["reaction_N_per_m"]), 1e-300),
            "energy_relative_change": abs(growth["energy_J_per_m"] - promotion["energy_J_per_m"]) /
                max(abs(growth["energy_J_per_m"]), 1e-300),
        },
        "support_audit": None,
        "passed": promotion["field_transfer_audit"]["projected_fields_nonzero"]
                  and growth["field_transfer_audit"]["projected_fields_nonzero"],
    })

    by_case = {row["case"]: row for row in rows}
    # Measured centered differences at fixed companion geometry.
    crack = [by_case[f"virtual_crack_extension_{i}"]["observables"] for i in (2, 3, 4)]
    cavity = [by_case[f"virtual_cavity_growth_{r:.2e}"]["observables"] for r in (4.5e-5, 5.0e-5, 5.5e-5)]
    derivatives = {}
    if all(crack) and all(cavity):
        derivatives = {
            "fixed_opening_crack_release_rate_J_per_m2": -(crack[2]["stored_energy_J_per_m"] - crack[0]["stored_energy_J_per_m"]) /
                (crack[2]["crack_graph_length_m"] - crack[0]["crack_graph_length_m"]),
            "cavity_area_conjugate_J_per_m3": -(cavity[2]["stored_energy_J_per_m"] - cavity[0]["stored_energy_J_per_m"]) /
                (cavity[2]["cavity_area_m2"] - cavity[0]["cavity_area_m2"]),
        }
    refinement = [by_case[f"mesh_refinement_{i}"]["observables"] for i in (10, 12, 16)]
    convergence = ({
        "reaction_relative_12_to_16": abs(refinement[2]["reaction_top_N_per_m"] - refinement[1]["reaction_top_N_per_m"]) /
            max(abs(refinement[2]["reaction_top_N_per_m"]), 1e-300),
        "energy_relative_12_to_16": abs(refinement[2]["stored_energy_J_per_m"] - refinement[1]["stored_energy_J_per_m"]) /
            max(abs(refinement[2]["stored_energy_J_per_m"]), 1e-300),
    } if all(refinement) else {})
    declared_tolerances = {
        "reaction_relative_12_to_16_max": 0.06,
        "energy_relative_12_to_16_max": 0.06,
        "mirror_reaction_relative_max": 1.0e-10,
        "kirsch_128_relative_to_analytic_max": 0.10,
        "kirsch_64_to_128_relative_max": 0.02,
    }
    mirror = [by_case[f"offset_{offset:+.2e}"]["observables"] for offset in (-2.5e-5, 2.5e-5)]
    mirror_error = abs(mirror[1]["reaction_top_N_per_m"] - mirror[0]["reaction_top_N_per_m"]) / max(
        abs(mirror[1]["reaction_top_N_per_m"]), 1.0e-300
    )
    kirsch = [by_case[f"kirsch_{segments}"]["observables"]["hoop_stress_concentration"]
              for segments in (32, 64, 128)]
    kirsch_analytic_error = abs(kirsch[-1] - 3.0) / 3.0
    kirsch_fine_change = abs(kirsch[-1] - kirsch[-2]) / abs(kirsch[-1])
    passed = (
        all(row["passed"] for row in rows) and len(derivatives) == 2
        and all(value > 0.0 for value in derivatives.values())
        and convergence["reaction_relative_12_to_16"] <= declared_tolerances["reaction_relative_12_to_16_max"]
        and convergence["energy_relative_12_to_16"] <= declared_tolerances["energy_relative_12_to_16_max"]
        and mirror_error <= declared_tolerances["mirror_reaction_relative_max"]
        and kirsch_analytic_error <= declared_tolerances["kirsch_128_relative_to_analytic_max"]
        and kirsch_fine_change <= declared_tolerances["kirsch_64_to_128_relative_max"]
    )
    implementation_sha = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    source_rows = {f"raw:{index}": row for index, row in enumerate(rows)}
    evidence_rows = []
    for index, row in enumerate(rows):
        configuration = row["configuration"]
        observables = clean(row.get("observables", {}))
        geometry = {"case": row["case"], "configuration": configuration,
                    "mesh_nodes": observables.get("mesh_nodes"), "mesh_elements": observables.get("mesh_elements")}
        evidence_rows.append({
            "case_id": f"static:{index}:{row['case']}", "execution_id": f"static-execution:{index}",
            "input_configuration": configuration, "input_hash": canonical_hash(configuration),
            "actual_realized_geometry": geometry, "actual_geometry_fingerprint": canonical_hash(geometry),
            "actual_operation_trace": [row["executed_operation"]],
            "initial_fingerprint": canonical_hash(configuration), "terminal_fingerprint": canonical_hash(observables),
            "measurement_source": "scripts/qualify_crack_void_static_v5.py",
            "predicate_name": "source_boolean", "predicate_inputs": {"source_bindings": {
                "measurement": {"source_row_id": f"raw:{index}", "path": ["passed"]}
            }},
            "predicate_result": row["passed"], "source_row_ids": [f"raw:{index}"],
            "implementation_sha": implementation_sha,
        })
    signatures = {}
    for evidence in evidence_rows:
        signature = (evidence["input_hash"], evidence["initial_fingerprint"], evidence["terminal_fingerprint"])
        signatures[signature] = signatures.get(signature, 0) + 1
    for evidence in evidence_rows:
        signature = (evidence["input_hash"], evidence["initial_fingerprint"], evidence["terminal_fingerprint"])
        if signatures[signature] > 1:
            evidence["equality_classification"] = "EXPECTED_EQUIVALENCE"
    validate_evidence_rows(evidence_rows, source_rows=source_rows, implementation_sha=implementation_sha)
    payload = {
        "schema": "v12.crack-void-static-qualification/5",
        "implementation_git_sha": implementation_sha,
        "gate": "PASS" if passed else "FAIL",
        "rows": rows, "evidence_rows": evidence_rows, "evidence_ontology_validation": "PASS",
        "fixed_opening_release_rates": derivatives,
        "mesh_convergence": convergence, "mirror_reaction_relative_error": mirror_error,
        "kirsch_hoop_concentration": kirsch,
        "kirsch_128_relative_to_analytic": kirsch_analytic_error,
        "kirsch_64_to_128_relative_change": kirsch_fine_change,
        "declared_tolerances": declared_tolerances,
    }
    payload = clean(payload)
    path = out / "case_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({"case_rows.json": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps({"gate": payload["gate"], "case_count": len(rows), **derivatives, **convergence}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
