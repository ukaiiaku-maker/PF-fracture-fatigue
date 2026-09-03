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
from arrhenius_fracture.crack_void_mechanics_v4 import solve_crack_void_case


def main(argv=None):
    out = Path(argv[0] if argv else "artifacts/voiding_v4/static")
    out.mkdir(parents=True, exist_ok=True)
    specifications = []
    for segments in (32, 64):
        specifications.append((f"kirsch_{segments}", dict(boundary_segments=segments, crack_enabled=False)))
    specifications.append(("crack_only_matched", dict(cavity_enabled=False)))
    specifications.append(("far_void", dict(cavity_center_m=(7.5e-4, 0.0))))
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
        rows.append({"case": case, "executed_operation": "production_static_fem_solve",
                     "configuration": result["configuration"], "observables": obs,
                     "support_audit": result["support_audit"], "passed": finite and support_ok})

    by_case = {row["case"]: row for row in rows}
    # Measured centered differences at fixed companion geometry.
    crack = [by_case[f"virtual_crack_extension_{i}"]["observables"] for i in (2, 3, 4)]
    cavity = [by_case[f"virtual_cavity_growth_{r:.2e}"]["observables"] for r in (4.5e-5, 5.0e-5, 5.5e-5)]
    derivatives = {}
    if all(crack) and all(cavity):
        derivatives = {
            "virtual_crack_energy_derivative_J_per_m2": (crack[2]["stored_energy_J_per_m"] - crack[0]["stored_energy_J_per_m"]) /
                (crack[2]["crack_graph_length_m"] - crack[0]["crack_graph_length_m"]),
            "virtual_cavity_energy_derivative_J_per_m2": (cavity[2]["stored_energy_J_per_m"] - cavity[0]["stored_energy_J_per_m"]) /
                (cavity[2]["cavity_area_m2"] - cavity[0]["cavity_area_m2"]),
        }
    refinement = [by_case[f"mesh_refinement_{i}"]["observables"] for i in (10, 12, 16)]
    convergence = ({
        "reaction_relative_12_to_16": abs(refinement[2]["reaction_top_N_per_m"] - refinement[1]["reaction_top_N_per_m"]) /
            max(abs(refinement[2]["reaction_top_N_per_m"]), 1e-300),
        "energy_relative_12_to_16": abs(refinement[2]["stored_energy_J_per_m"] - refinement[1]["stored_energy_J_per_m"]) /
            max(abs(refinement[2]["stored_energy_J_per_m"]), 1e-300),
    } if all(refinement) else {})
    passed = all(row["passed"] for row in rows) and len(derivatives) == 2
    payload = {
        "schema": "v12.crack-void-static-qualification/4",
        "implementation_git_sha": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "gate": "PASS" if passed else "FAIL",
        "rows": rows, "energy_derivatives": derivatives, "mesh_convergence": convergence,
    }
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [clean(item) for item in value]
        return value
    payload = clean(payload)
    path = out / "case_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({"case_rows.json": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps({"gate": payload["gate"], "case_count": len(rows), **derivatives, **convergence}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
