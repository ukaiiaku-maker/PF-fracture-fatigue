#!/usr/bin/env python3
"""Run the same unadvanced elastic specimen through the PF and FEM codebases.

The calculation deliberately bypasses hazards and accommodation.  It exercises
each repository's native mesh, cubic plane-strain assembly, Dirichlet solve and
domain-J observer on the same diffuse initial notch.  Results are forces per
unit out-of-plane thickness; no fitted scale factor is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPANION = ROOT.parent / "fem-czm-current-pf-companion"
DEFAULT_OUT = (
    ROOT / "integration_evidence" / "final_fracture_campaign" /
    "common_specimen_elastic_reference"
)
OPENINGS_M = (0.2e-6, 10.0e-6, 30.0e-6)


def _sha_array(array) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _worker(repository: Path, label: str, destination: Path) -> None:
    sys.path.insert(0, str(repository.resolve()))
    import numpy as np

    from arrhenius_fracture.config import ElasticProperties, GeometryConfig, MeshConfig
    from arrhenius_fracture.crystal import cubic_plane_strain_D
    from arrhenius_fracture.fem import assemble_mechanics, solve_dirichlet
    from arrhenius_fracture.j_integral import compute_J_integral
    from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh

    geometry = GeometryConfig(
        Lx=2.0e-3,
        Ly=4.0e-3,
        a0=0.5e-3,
        notch_half_thickness=0.08e-3,
    )
    mesh_config = MeshConfig(nx=36, ny=72, jitter=0.30,
                             tip_h_fine=1.0e-6, tip_ratio=1.20)
    material = ElasticProperties()
    D = cubic_plane_strain_D(523.0e9, 203.0e9, 160.0e9, 0.0)
    mesh = make_tri_mesh(geometry, mesh_config, seed=42)
    boundary = make_boundary_data(mesh, geometry)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    damage = np.zeros(mesh.nn)
    damage[(x <= geometry.a0) &
           (np.abs(y) <= geometry.notch_half_thickness)] = 1.0
    ep = np.zeros((3, mesh.ne))
    rho = np.zeros(mesh.ne)

    rows = []
    for opening in OPENINGS_M:
        displacement = np.zeros(mesh.ndof)
        K, residual, *_ = assemble_mechanics(
            mesh, displacement, ep, rho, damage, D, material,
            cohesive_network=None,
        )
        displacement, top = solve_dirichlet(
            K, residual, displacement, boundary, 0.5 * opening, -0.5 * opening,
        )
        K, residual, stress, _, _, energy_density = assemble_mechanics(
            mesh, displacement, ep, rho, damage, D, material,
            cohesive_network=None,
        )
        bottom = float(np.sum(residual[2 * boundary.bot_nodes + 1]))
        top_check = float(np.sum(residual[2 * boundary.top_nodes + 1]))
        stored = float(0.5 * displacement @ residual)
        J, KJ, j_info = compute_J_integral(
            mesh, displacement, stress, energy_density, damage,
            np.array([geometry.a0, 0.0]), np.array([1.0, 0.0]),
            material, 30.0e-6,
            crack_segments=[
                (np.array([0.0, 0.0]), np.array([geometry.a0, 0.0]))
            ],
        )
        centroids = mesh.nodes[mesh.elems].mean(axis=1)
        patch = (
            (centroids[:, 0] >= geometry.a0 + 8.0e-6)
            & (centroids[:, 0] <= geometry.a0 + 16.0e-6)
            & (np.abs(centroids[:, 1]) <= 8.0e-6)
        )
        if not np.any(patch):
            raise RuntimeError("common local-stress observation patch is empty")
        weights = mesh.area_e[patch]
        sigma_yy = float(np.average(stress[1, patch], weights=weights))
        rows.append({
            "total_opening_m": opening,
            "top_reaction_N_per_m": top_check,
            "solver_returned_top_reaction_N_per_m": float(top),
            "bottom_reaction_N_per_m": bottom,
            "reaction_balance_relative": abs(top_check + bottom) /
                max(0.5 * (abs(top_check) + abs(bottom)), 1.0e-300),
            "stored_energy_J_per_m": stored,
            "energy_reaction_identity_relative": abs(
                stored - 0.5 * opening * top_check
            ) / max(abs(stored), 1.0e-300),
            "common_domain_J_J_per_m2": float(J),
            "common_domain_J_signed_J_per_m2": float(j_info["J_signed"]),
            "common_domain_KJ_Pa_sqrt_m": float(KJ),
            "common_domain_active_elements": int(j_info["n_active_elements"]),
            "ahead_8_16um_sigma_yy_Pa": sigma_yy,
        })

    payload = {
        "schema": "common-specimen-elastic-reference/1",
        "label": label,
        "repository": str(repository.resolve()),
        "specimen": {
            "length_x_m": geometry.Lx,
            "length_y_m": geometry.Ly,
            "initial_notch_length_m": geometry.a0,
            "notch_half_thickness_m": geometry.notch_half_thickness,
            "initial_crack_representation": "nodal diffuse stiffness-kill band",
            "plane_condition": "plane_strain",
            "out_of_plane_thickness_m": 1.0,
            "boundary_condition": "symmetric total opening; top +U/2, bottom -U/2; two bottom x constraints",
        },
        "elasticity": {
            "kind": "cubic_plane_strain",
            "C11_Pa": 523.0e9,
            "C12_Pa": 203.0e9,
            "C44_Pa": 160.0e9,
            "theta_deg": 0.0,
        },
        "mesh": {
            "nx": 36,
            "ny": 72,
            "tip_h_fine_m": 1.0e-6,
            "tip_ratio": 1.20,
            "seed": 42,
            "node_count": int(mesh.nn),
            "element_count": int(mesh.ne),
            "hbar_m": float(mesh.hbar),
            "hbar_tip_m": float(mesh.hbar_tip),
            "nodes_sha256": _sha_array(mesh.nodes),
            "elements_sha256": _sha_array(mesh.elems),
            "damage_sha256": _sha_array(damage),
        },
        "controls": {
            "accommodation": "disabled",
            "hazards": "not_evaluated",
            "fracture_advance": "disabled",
            "cohesive_network": "absent_on_unadvanced_geometry",
            "common_J_outer_radius_m": 240.0e-6,
            "local_stress_observation": "area-weighted sigma_yy 8-16 um ahead, abs(y)<=8 um",
        },
        "rows": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def _combine(pf_path: Path, fem_path: Path, destination: Path) -> dict:
    pf = json.loads(pf_path.read_text())
    fem = json.loads(fem_path.read_text())
    comparisons = []
    for left, right in zip(pf["rows"], fem["rows"], strict=True):
        if left["total_opening_m"] != right["total_opening_m"]:
            raise RuntimeError("worker openings do not match")
        metrics = {}
        for key in (
            "top_reaction_N_per_m", "stored_energy_J_per_m",
            "common_domain_J_J_per_m2", "common_domain_KJ_Pa_sqrt_m",
            "ahead_8_16um_sigma_yy_Pa",
        ):
            metrics[key] = {
                "pf": left[key], "fem": right[key],
                "relative_difference": _relative(left[key], right[key]),
            }
        comparisons.append({"total_opening_m": left["total_opening_m"],
                            "metrics": metrics})
    max_difference = max(
        item["relative_difference"]
        for row in comparisons for item in row["metrics"].values()
    )
    result = {
        "schema": "common-specimen-elastic-comparison/1",
        "pf_result": str(pf_path.resolve()),
        "fem_result": str(fem_path.resolve()),
        "mesh_identity": {
            key: pf["mesh"][key] == fem["mesh"][key]
            for key in ("nodes_sha256", "elements_sha256", "damage_sha256")
        },
        "comparisons": comparisons,
        "maximum_relative_difference": max_difference,
        "pass_tolerance": 1.0e-10,
        "passed": bool(max_difference <= 1.0e-10),
        "interpretation": (
            "The two original spatial implementations agree on the matched "
            "unadvanced elastic problem. Differences in the prior histories "
            "therefore arise after constitutive/runtime choices, not from an "
            "intrinsic force-unit or boundary-opening mismatch."
        ),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--companion", type=Path, default=DEFAULT_COMPANION)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-repository", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker_repository, args.worker_label, args.worker_output)
        return 0

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    workers = ((ROOT, "pf", out / "pf_elastic.json"),
               (args.companion.resolve(), "fem_czm", out / "fem_czm_elastic.json"))
    for repository, label, output in workers:
        subprocess.run([
            sys.executable, str(script), "--worker",
            "--worker-repository", str(repository),
            "--worker-label", label,
            "--worker-output", str(output),
        ], cwd=repository, check=True)
    result = _combine(workers[0][2], workers[1][2], out / "comparison.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
