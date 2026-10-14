#!/usr/bin/env python3
"""Run a compact affine triangular-patch qualification of transverse controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.mechanics_state_contract import (
    ConstraintControl, MechanicsStateOptions, PlaneState,
    constrained_elastic_potential_density, full_tensor_from_in_plane,
    rotate_tensor,
)
from arrhenius_fracture.mesh import rebuild_tri_mesh


CASES = {
    "plane_strain": MechanicsStateOptions(),
    "plane_stress": MechanicsStateOptions(
        plane_state=PlaneState.PLANE_STRESS,
        constraint_control=ConstraintControl.FREE_TRANSVERSE,
        eps_zz=None,
    ),
    "generalized_prescribed_eps_zz": MechanicsStateOptions(
        plane_state=PlaneState.GENERALIZED_PLANE_STRAIN,
        constraint_control=ConstraintControl.PRESCRIBED_EPS_ZZ,
        eps_zz=2.5e-4,
    ),
    "generalized_prescribed_sigma_zz": MechanicsStateOptions(
        plane_state=PlaneState.GENERALIZED_PLANE_STRAIN,
        constraint_control=ConstraintControl.PRESCRIBED_SIGMA_ZZ,
        eps_zz=None,
        sigma_zz_Pa=75e6,
    ),
}


def _patch_strains(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    mesh = rebuild_tri_mesh(nodes, elems)
    exx, eyy, gamma = q
    u = np.column_stack((exx * nodes[:, 0] + 0.5 * gamma * nodes[:, 1],
                         0.5 * gamma * nodes[:, 0] + eyy * nodes[:, 1])).ravel()
    edofs = np.column_stack((2 * elems[:, 0], 2 * elems[:, 0] + 1,
                             2 * elems[:, 1], 2 * elems[:, 1] + 1,
                             2 * elems[:, 2], 2 * elems[:, 2] + 1))
    return np.einsum("eij,ej->ei", mesh.B_e, u[edofs]), mesh.area_e


def _potential(q: np.ndarray, options: MechanicsStateOptions,
               E_Pa: float, nu: float) -> float:
    eps, sig = full_tensor_from_in_plane(*q, E_Pa, nu, options)
    return constrained_elastic_potential_density(sig, eps, options)


def qualify() -> dict:
    E_Pa, nu = 210e9, 0.29
    target = np.array([1.2e-3, -2.0e-4, 4.0e-4])
    patch, areas = _patch_strains(target)
    angle = np.deg2rad(37.0)
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rows = {}
    for name, options in CASES.items():
        tensors = [full_tensor_from_in_plane(*row, E_Pa, nu, options)
                   for row in patch]
        strains = np.stack([item[0] for item in tensors])
        stresses = np.stack([item[1] for item in tensors])
        h = 1e-8
        derivative = np.array([
            (_potential(target + np.eye(3)[i] * h, options, E_Pa, nu)
             - _potential(target - np.eye(3)[i] * h, options, E_Pa, nu))
            / (2.0 * h)
            for i in range(3)
        ])
        expected = np.array([stresses[0, 0, 0], stresses[0, 1, 1],
                             stresses[0, 0, 1]])
        derivative_error = float(np.linalg.norm(derivative - expected)
                                 / np.linalg.norm(expected))
        rotation_work_error = max(abs(
            constrained_elastic_potential_density(rotate_tensor(sig, Q),
                                                   rotate_tensor(eps, Q), options)
            - constrained_elastic_potential_density(sig, eps, options)
        ) for eps, sig in tensors)
        transverse_target = (0.0 if options.plane_state is PlaneState.PLANE_STRESS
                             else options.sigma_zz_Pa)
        transverse_error = (0.0 if transverse_target is None else
                            float(np.max(np.abs(stresses[:, 2, 2]
                                                - transverse_target))))
        uniform_error = float(max(np.max(np.abs(patch - target)),
                                  np.max(np.abs(strains - strains[0])),
                                  np.max(np.abs(stresses - stresses[0]))))
        passed = (uniform_error <= 1e-6 and transverse_error <= 1e-5
                  and derivative_error <= 1e-9 and rotation_work_error <= 1e-6)
        rows[name] = {
            "status": "PASS" if passed else "FAIL",
            "options": options.to_dict(),
            "element_count": int(len(areas)),
            "total_area": float(np.sum(areas)),
            "uniform_patch_absolute_error": uniform_error,
            "transverse_stress_absolute_error_Pa": transverse_error,
            "reduced_potential_derivative_relative_error": derivative_error,
            "rotation_potential_absolute_error_J_m3": float(rotation_work_error),
            "strain_tensor": strains[0].tolist(),
            "stress_tensor_Pa": stresses[0].tolist(),
        }
    return {
        "schema": "mechanics-constraint-spatial-patch/1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL",
        "classification": "compact affine triangular-patch readiness qualification",
        "production_execution_authorized": False,
        "scope": "full tensor, transverse control, rotation, work potential, and derivative on a two-element affine patch; not a cracked-body or 3-D qualification",
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
