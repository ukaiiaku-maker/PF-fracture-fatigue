#!/usr/bin/env python3
"""Genuine annular Kirsch boundary-value sentinel for the unconstrained operator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.fem import assemble_mechanics, plane_strain_D
from qualify_cavity_boundary_patch_recovery_v1 import kirsch_tensor, recover_arcs


def kirsch_displacement(points, E, nu, remote_stress):
    """Plane-strain displacement obtained by integrating exact Kirsch strains.

    epsilon_rr=((1-nu)*sigma_rr-nu*sigma_tt)/(2*mu), likewise theta.
    With u_r=A+B*cos(2theta), u_theta=C*sin(2theta), the following A,B,C
    satisfy both normal strains and sigma_rtheta=2*mu*epsilon_rtheta.
    """
    radius = np.linalg.norm(points, axis=1)
    angle = np.arctan2(points[:, 1], points[:, 0])
    mu = E/(2*(1+nu)); scale = remote_stress/(4*mu)
    a = (1-2*nu)*radius+1/radius
    b = radius+(4-4*nu)/radius-1/radius**3
    c = -radius+(4*nu-2)/radius-1/radius**3
    ur, ut = scale*(a+b*np.cos(2*angle)), scale*c*np.sin(2*angle)
    return np.column_stack((ur*np.cos(angle)-ut*np.sin(angle), ur*np.sin(angle)+ut*np.cos(angle)))


def validate_raw_source(raw):
    """Independently reconstruct stiffness/stress/residual from retained u."""
    mesh = rebuild_tri_mesh(raw["nodes"], raw["elements"])
    material = ElasticProperties(E=210e9, nu=.3)
    D = plane_strain_D(material)
    K, residual, stress, *_ = assemble_mechanics(mesh, raw["displacement"],
        np.zeros((3, mesh.ne)), np.zeros(mesh.ne), np.zeros(mesh.nn), D, material, kappa=0.)
    K = K.tocsr(); K.sum_duplicates(); K.sort_indices()
    for key, value in (("K_data", K.data), ("K_indices", K.indices), ("K_indptr", K.indptr),
                       ("K_shape", K.shape), ("elasticity_D", D), ("stress", stress),
                       ("assembled_residual", residual)):
        if not np.array_equal(raw[key], value):
            raise ValueError("Kirsch source reassembly mismatch: "+key)
    fixed = raw["prescribed_dofs"]
    expected = kirsch_displacement(raw["nodes"][np.unique(fixed//2)], material.E, material.nu, 1e6).ravel()
    if not np.array_equal(raw["displacement"][fixed], expected):
        raise ValueError("Kirsch outer displacement mismatch")
    return dict(reassembled_arrays_exact=True, prescribed_displacement_exact=True)


def solve(segments, layers):
    angles = np.arange(segments)*2*np.pi/segments
    radii = np.geomspace(1., 5., layers+1)
    nodes = np.array([[r*np.cos(t), r*np.sin(t)] for r in radii for t in angles])
    elements = []
    for j in range(layers):
        for i in range(segments):
            a, b = j*segments+i, j*segments+(i+1)%segments
            c, d = a+segments, b+segments
            elements.extend(((a, c, d), (a, d, b)))
    mesh = rebuild_tri_mesh(nodes, np.array(elements))
    material = ElasticProperties(E=210e9, nu=.3)
    D = plane_strain_D(material)
    ep, rho, damage = np.zeros((3, mesh.ne)), np.zeros(mesh.ne), np.zeros(mesh.nn)
    u = np.zeros(2*mesh.nn)
    K, *_ = assemble_mechanics(mesh, u, ep, rho, damage, D, material, kappa=0.)
    fixed = np.arange(2*segments*layers, 2*mesh.nn)
    u[fixed] = kirsch_displacement(nodes[segments*layers:], material.E, material.nu, 1e6).ravel()
    free = np.setdiff1d(np.arange(len(u)), fixed)
    u[free] = spsolve(K[free][:, free], -(K[free][:, fixed]@u[fixed]))
    _, residual, stress, *_ = assemble_mechanics(mesh, u, ep, rho, damage, D, material, kappa=0.)
    K = K.tocsr(); K.sum_duplicates(); K.sort_indices()
    raw = dict(nodes=mesh.nodes, elements=mesh.elems, displacement=u, stress=stress,
        cavity_edges=np.array([(i, (i+1)%segments) for i in range(segments)]),
        support_mask=np.zeros(mesh.ne, dtype=bool), assembled_residual=residual,
        K_data=K.data, K_indices=K.indices, K_indptr=K.indptr, K_shape=np.array(K.shape),
        elasticity_D=D, prescribed_dofs=fixed, remote_stress_Pa=np.array(1e6))
    validation = validate_raw_source(raw)
    rows = recover_arcs(raw, (0., 0.))
    errors = []
    for row in rows:
        point = row["recovery"]["source_point_m"]
        exact = 1e6*kirsch_tensor(np.array([point]))[0]
        error = float(np.linalg.norm(np.asarray(row["recovery"]["tensor_Pa"])-exact)/max(np.linalg.norm(exact), 1e6))
        row.update(analytic_tensor_Pa=exact.tolist(), analytic_error=error)
        errors.append(error)
    return raw, dict(segments=segments, layers=layers, rows=rows, maximum_error=max(errors), source_validation=validation,
        free_residual_relative=float(np.linalg.norm(residual[free])/np.linalg.norm(residual[fixed])),
        global_minimum_quality=float(np.min(4*np.sqrt(3)*mesh.area_e/
            np.sum((mesh.nodes[mesh.elems[:, [1, 2, 0]]]-mesh.nodes[mesh.elems])**2, axis=(1, 2)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fine-extension", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    rows = []
    levels = ((256, 96), (512, 192)) if args.fine_extension else ((32, 12), (64, 24), (128, 48))
    for segments, layers in levels:
        raw, row = solve(segments, layers)
        path = args.output/f"kirsch_{segments}_{layers}.npz"
        np.savez_compressed(path, **raw)
        row.update(source_file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    errors = [r["maximum_error"] for r in rows]
    result = dict(schema="v5.cavity-boundary-patch-kirsch-fem-fine-extension/1" if args.fine_extension else "v5.cavity-boundary-patch-kirsch-fem/1", rows=rows,
        fixed_criteria=dict(kirsch_relative=.03, strictly_decreasing=True),
        operator_id="CAVITY_BOUNDARY_PATCH_RECOVERY_V1",
        passed=errors[-1] <= .03 and all(b < a for a, b in zip(errors, errors[1:])) and
               (not args.fine_extension or all(r["global_minimum_quality"] >= .05 for r in rows)),
        implementation_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/"docs/CAVITY_BOUNDARY_PATCH_RECOVERY_V1_PROTOCOL.md").read_bytes()).hexdigest())
    if args.fine_extension:
        result["fine_protocol_sha256"] = hashlib.sha256((ROOT/"docs/CAVITY_PATCH_KIRSCH_FINE_EXTENSION_V1_PROTOCOL.md").read_bytes()).hexdigest()
    (args.output/"report.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=result["passed"], errors=errors)))


if __name__ == "__main__": main()
