#!/usr/bin/env python3
"""Bounded manufactured qualification of the prospectively frozen V2 verifier."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2 import (
    recover_piecewise_equilibrated_boundary_stress_v2,
)
from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2_contract import (
    ACCEPTANCE_LIMITS, CONTRACT_ID, OPERATOR_ID,
)

OUTPUT = ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/qualification.json"


def _annulus(radial_layers=5, sectors=48, rotation=0.0, reflected=False):
    theta = rotation + np.arange(sectors) * 2 * np.pi / sectors
    if reflected: theta = -theta
    radii = np.linspace(1., 1.5, radial_layers + 1)
    nodes = np.vstack([np.column_stack((r*np.cos(theta), r*np.sin(theta))) for r in radii])
    elements = []
    for layer in range(radial_layers):
        for index in range(sectors):
            nxt, inner, outer = (index+1) % sectors, layer*sectors, (layer+1)*sectors
            for original in ((inner+index, outer+index, outer+nxt),
                             (inner+index, outer+nxt, inner+nxt)):
                triangle = list(original); points = nodes[triangle]
                a, b = points[1]-points[0], points[2]-points[0]
                if a[0]*b[1]-a[1]*b[0] < 0: triangle[1], triangle[2] = triangle[2], triangle[1]
                elements.append(triangle)
    return nodes, np.asarray(elements), [(i, (i+1) % sectors) for i in range(sectors)], 0


def _constant(points):
    out = np.zeros((len(points), 2, 2)); out[:,0,0]=3.; out[:,1,1]=7.; out[:,0,1]=out[:,1,0]=2.
    return out


def _affine(points):
    x,y=points.T; out=np.zeros((len(points),2,2)); out[:,0,0]=3+2*x+.5*y
    out[:,1,1]=7-1.5*x+4*y; out[:,0,1]=out[:,1,0]=2-4*x-2*y; return out


def _kirsch(points):
    x,y=points.T; r=np.hypot(x,y); theta=np.arctan2(y,x); c2=np.cos(2*theta); s2=np.sin(2*theta)
    rr=5*(1-r**-2)+5*(1-4*r**-2+3*r**-4)*c2
    tt=5*(1+r**-2)-5*(1+3*r**-4)*c2; rt=-5*(1+2*r**-2-3*r**-4)*s2
    er=np.column_stack((np.cos(theta),np.sin(theta))); et=np.column_stack((-np.sin(theta),np.cos(theta)))
    return (rr[:,None,None]*er[:,:,None]*er[:,None,:]+tt[:,None,None]*et[:,:,None]*et[:,None,:]
            +rt[:,None,None]*(er[:,:,None]*et[:,None,:]+et[:,:,None]*er[:,None,:]))


def _rotated_kirsch(angle):
    rotation=np.asarray(((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle))))
    def field(points):
        unrotated=np.asarray(points)@rotation
        return np.einsum("ia,nab,jb->nij",rotation,_kirsch(unrotated),rotation)
    return field


def _run(field, *, reverse=False, **mesh):
    nodes,elements,edges,source=_annulus(**mesh)
    if reverse: edges=[edge[::-1] for edge in edges[::-1]]
    return recover_piecewise_equilibrated_boundary_stress_v2(
        nodes=nodes,elements=elements,stress_Pa=field(nodes[elements].mean(axis=1)),
        boundary_node=source,cavity_center_m=(0.,0.),cavity_radius_m=1.,process_length_m=1.,
        owned_boundary_edges=edges,remote_traction_scale_Pa=10.,state_fingerprint="V2_QUALIFICATION")


def _compact(row):
    return {key: row[key] for key in (
        "normalized_boundary_traction", "stress_fit_residual", "equilibrium_residual",
        "interface_traction_jump_residual", "resultant_residual", "moment_residual",
        "rank", "required_rank", "condition", "minimum_support_quality", "window_identity",
    )}


def build_record():
    cases = {
        "constant_stress": _run(_constant, radial_layers=5, sectors=48),
        "affine_self_equilibrated_stress": _run(_affine, radial_layers=5, sectors=48),
        "kirsch_coarse": _run(_kirsch, radial_layers=5, sectors=48),
        "kirsch_fine": _run(_kirsch, radial_layers=8, sectors=96),
        "rigid_rotation": _run(_rotated_kirsch(np.pi/7), radial_layers=8, sectors=96, rotation=np.pi/7),
        "reflected_cavity_geometry": _run(_kirsch, radial_layers=8, sectors=96, reflected=True),
        "reversed_normals_and_edge_order": _run(_kirsch, radial_layers=8, sectors=96, reverse=True),
    }
    common = all(
        row["stress_fit_residual"] <= ACCEPTANCE_LIMITS["maximum_stress_fit_residual"]
        and row["equilibrium_residual"] <= ACCEPTANCE_LIMITS["maximum_equilibrium_residual"]
        and row["interface_traction_jump_residual"] <= ACCEPTANCE_LIMITS["maximum_interface_traction_jump_relative"]
        and row["resultant_residual"] <= ACCEPTANCE_LIMITS["maximum_resultant_residual_relative"]
        and row["moment_residual"] <= ACCEPTANCE_LIMITS["maximum_moment_residual_relative"]
        and row["rank"] == row["required_rank"]
        and row["condition"] <= ACCEPTANCE_LIMITS["maximum_scaled_condition"]
        for row in cases.values()
    )
    coarse, fine = cases["kirsch_coarse"], cases["kirsch_fine"]
    predicates = {
        "manufactured_and_balance_constraints": common,
        "kirsch_boundary_traction": max(coarse["normalized_boundary_traction"], fine["normalized_boundary_traction"]) <= .05,
        "kirsch_mesh_convergence": abs(fine["normalized_boundary_traction"]-coarse["normalized_boundary_traction"]) <= .05,
        "rotation": abs(cases["rigid_rotation"]["normalized_boundary_traction"]-fine["normalized_boundary_traction"]) <= 1e-10,
        "reflection": cases["reflected_cavity_geometry"]["normalized_boundary_traction"] <= .05,
        "edge_order": abs(cases["reversed_normals_and_edge_order"]["normalized_boundary_traction"]-fine["normalized_boundary_traction"]) <= 1e-13,
        "zero_boundary_traction_not_imposed": all(not row["zero_boundary_traction_imposed"] for row in cases.values()),
    }
    return {"schema":"unified-2d-one-void.piecewise-equilibrated-qualification/1",
            "contract":CONTRACT_ID,"operator":OPERATOR_ID,"cases":{k:_compact(v) for k,v in cases.items()},
            "predicates":predicates,"passed":all(predicates.values())}


def main():
    record=build_record(); OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("PIECEWISE_V2_QUALIFICATION="+("PASS" if record["passed"] else "FAIL")); return 0 if record["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
