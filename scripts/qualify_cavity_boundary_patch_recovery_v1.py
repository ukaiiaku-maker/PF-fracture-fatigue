#!/usr/bin/env python3
"""Read-only recovery sentinels on retained raw FEM evidence, not new FEM solves."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.cavity_boundary_patch_recovery_v1 import (
    OPERATOR_ID, RecoveryUnavailable, polygon_arc_source, recover_boundary_tensor,
)

ARCS = (0., .125, .25, .375, .5)
PLANES = ((1., 0.), (0., 1.), (2**-.5, 2**-.5), (2**-.5, -2**-.5))
LEVELS = ((32, 12), (64, 24), (128, 48), (512, 192))


def kirsch_tensor(points):
    """Infinite unit-radius cavity, unit x-direction far-field tension."""
    points = np.asarray(points)
    angle = np.arctan2(points[:, 1], points[:, 0])
    ratio = 1/np.sum(points**2, axis=1)
    rr = .5*(1-ratio)+.5*(1-4*ratio+3*ratio**2)*np.cos(2*angle)
    tt = .5*(1+ratio)-.5*(1+3*ratio**2)*np.cos(2*angle)
    rt = -.5*(1+2*ratio-3*ratio**2)*np.sin(2*angle)
    normals = np.column_stack((np.cos(angle), np.sin(angle)))
    tangents = np.column_stack((-np.sin(angle), np.cos(angle)))
    basis = np.stack((normals, tangents), axis=2)
    local = np.zeros((len(points), 2, 2))
    local[:, 0, 0], local[:, 1, 1] = rr, tt
    local[:, 0, 1] = local[:, 1, 0] = rt
    tensor = basis@local@basis.transpose(0, 2, 1)
    return .5*(tensor+tensor.transpose(0, 2, 1))


def annulus(segments, layers):
    angles = np.arange(segments)*2*np.pi/segments
    nodes = np.array([[r*np.cos(a), r*np.sin(a)] for r in 1+np.arange(4)/layers for a in angles])
    elems = []
    for j in range(3):
        for i in range(segments):
            a, b = j*segments+i, j*segments+(i+1)%segments
            c, d = a+segments, b+segments
            elems.extend(((a, c, d), (a, d, b)))
    return nodes, np.array(elems), np.array([(i, (i+1)%segments) for i in range(segments)])


def recover_arcs(raw, center):
    rows = []
    for arc in ARCS:
        point, normal = polygon_arc_source(raw["nodes"], raw["cavity_edges"], center=center, arc_fraction=arc)
        kwargs = dict(boundary_edges=raw["cavity_edges"], source_point=point, normal=normal,
                      intact_mask=~np.asarray(raw["support_mask"], dtype=bool), candidate_normals=PLANES)
        try:
            result = recover_boundary_tensor(raw["nodes"], raw["elements"], raw["stress"], **kwargs)
            peer = recover_boundary_tensor(raw["nodes"], raw["elements"], raw["stress"], **kwargs)
            reverse = recover_boundary_tensor(raw["nodes"], raw["elements"], raw["stress"],
                **dict(kwargs, boundary_edges=raw["cavity_edges"][::-1, ::-1]))
            rows.append(dict(arc_fraction=arc, recovery=result, repeat_exact=result == peer,
                             reversed_edge_order_exact=result == reverse))
        except RecoveryUnavailable as error:
            rows.append(dict(arc_fraction=arc, status="RECOVERY_UNAVAILABLE", reason=str(error)))
    return rows


def verified_npz(path, manifest):
    key = str(path.relative_to(manifest.parent))
    expected = json.loads(manifest.read_text())[key]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("retained source hash mismatch: "+str(path))
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def compare(rows, reference):
    comparisons = []
    for row, ref in zip(rows, reference):
        if "recovery" not in row or "recovery" not in ref:
            comparisons.append(dict(arc_fraction=row["arc_fraction"], qualified=False, reason="RECOVERY_UNAVAILABLE"))
            continue
        a, b = np.asarray(row["recovery"]["tensor_Pa"]), np.asarray(ref["recovery"]["tensor_Pa"])
        error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), np.finfo(float).tiny))
        comparisons.append(dict(arc_fraction=row["arc_fraction"], tensor_relative_error=error,
                                tensor_gate=error <= .05, source_qualified=False))
    return comparisons


def execute(retained):
    result = dict(schema="v5.cavity-boundary-patch-recovery-sentinel/1", operator_id=OPERATOR_ID,
        implementation_sha=subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        operator_file_sha256=hashlib.sha256((ROOT/"arrhenius_fracture/cavity_boundary_patch_recovery_v1.py").read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/"docs/CAVITY_BOUNDARY_PATCH_RECOVERY_V1_PROTOCOL.md").read_bytes()).hexdigest(),
        interpretation="POSTPROCESSING_EXISTING_SOLVES_NOT_NEW_PHYSICAL_EXECUTIONS",
        kirsch=[], static={}, fixed_crack_offsets={}, production=[], source_qualified=False)
    for segments, layers in LEVELS:
        nodes, elems, edges = annulus(segments, layers)
        stress = kirsch_tensor(nodes[elems].mean(axis=1))
        rows = recover_arcs(dict(nodes=nodes, elements=elems, cavity_edges=edges, stress=stress,
            support_mask=np.zeros(len(elems), dtype=bool)), (0., 0.))
        errors = []
        for row in rows:
            point = np.asarray(row["recovery"]["source_point_m"])
            reference = kirsch_tensor(np.array([point]))[0]
            error = float(np.linalg.norm(np.asarray(row["recovery"]["tensor_Pa"])-reference)/max(np.linalg.norm(reference), 1.))
            row["analytic_tensor_Pa"] = reference.tolist()
            row["analytic_error_normalized_by_max_tensor_or_unit_remote_stress"] = error
            errors.append(error)
        result["kirsch"].append(dict(segments=segments, layers=layers, rows=rows, maximum_error=max(errors)))
    errors = [row["maximum_error"] for row in result["kirsch"]]
    result["kirsch_operator_gate"] = errors[-1] <= .03 and all(b < a for a, b in zip(errors, errors[1:]))
    for kind in ("cavity", "cracked"):
        levels = LEVELS if kind == "cavity" else ((128, 48), (256, 96), (512, 192))
        output = {}
        for segments, layers in levels:
            path = retained/"traction/sources"/f"{kind}_{segments}_{layers}.npz"
            raw = verified_npz(path, retained/"traction/sha256_manifest.json")
            output[f"{segments}/{layers}"] = dict(source_path=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=recover_arcs(raw, (7e-4, 0.)))
        for value in output.values():
            value["comparison_to_512_192"] = compare(value["rows"], output["512/192"]["rows"])
        result["static"][kind] = output
    from arrhenius_fracture.closure_mechanics_evidence import GROUPS, REGISTRY
    for offset in (-4e-5, 4e-5):
        output = {}
        for segments, layers in ((128, 48), (256, 96)):
            key = GROUPS[f"offset:{offset}:{segments}:{layers}"]
            path = retained/"mechanics/sources"/(key+".npz")
            raw = verified_npz(path, retained/"mechanics/sha256_manifest.json")
            output[f"{segments}/{layers}"] = dict(source_path=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fixed_crack_path_m=REGISTRY[key]["crack_path_m"],
                cavity_center_m=REGISTRY[key]["cavity_center_m"],
                rows=recover_arcs(raw, REGISTRY[key]["cavity_center_m"]))
        output["128/48"]["comparison_to_256_96"] = compare(output["128/48"]["rows"], output["256/96"]["rows"])
        result["fixed_crack_offsets"][str(offset)] = output
    # The independent checkpoint loader verifies payload hashes before decoding.
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
    from arrhenius_fracture.fem import assemble_mechanics
    from arrhenius_fracture.voiding_production_v5 import _actual_cavity_boundary_edges
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    manifest_path = retained/"production/sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for segments, layers in LEVELS:
        path = retained/"production/checkpoints"/f"production_{segments}_{layers}_connected.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[str(path.relative_to(manifest_path.parent))]:
            raise ValueError("retained checkpoint manifest hash mismatch")
        state = restore_checkpoint(path)
        before = complete_accepted_state_fingerprint(state)
        _, _, stress, *_ = assemble_mechanics(state.mesh, state.displacement, state.ep_gp, state.rho_gp,
            state.damage, state.elasticity_D, state.material, cohesive_network=state.cohesive_network)
        cavity = state.void_state.cavities[0]
        point = np.asarray(cavity.connection_exit_m)
        normal = point-np.asarray(cavity.center_m); normal /= np.linalg.norm(normal)
        mask = np.asarray(state.mesh.element_damage_gp) if state.mesh.element_damage_gp is not None else np.zeros(state.mesh.ne)
        kwargs = dict(boundary_edges=_actual_cavity_boundary_edges(state), source_point=point,
                      normal=normal, intact_mask=mask == 0, candidate_normals=PLANES)
        row = dict(segments=segments, layers=layers, source_path=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        try:
            recovered = recover_boundary_tensor(state.mesh.nodes, state.mesh.elems, stress, **kwargs)
            peer = recover_boundary_tensor(state.mesh.nodes, state.mesh.elems, stress, **kwargs)
            row.update(recovery=recovered, repeat_exact=recovered == peer)
        except RecoveryUnavailable as error:
            row.update(status="RECOVERY_UNAVAILABLE", reason=str(error))
        row["accepted_state_unchanged"] = complete_accepted_state_fingerprint(state) == before
        result["production"].append(row)
    reference = result["production"][-1]
    for row in result["production"]:
        row["comparison_to_512_192"] = compare([dict(row, arc_fraction="owned_connection_exit")],
                                                 [dict(reference, arc_fraction="owned_connection_exit")])[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retained", type=Path, default=ROOT/"artifacts/voiding_v5_finalization_v3_closure/complete_attempt_20260907/a")
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("refusing to overwrite evidence")
    result = execute(args.retained)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(kirsch_operator_gate=result["kirsch_operator_gate"],
        kirsch_errors=[r["maximum_error"] for r in result["kirsch"]],
        production=[dict(segments=r["segments"], comparison=r["comparison_to_512_192"]) for r in result["production"]]), indent=2))


if __name__ == "__main__": main()
