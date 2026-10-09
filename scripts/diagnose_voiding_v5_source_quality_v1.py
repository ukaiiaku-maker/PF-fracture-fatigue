#!/usr/bin/env python3
"""Capture the actual rejected refinement without changing qualification.

Trusted repository checkpoints only; inventory hashes are verified first.
This diagnostic does not advance any clock, fire an event, or qualify a source.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture import adaptive_multitip_mesh_v11 as adaptive
from arrhenius_fracture import voiding_production_v5 as production
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.closure_mechanics_evidence import canonical_data


def edge_owners(elements):
    owners = defaultdict(list)
    for eid, (a, b, c) in enumerate(elements):
        for x, y in ((a, b), (b, c), (c, a)):
            owners[tuple(sorted((int(x), int(y))))].append(eid)
    return owners


def distance_to_segments(point, segments):
    if not len(segments):
        return None
    a, b = segments[:, 0], segments[:, 1]
    delta = b-a
    fraction = np.clip(np.einsum('ij,ij->i', point-a, delta)/np.maximum(np.sum(delta*delta, axis=1), 1e-300), 0, 1)
    return float(np.min(np.linalg.norm(point-(a+fraction[:, None]*delta), axis=1)))


def diagnose(state, *, lineage=None, subdivided_elements=None):
    nodes, elements = np.asarray(state.mesh.nodes), np.asarray(state.mesh.elems)
    tri = nodes[elements]
    lengths = np.linalg.norm(tri-tri[:, [1, 2, 0]], axis=2)
    areas = np.asarray(state.mesh.area_e)
    quality = 4*np.sqrt(3)*areas/np.sum(lengths*lengths, axis=1)
    owners = edge_owners(elements)
    boundary_edges = [e for e, ids in owners.items() if len(ids) == 1]
    cavity_edges = {tuple(e) for e in production._actual_cavity_boundary_edges(state)}
    outer_edges = set(boundary_edges)-cavity_edges
    outer_nodes = {n for e in outer_edges for n in e}
    prescribed = set(map(int, state.boundary.top_nodes)) | set(map(int, state.boundary.bot_nodes))
    prescribed.update((int(state.boundary.left_bot), int(state.boundary.right_bot)))
    cavity = state.void_state.cavities[0]
    source = np.asarray(cavity.connection_exit_m)
    _, stencil = production.cavity_boundary_tensor(state, boundary_node=int(np.argmin(np.linalg.norm(nodes-source, axis=1))))
    graph_segments = np.asarray([(a, b) for branch in state.crack_network.branches for a, b in zip(branch.path, branch.path[1:])])
    damage = getattr(state.mesh, 'element_damage_gp', None)
    support = np.zeros(len(elements), dtype=bool) if damage is None else np.asarray(damage) > 0
    support_edges = [e for e, ids in owners.items() if any(support[i] for i in ids) and (len(ids) == 1 or not all(support[i] for i in ids))]
    parent_lookup = {} if lineage is None else {child: parent for parent, children in lineage.parent_to_child_element_map.items() for child in children}
    rows = []
    for eid in np.flatnonzero(quality < .05):
        eid = int(eid); p = tri[eid]; centroid = p.mean(axis=0); side = lengths[eid]
        angles = []
        for k in range(3):
            u, v = p[(k+1) % 3]-p[k], p[(k+2) % 3]-p[k]
            angles.append(float(np.degrees(np.arccos(np.clip((u@v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)))))
        parent = parent_lookup.get(eid)
        changed_by_flip = bool(subdivided_elements is not None and not np.array_equal(elements[eid], subdivided_elements[eid]))
        split = bool(parent is not None and len(lineage.parent_to_child_element_map[parent]) > 1)
        marked = bool(parent is not None and parent in lineage.refined_parent_element_ids)
        rows.append({'element_id': eid, 'node_ids': elements[eid].tolist(), 'coordinates_m': p.tolist(),
            'quality': float(quality[eid]), 'aspect_ratio_longest_edge_over_shortest_altitude': float(side.max()**2/(2*areas[eid])),
            'area_m2': float(areas[eid]), 'minimum_angle_deg': min(angles), 'centroid_m': centroid.tolist(),
            'distance_definition': 'centroid to exact owned point / piecewise-linear boundary or support set',
            'distance_to_source_m': float(np.linalg.norm(centroid-source)),
            'distance_to_cavity_boundary_m': distance_to_segments(centroid, nodes[list(sorted(cavity_edges))]),
            'distance_to_crack_m': distance_to_segments(centroid, graph_segments),
            'distance_to_wake_support_m': 0. if support[eid] else distance_to_segments(centroid, nodes[support_edges]),
            'in_source_recovery_stencil': eid in stencil,
            'touches_outer_boundary': bool(set(elements[eid]) & outer_nodes),
            'touches_prescribed_boundary': bool(set(elements[eid]) & prescribed),
            'current_refinement_parent_element_id': parent,
            'refinement_generation': None if lineage is None else lineage.mesh_generation,
            'created_by_source_cavity_refinement': split and marked,
            'created_by_conformity_edge_split': split and not marked,
            'changed_by_quality_flip': changed_by_flip,
            'inherited_unchanged_from_connected_mesh': not split and not changed_by_flip,
            'earlier_graph_or_field_transfer_parentage': 'NOT_RETAINED_IN_BASE_CHECKPOINT_NO_INFERENCE'})
    return {'schema': 'v5.source-quality-element-diagnosis/1', 'minimum_quality': float(quality.min()),
        'required_minimum_quality': .05, 'bad_element_count': len(rows), 'elements': rows,
        'controlling_element_id': int(np.argmin(quality)), 'source_stencil': list(stencil),
        'nodes': len(nodes), 'triangles': len(elements), 'state_fingerprint': fingerprint(state)}


def capture_rejected_trial(connected):
    captured = {}; refinements = []
    original_qualifier, original_refine = production._qualified_cavity_source, adaptive.refine_accepted_state
    def observe_qualification(state, tensor):
        captured['state'] = state
        return original_qualifier(state, tensor)
    def observe_refine(*args, **kwargs):
        result, lineage = original_refine(*args, **kwargs)
        refinements.append((lineage, result.mesh.elems.copy()))
        return result, lineage
    original = fingerprint(connected)
    with patch.object(production, '_qualified_cavity_source', observe_qualification), patch.object(adaptive, 'refine_accepted_state', observe_refine):
        returned, audit = production.refine_downstream_source(connected, max_refinement_levels=1,
            refinement_region='complete_cavity_ring', quality_improvement=True)
    assert fingerprint(connected) == original
    assert captured['state'].competition == connected.competition and captured['state'].rng_state == connected.rng_state
    return captured['state'], audit, refinements, fingerprint(returned) == original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_bundle', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); source, output = args.source_bundle, args.output
    if output.exists(): raise ValueError('refusing to overwrite evidence directory')
    manifest = json.loads((source/'sha256_manifest.json').read_text())
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file() and p.name != 'sha256_manifest.json'}
    if actual != manifest: raise ValueError('trusted source inventory/hash mismatch')
    connected = restore_checkpoint(source/'checkpoints/production_512_192_connected.json')
    print('Capturing actual bounded fine refinement; no event/qualification override', flush=True)
    trial, audit, refinements, returned_original = capture_rejected_trial(connected)
    lineage, subdivided = refinements[-1]
    report = diagnose(trial, lineage=lineage, subdivided_elements=subdivided)
    report.update({'executed_code_sha': subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=ROOT, text=True).strip(),
        'baseline_bundle_manifest_sha256': hashlib.sha256((source/'sha256_manifest.json').read_bytes()).hexdigest(),
        'refinement_audit': audit, 'lineage': asdict(lineage), 'original_returned_by_fail_closed_gate': returned_original})
    output.mkdir(parents=True)
    write_checkpoint(trial, output/'rejected_refined_trial.json')
    np.savez_compressed(output/'mesh.npz', nodes=trial.mesh.nodes, elements=trial.mesh.elems, element_damage_gp=trial.mesh.element_damage_gp)
    (output/'bad_elements.json').write_text(json.dumps(canonical_data(report), sort_keys=True, indent=2, allow_nan=False)+'\n')
    (output/'sha256_manifest.json').write_text(json.dumps({str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('minimum_quality', 'bad_element_count', 'controlling_element_id', 'original_returned_by_fail_closed_gate')}), flush=True)


if __name__ == '__main__': main()
