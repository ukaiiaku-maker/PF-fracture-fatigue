"""Deterministic quality repair with immutable geometry/material interfaces.

This module proposes meshes, never qualifies a source or commits physical state.
Its caller must project state, rebuild/certify V12 support, equilibrate and check
all frozen mechanics/source predicates before accepting a transaction.
"""
from collections import defaultdict
from dataclasses import replace
import hashlib

import numpy as np
from scipy.optimize import minimize

from .mesh import rebuild_tri_mesh

SCHEMA = 'v5.quality-constrained-mesh/1'


def qualities(nodes, elements):
    p = np.asarray(nodes)[np.asarray(elements)]
    twice_area = np.abs((p[:, 1, 0]-p[:, 0, 0])*(p[:, 2, 1]-p[:, 0, 1])
                        -(p[:, 1, 1]-p[:, 0, 1])*(p[:, 2, 0]-p[:, 0, 0]))
    denominator = np.sum((p-p[:, [1, 2, 0]])**2, axis=(1, 2))
    return 2*np.sqrt(3)*twice_area/np.maximum(denominator, 1e-300)


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _owners(elements):
    out = defaultdict(list)
    for eid, tri in enumerate(elements):
        for a, b in zip(tri, tri[[1, 2, 0]]):
            out[tuple(sorted((int(a), int(b))))].append(eid)
    return out


def _fingerprint(nodes, elements):
    digest = hashlib.sha256()
    for value in (nodes, elements): digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def production_geometry_constraints(state):
    """Freeze exact graph, cavity anchors, specimen/material boundaries."""
    nodes = np.asarray(state.mesh.nodes); fixed = set(); graph_nodes = set()
    for branch in state.crack_network.branches:
        for a, b in zip(branch.path, branch.path[1:]):
            a, b = np.asarray(a), np.asarray(b); delta = b-a; length = np.linalg.norm(delta)
            axial = (nodes-a)@delta/length
            normal = np.abs((nodes[:, 0]-a[0])*delta[1]-(nodes[:, 1]-a[1])*delta[0])/length
            graph_nodes.update(map(int, np.flatnonzero((normal <= 1e-12) & (axial >= -1e-12) & (axial <= length+1e-12))))
    fixed |= graph_nodes
    for cavity in state.void_state.cavities:
        for point in (cavity.connection_entry_m, cavity.connection_exit_m):
            if point is not None:
                distance = np.linalg.norm(nodes-np.asarray(point), axis=1)
                if distance.min() > 1e-12: raise ValueError('owned cavity anchor missing')
                fixed.add(int(np.argmin(distance)))
    protected = [edge for edge in _owners(state.mesh.elems) if set(edge).issubset(graph_nodes)]
    return fixed, protected


def fixed_path_constraints(mesh, path):
    """Retain every conforming node and edge on a fixed static crack path."""
    nodes = np.asarray(mesh.nodes); fixed = set()
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b); delta = b-a; length = np.linalg.norm(delta)
        axial = (nodes-a)@delta/length
        distance = np.abs((nodes[:,0]-a[0])*delta[1]-(nodes[:,1]-a[1])*delta[0])/length
        fixed.update(map(int, np.flatnonzero((distance <= 1e-12)&(axial >= -1e-12)&(axial <= length+1e-12))))
    return fixed, tuple(edge for edge in _owners(mesh.elems) if set(edge).issubset(fixed))


def constrained_quality_mesh(mesh, *, fixed_nodes, protected_edges=(), strategy='flips',
                             minimum_quality=.05, maximum_sweeps=12):
    """Use fixed-node flips, optionally followed by free-node optimization.

    Every accepted operation strictly improves the affected minimum quality,
    preserves triangle orientation, and respects constrained edges and P0
    material interfaces. A failed target remains a failed proposal; no global
    acceptance tolerance is weakened here.
    """
    if strategy not in ('flips', 'flips_and_free_node_optimization'):
        raise ValueError('unregistered quality strategy')
    nodes = np.array(mesh.nodes, copy=True); elements = np.array(mesh.elems, copy=True)
    original_nodes = nodes.copy(); original_elements = elements.copy()
    fixed = set(map(int, fixed_nodes)); protected = {tuple(sorted(map(int, e))) for e in protected_edges}
    if any(i < 0 or i >= len(nodes) for i in fixed): raise ValueError('fixed node out of range')
    values = getattr(mesh, 'element_damage_gp', None)
    material = np.zeros(len(elements)) if values is None else np.asarray(values).copy()
    initial_owners = _owners(elements)
    boundary = {e for e, ids in initial_owners.items() if len(ids) == 1}
    protected |= boundary
    protected |= {e for e, ids in initial_owners.items() if len(ids) == 2 and material[ids[0]] != material[ids[1]]}
    fixed |= {n for e in protected for n in e}
    if not protected.issubset(initial_owners): raise ValueError('protected edge absent from original mesh')
    original_signed = np.array([_cross(nodes[t[1]]-nodes[t[0]], nodes[t[2]]-nodes[t[0]]) for t in elements])
    if np.any(original_signed == 0): raise ValueError('degenerate input triangle')
    operations = []
    for sweep in range(maximum_sweeps):
        q = qualities(nodes, elements)
        if float(q.min()) >= minimum_quality: break
        owners = _owners(elements); touched = set(); progressed = False
        for edge, incident in sorted(owners.items()):
            if edge in protected or len(incident) != 2 or touched.intersection(incident): continue
            i, j = incident
            before = min(q[i], q[j])
            if before >= minimum_quality or material[i] != material[j]: continue
            a, b = edge
            c = next(int(n) for n in elements[i] if n not in edge)
            d = next(int(n) for n in elements[j] if n not in edge)
            if c == d or tuple(sorted((c, d))) in owners: continue
            if (_cross(nodes[b]-nodes[a], nodes[c]-nodes[a])*_cross(nodes[b]-nodes[a], nodes[d]-nodes[a]) >= 0
                or _cross(nodes[d]-nodes[c], nodes[a]-nodes[c])*_cross(nodes[d]-nodes[c], nodes[b]-nodes[c]) >= 0): continue
            replacements = np.array(((c, d, a), (d, c, b)))
            for k, eid in enumerate((i, j)):
                t = replacements[k]
                if _cross(nodes[t[1]]-nodes[t[0]], nodes[t[2]]-nodes[t[0]])*original_signed[eid] < 0:
                    replacements[k] = t[[0, 2, 1]]
            after = float(qualities(nodes, replacements).min())
            if after <= before: continue
            elements[[i, j]] = replacements
            touched.update(incident); progressed = True
            operations.append({'operation': 'legal_fixed_node_flip', 'sweep': sweep,
                'removed_edge': edge, 'added_edge': (c, d), 'element_ids': incident,
                'minimum_quality_before': float(before), 'minimum_quality_after': after})
        if strategy == 'flips_and_free_node_optimization':
            q = qualities(nodes, elements)
            bad_nodes = sorted(set(elements[q < minimum_quality].ravel())-fixed)
            adjacency = defaultdict(list)
            for eid, tri in enumerate(elements):
                for node in tri: adjacency[int(node)].append(eid)
            for node in bad_nodes:
                ids = np.array(adjacency[int(node)], dtype=int); patch = elements[ids]
                before = float(qualities(nodes, patch).min())
                if before >= minimum_quality: continue
                neighbors = sorted(set(patch.ravel())-{node}); origin = nodes[node].copy()
                scale = float(np.max(np.linalg.norm(nodes[neighbors]-origin, axis=1)))
                if scale <= 0: continue
                local_sign = original_signed[ids]
                def quality_at(offset):
                    coordinates = nodes[patch].copy()
                    coordinates[patch == node] = origin+scale*np.asarray(offset)
                    u, v = coordinates[:, 1]-coordinates[:, 0], coordinates[:, 2]-coordinates[:, 0]
                    area2 = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
                    if np.any(area2*local_sign <= 0): return -1.
                    return float(np.min(2*np.sqrt(3)*np.abs(area2)/np.sum((coordinates-coordinates[:, [1, 2, 0]])**2, axis=(1, 2))))
                centroid_offset = (nodes[neighbors].mean(axis=0)-origin)/scale
                candidates = [(quality_at(centroid_offset*2.**-k), centroid_offset*2.**-k) for k in range(8)]
                optimum = minimize(lambda z: -quality_at(z), np.zeros(2), method='Nelder-Mead',
                    options={'initial_simplex': np.array(((0., 0.), (.05, 0.), (0., .05))),
                             'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                candidates.append((quality_at(optimum.x), optimum.x))
                after, offset = max(candidates, key=lambda pair: (pair[0], -float(np.linalg.norm(pair[1]))))
                if after <= before: continue
                nodes[node] = origin+scale*offset; progressed = True
                operations.append({'operation': 'constrained_free_node_optimization', 'sweep': sweep,
                    'node_id': int(node), 'before_m': origin.tolist(), 'after_m': nodes[node].tolist(),
                    'minimum_quality_before': before, 'minimum_quality_after': after})
        if not progressed: break
    final_owners = _owners(elements)
    if not protected.issubset(final_owners): raise RuntimeError('quality proposal lost protected edge')
    if not np.array_equal(nodes[sorted(fixed)], original_nodes[sorted(fixed)]): raise RuntimeError('fixed geometry moved')
    if {e for e, ids in final_owners.items() if len(ids) == 1} != boundary: raise RuntimeError('physical boundary changed')
    if any(len(ids) not in (1, 2) for ids in final_owners.values()): raise RuntimeError('nonmanifold quality proposal')
    result = rebuild_tri_mesh(nodes, elements)
    if values is not None: result = replace(result, element_damage_gp=material)
    q = qualities(nodes, elements)
    return result, {'schema': SCHEMA, 'strategy': strategy, 'target_minimum_quality': minimum_quality,
        'minimum_quality_before': float(qualities(original_nodes, original_elements).min()),
        'minimum_quality_after': float(q.min()), 'bad_elements_remaining': np.flatnonzero(q < minimum_quality).tolist(),
        'geometry_protected_nodes': sorted(fixed), 'protected_edges': sorted(protected),
        'original_mesh_fingerprint': _fingerprint(original_nodes, original_elements),
        'proposed_mesh_fingerprint': _fingerprint(nodes, elements), 'operations': operations,
        'geometrical_quality_target_passed': bool(np.min(q) >= minimum_quality),
        'scientific_source_qualification': 'NOT_EVALUATED_REQUIRES_FULL_STATE_TRANSACTION'}
