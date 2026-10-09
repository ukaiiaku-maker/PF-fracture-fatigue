#!/usr/bin/env python3
"""Development-only geometry proposals; no source PASS or event authority."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.quality_constrained_mesh_v1 import constrained_quality_mesh, _owners


def geometry_constraints(state):
    nodes = np.asarray(state.mesh.nodes); fixed = set()
    graph_nodes = set()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite proposal evidence')
    manifest = json.loads((args.diagnostic/'sha256_manifest.json').read_text())
    actual = {str(p.relative_to(args.diagnostic)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.diagnostic.rglob('*') if p.is_file() and p.name != 'sha256_manifest.json'}
    assert manifest == actual, 'diagnostic manifest mismatch'
    state = restore_checkpoint(args.diagnostic/'rejected_refined_trial.json')
    fixed, edges = geometry_constraints(state); args.output.mkdir(parents=True)
    for strategy in ('flips', 'flips_and_free_node_optimization'):
        print('Executing materially distinct quality proposal: '+strategy, flush=True)
        mesh, report = constrained_quality_mesh(state.mesh, fixed_nodes=fixed, protected_edges=edges, strategy=strategy)
        report['evidence_classification'] = 'DEVELOPMENT_GEOMETRY_PROPOSAL_NOT_SCIENTIFIC_QUALIFICATION'
        report['executed_head_sha'] = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=ROOT, text=True).strip()
        report['executed_code_file_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (Path(__file__).resolve(), ROOT/'arrhenius_fracture/quality_constrained_mesh_v1.py')}
        np.savez_compressed(args.output/(strategy+'.npz'), nodes=mesh.nodes, elements=mesh.elems, element_damage_gp=mesh.element_damage_gp)
        (args.output/(strategy+'.json')).write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('strategy', 'minimum_quality_before', 'minimum_quality_after', 'bad_elements_remaining')}), flush=True)
    (args.output/'sha256_manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.is_file()}, sort_keys=True, indent=2)+'\n')


if __name__ == '__main__': main()
