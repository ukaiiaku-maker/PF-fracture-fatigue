import numpy as np
import pytest

from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.quality_constrained_mesh_v1 import constrained_quality_mesh, qualities


def poor_fan():
    return rebuild_tri_mesh(np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.5, .001))),
                            np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))))


def test_fixed_boundary_free_node_strategy_is_distinct_and_preserves_input():
    mesh = poor_fan(); original = mesh.nodes.copy()
    fixed, a = constrained_quality_mesh(mesh, fixed_nodes=range(5), protected_edges=((0, 4), (1, 4)), strategy='flips')
    improved, b = constrained_quality_mesh(mesh, fixed_nodes=range(4), strategy='flips_and_free_node_optimization')
    assert not a['geometrical_quality_target_passed']
    assert b['geometrical_quality_target_passed']
    assert np.array_equal(mesh.nodes, original)
    assert np.array_equal(improved.nodes[:4], original[:4])
    assert np.min(qualities(improved.nodes, improved.elems)) >= .05


def test_repeated_proposal_is_byte_deterministic():
    a, aa = constrained_quality_mesh(poor_fan(), fixed_nodes=range(4), strategy='flips_and_free_node_optimization')
    b, bb = constrained_quality_mesh(poor_fan(), fixed_nodes=range(4), strategy='flips_and_free_node_optimization')
    assert aa == bb and np.array_equal(a.nodes, b.nodes) and np.array_equal(a.elems, b.elems)
    assert all(r['minimum_quality_after'] > r['minimum_quality_before'] for r in aa['operations'])


def test_missing_protected_edge_fails_closed():
    with pytest.raises(ValueError, match='protected edge absent'):
        constrained_quality_mesh(poor_fan(), fixed_nodes=range(4), protected_edges=((0, 2),))


def test_good_mesh_is_unchanged_and_never_self_certifies_source():
    mesh = rebuild_tri_mesh(np.array(((0., 0.), (1., 0.), (.5, 1.))), np.array(((0, 1, 2),)))
    result, audit = constrained_quality_mesh(mesh, fixed_nodes=range(3))
    assert np.array_equal(result.nodes, mesh.nodes) and np.array_equal(result.elems, mesh.elems)
    assert audit['operations'] == []
    assert audit['scientific_source_qualification'].startswith('NOT_EVALUATED')
