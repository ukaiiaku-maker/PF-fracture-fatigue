import numpy as np
import pytest

from arrhenius_fracture.cavity_boundary_patch_recovery_v1 import (
    RecoveryUnavailable, polygon_arc_source, recover_boundary_tensor,
)


def annulus(segments=32, layers=12):
    angles = np.arange(segments)*2*np.pi/segments
    nodes = np.array([[r*np.cos(a), r*np.sin(a)] for r in 1+np.arange(4)/layers for a in angles])
    elements = []
    for j in range(3):
        for i in range(segments):
            a, b = j*segments+i, j*segments+(i+1)%segments
            c, d = a+segments, b+segments
            elements.extend(((a, c, d), (a, d, b)))
    edges = np.array([(i, (i+1)%segments) for i in range(segments)])
    return nodes, np.array(elements), edges


def field(nodes, elems, affine=True):
    xy = nodes[elems].mean(axis=1)
    result = np.broadcast_to(np.array([[3., .4], [.4, -2.]]), (len(elems), 2, 2)).copy()
    if affine:
        result += xy[:, 0, None, None]*np.array([[.7, -.2], [-.2, .5]])
        result += xy[:, 1, None, None]*np.array([[-.3, .9], [.9, .2]])
    return result


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("fraction", [0., .125, .17, .5])
def test_manufactured_reproduction_and_no_input_mutation(affine, fraction):
    nodes, elems, edges = annulus()
    stress = field(nodes, elems, affine)
    snapshots = [a.copy() for a in (nodes, elems, edges, stress)]
    point, normal = polygon_arc_source(nodes, edges, center=(0., 0.), arc_fraction=fraction)
    result = recover_boundary_tensor(nodes, elems, stress, boundary_edges=edges, source_point=point, normal=normal)
    exact = field(np.array([point]), np.array([[0, 0, 0]]), affine)[0]
    np.testing.assert_allclose(result["tensor_Pa"], exact, rtol=1e-11, atol=1e-12)
    for before, after in zip(snapshots, (nodes, elems, edges, stress)):
        assert np.array_equal(before, after)
    rebuilt = np.einsum("i,ijk->jk", result["evaluation_weights"], result["raw_CST_tensors_Pa"])
    np.testing.assert_allclose(rebuilt, result["tensor_Pa"], rtol=1e-14)


def test_rigid_rotation_tensor_and_mesh():
    nodes, elems, edges = annulus()
    stress = field(nodes, elems)
    point, normal = polygon_arc_source(nodes, edges, center=(0., 0.), arc_fraction=.125)
    a = recover_boundary_tensor(nodes, elems, stress, boundary_edges=edges, source_point=point, normal=normal)
    theta = .713
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = rotation@stress@rotation.T
    # Roundoff in independent off-diagonal products is not physical asymmetry.
    rotated = (rotated+rotated.transpose(0, 2, 1))*.5
    b = recover_boundary_tensor(nodes@rotation.T, elems, rotated, boundary_edges=edges,
                                source_point=rotation@point, normal=rotation@normal)
    assert a["element_ids"] == b["element_ids"]
    np.testing.assert_allclose(b["tensor_Pa"], rotation@np.array(a["tensor_Pa"])@rotation.T, rtol=1e-11, atol=1e-12)


def test_exact_repeat_and_reversed_edges():
    nodes, elems, edges = annulus()
    kwargs = dict(source_point=nodes[0], normal=np.array([1., 0.]), candidate_normals=((1., 0.), (0., 1.)))
    a = recover_boundary_tensor(nodes, elems, field(nodes, elems), boundary_edges=edges, **kwargs)
    b = recover_boundary_tensor(nodes, elems, field(nodes, elems), boundary_edges=edges[::-1, ::-1], **kwargs)
    assert a == b


def test_excluded_support_is_never_sampled():
    nodes, elems, edges = annulus()
    intact = np.ones(len(elems), dtype=bool)
    intact[0] = False
    result = recover_boundary_tensor(nodes, elems, field(nodes, elems), boundary_edges=edges,
                                     source_point=nodes[0], normal=(1., 0.), intact_mask=intact)
    assert 0 not in result["element_ids"]


@pytest.mark.parametrize("failure", ["missing_source", "nonfinite", "nonunit", "rank", "ownership"])
def test_fail_closed(failure):
    nodes, elems, edges = annulus()
    stress = field(nodes, elems)
    point, normal, intact = nodes[0].copy(), np.array([1., 0.]), np.ones(len(elems), dtype=bool)
    if failure == "missing_source": point = np.array([0., 0.])
    if failure == "nonfinite": stress[0, 0, 0] = np.nan
    if failure == "nonunit": normal *= 2
    if failure == "rank": intact[:] = False; intact[1] = True
    if failure == "ownership": edges = np.vstack((edges, elems[0, :2]))
    with pytest.raises(RecoveryUnavailable):
        recover_boundary_tensor(nodes, elems, stress, boundary_edges=edges, source_point=point, normal=normal, intact_mask=intact)


def test_ill_conditioned_stencil_is_not_source():
    nodes, elems, edges = annulus()
    nodes[:, 1] *= 1e-9
    with pytest.raises(RecoveryUnavailable, match="rank/conditioning"):
        recover_boundary_tensor(nodes, elems, field(nodes, elems), boundary_edges=edges,
                                source_point=nodes[0], normal=(1., 0.))


def test_exact_kirsch_displacement_reproduces_independent_stress():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    try:
        from qualify_cavity_patch_kirsch_fem_v1 import kirsch_displacement
        from qualify_cavity_boundary_patch_recovery_v1 import kirsch_tensor
        from arrhenius_fracture.config import ElasticProperties
        from arrhenius_fracture.fem import plane_strain_D
        points = np.array([[1.2, .3], [1.1, -.8], [-2.3, .7], [3., 1.]])
        h = 1e-5
        derivatives = [(kirsch_displacement(points+np.eye(2)[i]*h, 210e9, .3, 1e6)-
                        kirsch_displacement(points-np.eye(2)[i]*h, 210e9, .3, 1e6))/(2*h) for i in range(2)]
        strain = np.array([derivatives[0][:, 0], derivatives[1][:, 1],
                           derivatives[0][:, 1]+derivatives[1][:, 0]])
        calculated = plane_strain_D(ElasticProperties(E=210e9, nu=.3))@strain
        exact = 1e6*kirsch_tensor(points)
        expected = np.array([exact[:, 0, 0], exact[:, 1, 1], exact[:, 0, 1]])
        np.testing.assert_allclose(calculated, expected, rtol=1e-8, atol=.002)
    finally:
        sys.path.remove(str(scripts))


def test_kirsch_raw_source_reassembly_rejects_tampering():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    try:
        from qualify_cavity_patch_kirsch_fem_v1 import solve, validate_raw_source
        raw, row = solve(16, 6)
        assert row["source_validation"]["reassembled_arrays_exact"]
        raw["stress"] = raw["stress"].copy()
        raw["stress"][0, 0] += 1.
        with pytest.raises(ValueError, match="stress"):
            validate_raw_source(raw)
    finally:
        sys.path.remove(str(scripts))
