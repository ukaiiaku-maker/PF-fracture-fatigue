import numpy as np
import pytest

from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2 import (
    recover_piecewise_equilibrated_boundary_stress_v2,
)


def _annulus(radial_layers=5, sectors=48, rotation=0.0, reflected=False):
    theta = rotation + np.arange(sectors) * 2 * np.pi / sectors
    if reflected:
        theta = -theta
    radii = np.linspace(1.0, 1.5, radial_layers + 1)
    nodes = np.vstack([np.column_stack((r * np.cos(theta), r * np.sin(theta))) for r in radii])
    elements = []
    for layer in range(radial_layers):
        for index in range(sectors):
            following = (index + 1) % sectors
            inner, outer = layer * sectors, (layer + 1) * sectors
            for original in ((inner + index, outer + index, outer + following),
                             (inner + index, outer + following, inner + following)):
                triangle = list(original)
                points = nodes[triangle]
                first, second = points[1] - points[0], points[2] - points[0]
                if first[0] * second[1] - first[1] * second[0] < 0:
                    triangle[1], triangle[2] = triangle[2], triangle[1]
                elements.append(triangle)
    edges = [(index, (index + 1) % sectors) for index in range(sectors)]
    source = 0
    return nodes, np.asarray(elements), edges, source


def _constant(points):
    result = np.zeros((len(points), 2, 2))
    result[:, 0, 0], result[:, 1, 1] = 3., 7.
    result[:, 0, 1] = result[:, 1, 0] = 2.
    return result


def _affine(points):
    x, y = points.T
    result = np.zeros((len(points), 2, 2))
    result[:, 0, 0] = 3. + 2. * x + .5 * y
    result[:, 1, 1] = 7. - 1.5 * x + 4. * y
    result[:, 0, 1] = result[:, 1, 0] = 2. - 4. * x - 2. * y
    return result


def _kirsch(points):
    x, y = points.T
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    rr = 5 * (1 - r**-2) + 5 * (1 - 4*r**-2 + 3*r**-4) * c2
    tt = 5 * (1 + r**-2) - 5 * (1 + 3*r**-4) * c2
    rt = -5 * (1 + 2*r**-2 - 3*r**-4) * s2
    er = np.column_stack((np.cos(theta), np.sin(theta)))
    et = np.column_stack((-np.sin(theta), np.cos(theta)))
    return (rr[:, None, None] * er[:, :, None] * er[:, None, :]
            + tt[:, None, None] * et[:, :, None] * et[:, None, :]
            + rt[:, None, None] * (er[:, :, None] * et[:, None, :]
                                    + et[:, :, None] * er[:, None, :]))


def _rotated_kirsch(angle):
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)),
                           (np.sin(angle), np.cos(angle))))
    def field(points):
        return np.einsum("ia,nab,jb->nij", rotation, _kirsch(points @ rotation), rotation)
    return field


def _run(field, **kwargs):
    nodes, elements, edges, source = _annulus(**kwargs)
    return recover_piecewise_equilibrated_boundary_stress_v2(
        nodes=nodes, elements=elements, stress_Pa=field(nodes[elements].mean(axis=1)),
        boundary_node=source, cavity_center_m=(0., 0.), cavity_radius_m=1.,
        process_length_m=1., owned_boundary_edges=edges,
        remote_traction_scale_Pa=10., state_fingerprint="qualification",
    )


@pytest.mark.parametrize("field", (_constant, _affine))
def test_manufactured_equilibrated_fields_satisfy_piecewise_constraints(field):
    result = _run(field, radial_layers=5, sectors=48)
    assert result["zero_boundary_traction_imposed"] is False
    assert result["stress_fit_residual"] <= result["stress_fit_residual_limit"]
    assert result["equilibrium_residual"] <= result["constraint_limits"]["equilibrium"]
    assert result["interface_traction_jump_residual"] <= result["constraint_limits"]["interface_traction_jump"]
    assert result["resultant_residual"] <= result["constraint_limits"]["resultant"]
    assert result["moment_residual"] <= result["constraint_limits"]["moment"]


def test_kirsch_refinement_rotation_reflection_and_edge_order_qualify():
    coarse = _run(_kirsch, radial_layers=5, sectors=48)
    fine = _run(_kirsch, radial_layers=8, sectors=96)
    rotated = _run(_rotated_kirsch(np.pi / 7), radial_layers=8, sectors=96, rotation=np.pi / 7)
    reflected = _run(_kirsch, radial_layers=8, sectors=96, reflected=True)
    assert coarse["normalized_boundary_traction"] <= 0.05
    assert fine["normalized_boundary_traction"] <= 0.05
    assert abs(fine["normalized_boundary_traction"] - coarse["normalized_boundary_traction"]) <= 0.05
    assert rotated["normalized_boundary_traction"] == pytest.approx(
        fine["normalized_boundary_traction"], abs=1e-10,
    )
    assert reflected["normalized_boundary_traction"] <= 0.05

    nodes, elements, edges, source = _annulus(radial_layers=8, sectors=96)
    kwargs = dict(nodes=nodes, elements=elements,
                  stress_Pa=_kirsch(nodes[elements].mean(axis=1)), boundary_node=source,
                  cavity_center_m=(0., 0.), cavity_radius_m=1., process_length_m=1.,
                  remote_traction_scale_Pa=10., state_fingerprint="orientation")
    reverse = recover_piecewise_equilibrated_boundary_stress_v2(
        owned_boundary_edges=[edge[::-1] for edge in edges[::-1]], **kwargs,
    )
    assert reverse["normalized_boundary_traction"] == pytest.approx(
        fine["normalized_boundary_traction"], abs=1e-13,
    )
    assert reverse["window_identity"] == fine["window_identity"]
