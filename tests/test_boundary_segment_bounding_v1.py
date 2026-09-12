import numpy as np
import pytest
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import _segments_intersect


@pytest.mark.parametrize('reverse_a', [False, True])
@pytest.mark.parametrize('reverse_b', [False, True])
def test_refined_external_edge_outside_root_certificate_tube_is_not_intersection(reverse_a, reverse_b):
    edge = [np.array((0., 1.0738404582121311e-6)), np.array((0., 5.369202291061085e-7))]
    tube = [np.array((1.2813516934366172e-6, -1.922027540154926e-6)),
            np.array((1.2813516934366172e-6, 1.922027540154926e-6))]
    if reverse_a: edge.reverse()
    if reverse_b: tube.reverse()
    assert not _segments_intersect(*edge, *tube)


@pytest.mark.parametrize('scale', [1., 1e-3, 1e-6])
def test_true_crossing_and_exact_contact_remain_intersections(scale):
    def p(x, y): return scale*np.array((x, y))
    assert _segments_intersect(p(-1, 0), p(1, 0), p(0, -1), p(0, 1))
    assert _segments_intersect(p(-1, 0), p(0, 0), p(0, 0), p(0, 1))
    assert not _segments_intersect(p(-1, 0), p(1, 0), p(0, 2), p(0, 3))
