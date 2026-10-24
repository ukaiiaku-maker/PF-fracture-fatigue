import numpy as np
import pytest

from experimental_fits.ast2018.objective import (
    _quadrature_weights, robust_quadrature_residual,
)


def test_robust_curve_cost_does_not_depend_on_raster_density():
    expected=np.sqrt(1+3**2)-1
    for n in (3,213,629):
        q=_quadrature_weights(np.linspace(0,1000,n))
        r=robust_quadrature_residual(np.full(n,3.),q,1.)
        assert .5*np.sum(r*r)==pytest.approx(expected)


def test_family_weights_normalize_per_condition():
    q=_quadrature_weights(np.linspace(0,1,25))
    r=robust_quadrature_residual(np.full(25,2.),q,.25)
    assert 4*(.5*np.sum(r*r))==pytest.approx(np.sqrt(5)-1)
