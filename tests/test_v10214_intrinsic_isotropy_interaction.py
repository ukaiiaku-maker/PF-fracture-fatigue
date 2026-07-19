import numpy as np
import pytest

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.crystal import cubic_plane_strain_D
from arrhenius_fracture.interaction_integral_v10214 import (
    fit_intrinsic_isotropic_plane_strain,
)


def test_exact_cubic_tungsten_is_accepted_despite_rounded_E_nu_metadata():
    mat = ElasticProperties(E=410.0e9, nu=0.28)
    D = cubic_plane_strain_D(
        C11=523.0e9,
        C12=203.0e9,
        C44=160.0e9,
        theta_deg=45.0,
    )
    fit = fit_intrinsic_isotropic_plane_strain(D, mat)
    assert fit.maximum_relative_residual < 1.0e-12
    assert fit.metadata_D_maximum_relative_difference == pytest.approx(
        2.190e-3, rel=2.0e-3
    )
    assert fit.lame_lambda_Pa == pytest.approx(203.0e9, rel=1.0e-12)
    assert fit.lame_mu_Pa == pytest.approx(160.0e9, rel=1.0e-12)
    assert fit.E_Pa == pytest.approx(409.476584022e9, rel=1.0e-10)
    assert fit.poisson == pytest.approx(0.279614325069, rel=1.0e-10)


def test_rotated_intrinsically_isotropic_cubic_matrix_is_orientation_invariant():
    mat = ElasticProperties()
    fits = []
    for theta in (0.0, 17.0, 45.0, 83.0):
        D = cubic_plane_strain_D(
            C11=523.0e9,
            C12=203.0e9,
            C44=160.0e9,
            theta_deg=theta,
        )
        fits.append(fit_intrinsic_isotropic_plane_strain(D, mat))
    assert max(f.maximum_relative_residual for f in fits) < 1.0e-12
    assert np.ptp([f.E_Pa for f in fits]) / fits[0].E_Pa < 1.0e-12
    assert np.ptp([f.poisson for f in fits]) < 1.0e-12


def test_true_cubic_anisotropy_remains_fail_closed():
    mat = ElasticProperties()
    D = cubic_plane_strain_D(
        C11=523.0e9,
        C12=203.0e9,
        C44=176.0e9,
        theta_deg=45.0,
    )
    with pytest.raises(ValueError, match="materially anisotropic"):
        fit_intrinsic_isotropic_plane_strain(D, mat)


def test_shear_normal_coupling_remains_fail_closed():
    mat = ElasticProperties()
    D = cubic_plane_strain_D(theta_deg=0.0)
    D = D.copy()
    D[0, 2] = D[2, 0] = 1.0e9
    with pytest.raises(ValueError, match="materially anisotropic"):
        fit_intrinsic_isotropic_plane_strain(D, mat)
