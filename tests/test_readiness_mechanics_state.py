import numpy as np
import pytest

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import plane_strain_D, plane_stress_D, transverse_stress
from arrhenius_fracture.mechanics_state_contract import (
    ConstraintControl, InPlaneMode, MechanicsStateOptions, PlaneState,
    constrained_elastic_potential_density, elastic_work_density,
    full_tensor_from_in_plane, resolved_traction, rotate_tensor,
)
from arrhenius_fracture.readiness_options import ReadinessOptions
from arrhenius_fracture.shared_options import SharedOptions


def test_v2_wrapper_preserves_v1_identity_and_is_fail_closed():
    base = SharedOptions()
    before = base.fingerprint
    opts = ReadinessOptions(base, MechanicsStateOptions())
    assert opts.base.fingerprint == before
    assert opts.to_dict()["production_execution_authorized"] is False
    with pytest.raises(ValueError, match="cannot authorize production"):
        ReadinessOptions(base, MechanicsStateOptions(), True)


def test_plane_stress_and_plane_strain_have_actual_transverse_conditions():
    mat = ElasticProperties(E=200e9, nu=0.3)
    strain = np.array([1.2e-3, -0.2e-3, 0.4e-3])
    sig_pe = strain @ plane_strain_D(mat).T
    sig_ps = strain @ plane_stress_D(mat).T
    assert transverse_stress(sig_ps[None, :], mat, "plane_stress")[0] == 0.0
    assert transverse_stress(sig_pe[None, :], mat, "plane_strain")[0] == pytest.approx(
        mat.nu * (sig_pe[0] + sig_pe[1]))
    assert not np.allclose(sig_pe, sig_ps)


def test_generalized_constraint_rotation_resolution_and_work_conjugacy():
    options = MechanicsStateOptions(
        plane_state=PlaneState.GENERALIZED_PLANE_STRAIN,
        constraint_control=ConstraintControl.PRESCRIBED_SIGMA_ZZ,
        eps_zz=None, sigma_zz_Pa=75e6,
        in_plane_mode=InPlaneMode.MIXED_I_II, mode_I_fraction=0.6,
    )
    eps, sig = full_tensor_from_in_plane(1e-3, -2e-4, 3e-4, 210e9, 0.29, options)
    assert sig[2, 2] == pytest.approx(75e6)
    angle = np.deg2rad(37.0)
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    sig_r, eps_r = rotate_tensor(sig, Q), rotate_tensor(eps, Q)
    assert elastic_work_density(sig_r, eps_r) == pytest.approx(elastic_work_density(sig, eps))
    normal, shear = resolved_traction(sig, np.array([1.0, 0.0, 0.0]),
                                      np.array([0.0, 1.0, 0.0]))
    assert normal == pytest.approx(sig[0, 0])
    assert shear == pytest.approx(sig[1, 0])


def test_constraint_contract_rejects_fake_or_ambiguous_states():
    with pytest.raises(ValueError):
        MechanicsStateOptions(plane_state="plane_stress")
    with pytest.raises(ValueError):
        MechanicsStateOptions(plane_state="generalized_plane_strain", eps_zz=0.0,
                              sigma_zz_Pa=0.0)
    with pytest.raises(ValueError):
        MechanicsStateOptions(in_plane_mode="mixed_I_II", mode_I_fraction=1.0)


@pytest.mark.parametrize("options", [
    MechanicsStateOptions(),
    MechanicsStateOptions(
        plane_state=PlaneState.PLANE_STRESS,
        constraint_control=ConstraintControl.FREE_TRANSVERSE,
        eps_zz=None,
    ),
    MechanicsStateOptions(
        plane_state=PlaneState.GENERALIZED_PLANE_STRAIN,
        constraint_control=ConstraintControl.PRESCRIBED_EPS_ZZ,
        eps_zz=2.5e-4,
    ),
    MechanicsStateOptions(
        plane_state=PlaneState.GENERALIZED_PLANE_STRAIN,
        constraint_control=ConstraintControl.PRESCRIBED_SIGMA_ZZ,
        eps_zz=None,
        sigma_zz_Pa=75e6,
    ),
])
def test_reduced_potential_derivative_is_in_plane_stress(options):
    E, nu = 210e9, 0.29
    q = np.array([1.2e-3, -2e-4, 4e-4])

    def potential(values):
        eps, sig = full_tensor_from_in_plane(*values, E, nu, options)
        return constrained_elastic_potential_density(sig, eps, options)

    eps, sig = full_tensor_from_in_plane(*q, E, nu, options)
    h = 1e-8
    derivative = np.array([
        (potential(q + np.eye(3)[i] * h)
         - potential(q - np.eye(3)[i] * h)) / (2.0 * h)
        for i in range(3)
    ])
    np.testing.assert_allclose(derivative, [sig[0, 0], sig[1, 1], sig[0, 1]],
                               rtol=2e-10, atol=20.0)
    if options.sigma_zz_Pa is not None:
        assert sig[2, 2] == pytest.approx(options.sigma_zz_Pa)
        assert eps[2, 2] != 0.0


def test_compact_spatial_patch_qualification_passes():
    from scripts.qualify_mechanics_constraint_patch import qualify
    report = qualify()
    assert report["status"] == "PASS"
    assert set(report["cases"]) == {
        "plane_strain", "plane_stress", "generalized_prescribed_eps_zz",
        "generalized_prescribed_sigma_zz",
    }
