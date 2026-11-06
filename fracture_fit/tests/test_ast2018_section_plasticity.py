"""Full-section bulk plasticity: beam statics and native-rate bookkeeping."""
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from experimental_fits.ast2018.mechanics import Specimen
from experimental_fits.ast2018.section_plasticity import (
    SectionPlasticity, SectionPlasticityConfig, SectionSaturation)

KB = 8.617333262e-5


class LinearSurface:
    def __init__(self, G0, V_eV_per_Pa):
        self.G0, self.V = G0, V_eV_per_Pa

    def barrier_eV(self, stress, T):
        return np.maximum(self.G0 - self.V * np.asarray(stress, float), 0.0)


class NativeRates:
    """Same signatures and forward-minus-reverse form as the native class."""

    @staticmethod
    def _arrhenius_rate(barrier, T, nu0):
        return nu0 * np.exp(np.clip(-np.asarray(barrier, float) / (KB * T), -700, 0))

    @classmethod
    def _signed_rate(cls, surface, stress, T, nu0):
        stress = np.asarray(stress, float)
        return (cls._arrhenius_rate(surface.barrier_eV(np.maximum(stress, 0), T), T, nu0) -
                cls._arrhenius_rate(surface.barrier_eV(np.maximum(-stress, 0), T), T, nu0))


def _mechanism(G0, V, nu0):
    surface = LinearSurface(G0, V)
    return SimpleNamespace(surface=lambda parent: surface, stress_fraction=1 / math.sqrt(3),
                           nu0_s=nu0)


def _fixture(peierls_G0=0.9, emission_G0=0.9):
    candidate = SimpleNamespace(
        emission=LinearSurface(emission_G0, 1e-10),
        peierls=_mechanism(peierls_G0, 4e-10, 1e12),
        taylor=_mechanism(0.5, 1e-12, 1e11),
        rho_source0_m2=1e14, taylor_corr_scale=1.0, taylor_corr_rho_c_m2=1e16)
    physics = SimpleNamespace(
        n_systems=2, emission_schmid_factors=(1 / math.sqrt(3),) * 2,
        activation_to_line_content_per_system=(1.0, 1.0),
        persistent_backstress_scale=1.0, G_Pa=160e9, b_m=2.74e-10,
        forest_interaction_matrix=((1.0, 1.0), (1.0, 1.0)), rho_forest_floor_m2=5e12,
        jump_fraction_of_forest_spacing=1.0, mobile_transport_velocity_scale=1.0,
        taylor_phi_max=float("inf"), mean_free_path_coefficient=1.0,
        encounter_efficiency=1.0, emission_nu0_s=1e11)
    specimen = Specimen(4e-6, 6e-6, 12.5e-6, 2.25e-6, 411e9, .28, 1.6e-4)
    return candidate, physics, specimen


def _section(nucleation=True, **kw):
    candidate, physics, specimen = _fixture(**kw)
    config = (SectionPlasticityConfig(mode="full_section", bulk_nucleation=True,
                                      initial_mobile="zero") if nucleation
              else SectionPlasticityConfig(mode="full_section"))
    return SectionPlasticity(candidate, physics, specimen, config, NativeRates)


def test_default_bulk_has_forest_floor_carriers_and_no_nucleation():
    config = SectionPlasticityConfig(mode="full_section")
    assert not config.bulk_nucleation and config.initial_mobile == "forest_floor"
    with pytest.raises(ValueError):
        SectionPlasticityConfig(mode="full_section", initial_mobile="zero")
    section = _section(nucleation=False)
    assert np.all(section.mobile_m2 == 5e12)
    before = float(np.sum(section.mobile_m2 + section.retained_m2))
    du = section.advance(1.0, 1.5e-3, 1.5e-3, 2.25e-6, 294.15)
    assert du > 0 and section.born_line_m == 0.0
    assert np.isclose(float(np.sum(section.mobile_m2 + section.retained_m2)), before,
                      rtol=1e-12)


def test_clone_isolates_evolving_fields():
    section = _section(nucleation=False)
    trial = section.clone()
    trial.advance(1.0, 1.5e-3, 1.5e-3, 2.25e-6, 294.15)
    assert np.all(section.eps_p == 0) and np.any(trial.eps_p != 0)
    assert section.upl_m == 0 and trial.upl_m > 0


def test_tip_only_is_the_default_and_cannot_build_a_section_state():
    from experimental_fits.ast2018.spatial import predict_spatial
    assert inspect.signature(predict_spatial).parameters["section"].default is None
    assert not SectionPlasticityConfig().active
    candidate, physics, specimen = _fixture()
    with pytest.raises(ValueError):
        SectionPlasticity(candidate, physics, specimen, SectionPlasticityConfig(), NativeRates)


def test_crack_cuts_only_the_hinge_station():
    section = _section()
    w = section.weights(2.25e-6)
    hinge = section.hinge
    assert np.all(w[~hinge] == 1.0)
    uncracked = np.sum(w[hinge]) * section.dy
    assert np.isclose(uncracked, 6e-6 - 2.25e-6, rtol=0, atol=1e-15)
    assert np.isclose(section.hinge_length_m, 3.75e-6)


def test_elastic_bending_stress_and_self_equilibrated_eigenstrain_residual():
    section = _section()
    a = 2.25e-6
    w, area, arm, inertia = section.section(a)
    h = int(np.flatnonzero(section.hinge)[0])
    F = 2e-3
    sigma = section.stress(F, a)
    ligament = 6e-6 - a
    # Outer tension fibre of the ligament against M c / I for a B x b section.
    active = np.flatnonzero(w[h] > 0)
    expected = F * section.s[h] * arm[h, active] / (section.B * inertia[h])
    assert np.allclose(sigma[h, active], expected)
    assert np.isclose(np.max(sigma[h]), 6 * F * section.s[h] / (section.B * ligament ** 2),
                      rtol=.05)
    rng = np.random.default_rng(3)
    section.eps_p = rng.normal(scale=1e-3, size=section.eps_p.shape)
    residual = section.stress(0.0, a)
    assert np.allclose(np.sum(w * residual, axis=1) * section.dy, 0, atol=1e-3)
    assert np.allclose(np.sum(w * residual * arm, axis=1) * section.dy, 0, atol=1e-9)


def test_plastic_displacement_is_hinge_rotation_times_lever_arm():
    section = _section()
    a = 2.25e-6
    w, _, arm, _ = section.section(a)
    h = int(np.flatnonzero(section.hinge)[0])
    kappa = 1e3
    deps = np.zeros_like(section.eps_p)
    deps[h] = kappa * arm[h] * (w[h] > 0)
    du = section.plastic_displacement_increment(deps, a)
    assert np.isclose(du, kappa * section.hinge_length_m * section.s[h], rtol=1e-12)
    # A uniform axial eigenstrain neither rotates the beam nor stresses it.
    uniform = np.full_like(section.eps_p, 2e-3)
    assert abs(section.plastic_displacement_increment(uniform, a)) < 1e-18
    section.eps_p = uniform
    assert np.allclose(section.stress(0.0, a), 0.0, atol=1e-3)


def test_zero_load_gives_zero_glide_and_conserves_line_content():
    section = _section()
    du = section.advance(10.0, 0.0, 0.0, 2.25e-6, 294.15)
    assert du == 0.0
    assert np.all(section.slip == 0.0)
    section = _section(emission_G0=0.8)
    section.advance(1.0, 1e-3, 1e-3, 2.25e-6, 294.15)
    volume = section.B * section.dy * section.ds[None, :, None]
    stored = float(np.sum((section.mobile_m2 + section.retained_m2) * volume))
    assert section.born_line_m > 0
    assert np.isclose(stored, section.born_line_m, rtol=1e-10)
    peak = float(np.max(section.mobile_m2 + section.retained_m2))
    assert section.max_line_balance_residual <= 1e-12 * peak


def test_fast_capture_and_release_conserve_line_content():
    # Regression: the expanded BE determinant cancelled at large h*rate.
    section = _section()
    section.mobile_m2[:] = 3e13
    section.retained_m2[:] = 1e13
    born = np.zeros_like(section.mobile_m2)
    e = np.full_like(born, 1e9)
    r = np.full_like(born, 2e9)
    section._rates = lambda sigma, T: (np.zeros_like(born), e, r, np.zeros_like(born),
                                       section.mobile_m2 + section.retained_m2)
    _, _, mobile, retained, _, _ = section._substep(10.0, 0.0, 2.25e-6, 294.15)
    assert np.allclose(mobile + retained, 4e13, rtol=1e-14)
    assert np.allclose(mobile / (mobile + retained), 2 / 3, rtol=1e-6)


def test_bulk_flow_is_monotone_in_force_and_runs_away_above_its_flow_stress():
    responses = []
    for F in (0.5e-3, 1.0e-3, 1.5e-3):
        section = _section(emission_G0=0.85)
        responses.append(section.advance(1.0, F, F, 2.25e-6, 294.15))
    assert 0 < responses[0] < responses[1] < responses[2]
    # Strongly nonlinear Arrhenius flow: a 3x force raises du by >100x.
    assert responses[2] > 100 * responses[0]
    section = _section(emission_G0=0.85)
    with pytest.raises(SectionSaturation):
        section.advance(1.0, 6e-3, 6e-3, 2.25e-6, 294.15, du_limit_m=1e-7)
