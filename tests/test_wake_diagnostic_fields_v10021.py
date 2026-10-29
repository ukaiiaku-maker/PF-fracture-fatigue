from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front import _build_parser
from arrhenius_fracture.unified_front import UnifiedMPZFrontEngine
from arrhenius_fracture.sharp_front_v10_1 import _diagnostics_with_csv_aliases
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState
from arrhenius_fracture.incremental_opening_advection import (
    density_profile_in_active_zone,
    incoming_material_profile,
    operator_from_states,
)
from arrhenius_fracture.signed_gnd_projection import (
    signed_gnd_density, signed_profiles, swept_tip_content,
)
import scripts.qste340tm_signed_bulk_1d as signed_bulk_1d
from scripts.qste340tm_faithful_reduction import PlasticParameters
from scripts.qste340tm_emission_blunting_reduction import BluntingParameters


def test_persistent_wake_diagnostics_and_csv_aliases():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=100e-6, n_bins=200, wake_length_m=100e-6, wake_shielding=True),
    )
    state.mobile[:, :2] = 1.0
    state.retained[:, :2] = 2.0
    state.advance(5e-6)
    d = _diagnostics_with_csv_aliases(state, 160e9, 0.28, 2.74e-10, 1e-6)
    assert d["mpz_wake_retained_count"] > 0.0
    assert d["mpz_wake_retained_total"] == d["mpz_wake_retained_count"]
    assert d["mpz_K_shield_Pa_sqrt_m"] == d["mpz_total_K_shield_Pa_sqrt_m"]
    assert d["mpz_local_slip_count"] >= 0.0


def test_forward_passage_blunting_is_opt_in_and_convects_with_tip():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    off = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=1e-6, n_bins=20, blunting_length_m=5e-6,
                  forward_slip_blunting_coefficient=0.0),
    )
    on = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=1e-6, n_bins=20, blunting_length_m=5e-6,
                  forward_slip_blunting_coefficient=3.0),
    )
    for state in (off, on):
        state.register_forward_passage(10.0)
    r0 = 1e-9
    b = 2.5e-10
    assert off.blunted_radius(r0, b) == pytest.approx(r0)
    before = on.blunted_radius(r0, b)
    assert before == pytest.approx(r0 + 3.0 * b * 10.0)
    survival = on.convect_forward_blunting(5e-6)
    assert survival == pytest.approx(1.0 / math.e)
    assert on.forward_passage_slip_count == pytest.approx(10.0 / math.e)
    assert r0 < on.blunted_radius(r0, b) < before


def test_forward_passage_ledger_splits_without_duplication():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(manifest, MPZConfig())
    state.register_forward_passage(12.0)
    child = state.split(0.25)
    assert child.forward_passage_slip_count == pytest.approx(3.0)
    assert state.forward_passage_slip_count == pytest.approx(9.0)


def test_analytical_source_ledger_matches_exponential_convection():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest,
        MPZConfig(blunting_length_m=5e-6, analytical_source_slip_ledger=True),
    )
    state.register_source_passage(20.0)
    r0 = 1e-9
    b = 2.5e-10
    before = state.blunted_radius(r0, b)
    state.convect_forward_blunting(5e-6)
    assert state.source_passage_slip_count == pytest.approx(20.0 / math.e)
    expected = r0 + manifest.c_blunt * b * 20.0 / math.e
    assert state.blunted_radius(r0, b) == pytest.approx(expected)
    assert state.blunted_radius(r0, b) < before


def test_passage_ledgers_can_use_analytical_source_normalization():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest,
        MPZConfig(
            analytical_source_slip_ledger=True,
            passage_count_normalization=200.0,
            forward_slip_blunting_coefficient=2.0,
        ),
    )
    state.register_source_passage(100.0)
    state.register_forward_passage(50.0)
    assert state.source_passage_slip_count == pytest.approx(0.5)
    assert state.forward_passage_slip_count == pytest.approx(0.25)
    diagnostic = state.diagnostics(160e9, 0.28, 2.5e-10, 1e-9)
    assert diagnostic["mpz_passage_count_normalization"] == pytest.approx(200.0)


def test_fixed_local_kmax_loading_is_separate_and_opt_in():
    parser = _build_parser()
    historical = parser.parse_args([])
    assert historical.fatigue_loading_control == "fixed_displacement"
    assert historical.mpz_passage_count_normalization == pytest.approx(1.0)
    assert historical.signed_gnd_tip_absorption is False
    controlled = parser.parse_args([
        "--fatigue-loading-control", "fixed_local_kmax",
        "--fatigue-target-Kmax-MPa-sqrt-m", "42.857142857",
        "--mpz-passage-count-normalization", "282.1181135",
    ])
    assert controlled.fatigue_loading_control == "fixed_local_kmax"
    assert controlled.fatigue_target_Kmax_MPa_sqrt_m == pytest.approx(42.857142857)
    assert controlled.mpz_passage_count_normalization == pytest.approx(282.1181135)


def test_signed_gnd_swept_tip_content_preserves_sign_and_face_ownership():
    triangles = np.array([
        [[0.0, -1.0], [1.0, -1.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ])
    directions = np.array([
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
    ])
    rho = np.array([[-2.0, 2.0], [0.0, 0.0]])
    result = swept_tip_content(
        triangles=triangles, rho_signed_m2=rho, directions=directions,
        old_tip_x_m=0.0, new_tip_x_m=1.0, capture_width_m=2.0,
    )
    assert result["signed_count_by_system_face"][0].tolist() == pytest.approx([-1.0, 1.0])
    assert result["absolute_count_total"] == pytest.approx(2.0)
    assert result["opening_effective_count"] == pytest.approx(math.sqrt(2.0))
    assert result["scope"] == "crack_swept_signed_GND_not_continuous_mobile_flux"


def test_bulk_tip_absorption_blunts_through_existing_coefficient_and_convects():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest, MPZConfig(blunting_length_m=5e-6, n_systems=2)
    )
    signed = np.array([[-1.0, 1.0], [0.0, 0.0]])
    opening = np.array([[0.5, 0.5], [0.0, 0.0]])
    committed = state.register_bulk_tip_absorption(signed, opening)
    r0, b = 1e-9, 2.5e-10
    assert committed == pytest.approx(1.0)
    assert state.blunted_radius(r0, b) == pytest.approx(
        r0 + manifest.c_blunt * b
    )
    state.convect_forward_blunting(5e-6)
    assert state.bulk_tip_opening_slip_count == pytest.approx(1.0 / math.e)
    diagnostic = state.diagnostics(160e9, 0.28, b, r0)
    assert diagnostic["mpz_bulk_tip_absorbed_absolute_total"] == pytest.approx(2.0)
    assert diagnostic["mpz_bulk_tip_absorbed_opening_total"] == pytest.approx(1.0)


def test_unified_front_declares_moving_state_ownership():
    assert UnifiedMPZFrontEngine.unified_mpz_active is True
    assert UnifiedMPZFrontEngine.state_model == "moving_pz"


def test_incremental_advection_projection_is_area_conservative():
    x = np.array([0.25, 0.75, 1.25, 1.75]) * 1e-6
    y = np.zeros(4)
    area = np.full(4, 0.5e-12)
    rho = np.full(4, 2.0e12)
    profile = density_profile_in_active_zone(
        centers_x=x, centers_y=y, areas=area, density_m2=rho,
        tip_x_m=0.0, length_m=2e-6, n_bins=4, capture_width_m=1e-6,
    )
    assert np.sum(profile) == pytest.approx(np.sum(rho * area))
    incoming = incoming_material_profile(
        centers_x=x, centers_y=y, areas=area, excess_density_m2=rho,
        old_tip_x_m=0.0, new_tip_x_m=0.5e-6,
        length_m=1e-6, n_bins=2, capture_width_m=1e-6,
    )
    assert np.sum(incoming) == pytest.approx(1.0)


def test_incremental_advection_uses_triangle_rectangle_overlap():
    triangles = np.array([
        [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5]],
        [[0.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
    ])
    centers = triangles.mean(axis=1)
    area = np.full(2, 0.5)
    rho = np.full(2, 2.0)
    profile = density_profile_in_active_zone(
        centers_x=centers[:, 0], centers_y=centers[:, 1], areas=area,
        density_m2=rho, tip_x_m=0.25, length_m=0.5, n_bins=2,
        capture_width_m=0.5, element_vertices=triangles,
    )
    assert np.sum(profile) == pytest.approx(0.5)
    incoming = incoming_material_profile(
        centers_x=centers[:, 0], centers_y=centers[:, 1], areas=area,
        excess_density_m2=rho, old_tip_x_m=-0.5, new_tip_x_m=-0.25,
        length_m=1.0, n_bins=2, capture_width_m=0.5,
        element_vertices=triangles,
    )
    assert np.sum(incoming) == pytest.approx(0.25)
    assert incoming[-1] == pytest.approx(0.25)


def test_signed_gnd_projection_recovers_linear_zero_spin_nye_field():
    # Structured triangles with element-centred ep_xx=y/L; alpha_13=-1/L.
    nodes = np.array([[x, y] for y in (0.0, 0.5, 1.0, 1.5)
                      for x in (0.0, 0.5, 1.0, 1.5)])
    def nid(ix, iy): return iy * 4 + ix
    elems = []
    for iy in range(3):
        for ix in range(3):
            a, b = nid(ix, iy), nid(ix + 1, iy)
            c, d = nid(ix, iy + 1), nid(ix + 1, iy + 1)
            elems.extend(((a, b, d), (a, d, c)))
    elems = np.asarray(elems)
    centers = nodes[elems].mean(axis=1)
    ep = np.zeros((3, len(elems)))
    ep[0] = centers[:, 1]
    out = signed_gnd_density(nodes, elems, ep, burgers_m=1.0,
                             slip_angles_deg=(0.0, 90.0))
    assert np.median(out["rho_signed_m2"][0]) == pytest.approx(-1.0, abs=0.15)
    assert np.median(out["rho_signed_m2"][1]) == pytest.approx(0.0, abs=0.15)


def test_signed_profile_preserves_cancellation_and_sign():
    triangles = np.array([
        [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5]],
        [[0.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
    ])
    rho = np.array([[2.0, -2.0], [-1.0, 1.0]])
    active, wake = signed_profiles(
        triangles=triangles, rho_signed_m2=rho, tip_x_m=0.0,
        active_length_m=1.0, active_bins=2, wake_length_m=1.0,
        wake_bins=2, capture_width_m=1.0)
    assert np.sum(active, axis=1) == pytest.approx([0.0, 0.0])
    assert np.all(wake == 0.0)


def test_incremental_operator_reproduces_additive_state_without_duplication():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    cfg = MPZConfig(length_m=1e-6, n_bins=10, wake_length_m=5e-6,
                    wake_n_bins=20, blunting_length_m=5e-6)
    pre = UnifiedMPZState(manifest, cfg)
    pre.retained[:, :4] = 2.0
    pure = pre.copy()
    pure.advance(0.5e-6)
    pure.convect_forward_blunting(0.5e-6)
    explicit = pure.copy()
    injected = np.zeros(explicit.n_bins)
    injected[-2:] = (3.0, 5.0)
    explicit.inject_retained_profile(injected, owner="bulk_inherited")
    explicit.register_forward_passage(4.0)
    operator = operator_from_states(
        explicit=explicit, pure_advection=pure, distance_m=0.5e-6,
        temperature_K=298.15, Kmax_Pa_sqrt_m=50e6, R=0.7,
        microsteps=5, event_duration_s=1e-6,
        inherited_count=8.0, opening_generated_count=0.0,
    )
    coarse = pre.copy()
    coarse.advance(0.5e-6)
    coarse.convect_forward_blunting(0.5e-6)
    operator.apply(coarse)
    assert coarse.retained == pytest.approx(explicit.retained)
    assert coarse.wake_retained == pytest.approx(explicit.wake_retained)
    assert coarse.forward_passage_slip_count == pytest.approx(
        explicit.forward_passage_slip_count)


def test_incremental_opening_options_are_opt_in():
    parser = _build_parser()
    default = parser.parse_args([])
    assert default.incremental_opening_advection is False
    enabled = parser.parse_args([
        "--incremental-opening-advection",
        "--opening-advection-step-um", "0.2",
        "--opening-advection-duration-s", "1e-6",
    ])
    assert enabled.incremental_opening_advection is True
    assert enabled.opening_advection_step_um == pytest.approx(0.2)


def test_signed_bulk_1d_sweep_preserves_face_sign_and_blunts_from_opening_projection():
    plastic = PlasticParameters(
        emission_G0_298_eV=1.0, plastic_sigc_298_Pa=2.0e9,
        transport_G0_298_eV=1.0, retained_recovery_per_s=0.0,
        initial_mobile_count=0.0, initial_retained_count=0.0,
    )
    blunting = BluntingParameters(
        source_slip_coefficient=10.0, forward_slip_coefficient=0.0,
        blunting_length_m=5.0e-6,
    )
    state = signed_bulk_1d.initial_state(plastic)
    state[signed_bulk_1d.BULK_LOWER_SIGNED] = 12.0
    state[signed_bulk_1d.BULK_UPPER_SIGNED] = -12.0
    before = signed_bulk_1d.effective_radius(state, blunting)
    advanced, receipt = signed_bulk_1d.sweep_increment(
        state, signed_bulk_1d.PROCESS_ZONE_M, blunting,
        maximum_step_m=0.25e-6,
    )
    assert advanced[signed_bulk_1d.BULK_LOWER_SIGNED] > 0.0
    assert advanced[signed_bulk_1d.BULK_UPPER_SIGNED] < 0.0
    assert receipt["bulk_lower_signed_swept"] == pytest.approx(
        -receipt["bulk_upper_signed_swept"]
    )
    assert receipt["bulk_opening_absorbed"] > 0.0
    assert receipt["bulk_signed_balance_residual"] == pytest.approx(0.0, abs=1e-12)
    assert signed_bulk_1d.effective_radius(advanced, blunting) > before


def test_signed_bulk_1d_forward_passage_birth_is_antisymmetric():
    panel = np.eye(signed_bulk_1d.base.HOMOGENEOUS_SIZE)
    panel[signed_bulk_1d.LOCAL_FORWARD_SLIP, signed_bulk_1d.MOBILE] = 0.25
    embedded = signed_bulk_1d._embed_base_panel(panel)
    state = np.zeros(signed_bulk_1d.STATE_SIZE)
    state[signed_bulk_1d.MOBILE] = 8.0
    out = embedded @ np.r_[state, 1.0]
    assert out[signed_bulk_1d.BULK_LOWER_SIGNED] == pytest.approx(1.0)
    assert out[signed_bulk_1d.BULK_UPPER_SIGNED] == pytest.approx(-1.0)


def test_fixed_reaction_loading_is_an_explicit_optional_control():
    parser = _build_parser()
    args = parser.parse_args([
        "--fatigue-loading-control", "fixed_reaction",
        "--fatigue-target-Kmax-MPa-sqrt-m", "40",
    ])
    assert args.fatigue_loading_control == "fixed_reaction"
    assert args.fatigue_target_Kmax_MPa_sqrt_m == pytest.approx(40.0)
    assert args.fatigue_target_reaction_N_per_m is None
    calibrated = parser.parse_args([
        "--fatigue-loading-control", "calibrated_fixed_displacement",
        "--fatigue-target-Kmax-MPa-sqrt-m", "40",
    ])
    assert calibrated.fatigue_loading_control == "calibrated_fixed_displacement"
