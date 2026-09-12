import copy

import numpy as np
import pytest

from arrhenius_fracture.closure_static_evidence import (
    TRACTION_REGISTRY, configuration, fixed_arc_probes, recompute_measurements,
    source_fingerprints, validate_solver_capture, resolution_screen,
)
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.finalization_v3_schema import canonical_hash


@pytest.fixture(scope="module")
def captured():
    cfg = configuration(False, 32, 12)
    return cfg, solve_crack_void_case(**cfg)["source_capture"]


def test_registry_has_eleven_physical_bases_not_fourteen_family_aliases():
    assert len(TRACTION_REGISTRY) == 11
    assert len({canonical_hash(v) for v in TRACTION_REGISTRY.values()}) == 11


def test_capture_reassembles_the_actual_production_system(captured):
    cfg, raw = captured
    validate_solver_capture(raw, cfg)
    assert set(source_fingerprints(raw)) == {"mesh", "support", "assembled_system", "solution"}


@pytest.mark.parametrize("field", ["stress", "K_data", "assembled_residual", "elasticity_D"])
def test_reassembly_rejects_mutation_even_if_fingerprints_are_recomputed(captured, field):
    cfg, raw = captured
    raw = copy.deepcopy(raw); raw[field].flat[0] += 10.
    with pytest.raises(ValueError, match="source reassembly mismatch"):
        validate_solver_capture(raw, cfg)


def test_recorded_nondefault_material_and_opening_are_passed_to_solver(captured):
    cfg, original = captured
    altered = dict(cfg, material_E_Pa=105e9, opening_m=8e-7)
    raw = solve_crack_void_case(**altered)["source_capture"]
    validate_solver_capture(raw, altered)
    np.testing.assert_allclose(raw["stress"], original["stress"], rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(raw["displacement"], original["displacement"]*2, rtol=1e-10, atol=1e-20)


def test_fixed_arc_operator_preserves_constant_full_tensor_and_wraparound(captured):
    cfg, raw = captured
    edges = recompute_measurements(raw, cfg)["edge_records"]
    tensor = [[3., 2.], [2., 5.]]
    for e in edges: e["adjacent_element_stress_tensor_Pa"] = tensor
    probes = fixed_arc_probes(edges, np.asarray(cfg["cavity_center_m"]), cfg["cavity_radius_m"])
    assert [p["arc_fraction"] for p in probes] == [0., .125, .25, .375, .5]
    for probe in probes:
        np.testing.assert_allclose(probe["tensor_Pa"], tensor)
        assert sum(probe["weights"]) == 1.
        assert min(probe["weights"]) >= 0.
        assert len(probe["candidate_planes"]) == 4


def test_crack_tip_probe_is_full_tensor_at_same_physical_point_on_two_meshes():
    points = []
    for n, r in ((128, 48), (128, 64)):
        cfg = configuration(True, n, r)
        result = solve_crack_void_case(**cfg)
        raw = result["source_capture"]
        validate_solver_capture(raw, cfg)
        probe = recompute_measurements(raw, cfg)["fixed_tip_probe"]
        points.append(probe["physical_position_m"])
        assert np.asarray(probe["tensor_Pa"]).shape == (2, 2)
        assert len(probe["element_ids"]) == len(probe["weights"])
    assert points[0] == points[1] == [0.000525, 0.]


def test_fixed_tip_probe_rejects_coarse_support_covering_the_probe():
    cfg = configuration(True, 32, 12)
    raw = solve_crack_void_case(**cfg)["source_capture"]
    with pytest.raises(ValueError, match="outside live solid"):
        recompute_measurements(raw, cfg)


def test_resolution_screen_cannot_pass_by_edge_count_alone():
    m = dict(eta_n_max=.026, eta_t_max=.0246, local_aspect_ratio_max=6.8, minimum_quality=.2)
    assert resolution_screen(m)
    assert not resolution_screen(dict(m, eta_n_max=.106, eta_t_max=.0123))
    assert not resolution_screen(dict(m, local_aspect_ratio_max=8.7))
    assert not resolution_screen(dict(m, minimum_quality=.049))


def test_raw_traction_matches_original_solver_without_projecting_stress(captured):
    cfg, raw = captured
    m = recompute_measurements(raw, cfg)
    expected = solve_crack_void_case(**cfg)["observables"]
    assert m["normalized_traction"] == pytest.approx(expected["cavity_traction_l2_normalized"], rel=1e-14)
    assert m["normalized_traction"] > .05
    assert m["assembled_cavity_node_residual_relative"] < 1e-8
