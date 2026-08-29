import copy
import math

import numpy as np
import pytest

from arrhenius_fracture.voiding_v2 import (
    Clock, LifecycleRates, Registry, Site, SiteState, VoidingV2Config,
    advance_lifecycle_localized, build_explicit_hole_mesh, fill_explicit_hole_mesh,
    series_limited_growth_rate, solve_static_hole, triangle_intersects_open_disk,
)


@pytest.mark.parametrize("a,b,expected", [
    (2.0,3.0,1.2),(0.0,3.0,0.0),(2.0,0.0,0.0),(-2.0,3.0,0.0),
    (2.0,-3.0,0.0),(-2.0,-3.0,0.0),
])
def test_series_growth_requires_two_positive_rates(a,b,expected):
    assert series_limited_growth_rate(a,b) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.nan,math.inf,-math.inf])
def test_series_growth_rejects_nonfinite(value):
    with pytest.raises(ValueError): series_limited_growth_rate(value,1.0)


def registry(birth=.5,stable=1.0,heal=10.0):
    return Registry(VoidingV2Config(enabled=True), sites={"s":Site(
        "s",(1.0,0.0),SiteState.AVAILABLE,Clock(0,birth),Clock(0,stable),Clock(0,heal),1.0)})


def fixed_rates(_registry,_site): return LifecycleRates(1.0,1.0,1.0)


def test_birth_does_not_reuse_full_step_for_stabilization():
    r=registry(birth=.75,stable=.5)
    advance_lifecycle_localized(r,"s",1.0,fixed_rates,.1)
    assert r.sites["s"].state == SiteState.EMBRYO
    assert r.sites["s"].stabilization.hazard == pytest.approx(.25)


def test_birth_then_stabilization_uses_only_residual_time():
    r=registry(birth=.25,stable=.5,heal=10)
    advance_lifecycle_localized(r,"s",1.0,fixed_rates,.1)
    assert r.sites["s"].state == SiteState.CONSUMED
    assert r.events[1]["time_within_step_s"] == pytest.approx(.75)


def test_birth_then_healing_and_tie_policy():
    r=registry(birth=.25,stable=.7,heal=.3)
    advance_lifecycle_localized(r,"s",1.0,fixed_rates,.1)
    assert r.sites["s"].state == SiteState.HEALED and not r.voids
    tie=registry(birth=.1,stable=.2,heal=.2)
    advance_lifecycle_localized(tie,"s",1.0,fixed_rates,.1)
    assert tie.sites["s"].state == SiteState.HEALED


def test_timestep_partition_invariance():
    one=registry(birth=.25,stable=.5,heal=10); split=copy.deepcopy(one)
    advance_lifecycle_localized(one,"s",1.0,fixed_rates,.1)
    advance_lifecycle_localized(split,"s",.4,fixed_rates,.1)
    advance_lifecycle_localized(split,"s",.6,fixed_rates,.1)
    assert one.sites["s"].state == split.sites["s"].state
    assert one.sites["s"].birth.hazard == split.sites["s"].birth.hazard
    assert one.sites["s"].stabilization.hazard == split.sites["s"].stabilization.hazard


@pytest.mark.parametrize("dt",[0,-1,math.nan])
def test_lifecycle_rejects_invalid_dt(dt):
    with pytest.raises(ValueError): advance_lifecycle_localized(registry(),"s",dt,fixed_rates,.1)


def test_post_transition_veto_rolls_back_exactly():
    r=registry(); before=copy.deepcopy(r)
    with pytest.raises(RuntimeError):
        advance_lifecycle_localized(r,"s",1.0,fixed_rates,.1,
            post_transition_veto=lambda *_: (_ for _ in ()).throw(RuntimeError("veto")))
    assert r == before


def test_triangle_disk_intersection_catches_edge_and_center_cases():
    assert triangle_intersects_open_disk(np.array([[-2,.5],[2,.5],[0,2.]]),(0,0),1)
    assert triangle_intersects_open_disk(np.array([[-2,-2],[2,-2],[0,2.]]),(0,0),1)
    assert not triangle_intersects_open_disk(np.array([[2,2],[3,2],[2,3]]),(0,0),1)


@pytest.mark.parametrize("h,n",[(2e-4,48),(1e-4,96)])
def test_actual_connectivity_derived_hole_invariants(h,n):
    hole=build_explicit_hole_mesh(.008,.008,(.004,0),.0005,h,n)
    v=hole.validation
    assert v["actual_internal_components"] == 1 and v["cavity_cycle"]
    assert v["triangle_disk_intersections"] == 0 and v["orphan_nodes"] == 0
    assert v["polygon_match_max_radius_error_m"] < 1e-14
    assert len(hole.cavity_edges) == n


def test_production_fem_hole_solution_is_equilibrated_and_symmetric():
    hole=build_explicit_hole_mesh(.008,.008,(.004,0),.0005,1e-4,96)
    result=solve_static_hole(hole,8e-6)
    assert result.free_residual_norm_N_per_m/result.reaction_top_N_per_m < 1e-12
    assert result.symmetry_error < 1e-12
    assert np.isfinite(result.stored_energy_J_per_m) and result.stored_energy_J_per_m > 0
    assert 2.5 < result.hoop_stress_concentration < 3.5


def test_filled_control_preserves_every_element_outside_cavity_patch():
    hole=build_explicit_hole_mesh(.008,.008,(.006,0),.00025,2e-4,48)
    control=fill_explicit_hole_mesh(hole)
    assert np.array_equal(control.mesh.nodes[:hole.mesh.nn],hole.mesh.nodes)
    assert np.array_equal(control.mesh.elems[:hole.mesh.ne],hole.mesh.elems)
    assert control.mesh.nn == hole.mesh.nn+1
    assert control.mesh.ne == hole.mesh.ne+len(hole.cavity_edges)
