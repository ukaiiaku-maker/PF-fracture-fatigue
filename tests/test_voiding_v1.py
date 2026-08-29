import copy
import math

import numpy as np
import pytest

from arrhenius_fracture.voiding_v1 import (
    Cavity, CavityStatus, FirstPassageState, SiteClass, SiteStatus, VoidRegistry,
    VoidSite, VoidingConfig, activate_downstream_front, advance_site_lifecycle,
    birth_intensity, connect_crack_to_void, crack_to_void_ligament_candidate,
    empty_registry, make_explicit_circular_hole_mesh, poisson_completion_probability,
    promote_cavity, promotion_is_resolved, series_limited_growth_rate,
    validate_explicit_hole,
)


def test_voiding_disabled_by_default_and_empty_registry_exact_roundtrip():
    reg = empty_registry()
    assert not reg.config.enabled and reg.sites == {} and reg.cavities == {}
    fp = reg.fingerprint()
    assert copy.deepcopy(reg).fingerprint() == fp
    with pytest.raises(RuntimeError):
        reg.instantiate_site(VoidSite("s", SiteClass.PRESCRIBED_TEST_SITE, (0, 0)))


def test_multihit_and_candidate_weight_enter_birth_only():
    site = VoidSite("s", SiteClass.PRESCRIBED_TEST_SITE, (0, 0),
                    statistical_weight_birth_only=5, required_hits=3, completion_lambda=2)
    assert poisson_completion_probability(3, 2) == pytest.approx(1-5*math.exp(-2))
    assert birth_intensity(site, 1000, 2, 0, 0) == pytest.approx(10*(1-5*math.exp(-2)))
    assert series_limited_growth_rate(2, 3) == pytest.approx(1.2)


def test_distinct_embryo_heal_and_single_stabilization_paths():
    reg = VoidRegistry(VoidingConfig(enabled=True))
    site = VoidSite("s", SiteClass.PRESCRIBED_TEST_SITE, (1, 0),
                    birth=FirstPassageState(0, .1), stabilization=FirstPassageState(0, .2),
                    healing=FirstPassageState(0, 10))
    reg.instantiate_site(site)
    cavity = advance_site_lifecycle(reg, "s", 1, 1, 1, 0, .2)
    assert cavity is not None and site.status == SiteStatus.CONSUMED_SITE
    assert len(reg.cavities) == 1
    assert advance_site_lifecycle(reg, "s", 1, 1, 1, 0, .2) is None

    healed = VoidRegistry(VoidingConfig(enabled=True))
    s2 = VoidSite("h", SiteClass.PRESCRIBED_TEST_SITE, (1, 0),
                  birth=FirstPassageState(0, .1), stabilization=FirstPassageState(0, 10),
                  healing=FirstPassageState(0, .2))
    healed.instantiate_site(s2)
    assert advance_site_lifecycle(healed, "h", 1, 1, 0, 1, .2) is None
    assert s2.status == SiteStatus.HEALED_SITE and healed.cavities == {}


@pytest.mark.parametrize("segments", [24, 48, 96])
def test_true_closed_explicit_hole_converges(segments):
    hole = make_explicit_circular_hole_mesh(4, 4, (2, 0), .5, .16, segments)
    valid = validate_explicit_hole(hole)
    assert valid["no_triangles_inside"] and valid["closed_boundary_components"] == 1
    assert valid["true_hole_no_void_material"] and valid["traction_boundary_condition"] == "NATURAL_ZERO_TRACTION"
    assert abs(hole.area_m2-math.pi*.25)/(math.pi*.25) < .025
    assert abs(hole.perimeter_m-math.pi)/(math.pi) < .012
    assert hole.minimum_quality > 0


def test_ligament_intersection_miss_length_accounting_and_rollback():
    reg = VoidRegistry(VoidingConfig(enabled=True))
    cav = Cavity("v", "s", SiteClass.PRESCRIBED_TEST_SITE, (2, 0), CavityStatus.RESOLVED_VOID)
    cav.set_radius(.5); reg.cavities["v"] = cav
    assert crack_to_void_ligament_candidate((0, 0), (0, 1), cav, "existing-cleavage") is None
    candidate = crack_to_void_ligament_candidate((0, 0), (1, 0), cav, "existing-cleavage")
    assert candidate["fractured_ligament_length_m"] == pytest.approx(1.5)
    assert candidate["free_void_span_m"] == 0 and candidate["uses_existing_cleavage_barrier"]
    before = reg.fingerprint()
    with pytest.raises(RuntimeError):
        connect_crack_to_void(reg, "v", candidate, "crack:0", lambda: (_ for _ in ()).throw(RuntimeError("veto")))
    assert reg.fingerprint() == before
    connect_crack_to_void(reg, "v", candidate, "crack:0")
    assert reg.ledger.fractured_ligament_increment == pytest.approx(1.5)
    assert reg.ledger.free_void_span_increment == 0


def test_promotion_is_nonduplicative_and_tip_void_radii_stay_distinct():
    cav = Cavity("v", "s", SiteClass.PRESCRIBED_TEST_SITE, (2, 0)); cav.set_radius(.5)
    cfg = VoidingConfig(enabled=True)
    assert promotion_is_resolved(cav, .1, .5, cfg, 48)
    hole = make_explicit_circular_hole_mesh(4, 4, (2, 0), .5, .1, 48)
    promote_cavity(cav, hole, 3)
    assert cav.void_or_site_id == "v" and cav.parent_site_id == "s"
    reg = VoidRegistry(cfg, cavities={"v": cav})
    candidate = crack_to_void_ligament_candidate((0, 0), (1, 0), cav, "cleavage-v1")
    connect_crack_to_void(reg, "v", candidate, "crack:0")
    event = activate_downstream_front(reg, "v", (2.5, 0), (1, 0), "front:1", .02)
    assert event["r_tip_m"] == .02 and event["R_void_m"] == .5
    assert not event["analytical_tip_amplification_used"]
    assert reg.ledger.fractured_ligament_increment != reg.ledger.free_void_span_increment


def test_prohibited_model_terms_absent_from_production_module():
    from pathlib import Path
    text = Path("arrhenius_fracture/voiding_v1.py").read_text().lower()
    for token in ("a"+"t1", "a"+"t2", "gurson", "gtn", "prony", "mori", "area_loss"):
        assert token not in text
