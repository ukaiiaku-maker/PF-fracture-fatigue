from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, build_production_void_state,
    deterministic_trajectory, natural_trajectory,
)
from arrhenius_fracture.voiding_v5 import (
    Cavity2D, VoidPhase, VoidingConfig, advance_site, arrhenius_rates,
    create_subgrid_cavity, grow_cavity_2d, grow_cavity_from_rate,
)


def test_plane_strain_cavity_inventory_is_area_not_spherical_inventory():
    radius = 5.0e-5
    cavity = Cavity2D("v", "s", (0.0, 0.0), radius, math.pi * radius**2,
                      math.pi * radius**2, VoidPhase.RESOLVED_VOID)
    grown = grow_cavity_2d(cavity, 1.0e-5)
    assert grown.area_m2 == math.pi * grown.radius_m**2
    assert grown.inventory_area_m2 - cavity.inventory_area_m2 == grown.area_m2 - cavity.area_m2


def test_deterministic_driver_reaches_real_continued_graph_event():
    final, rows = deterministic_trajectory()
    assert rows[-1]["operation"] == "continued_accepted_event"
    assert rows[-1]["event_counters"]["topology_actions"] >= 3
    assert final.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE


def test_natural_driver_integrates_actual_stress_history():
    final, rows = natural_trajectory()
    assert len(rows) == 6
    assert all(row["rates"]["local_max_principal_stress_Pa"] >= 0.0 for row in rows)
    assert final.void_state.sites[0].birth.accumulated >= 0.0


def test_multihit_renews_stochastic_threshold_and_rng_state():
    state, _ = build_production_void_state(stochastic=True)
    site = state.void_state.sites[0]
    rates = {"birth_s": 1.0, "stabilization_s": 0.0, "healing_s": 0.0}
    advanced, events = advance_site(
        state.void_state, site.site_id, site.birth.threshold / site.candidate_weight,
        rates=rates,
    )
    renewed = advanced.sites[0]
    assert events == ("BIRTH_HIT",)
    assert renewed.birth.threshold != site.birth.threshold
    assert advanced.rng_state != state.void_state.rng_state


def test_series_limiter_and_duplicate_cavity_guard():
    cfg = VoidingConfig(enabled=True)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=np.eye(2) * 1.0e9)
    assert rates["series_limited_growth_s"] < min(
        rates["surface_reaction_s"], rates["vacancy_transport_s"]
    )
    cavity = Cavity2D("v", "s", (0.0, 0.0), 1.0e-6, math.pi * 1.0e-12,
                      math.pi * 1.0e-12, VoidPhase.STABLE_SUBGRID_VOID)
    grown = grow_cavity_from_rate(
        cavity, rates=rates, dt_s=1.0e-12,
        radial_growth_scale_m=cfg.radial_growth_scale_m,
    )
    assert grown.radius_m > cavity.radius_m

    state, _ = build_production_void_state()
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    voids = replace(state.void_state, sites=(site,))
    voids = create_subgrid_cavity(voids, site.site_id, 1.0e-6)
    with pytest.raises(ValueError, match="only one cavity"):
        create_subgrid_cavity(voids, site.site_id, 1.0e-6)


def test_no_zero_drive_or_unit_rate_bypass_for_cleavage_first_passage():
    state, _ = build_production_void_state()
    with pytest.raises(RuntimeError, match="cannot reach first passage"):
        _complete_next_clock(state, np.zeros((2, 2)))
