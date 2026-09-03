import math

from arrhenius_fracture.voiding_production_v4 import deterministic_trajectory, natural_trajectory
from arrhenius_fracture.voiding_v4 import Cavity2D, VoidPhase, grow_cavity_2d


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

