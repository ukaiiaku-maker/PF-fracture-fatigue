import math

from arrhenius_fracture.voiding_v5 import (
    HazardClock, ProductionVoidState, VoidPhase, VoidSite,
    initialize_preexisting_resolved_cavity,
)


def test_preexisting_cavity_is_initial_geometry_not_vacancy_capture():
    clock = HazardClock(0.0, 1.0)
    state = ProductionVoidState((
        VoidSite("site-1", (6.0e-4, 0.0), VoidPhase.AVAILABLE_SITE,
                 0, 2, 1.0, clock, clock, clock),
    ), available_defect_inventory_area_m2=3.0e-10)
    result = initialize_preexisting_resolved_cavity(state, "site-1", 5.0e-6)
    cavity = result.cavities[0]
    growth = result.embryo_growth_states[0]
    assert cavity.phase is VoidPhase.RESOLVED_VOID
    assert cavity.inventory_area_m2 == 0.0
    assert result.available_defect_inventory_area_m2 == state.available_defect_inventory_area_m2
    assert result.consumed_defect_inventory_area_m2 == 0.0
    assert growth.initial_area_m2 == math.pi * (5.0e-6) ** 2
    assert growth.vacancy_derived_area_m2 == 0.0
    assert result.length_ledgers["preexisting_void_free_span_m"] == 1.0e-5
    assert result.event_history[-1]["vacancy_inventory_debit_area_m2"] == 0.0
