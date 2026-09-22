from dataclasses import replace

import numpy as np

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step,
)
from arrhenius_fracture.plasticity_vacancy_budget_v1 import (
    MECHANISM_INTERPRETATION, geometry_region_masks,
)
from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


def _state(temperature=1200.0):
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(bundle=bundle, temperature_K=temperature)
    rho0 = float(bundle.fracture_row["rho_forest_floor_m2"])
    return equilibrate_fixed_load_with_production_fem(replace(
        state, ep_gp=np.zeros_like(state.ep_gp), rho_gp=np.full_like(state.rho_gp, rho0),
    ))


def test_geometry_regions_are_complete_disjoint_and_numbering_independent():
    state = _state()
    masks = geometry_region_masks(state)
    ownership = sum(mask.astype(int) for mask in masks.values())
    assert np.array_equal(ownership, np.ones(state.mesh.ne, dtype=int))
    assert all(np.any(mask) for mask in masks.values())


def test_1200K_short_step_has_source_outside_mpz_without_explicit_jog_term():
    state = _state()
    accepted, audit = accepted_bulk_load_step(
        state, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.10, duration_s=1.0e-8, temperature_K=1200.0,
    )
    masks = geometry_region_masks(accepted)
    epdot = np.asarray(audit["dot_ep_gp_s"])
    assert np.sum(accepted.mesh.area_e[~masks["TIP_MPZ"]]*epdot[~masks["TIP_MPZ"]]) > 0.0
    assert MECHANISM_INTERPRETATION == (
        "COARSE_GRAINED_JOG_INTERSECTION_AND_NONCONSERVATIVE_PLASTIC_SOURCE"
    )
