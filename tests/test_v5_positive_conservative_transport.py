from types import SimpleNamespace

import numpy as np
from scipy import sparse

from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    SurfaceKind,
    VacancyTransportParameterRowV2,
    _project_face_fluxes_to_positive_cell_balance,
    build_transport_surface_registry,
    initial_transport_state,
    solve_transport_interval,
)


def _nonorthogonal_mesh():
    nodes = np.asarray([
        (0.0, 0.0), (2.0e-6, 0.0), (2.0e-6, 1.0e-6),
        (0.0, 1.0e-6), (0.72e-6, 0.63e-6),
    ])
    elems = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    return rebuild_tri_mesh(nodes, elems)


def _closed_registry(mesh):
    owners = {}
    for eid, tri in enumerate(mesh.elems):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(eid)
    tags = {
        edge: SurfaceKind.CONTACT_OR_GRIP_NO_FLUX
        for edge, adjacent in owners.items() if len(adjacent) == 1
    }
    return build_transport_surface_registry(mesh, boundary_tags=tags)


def test_shared_face_flux_projection_is_positive_and_locally_conservative():
    # One interior face: its single oriented flux enters the two balances with
    # opposite signs.  The high-order value makes cell 0 negative.
    divergence = sparse.csr_matrix(np.asarray(((1.0,), (-1.0,))))
    high_flux = np.asarray((2.0,))
    diagonal = np.ones(2)
    rhs = np.ones(2)
    corrected, concentration, audit = _project_face_fluxes_to_positive_cell_balance(
        divergence, high_flux, diagonal, rhs, concentration_floor=0.1,
    )
    residual = diagonal*concentration + divergence@corrected-rhs
    assert np.min(concentration) >= 0.1-1.0e-11
    assert np.linalg.norm(residual, ord=np.inf) < 1.0e-14
    assert corrected[0] < high_flux[0]
    assert audit["cell_balance_residual_inf"] < 1.0e-14


def test_admissible_uniform_state_keeps_original_rt0_operator_on_nonorthogonal_mesh():
    mesh = _nonorthogonal_mesh()
    registry = _closed_registry(mesh)
    row = VacancyTransportParameterRowV2.from_material(
        SimpleNamespace(b=2.74e-10)
    )
    initial_c = np.full(mesh.ne, 1.0e-5)
    state = initial_transport_state(
        mesh, temperature_K=1800.0, row=row, registry=registry,
        vacancy_fraction_gp=initial_c,
    )
    zeros = np.zeros(mesh.ne)
    advanced = solve_transport_interval(
        state, mesh, registry, row=row, duration_s=0.01,
        accepted_equivalent_plastic_strain_rate_s=zeros,
        accepted_dislocation_density_m2=zeros,
        accepted_mean_stress_Pa=zeros,
    )
    audit = advanced.last_solver_audit
    assert np.allclose(advanced.vacancy_fraction_gp, initial_c, rtol=0.0, atol=1e-19)
    assert audit["conservative_positive_flux_correction_used"] is False
    assert audit["accepted_undamped_high_order_fixed_point"] is True
    assert audit["local_element_conservation_residual_inf"] < 1.0e-25
    assert abs(audit["global_conservation_residual_area_m2"]) <= audit[
        "conservation_roundoff_budget_area_m2"
    ]
