from types import SimpleNamespace

import numpy as np
import pytest

import arrhenius_fracture.voiding_production_v5 as production


@pytest.mark.parametrize("polygon", [
    np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
    np.asarray(((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))),
])
def test_polygon_open_interior_excludes_terminal_boundary_contact(polygon):
    check = production._segment_intersects_convex_polygon_open_interior
    assert not check((-1.0, 0.5), (0.0, 0.5), polygon)
    assert check((-1.0, 0.5), (0.5, 0.5), polygon)
    assert check((-1.0, 0.5), (2.0, 0.5), polygon)
    assert not check((-1.0, 0.0), (0.5, 0.0), polygon)


def test_only_intersecting_candidate_uses_complete_ligament_observation(monkeypatch):
    candidates = (
        SimpleNamespace(candidate_id="contact", direction_xy=(1.0, 0.0)),
        SimpleNamespace(candidate_id="solid", direction_xy=(0.0, 1.0)),
    )
    branch = SimpleNamespace(tip=(0.0, 0.0))
    state = SimpleNamespace(
        competition=SimpleNamespace(candidates=candidates),
        crack_network=SimpleNamespace(branch=lambda branch_id: branch),
        material=SimpleNamespace(Eprime=9.0),
    )
    monkeypatch.setattr(
        production,
        "_first_ray_cavity_intersection",
        lambda state, start, direction: (2.0, 0.0) if direction == (1.0, 0.0) else None,
    )
    monkeypatch.setattr(
        production,
        "matched_ligament_energy_observation",
        lambda *args, **kwargs: {
            "status": "AVAILABLE",
            "G_ligament_J_per_m2": 4.0,
            "schema": "test",
        },
    )
    source_rows = (
        {"candidate_id": "contact", "directional_drive_status": "DIRECTIONAL_DRIVE_UNAVAILABLE"},
        {"candidate_id": "solid", "directional_drive_status": "AVAILABLE", "sentinel": 7},
    )
    rows = production.dispatch_cavity_intersecting_candidate_drives(
        state,
        branch_id="root",
        directional_rows=source_rows,
        maximum_event_advance_m=3.0,
    )
    assert rows[0]["directional_drive_status"] == "AVAILABLE"
    assert rows[0]["drive_source"] == "MATCHED_COMPLETE_LIGAMENT_TOPOLOGY_ENERGY"
    assert rows[0]["G_kinetic_used_J_per_m2"] == 4.0
    assert rows[0]["K_directional_Pa_sqrt_m"] == 6.0
    assert rows[0]["local_J_substituted"] is False
    assert rows[1] == source_rows[1]


def test_unconverged_complete_ligament_observation_remains_fail_closed(monkeypatch):
    candidate = SimpleNamespace(candidate_id="contact", direction_xy=(1.0, 0.0))
    state = SimpleNamespace(
        competition=SimpleNamespace(candidates=(candidate,)),
        crack_network=SimpleNamespace(
            branch=lambda branch_id: SimpleNamespace(tip=(0.0, 0.0))
        ),
        material=SimpleNamespace(Eprime=9.0),
    )
    monkeypatch.setattr(
        production,
        "_first_ray_cavity_intersection",
        lambda *args, **kwargs: (1.0, 0.0),
    )
    monkeypatch.setattr(
        production,
        "matched_ligament_energy_observation",
        lambda *args, **kwargs: {
            "status": "DIRECTIONAL_DRIVE_UNAVAILABLE",
            "G_ligament_J_per_m2": 4.0,
        },
    )
    rows = production.dispatch_cavity_intersecting_candidate_drives(
        state,
        branch_id="root",
        directional_rows=({"candidate_id": "contact"},),
        maximum_event_advance_m=2.0,
    )
    assert rows[0]["directional_drive_status"] == "DIRECTIONAL_DRIVE_UNAVAILABLE"
    assert rows[0]["directional_drive_unavailable_reason"] == (
        "MATCHED_COMPLETE_LIGAMENT_ENERGY_NOT_MESH_CONVERGED"
    )
    assert "K_directional_Pa_sqrt_m" not in rows[0]
