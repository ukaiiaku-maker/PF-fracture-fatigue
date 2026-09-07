from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.checkpoint_v11 import write_checkpoint, restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, deterministic_trajectory, downstream_front_transaction,
    directional_clock_rates, build_production_void_state,
    _geometry, _grow_hole_boundary,
    _actual_cavity_boundary_edges, _first_ray_cavity_intersection,
)
from arrhenius_fracture.voiding_v5 import VoidPhase


@pytest.fixture(scope="module")
def connected():
    state, rows = deterministic_trajectory()
    assert rows[-1]["operation"] == "downstream_first_passage_unavailable"
    return state


def test_accepted_ligament_survives_unqualified_source_without_child(connected):
    assert connected.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID
    assert connected.crack_network.active_tip_ids == ()
    assert len(connected.crack_network.branches) == 1
    assert connected.junction_process_state["latest_crack_void_connection_certificate"]["passed"]
    advanced, result, operations, audit = downstream_front_transaction(connected)
    assert fingerprint(advanced) == fingerprint(connected)
    assert result is None and not operations
    assert audit["status"] == "UNQUALIFIED_CAVITY_SOURCE_TENSOR"


@pytest.mark.parametrize("partitions", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("source_kind", ["sharp_front", "cavity_surface"])
def test_no_unqualified_source_can_advance_clock_even_with_source_kind_spoofing(connected, partitions, source_kind):
    before = fingerprint(connected); state = connected
    for _ in range(partitions):
        state, audit = _complete_next_clock(state, np.eye(2)*1e9,
            source_kind=source_kind, maximum_advance_duration_s=1./partitions)
        assert all(r["effective_rate_s"] == 0. and math.isinf(r["crossing_time_s"]) and
                   not r["emitted_event_ids"] and not r["winner"] for r in audit)
    assert fingerprint(state) == before


def test_user_authored_qualification_boolean_cannot_bypass_guard(connected):
    junction = dict(connected.junction_process_state, production_resolution_cavity_tensor_qualified=True)
    forged = replace(connected, junction_process_state=junction)
    state, audit = _complete_next_clock(forged, np.eye(2)*1e9)
    assert state is forged
    assert not any(r["winner"] for r in audit)


def test_zero_drive_preserves_complete_candidate_provenance(connected):
    state, audit = _complete_next_clock(connected, np.zeros((2, 2)), maximum_advance_duration_s=1.)
    assert state.competition == connected.competition
    assert state.rng_state == connected.rng_state
    assert state.junction_process_state["active_event_source"] == connected.junction_process_state["active_event_source"]
    assert state.junction_process_state["production_time_s"] == connected.junction_process_state["production_time_s"]+1.
    assert all(r["instantaneous_status"] == "ZERO_DOWNSTREAM_DRIVE" for r in audit)


def test_checkpoint_restart_retains_source_guard_and_all_ownership(connected, tmp_path):
    path = tmp_path/"connected.json"
    write_checkpoint(connected, path)
    restored = restore_checkpoint(path)
    direct, a = _complete_next_clock(connected, np.eye(2)*1e9)
    resumed, b = _complete_next_clock(restored, np.eye(2)*1e9)
    assert fingerprint(direct) == fingerprint(resumed) == fingerprint(connected)
    assert a == b


def test_readonly_rate_diagnostic_uses_existing_hazard_without_consuming_it(connected):
    before = fingerprint(connected)
    rates = directional_clock_rates(connected, np.eye(2)*1e9)
    for hazard, row in zip(connected.competition.hazard_states, rates):
        assert row["crossing_time_s"] == (hazard.current_threshold_action-hazard.action)/row["effective_rate_s"]
    assert fingerprint(connected) == before


def test_checkpoint_candidate_measurement_contract_matches_production_runner(connected):
    from arrhenius_fracture.closure_production_evidence import candidate_measurements
    from scripts.qualify_voiding_v5_production_transfer import candidate_measurements as runner, canonical
    before=fingerprint(connected)
    assert candidate_measurements(connected)==canonical(runner(connected))
    assert fingerprint(connected)==before


def test_transfer_ontology_rejects_missing_physical_registry():
    from arrhenius_fracture.closure_production_evidence import validate_production, SCHEMA
    with pytest.raises(ValueError,match="physical registry"):
        validate_production({"transfer_manifest":{"schema":SCHEMA,"executed_code_sha":"a"},
                             "rows":[]},{},executed_code_sha="a")


def test_resolution_change_requires_fixed_physical_crack():
    with pytest.raises(ValueError, match="explicit fixed crack path"):
        build_production_void_state(boundary_segments=64, radial_layers=24)


def test_resolved_growth_rebuilds_normal_layers_without_overtaking_solid():
    from arrhenius_fracture.explicit_cavity_v5 import triangle_intersects_open_disk
    hole, _ = _geometry(boundary_segments=128, radial_layers=48)
    path = ((0., 0.), (0.0005725993004046688, 0.))
    grown = _grow_hole_boundary(hole, 5.5e-5, crack_path_m=path)
    assert grown.validation["triangle_disk_intersections"] == 0
    assert all(not triangle_intersects_open_disk(grown.mesh.nodes[e], grown.center_m, grown.radius_m)
               for e in grown.mesh.elems)
    for point in path:
        assert min(np.linalg.norm(grown.mesh.nodes-point, axis=1)) <= 1e-12
    assert grown.radius_m == 5.5e-5


def test_fine_radial_nodes_inside_identification_band_are_not_surface_nodes():
    from types import SimpleNamespace
    hole,_=_geometry(radius_m=5.5e-5,boundary_segments=256,radial_layers=192)
    state=SimpleNamespace(mesh=hole.mesh,void_state=SimpleNamespace(cavities=(SimpleNamespace(
        center_m=hole.center_m,radius_m=hole.radius_m),)))
    radii=np.linalg.norm(hole.mesh.nodes-np.asarray(hole.center_m),axis=1)
    assert np.count_nonzero(radii<=hole.radius_m*1.02)>256
    assert len(np.unique(_actual_cavity_boundary_edges(state)))==256
    hit=_first_ray_cavity_intersection(state,(.0005725993004046688,0.),(1.,0.))
    assert hit[0]==pytest.approx(7e-4-5.5e-5/math.cos(math.pi/256),abs=1e-15)
