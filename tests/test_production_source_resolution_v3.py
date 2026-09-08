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


def test_stored_unqualified_candidate_metadata_matches_zero_effective_kinetics(connected):
    before=fingerprint(connected)
    hazards={h.candidate_id:h for h in connected.competition.hazard_states}
    rows=connected.junction_process_state["active_event_source"]["candidate_source_states"]
    source=connected.junction_process_state["active_event_source"]
    assert rows and any(row["resolved_opening_stress_Pa"]>0 for row in rows)
    for row in rows:
        positive=row["resolved_opening_stress_Pa"]>0
        assert row["geometry_status"] == ("GEOMETRICALLY_VALID_KINETICALLY_UNAVAILABLE" if positive
            else "GEOMETRICALLY_VALID_KINETICALLY_DORMANT")
        assert row["instantaneous_status"] == ("SOURCE_TENSOR_UNQUALIFIED" if positive else "ZERO_DOWNSTREAM_DRIVE")
        assert row["effective_rate_s"]==0 and row["crossing_time_s"]=="infinity"
        assert row["threshold_identity"]["threshold_action"] == hazards[row["candidate_id"]].current_threshold_action
        for name in ("source_kind","source_cavity_id","source_boundary_site_id","source_position_m",
                     "source_geometry_generation","source_tensor_fingerprint","source_probe_identity"):
            assert row[name]==source[name]
        assert row["source_mesh_generation"]==connected.event_counters["mesh_generation"]
    advanced,_,_,_=downstream_front_transaction(connected)
    assert fingerprint(advanced)==before
    assert advanced.competition==connected.competition and advanced.rng_state==connected.rng_state


def test_ligament_retires_arriving_front_before_first_support_rebuild():
    from arrhenius_fracture.voiding_production_v5 import ligament_transaction
    pre, _ = deterministic_trajectory(stop_before_ligament=True)
    before = fingerprint(pre)
    operations = []
    state, result = ligament_transaction(pre, operation_log=operations)
    assert result.accepted and fingerprint(pre) == before
    assert operations.index("root_status_change") < operations.index("support_rebuild")
    assert operations.index("cavity_phase_update") < operations.index("support_rebuild")
    assert state.crack_network.active_tip_ids == ()
    contexts = state.junction_process_state["boundary_terminal_context"]
    cavity_contexts = [item for values in contexts.values() for item in values
                       if item["boundary_kind"] == "cavity_free_surface"]
    assert cavity_contexts and all(item["endpoint_role"] == "inactive_terminal" for item in cavity_contexts)
    assert all(item["cavity_cycle_certified"] for item in cavity_contexts)


@pytest.mark.parametrize("stage", ["cavity_phase_update", "root_status_change", "support_rebuild"])
def test_pre_support_contact_transition_rolls_back_exactly(stage):
    from arrhenius_fracture.voiding_production_v5 import ligament_transaction
    pre, _ = deterministic_trajectory(stop_before_ligament=True)
    before = fingerprint(pre); operations = []
    with pytest.raises(RuntimeError, match="injected:" + stage):
        ligament_transaction(pre, failure_stage=stage, operation_log=operations)
    assert stage in operations and fingerprint(pre) == before


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
    from arrhenius_fracture.canonical_kinetic_time_v1 import AcceptedTime
    before=connected.junction_process_state.get('canonical_accepted_time_v1',
        AcceptedTime.from_seconds(connected.junction_process_state['production_time_s']))
    after=state.junction_process_state['canonical_accepted_time_v1']
    assert after.seconds_exact()-before.seconds_exact()==1
    assert state.junction_process_state['production_time_s']==after.seconds()
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


def test_bounded_local_refinement_retains_failed_traction_and_complete_accepted_state(connected):
    from arrhenius_fracture.voiding_production_v5 import refine_downstream_source
    before = fingerprint(connected)
    state, audit = refine_downstream_source(connected,max_refinement_levels=3)
    assert audit["status"] == "SOURCE_TENSOR_UNQUALIFIED"
    assert len(audit["attempts"]) == 3
    metric = audit["attempts"][-1]["proof"]["current_metrics"]
    assert metric["source_neighborhood_eta_n_max"] <= .03 and metric["source_neighborhood_eta_t_max"] <= .025
    assert metric["eta_t_max"] > .025  # The unrefined cavity is not relabelled resolved.
    assert metric["normalized_traction"] > .05
    assert state is connected and fingerprint(state) == before
    assert state.competition == connected.competition and state.rng_state == connected.rng_state


@pytest.mark.parametrize("stage", ["downstream_source_refinement", "source_support_rebuild", "source_equilibrium"])
def test_source_refinement_failure_restores_complete_connected_state(connected,stage):
    from arrhenius_fracture.voiding_production_v5 import refine_downstream_source
    before = fingerprint(connected); operations = []
    with pytest.raises(RuntimeError,match="injected:"+stage):
        refine_downstream_source(connected,max_refinement_levels=1,
            failure_stage=stage,operation_log=operations)
    assert stage in operations and fingerprint(connected) == before


def test_bounded_quality_flip_preserves_nodes_boundary_and_protected_crack_edge():
    from types import SimpleNamespace
    from arrhenius_fracture.mesh import rebuild_tri_mesh
    from arrhenius_fracture.voiding_production_v5 import _bounded_quality_edge_flips
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    nodes = np.array(((0.,0.),(.5,-.001),(1.,0.),(.5,1.)))
    mesh = rebuild_tri_mesh(nodes,np.array(((0,1,2),(0,2,3))))
    state = SimpleNamespace(mesh=mesh,crack_network=SimpleNamespace(branches=()))
    improved, flips = _bounded_quality_edge_flips(state)
    assert flips == ((0,2,1,3),)
    assert np.array_equal(improved.nodes,mesh.nodes)
    def boundary(elems):
        edges = np.sort(np.concatenate((elems[:,[0,1]],elems[:,[1,2]],elems[:,[2,0]])),axis=1)
        unique, counts = np.unique(edges,axis=0,return_counts=True)
        return unique[counts==1]
    assert np.array_equal(boundary(improved.elems),boundary(mesh.elems))
    protected = SimpleNamespace(mesh=mesh,crack_network=CrackNetworkState.one_tip(((0.,0.),(1.,0.))))
    unchanged, flips = _bounded_quality_edge_flips(protected)
    assert not flips and np.array_equal(unchanged.elems,mesh.elems)


def test_source_binding_includes_authoritative_p0_not_just_nodal_visualization(connected):
    from arrhenius_fracture.voiding_production_v5 import _cavity_resolution_binding
    before = _cavity_resolution_binding(connected)
    p0 = np.asarray(connected.mesh.element_damage_gp).copy()
    assert p0.shape == (connected.mesh.ne,)
    p0[0] = 1.-p0[0]
    tampered = replace(connected,mesh=replace(connected.mesh,element_damage_gp=p0))
    assert np.array_equal(tampered.damage,connected.damage)
    assert _cavity_resolution_binding(tampered) != before


def test_source_proof_without_previous_p0_capture_fails_closed(connected):
    from arrhenius_fracture.voiding_production_v5 import _cavity_resolution_binding,_qualified_cavity_source
    proof = {"schema":"v12.cavity-source-local-refinement/1","refinement_count":1,
        "current_binding":_cavity_resolution_binding(connected),
        "previous_binding":"irrelevant","current_metrics":{},"previous_metrics":{},
        "previous_source_capture":{"nodes":connected.mesh.nodes,"elements":connected.mesh.elems}}
    forged = replace(connected,junction_process_state={**connected.junction_process_state,
        "cavity_source_resolution_proof":proof})
    assert not _qualified_cavity_source(forged,np.eye(2)*1e9)


def test_tampered_reference_p0_capture_cannot_authorize_source(connected):
    from arrhenius_fracture.voiding_production_v5 import _cavity_resolution_binding,_qualified_cavity_source
    p0=np.asarray(connected.mesh.element_damage_gp).copy();p0[0]=1.-p0[0]
    capture={"nodes":connected.mesh.nodes,"elements":connected.mesh.elems,"boundary":connected.boundary,
        "element_damage_gp":p0,"displacement":connected.displacement,"damage":connected.damage,
        "ep_gp":connected.ep_gp,"rho_gp":connected.rho_gp,"energy_ledgers":connected.energy_ledgers}
    proof={"schema":"v12.cavity-source-local-refinement/1","refinement_count":1,
        "current_binding":_cavity_resolution_binding(connected),
        "previous_binding":_cavity_resolution_binding(connected),"current_metrics":{},"previous_metrics":{},
        "previous_source_capture":capture}
    forged=replace(connected,junction_process_state={**connected.junction_process_state,
        "cavity_source_resolution_proof":proof})
    assert not _qualified_cavity_source(forged,np.eye(2)*1e9)
