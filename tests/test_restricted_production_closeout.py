from copy import deepcopy

import pytest

from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
    INVALID_COMPARISON, NOT_EVALUATED, PASS, SCIENTIFICALLY_UNAVAILABLE,
    authenticated_payload, evaluate_resolved_sigma_tt_two_level_records,
)
from arrhenius_fracture.restricted_production_closeout_v1 import (
    DIRECT_OPERATIONS, VACANCY_OPERATIONS, aggregate_restricted_closeout,
    authenticate_operation_evidence, authenticate_stage, validate_lifecycle_trace,
)
from scripts.qualify_bounded_scheduler_closeout import (
    bind_cache, cache_dependency, load_bound_cache, validate_receipt_chain,
)


def _source(source_id="s0", *, traction=True, sigma=100.0, rate=2.0):
    return authenticated_payload({
        "source_id": source_id, "orientation_rad": 0.0,
        "nominal_coordinate_m": [1.0, 0.0], "realized_coordinate_m": [1.0, 0.0],
        "local_normal": [1.0, 0.0], "local_tangent": [0.0, 1.0],
        "physical_window": {"center": [0.0, 0.0], "dimensions": [2.0, 1.0], "orientation": 0.0},
        "mesh_support_identity": {"elements": [1, 2]}, "available": True,
        "sigma_tt_Pa": sigma, "activation_barrier_J": 3.0, "loop_emission_rate_s": rate,
        "predicates": {"boundary_traction": traction, "fit_residual": True,
                       "design_condition": True, "interface_continuity": True,
                       "resultant_balance": True, "moment_balance": True,
                       "mesh_quality": True, "source_window_ownership": True},
    })


def _evidence():
    identity = {
        "material": "bundle", "load": 1.0, "clocks": [0.2, 1.0], "rng": "r",
        "emission_parameters": "p", "emission_population": "population",
    }
    return authenticated_payload({
        "coarse_state_fingerprint": "state-c", "fine_state_fingerprint": "state-f",
        "coarse_mesh_fingerprint": "mesh-c", "fine_mesh_fingerprint": "mesh-f",
        "coarse_mesh_observation_fingerprint": "obs-c", "fine_mesh_observation_fingerprint": "obs-f",
        "coarse_material_observation_fingerprint": "mat", "fine_material_observation_fingerprint": "mat",
        "coarse_equilibrium": {"source_state_fingerprint": "eq-c", "mesh_fingerprint": "obs-c", "material_fingerprint": "mat"},
        "fine_equilibrium": {"source_state_fingerprint": "eq-f", "mesh_fingerprint": "obs-f", "material_fingerprint": "mat"},
        "coarse_physical_identity": identity, "fine_physical_identity": identity,
        "common_parent_state_fingerprint": "state-c",
        "refinement_lineage": {"parent_mesh_fingerprint": "mesh-c", "current_mesh_fingerprint": "mesh-f",
                               "refined_parent_element_ids": [1]},
        "transfer_conservation": {"material": True, "kinetics": True, "inventory": True, "geometry": True},
        "boundary_representation_fingerprint": "boundary",
    })


def _cert(*, evidence=None, coarse=None, fine=None):
    return evaluate_resolved_sigma_tt_two_level_records(
        _evidence() if evidence is None else evidence,
        [_source()] if coarse is None else coarse,
        [_source(sigma=101.0, rate=2.02)] if fine is None else fine)


def _reauthenticate_evidence(evidence):
    evidence.pop("authentication_sha256", None)
    return authenticated_payload(evidence)


def _reauthenticate_source(source):
    source.pop("authentication_sha256", None)
    return authenticated_payload(source)


def test_certificate_outcomes_are_disjoint():
    assert _cert().terminal_scientific_availability == PASS
    failed_traction = _source(traction=False)
    assert _cert(coarse=[failed_traction]).terminal_scientific_availability == SCIENTIFICALLY_UNAVAILABLE
    assert _cert(fine=[_source(sigma=200.0, rate=4.0)]).terminal_scientific_availability == SCIENTIFICALLY_UNAVAILABLE
    invalid = deepcopy(_evidence()); invalid["fine_mesh_fingerprint"] = "mesh-c"
    invalid = _reauthenticate_evidence(invalid)
    result = _cert(evidence=invalid, coarse=[failed_traction])
    assert result.terminal_scientific_availability == INVALID_COMPARISON
    assert evaluate_resolved_sigma_tt_two_level_records(None, None, None).terminal_scientific_availability == NOT_EVALUATED


@pytest.mark.parametrize("field", ["load", "material", "clocks", "rng", "emission_parameters", "emission_population"])
def test_changed_physical_identity_is_rejected(field):
    evidence = deepcopy(_evidence())
    evidence["fine_physical_identity"][field] = "changed"
    assert _cert(evidence=_reauthenticate_evidence(evidence)).terminal_scientific_availability == INVALID_COMPARISON


def test_coordinate_window_mesh_equilibrium_source_and_fingerprint_attacks_rejected():
    attacks = []
    shifted = deepcopy(_source()); shifted["realized_coordinate_m"] = [1.1, 0.0]
    attacks.append((_evidence(), [_source()], [_reauthenticate_source(shifted)]))
    window = deepcopy(_source()); window["physical_window"]["dimensions"] = [3.0, 1.0]
    attacks.append((_evidence(), [_source()], [_reauthenticate_source(window)]))
    duplicate = [_source("s0"), _source("s0")]
    attacks.append((_evidence(), duplicate, [_source("s0")]))
    missing = [_source("other")]
    attacks.append((_evidence(), [_source("s0")], missing))
    same_mesh = deepcopy(_evidence()); same_mesh["fine_mesh_fingerprint"] = "mesh-c"
    attacks.append((_reauthenticate_evidence(same_mesh), [_source()], [_source()]))
    stale = deepcopy(_evidence()); stale["fine_equilibrium"]["mesh_fingerprint"] = "old"
    attacks.append((_reauthenticate_evidence(stale), [_source()], [_source()]))
    forged = deepcopy(_evidence()); forged["fine_physical_identity"]["load"] = 2.0
    attacks.append((forged, [_source()], [_source()]))
    for evidence, coarse, fine in attacks:
        assert _cert(evidence=evidence, coarse=coarse, fine=fine).terminal_scientific_availability == INVALID_COMPARISON


def _trace(path):
    operations = VACANCY_OPERATIONS if path == "VACANCY_MEDIATED" else DIRECT_OPERATIONS
    rows = []
    for index, operation in enumerate(operations):
        mechanism = ("DIRECT_DISLOCATION_INTERACTION" if operation == "DIRECT_INTERACTION_BIRTH" else
                     "CAVITY_SURFACE_LOOP_EMISSION" if path == "DIRECT_DISLOCATION" and operation in {"STABLE_SUBGRID_GROWTH", "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED"} else
                     "CANONICAL_FRACTURE" if operation in {"LIGAMENT_CONNECTION", "DOWNSTREAM_CHILD", "CHILD_CONTINUATION"} else
                     "VACANCY_TRANSPORT_GROWTH" if path == "VACANCY_MEDIATED" and index else
                     "VACANCY_CLUSTERING")
        api = ("arrhenius_fracture.voiding_production_v5.downstream_front_transaction"
               if operation in {"DOWNSTREAM_CHILD", "CHILD_CONTINUATION"}
               else "arrhenius_fracture.embryo_growth_v1.EmbryoGrowthProcessOwner")
        balance = {"nonnegative": True, "residual_m2": 0.0, "roundoff_budget_m2": 1e-20}
        if operation == "DIRECT_INTERACTION_BIRTH": balance["vacancy_capture_required"] = False
        if path == "DIRECT_DISLOCATION" and operation == "STABLE_SUBGRID_GROWTH":
            balance["direct_birth_hits_are_cavity_loops"] = False
        if operation == "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED": balance["emission_rate_supplied"] = False
        proof = {"executed": True, "operation": operation,
                 "committed_duration_s": 0.1,
                 "committed_event_type": operation}
        scheduler_event = {
            "VACANCY_INVENTORY_BIRTH": "VACANCY_BIRTH_HIT",
            "DIRECT_INTERACTION_BIRTH": "DISLOCATION_INTERACTION_HIT",
            "EMBRYO_STABILIZED": "STABILIZATION",
            "PROMOTION": "GROWTH_PROMOTION",
            "LIGAMENT_CONNECTION": "LIGAMENT_RUPTURE",
            "DOWNSTREAM_CHILD": "DOWNSTREAM_FIRST_PASSAGE",
            "CHILD_CONTINUATION": "CHILD_CONTINUATION",
        }.get(operation)
        if operation in {
                "VACANCY_INVENTORY_BIRTH", "DIRECT_INTERACTION_BIRTH",
                "EMBRYO_STABILIZED", "STABLE_SUBGRID_GROWTH", "PROMOTION",
                "RESOLVED_VACANCY_GROWTH", "LIGAMENT_CONNECTION",
                "DOWNSTREAM_CHILD", "CHILD_CONTINUATION"}:
            proof["scheduler_audit"] = {
                "schema": "unified-2d.global-physical-event-scheduler/1",
                "events": ({"selected_event": scheduler_event},),
                "preview_nonmutating": True,
                "no_owner_advanced_twice": True,
            }
        if operation in {"CAVITY_LOOP_SUBGRID_GROWTH", "STABLE_SUBGRID_GROWTH",
                         "RESOLVED_VACANCY_GROWTH"}:
            proof["measured_growth_delta"] = {
                "radius_before_m": 1.0, "radius_after_m": 2.0,
                "area_before_m2": 1.0, "area_after_m2": 3.0,
                "mechanism_area_increments_m2": {"loop": 2.0},
                "conservation_residual_m2": 0.0,
                "conservation_budget_m2": 1e-12,
            }
        if operation == "DOWNSTREAM_SOURCE_REFINEMENT":
            proof.update({"committed_event_type": "REPRESENTATION_TRANSFER",
                          "physical_identity_preserved": True,
                          "kinetic_owners_preserved": True,
                          "transfer_checks_passed": True})
        if operation == "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED":
            proof["consumer_result"] = authenticate_operation_evidence({
                "mechanism_requested": True, "source_status": "UNAVAILABLE_NO_RATE",
                "production_rate_s": None, "actual_reason": "measured recovery failed",
                "accepted_state_unchanged": True})
        output = f"f{index+1}"
        if operation == "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED": output = f"f{index}"
        rows.append(authenticate_stage({
            "run_id": "run", "source_identity": {"source": "controlled-functional"},
            "physical_time_s": float(index), "operation": operation,
            "input_state_fingerprint": f"f{index}", "output_state_fingerprint": output,
            "site_id": "site-1", "embryo_id": "site-1", "cavity_id": "cavity-1",
            "front_id": "root", "active_mechanism": mechanism, "production_api": api,
            "inventory_balance": balance, "representation": "RESOLVED" if index >= 4 else "SUBGRID",
            "topology_state": operation, "owner_replaced": False, "phase_assignment_only": False,
            "evidence_kind": "CONTROLLED_FUNCTIONAL_SENTINEL",
            "operation_evidence": authenticate_operation_evidence(proof),
            **({"source_consumer_status": "UNAVAILABLE_NO_RATE"}
               if operation == "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED" else {}),
        }))
    return rows


def test_lifecycle_and_aggregate_fail_closed_then_accept_restricted():
    vacancy_rows, direct_rows = _trace("VACANCY_MEDIATED"), _trace("DIRECT_DISLOCATION")
    vacancy = validate_lifecycle_trace("VACANCY_MEDIATED", vacancy_rows)
    direct = validate_lifecycle_trace("DIRECT_DISLOCATION", direct_rows)
    assert vacancy.valid and direct.valid
    assert not validate_lifecycle_trace("VACANCY_MEDIATED", vacancy_rows[:-1]).valid
    unrelated = deepcopy(vacancy_rows); unrelated[-1]["site_id"] = "other"
    unrelated[-1] = _reauthenticate_source(unrelated[-1])
    assert not validate_lifecycle_trace("VACANCY_MEDIATED", unrelated).valid
    broken = deepcopy(vacancy_rows); broken[2]["input_state_fingerprint"] = "broken"
    broken[2] = _reauthenticate_source(broken[2])
    assert not validate_lifecycle_trace("VACANCY_MEDIATED", broken).valid
    available = deepcopy(direct_rows)
    proof = dict(available[-1]["operation_evidence"]); proof.pop("authentication_sha256")
    consumer = dict(proof["consumer_result"]); consumer.pop("authentication_sha256")
    consumer.update({"source_status": "RATE_SUPPLIED", "production_rate_s": [1.0]})
    proof["consumer_result"] = authenticate_operation_evidence(consumer)
    available[-1]["operation_evidence"] = authenticate_operation_evidence(proof)
    available[-1] = _reauthenticate_source(available[-1])
    assert not validate_lifecycle_trace("DIRECT_DISLOCATION", available).valid
    invalid_certificate = _cert(evidence=_reauthenticate_evidence({**deepcopy(_evidence()), "fine_mesh_fingerprint": "mesh-c"}))
    assert aggregate_restricted_closeout(invalid_certificate, vacancy, direct).framework == "PENDING_OR_FAILED_CLOSEOUT_GATES"
    unavailable = _cert(coarse=[_source(traction=False)])
    assert aggregate_restricted_closeout(unavailable, vacancy, direct).framework == \
        "PASS_WITH_RESOLVED_LOOP_EMISSION_SCIENTIFICALLY_UNAVAILABLE"


def test_operation_evidence_rejects_labels_noops_and_hidden_refinement():
    direct = _trace("DIRECT_DISLOCATION")
    no_op = deepcopy(direct)
    index = DIRECT_OPERATIONS.index("STABLE_SUBGRID_GROWTH")
    no_op[index]["output_state_fingerprint"] = no_op[index]["input_state_fingerprint"]
    proof = dict(no_op[index]["operation_evidence"])
    proof.pop("authentication_sha256")
    proof["measured_growth_delta"]["area_after_m2"] = 1.0
    no_op[index]["operation_evidence"] = authenticate_operation_evidence(proof)
    no_op[index] = _reauthenticate_source(no_op[index])
    assert not validate_lifecycle_trace("DIRECT_DISLOCATION", no_op).valid

    equilibrium = deepcopy(direct)
    proof = dict(equilibrium[index]["operation_evidence"]); proof.pop("authentication_sha256")
    proof["committed_event_type"] = "EQUILIBRATION"
    equilibrium[index]["operation_evidence"] = authenticate_operation_evidence(proof)
    equilibrium[index] = _reauthenticate_source(equilibrium[index])
    assert not validate_lifecycle_trace("DIRECT_DISLOCATION", equilibrium).valid

    disabled = deepcopy(direct); last = len(disabled)-1
    proof = dict(disabled[last]["operation_evidence"]); proof.pop("authentication_sha256")
    result = dict(proof["consumer_result"]); result.pop("authentication_sha256")
    result["mechanism_requested"] = False
    proof["consumer_result"] = authenticate_operation_evidence(result)
    disabled[last]["operation_evidence"] = authenticate_operation_evidence(proof)
    disabled[last] = _reauthenticate_source(disabled[last])
    assert not validate_lifecycle_trace("DIRECT_DISLOCATION", disabled).valid

    vacancy = _trace("VACANCY_MEDIATED")
    refine = VACANCY_OPERATIONS.index("DOWNSTREAM_SOURCE_REFINEMENT")
    hidden = vacancy[:refine] + vacancy[refine+1:]
    assert not validate_lifecycle_trace("VACANCY_MEDIATED", hidden).valid

    label_only = deepcopy(direct)
    proof = dict(label_only[index]["operation_evidence"])
    proof.pop("authentication_sha256")
    proof.pop("scheduler_audit")
    label_only[index]["operation_evidence"] = authenticate_operation_evidence(proof)
    label_only[index]["production_api"] = \
        "arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval"
    label_only[index] = _reauthenticate_source(label_only[index])
    result = validate_lifecycle_trace("DIRECT_DISLOCATION", label_only)
    assert not result.valid
    assert "global_scheduler_common_clock_receipts" in result.failed_predicates


def test_scheduler_receipt_chain_rejects_broken_endpoints_time_owner_and_order():
    rows = ({
        "input_state_fingerprint": "a", "output_state_fingerprint": "b",
        "elapsed_time_s": 0.25, "selected_event": None,
        "selected_owner": None,
    }, {
        "input_state_fingerprint": "b", "output_state_fingerprint": "c",
        "elapsed_time_s": 0.75, "selected_event": "CHILD_CONTINUATION",
        "selected_owner": ("downstream_front_transaction", "void-front-1"),
    })
    kwargs = dict(
        input_fingerprint="a", output_fingerprint="c",
        expected_events=("CHILD_CONTINUATION",),
        expected_owner_kinds=("downstream_front_transaction",),
        expected_duration_s=1.0)
    assert validate_receipt_chain(rows, **kwargs)
    for field, value in (
        ("input_state_fingerprint", "broken"),
        ("elapsed_time_s", -0.1),
        ("selected_owner", ("", "void-front-1")),
        ("selected_event", "DOWNSTREAM_FIRST_PASSAGE"),
    ):
        broken = deepcopy(rows)
        broken[1][field] = value
        assert not validate_receipt_chain(broken, **kwargs)


def test_scheduler_cache_is_bound_to_configuration_input_and_behavior_sources(tmp_path):
    dependency = cache_dependency(
        "unit-cache", input_fingerprint="state-a",
        configuration={"scale": 1.0})
    path = tmp_path/"cache.json"
    path.write_text(__import__("json").dumps(bind_cache(
        {"status": "PASS"}, dependency)))
    assert load_bound_cache(path, dependency)["status"] == "PASS"
    changed_config = cache_dependency(
        "unit-cache", input_fingerprint="state-a",
        configuration={"scale": 2.0})
    changed_input = cache_dependency(
        "unit-cache", input_fingerprint="state-b",
        configuration={"scale": 1.0})
    assert load_bound_cache(path, changed_config) is None
    assert load_bound_cache(path, changed_input) is None
    forged = __import__("json").loads(path.read_text())
    forged["cache_dependency"]["payload"]["configuration"]["scale"] = 9.0
    path.write_text(__import__("json").dumps(forged))
    assert load_bound_cache(path, dependency) is None
