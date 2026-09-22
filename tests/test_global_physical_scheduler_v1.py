from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.global_physical_scheduler_v1 import (
    _advance_front_engines,
    PhysicalEventPreview, advance_directional_preview,
    advance_global_physical_interval,
    commit_directional_preview, commit_front_engine_emission,
    preview_front_engine_emission, preview_vacancy_birth_hit,
    preview_downstream_first_passage, preview_healing, preview_stabilization,
    select_physical_event,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, _complete_next_clock_fused,
    build_production_void_state, crack_tip_tensor,
    preview_directional_first_passage,
)
from arrhenius_fracture.voiding_v5 import VoidPhase
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    commit_site_inventory_hit, locate_site_inventory_event,
    preview_site_inventory_event,
)


@pytest.fixture(scope="module")
def accepted_state():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    return state


def _directional_kwargs(state, duration=1.0e-8):
    tensor, elements = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    return tensor, {
        "source_front_id": ROOT_BRANCH_ID,
        "source_position_m": state.crack_network.branch(ROOT_BRANCH_ID).tip,
        "source_probe_identity": {"kind": "scheduler-test", "element_ids": list(elements)},
        "maximum_advance_duration_s": duration,
    }


def test_directional_split_is_pure_and_exactly_equivalent_to_fused(accepted_state):
    tensor, kwargs = _directional_kwargs(accepted_state)
    before = complete_accepted_state_fingerprint(accepted_state)
    fused, fused_audit = _complete_next_clock_fused(accepted_state, tensor, **kwargs)
    preview = preview_directional_first_passage(accepted_state, tensor, **kwargs)
    assert complete_accepted_state_fingerprint(accepted_state) == before
    advanced, audit = advance_directional_preview(accepted_state, preview)
    committed = commit_directional_preview(advanced, preview)
    assert complete_accepted_state_fingerprint(committed) == complete_accepted_state_fingerprint(fused)
    assert audit == fused_audit
    wrapped, wrapped_audit = _complete_next_clock(accepted_state, tensor, **kwargs)
    assert complete_accepted_state_fingerprint(wrapped) == complete_accepted_state_fingerprint(fused)
    assert wrapped_audit == fused_audit


def test_directional_rate_scale_is_guarded_and_uses_the_same_clock_owner(accepted_state):
    tensor, kwargs = _directional_kwargs(accepted_state, duration=1.0e-12)
    baseline = preview_directional_first_passage(accepted_state, tensor, **kwargs)
    with pytest.raises(RuntimeError, match="restricted to labeled functional qualification"):
        preview_directional_first_passage(
            accepted_state, tensor, functional_test_rate_scale=2.0, **kwargs)
    controlled = preview_directional_first_passage(
        accepted_state, tensor, functional_test_rate_scale=2.0,
        test_only_functional_qualification=True, **kwargs)
    base_rows = {row["candidate_id"]: row for row in baseline.commit_payload["audit"]}
    controlled_rows = {
        row["candidate_id"]: row for row in controlled.commit_payload["audit"]}
    assert controlled.threshold_identity == baseline.threshold_identity
    assert controlled.candidate_ids == baseline.candidate_ids
    assert controlled.source_fingerprint == baseline.source_fingerprint
    assert controlled_rows.keys() == base_rows.keys()
    for candidate_id, row in controlled_rows.items():
        reference = base_rows[candidate_id]
        assert row["hazard_barrier_J"] == reference["hazard_barrier_J"]
        assert row["effective_rate_s"] == pytest.approx(
            2.0*reference["effective_rate_s"])
        assert row["functional_test_rate_scale"] == 2.0
        assert row["rate_scale_classification"] == (
            "CONTROLLED_NOT_CALIBRATED_REFERENCE")


def test_selected_vacancy_birth_preview_is_atomically_committed():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1200.0,
        initial_vacancy_fraction_gp=0.0099567)
    transport = state.vacancy_transport_state_v2
    controlled_site = replace(
        transport.site_states[0], capture_rate_area_per_s=1.0e10,
        support_radius_m=7.2e-4)
    state = replace(state, vacancy_transport_state_v2=replace(
        transport, site_states=(controlled_site,)))
    parent = np.asarray(
        state.vacancy_transport_state_v2.parent_mechanics_element_ids, int)
    preview = preview_vacancy_birth_hit(
        state, maximum_duration_s=1.0e-6,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(len(parent)),
        accepted_dislocation_density_m2=np.asarray(state.rho_gp)[parent],
        accepted_mean_stress_Pa=np.zeros(len(parent)))
    assert preview.event_type == "VACANCY_BIRTH_HIT"

    advanced, audit = advance_global_physical_interval(
        state, state, preview.crossing_time_s, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 16,
            "diagnostic_freeze_front_process": True,
            "stress_tensor_Pa": np.zeros((2, 2)),
        })

    assert [event["selected_event"] for event in audit["events"]] == [
        "VACANCY_BIRTH_HIT"]
    assert advanced.vacancy_transport_state_v2.site_states[0].hit_count == 1
    assert advanced.void_state.sites[0].hits == 1


def test_directional_preview_rejects_stale_state(accepted_state):
    tensor, kwargs = _directional_kwargs(accepted_state)
    preview = preview_directional_first_passage(accepted_state, tensor, **kwargs)
    stale = replace(accepted_state, stored_energy_J_per_m=accepted_state.stored_energy_J_per_m+1.0)
    with pytest.raises(RuntimeError, match="stale"):
        advance_directional_preview(stale, preview)


def test_directional_partition_invariance_and_preview_advance_commit_rollback(accepted_state):
    tensor, kwargs = _directional_kwargs(accepted_state, duration=1.0e-8)
    direct, _ = _complete_next_clock(accepted_state, tensor, **kwargs)
    half_kwargs = dict(kwargs, maximum_advance_duration_s=5.0e-9)
    half, _ = _complete_next_clock(accepted_state, tensor, **half_kwargs)
    partitioned, _ = _complete_next_clock(half, tensor, **half_kwargs)
    assert complete_accepted_state_fingerprint(direct) == complete_accepted_state_fingerprint(partitioned)
    original = complete_accepted_state_fingerprint(accepted_state)
    for boundary in ("preview", "advance", "commit"):
        def fail(stage, _state, expected=boundary):
            if stage == expected:
                raise RuntimeError("injected:"+stage)
        if boundary == "preview":
            with pytest.raises(RuntimeError, match="injected:preview"):
                preview_directional_first_passage(
                    accepted_state, tensor, failure_injector=fail, **kwargs,
                )
        else:
            preview = preview_directional_first_passage(accepted_state, tensor, **kwargs)
            if boundary == "advance":
                with pytest.raises(RuntimeError, match="injected:advance"):
                    advance_directional_preview(accepted_state, preview, failure_injector=fail)
            else:
                advanced, _ = advance_directional_preview(accepted_state, preview)
                with pytest.raises(RuntimeError, match="injected:commit"):
                    commit_directional_preview(advanced, preview, failure_injector=fail)
        assert complete_accepted_state_fingerprint(accepted_state) == original


def test_front_engine_emission_is_continuous_diagnostic_not_event(accepted_state):
    before = complete_accepted_state_fingerprint(accepted_state)
    preview = preview_front_engine_emission(
        accepted_state, ROOT_BRANCH_ID, 1.0e8, 900.0, 1.0e-6,
    )
    assert complete_accepted_state_fingerprint(accepted_state) == before
    assert preview.event_type == "NO_DISCRETE_EMISSION_EVENT"
    assert not preview.eligible
    assert not preview.event_ids
    assert preview.commit_payload["event_kind"] == "CONTINUOUS_EMISSION_DIAGNOSTIC"
    assert preview.commit_payload["emission_rate_s"] > 0.0
    assert preview.commit_payload["pre_engine_fingerprint"] != preview.commit_payload["post_engine_fingerprint"]
    assert dict(preview.threshold_identity)["renewal_threshold"] == 1.0
    assert preview.commit_payload["emitted_population_increment"] > 0.0
    with pytest.raises(ValueError, match="continuous state evolution"):
        commit_front_engine_emission(accepted_state, preview)


def test_active_front_process_requires_complete_load_map_and_advances(accepted_state):
    with pytest.raises(RuntimeError, match="loads are incomplete"):
        _advance_front_engines(accepted_state, accepted_state, 1.0e-9, {}, 900.0)
    advanced, rows = _advance_front_engines(
        accepted_state, accepted_state, 1.0e-9,
        {ROOT_BRANCH_ID: 1.0e6}, 900.0)
    assert len(rows) == 1
    owner, row = rows[0]
    assert owner == ROOT_BRANCH_ID
    assert row.audit["duration_s"] == pytest.approx(1.0e-9)
    assert advanced.tip_process_state["by_branch"][owner] != \
        accepted_state.tip_process_state["by_branch"][owner]


def _controlled_phase_state(base, phase, *, child=False):
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    from arrhenius_fracture.voiding_v5 import Cavity2D

    root = replace(base.crack_network.branches[0], status="arrested")
    branches = (root,)
    tip_process = dict(base.tip_process_state)
    by_branch = dict(tip_process["by_branch"])
    owner_by_front = {}
    if child:
        child_id = "void-front-1"
        child_branch = replace(
            base.crack_network.branches[0], branch_id=child_id,
            parent_branch_id=ROOT_BRANCH_ID, generation=1,
            initiation_event=1, status="active",
            local_state={"front_engine_state_owner": child_id})
        branches = (root, child_branch)
        by_branch[child_id] = by_branch[ROOT_BRANCH_ID]
        owner_by_front[child_id] = child_id
    tip_process["by_branch"] = by_branch
    tip_process["owner_by_front"] = owner_by_front
    tip_process["active_branch_id"] = "void-front-1" if child else None
    network = CrackNetworkState(
        branches=branches, primary_branch_id=ROOT_BRANCH_ID,
        branching_enabled=child)
    site = replace(base.void_state.sites[0], phase=phase)
    radius = 1.0e-6
    cavity = Cavity2D(
        cavity_id="void:"+site.site_id, parent_site_id=site.site_id,
        center_m=site.center_m, radius_m=radius,
        area_m2=math.pi*radius**2, inventory_area_m2=math.pi*radius**2,
        phase=phase)
    configuration = replace(
        base.void_state.embryo_growth_configuration,
        vacancy_clustering_birth_enabled=False,
        direct_interaction_birth_enabled=False,
        cavity_surface_loop_emission_enabled=False)
    hazards = tuple(
        replace(item, action=0.0, residual_action=0.0,
                current_threshold_action=(1.0 if index == 0 else 1.0e300))
        for index, item in enumerate(base.competition.hazard_states))
    return replace(
        base, crack_network=network, tip_process_state=tip_process,
        competition=replace(base.competition, hazard_states=hazards),
        void_state=replace(
            base.void_state, sites=(site,), cavities=(cavity,),
            embryo_growth_states=(), embryo_growth_configuration=configuration))


def _install_controlled_downstream_owner(monkeypatch):
    import arrhenius_fracture.global_physical_scheduler_v1 as scheduler
    import arrhenius_fracture.voiding_production_v5 as production

    def preview(state, *, continuation, maximum_duration_s, **_kwargs):
        hazard = state.competition.hazard_states[0]
        remaining = max(hazard.current_threshold_action-hazard.action, 0.0)
        elapsed = min(float(maximum_duration_s), remaining)
        advanced_action = hazard.action+elapsed
        advanced_hazard = replace(
            hazard, action=advanced_action,
            residual_action=advanced_action-math.floor(advanced_action+1.0e-14))
        advanced = replace(state, competition=replace(
            state.competition,
            hazard_states=(advanced_hazard,)+state.competition.hazard_states[1:]))
        crossing = remaining if remaining <= maximum_duration_s else math.inf
        event_ids = (("controlled-downstream",) if math.isfinite(crossing) else ())
        candidate_ids = ((hazard.candidate_id,) if event_ids else ())
        rows = ({"directional_drive_status": "AVAILABLE",
                 "K_effective_Pa_sqrt_m": 0.0},)
        inner = PhysicalEventPreview(
            event_type="DIRECTIONAL_CLEAVAGE" if event_ids else "NO_DIRECTIONAL_EVENT",
            owner_kind="DirectionalCompetitionState", owner_id="void-front-1",
            crossing_time_s=crossing,
            accepted_state_fingerprint=complete_accepted_state_fingerprint(state),
            source_fingerprint="controlled-source", topology_fingerprint="controlled-topology",
            event_ids=event_ids, candidate_ids=candidate_ids,
            threshold_identity=((hazard.candidate_id, hazard.current_threshold_action),),
            rng_fingerprint="controlled-rng",
            commit_payload={"advanced_state": advanced, "audit": rows,
                            "elapsed_time_s": elapsed})
        return PhysicalEventPreview(
            event_type=("CHILD_CONTINUATION" if continuation else
                        "DOWNSTREAM_FIRST_PASSAGE") if event_ids else
                       "NO_DOWNSTREAM_EVENT",
            owner_kind="downstream_front_transaction", owner_id="void-front-1",
            crossing_time_s=crossing,
            accepted_state_fingerprint=complete_accepted_state_fingerprint(state),
            source_fingerprint="controlled-source", topology_fingerprint="controlled-topology",
            event_ids=event_ids, candidate_ids=candidate_ids,
            threshold_identity=inner.threshold_identity,
            rng_fingerprint="controlled-rng",
            commit_payload={
                "advanced_state": advanced, "audit": (),
                "directional_preview": inner, "continuation": continuation,
                "source": {"source_front_id": "void-front-1",
                           "front_load_rows": rows}})

    def transport(current, mechanical, *, duration_s, **_kwargs):
        transport_state = replace(
            current.vacancy_transport_state_v2,
            transport_time_s=current.vacancy_transport_state_v2.transport_time_s+duration_s,
            last_solver_audit={"duration_s": duration_s})
        return replace(mechanical, vacancy_transport_state_v2=transport_state), {
            "elapsed_time_s": duration_s}

    committed_front_process = []

    def commit(state, *, _precompleted_front_process=None, **_kwargs):
        committed_front_process.append(_precompleted_front_process)
        return state, {}, None, None

    monkeypatch.setattr(scheduler, "preview_downstream_first_passage", preview)
    monkeypatch.setattr(production, "advance_production_vacancy_transport_interval", transport)
    monkeypatch.setattr(production, "downstream_front_transaction", commit)
    return committed_front_process


@pytest.mark.parametrize("child", [False, True])
def test_nonwinning_downstream_owner_accumulates_across_scheduler_calls(
        accepted_state, monkeypatch, child):
    phase = VoidPhase.DOWNSTREAM_FRONT_ACTIVE if child else VoidPhase.CONNECTED_VOID
    state = _controlled_phase_state(accepted_state, phase, child=child)
    committed_front_process = _install_controlled_downstream_owner(monkeypatch)
    actions = []
    policies = []
    selected = []
    for duration in (0.25, 0.25, 0.5):
        state, audit = advance_global_physical_interval(
            state, state, duration, {
                "kind": "hold", "temperature_K": 900.0,
                "maximum_events": 4,
                "stress_tensor_Pa": np.zeros((2, 2)),
            })
        actions.append(state.competition.hazard_states[0].action)
        policies.extend(event["front_process_load_policy"] for event in audit["events"])
        selected.extend(event["selected_event"] for event in audit["events"]
                        if event["selected_event"] is not None)
        assert all(event["directional_owner_advance"] is not None
                   for event in audit["events"])
    assert actions == pytest.approx((0.25, 0.5, 1.0), abs=2.0e-15)
    expected = "CANONICAL_DOWNSTREAM_PROVIDER" if child else \
        "EXPLICIT_COMPLETE_STATE_BOUND_MAP"
    assert set(policies) == {expected}
    assert selected == (["CHILD_CONTINUATION"] if child else
                        ["DOWNSTREAM_FIRST_PASSAGE"])
    assert len(committed_front_process) == 1
    assert (committed_front_process[0] is not None) is child


def test_actual_connected_owner_accumulates_through_restart_then_crosses_once(tmp_path):
    from pathlib import Path
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint

    checkpoint = Path(
        "artifacts/restricted_production_closeout/vacancy_connected_source_qualified.json")
    if not checkpoint.exists():
        pytest.skip("qualified connected-state fixture is unavailable")
    state = restore_checkpoint(checkpoint)
    baseline = preview_downstream_first_passage(
        state, temperature_K=900.0, maximum_duration_s=2.5e-7)
    directional_rows = baseline.commit_payload[
        "directional_preview"].commit_payload["audit"]
    positive = [row for row in directional_rows
                if float(row["effective_rate_s"]) > 0.0]
    assert len(positive) == 1
    hazard = state.competition.hazard_states[0]
    remaining = hazard.current_threshold_action-hazard.action
    scale = remaining/(float(positive[0]["effective_rate_s"])*1.0e-6)
    initial_action = hazard.action
    actions = []
    selected = []
    for index, duration in enumerate((2.5e-7, 2.5e-7, 5.0e-7)):
        state, audit = advance_global_physical_interval(
            state, state, duration, {
                "kind": "hold", "temperature_K": 900.0,
                "maximum_events": 8,
                "functional_test_directional_rate_scale": scale,
                "test_only_functional_qualification": True,
            })
        actions.append(state.competition.hazard_states[0].action)
        selected.extend(row["selected_event"] for row in audit["events"]
                        if row["selected_event"] is not None)
        if index == 0:
            restart = tmp_path/"actual-connected-clock.json"
            write_checkpoint(state, restart, compression="gzip")
            state = restore_checkpoint(restart)
    assert actions[0] == pytest.approx(initial_action+0.25*remaining, rel=2e-12)
    assert actions[1] == pytest.approx(initial_action+0.50*remaining, rel=2e-12)
    assert selected == ["DOWNSTREAM_FIRST_PASSAGE"]
    assert state.crack_network.active_tip_ids == ("void-front-1",)


def test_void_lifecycle_previews_enforce_phase_eligibility(accepted_state):
    site = accepted_state.void_state.sites[0]
    unavailable = preview_stabilization(accepted_state, site.site_id, 2.0)
    assert not unavailable.eligible
    embryo = replace(site, phase=VoidPhase.EMBRYO)
    state = replace(accepted_state, void_state=replace(
        accepted_state.void_state, sites=(embryo,),
    ))
    stabilize = preview_stabilization(state, site.site_id, 2.0)
    heal = preview_healing(state, site.site_id, 4.0)
    assert math.isclose(stabilize.crossing_time_s, site.stabilization.threshold/2.0)
    assert math.isclose(heal.crossing_time_s, site.healing.threshold/4.0)


def test_vacancy_birth_locator_split_is_pure_and_equivalent(accepted_state):
    transport = accepted_state.vacancy_transport_state_v2
    site = replace(
        transport.site_states[0], capture_rate_area_per_s=1.0e-12,
        required_embryo_inventory_area_m2=1.0e-30,
        birth_barrier_J=0.0, birth_stress_work_J=0.0,
        threshold=transport.site_states[0].birth_action+1.0e-40,
    )
    concentration = np.full(transport.transport_mesh.ne, 1.0e-5)
    ledgers = dict(transport.ledgers)
    ledgers["initial_owned_inventory_area_m2"] = float(np.sum(
        transport.transport_mesh.area_e*concentration
    ))
    source = replace(transport, vacancy_fraction_gp=concentration,
                     ledgers=ledgers, site_states=(site,))
    zeros = np.zeros(source.transport_mesh.ne)
    kwargs = dict(
        row=source.parameter_row, maximum_duration_s=1.0,
        accepted_equivalent_plastic_strain_rate_s=zeros,
        accepted_dislocation_density_m2=zeros,
        accepted_mean_stress_Pa=zeros,
    )
    time, advanced = preview_site_inventory_event(
        source, source.transport_mesh, source.surface_registry, **kwargs,
    )
    assert time is not None
    assert source.site_states[0].hit_count == site.hit_count
    assert advanced.site_states[0].hit_count == site.hit_count
    committed = commit_site_inventory_hit(source, advanced, expected_site_id=site.site_id)
    fused_time, fused = locate_site_inventory_event(
        source, source.transport_mesh, source.surface_registry, **kwargs,
    )
    assert fused_time == time
    assert fused.site_states == committed.site_states


def test_declared_controlled_winners_use_one_deterministic_selector():
    fingerprints = ("accepted", "source", "topology", "rng")
    event_types = (
        "CLEAVAGE_WINS", "VACANCY_BIRTH_WINS",
        "STABILIZATION_WINS", "HEALING_WINS", "GROWTH_PROMOTION_WINS",
        "LIGAMENT_RUPTURE_WINS", "DOWNSTREAM_FIRST_PASSAGE_WINS",
    )
    for winner_type in event_types:
        previews = tuple(PhysicalEventPreview(
            event_type=kind, owner_kind="controlled-owner", owner_id=kind,
            crossing_time_s=1.0 if kind == winner_type else 2.0,
            accepted_state_fingerprint=fingerprints[0], source_fingerprint=fingerprints[1],
            topology_fingerprint=fingerprints[2], event_ids=(kind,), candidate_ids=(kind,),
            threshold_identity=(("controlled_action", 1.0),),
            rng_fingerprint=fingerprints[3], commit_payload={},
        ) for kind in event_types)
        selected, tie = select_physical_event(previews)
        assert selected.event_type == winner_type
        assert tie == (selected,)


def test_unsupported_cross_owner_tie_fails_closed():
    common = dict(
        crossing_time_s=1.0, accepted_state_fingerprint="accepted",
        source_fingerprint="source", topology_fingerprint="topology",
        candidate_ids=(), threshold_identity=(), rng_fingerprint="rng",
        commit_payload={},
    )
    previews = (
        PhysicalEventPreview(event_type="CLEAVAGE", owner_kind="front", owner_id="a",
                             event_ids=("a",), **common),
        PhysicalEventPreview(event_type="VACANCY_BIRTH", owner_kind="site", owner_id="b",
                             event_ids=("b",), **common),
    )
    with pytest.raises(RuntimeError, match="unsupported simultaneous"):
        select_physical_event(previews)


def test_growth_packet_boundary_tie_defers_to_physical_event():
    common = dict(
        crossing_time_s=1.0, accepted_state_fingerprint="accepted",
        source_fingerprint="source", topology_fingerprint="topology",
        candidate_ids=(), threshold_identity=(), rng_fingerprint="rng",
        commit_payload={},
    )
    physical = PhysicalEventPreview(
        event_type="GROWTH_PROMOTION", owner_kind="EmbryoGrowthProcessOwner",
        owner_id="site", event_ids=("promotion",), **common)
    packet = PhysicalEventPreview(
        event_type="GROWTH_PACKET_BOUNDARY", owner_kind="GlobalPhysicalScheduler",
        owner_id="growth-packet", event_ids=("packet",), **common)
    winner, tie = select_physical_event((packet, physical))
    assert winner is physical
    assert {item.event_type for item in tie} == {
        "GROWTH_PACKET_BOUNDARY", "GROWTH_PROMOTION"}


def test_same_embryo_stabilization_healing_tie_uses_lifecycle_owner_policy():
    common = dict(
        crossing_time_s=1.0, accepted_state_fingerprint="accepted",
        source_fingerprint="source", topology_fingerprint="topology",
        candidate_ids=("site-1",), threshold_identity=(), rng_fingerprint="rng",
        commit_payload={}, owner_kind="ProductionVoidState", owner_id="site-1",
    )
    stabilization = PhysicalEventPreview(
        event_type="STABILIZATION", event_ids=("stabilization:site-1",), **common)
    healing = PhysicalEventPreview(
        event_type="HEALING", event_ids=("healing:site-1",), **common)

    winner, tie = select_physical_event((stabilization, healing))

    assert winner is healing
    assert {item.event_type for item in tie} == {"STABILIZATION", "HEALING"}


def test_ramp_growth_packet_uses_interior_mechanics_and_consistent_plastic_rate():
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import (
        CavityEmissionParameterRowV1, cavity_surface_emission_rates,
        source_local_fem_tensors,
    )
    from arrhenius_fracture.vacancy_coupling_v1 import (
        equivalent_plastic_strain_rate_s,
    )
    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        low_count_mu_max=1.0e-3, continuum_mu_min=1.0e-2)
    state = production_growth_state(row, 91)
    site = replace(state.void_state.sites[0],
                   phase=VoidPhase.STABLE_SUBGRID_VOID)
    state = replace(state, void_state=replace(state.void_state, sites=(site,)))
    growth = state.void_state.embryo_growth_states[0]
    tensors, zz = source_local_fem_tensors(state, growth)
    rates, _ = cavity_surface_emission_rates(
        growth, stress_tensor_Pa=tensors, sigma_zz_Pa=zz,
        temperature_K=1200.0)
    duration = 2.0/float(rates[0])
    delta_ep = np.zeros_like(state.ep_gp)
    delta_ep[0] = 1.0e-6
    callback_fractions = []

    def mechanics_at_fraction(fraction):
        callback_fractions.append(float(fraction))
        return replace(state, ep_gp=np.asarray(state.ep_gp)+fraction*delta_ep)

    endpoint = mechanics_at_fraction(1.0)
    advanced, audit = advance_global_physical_interval(
        state, endpoint, duration_s=duration,
        load_history={
            "kind": "ramp", "temperature_K": 1200.0,
            "mechanical_state_at_fraction": mechanics_at_fraction,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })
    first = audit["events"][0]
    assert first["selected_event"] == "GROWTH_PACKET_BOUNDARY"
    assert first["growth_packet_global_scheduler_boundary"]
    fraction = first["ramp_event_time_s"]/duration
    assert 0.0 < fraction < 1.0
    assert audit["ramp_action_integration"] == \
        "FULL_VERSUS_TWO_HALF_STATE_REPLAY"
    assert audit["maximum_accepted_error"] <= \
        audit["ramp_action_relative_tolerance"]
    assert first["ramp_quadrature_fraction"] in callback_fractions
    expected_rate = np.max(equivalent_plastic_strain_rate_s(
        state.ep_gp, endpoint.ep_gp, duration))
    observed_rate = first["transport"][
        "accepted_plastic_increment_rate_max_s"]
    assert 0.0 < observed_rate < expected_rate
    # The accepted endpoint additionally contains physical loop-deposition
    # plasticity, so it need not equal the prescribed ramp field bitwise.
    assert np.max(np.abs(np.asarray(advanced.ep_gp)-np.asarray(endpoint.ep_gp))) > 0.0


def _controlled_lifecycle_ramp_state(*, stabilization_threshold,
                                     healing_threshold=1.0e300):
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
    from arrhenius_fracture.voiding_v5 import HazardClock

    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    state = production_growth_state(row, 193)
    site = replace(
        state.void_state.sites[0],
        stabilization=HazardClock(0.0, stabilization_threshold),
        healing=HazardClock(0.0, healing_threshold),
    )
    growth = replace(
        state.void_state.embryo_growth_states[0],
        cavity_emission_state=replace(
            state.void_state.embryo_growth_states[0].cavity_emission_state,
            enabled=False,
        ),
    )
    configuration = replace(
        state.void_state.embryo_growth_configuration,
        vacancy_clustering_birth_enabled=False,
        direct_interaction_birth_enabled=False,
        cavity_surface_loop_emission_enabled=False,
    )
    hazards = tuple(replace(item, current_threshold_action=1.0e300)
                    for item in state.competition.hazard_states)
    state = replace(
        state,
        competition=replace(state.competition, hazard_states=hazards),
        void_state=replace(
            state.void_state, sites=(site,), embryo_growth_states=(growth,),
            embryo_growth_configuration=configuration,
        ),
    )
    return state


def _install_controlled_lifecycle_ramp(
        monkeypatch, state, stabilization_rate, healing_rate=lambda _fraction: 0.0):
    import arrhenius_fracture.vacancy_transport_phase_field_v2 as transport_module
    import arrhenius_fracture.voiding_v5 as voiding_module

    element_count = state.mesh.ne

    def stress3d(trial):
        fraction = float(trial.stored_energy_J_per_m)
        return {
            "sigma_xx_Pa": np.full(element_count, fraction),
            "sigma_yy_Pa": np.zeros(element_count),
            "sigma_xy_Pa": np.zeros(element_count),
            "sigma_zz_Pa": np.zeros(element_count),
            "mean_stress_Pa": np.zeros(element_count),
            "source": "CONTROLLED_RAMP_TEST",
        }

    def rates(_config, *, stress_tensor_Pa, **_kwargs):
        fraction = float(np.asarray(stress_tensor_Pa)[0, 0])
        return {"stabilization_s": float(stabilization_rate(fraction)),
                "healing_s": float(healing_rate(fraction))}

    monkeypatch.setattr(transport_module, "accepted_elastoplastic_stress_3d", stress3d)
    monkeypatch.setattr(voiding_module, "arrhenius_rates", rates)


def test_ramp_integrates_linear_lifecycle_action_and_consumes_event_once(monkeypatch):
    state = _controlled_lifecycle_ramp_state(stabilization_threshold=0.75)
    _install_controlled_lifecycle_ramp(monkeypatch, state, lambda fraction: 1.0+fraction)

    def mechanics(fraction, _current=None):
        return replace(state, stored_energy_J_per_m=float(fraction))

    before = complete_accepted_state_fingerprint(state)
    advanced, audit = advance_global_physical_interval(
        state, mechanics(1.0), 1.0, {
            "kind": "ramp", "temperature_K": 1200.0,
            "mechanical_state_at_fraction": mechanics,
            "state_coupled_mechanical_state_at_fraction": mechanics,
            "ramp_action_relative_tolerance": 1.0e-3,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })
    expected = math.sqrt(2.5)-1.0
    stabilization_events = tuple(
        event for event in audit["events"]
        if event["selected_event"] == "STABILIZATION")
    assert len(stabilization_events) == 1
    assert stabilization_events[0]["ramp_event_time_s"] == pytest.approx(
        expected, rel=2.0e-3)
    assert advanced.void_state.sites[0].phase == VoidPhase.STABLE_SUBGRID_VOID
    assert complete_accepted_state_fingerprint(state) == before


def test_two_embryos_own_independent_lifecycle_clocks(monkeypatch):
    """An explicit-cavity count limit must not freeze a second subgrid owner."""
    state = _controlled_lifecycle_ramp_state(stabilization_threshold=0.10)
    first_site = state.void_state.sites[0]
    second_site = replace(
        first_site, site_id="site-2", center_m=(7.5e-4, 0.0),
        stabilization=replace(first_site.stabilization, threshold=0.05),
    )
    first_growth = state.void_state.embryo_growth_states[0]
    second_growth = replace(
        first_growth, site_id="site-2", cavity_id=None,
        lineage=("EMBRYO_BIRTH", "INDEPENDENT_SITE_2"),
    )
    first_transport_site = state.vacancy_transport_state_v2.site_states[0]
    second_transport_site = replace(
        first_transport_site, site_id="site-2", center_xy_m=second_site.center_m,
        rng_fingerprint="controlled-two-embryo-site-2",
    )
    direct = state.void_state.dislocation_interaction_state
    first_direct_site = direct.sites[0]
    second_direct_site = replace(
        first_direct_site, site_id="site-2", center_xy_m=second_site.center_m,
        interaction_rng_identity="controlled-two-embryo-site-2",
    )
    state = replace(
        state,
        vacancy_transport_state_v2=replace(
            state.vacancy_transport_state_v2,
            site_states=(first_transport_site, second_transport_site),
        ),
        void_state=replace(
            state.void_state,
            sites=(first_site, second_site),
            embryo_growth_states=(first_growth, second_growth),
            dislocation_interaction_state=replace(
                direct, sites=(first_direct_site, second_direct_site),
            ),
        ),
    )
    _install_controlled_lifecycle_ramp(monkeypatch, state, lambda _fraction: 1.0)

    before = complete_accepted_state_fingerprint(state)
    advanced, audit = advance_global_physical_interval(
        state, state, 0.20, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })

    stabilization_events = tuple(
        event for event in audit["events"]
        if event["selected_event"] == "STABILIZATION")
    assert tuple(event["selected_owner"][1] for event in stabilization_events) == (
        "site-2", "site-1")
    assert tuple(site.phase for site in advanced.void_state.sites) == (
        VoidPhase.STABLE_SUBGRID_VOID, VoidPhase.STABLE_SUBGRID_VOID)
    assert {cavity.parent_site_id for cavity in advanced.void_state.cavities} == {
        "site-1", "site-2"}
    assert complete_accepted_state_fingerprint(state) == before


def test_ramp_exponential_action_matches_independent_quadrature(monkeypatch):
    from scipy.integrate import quad
    from scipy.optimize import brentq

    threshold = 1.0
    state = _controlled_lifecycle_ramp_state(
        stabilization_threshold=threshold)
    _install_controlled_lifecycle_ramp(
        monkeypatch, state, lambda fraction: math.exp(2.0*fraction))

    def mechanics(fraction, _current=None):
        return replace(state, stored_energy_J_per_m=float(fraction))

    reference = brentq(
        lambda time: quad(lambda value: math.exp(2.0*value), 0.0, time,
                          epsabs=1.0e-13, epsrel=1.0e-13)[0]-threshold,
        0.0, 1.0, xtol=1.0e-13)
    _, audit = advance_global_physical_interval(
        state, mechanics(1.0), 1.0, {
            "kind": "ramp", "temperature_K": 1200.0,
            "mechanical_state_at_fraction": mechanics,
            "state_coupled_mechanical_state_at_fraction": mechanics,
            "ramp_action_relative_tolerance": 2.5e-4,
            "ramp_action_maximum_depth": 13,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })
    events = tuple(event for event in audit["events"]
                   if event["selected_event"] == "STABILIZATION")
    assert len(events) == 1
    assert events[0]["ramp_event_time_s"] == pytest.approx(reference, rel=7.5e-4)


def test_ramp_competing_owner_order_uses_accumulated_action(monkeypatch):
    state = _controlled_lifecycle_ramp_state(
        stabilization_threshold=0.9, healing_threshold=0.6)
    _install_controlled_lifecycle_ramp(
        monkeypatch, state,
        stabilization_rate=lambda fraction: 0.4+1.6*fraction,
        healing_rate=lambda _fraction: 1.0)

    def mechanics(fraction, _current=None):
        return replace(state, stored_energy_J_per_m=float(fraction))

    advanced, audit = advance_global_physical_interval(
        state, mechanics(1.0), 1.0, {
            "kind": "ramp", "temperature_K": 1200.0,
            "mechanical_state_at_fraction": mechanics,
            "state_coupled_mechanical_state_at_fraction": mechanics,
            "ramp_action_relative_tolerance": 1.0e-3,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })
    physical = tuple(event["selected_event"] for event in audit["events"]
                     if event["selected_event"] is not None)
    assert physical == ("HEALING",)
    assert advanced.void_state.sites[0].phase == VoidPhase.HEALED_SITE


def test_ramp_partition_and_checkpoint_restart_preserve_accumulated_action(
        monkeypatch, tmp_path):
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint

    state = _controlled_lifecycle_ramp_state(stabilization_threshold=1.0e300)
    _install_controlled_lifecycle_ramp(monkeypatch, state, lambda fraction: 1.0+fraction)

    def advance_partitioned(parts, *, checkpoint_at=None):
        current = state
        for index in range(parts):
            def mechanics(local_fraction, _current=None, index=index):
                fraction = (index+float(local_fraction))/parts
                return replace(state, stored_energy_J_per_m=fraction)
            current, _ = advance_global_physical_interval(
                current, mechanics(1.0), 1.0/parts, {
                    "kind": "ramp", "temperature_K": 1200.0,
                    "mechanical_state_at_fraction": mechanics,
                    "state_coupled_mechanical_state_at_fraction": mechanics,
                    "ramp_action_relative_tolerance": 1.0e-3,
                    "maximum_events": 64,
                    "diagnostic_freeze_front_process": True,
                })
            if checkpoint_at == index+1:
                path = tmp_path/f"ramp-{parts}-{index+1}.json"
                write_checkpoint(current, path)
                current = restore_checkpoint(path)
        return current

    actions = {}
    for parts in (1, 2, 4, 8, 16):
        result = advance_partitioned(parts)
        actions[parts] = result.void_state.sites[0].stabilization.accumulated
    restarted = advance_partitioned(2, checkpoint_at=1)
    assert all(value == pytest.approx(1.5, rel=2.0e-3)
               for value in actions.values())
    assert max(actions.values())-min(actions.values()) <= 2.0e-3
    assert restarted.void_state.sites[0].stabilization.accumulated == pytest.approx(
        actions[2], rel=0.0, abs=2.0e-12)


def test_waiting_stabilization_commits_atomically_at_nonlinear_radius_root():
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
    from arrhenius_fracture.voiding_v5 import HazardClock

    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        minimum_stable_radius_m=1.000001e-8,
        low_count_mu_max=1.0e-3, continuum_mu_min=1.0e-2)
    state = production_growth_state(row, 91)
    site = replace(
        state.void_state.sites[0],
        stabilization=HazardClock(1.0, 1.0),
        healing=HazardClock(0.0, 1.0e300))
    growth = replace(
        state.void_state.embryo_growth_states[0],
        stabilization_action_complete_waiting_for_radius=True)
    state = replace(state, void_state=replace(
        state.void_state, sites=(site,), embryo_growth_states=(growth,)))
    initial_fingerprint = complete_accepted_state_fingerprint(state)

    advanced, audit = advance_global_physical_interval(
        state, state, duration_s=1.0e-9,
        load_history={
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
        })

    final_site = advanced.void_state.sites[0]
    final_growth = advanced.void_state.embryo_growth_states[0]
    stable_events = tuple(
        event for event in advanced.void_state.event_history
        if event.get("event") in {
            "STABILIZED", "STABILIZED_AT_STABLE_RADIUS"})
    radius_events = tuple(
        event for event in audit["events"]
        if event["selected_event"] == "GROWTH_STABLE_RADIUS")
    assert final_site.phase == VoidPhase.STABLE_SUBGRID_VOID
    assert not final_growth.stabilization_action_complete_waiting_for_radius
    assert len(advanced.void_state.cavities) == 1
    assert len(stable_events) == 1
    assert len(radius_events) == 1
    assert radius_events[0]["radius_transition_verified"]
    assert radius_events[0]["verified_radius_m"] == pytest.approx(
        row.minimum_stable_radius_m, rel=1.0e-12)
    assert audit["nonlinear_radius_replay_total"] > 0
    assert all(event["elapsed_time_s"] > 0.0 for event in audit["events"])
    assert complete_accepted_state_fingerprint(state) == initial_fingerprint

    for injected_stage in (
        "radius_root_after_transport",
        "growth_packet_before_equilibrium",
        "growth_packet_committed",
        "radius_root_overshoot",
        "radius_root_before_topology_publication",
        "radius_transition_commit",
    ):
        def fail(stage, _state, expected=injected_stage):
            if stage == expected:
                raise RuntimeError("injected:"+stage)

        with pytest.raises(RuntimeError, match="injected:"+injected_stage):
            advance_global_physical_interval(
                state, state, duration_s=1.0e-9,
                load_history={
                    "kind": "hold", "temperature_K": 1200.0,
                    "maximum_events": 64, "failure_injector": fail,
                    "diagnostic_freeze_front_process": True,
                })
        assert complete_accepted_state_fingerprint(state) == initial_fingerprint


def test_real_cavity_loop_event_is_committed_by_global_scheduler_without_area_supply():
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import (
        CavityEmissionParameterRowV1, loop_area_kinematics,
        make_cavity_loop_event_preview,
    )
    from arrhenius_fracture.voiding_v5 import create_subgrid_cavity

    # A fixed functional row keeps the real emission source in low-count mode;
    # it does not alter any production material row or inject cavity area.
    row = replace(
        CavityEmissionParameterRowV1(), row_id="functional-real-loop-v1",
        source_orientations_rad=(0.0,), attempt_frequency_s=1.0e4)
    state = production_growth_state(row, 901)
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    void_state = create_subgrid_cavity(
        replace(state.void_state, sites=(site,)), site.site_id, 1.0e-8)
    state = replace(state, void_state=void_state)
    growth_before = state.void_state.embryo_growth_states[0]
    preview = make_cavity_loop_event_preview(
        state, growth_before, temperature_K=1200.0, maximum_duration_s=30.0)
    assert preview.event_type == "CAVITY_LOOP_EMISSION"

    advanced, audit = advance_global_physical_interval(
        state, state, preview.crossing_time_s, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 20, "promotion_radius_m": 1.0e-6,
            "diagnostic_freeze_front_process": True,
            "functional_test_growth_supply_area_rate_m2_s": 0.0,
            "test_only_functional_qualification": True,
        })
    growth_after = advanced.void_state.embryo_growth_states[0]
    selected = tuple(event for event in audit["events"]
                     if event["selected_event"] == "CAVITY_LOOP_EMISSION")
    assert len(selected) == 1
    assert growth_after.cavity_emission_state.cumulative_loop_count_effective == 1.0
    expected_area = loop_area_kinematics(row, growth_before.radius_m)["selected_area_m2"]
    assert growth_after.area_m2-growth_before.area_m2 == pytest.approx(expected_area)
    assert growth_after.radius_m > growth_before.radius_m
    growth_audit = selected[0]["embryo_growth"][0]
    packet = growth_audit["packets"][0]
    assert packet["mechanical_deposition"]["deposited_loop_count"] == 1.0
    assert growth_audit["equilibrated_after_every_packet"] is True


def test_five_percent_stable_radius_root_and_healing_preemption():
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
    from arrhenius_fracture.voiding_v5 import HazardClock

    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        minimum_stable_radius_m=1.05e-8)
    state = production_growth_state(row, 317)
    growth = state.void_state.embryo_growth_states[0]
    growth = replace(
        growth,
        stabilization_action_complete_waiting_for_radius=True,
        cavity_emission_state=replace(growth.cavity_emission_state, enabled=False))
    site = replace(
        state.void_state.sites[0],
        stabilization=HazardClock(1.0, 1.0),
        healing=HazardClock(0.0, 1.0e300))
    config = replace(
        state.void_state.embryo_growth_configuration,
        cavity_surface_loop_emission_enabled=False,
        vacancy_clustering_birth_enabled=False,
        direct_interaction_birth_enabled=False)
    state = replace(state, void_state=replace(
        state.void_state, sites=(site,), embryo_growth_states=(growth,),
        embryo_growth_configuration=config))
    before = complete_accepted_state_fingerprint(state)
    advanced, audit = advance_global_physical_interval(
        state, state, 0.04, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
            "functional_test_growth_supply_area_rate_m2_s": 1.0e-15,
            "test_only_functional_qualification": True,
        })
    roots = tuple(event for event in audit["events"]
                  if event["selected_event"] == "GROWTH_STABLE_RADIUS")
    assert len(roots) == 1 and roots[0]["radius_transition_verified"]
    assert roots[0]["verified_radius_m"] == pytest.approx(1.05e-8, rel=1.0e-12)
    assert advanced.void_state.sites[0].phase == VoidPhase.STABLE_SUBGRID_VOID
    assert complete_accepted_state_fingerprint(state) == before

    current_radius = advanced.void_state.embryo_growth_states[0].radius_m
    interval_target = 1.2e-8
    interval_supply = 1.0e-14
    interval_duration = math.pi*(interval_target**2-current_radius**2)/interval_supply
    interval_grown, interval_audit = advance_global_physical_interval(
        advanced, advanced, interval_duration, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64, "promotion_radius_m": 1.0e-6,
            "diagnostic_freeze_front_process": True,
            "functional_test_growth_supply_area_rate_m2_s": interval_supply,
            "test_only_functional_qualification": True,
        })
    assert interval_grown.void_state.embryo_growth_states[0].radius_m == \
        pytest.approx(interval_target, rel=2.0e-10)
    assert interval_grown.void_state.sites[0].phase == VoidPhase.STABLE_SUBGRID_VOID
    assert interval_audit["advanced_duration_s"] == pytest.approx(interval_duration)

    healing_site = replace(site, healing=HazardClock(0.0, 1.0e-3))
    healing_state = replace(state, void_state=replace(
        state.void_state, sites=(healing_site,)))
    healed, healing_audit = advance_global_physical_interval(
        healing_state, healing_state, 0.04, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64,
            "diagnostic_freeze_front_process": True,
            "functional_test_growth_supply_area_rate_m2_s": 1.0e-15,
            "test_only_functional_qualification": True,
        })
    selected = tuple(event["selected_event"] for event in healing_audit["events"]
                     if event["selected_event"] is not None)
    assert selected[0] == "HEALING"
    assert "GROWTH_STABLE_RADIUS" not in selected
    assert healed.void_state.sites[0].phase == VoidPhase.HEALED_SITE


def test_directional_cleavage_preempts_stable_radius_and_rolls_back_before_publication(monkeypatch):
    from scripts.qualify_embryo_to_large_void_growth import production_growth_state
    from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
    from arrhenius_fracture.voiding_production_v5 import directional_clock_rates
    from arrhenius_fracture.voiding_v5 import HazardClock

    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        minimum_stable_radius_m=1.05e-8)
    state = production_growth_state(row, 317)
    growth = replace(
        state.void_state.embryo_growth_states[0],
        stabilization_action_complete_waiting_for_radius=True,
        cavity_emission_state=replace(
            state.void_state.embryo_growth_states[0].cavity_emission_state,
            enabled=False))
    site = replace(
        state.void_state.sites[0],
        stabilization=HazardClock(1.0, 1.0),
        healing=HazardClock(0.0, 1.0e300))
    configuration = replace(
        state.void_state.embryo_growth_configuration,
        cavity_surface_loop_emission_enabled=False,
        vacancy_clustering_birth_enabled=False,
        direct_interaction_birth_enabled=False)
    state = replace(state, void_state=replace(
        state.void_state, sites=(site,), embryo_growth_states=(growth,),
        embryo_growth_configuration=configuration))

    tensor, _ = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    rate = directional_clock_rates(
        state, tensor, temperature_K=1200.0)[0]["effective_rate_s"]
    assert rate > 0.0
    cleavage_time = 1.0e-4
    hazard = replace(
        state.competition.hazard_states[0],
        current_threshold_action=rate*cleavage_time,
        threshold_process="unit_deterministic")
    state = replace(
        state, competition=replace(state.competition, hazard_states=(hazard,)))
    before = complete_accepted_state_fingerprint(state)
    observed = []

    def fail(stage, _trial):
        observed.append(stage)
        if stage == "directional_cleavage_before_topology_publication":
            raise RuntimeError("injected:"+stage)

    with pytest.raises(
            RuntimeError,
            match="injected:directional_cleavage_before_topology_publication"):
        advance_global_physical_interval(
            state, state, 2.0e-4, {
                "kind": "hold", "temperature_K": 1200.0,
                "maximum_events": 64,
                "diagnostic_freeze_front_process": True,
                "functional_test_growth_supply_area_rate_m2_s": 1.0e-15,
                "test_only_functional_qualification": True,
                "failure_injector": fail,
            })
    assert "directional_cleavage_before_topology_publication" in observed
    assert complete_accepted_state_fingerprint(state) == before

    import arrhenius_fracture.v12_production_driver as production_driver
    captured = {}
    def capture_physical_event(current, endpoint, **kwargs):
        captured.update(kwargs)
        return current, {
            "energy_release_J_per_m": 2.0,
            "hazard_dissipation_J_per_m": 1.0,
            "energy_margin_J_per_m": 1.0,
        }
    monkeypatch.setattr(production_driver, "execute_event", capture_physical_event)
    _, committed = advance_global_physical_interval(
        state, state, 2.0e-4, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_transactions_per_call": 1,
            "diagnostic_freeze_front_process": True,
        })
    assert captured["physical_context"].hazard_barrier_J > 0.0
    assert captured["physical_context"].cooperative_hits > 0.0
    event = next(item for item in committed["events"]
                 if item["selected_event"] == "DIRECTIONAL_CLEAVAGE")
    assert event["topology_transaction"]["hazard_dissipation_J_per_m"] == 1.0


def test_nonlinear_promotion_root_commits_explicit_geometry_and_fails_loop_closed():
    from arrhenius_fracture.embryo_growth_v1 import (
        CavityEmissionParameterRowV1, initialize_embryo_growth_state,
    )
    from arrhenius_fracture.voiding_production_v5 import (
        build_production_void_state, remesh_cavity,
    )
    from arrhenius_fracture.voiding_v5 import create_subgrid_cavity, promote_cavity

    state, hole = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1200.0)
    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        low_count_mu_max=1.0e-6, continuum_mu_min=1.0e-5,
        depletion_per_loop=1.0e-12, maximum_packet_area_fraction=0.5)
    growth = initialize_embryo_growth_state(
        site_id="site-1", radius_m=4.9e-5, row=row, emission_enabled=True)
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    voids = replace(
        state.void_state, sites=(site,), embryo_growth_states=(growth,),
        embryo_growth_configuration=replace(
            state.void_state.embryo_growth_configuration,
            vacancy_clustering_birth_enabled=False,
            direct_interaction_birth_enabled=False,
            cavity_surface_loop_emission_enabled=True))
    voids = create_subgrid_cavity(voids, site.site_id, growth.radius_m)
    state = replace(state, void_state=voids)
    before = complete_accepted_state_fingerprint(state)

    def commit_promotion(live, _preview):
        promoted = promote_cavity(
            live.void_state, live.void_state.cavities[0].cavity_id, 5.0e-5)
        promoted_growth = promoted.embryo_growth_states[0]
        promoted_growth = replace(
            promoted_growth,
            cavity_emission_state=replace(
                promoted_growth.cavity_emission_state, enabled=False))
        promoted = replace(
            promoted, embryo_growth_states=(promoted_growth,),
            embryo_growth_configuration=replace(
                promoted.embryo_growth_configuration,
                cavity_surface_loop_emission_enabled=False))
        return remesh_cavity(live, hole, promoted, "scheduler-promotion")

    advanced, audit = advance_global_physical_interval(
        state, state, 5.0e-6, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 128, "promotion_radius_m": 5.0e-5,
            "diagnostic_freeze_front_process": True,
            "commit_growth_promotion": commit_promotion,
        })
    events = tuple(event for event in audit["events"]
                   if event["selected_event"] == "GROWTH_PROMOTION")
    assert len(events) == 1 and events[0]["radius_transition_verified"]
    assert events[0]["verified_radius_m"] == pytest.approx(5.0e-5, rel=1.0e-12)
    assert advanced.void_state.cavities[0].phase == VoidPhase.RESOLVED_VOID
    assert advanced.void_state.embryo_growth_states[0].representation == \
        "ACCEPTED_FEM_CAVITY_SURFACE"
    assert not advanced.void_state.embryo_growth_states[0].cavity_emission_state.enabled
    assert any(face.kind.value == "ACTIVE_CAVITY_SURFACE"
               for face in advanced.vacancy_transport_state_v2.surface_registry.faces)
    assert audit["nonlinear_radius_replay_total"] > 0
    assert complete_accepted_state_fingerprint(state) == before


def test_resolved_remesh_radius_is_state_owned_and_published_at_exact_root():
    from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
    from arrhenius_fracture.embryo_growth_v1 import (
        RESOLVED, initialize_embryo_growth_state,
    )
    from arrhenius_fracture.voiding_production_v5 import (
        _grow_hole_boundary, build_production_void_state, remesh_cavity,
    )
    from arrhenius_fracture.voiding_v5 import Cavity2D

    state, hole = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1800.0)
    site = state.void_state.sites[0]
    radius = hole.radius_m
    cavity = Cavity2D(
        "void:"+site.site_id, site.site_id, hole.center_m, radius,
        math.pi*radius**2, math.pi*radius**2, VoidPhase.RESOLVED_VOID)
    growth = initialize_embryo_growth_state(
        site_id=site.site_id, radius_m=radius, initial_area_m2=cavity.area_m2,
        representation=RESOLVED, cavity_id=cavity.cavity_id,
        emission_enabled=False)
    voids = replace(
        state.void_state,
        sites=(replace(site, phase=VoidPhase.RESOLVED_VOID),),
        cavities=(cavity,), embryo_growth_states=(growth,),
        embryo_growth_configuration=replace(
            state.void_state.embryo_growth_configuration,
            vacancy_clustering_birth_enabled=False,
            direct_interaction_birth_enabled=False,
            cavity_surface_loop_emission_enabled=False))
    state = remesh_cavity(state, hole, voids, "test-promotion")
    before = complete_accepted_state_fingerprint(state)
    published = []

    def commit_remesh(live, before_growth, after_growth):
        published.append((before_growth.radius_m, after_growth.radius_m))
        grown_hole = _grow_hole_boundary(
            hole, after_growth.radius_m,
            crack_path_m=live.crack_network.branch(ROOT_BRANCH_ID).path)
        return remesh_cavity(
            live, grown_hole, live.void_state, "scheduler-resolved-remesh")

    with pytest.raises(RuntimeError, match="restricted to labeled"):
        advance_global_physical_interval(
            state, state, 0.01, {
                "kind": "hold", "temperature_K": 1800.0,
                "functional_test_growth_supply_area_rate_m2_s": 1.0e-8,
            })
    assert complete_accepted_state_fingerprint(state) == before

    advanced, audit = advance_global_physical_interval(
        state, state, 0.055, {
            "kind": "hold", "temperature_K": 1800.0,
            "maximum_events": 64, "promotion_radius_m": radius,
            "diagnostic_freeze_front_process": True,
            "embryo_remesh_fraction": 0.25,
            "commit_embryo_growth_remesh": commit_remesh,
            "functional_test_growth_supply_area_rate_m2_s": 1.0e-8,
            "test_only_functional_qualification": True,
        })
    events = tuple(event for event in audit["events"]
                   if event["selected_event"] == "GROWTH_REMESH_CADENCE")
    assert len(events) == 1 and events[0]["radius_transition_verified"]
    assert len(published) == 1
    assert events[0]["verified_radius_m"] == pytest.approx(
        events[0]["verified_target_radius_m"], rel=1.0e-12)
    assert advanced.void_state.embryo_growth_states[0].last_remesh_radius_m == \
        pytest.approx(events[0]["verified_radius_m"], rel=1.0e-12)
    assert not np.array_equal(advanced.mesh.nodes, state.mesh.nodes)
    assert complete_accepted_state_fingerprint(state) == before
