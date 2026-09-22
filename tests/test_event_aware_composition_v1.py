from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_event_aware_coupled_interval,
    numerical_provider_dependency_identity,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


@pytest.fixture(scope="module")
def state():
    accepted, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0)
    return accepted


def _install_outer_controls(monkeypatch, *, second="complete", packet=False,
                            geometric=False, competing_owner=False,
                            first_veto=False):
    import arrhenius_fracture.bulk_plasticity_v104 as bulk
    import arrhenius_fracture.global_physical_scheduler_v1 as scheduler
    from arrhenius_fracture.v12_production_driver import PhysicalEnergyVeto

    def plastic(current, **_kwargs):
        return current, {"controlled": True}

    def global_interval(current, _mechanical, horizon, _history):
        marker = int(current.event_counters.get("controlled_marker", 0))
        if marker == 0:
            if first_veto and horizon >= 0.5-1.0e-12:
                raise PhysicalEnergyVeto(current, {
                    "elapsed_time_s":0.5,"selected_event":"DIRECTIONAL_CLEAVAGE",
                    "selected_owner":["DirectionalCompetitionState","front-a"],
                    "event_ids":["cleave:a#1"],"candidate_ids":["cleave:a"],
                    "energy_release_J_per_m":0.25,
                    "hazard_dissipation_J_per_m":0.5,
                    "energy_margin_J_per_m":-0.25,
                })
            crossing = 1.0
            if horizon < crossing-1.0e-12:
                return current, {"advanced_duration_s": horizon, "events": [{
                    "selected_event": None,
                    "growth_packet_global_scheduler_boundary": False,
                }]}
            counters = dict(current.event_counters)
            counters["controlled_marker"] = 1
            boundary = replace(current, event_counters=counters)
            if geometric:
                branch = boundary.crack_network.branches[0]
                path = tuple(branch.path)+((branch.tip[0]+1.0e-9, branch.tip[1]),)
                advanced = replace(
                    branch, path=path,
                    orientation_history_rad=tuple(branch.orientation_history_rad)+(0.0,))
                boundary = replace(boundary, crack_network=replace(
                    boundary.crack_network, branches=(advanced,),
                    geometry_generation=boundary.crack_network.geometry_generation+1))
            chosen_owner = "owner-b" if competing_owner else "owner-a"
            event_type = None if packet else "CONTROLLED_LOOP"
            previews = ({"event_type":"CONTROLLED_LOOP", "owner_kind":"site",
                         "owner_id":"owner-a", "event_ids":["loop:a"],
                         "candidate_ids":["candidate:a"]},
                        {"event_type":"CONTROLLED_LOOP", "owner_kind":"site",
                         "owner_id":"owner-b", "event_ids":["loop:b"],
                         "candidate_ids":["candidate:b"]})
            return boundary, {"advanced_duration_s": crossing, "events": [{
                "selected_event": event_type,
                "selected_owner": None if packet else ["site", chosen_owner],
                "eligible_previews": previews,
                "elapsed_time_s": crossing,
                "growth_packet_global_scheduler_boundary": packet,
            }]}
        if second == "veto" and horizon >= 0.25-1.0e-12:
            counters = dict(current.event_counters)
            counters["controlled_veto_prefix"] = 1
            prefix = replace(current, event_counters=counters)
            raise PhysicalEnergyVeto(prefix, {
                "elapsed_time_s": 0.25,
                "selected_event": "DIRECTIONAL_CLEAVAGE",
                "selected_owner": ["DirectionalCompetitionState", "front-a"],
                "event_ids": ["cleave:a#1"], "candidate_ids": ["cleave:a"],
                "energy_release_J_per_m":0.25,
                "hazard_dissipation_J_per_m":0.5,
                "energy_margin_J_per_m":-0.25,
            })
        if second == "failure":
            raise RuntimeError("controlled remainder failure")
        return current, {"advanced_duration_s": horizon, "events": [{
            "selected_event": None,
            "growth_packet_global_scheduler_boundary": False,
        }]}

    monkeypatch.setattr(bulk, "accepted_bulk_load_step", plastic)
    monkeypatch.setattr(scheduler, "advance_global_physical_interval", global_interval)


def _directional(current):
    return np.zeros((2, 2)), 0.0, {"controlled": True}, ()


def test_fixed_topology_loop_and_packet_complete_without_rebuild(state, monkeypatch):
    for packet in (False, True):
        _install_outer_controls(monkeypatch, packet=packet)
        final, audit = advance_event_aware_coupled_interval(
            state, 2.0, temperature_K=900.0,
            directional_observation_provider=_directional)
        assert final.event_counters["controlled_marker"] == 1
        assert audit["consumed_duration_s"] == pytest.approx(2.0)
        assert audit["remaining_duration_s"] == 0.0
        assert audit["provider_rebuild_count"] == 0
        assert audit["native_topology_transition_count"] == 0
        assert audit["accepted_event_segments"][0]["decision"] == (
            "NUMERICAL_BOUNDARY" if packet else "ACCEPTED")

    root = replace(state.crack_network.branches[0], status="arrested")
    no_front = replace(state, crack_network=replace(
        state.crack_network, branches=(root,)))
    _install_outer_controls(monkeypatch)
    _, audit = advance_event_aware_coupled_interval(
        no_front, 2.0, temperature_K=900.0,
        directional_observation_provider=_directional)
    assert audit["remaining_duration_s"] == 0.0


def test_competing_same_type_owner_identity_is_bound(state, monkeypatch):
    _install_outer_controls(monkeypatch, competing_owner=True)
    _, audit = advance_event_aware_coupled_interval(
        state, 2.0, temperature_K=900.0,
        directional_observation_provider=_directional)
    segment = audit["accepted_event_segments"][0]
    assert segment["owner_id"] == "owner-b"
    assert segment["event_ids"] == ("loop:b",)
    assert segment["candidate_ids"] == ("candidate:b",)


def test_child_veto_propagates_total_time_and_preserves_accepted_geometry(
        state, monkeypatch):
    _install_outer_controls(monkeypatch, second="veto", geometric=True)
    before = complete_accepted_state_fingerprint(state)
    final, audit = advance_event_aware_coupled_interval(
        state, 2.0, temperature_K=900.0,
        directional_observation_provider=_directional)
    assert audit["outcome"] == "ENERGY_VETO"
    assert audit["terminal"] is True
    assert audit["consumed_duration_s"] == pytest.approx(1.25)
    assert audit["remaining_duration_s"] == pytest.approx(0.75)
    assert audit["first_event_consumed_duration_s"] == pytest.approx(1.0)
    assert final.event_counters["controlled_marker"] == 1
    assert final.event_counters["controlled_veto_prefix"] == 1
    assert complete_accepted_state_fingerprint(state) == before
    assert audit["native_topology_transition_count"] == 1
    assert audit["provider_rebuild_count"] == 0
    assert audit["energy_decision"]["energy_margin_J_per_m"] == -0.25


def test_immediate_veto_exposes_canonical_terminal_energy(state, monkeypatch):
    _install_outer_controls(monkeypatch, first_veto=True)
    final,audit=advance_event_aware_coupled_interval(
        state,2.0,temperature_K=900.0,
        directional_observation_provider=_directional)
    assert final is state
    assert audit["outcome"]=="ENERGY_VETO" and audit["terminal"]
    assert audit["consumed_duration_s"]==pytest.approx(0.5)
    assert audit["remaining_duration_s"]==pytest.approx(1.5)
    assert audit["energy_decision"]["energy_margin_J_per_m"]==-0.25


def test_dependency_identity_ignores_histories_but_detects_operator_owner(state):
    base = numerical_provider_dependency_identity(state)
    counters = dict(state.event_counters)
    counters["loop_count_only"] = 17
    history = replace(state, event_counters=counters,
                      rho_gp=np.asarray(state.rho_gp)+1.0)
    assert numerical_provider_dependency_identity(history) == base
    changed_damage = np.asarray(state.damage).copy()
    changed_damage[0] = min(float(changed_damage[0])+1.0e-6, 1.0)
    assert numerical_provider_dependency_identity(
        replace(state, damage=changed_damage)) != base


class _Manager:
    def __init__(self):
        self.generation = 0
        self.mechanics = object()
        self.transport = object()

    def snapshot_binding(self):
        return (self.generation, self.mechanics, self.transport)

    def restore_binding(self, snapshot):
        self.generation, self.mechanics, self.transport = snapshot

    def rebuild_after_native_commit(self, _before, _after, builder):
        self.mechanics, self.transport, build_audit = builder(_after, [_after])
        self.generation += 1
        return self.mechanics, self.transport, {
            "generation": self.generation, "basis": build_audit}


def test_reduced_rebuilds_only_on_dependency_change_and_rolls_back_failure(
        state, monkeypatch):
    _install_outer_controls(monkeypatch)
    # Evolving clocks/history use the existing providers and need no manager.
    _, fixed = advance_event_aware_coupled_interval(
        state, 2.0, temperature_K=900.0,
        directional_observation_provider=_directional,
        mechanical_equilibrium_provider=object(),
        transport_linear_solver_provider=object())
    assert fixed["provider_rebuild_count"] == 0

    def builder(_accepted, _training):
        return object(), object(), {"controlled": True}

    _install_outer_controls(monkeypatch, geometric=True)
    manager = _Manager()
    _, changed = advance_event_aware_coupled_interval(
        state, 2.0, temperature_K=900.0,
        directional_observation_provider=_directional,
        mechanical_equilibrium_provider=manager.mechanics,
        transport_linear_solver_provider=manager.transport,
        provider_transition_manager=manager, provider_builder=builder)
    assert changed["provider_rebuild_count"] == manager.generation == 1

    _install_outer_controls(monkeypatch, geometric=True, second="failure")
    manager = _Manager()
    initial = manager.snapshot_binding()
    with pytest.raises(RuntimeError, match="controlled remainder failure"):
        advance_event_aware_coupled_interval(
            state, 2.0, temperature_K=900.0,
            directional_observation_provider=_directional,
            mechanical_equilibrium_provider=manager.mechanics,
            transport_linear_solver_provider=manager.transport,
            provider_transition_manager=manager, provider_builder=builder)
    assert manager.snapshot_binding() == initial
