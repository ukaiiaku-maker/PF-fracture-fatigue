"""Narrow step-419 transaction and unchanged marked-birth regressions."""
from dataclasses import replace
import json
import pickle
import subprocess

import pytest

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint, write_branch_checkpoint
from arrhenius_fracture.directional_competition_v11 import construct_action_proposals, reserve_action, accept_reservation
from arrhenius_fracture.production_step_loop_v11 import _select, ActionTrialDiagnostic
from arrhenius_fracture.tip_local_proposals_v13 import construct_tip_local_proposals
from arrhenius_fracture.tip_directional_observation_v11 import selected_event_owner, candidate_tip_owners
from arrhenius_fracture.topology_transaction_v11 import mark_coalesced
from scripts.run_v13_tip_local_recovery import DEST, EXPECTED, load_frozen_interval
from scripts.run_pf_current_source_multifront_field_atlas_v12 import sha256
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
from scripts.audit_v13_marked_event_bookkeeping import audit_pair


@pytest.fixture(scope='module')
def frozen():
    record = json.loads((DEST/'frozen_failure/failed_proposal.json').read_text())
    source = DEST/'frozen_failure/failed_interval.pkl'
    assert sha256(source) == record['frozen_payload_sha256']
    payload = load_frozen_interval()
    corrected = restore_branch_checkpoint(DEST/'frozen_corrected/checkpoint/latest.json')
    proposals = construct_tip_local_proposals(payload['interval_state'], correlation_interval_s=1e-6)
    cached = {d.proposal.action_id: d.result for d in payload['diagnostics']}
    diagnostics = [ActionTrialDiagnostic(p, cached[p.action_id]) for p in proposals]
    chosen = _select(diagnostics, payload['interval_state'].competition)
    return payload, record, corrected, proposals, chosen


def test_frozen_failure_classification_and_unchanged_guard(frozen):
    p, record, *_ = frozen
    assert record['context']['step'] == 420
    assert record['reproduced_failure'] == 'one topology transaction spans multiple pre-event tips'
    assert len({m['pre_event_front_id'] for m in record['members']}) == 2
    assert len({m['process_owner_id'] for m in record['members']}) == 1
    assert not record['created_front_ids'] and not record['coalesced_front_ids']
    with pytest.raises(RuntimeError, match='multiple pre-event tips'):
        selected_event_owner(p['proposal'], p['observations'])
    path = 'arrhenius_fracture/tip_directional_observation_v11.py'
    from pathlib import Path
    assert Path(path).read_bytes() == subprocess.check_output(['git','show','651d32e:'+path])


def test_cross_tip_close_completions_partition_full_event_identity(frozen):
    p, record, _, proposals, _ = frozen
    assert abs(record['proposal']['completion_times_s'][0]-record['proposal']['completion_times_s'][1]) < 1e-6
    assert len(proposals) == 2 and all(q.action_type == 'one_arm' for q in proposals)
    assert len({(q.event_owner_tip_id,q.branch_opportunity_id) for q in proposals}) == 2
    for q in proposals:
        assert q.scoped_event_identities == ((q.event_owner_tip_id, q.member_candidate_ids[0], q.member_event_ordinals[0]),)
    assert fp(p['interval_state'].competition) == fp(load_frozen_interval()['interval_state'].competition)


def test_global_earliest_selected_and_other_event_pending(frozen):
    p, _, cp, proposals, chosen = frozen
    assert chosen.proposal.completion_times_s[0] == min(q.completion_times_s[0] for q in proposals)
    assert chosen.proposal.event_owner_tip_id == 'b042c2d7b4cc6a46'
    old = p['interval_state'].competition
    new = cp.state.competition
    assert set(new.consumed_event_ids)-set(old.consumed_event_ids) == set(chosen.proposal.member_event_ids)
    other = next(q for q in proposals if q != chosen.proposal)
    assert set(other.member_event_ids).issubset({e.event_id for e in new.pending_events})
    assert not set(other.member_event_ids).intersection(new.consumed_event_ids)


def test_nonselected_clock_rng_and_ordinals_unchanged_by_transaction(frozen):
    p, _, cp, _, chosen = frozen
    before = p['interval_state']
    for h in before.competition.hazard_states:
        if h.candidate_id not in chosen.proposal.member_candidate_ids:
            actual = next(v for v in cp.state.competition.hazard_states if v.candidate_id == h.candidate_id)
            assert fp(actual) == fp(h)
    assert fp(cp.state.rng_state) == fp(before.rng_state)
    # This comparison begins AFTER the common physical interval integration;
    # it does not pretend that physical hazard integration leaves actions fixed.


def test_selected_owner_renewal_exactly_once(frozen):
    p, _, _, _, chosen = frozen
    calls = json.loads((DEST/'frozen_corrected/renewal_calls.json').read_text())
    assert len(calls) == 1 and calls[0]['event_selected']
    assert calls[0]['event_distance_m'] == pytest.approx(5e-6)
    assert selected_event_owner(chosen.proposal,p['observations'])[1] == chosen.proposal.event_owner_tip_id


def test_same_tip_marked_birth_primary_once_companion_not_consumed():
    for case in ('Peak_300K_seed3621', 'Peak_1000K_seed3621'):
        audit = audit_pair(case)
        assert all(audit['checks'].values())
        assert audit['checks']['primary_consumed_once']
        assert audit['checks']['companion_not_consumed_at_birth']


def test_one_tip_structural_coalescence_does_not_consume_other_tip(frozen):
    p, _, _, proposals, chosen = frozen
    before = p['interval_state']
    owner = chosen.proposal.event_owner_tip_id
    other = next(q for q in proposals if q != chosen.proposal)
    network = mark_coalesced(before.crack_network, owner, other.event_owner_tip_id)
    state = accept_reservation(reserve_action(before.competition,chosen.proposal,event_rewards_m=(5e-6,)), chosen.proposal.action_id)
    assert network.branch(owner).local_state['merge_target_branch_id'] == other.event_owner_tip_id
    assert network.branch(other.event_owner_tip_id) == before.crack_network.branch(other.event_owner_tip_id)
    assert selected_event_owner(chosen.proposal,p['observations'])[1] == owner
    assert set(state.consumed_event_ids)-set(before.competition.consumed_event_ids) == set(chosen.proposal.member_event_ids)
    assert set(other.member_event_ids).issubset({e.event_id for e in state.pending_events})


def test_corrected_geometry_energy_charged_once(frozen):
    p, _, cp, _, chosen = frozen
    before = p['interval_state']
    assert fp(cp.state.crack_network) == fp(chosen.result.state.crack_network)
    assert cp.state.crack_network.total_physical_crack_length_m-before.crack_network.total_physical_crack_length_m == pytest.approx(5e-6)
    assert cp.state.energy_ledgers['topology_release_J_per_m'] == chosen.result.state.energy_ledgers['topology_release_J_per_m']
    assert cp.state.energy_ledgers['hazard_dissipation_J_per_m'] == chosen.result.state.energy_ledgers['hazard_dissipation_J_per_m']
    assert cp.state.event_counters['topology_actions'] == before.event_counters['topology_actions'] + 1


def test_actual_corrected_checkpoint_reload_owner_registry(tmp_path,frozen):
    _, _, cp, _, _ = frozen
    assert cp.state.event_counters['accepted_steps'] == 420
    write_branch_checkpoint(cp, tmp_path/'roundtrip.json')
    restored = restore_branch_checkpoint(tmp_path/'roundtrip.json')
    assert fp(restored) == fp(cp)
    assert set(cp.front_competitions) == set(cp.state.crack_network.active_tip_ids)
    owners = candidate_tip_owners(cp.state.crack_network,[h.candidate_id for h in cp.state.competition.hazard_states])
    assert set(owners.values()) == set(cp.front_competitions)
    assert set(cp.branch_clusters[0].arm_branch_ids) == set(cp.front_competitions)
    assert cp.branch_clusters[0].cluster_id == cp.state.junction_process_state['cluster'].cluster_id


def test_same_tip_proposals_identical_and_original_checkpoint_pinned(frozen):
    p, _, _, _, _ = frozen
    from tests.test_topology_transaction_v11 import fem_state
    state = fem_state(p['interval_state'].competition)
    old = construct_action_proposals(state.competition.hazard_states, correlation_interval_s=1e-6)
    scoped = construct_tip_local_proposals(state,correlation_interval_s=1e-6)
    assert tuple(replace(q,event_owner_tip_id=None,branch_opportunity_id=None) for q in scoped) == old
    source = DEST/'frozen_failure/input_checkpoint/latest.json.state.pkl'
    assert sha256(source) == EXPECTED
