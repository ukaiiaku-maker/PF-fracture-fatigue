from copy import deepcopy

import pytest

from arrhenius_fracture.future_causal_state_fingerprint_v1 import components, fingerprint


def representative():
    return {
        'void_state': None,
        'energy_ledgers': {
            'latest_fem_energy_J_per_m': 2.0,
            'latest_reaction_N_per_m': 3.0,
            'latest_free_dof_residual_l2_N_per_m': 1e-12,
            'latest_constrained_reaction_l2_N_per_m': 4.0,
            'latest_top_bottom_reaction_balance': 1e-13,
            'latest_energy_reaction_identity': 2e-13,
        },
        'junction_process_state': {
            'v12_boundary_terminal_certificates': [],
            'v12_graph_support_audit': {'boundary_terminal_certificates': [], 'passed': True},
        },
        'v12_support_state': {'source_commit': 'old', 'mesh_geometry_fingerprint': 'mesh'},
        'hazards': [{'action': .25, 'threshold': .75}],
        'rng_state': {'state': 17},
    }


def test_only_prospectively_declared_noncausal_fields_are_excluded():
    first = representative(); second = deepcopy(first)
    second['energy_ledgers']['latest_free_dof_residual_l2_N_per_m'] = 9e-12
    second['v12_support_state']['source_commit'] = 'new'
    second['junction_process_state'].pop('v12_boundary_terminal_certificates')
    second.pop('void_state')
    assert fingerprint(first) == fingerprint(second)
    assert set(components(first)[1]) == {
        '/energy_ledgers/latest_free_dof_residual_l2_N_per_m',
        '/energy_ledgers/latest_constrained_reaction_l2_N_per_m',
        '/energy_ledgers/latest_top_bottom_reaction_balance',
        '/energy_ledgers/latest_energy_reaction_identity',
        '/junction_process_state/v12_boundary_terminal_certificates',
        '/junction_process_state/v12_graph_support_audit/boundary_terminal_certificates',
        '/v12_support_state/source_commit',
    }


@pytest.mark.parametrize('path', ['energy', 'reaction', 'hazard', 'rng', 'mesh', 'topology'])
def test_future_physical_consumers_remain_in_fingerprint(path):
    first = representative(); second = deepcopy(first)
    if path == 'energy': second['energy_ledgers']['latest_fem_energy_J_per_m'] += 1
    elif path == 'reaction': second['energy_ledgers']['latest_reaction_N_per_m'] += 1
    elif path == 'hazard': second['hazards'][0]['action'] += .1
    elif path == 'rng': second['rng_state']['state'] += 1
    elif path == 'mesh': second['v12_support_state']['mesh_geometry_fingerprint'] = 'other'
    else: second['junction_process_state']['v12_graph_support_audit']['passed'] = False
    assert fingerprint(first) != fingerprint(second)


def test_enabled_void_state_cannot_be_hidden():
    value = representative(); value['void_state'] = {'enabled': True}
    with pytest.raises(ValueError): components(value)
