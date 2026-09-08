import sys
from pathlib import Path
from copy import deepcopy
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from qualify_v5_disabled_causal_neutrality_v1 import causal, DIAGNOSTICS


def base():
    return {'energy_ledgers': {'physical_energy': 2.}, 'junction_process_state': {'v12_graph_support_audit': {}},
        'v12_support_state': {'source_commit': 'old', 'support': [1, 2]}, 'hazards': [0.1], 'rng': 3621}


def test_audited_empty_inactive_extensions_are_not_physical_but_raw_is_retained():
    first = base(); second = deepcopy(first); second['void_state'] = None
    second['v12_support_state']['source_commit'] = 'new'
    second['junction_process_state']['v12_boundary_terminal_certificates'] = []
    second['junction_process_state']['v12_graph_support_audit']['boundary_terminal_certificates'] = []
    second['energy_ledgers'].update({k: 1. for k in DIAGNOSTICS})
    before = deepcopy(second)
    assert first != second and causal(first)[0] == causal(second)[0]
    assert second == before


@pytest.mark.parametrize('field', ['hazards', 'rng'])
def test_any_causal_difference_remains_visible(field):
    first = base(); second = deepcopy(first); second[field] = 'changed'
    assert causal(first)[0] != causal(second)[0]


def test_nonempty_certificate_and_active_void_are_not_excluded():
    first = base(); second = deepcopy(first)
    second['junction_process_state']['v12_boundary_terminal_certificates'] = [{'passed': False}]
    assert causal(first)[0] != causal(second)[0]
    second['void_state'] = {'active': True}
    with pytest.raises(ValueError): causal(second)
