import inspect
import json
from pathlib import Path
import pytest
from scripts.run_v13_later_clean_parents import certify, CASES, SOURCE_ROOT, main, LaterCapture


@pytest.mark.parametrize('case',CASES)
def test_only_exact_committed_first_parent_is_certified(case):
    source,manifest,record=certify(case)
    assert record['fresh_initialization']
    assert not manifest['crack_network']['branching_enabled']
    argv=json.loads((source.parent.parent/'launch.json').read_text())['arguments']
    assert '--out' in argv and '--maximum-fronts' in argv
    assert argv[argv.index('--maximum-fronts')+1]=='1'


def test_nonselected_case_is_not_authorized():
    with pytest.raises(ValueError):
        certify('DBTT_300K')


def test_continuation_never_invokes_branch_mark_or_parent_initialization():
    source=inspect.getsource(main)
    assert '--v11-restart-checkpoint' in source
    assert 'first_selected_event_step' not in source
    assert 'apply_cooperative_pair_transition' not in source
    assert 'sample' not in source
    assert 'branching_enabled' in inspect.getsource(LaterCapture.begin)
