import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('repair', str(Path(__file__).with_name('v5_retained_neutrality_postprocess.py')))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
ROOT = Path(os.environ.get('V5_PHYSICAL_WORKTREE', '/private/tmp/pf-sharp-front-stateful-voiding-v5'))
CAPTURE = Path(os.environ.get('V5_FAILED_CAPTURE_ROOT', '/private/tmp/v5-final-causal-neutrality-63d/causal-neutrality'))


def rows_only(tmp_path, side):
    output = tmp_path / side
    for mode in ('base', 'disabled'):
        (output / mode).mkdir(parents=True)
        shutil.copy2(CAPTURE / side / 'evidence' / mode / 'rows.json', output / mode / 'rows.json')
    return output


def test_original_suffix_is_exact_and_cannot_launch_workers(tmp_path, monkeypatch):
    original_run = subprocess.run
    def rg_only(command, *args, **kwargs):
        assert command[0] == 'rg', 'physical worker invocation prohibited'
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', rg_only)
    results = []
    for side in ('a', 'b'):
        output = rows_only(tmp_path, side)
        original = repair.inventory(output)
        completed = repair.report_only(ROOT, output)
        assert completed['frozen_script_sha256'] == repair.FROZEN_SCRIPT_SHA256
        for name, sha in original.items():
            assert repair.digest(output / name) == sha
        result = json.loads((output / 'report.json').read_text())
        assert result['passed'] is False and len(result['rows']) == 4
        assert all(row['observed_causal_states_exact'] for row in result['rows'])
        assert all(not row['passed'] for row in result['rows'])
        results.append(repair.inventory(output))
    assert results[0] == results[1]


def test_changed_causal_state_remains_visible(tmp_path):
    output = rows_only(tmp_path, 'a')
    path = output / 'disabled/rows.json'
    data = json.loads(path.read_text())
    data['rows'][0]['raw_states'][0]['displacement']['array_sha256'] = '0' * 64
    path.write_text(json.dumps(data))
    repair.report_only(ROOT, output)
    report = json.loads((output / 'report.json').read_text())
    assert not report['rows'][0]['observed_causal_states_exact']
    assert not report['passed']


def test_refuses_changed_predicate_source(tmp_path):
    path = tmp_path / 'script.py'
    path.write_text('def main():\n    a = {}\n')
    with pytest.raises(ValueError, match='frozen scientific report script changed'):
        repair.suffix(path)
