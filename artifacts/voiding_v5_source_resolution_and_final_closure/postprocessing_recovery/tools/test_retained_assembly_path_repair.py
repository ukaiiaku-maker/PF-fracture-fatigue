import ast
import importlib.util
import os
from pathlib import Path
import sys
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('path_repair', Path(__file__).with_name('v5_retained_assembly_path_repair.py'))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
ROOT = Path(os.environ.get('V5_PHYSICAL_WORKTREE', '/private/tmp/pf-sharp-front-stateful-voiding-v5'))


def test_only_the_identified_path_expression_changes():
    path = ROOT / 'scripts/assemble_v5_source_resolution_campaign_v1.py'
    original = path.read_text()
    corrected = repair.corrected_source(path)
    assert corrected.replace(repair.CORRECTED, repair.BROKEN) == original
    before, after = ast.parse(original), ast.parse(corrected)
    for left, right in zip(before.body, after.body, strict=True):
        if not isinstance(left, ast.FunctionDef) or left.name != 'main':
            assert ast.dump(left) == ast.dump(right)


def test_changed_frozen_source_is_rejected(tmp_path):
    path = tmp_path / 'changed.py'
    path.write_text((ROOT / 'scripts/assemble_v5_source_resolution_campaign_v1.py').read_text() + '\n')
    with pytest.raises(ValueError, match='frozen complete assembler source changed'):
        repair.corrected_source(path)


def test_real_main_resolves_all_35_shards_without_any_physical_worker(tmp_path, monkeypatch):
    namespace, _ = repair.load_assembler(ROOT)
    import v5_numerical_runtime_v1 as runtime
    monkeypatch.setattr(runtime, 'runtime_record', lambda: {})
    monkeypatch.setattr(runtime, 'require_pinned', lambda _: {})
    def no_worker(*args, **kwargs):
        raise AssertionError('physical worker invocation prohibited during assembly')
    monkeypatch.setattr(subprocess, 'run', no_worker)
    monkeypatch.setattr(subprocess, 'Popen', no_worker)
    specs = namespace['matrix']()
    by_id = {s['id']: s for s in specs}
    inputs, output = tmp_path / 'inputs', tmp_path / 'output'
    inventory_calls, assemblies, copied, validated = [], [], [], []
    def inventory(path):
        inventory_calls.append(path)
        return {}
    def read(path, name):
        assert name == 'execution.json'
        selected = by_id[path.parent.name]
        return {'execution_completed': True, 'selected_numerical_kernels': {},
                'phase': selected['phase'], 'section': selected['section'],
                'shard_index': selected['index'], 'shard_count': selected['count'],
                'executed_code_sha': repair.PHYSICAL_SHA}
    namespace.update(verify_inventory=inventory, read=read,
                     assemble=lambda kind, paths, dest: assemblies.append((kind, paths, dest)),
                     validate_positive=lambda path: validated.append(path),
                     validate_causality=lambda path: validated.append(path),
                     derive_ledger=lambda _: {'scientific_decision': 'BLOCKED'})
    monkeypatch.setattr(namespace['shutil'], 'copytree', lambda source, dest, **_: copied.append((source, dest)))
    monkeypatch.setattr(sys, 'argv', ['assembler', str(inputs), str(output)])
    namespace['main']()
    assert len(assemblies) == 4
    for kind, paths, dest in assemblies:
        assert paths == [inputs / s['id'] / dest.parent.name / 'evidence' for s in specs if s['phase'] == kind]
    assert len(inventory_calls) == 142  # 70 before, A/B pair, 70 after.
    assert len(copied) == 6 and len(validated) == 4
    assert (output / 'a/scientific_ledger.json').read_text() == (output / 'b/scientific_ledger.json').read_text()
