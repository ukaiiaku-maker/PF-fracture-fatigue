"""Complete a missing-tool report suffix; never execute a physical worker.

Original failed attempts are retained byte-for-byte. The report statements
come from the clean physical implementation, not a replacement predicate.
"""
import argparse
import ast
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
import runpy
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout

IMPLEMENTATION = '63d53cdbad9125ef0d5180551974ac0e3a3ef721'
BASE = '326e3f5973ef623781ab8568798c495a3f68238c'
FROZEN_SCRIPT_SHA256 = '6a798ecd825fc495e4b722692491efb87ebfb3cf5ff991e2b7594c9c97c74a0e'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    paths = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('symlink in retained capture')
    return {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}


def clean(repository, expected):
    def git(*args):
        return subprocess.check_output(('git', *args), cwd=repository, text=True).strip()
    if git('rev-parse', 'HEAD') != expected or git('status', '--porcelain'):
        raise ValueError('requires clean exact implementation ' + expected)


def suffix(script):
    if digest(script) != FROZEN_SCRIPT_SHA256:
        raise ValueError('frozen scientific report script changed')
    tree = ast.parse(script.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'a')
    nodes = main.body[start:]
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == 'worker' for statement in nodes for node in ast.walk(statement)):
        raise ValueError('physical worker call in report suffix')
    return ast.Module(body=nodes, type_ignores=[])


def report_only(implementation, evidence):
    script = implementation / 'scripts/qualify_v5_disabled_causal_neutrality_v1.py'
    sys.path.insert(0, str(implementation / 'scripts'))
    namespace = runpy.run_path(str(script), run_name='retained_report_definitions_only')
    namespace.update(root=implementation, current=IMPLEMENTATION,
                     args=SimpleNamespace(output=evidence))
    tree = suffix(script)
    capture = io.StringIO()
    with redirect_stdout(capture):
        exec(compile(tree, str(script), 'exec'), namespace)
    return {'frozen_script_sha256': digest(script),
            'frozen_suffix_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            'stdout': capture.getvalue()}


def audit_worker(implementation, repository, evidence, mode):
    clean(repository, BASE if mode == 'base' else IMPLEMENTATION)
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(implementation / 'scripts'))
    namespace = runpy.run_path(str(implementation / 'scripts/qualify_v5_disabled_causal_neutrality_v1.py'),
                              run_name='retained_audit_definitions_only')
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
    rows = json.loads((evidence / mode / 'rows.json').read_text())
    expected_sha = BASE if mode == 'base' else IMPLEMENTATION
    if rows['sha'] != expected_sha or [r['case_id'] for r in rows['rows']] != list(namespace['CASES']):
        raise ValueError('wrong retained worker identity or case registry')
    for row in rows['rows']:
        if row['configuration'] != namespace['CASES'][row['case_id']]:
            raise ValueError('retained physical inputs changed')
        state = restore_checkpoint(evidence / mode / 'checkpoints' / (row['case_id'] + '_terminal.json'))
        components = {field.name: namespace['normalize'](getattr(state, field.name)) for field in fields(state)}
        if components != row['terminal_components']:
            raise ValueError('raw terminal components do not reconstruct: ' + row['case_id'])
    clean(repository, expected_sha)
    return {'mode': mode, 'worker_sha': expected_sha, 'terminal_captures_reconstructed': len(rows['rows'])}


def repair(source, output, implementation, base, recovery_sha):
    if not re.fullmatch('[0-9a-f]{40}', recovery_sha or ''):
        raise ValueError('exact recovery implementation SHA required')
    recovery_root = Path(subprocess.check_output(('git', 'rev-parse', '--show-toplevel'),
        cwd=Path(__file__).resolve().parent, text=True).strip())
    clean(recovery_root, recovery_sha)
    clean(implementation, IMPLEMENTATION)
    clean(base, BASE)
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError('refusing to overwrite original attempt or repair')
    files = inventory(source)
    original_manifest = json.loads((source / 'sha256_manifest.json').read_text())
    if {k: v for k, v in files.items() if k != 'sha256_manifest.json'} != original_manifest:
        raise ValueError('original failed attempt inventory mismatch')
    original = json.loads((source / 'execution.json').read_text())
    if original['executed_code_sha'] != IMPLEMENTATION or original['phase'] != 'causal-neutrality' or original['execution_completed']:
        raise ValueError('not the identified failed causal postprocessing attempt')
    if original['operations'] != [{'operation': 'causal_neutrality', 'script': 'qualify_v5_disabled_causal_neutrality_v1.py', 'returncode': 1}]:
        raise ValueError('unexpected original failure operation')
    log = (source / 'causal_neutrality.log').read_text()
    if not log.rstrip().endswith("FileNotFoundError: [Errno 2] No such file or directory: 'rg'"):
        raise ValueError('not the isolated missing-ripgrep failure')
    if (source / 'evidence/report.json').exists() or not shutil.which('rg'):
        raise ValueError('unexpected prior report or missing repaired dependency')
    sys.path.insert(0, str(implementation / 'scripts'))
    from v5_numerical_runtime_v1 import require_pinned, runtime_record
    kernels = require_pinned(runtime_record())
    if kernels != original['selected_numerical_kernels']:
        raise ValueError('repair kernels differ from the original physical worker')
    audited = []
    for mode, repository in (('base', base), ('disabled', implementation)):
        raw = subprocess.check_output((sys.executable, str(Path(__file__).resolve()), 'audit-worker',
            '--implementation', str(implementation), '--repository', str(repository),
            '--source', str(source / 'evidence'), '--mode', mode), text=True, cwd=repository)
        audited.append(json.loads(raw))
    shutil.copytree(source, output / 'retained_failed_attempt')
    shutil.copytree(source / 'evidence', output / 'evidence')
    completion = report_only(implementation, output / 'evidence')
    if inventory(source) != files or inventory(output / 'retained_failed_attempt') != files:
        raise ValueError('original capture changed during postprocessing')
    for name, sha in files.items():
        if name.startswith('evidence/') and digest(output / name) != sha:
            raise ValueError('retained raw evidence changed')
    receipt = {'schema': 'v5.retained-neutrality-postprocessing-repair/1',
        'record_kind': 'POSTPROCESSING_ONLY_NO_NEW_PHYSICAL_EXECUTION',
        'physical_implementation_sha': IMPLEMENTATION, 'original_campaign_run_id': 34205485025,
        'original_job_id': 101995141502, 'recovery_implementation_sha': recovery_sha,
        'recovery_script_sha256': digest(Path(__file__)), 'original_inventory': files,
        'repaired_dependency_version': subprocess.check_output(('rg', '--version'), text=True).splitlines()[0],
        'original_execution_completed': False, 'original_failure_retained_exactly': True,
        'physical_worker_invocations_in_repair': 0, 'terminal_source_audits': audited,
        'scientific_predicates_unchanged': True, **completion}
    (output / 'completion_repair.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    envelope = {**original, 'schema': 'v5.source-resolution-final-phase-repaired-postprocessing/1',
        'record_kind': 'ORIGINAL_PHYSICAL_CAPTURE_WITH_SEPARATELY_ATTESTED_REPORT_COMPLETION',
        'postprocessing_recovery_sha': recovery_sha, 'physical_reexecution_performed': False,
        'original_failed_execution': 'retained_failed_attempt/execution.json',
        'completion_repair_sha256': digest(output / 'completion_repair.json'),
        'operations': [{'operation': 'source_bound_terminal_capture_audit', 'returncode': 0},
                       {'operation': 'frozen_original_report_suffix_completion', 'returncode': 0}],
        'execution_completed': True}
    (output / 'execution.json').write_text(json.dumps(envelope, sort_keys=True, indent=2) + '\n')
    (output / 'sha256_manifest.json').write_text(json.dumps(inventory(output), sort_keys=True, indent=2) + '\n')
    clean(implementation, IMPLEMENTATION)
    clean(base, BASE)
    clean(recovery_root, recovery_sha)
    return {'execution_completed': True, 'physical_reexecution_performed': False,
            'scientific_pass': json.loads((output / 'evidence/report.json').read_text())['passed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('repair', 'audit-worker'))
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--recovery-sha')
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--mode', choices=('base', 'disabled'))
    args = parser.parse_args()
    result = audit_worker(args.implementation, args.repository, args.source, args.mode) if args.operation == 'audit-worker' else repair(args.source, args.output, args.implementation, args.base, args.recovery_sha)
    print(json.dumps(result, sort_keys=True))
