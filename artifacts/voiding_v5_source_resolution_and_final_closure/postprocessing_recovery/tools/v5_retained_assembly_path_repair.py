"""One SHA-bound path-join repair; all frozen scientific functions stay exact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PHYSICAL_SHA = '63d53cdbad9125ef0d5180551974ac0e3a3ef721'
FROZEN_SHA256 = '7280e6e03f50915485aa87fb6700edfb99dcdd187760be48a8367cf025711601'
BROKEN = "args.shards[s['id']]"
CORRECTED = "(args.shards/s['id'])"


def corrected_source(path):
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != FROZEN_SHA256:
        raise ValueError('frozen complete assembler source changed')
    source = original.decode()
    if source.count(BROKEN) != 1:
        raise ValueError('path-join repair is not uniquely bounded')
    corrected = source.replace(BROKEN, CORRECTED)
    # Every top-level definition except the mechanical main orchestration is
    # byte/AST identical; main itself differs in exactly one expression.
    before, after = ast.parse(source), ast.parse(corrected)
    for left, right in zip(before.body, after.body, strict=True):
        if isinstance(left, ast.FunctionDef) and left.name == 'main':
            continue
        if ast.dump(left) != ast.dump(right):
            raise ValueError('non-orchestration source changed')
    return corrected


def load_assembler(implementation):
    path = implementation / 'scripts/assemble_v5_source_resolution_campaign_v1.py'
    source = corrected_source(path)
    namespace = {'__name__': 'v5_explicit_retained_assembly_path_repair', '__file__': str(path)}
    sys.path.insert(0, str(implementation / 'scripts'))
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', required=True, type=Path)
    parser.add_argument('--recovery-sha', required=True)
    parser.add_argument('shards', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    implementation = args.implementation.resolve()
    def git(*values):
        return subprocess.check_output(('git', *values), cwd=implementation, text=True).strip()
    if git('rev-parse', 'HEAD') != PHYSICAL_SHA or git('status', '--porcelain'):
        raise ValueError('clean exact physical implementation required')
    if git('rev-parse', args.recovery_sha) != args.recovery_sha:
        raise ValueError('exact recovery SHA required')
    namespace, source = load_assembler(implementation)
    previous = sys.argv
    try:
        sys.argv = [namespace['__file__'], str(args.shards), str(args.output)]
        namespace['main']()
    finally:
        sys.argv = previous
    if git('rev-parse', 'HEAD') != PHYSICAL_SHA or git('status', '--porcelain'):
        raise ValueError('frozen implementation changed during assembly')
    receipt = {
        'schema': 'v5.retained-complete-assembly-path-repair/1',
        'record_kind': 'POSTPROCESSING_REPAIR_NOT_NEW_PHYSICAL_EXECUTION',
        'physical_implementation_sha': PHYSICAL_SHA,
        'recovery_sha': args.recovery_sha,
        'original_failed_assembly_run_id': 34255392088,
        'original_failed_assembly_job_id': 102159780607,
        'original_failure': "TypeError: 'PosixPath' object is not subscriptable",
        'frozen_assembler_sha256': FROZEN_SHA256,
        'executed_repaired_assembler_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'repair_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exact_single_replacement': {'before': BROKEN, 'after': CORRECTED},
        'scientific_functions_and_predicates_unchanged': True,
        'physical_worker_invocations': 0,
        'clean_physical_worktree_at_end': True,
    }
    namespace['write'](args.output / 'assembly_repair.json', receipt)
    namespace['write'](args.output / 'sha256_manifest.json', {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*'))
        if p.is_file() and p != args.output / 'sha256_manifest.json'})
    namespace['verify_inventory'](args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
