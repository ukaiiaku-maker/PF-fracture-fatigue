#!/usr/bin/env python3
"""Verify frozen evidence and publish one compact clock/mechanism review."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.run_pf_current_source_multifront_field_atlas_v12 import sha256, atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original-root',type=Path,required=True)
    args=parser.parse_args()
    old=json.loads((args.original_root/'artifact_inventory.json').read_text())
    for entry in old['files']:
        if sha256(args.original_root/entry['path'])!=entry['sha256']:
            raise RuntimeError('immutable original artifact changed: '+entry['path'])
    modules=['test_v13_clean_parent_capture','test_v13_physical_companion_contract',
        'test_topology_transaction_v11','test_directional_competition_transactions_v11',
        'test_multifront_checkpoint_output_v12','test_v13_clock_inverse_audit',
        'test_v13_cooperative_pair_transition','test_v13_later_parent_contract','test_v13_revised_ensemble_gate',
        'test_v13_inherited_residual_clock']
    test=subprocess.run([sys.executable,'-m','pytest','-q',*(f'tests/{m}.py' for m in modules)],
        capture_output=True,text=True,check=True)
    compiled=subprocess.run([sys.executable,'-m','compileall','-q','arrhenius_fracture',
        'scripts/audit_v13_inherited_clocks.py','scripts/run_v13_later_clean_parents.py',
        'scripts/qualify_v13_later_companions.py','scripts/report_v13_clock_and_pair_mechanism.py',
        'scripts/package_v13_clock_review.py'],capture_output=True,text=True,check=True)
    diff=subprocess.run(['git','diff','--check'],capture_output=True,text=True,check=True)
    atomic_json(args.root/'verification.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'original_immutable_files_verified':len(old['files']),'focused_tests_stdout':test.stdout,'focused_tests_stderr':test.stderr,
        'focused_tests_exit_code':test.returncode,'compileall_exit_code':compiled.returncode,'diff_check_exit_code':diff.returncode,
        'full_suite_or_sampler_Monte_Carlo_repeated':False})
    inventory=[]
    for folder in ('later_parents','later_companions'):
        for path in sorted((args.root/folder).rglob('*')):
            if path.is_file():
                inventory.append({'path':str(path.relative_to(args.root)),'bytes':path.stat().st_size,'sha256':sha256(path)})
    atomic_json(args.root/'artifact_inventory.json',{'root':str(args.root.resolve()),'files':inventory,
        'large_checkpoints_fields_and_FEM_cache_in_compact_archive':False})
    selected=[args.root/name for name in ('V13_CLOCK_AND_PAIR_MECHANISM.md','clock_audit.json',
        'later_parent_plan.json','revised_ensemble_gate.json','verification.json','artifact_inventory.json','source_preservation.json',
        'later_companions/summary.json')]
    for case in ('Peak_1000K','weakT_1000K'):
        folder=args.root/'later_parents'/case
        selected.extend(folder/name for name in ('launch.json','restart_fork_provenance.json','terminal.json','progress.json'))
        selected.extend(folder.glob('events/*/parent/parent_record.json'))
    archive=args.root/'V13_CLOCK_AND_PAIR_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as stream:
        for path in selected:
            stream.write(path,path.relative_to(args.root))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    atomic_json(args.root/'archive_manifest.json',{'path':archive.name,'sha256':sha256(archive),
        'bytes':archive.stat().st_size,'zip_integrity':'PASS'})
    print(test.stdout.strip())
    print('Original immutable artifacts PASS',len(old['files']))
    print(archive,archive.stat().st_size)


if __name__=='__main__':
    main()
