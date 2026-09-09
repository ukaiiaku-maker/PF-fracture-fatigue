#!/usr/bin/env python3
"""Require every currently collected focused test in the actual full-run XML."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
FOCUSED_FILES=(
    'test_finalization_v3_closure_schema.py','test_closure_static_evidence.py','test_explicit_cavity_v5.py',
    'test_crack_void_fixed_geometry_v3.py','test_production_source_resolution_v3.py',
    'test_closure_mechanics_evidence.py','test_closure_lifecycle_evidence.py','test_closure_lifecycle_rollback.py',
    'test_voiding_lifecycle_driver_v5.py','test_v5_clean_historical_neutrality.py',
    'test_closure_publication_package.py','test_closure_production_topology_audit.py',
    'test_no_legacy_variational_modules.py','test_quality_constrained_mesh_v1.py',
    'test_source_transfer_budget_v1.py','test_cavity_boundary_patch_recovery_v1.py',
    'test_cavity_recovery_transfer_budget_v1.py','test_canonical_kinetic_time_v1.py',
    'test_boundary_segment_bounding_v1.py','test_v12_mechanically_separating_wake.py',
    'test_voiding_v5.py','test_static_numerical_family_v1.py','test_compressed_closure_checkpoint_v1.py',
    'test_disabled_causal_neutrality_v1.py',
    'test_future_causal_state_fingerprint_v1.py',
    'test_source_resolution_campaign_v1.py',
    'test_common_restart_protocol_v1.py',
    'test_final_campaign_archive_v1.py',
    'test_v5_numerical_runtime_v1.py',
)


def summarize(xml,expected):
    rows=[]
    for case in ET.parse(xml).getroot().iter('testcase'):
        classname=case.attrib['classname'];node=None
        for file in FOCUSED_FILES:
            prefix='tests.'+file[:-3]
            if classname==prefix or classname.startswith(prefix+'.'):
                suffix=classname[len(prefix):].lstrip('.')
                node='tests/'+file+'::'+(suffix.replace('.','::')+'::' if suffix else '')+case.attrib['name']
                break
        if node is not None:rows.append({'nodeid':node,'failed':case.find('failure') is not None or case.find('error') is not None,
            'skipped':case.find('skipped') is not None})
    observed=Counter(r['nodeid'] for r in rows)
    complete=observed==Counter(expected)
    return {'schema':'v5.source-resolution-full-focused-regression/1','expected_count':len(expected),
        'executed_count':len(rows),'failed':sum(r['failed'] for r in rows),'skipped':sum(r['skipped'] for r in rows),
        'missing':sorted(set(expected)-set(observed)),'unexpected':sorted(set(observed)-set(expected)),
        'complete_registry_exact':complete,'tests':rows,
        'passed':complete and bool(rows) and not any(r['failed'] or r['skipped'] for r in rows)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('xml',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--execute',action='store_true',help='Run the entire focused registry before interpreting its actual XML')
    args=parser.parse_args()
    if args.execute:
        args.xml.parent.mkdir(parents=True,exist_ok=True)
        completed=subprocess.run([sys.executable,'-m','pytest','-q',*('tests/'+name for name in FOCUSED_FILES),
            '--junitxml='+str(args.xml)],cwd=ROOT)
        if completed.returncode not in (0,1):raise SystemExit(completed.returncode)
    collected=subprocess.check_output([sys.executable,'-m','pytest','--collect-only','-q',*('tests/'+name for name in FOCUSED_FILES)],cwd=ROOT,text=True)
    expected=[line.strip() for line in collected.splitlines() if line.startswith('tests/') and '::' in line]
    report=summarize(args.xml,expected)
    report.update(executed_code_sha=subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        full_run_xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'}))
    if not report['passed']:raise SystemExit(1)
