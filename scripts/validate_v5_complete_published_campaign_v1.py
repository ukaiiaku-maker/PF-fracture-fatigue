#!/usr/bin/env python3
"""Reconstruct the published complete evidence in its exact implementation worker."""
import argparse
import json
from importlib.metadata import version,PackageNotFoundError
import platform
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validate_v5_source_resolution_evidence_v1 import verify_inventory,validate_static,validate_positive,validate_causality,same
from assemble_v5_source_resolution_campaign_v1 import derive_ledger
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from package_v5_final_campaign_v1 import verify_publication_binding


def validate(root,publication_manifest,publication_sha):
    def git(*args):return subprocess.check_output(('git',*args),cwd=ROOT,text=True).strip()
    report=json.loads(publication_manifest.read_text());sha=git('rev-parse','HEAD')
    if sha!=report['implementation_sha'] or git('status','--porcelain'):
        raise ValueError('complete reconstruction requires the clean exact implementation worker')
    runtime=report['numerical_runtime'];packages={}
    for name in runtime['numerical_package_versions']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    if platform.python_version()!=runtime['python_version'] or packages!=runtime['numerical_package_versions']:
        raise ValueError('publication reconstruction numerical runtime differs from its producer')
    from v5_numerical_runtime_v1 import require_pinned,runtime_record
    kernels=require_pinned(runtime_record())
    same(kernels,runtime['selected_numerical_kernels'],'publication reconstruction selected kernels differ from producer')
    same(kernels['environment'],runtime['numerical_environment'],'publication reconstruction numerical environment differs')
    binding=verify_publication_binding(report,publication_sha);verify_inventory(root);results={}
    for side in ('a','b'):
        directory=root/side;verify_inventory(directory)
        static=validate_static(directory/'static')
        positive=validate_positive(directory/'source/positive')
        causality=validate_causality(directory/'source/causality')
        lifecycle=json.loads((directory/'lifecycle/lifecycle_rows.json').read_text())
        class Sources:
            def __getitem__(self,key):return restore_checkpoint(directory/'lifecycle'/key)
        ontology=validate_closure_evidence(lifecycle,Sources(),executed_code_sha=sha)
        same(derive_ledger(directory),json.loads((directory/'scientific_ledger.json').read_text()),
            'published complete scientific classification does not recompute')
        results[side]={'static':static,'positive':positive,'causality':causality,'lifecycle':ontology}
    a,b=(verify_inventory(root/side) for side in ('a','b'))
    paired=json.loads((root/'paired_comparison.json').read_text())
    same(paired['exact_recursive_comparison'],a==b,'published A/B comparison mismatch')
    same(paired['file_count_per_side'],[len(a),len(b)],'published A/B inventory count mismatch')
    if not a==b:raise ValueError('published full A/B evidence differs')
    if git('status','--porcelain') or git('rev-parse','HEAD')!=sha:raise ValueError('implementation worker changed')
    return {'schema':'v5.exact-publication-head-complete-evidence-audit/1',**binding,
        'exact_recursive_evidence_comparison':True,'reconstruction_results':results,
        'clean_implementation_worker_at_end':True,'evidence_valid':True,
        'recorded_numerical_runtime_exact':True,
        'scientific_decision':json.loads((root/'a/scientific_ledger.json').read_text())['scientific_decision'],
        'record_kind':'EXACT_PUBLICATION_HEAD_AUDIT_OF_SEPARATELY_IDENTIFIED_IMPLEMENTATION_EXECUTION'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('evidence',type=Path)
    parser.add_argument('publication_manifest',type=Path);parser.add_argument('publication_sha');parser.add_argument('output',type=Path)
    args=parser.parse_args();result=validate(args.evidence,args.publication_manifest,args.publication_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'evidence_valid':result['evidence_valid'],'scientific_decision':result['scientific_decision']}))
