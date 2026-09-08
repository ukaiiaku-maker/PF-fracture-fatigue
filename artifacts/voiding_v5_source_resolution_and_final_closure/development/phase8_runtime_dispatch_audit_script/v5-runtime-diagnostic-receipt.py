import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
from validate_v5_source_resolution_evidence_v1 import verify_inventory
from v5_numerical_runtime_v1 import require_pinned
root=ROOT/'artifacts/voiding_v5_source_resolution_and_final_closure/development/phase8_runtime_dispatch_interventions_c8b'
verify_inventory(root)
paths=sorted(root.rglob('report.json'));assert len(paths)==3
reports=[json.loads(path.read_text()) for path in paths]
fixed=[report['results']['fixed-policy'] for report in reports]
assert all(value==fixed[0] for value in fixed)
assert all(report['results']['fixed-policy']==report['results']['fixed-policy-repeat'] for report in reports)
for value in fixed:require_pinned(value['runtime'])
run=json.loads(subprocess.check_output(['gh','run','view','34197537829','--repo','ukaiiaku-maker/PF-fracture-fatigue',
    '--json','databaseId,headSha,status,conclusion,url,jobs'],text=True))
assert run['status']=='completed' and run['conclusion']=='success'
jobs=[{key:job[key] for key in ('databaseId','name','status','conclusion','startedAt','completedAt')}
      for job in run['jobs'] if job['name'].startswith('runtime-diagnostic / runtime')]
assert len(jobs)==3 and all(job['conclusion']=='success' for job in jobs)
record={'schema':'v5.cross-worker-dispatch-reconstruction-receipt/1',
    'record_kind':'READ_ONLY_RECONSTRUCTION_AND_DISPATCH_INTERVENTION_NOT_NEW_PHYSICAL_EXECUTION',
    'run_id':run['databaseId'],'source_code_sha':run['headSha'],'run_conclusion':run['conclusion'],'url':run['url'],'jobs':jobs,
    'source_pack_manifest_sha256':hashlib.sha256((root/'sha256_manifest.json').read_bytes()).hexdigest(),
    'three_fixed_policy_reports_exact':True,'all_independent_fixed_policy_repeats_exact':True,
    'actual_fixed_kernel_contract':fixed[0]['runtime'],
    'retained_checkpoint_fingerprint':fixed[0]['checkpoint_fingerprint'],
    'workers':[{'relative_report_path':str(path.relative_to(root)),
        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'native_blas_targets':[library['architecture'] for library in report['results']['native']['runtime']['blas_libraries']],
        'topology_hashes':{mode:value['topology_sha256'] for mode,value in report['results'].items()},
        'old_recorded_topology_exact':{mode:value['topology_exact'] for mode,value in report['results'].items()}}
        for path,report in zip(paths,reports)],
    'scientific_qualification':False,'old_mixed_kernel_restart_gate_requalified':False,
    'scope':'Fixed-policy reproducibility on one identical retained checkpoint and small algebra sentinel; final physical A/B still required.'}
output=root.parent/'phase8_runtime_dispatch_reconstruction.json'
output.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(json.dumps({'output':str(output),'run_id':record['run_id'],'three_fixed_policy_reports_exact':True}))
