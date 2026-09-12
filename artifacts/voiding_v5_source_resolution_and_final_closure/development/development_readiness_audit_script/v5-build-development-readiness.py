"""Generate readiness only after all registered development executions/audits exist."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
from validate_v5_source_resolution_evidence_v1 import verify_inventory
from validate_v5_development_lifecycle_shards_v1 import require_registry
from v5_source_resolution_campaign_matrix_v1 import require_development_complete
root=ROOT/'artifacts/voiding_v5_source_resolution_and_final_closure';dev=root/'development'

def reference(path):
    return {'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def manifest(name):
    directory=dev/name;verify_inventory(directory)
    return reference(directory/'sha256_manifest.json')

def audit(name,section,count):
    directory=dev/name;verify_inventory(directory)
    paths=list(directory.rglob('report.json'));assert len(paths)==1
    data=json.loads(paths[0].read_text())
    assert data['section']==section and data['actual_section_cases']==count and data['complete_section_registry']
    assert len(data['row_reconstruction_results'])==data['actual_source_rows']
    return data,[reference(directory/'sha256_manifest.json'),reference(paths[0])]

transition,transition_refs=audit('phase7_transitions_linux_reconstruction_2fa','transitions',45)
rollback,rollback_refs=audit('phase8_rollback_linux_reconstruction_2fa','rollback',39)
restart,restart_refs=audit('phase8_restarts_linux_reconstruction','restarts',11)
restart_rows=[];restart_packs=[]
for i in range(4):
    name='phase8_matched_restarts_ci_787_shard'+str(i)
    restart_packs.append(manifest(name))
    report=json.loads((dev/name/'evidence/lifecycle_rows.json').read_text())
    assert report['executed_code_sha']=='787ebf692a8d5058050553f4160b78af062f1d48'
    restart_rows.extend(report['rows'])
require_registry('restarts',restart_rows)
runtime=json.loads((dev/'phase8_runtime_dispatch_reconstruction.json').read_text())
assert runtime['three_fixed_policy_reports_exact'] and runtime['all_independent_fixed_policy_repeats_exact']

phases={}
def phase(n,passed,note,packs=(),extra=()):
    phases[str(n)]={'classification':'EXECUTED_PASS' if passed else 'EXECUTED_FAIL',
        'interpretation':note,'source_records':[*(manifest(name) for name in packs),*extra]}

phase(1,True,'All 19 bad triangles diagnosed; inherited near old tip, earlier unavailable lineage explicitly UNKNOWN.',
      ['phase1_quality_diagnosis'])
phase(2,True,'Two materially distinct constrained strategies executed; physical transaction/source qualification separately bound in Phase 5.',
      ['phase2_quality_strategies','phase5_positive_clean_38a'])
phase(3,False,'Independent manufactured/Kirsch/recovery A/B executed. Fine recovery converges, but practical 32/64/128 source identity and kinetic transfer remain unqualified; raw fine route separately qualifies.',
      ['phase3_operator_a','phase3_operator_b','phase3_kirsch_coarse_a','phase3_kirsch_coarse_b',
       'phase3_kirsch_fine_a','phase3_kirsch_fine_b','phase4_recovery_budget_a','phase4_recovery_budget_b'])
phase(4,True,'Prospective complete transfer budget defined before qualifying executions; recovered transfer failures are preserved, not fitted away.',
      ['phase4_recovery_budget_a','phase4_recovery_budget_b','phase5_positive_clean_38a'])
phase(5,True,'Clean qualified fine source, real preserved-threshold first passage, sole child, checkpoint/rollback exercised; coarse negative retained.',
      ['phase5_positive_clean_38a','phase8_rollback_ci_facbc'],rollback_refs)
phase(6,False,'All six actual source interventions executed: source separation and zero-drive pass, doubled owned child radius leaves continuation rate unchanged. Distinct r_tip != R_void does not qualify radius-dependent physics.',
      ['phase6_causality_clean_38a'])
phase(7,False,'All 45 physical constant-source transitions executed. Coupled variable-rate/remesh-path partition invariance remains unqualified; original strict reconstruction classification is retained.',
      ['phase7_natural_harness_352','phase7_transitions_ci_ec796_shard0','phase7_transitions_ci_ec796_shard1','phase7_transitions_ci_ec796_shard2'],transition_refs)
phase(8,bool(restart['science_passed'] and rollback['science_passed']),
      'All 11 matched-return direct/replay continuations and all 39 reached rollback injections executed and audited. Mixed native-kernel common-terminal failure is retained; three-worker fixed-kernel intervention is exact but is not replacement physical evidence.',
      ['phase8_stronger_reload_two_level_c284','phase8_matched_return_source_v2_787','phase8_runtime_dispatch_interventions_c8b'],
      [*restart_packs,*restart_refs,*rollback_refs,reference(dev/'phase8_runtime_dispatch_reconstruction.json')])
phase(9,False,'All 96 unique static solves and independent reconstruction completed: 434/537 family predicates, 0/33 full families, 8/12 derivative checks. Historical 683/791 unchanged.',
      ['phase9_static_96_a34'],[reference(dev/'phase9_static_reconstruction.json')])
phase(10,False,'Four observed causal prefixes exact; full prospectively required future-sequence/restart neutrality fails. Historical raw full-state identity remains FAIL.',
      ['phase10_causal_a34'])
report={'schema':'v5.source-resolution-development-phase-ledger/1',
    'record_kind':'HASH_BOUND_DEVELOPMENT_CLASSIFICATION_NOT_FINAL_AB_EVIDENCE',
    'baseline_sha':'c4fbd2fec207d1a0964db1a02069af62bc0aacde','phases':phases,
    'development_execution_complete':True,'scientific_qualification':'BLOCKED',
    'final_complete_AB_started':False,'release_candidate_permitted':False,
    'next_required_action':'One full same-head final A/B campaign under frozen numerical kernels, then exact publication audit and final PR ledger.'}
output=root/'development_phase_status.json'
if output.exists():raise ValueError('refusing to overwrite readiness')
output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
require_development_complete(output)
print(json.dumps({'readiness':str(output),'phase_classifications':{key:value['classification'] for key,value in phases.items()}}))
