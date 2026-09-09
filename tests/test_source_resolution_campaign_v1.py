import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v5_source_resolution_campaign_matrix_v1 import matrix,require_development_complete
from validate_v5_source_resolution_evidence_v1 import verify_inventory
from qualify_v5_source_resolution_focused_results_v1 import summarize
from validate_v5_development_lifecycle_shards_v1 import expected_registry,require_registry,reconstruct_all_rows
from assemble_v5_source_resolution_shards_v1 import copy_owned
from recover_v5_causal_neutrality_postprocess_v1 import require_failed_execution
import assemble_v5_source_resolution_shards_v1 as shard_assembler
import classify_v5_exact_head_reconstruction_v1 as exact_head_classifier


def test_owned_assembly_storage_preserves_bytes_and_rejects_conflicts(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'original scientific bytes')
    target=tmp_path/'assembly'/'owned'
    assert copy_owned(source,target)==str(target)
    assert source.stat().st_ino==target.stat().st_ino
    assert source.read_bytes()==target.read_bytes()==b'original scientific bytes'
    copy_owned(source,target)
    conflicting=tmp_path/'conflicting';conflicting.write_bytes(b'different evidence')
    with pytest.raises(ValueError,match='conflicting owned'):copy_owned(conflicting,target)
    assert source.read_bytes()==b'original scientific bytes'


def test_owned_assembly_cross_filesystem_fallback_and_symlink_guard(monkeypatch,tmp_path):
    import assemble_v5_source_resolution_shards_v1 as module
    import errno
    source=tmp_path/'source';source.write_bytes(b'original')
    def cross_device(*args,**kwargs):raise OSError(errno.EXDEV,'cross device')
    monkeypatch.setattr(module.os,'link',cross_device)
    target=tmp_path/'copy';copy_owned(source,target)
    assert source.read_bytes()==target.read_bytes()
    link=tmp_path/'link';link.symlink_to(source)
    with pytest.raises(ValueError,match='symlink'):copy_owned(link,tmp_path/'unsafe')


def test_sdk_cleanup_refuses_to_run_in_a_user_environment():
    import os,subprocess
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_v5_ci_storage_v1.sh'
    environment={**os.environ,'GITHUB_ACTIONS':'false','RUNNER_ENVIRONMENT':'self-hosted','RUNNER_OS':'Linux'}
    result=subprocess.run(['bash',str(script)],env=environment,text=True,capture_output=True)
    assert result.returncode==2 and 'Refusing SDK cleanup' in result.stderr


@pytest.mark.parametrize('section,count',[('transitions',45),('restarts',11),('rollback',39)])
def test_development_reconstruction_requires_every_registered_actual_case(section,count):
    registry=expected_registry(section);assert len(registry)==count
    rows=[{'dataset':section,'case_identity':case,'partition_count':partition,'execution_id':str(i)}
        for i,(case,partition) in enumerate(sorted(registry))]
    assert len(require_registry(section,rows))==count
    with pytest.raises(ValueError,match='incomplete'):require_registry(section,rows[:-1])
    with pytest.raises(ValueError,match='aliased'):require_registry(section,rows+[rows[0]])


def test_final_matrix_is_disjoint_and_complete():
    rows=matrix();assert len(rows)==35 and len({r['id'] for r in rows})==35
    assert rows[0]['id']=='source'
    assert [r['id'] for r in rows[1:12]]==['restarts-'+str(i) for i in range(11)]
    assert sum(r['phase']=='static' for r in rows)==8
    for section,count in (('transitions',3),('restarts',11),('controlled',4),('natural',4),('neutrality',1),('rollback',1)):
        observed=[r for r in rows if r['phase']=='lifecycle' and r['section']==section]
        assert sorted(r['index'] for r in observed)==list(range(count))
        assert all(r['count']==count for r in observed)


def test_development_reconstruction_continues_independent_rows_without_hiding_failure():
    rows=[{'execution_id':str(n),'dataset':'restarts','case_identity':str(n),'partition_count':None} for n in range(3)]
    visited=[]
    def validator(selected,sources,*,executed_code_sha):
        visited.append(selected[0]['execution_id'])
        if selected[0]['execution_id']=='0':raise ValueError('strict topology mismatch')
    audited=reconstruct_all_rows(rows,{},'a'*40,validator)
    assert visited==['0','1','2']
    assert [r['valid'] for r in audited]==[False,True,True]
    assert audited[0]['failure']=={'type':'ValueError','message':'strict topology mismatch'}


def test_readiness_allows_honest_scientific_failure_but_not_unexecuted_phases(tmp_path):
    path=tmp_path/'readiness.json'
    source=tmp_path/'retained-source.json';source.write_text('{"actual_failure":"frozen_gate"}')
    reference={'path':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    report={'schema':'v5.source-resolution-development-phase-ledger/1','phases':{
        str(n):{'classification':'EXECUTED_FAIL','source_records':[reference]} for n in range(1,11)}}
    path.write_text(json.dumps(report));assert require_development_complete(path)==report
    report['phases']['7']['classification']='NOT_EXERCISED';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='not executed'):require_development_complete(path)


def test_readiness_cannot_reference_missing_or_altered_sources(tmp_path):
    path=tmp_path/'readiness.json'
    source=tmp_path/'source.json';source.write_text('{}')
    report={'schema':'v5.source-resolution-development-phase-ledger/1','phases':{
        str(n):{'classification':'EXECUTED_FAIL','source_records':[{'path':source.name,
            'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]} for n in range(1,11)}}
    path.write_text(json.dumps(report));source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash mismatch'):require_development_complete(path)
    source.unlink()
    with pytest.raises(ValueError,match='missing'):require_development_complete(path)


def test_manifest_binds_nested_manifests_and_all_files(tmp_path):
    nested=tmp_path/'inner';nested.mkdir();(nested/'sha256_manifest.json').write_text('{}')
    inventory={'inner/sha256_manifest.json':hashlib.sha256(b'{}').hexdigest()}
    (tmp_path/'sha256_manifest.json').write_text(json.dumps(inventory))
    assert verify_inventory(tmp_path)==inventory
    (nested/'sha256_manifest.json').write_text('{"altered":true}')
    with pytest.raises(ValueError,match='inventory/hash'):verify_inventory(tmp_path)


def test_causal_neutrality_recovery_accepts_only_the_exact_incomplete_operation():
    sha='a'*40
    report={'executed_code_sha':sha,'execution_completed':False,'phase':'causal-neutrality','section':'all',
        'clean_exact_head_at_end':True,'operations':[{'operation':'causal_neutrality',
        'script':'qualify_v5_disabled_causal_neutrality_v1.py','returncode':1}]}
    assert require_failed_execution(report,sha)==report
    for changed in ({**report,'execution_completed':True},{**report,'phase':'source'},
                    {**report,'clean_exact_head_at_end':False},{**report,'operations':[]}):
        with pytest.raises(ValueError):require_failed_execution(changed,sha)


def test_strict_ontology_failure_is_retained_without_predicate_relaxation(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('natural internal-stage independent production replay mismatch: exact bits')
    monkeypatch.setattr(shard_assembler,'validate_closure_evidence',fail)
    result=shard_assembler.classify_ontology({},object(),'a'*40)
    assert result['valid'] is False and result['classification']=='EXECUTED_BLOCKED'
    assert result['predicate_relaxed'] is False
    assert result['failure']['type']=='ValueError'
    assert 'independent production replay mismatch' in result['failure']['message']


def test_unexpected_ontology_failure_remains_a_workflow_error(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('missing checkpoint')
    monkeypatch.setattr(shard_assembler,'validate_closure_evidence',fail)
    with pytest.raises(ValueError,match='missing checkpoint'):
        shard_assembler.classify_ontology({},object(),'a'*40)


def test_exact_head_known_scientific_failure_is_terminal_blocked(monkeypatch,tmp_path):
    sha='a'*40;implementation=tmp_path/'implementation';scripts=implementation/'scripts';scripts.mkdir(parents=True)
    validator=scripts/'validate_v5_complete_published_campaign_v1.py';validator.write_text('# frozen validator\n')
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'paired_comparison.json').write_text(json.dumps({'executed_code_sha':sha,
        'exact_recursive_comparison':True,'classification':'PASS'}))
    publication=tmp_path/'publication.json';publication.write_text('{}')
    monkeypatch.setattr(exact_head_classifier,'git',lambda *args:sha if args[1:] == ('rev-parse','HEAD') else '')
    class Completed:
        returncode=1;stdout='';stderr='ValueError: natural internal-stage independent production replay mismatch: exact bits\n'
    monkeypatch.setattr(exact_head_classifier.subprocess,'run',lambda *args,**kwargs:Completed())
    output=tmp_path/'audit.json'
    result=exact_head_classifier.classify(implementation,evidence,publication,sha,output)
    assert result['classification']=='EXECUTED_BLOCKED' and result['exact_head_audit_completed'] is True
    assert result['evidence_valid'] is False and result['predicate_relaxed'] is False
    assert json.loads(output.read_text())==result


@pytest.mark.parametrize('child,passed',(('',True),('<skipped/>',False),('<failure/>',False)))
def test_focused_requires_executed_pass_not_skip_or_missing(tmp_path,child,passed):
    path=tmp_path/'run.xml';path.write_text('<testsuites><testsuite><testcase classname="tests.test_voiding_v5" name="test_case">'
        +child+'</testcase></testsuite></testsuites>')
    expected=['tests/test_voiding_v5.py::test_case']
    assert summarize(path,expected)['passed'] is passed
    assert not summarize(path,expected+['tests/test_voiding_v5.py::missing'])['passed']
