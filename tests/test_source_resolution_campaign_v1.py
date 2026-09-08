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


@pytest.mark.parametrize('child,passed',(('',True),('<skipped/>',False),('<failure/>',False)))
def test_focused_requires_executed_pass_not_skip_or_missing(tmp_path,child,passed):
    path=tmp_path/'run.xml';path.write_text('<testsuites><testsuite><testcase classname="tests.test_voiding_v5" name="test_case">'
        +child+'</testcase></testsuite></testsuites>')
    expected=['tests/test_voiding_v5.py::test_case']
    assert summarize(path,expected)['passed'] is passed
    assert not summarize(path,expected+['tests/test_voiding_v5.py::missing'])['passed']
