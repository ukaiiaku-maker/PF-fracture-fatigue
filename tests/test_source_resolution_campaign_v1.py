import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v5_source_resolution_campaign_matrix_v1 import matrix,require_development_complete
from validate_v5_source_resolution_evidence_v1 import verify_inventory
from qualify_v5_source_resolution_focused_results_v1 import summarize
from validate_v5_development_lifecycle_shards_v1 import expected_registry,require_registry


@pytest.mark.parametrize('section,count',[('transitions',45),('restarts',11),('rollback',39)])
def test_development_reconstruction_requires_every_registered_actual_case(section,count):
    registry=expected_registry(section);assert len(registry)==count
    rows=[{'dataset':section,'case_identity':case,'partition_count':partition,'execution_id':str(i)}
        for i,(case,partition) in enumerate(sorted(registry))]
    assert len(require_registry(section,rows))==count
    with pytest.raises(ValueError,match='incomplete'):require_registry(section,rows[:-1])
    with pytest.raises(ValueError,match='aliased'):require_registry(section,rows+[rows[0]])


def test_final_matrix_is_disjoint_and_complete():
    rows=matrix();assert len(rows)==28 and len({r['id'] for r in rows})==28
    assert rows[0]['id']=='source'
    assert [r['id'] for r in rows[1:5]]==['restarts-'+str(i) for i in range(4)]
    assert sum(r['phase']=='static' for r in rows)==8
    for section,count in (('transitions',3),('restarts',4),('controlled',4),('natural',4),('neutrality',1),('rollback',1)):
        observed=[r for r in rows if r['phase']=='lifecycle' and r['section']==section]
        assert sorted(r['index'] for r in observed)==list(range(count))
        assert all(r['count']==count for r in observed)


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
