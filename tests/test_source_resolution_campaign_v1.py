import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v5_source_resolution_campaign_matrix_v1 import matrix,require_development_complete
from validate_v5_source_resolution_evidence_v1 import verify_inventory
from qualify_v5_source_resolution_focused_results_v1 import summarize


def test_final_matrix_is_disjoint_and_complete():
    rows=matrix();assert len(rows)==28 and len({r['id'] for r in rows})==28
    assert sum(r['phase']=='static' for r in rows)==8
    for section,count in (('transitions',3),('restarts',4),('controlled',4),('natural',4),('neutrality',1),('rollback',1)):
        observed=[r for r in rows if r['phase']=='lifecycle' and r['section']==section]
        assert sorted(r['index'] for r in observed)==list(range(count))
        assert all(r['count']==count for r in observed)


def test_readiness_allows_honest_scientific_failure_but_not_unexecuted_phases(tmp_path):
    path=tmp_path/'readiness.json'
    report={'schema':'v5.source-resolution-development-phase-ledger/1','phases':{
        str(n):{'classification':'EXECUTED_FAIL','source_records':['retained-source.json']} for n in range(1,11)}}
    path.write_text(json.dumps(report));assert require_development_complete(path)==report
    report['phases']['7']['classification']='NOT_EXERCISED';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='not executed'):require_development_complete(path)


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
