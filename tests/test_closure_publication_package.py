import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('closure_package',Path(__file__).resolve().parents[1]/'scripts/assemble_voiding_v5_closure_publication.py')
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)


def test_publication_pair_inventory_includes_exact_manifest_bytes(tmp_path):
    (tmp_path/'source.json').write_text('{}\n')
    manifest=package.inventory(tmp_path)
    (tmp_path/'sha256_manifest.json').write_text(json.dumps(manifest))
    assert set(package.verified(tmp_path))=={'source.json','sha256_manifest.json'}
    (tmp_path/'source.json').write_text('{"tampered":true}\n')
    with pytest.raises(ValueError,match='manifest mismatch'):package.verified(tmp_path)


def test_publication_refuses_symlink_before_reading_source(tmp_path):
    (tmp_path/'link').symlink_to(tmp_path/'unavailable')
    with pytest.raises(ValueError,match='symlink'):package.verified(tmp_path)


@pytest.mark.parametrize('downstream_exercised',[False,True])
def test_observed_accounting_does_not_certify_unexercised_lifecycle(downstream_exercised):
    decision={'stagewise_topology_and_conservation':True,'transition_partitions':[
        {'execution_id':str(i),'actual_transition':i<35 or downstream_exercised} for i in range(45)]}
    result=package.accounting_coverage(decision)
    assert result['all_observed_states_passed']
    assert result['complete_lifecycle_passed'] is downstream_exercised
    assert len(result['unexercised_transition_execution_ids'])==(0 if downstream_exercised else 10)
