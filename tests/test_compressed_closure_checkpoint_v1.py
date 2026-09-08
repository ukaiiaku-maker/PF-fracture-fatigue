from pathlib import Path
import pytest
from arrhenius_fracture.checkpoint_v11 import write_checkpoint,restore_checkpoint
from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def test_compressed_checkpoint_is_exact_deterministic_and_legacy_compatible(tmp_path):
    state,_=build_production_void_state(); expected=fingerprint(state)
    first=write_checkpoint(state,tmp_path/'a.json',compression='gzip')
    second=write_checkpoint(state,tmp_path/'b.json',compression='gzip')
    legacy=write_checkpoint(state,tmp_path/'old.json')
    assert first['state_sha256']==second['state_sha256']
    assert 'state_encoding' not in legacy
    assert all(fingerprint(restore_checkpoint(tmp_path/name))==expected for name in ('a.json','b.json','old.json'))
    blob=tmp_path/first['state_file'];blob.write_bytes(blob.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):restore_checkpoint(tmp_path/'a.json')
