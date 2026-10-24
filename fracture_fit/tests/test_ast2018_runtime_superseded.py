"""Runtime accounting includes the profile-seeded and superseded stages."""
import json

from experimental_fits.ast2018.runtime import account


def test_runtime_counts_profile_seeded_and_archived_confirmations(tmp_path):
    local=tmp_path/'spatial_local_figure5_alternative_profile_seeded.jsonl'
    local.write_text(json.dumps({'runtime_s':2.0,'case_calls':11,
                                 'completed_unix_s':10.0,'error':None})+'\n')
    archived=tmp_path/'superseded_figure5_alternative_before_profile_seeded'
    final=archived/'confirmation_figure5_alternative'
    final.mkdir(parents=True)
    (final/'summary.json').write_text(json.dumps([
        {'runtime_s':3.0,'status':'complete'}]))
    stages=account(tmp_path)['stages']
    assert stages['spatial_local_profile_seeded_figure5_alternative']['forward_case_calls']==11
    assert stages['confirmation_figure5_alternative_superseded']['forward_case_calls']==1
