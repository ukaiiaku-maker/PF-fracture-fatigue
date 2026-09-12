"""Read-only reconstruction of the trusted authorized rollback CI bundle."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd()/'scripts'))
from validate_v5_source_resolution_evidence_v1 import verify_inventory, same
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.closure_lifecycle_evidence import conservation, stagewise_topology, lifecycle_decision
from arrhenius_fracture.closure_rollback_matrix_v5 import ROLLBACK_STAGES
from arrhenius_fracture.finalization_v3_schema import canonical_hash

root=Path(sys.argv[1]);verify_inventory(root);root=root/'evidence';verify_inventory(root)
report=json.loads((root/'lifecycle_rows.json').read_text())
rows=report['rows']
assert len(rows)==39 and all(r['dataset']=='rollback' for r in rows)
expected={'lifecycle:'+s for s in ROLLBACK_STAGES}
expected.update('promotion:'+s for s in ('field_projection','support_rebuild','equilibrium'))
expected.update('ligament:'+s for s in ('graph_edit','remesh','field_projection','support_rebuild','equilibrium','energy_gate','connected_surface_certification','dormant_support_rebuild'))
assert len(expected)==39 and {r['case_identity'] for r in rows}==expected
assert len({r['execution_id'] for r in rows})==39
class Sources:
    def __getitem__(self, key):
        p=(root/key).resolve()
        assert p.is_relative_to(root.resolve())
        return restore_checkpoint(p)
sources=Sources()
for row in rows:
    before=sources[row['initial_checkpoint']];after=sources[row['terminal_checkpoint']]
    assert row['executed_code_sha']==report['executed_code_sha']
    assert row['input_hash']==canonical_hash(row['input_configuration'])
    a,b=fingerprint(before),fingerprint(after)
    assert a==row['initial_fingerprint'] and b==row['terminal_fingerprint']
    assert row['restored_exactly']==(a==b)
    same(conservation(after,before),row['conservation'],'conservation source mismatch')
    same(stagewise_topology(after),row['stagewise_topology'],'independent topology mismatch')
    cfg=row['input_configuration'];stage=cfg['failure_stage']
    assert cfg['intended_stage_reached'] and stage in row['actual_operations']
    assert row['failure']['message']=='injected:'+stage
    print('RECONSTRUCTED '+row['case_identity'],flush=True)
decision=lifecycle_decision(rows,sources)
same(decision,report['decision'],'source-derived rollback decision mismatch')
print(json.dumps({'record_kind':'READ_ONLY_DEVELOPMENT_RECONSTRUCTION_NOT_NEW_PHYSICAL_EXECUTION',
    'source_sha':report['executed_code_sha'],'source_manifest_sha256':hashlib.sha256((root/'sha256_manifest.json').read_bytes()).hexdigest(),
    'valid':True,'actual_attempts':39,'passed':sum(r['passed'] for r in decision['rollback_attempts']),
    'downstream_rollback':decision['downstream_lifecycle_rollback'],
    'stagewise_topology_and_conservation':decision['stagewise_topology_and_conservation']}),flush=True)
