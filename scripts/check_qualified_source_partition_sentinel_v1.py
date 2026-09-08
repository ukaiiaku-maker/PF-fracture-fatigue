#!/usr/bin/env python3
"""Half-passage owned-source partitions: no child or broad lifecycle replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.voiding_production_v5 import _complete_next_clock,cavity_source_resolution_metrics,_qualified_cavity_source,directional_clock_rates
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('checkpoint',type=Path);args=parser.parse_args()
original=restore_checkpoint(args.checkpoint);m=cavity_source_resolution_metrics(original);tensor=np.asarray(m['tensor_Pa'])
assert _qualified_cavity_source(original,tensor)
cavity=original.void_state.cavities[0]
total=.5*min(row['crossing_time_s'] for row in directional_clock_rates(original,tensor));rows=[]
for parts in (1,2,4,8,16):
    current=original
    for _ in range(parts):
        current,audit=_complete_next_clock(current,tensor,maximum_advance_duration_s=total/parts,
            source_kind='cavity_surface',source_cavity_id=cavity.cavity_id,source_boundary_site_id='connection_exit',
            source_position_m=cavity.connection_exit_m,source_probe_identity={'kind':'direct_cavity_boundary_tensor',
                'boundary_node_id':m['boundary_node_id'],'element_ids':m['probe_element_ids']})
        assert all(not row['winner'] and row['effective_rate_s']>0 for row in audit)
    rows.append({'partitions':parts,'fingerprint':fingerprint(current),
        'source_remains_qualified':_qualified_cavity_source(current,tensor)})
print(json.dumps(rows,indent=2));assert len({r['fingerprint'] for r in rows})==1 and all(r['source_remains_qualified'] for r in rows)
