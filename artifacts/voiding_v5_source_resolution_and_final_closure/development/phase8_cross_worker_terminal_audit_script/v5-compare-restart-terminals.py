import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint

def state(shard):
    root=Path('/private/tmp/v5-common-restarts-ci-787')/shard/'evidence'
    row=json.loads((root/'lifecycle_rows.json').read_text())['rows'][0]
    return row,restore_checkpoint(root/row['terminal_checkpoint'])

differences=[]
def compare(a,b,path):
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        if a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True):return
        entry={'path':path,'kind':'array','shape_a':a.shape,'shape_b':b.shape,'dtype_a':str(a.dtype),'dtype_b':str(b.dtype)}
        if a.shape==b.shape:
            entry.update(count_different=int(np.count_nonzero(a!=b)),max_absolute_difference=float(np.max(np.abs(a-b))))
        differences.append(entry);return
    if hasattr(a,'to_dict') and hasattr(b,'to_dict'):return compare(a.to_dict(),b.to_dict(),path)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        return compare({k:getattr(a,k) for k in a.__dataclass_fields__},{k:getattr(b,k) for k in b.__dataclass_fields__},path)
    if isinstance(a,Mapping) and isinstance(b,Mapping):
        for k in sorted(a.keys()|b.keys(),key=str):
            if k not in a or k not in b:differences.append({'path':path+'/'+str(k),'kind':'missing','in_a':k in a,'in_b':k in b})
            else:compare(a[k],b[k],path+'/'+str(k))
        return
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        if len(a)>40 or len(b)>40:
            aa=json.dumps(a,default=str,sort_keys=True);bb=json.dumps(b,default=str,sort_keys=True)
            if aa!=bb:differences.append({'path':path,'kind':'long_sequence','length_a':len(a),'length_b':len(b),'sha_a':hashlib.sha256(aa.encode()).hexdigest(),'sha_b':hashlib.sha256(bb.encode()).hexdigest()})
            return
        if len(a)!=len(b):differences.append({'path':path,'kind':'length','a':len(a),'b':len(b)})
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        return
    if isinstance(a,np.generic):a=a.item()
    if isinstance(b,np.generic):b=b.item()
    if isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b):return
    if a!=b:differences.append({'path':path,'a':a,'b':b})

ra,a=state('shard2');rb,b=state('shard3')
for name in a.__dataclass_fields__:
    compare(getattr(a,name),getattr(b,name),'/'+name)
result={'source_sha':'787ebf692a8d5058050553f4160b78af062f1d48','case_a':ra['case_identity'],'case_b':rb['case_identity'],
        'fingerprint_a':complete_accepted_state_fingerprint(a),'fingerprint_b':complete_accepted_state_fingerprint(b),
        'difference_count':len(differences),'differences':differences}
Path('/private/tmp/v5-restart-cross-worker-differences.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(result,indent=2,default=str),flush=True)
