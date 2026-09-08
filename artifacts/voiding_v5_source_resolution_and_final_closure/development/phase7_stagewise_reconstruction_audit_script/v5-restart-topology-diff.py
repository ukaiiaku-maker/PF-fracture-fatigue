import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.closure_lifecycle_evidence import stagewise_topology
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/v5-common-restarts-ci-787/shard2/evidence')
sys.path.insert(0,str(Path.cwd()/'scripts'))
from validate_v5_source_resolution_evidence_v1 import verify_inventory
verify_inventory(p)
report=json.loads((p/'lifecycle_rows.json').read_text())
r=next(row for row in report['rows'] if row['case_identity']==sys.argv[2]) if len(sys.argv)>2 else report['rows'][0]
s=restore_checkpoint(p/r['terminal_checkpoint'])
a=canonical_data(stagewise_topology(s));b=r['stagewise_topology'];differences=[]
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:differences.append((path+'/'+k,'MISSING',k in a,k in b))
            else:diff(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):differences.append((path,'LENGTH',len(a),len(b)))
        else:
            for i,(x,y) in enumerate(zip(a,b)):diff(x,y,path+'/'+str(i))
    elif a!=b:differences.append((path,a,b))
diff(a,b)
result={'source_sha':report['executed_code_sha'],'case_identity':r['case_identity'],'actual_passed':a['passed'],'recorded_passed':b['passed'],'difference_count':len(differences),'first_differences':differences[:20]}
if len(sys.argv)>3:Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
