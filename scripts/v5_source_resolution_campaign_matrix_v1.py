#!/usr/bin/env python3
"""Frozen disjoint scheduling of one full source-resolution A/B campaign."""
import argparse
import hashlib
import json
from pathlib import Path


def matrix():
    rows=[]
    for index in range(8):rows.append({'id':'static-'+str(index),'phase':'static','section':'all','index':index,'count':8})
    for section,count in (('transitions',3),('restarts',4),('controlled',4),('natural',4),('neutrality',1),('rollback',1)):
        for index in range(count):rows.append({'id':section+'-'+str(index),'phase':'lifecycle','section':section,'index':index,'count':count})
    for phase in ('source','recovery','causal-neutrality'):
        rows.append({'id':phase,'phase':phase,'section':'all','index':0,'count':1})
    # Start the measured long-running source/restart operations first. This
    # changes queue order only, never a case, shard identity or execution count.
    priority={'source':0,'restarts':1,'rollback':2,'causal-neutrality':3,
        'transitions':4,'controlled':5,'natural':6,'static':7,'neutrality':8,'recovery':9}
    return sorted(rows,key=lambda row:(priority[row['section'] if row['phase']=='lifecycle'
        else row['phase']],row['index']))


def require_development_complete(path):
    path=Path(path);record=json.loads(path.read_text())
    if record.get('schema')!='v5.source-resolution-development-phase-ledger/1':raise ValueError('wrong readiness schema')
    if set(record.get('phases',{}))!={str(n) for n in range(1,11)}:raise ValueError('all ten development phases required')
    for number,row in record['phases'].items():
        if row.get('classification') not in ('EXECUTED_PASS','EXECUTED_FAIL') or not row.get('source_records'):
            raise ValueError('phase '+number+' not executed and classified')
        for source in row['source_records']:
            if not isinstance(source,dict) or set(source)!={'path','sha256'}:
                raise ValueError('phase '+number+' requires hash-bound source records')
            relative=Path(source['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('readiness source must remain under its evidence root')
            target=path.parent/relative
            if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError('readiness source missing or hash mismatch: '+source['path'])
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--readiness',type=Path);args=parser.parse_args()
    if args.readiness:require_development_complete(args.readiness)
    print(json.dumps({'include':matrix()},separators=(',',':')))
