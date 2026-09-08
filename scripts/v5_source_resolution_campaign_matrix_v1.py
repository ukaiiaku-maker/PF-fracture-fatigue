#!/usr/bin/env python3
"""Frozen disjoint scheduling of one full source-resolution A/B campaign."""
import argparse
import json
from pathlib import Path


def matrix():
    rows=[]
    for index in range(8):rows.append({'id':'static-'+str(index),'phase':'static','section':'all','index':index,'count':8})
    for section,count in (('transitions',3),('restarts',4),('controlled',4),('natural',4),('neutrality',1),('rollback',1)):
        for index in range(count):rows.append({'id':section+'-'+str(index),'phase':'lifecycle','section':section,'index':index,'count':count})
    for phase in ('source','recovery','causal-neutrality'):
        rows.append({'id':phase,'phase':phase,'section':'all','index':0,'count':1})
    return rows


def require_development_complete(path):
    record=json.loads(Path(path).read_text())
    if record.get('schema')!='v5.source-resolution-development-phase-ledger/1':raise ValueError('wrong readiness schema')
    if set(record.get('phases',{}))!={str(n) for n in range(1,11)}:raise ValueError('all ten development phases required')
    for number,row in record['phases'].items():
        if row.get('classification') not in ('EXECUTED_PASS','EXECUTED_FAIL') or not row.get('source_records'):
            raise ValueError('phase '+number+' not executed and classified')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--readiness',type=Path);args=parser.parse_args()
    if args.readiness:require_development_complete(args.readiness)
    print(json.dumps({'include':matrix()},separators=(',',':')))
