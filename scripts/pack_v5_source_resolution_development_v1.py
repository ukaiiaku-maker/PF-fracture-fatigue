#!/usr/bin/env python3
"""Preserve development results, optionally losslessly repacking large checkpoints.

This is packaging, not an additional physical execution or clean-run attestation.
Every changed checkpoint representation has original hashes and an exact complete
state fingerprint comparison. Original source trees are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.checkpoint_v11 import SCHEMA,restore_checkpoint,write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def pack(source,destination):
    if destination.exists():raise ValueError('refusing to overwrite packaged evidence')
    files=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):raise ValueError('development source contains symlink')
    base=source.parent if source.is_file() else source
    inventory={str(p.relative_to(base)):digest(p) for p in files}
    # Verify supplied inventories before restoring any trusted local pickle.
    for path in files:
        if path.name=='sha256_manifest.json':
            expected=json.loads(path.read_text())
            observed={str(p.relative_to(path.parent)):digest(p) for p in files
                if p.is_relative_to(path.parent) and p!=path}
            if observed!=expected:raise ValueError('original development manifest mismatch '+str(path))
    destination.mkdir(parents=True);converted=[];skip=set()
    for path in files:
        if path.suffix!='.json' or path.name=='sha256_manifest.json':continue
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or data.get('schema')!=SCHEMA:continue
        blob=path.with_name(data['state_file'])
        if blob.stat().st_size<90*1024*1024 or data.get('state_encoding')=='pickle+gzip/1':continue
        state,runtime=restore_checkpoint(path,with_provider_runtime=True)
        target=destination/path.relative_to(base)
        packed=write_checkpoint(state,target,provider_runtime=runtime,compression='gzip')
        if fingerprint(restore_checkpoint(target))!=fingerprint(state):raise ValueError('repacked checkpoint changed accepted state')
        skip.update((path,blob))
        converted.append({'checkpoint':str(path.relative_to(base)),'original_manifest_sha256':digest(path),
            'original_payload_sha256':digest(blob),'compressed_payload_sha256':packed['state_sha256'],
            'complete_state_fingerprint':fingerprint(state),'complete_state_exact':True})
        write(destination/'original_checkpoint_manifests'/path.relative_to(base),data)
    for path in files:
        if path in skip:continue
        relative=path.relative_to(base)
        if path.name=='sha256_manifest.json':relative=relative.with_name('retained_source_sha256_manifest.json')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(destination/'packaging_provenance.json',{'schema':'v5.retained-development-packaging/1',
        'record_kind':'PACKAGING_NOT_NEW_PHYSICAL_EXECUTION_OR_CLEAN_ATTESTATION','original_inventory':inventory,
        'lossless_checkpoint_representation_changes':converted})
    write(destination/'sha256_manifest.json',{str(p.relative_to(destination)):digest(p)
        for p in sorted(destination.rglob('*')) if p.is_file()})
    return {'source_file_count':len(files),'repacked_checkpoints':len(converted)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(pack(args.source,args.output)))
