#!/usr/bin/env python3
"""Lossless, deterministic, size-bounded publication of the complete CI bundle.

No scientific JSON or checkpoint is rewritten. Archive parts are storage only;
the original complete inventory is checked again after safe extraction.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='v5.complete-campaign-lossless-publication/1'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def inventory(root):
    paths=sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):raise ValueError('symlink in campaign')
    return {str(p.relative_to(root)):digest(p) for p in paths if p.is_file()}


def executable_tree(sha):
    if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('exact implementation SHA required')
    records=subprocess.check_output(('git','ls-tree','-rz','--full-tree',sha),cwd=ROOT).split(b'\0')
    # The later publication may add evidence, never change executable inputs.
    records=[r for r in records if r and not r.split(b'\t',1)[1].startswith(b'artifacts/')]
    return hashlib.sha256(b'\0'.join(records)+b'\0').hexdigest()


class Parts(io.RawIOBase):
    def __init__(self,root,limit):
        self.root=root;self.limit=limit;self.current=None;self.used=0;self.total=0;self.paths=[]
    def writable(self):return True
    def tell(self):return self.total
    def write(self,data):
        length=len(data);offset=0
        while offset<length:
            if self.current is None or self.used==self.limit:
                if self.current is not None:self.current.close()
                path=self.root/('evidence.tar.gz.part'+str(len(self.paths)+1).zfill(4))
                self.paths.append(path);self.current=path.open('xb');self.used=0
            count=min(length-offset,self.limit-self.used)
            self.current.write(data[offset:offset+count]);offset+=count;self.used+=count;self.total+=count
        return length
    def flush(self):
        if self.current is not None and not self.current.closed:self.current.flush()
    def close(self):
        if self.current is not None:self.current.close()
        super().close()


def pack(source,output,implementation_sha,*,part_bytes=80*1024*1024,code_tree=None):
    if output.exists():raise ValueError('refusing to overwrite publication')
    if part_bytes<=0:raise ValueError('positive archive-part size required')
    files=inventory(source)
    expected=json.loads((source/'sha256_manifest.json').read_text())
    if {k:v for k,v in files.items() if k!='sha256_manifest.json'}!=expected:
        raise ValueError('original complete campaign inventory mismatch')
    paired=json.loads((source/'paired_comparison.json').read_text())
    if paired['executed_code_sha']!=implementation_sha:raise ValueError('campaign implementation mismatch')
    output.mkdir(parents=True)
    with Parts(output,part_bytes) as sink:
        with gzip.GzipFile(fileobj=sink,mode='wb',mtime=0,filename='') as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
                for name in files:
                    path=source/name;entry=tarfile.TarInfo(name);entry.size=path.stat().st_size
                    entry.mode=0o644;entry.mtime=0;entry.uid=entry.gid=0;entry.uname=entry.gname=''
                    with path.open('rb') as stream:archive.addfile(entry,stream)
        sink.flush()
        parts=[{'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sink.paths]
    report={'schema':SCHEMA,'record_kind':'LOSSLESS_PACKAGING_NOT_NEW_PHYSICAL_EXECUTION',
        'implementation_sha':implementation_sha,'executable_tree_sha256':code_tree or executable_tree(implementation_sha),
        'part_limit_bytes':part_bytes,'parts':parts,'original_complete_inventory':files,
        'original_paired_comparison':paired}
    (output/'publication.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (output/'sha256_manifest.json').write_text(json.dumps(inventory(output),sort_keys=True,indent=2)+'\n')
    return report


class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                path=next(self.paths,None)
                if path is None:return 0
                self.current=path.open('rb')
            count=self.current.readinto(buffer)
            if count:return count
            self.current.close();self.current=None
    def close(self):
        if self.current is not None:self.current.close()
        super().close()


def unpack(source,output):
    if output.exists():raise ValueError('refusing to overwrite extracted evidence')
    actual=inventory(source);expected=json.loads((source/'sha256_manifest.json').read_text())
    if {k:v for k,v in actual.items() if k!='sha256_manifest.json'}!=expected:
        raise ValueError('publication inventory mismatch')
    report=json.loads((source/'publication.json').read_text())
    if report['schema']!=SCHEMA:raise ValueError('publication schema mismatch')
    names=[part['path'] for part in report['parts']]
    if names!=['evidence.tar.gz.part'+str(i+1).zfill(4) for i in range(len(names))] or not names:
        raise ValueError('archive part ordering/identity mismatch')
    for part in report['parts']:
        path=source/part['path']
        if path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:
            raise ValueError('archive part integrity mismatch')
    output.mkdir(parents=True);seen=set()
    with Joined([source/name for name in names]) as joined:
        with tarfile.open(fileobj=io.BufferedReader(joined),mode='r|gz') as archive:
            for entry in archive:
                path=Path(entry.name)
                if not entry.isfile() or path.is_absolute() or '..' in path.parts or entry.name in seen:
                    raise ValueError('unsafe or duplicate archive member')
                if entry.name not in report['original_complete_inventory']:raise ValueError('unregistered archive member')
                seen.add(entry.name);target=output/path;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(entry) as original,target.open('xb') as destination:
                    for block in iter(lambda:original.read(1024*1024),b''):destination.write(block)
    if inventory(output)!=report['original_complete_inventory']:raise ValueError('lossless complete inventory mismatch')
    return report


def verify_publication_binding(report,publication_sha):
    implementation=report['implementation_sha']
    subprocess.run(('git','merge-base','--is-ancestor',implementation,publication_sha),cwd=ROOT,check=True)
    if executable_tree(implementation)!=report['executable_tree_sha256'] or executable_tree(publication_sha)!=report['executable_tree_sha256']:
        raise ValueError('publication changed executable inputs')
    changed=subprocess.check_output(('git','diff','--name-only',implementation,publication_sha),cwd=ROOT,text=True).splitlines()
    if any(not p.startswith('artifacts/voiding_v5_source_resolution_and_final_closure/') for p in changed):
        raise ValueError('publication changed inputs outside authorized evidence root')
    return {'publication_sha':publication_sha,'implementation_sha':implementation,'executable_inputs_exact':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('pack','unpack'))
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--implementation-sha');parser.add_argument('--verify-publication-head',action='store_true')
    args=parser.parse_args()
    result=pack(args.source,args.output,args.implementation_sha) if args.operation=='pack' else unpack(args.source,args.output)
    if args.verify_publication_head:
        sha=subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip()
        print(json.dumps(verify_publication_binding(result,sha),sort_keys=True))
    else:print(json.dumps({'implementation_sha':result['implementation_sha'],'files':len(result['original_complete_inventory'])},sort_keys=True))
