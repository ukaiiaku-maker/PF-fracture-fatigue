import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_v5_final_campaign_v1 import pack,unpack,inventory


def source(tmp_path):
    root=tmp_path/'source';root.mkdir()
    (root/'paired_comparison.json').write_text(json.dumps({'executed_code_sha':'a'*40,'classification':'PASS'}))
    (root/'a').mkdir();(root/'b').mkdir()
    data=bytes(range(256))*20
    for side in ('a','b'):(root/side/'actual-checkpoint.bin').write_bytes(data)
    long_name='source-bound-checkpoint-'+'x'*110+'.json'
    (root/'a'/long_name).write_text('{"retained":"exact bytes"}')
    (root/'sha256_manifest.json').write_text(json.dumps(inventory(root),sort_keys=True))
    return root


def test_complete_archive_is_deterministic_bounded_and_lossless(tmp_path):
    original=source(tmp_path)
    a=pack(original,tmp_path/'a','a'*40,part_bytes=91,code_tree='b'*64)
    b=pack(original,tmp_path/'b','a'*40,part_bytes=91,code_tree='b'*64)
    assert a==b and inventory(tmp_path/'a')==inventory(tmp_path/'b')
    assert len(a['parts'])>1 and all(p['bytes']<=91 for p in a['parts'])
    assert unpack(tmp_path/'a',tmp_path/'restored')==a
    assert inventory(original)==inventory(tmp_path/'restored')


def test_archive_rejects_changed_payload_before_extraction(tmp_path):
    original=source(tmp_path);report=pack(original,tmp_path/'a','a'*40,part_bytes=91,code_tree='b'*64)
    (tmp_path/'a'/report['parts'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='inventory'):unpack(tmp_path/'a',tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_archive_rejects_unregistered_source_and_overwrite(tmp_path):
    original=source(tmp_path)
    pack(original,tmp_path/'a','a'*40,code_tree='b'*64)
    with pytest.raises(ValueError,match='overwrite'):pack(original,tmp_path/'a','a'*40,code_tree='b'*64)
    (original/'unregistered').write_text('not in the original manifest')
    with pytest.raises(ValueError,match='inventory'):pack(original,tmp_path/'b','a'*40,code_tree='b'*64)


def test_archive_rejects_traversal_even_with_matching_storage_hashes(tmp_path):
    import gzip,hashlib,io,tarfile
    original=source(tmp_path);report=pack(original,tmp_path/'a','a'*40,code_tree='b'*64)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        entry=tarfile.TarInfo('../escaped');entry.size=3
        archive.addfile(entry,io.BytesIO(b'bad'))
    path=tmp_path/'a'/report['parts'][0]['path'];path.write_bytes(gzip.compress(buffer.getvalue(),mtime=0))
    report['parts']=[{'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
    (tmp_path/'a/publication.json').write_text(json.dumps(report))
    (tmp_path/'a/sha256_manifest.json').write_text(json.dumps({k:v for k,v in inventory(tmp_path/'a').items() if k!='sha256_manifest.json'}))
    with pytest.raises(ValueError,match='unsafe'):unpack(tmp_path/'a',tmp_path/'restored')
    assert not (tmp_path/'escaped').exists()
