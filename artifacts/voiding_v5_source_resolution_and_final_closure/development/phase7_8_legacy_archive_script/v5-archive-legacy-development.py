"""Losslessly retain an oversized completed legacy development bundle."""
import gzip,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
from package_v5_final_campaign_v1 import Parts,Joined,inventory,digest
from validate_v5_source_resolution_evidence_v1 import verify_inventory
source=Path('/private/tmp/v5-development-phases-7-8-a34')
output=ROOT/'artifacts/voiding_v5_source_resolution_and_final_closure/development/phase7_8_legacy_a34_lossless_archive'
verify_inventory(source);files=inventory(source)
report=json.loads((source/'lifecycle_rows.json').read_text())
sha=report['executed_code_sha'];worker=Path('/private/tmp/v5-source-resolution-worker-a34cdf0')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain'],cwd=worker,text=True).strip()
if output.exists():raise ValueError('refusing to overwrite retained development archive')
output.mkdir(parents=True)
with Parts(output,80*1024*1024) as sink:
    with gzip.GzipFile(fileobj=sink,mode='wb',mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name in files:
                path=source/name;entry=tarfile.TarInfo(name);entry.size=path.stat().st_size
                entry.mode=0o644;entry.mtime=0;entry.uid=entry.gid=0;entry.uname=entry.gname=''
                with path.open('rb') as stream:archive.addfile(entry,stream)
    sink.flush();parts=[{'path':path.name,'sha256':digest(path),'bytes':path.stat().st_size} for path in sink.paths]
observed={}
with io.BufferedReader(Joined([output/part['path'] for part in parts])) as joined:
    with gzip.GzipFile(fileobj=joined,mode='rb') as compressed:
        with tarfile.open(fileobj=compressed,mode='r|') as archive:
            for entry in archive:
                if not entry.isfile() or entry.name in observed or entry.name not in files:raise ValueError('unexpected archive entry')
                h=hashlib.sha256()
                with archive.extractfile(entry) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                observed[entry.name]=h.hexdigest()
assert observed==files
rows=report['rows'];restarts=[row for row in rows if row['dataset']=='restarts'];decision=report['decision']
summary={'schema':'v5.complete-legacy-development-lossless-archive/1',
    'record_kind':'LOSSLESS_STORAGE_NOT_NEW_PHYSICAL_EXECUTION_OR_FINAL_AB_QUALIFICATION',
    'source_sha':sha,'clean_source_worker_at_completion':True,'original_inventory':files,
    'parts':parts,'streaming_roundtrip_every_original_file_exact':True,
    'source_schema':report['schema'],'actual_source_rows':len(rows),
    'transition_cases':len(decision['transition_partitions']),
    'transition_passed':sum(row['passed'] for row in decision['transition_partitions']),
    'actual_transitions':sum(row['actual_transition'] for row in decision['transition_partitions']),
    'rollback_attempts':len(decision['rollback_attempts']),
    'rollback_producer_passed':sum(row['passed'] for row in decision['rollback_attempts']),
    'restart_cases':len(restarts),
    'actual_successful_direct_replay_continuations':sum(row['failure'] is None and row['restart_exact']
        and row['continued_front_terminal_reached'] and row['subsequent_history_exact'] for row in restarts),
    'legacy_zero_drive_case':{'failure':restarts[-1]['failure'],
        'reported_restart_exact':restarts[-1]['restart_exact'],
        'direct_operation_apis':[op['api'] for op in restarts[-1]['actual_operations']],
        'restarted_operation_apis':[op['api'] for op in restarts[-1]['restarted_operations']],
        'interpretation':'Both load preparations ran; direct resume_to_guard failed and the shared try skipped replay resume_to_guard. The reported endpoint equality is NOT a completed exact restart.'},
    'common_terminal_passed':False,'full_closure_campaign':False,
    'source_runner_sha256':digest(worker/'scripts/run_voiding_v5_closure_lifecycle.py')}
(output/'archive.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
(output/'sha256_manifest.json').write_text(json.dumps(inventory(output),sort_keys=True,indent=2)+'\n')
print(json.dumps({key:summary[key] for key in ('source_sha','actual_source_rows','transition_cases','transition_passed','actual_transitions',
    'rollback_attempts','rollback_producer_passed','restart_cases','actual_successful_direct_replay_continuations','streaming_roundtrip_every_original_file_exact')}),flush=True)
