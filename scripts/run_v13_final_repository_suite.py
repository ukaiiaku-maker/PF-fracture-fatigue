"""One complete repository-suite invocation, only after the held-out queue ends."""
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys

from scripts.run_v13_heldout_materials import OUT,DEST,PLAN,ROOT,verify_source,verify_accepted
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json


def main():
    p=verify_source();before=verify_accepted()
    if shutil.disk_usage(OUT).free<4*1024**3:raise RuntimeError('less than 4 GiB durable publication/test headroom')
    q=json.loads((DEST/'queue_status.json').read_text())
    assert not q['active'] and not q['pending'] and not q['paused']
    for case in p['cases']:
        t=json.loads((DEST/case/'terminal.json').read_text())
        assert t['status'] in ('TERMINATED','EXISTING_GATE_STOP')
    command=[sys.executable,'-m','pytest','-q','--junitxml='+str(OUT/'full_suite.xml')]
    start=datetime.now(timezone.utc).isoformat()
    with (OUT/'full_suite_claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),command=command,start_utc=start,full_suite_invocations=1,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),stream,indent=2)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/pf-current-source-v13-parent-mpl')
    atomic_json(OUT/'full_suite_environment.json',{k:env.get(k) for k in ('PYTHONPATH','PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MPLCONFIGDIR','TMPDIR')})
    with (OUT/'full_suite.log').open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        after=verify_accepted()
    except Exception as exc:
        after=dict(status='FAIL',reason=str(exc))
    atomic_json(OUT/'full_suite_result.json',dict(returncode=result.returncode,command=command,start_utc=start,
        end_utc=datetime.now(timezone.utc).isoformat(),full_suite_invocations=1,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        collection_or_environment_failure=result.returncode in (2,3,4,5),
        accepted_freeze_before=before,accepted_freeze_after=after))
    print(json.dumps(dict(returncode=result.returncode,simulation_program_stopped=True)))
    return result.returncode if after.get('status')!='FAIL' else 1


if __name__=='__main__':sys.exit(main())
