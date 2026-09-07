#!/usr/bin/env python3
"""One fresh, non-resumable worker launched by the R/nominal-K controller."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--contract",type=Path,required=True); a=ap.parse_args()
    contract=json.loads(a.contract.read_text()); out=Path(contract["result_path"])
    if out.exists():
        raise SystemExit("fresh worker refuses an existing result path")
    out.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update(contract["environment"])
    for key in list(env):
        if "RESUME" in key.upper() or "RESTART" in key.upper():
            env.pop(key,None)
    started=time.time()
    with Path(contract["log_path"]).open("w") as log:
        proc=subprocess.run(["bash","scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],env=env,
                            stdout=log,stderr=subprocess.STDOUT)
    terminal={"schema":"R_nominal_worker_terminal_v1","job_id":contract["job_id"],
              "pid":os.getpid(),"exit_code":proc.returncode,"wall_seconds":time.time()-started,
              "fresh_virgin_start":True,"resume":False}
    atomic_json(Path(contract["terminal_path"]),terminal)
    return proc.returncode


if __name__=="__main__": raise SystemExit(main())

