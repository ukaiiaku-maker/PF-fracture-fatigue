from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_audited_smoke_runner_completes_one_fake_case(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_v10_2_21_top1_persistent_site_audited_smoke.sh"
    registry = (
        root
        / "arrhenius_fracture"
        / "data"
        / "materials"
        / "v10_2_21_v912_top1_persistent_site_registry.csv"
    )
    family = tmp_path / "family.json"
    family.write_text("{}\n")

    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == '-u' && ${2:-} == '-m' ]]; then
  shift 3
  out=''
  while [[ $# -gt 0 ]]; do
    if [[ $1 == '--out' ]]; then
      out=$2
      shift 2
    else
      shift
    fi
  done
  [[ -n $out ]]
  mkdir -p "$out"
  printf 'step,sigma_back_Pa,lambda_e,mpz_available_site_fraction\n0,0,0,1\n' \\
    > "$out/steps_900K.csv"
  exit 0
fi

if [[ ${1:-} == *'classify_v10_2_15_stage3_case.py' ]]; then
  shift
  case_root=''
  while [[ $# -gt 0 ]]; do
    if [[ $1 == '--case-root' ]]; then
      case_root=$2
      shift 2
    else
      shift
    fi
  done
  [[ -n $case_root ]]
  printf '{"complete": true}\n' > "$case_root/stage3_case_status.json"
  : > "$case_root/COMPLETE"
  exit 0
fi

if [[ ${1:-} == '-' ]]; then
  printf '1\n'
  exit 0
fi

echo "unexpected fake-python invocation: $*" >&2
exit 3
"""
    )
    fake_python.chmod(0o755)

    outroot = tmp_path / "smoke"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "CONDA_ENV": "arrhenius-sharp-front-v10",
            "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10",
            "OUTROOT": str(outroot),
            "FAMILY_JSON": str(family),
            "REGISTRY": str(registry),
            "TEMPS": "900",
            "MAX_JOBS": "1",
            "TARGET_EXT_UM": "1",
            "STEPS": "1",
            "THETA": "45",
        }
    )
    completed = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "FINISHED: T=900K" in completed.stdout
    assert "complete=1 failed=0" in completed.stdout
    assert "unbound variable" not in completed.stderr
    case = outroot / "T900K_th45_seed3621"
    assert (case / "COMPLETE").is_file()
    assert not (case / "RUN_FAILED").exists()
