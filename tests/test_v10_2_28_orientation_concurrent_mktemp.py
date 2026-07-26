from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_v10_2_28_paper_four_class_1000um_orientation.sh"


def test_orientation_wrapper_exports_macos_safe_scheduler_mktemp(tmp_path):
    command = r'''
set -euo pipefail
source <(sed '/^exec bash /d' "$LAUNCHER")
template="$TMPROOT/.v10_2_28_four_class_orientation_scheduler.XXXXXX.sh"
first=$(mktemp "$template")
second=$(mktemp "$template")
test "$first" != "$second"
test -f "$first"
test -f "$second"
case "$first" in *.sh) ;; *) exit 11 ;; esac
case "$second" in *.sh) ;; *) exit 12 ;; esac
rm -f "$first" "$second"
'''
    environment = dict(os.environ)
    environment.update(
        {
            "LAUNCHER": str(LAUNCHER),
            "TMPROOT": str(tmp_path),
        }
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
