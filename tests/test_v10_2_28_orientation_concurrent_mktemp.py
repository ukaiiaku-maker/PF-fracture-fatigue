from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"


def test_orientation_launcher_uses_portable_mktemp_suffix_sequence():
    source = IMPLEMENTATION.read_text()
    assert (
        'generated_scheduler_base=$(mktemp "$ROOT/scripts/'
        '.v10_2_28_four_class_orientation_scheduler.XXXXXX")'
        in source
    )
    assert 'generated_scheduler="${generated_scheduler_base}.sh"' in source
    assert 'mv "$generated_scheduler_base" "$generated_scheduler"' in source
    assert ".v10_2_28_four_class_orientation_scheduler.XXXXXX.sh" not in source


def test_portable_scheduler_sequence_creates_unique_shell_paths(tmp_path):
    command = r'''
set -euo pipefail
template="$TMPROOT/.v10_2_28_four_class_orientation_scheduler.XXXXXX"
base1=$(mktemp "$template")
first="${base1}.sh"
mv "$base1" "$first"
base2=$(mktemp "$template")
second="${base2}.sh"
mv "$base2" "$second"
test "$first" != "$second"
test -f "$first"
test -f "$second"
case "$first" in *.sh) ;; *) exit 11 ;; esac
case "$second" in *.sh) ;; *) exit 12 ;; esac
rm -f "$first" "$second"
'''
    environment = dict(os.environ)
    environment["TMPROOT"] = str(tmp_path)
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
