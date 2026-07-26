from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_v10_2_28_four_class_KJ_temperature_plots.sh"


def test_wrapper_has_valid_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_wrapper_separates_tail_policy_outputs():
    source = WRAPPER.read_text()
    assert 'POLICIES=${POLICIES:-"maximum fraction"}' in source
    assert 'maximum_${TAIL_LENGTH_UM}um_or_${TAIL_FRACTION}fraction' in source
    assert 'length_${TAIL_LENGTH_UM}um' in source
    assert 'fraction_${TAIL_FRACTION}' in source
    assert '--plot-dir "$plot_dir"' in source
    assert 'PLOT_COMPLETE: policy=$policy' in source
    assert 'ALL_PLOTS_COMPLETE:' in source
