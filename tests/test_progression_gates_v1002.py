from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_progression_shell_scripts_parse():
    for name in (
        "run_v10_0_2_three_class_progression.sh",
        "run_v10_0_2_50um_700K.sh",
        "run_v10_0_2_wake_ablation_700K.sh",
    ):
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_progression_uses_validated_transfer_modes_and_resolved_mesh():
    text = (SCRIPTS / "run_v10_0_2_three_class_progression.sh").read_text()
    assert "--bulk-plasticity-mode tip_only" in text
    assert "--directional-j-mode root_signed" in text
    assert "TIP_H_FINE=${TIP_H_FINE:-5e-7}" in text
    assert "NX=${NX:-48}" in text
    assert "NY=${NY:-96}" in text
    assert "--max-fronts 1" in text


def test_progression_does_not_mislabel_da_as_numerical_convergence():
    text = (SCRIPTS / "run_v10_0_2_three_class_progression.sh").read_text()
    assert "requires DA_PHYS_M=5e-6" in text
    assert "Physical renewal length and numerical geometry substep" in text


def test_wake_ablation_is_true_on_off_routing():
    text = (SCRIPTS / "run_v10_0_2_three_class_progression.sh").read_text()
    assert 'if [[ "$WAKE_SHIELDING" == "1" ]]' in text
    assert "wake_args+=(--wake-shielding)" in text
    assert "wake_args+=(--no-wake-shielding)" in text
    assert "wake routing audit failed" in text
    ablation = (SCRIPTS / "run_v10_0_2_wake_ablation_700K.sh").read_text()
    assert "for WAKE in 1 0" in ablation
