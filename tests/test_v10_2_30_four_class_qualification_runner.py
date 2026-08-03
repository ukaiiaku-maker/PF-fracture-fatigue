from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "scripts/run_v10_2_30_three_deltaK_energy_gate_qualification.sh"
FOUR_CLASS = (
    ROOT
    / "scripts/run_v10_2_30_four_class_three_deltaK_energy_gate_qualification.sh"
)


def test_four_class_qualification_runner_has_valid_shell_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(FOUR_CLASS.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_four_class_runner_uses_exact_canonical_options_and_references():
    text = FOUR_CLASS.read_text()
    for option in (
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
    ):
        assert option in text
    for variable in (
        "PEAK_REFERENCE_ROOT",
        "DBTT_REFERENCE_ROOT",
        "WEAKT_REFERENCE_ROOT",
        "CERAMIC_REFERENCE_ROOT",
    ):
        assert variable in text
    assert "run_v10_2_30_three_deltaK_energy_gate_qualification.sh" in text
    assert "v10_2_30_four_class_qualification_gate.json" in text


def test_four_class_gate_requires_each_class_to_bracket_and_converge():
    text = FOUR_CLASS.read_text()
    assert 'errors.append(f"{label}: no propagated qualification case")' in text
    assert 'errors.append(f"{label}: no censored qualification case")' in text
    assert 'errors.append(f"{label}: trial-fraction convergence failed")' in text
    assert "no positive committed event was energy-truncated" in text
    assert "class seed namespaces are not unique" in text


def test_single_class_runner_remains_the_low_level_primitive():
    text = SINGLE.read_text()
    assert "PARAMETER_OPTION=${PARAMETER_OPTION:-" in text
    assert "REFERENCE_ROOT" in text
    assert "CONVERGENCE_TRIAL_FRACTION" in text
