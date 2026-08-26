from pathlib import Path
import subprocess


def test_four_class_monotonic_parity_runner_is_bounded_and_direct():
    path = Path("scripts/run_v10_2_30_four_class_monotonic_parity.sh")
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "sharp_front_v10_2_28_audited" in text
    assert "sharp_front_v10_2_29_fatigue_audited" in text
    assert "validate_monotonic_pair" in text
    assert "peak v913_paper_peak01_0242980" in text
    assert "dbtt v913_paper_dbtt01_0202500" in text
    assert "weakt v913_paper_weakT01_0129902" in text
    assert "ceramic v913_paper_ceramic01_0077080" in text
    assert "rm -rf" not in text
