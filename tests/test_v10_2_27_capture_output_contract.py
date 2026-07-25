from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_configuration_capture_does_not_precreate_snapshot_root() -> None:
    text = (
        ROOT / "scripts" / "capture_v10_2_27_kernel_states_for_configuration.py"
    ).read_text()
    assert "snapshot_out.parent.mkdir(parents=True, exist_ok=True)" in text
    assert "snapshot_out.mkdir(" not in text
    assert "PhysicalFEMCapture owns creation of snapshot_out" in text
