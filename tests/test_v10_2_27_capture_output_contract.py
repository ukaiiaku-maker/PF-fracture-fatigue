from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registered_capture_does_not_precreate_snapshot_root() -> None:
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert 'SNAPSHOT_ROOT="$CACHE_DIR/snapshots"' in text
    assert 'rm -rf "$SNAPSHOT_ROOT"' in text
    assert 'mkdir -p "$(dirname "$SNAPSHOT_ROOT")"' in text
    assert 'mkdir -p "$SNAPSHOT_ROOT"' not in text
    assert "PhysicalFEMCapture owns creation of the output root" in text


def test_registered_capture_must_emit_complete_provenance() -> None:
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert 'capture_complete.json' in text
    assert 'kernel_capture_manifest.json' in text
    assert 'check_v10_2_27_active_endpoint_resolution.py' in text
    assert 'check_v10_2_27_capture_physics_contract.py' in text
