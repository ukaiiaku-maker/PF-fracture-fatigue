from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registered_capture_does_not_precreate_snapshot_root() -> None:
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert 'ITER_SNAPSHOTS="$ITER_ROOT/snapshots"' in text
    assert 'rm -rf "$ITER_SNAPSHOTS"' in text
    assert 'mkdir -p "$(dirname "$ITER_SNAPSHOTS")"' in text
    assert 'mkdir -p "$ITER_SNAPSHOTS"' not in text
    assert 'V10227_KERNEL_CAPTURE_OUTROOT="$ITER_SNAPSHOTS"' in text


def test_registered_capture_must_emit_complete_provenance() -> None:
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert 'capture_complete.json' in text
    assert 'kernel_capture_manifest.json' in text
    assert 'check_v10_2_27_active_endpoint_resolution.py' in text
    assert 'check_v10_2_27_capture_physics_contract.py' in text
