from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ensure_v10_2_27_signed_kernel.py"
SPEC = importlib.util.spec_from_file_location("ensure_v10227", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_mode_rejects_snapshot_root_reuse(monkeypatch):
    monkeypatch.setenv("KERNEL_SNAPSHOT_ROOT", "/tmp/preserved_snapshots")
    with pytest.raises(SystemExit, match="cannot be combined"):
        MODULE._install_automatic_endpoint_resolution([
            "--theta-deg", "18",
            "--target-extension-um", "50",
            "--mode", "build",
        ])


def test_build_mode_rejects_load_root_reuse(monkeypatch):
    monkeypatch.setenv("KERNEL_LOAD_INVARIANCE_ROOT", "/tmp/preserved_loads")
    with pytest.raises(SystemExit, match="cannot be combined"):
        MODULE._install_automatic_endpoint_resolution([
            "--theta-deg", "18",
            "--target-extension-um", "50",
            "--mode", "build",
        ])


def test_auto_mode_allows_snapshot_root_reuse(monkeypatch):
    monkeypatch.setenv("KERNEL_SNAPSHOT_ROOT", "/tmp/preserved_snapshots")
    result = MODULE._install_automatic_endpoint_resolution([
        "--theta-deg", "18",
        "--target-extension-um", "50",
        "--mode", "auto",
    ])
    assert "--tip-h-fine-um" in result
