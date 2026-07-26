from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
INSTALLER = ROOT / "scripts" / "install_v10_2_27_four_class_registry.py"


def _canonical_options() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("v10227_four_class_installer", INSTALLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(item[0] for item in module.CANONICAL_OPTIONS)


def test_launcher_has_valid_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_uses_current_canonical_four_class_rows():
    source = LAUNCHER.read_text()
    match = re.search(r'^OPTIONS=\$\{OPTIONS:-"([^"]+)"\}$', source, re.MULTILINE)
    assert match is not None
    assert tuple(match.group(1).split()) == _canonical_options()


def test_launcher_locks_requested_geometry_and_direct_provider():
    source = LAUNCHER.read_text()
    required_tokens = (
        "TARGET_EXT_UM=${TARGET_EXT_UM:-1000}",
        "THETA=${THETA:-30}",
        "--process-zone-length-um 50",
        "--process-zone-bins 80",
        "--mesh-nx 36",
        "--mesh-ny 72",
        "--tip-h-fine-um 1",
        "--tip-ratio 1.20",
        "--da-phys-um 5",
        "scripts/ensure_v10_2_28_signed_kernel.py",
        "v10.2.28_direct_prescribed_geometry_fem_v1",
        "v10.2.28_paper_four_class_theta30_1000um_campaign_lock_v1",
        "arrhenius_fracture.sharp_front_v10_2_28_audited",
    )
    for token in required_tokens:
        assert token in source


def test_launcher_prebuilds_before_cases_and_supports_preflight_only():
    source = LAUNCHER.read_text()
    ensure_index = source.index("scripts/ensure_v10_2_28_signed_kernel.py")
    scheduler_index = source.index("bash \"$generated_scheduler\"")
    assert ensure_index < scheduler_index
    assert "PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}" in source
    assert "PREFLIGHT_COMPLETE: direct kernel locked" in source
    assert "KERNEL_STRICT_FAMILY_OVERRIDE=1" in source


def test_stale_scheduler_rows_are_replaced_only_as_compatibility_inputs():
    source = LAUNCHER.read_text()
    assert source.count("v913_paper_weakT01_0257068_persistent_sites") == 1
    assert source.count("v913_paper_ceramic01_0189364_persistent_sites") == 1
    assert '"v913_paper_weakT01_0129902_persistent_sites"' in source
    assert '"v913_paper_ceramic01_0077080_persistent_sites"' in source
