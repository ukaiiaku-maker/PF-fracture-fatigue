from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v10_2_27_signed_kernel_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("kernel_coverage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family(path: Path, maximum_um: float) -> Path:
    payload = {
        "schema": "v10.2.14_active_only_real_signed_2d_shielding_atlas",
        "states": [
            {"crack_extension_m": 0.0},
            {"crack_extension_m": maximum_um * 1.0e-6},
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_800um_family_rejected_for_1000um_at_30deg(tmp_path: Path) -> None:
    module = _load()
    report = module.coverage_report(
        _family(tmp_path / "family.json", 800.0),
        target_extension_um=1000.0,
        theta_deg=30.0,
        da_phys_um=5.0,
        event_minimum_factor=0.5,
        event_maximum_factor=4.0,
        margin_events=1.0,
    )
    assert report["pass"] is False
    assert math.isclose(
        report["required_atlas_max_crack_path_extension_um"],
        1173.0792591442503,
        rel_tol=1.0e-12,
    )


def test_1200um_family_passes_for_1000um_at_30deg(tmp_path: Path) -> None:
    module = _load()
    report = module.coverage_report(
        _family(tmp_path / "family.json", 1200.0),
        target_extension_um=1000.0,
        theta_deg=30.0,
        da_phys_um=5.0,
        event_minimum_factor=0.5,
        event_maximum_factor=4.0,
        margin_events=1.0,
    )
    assert report["pass"] is True
    assert report["coverage_margin_um"] > 0.0
