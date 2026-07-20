from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.checked_spatial_station_projection_v10212 import (
    ACCEPTED_INTERACTION_SCHEMAS,
    INTRINSIC_INTERACTION_SCHEMA,
    LEGACY_INTERACTION_SCHEMA,
)


def test_extension_only_builder_accepts_uniform_legacy_or_intrinsic_schema():
    assert LEGACY_INTERACTION_SCHEMA in ACCEPTED_INTERACTION_SCHEMAS
    assert INTRINSIC_INTERACTION_SCHEMA in ACCEPTED_INTERACTION_SCHEMAS
    text = Path("scripts/build_v10_2_13_extension_only_family.py").read_text()
    ast.parse(text)
    assert "ACCEPTED_INTERACTION_SCHEMAS" in text
    assert "selected_interaction_schema" in text
    assert "schemas != {II_MODEL_ID}" not in text


def test_stage3_status_script_parses_and_reports_missing_run(tmp_path: Path):
    script = Path("scripts/status_v10_2_15_stage3.py")
    ast.parse(script.read_text())
    completed = subprocess.run(
        [sys.executable, str(script), "--outroot", str(tmp_path / "absent"), "--tail", "0"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "state:       not_running" in completed.stdout
    assert "launcher:    not running" in completed.stdout


def test_overnight_launcher_writes_durable_status_without_atlas_phase():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "overnight_status.json" in text
    assert "overnight_launcher.pid" in text
    assert "write_status running" in text
    assert "write_status complete" in text
    assert "write_status failed" in text
    assert "write_status assembling" not in text
    assert "PHASE=atlas_assembly" not in text
