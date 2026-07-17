import importlib.util
import json
from pathlib import Path

import arrhenius_fracture
from arrhenius_fracture.sharp_front_v10_1_6 import _write_refresh_scale_alias


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "scripts" / "repair_v10_1_6_matrix_audit_aliases.py"
WRAPPER = ROOT / "scripts" / "run_v10_1_6_1_temperature_matrix.sh"


def _repair_module():
    spec = importlib.util.spec_from_file_location("v10161_repair", REPAIR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v10161():
    assert arrhenius_fracture.__version__ == "10.1.9"


def test_refresh_scale_alias_is_written_without_removing_canonical_key(tmp_path: Path):
    out = tmp_path / "case"
    out.mkdir()
    path = out / "v10_1_driver_modes.json"
    path.write_text(json.dumps({"campaign_refresh_length_scale": 1.25}))

    _write_refresh_scale_alias(["--out", str(out)])

    payload = json.loads(path.read_text())
    assert payload["campaign_refresh_length_scale"] == 1.25
    assert payload["campaign_refresh_scale"] == 1.25
    assert payload["matrix_audit_key_compatibility"] == "v10.1.6.1"


def test_refresh_scale_alias_is_noop_without_output(tmp_path: Path):
    _write_refresh_scale_alias([])
    assert list(tmp_path.iterdir()) == []


def test_existing_case_repair_adds_alias(tmp_path: Path):
    case = tmp_path / "full" / "DBTT" / "T300_th45"
    case.mkdir(parents=True)
    path = case / "v10_1_driver_modes.json"
    path.write_text(json.dumps({"campaign_refresh_length_scale": 1.0}))

    scanned, changed = _repair_module().repair(tmp_path)

    payload = json.loads(path.read_text())
    assert scanned == 1
    assert changed == 1
    assert payload["campaign_refresh_scale"] == 1.0


def test_resume_wrapper_repairs_then_delegates():
    text = WRAPPER.read_text()
    assert "repair_v10_1_6_matrix_audit_aliases.py" in text
    assert "run_v10_1_6_temperature_matrix.sh" in text
    assert "exec bash" in text
