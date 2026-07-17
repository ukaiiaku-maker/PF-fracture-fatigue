import json
from pathlib import Path

import arrhenius_fracture
from arrhenius_fracture.sharp_front_v10_1_6 import _write_refresh_scale_alias


def test_version_is_v10161():
    assert arrhenius_fracture.__version__ == "10.1.6.1"


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
