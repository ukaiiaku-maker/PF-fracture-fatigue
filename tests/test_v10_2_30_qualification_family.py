import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def module():
    path = ROOT / "scripts/v10230_qualification_family.py"
    spec = importlib.util.spec_from_file_location("qualification_family", path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def test_authoritative_family_validates_exact_identity():
    result = module().validate()
    assert result["observed_sha256"] == "5aa74370e6419104684c52bbcf93323f905d2752d1bba59252f9b0b35c77e07c"
    assert result["observed_state_count"] == 7
    assert result["observed_coverage_m"] == 0.001175


def test_family_validation_fails_closed_on_changed_content(tmp_path):
    source = tmp_path / "family.json"; source.write_text("{}")
    descriptor = json.loads((ROOT / "runtime_inputs/v10_2_30/qualification_family_manifest.json").read_text())
    descriptor["path"] = str(source)
    path = tmp_path / "descriptor.json"; path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="hash or size"):
        module().validate(path)
