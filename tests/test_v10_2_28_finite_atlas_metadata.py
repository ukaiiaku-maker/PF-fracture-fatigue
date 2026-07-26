from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas_finite_metadata.py"
SPEC = importlib.util.spec_from_file_location("finite_atlas_metadata_v10228", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_review_diagnostics_may_become_json_null():
    payload = {
        "input_states": [
            {
                "maximum_relative_load_variation": math.inf,
                "maximum_within_load_relative_spread": float("nan"),
                "coefficient": 1.25,
            }
        ]
    }
    sanitized, replacements = MODULE._sanitize(payload)
    row = sanitized["input_states"][0]
    assert row["maximum_relative_load_variation"] is None
    assert row["maximum_within_load_relative_spread"] is None
    assert row["coefficient"] == pytest.approx(1.25)
    assert replacements == 2


def test_nonfinite_kernel_or_geometry_value_remains_fatal():
    with pytest.raises(ValueError, match="outside permitted review diagnostics"):
        MODULE._sanitize({"active_I": math.inf})


def test_direct_builder_routes_extended_assembler_through_metadata_wrapper():
    source = (
        ROOT / "scripts" / "build_v10_2_28_prescribed_geometry_kernel_numpy2.py"
    ).read_text()
    assert "build_v10_2_27_extended_active_only_atlas_finite_metadata.py" in source
    assert "subprocess.run = _run_with_finite_atlas_metadata" in source
