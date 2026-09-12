import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_retained_v3_polygon_vertex_mismatch_is_analytic_and_frozen():
    record = json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v4/v3_geometry_diagnosis.json").read_text()
    )
    radius = 5.5e-5
    sectors = 32
    vertex_radius = radius / math.cos(math.pi / sectors)
    assert vertex_radius == pytest.approx(5.526612148069713e-5, abs=1.0e-20)
    assert vertex_radius - radius == pytest.approx(2.6612148069713e-7, abs=1.0e-20)
    assert record["R_vertex_m"] == pytest.approx(vertex_radius, abs=1.0e-20)
    assert record["R_vertex_minus_R_void_m"] == pytest.approx(
        vertex_radius - radius, abs=1.0e-20
    )
    assert record["classifications"] == {
        "V3_OPERATOR_MANUFACTURED_AND_KIRSCH": "PASS",
        "V3_CENTRAL_DBTT_GEOMETRY_REGISTRATION": (
            "FAIL_POLYGON_VERTEX_VS_NOMINAL_CIRCLE"
        ),
        "V3_CENTRAL_DBTT_TENSOR_CONVERGENCE": "NOT_RUN",
    }
    assert record["retained_v3_files_modified"] is False
