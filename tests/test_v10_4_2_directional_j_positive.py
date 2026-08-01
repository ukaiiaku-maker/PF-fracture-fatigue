from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.directional_j_positive_v1042 import (
    effective_directional_J,
    transform_source,
)
from arrhenius_fracture.reuse_v1041_v1042 import (
    audit_positive_directional_J_history,
)

ROOT = Path(__file__).resolve().parents[1]


def test_effective_directional_J_keeps_negative_work_non_driving():
    assert effective_directional_J(-0.07157286780799363) == 0.0
    assert effective_directional_J(0.0) == 0.0
    assert effective_directional_J(24130077.750773273) == 24130077.750773273
    assert effective_directional_J(-2.0, allow_abs=True) == 2.0


def test_transformed_solver_removes_first_nonzero_sign_latch():
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)
    compile(transformed, "sharp_front.py[v10.4.2-positive-J-test]", "exec")
    assert "positive_raw_signed_J_is_forward_configurational_work" in transformed
    assert "J_eff = max(J_signed, 0.0)" in transformed
    assert "_J_SIGN_REF['sign'] == 0.0" not in transformed
    assert "first_nonzero_sign_latch_used'] = False" in transformed
    assert "J_eff = abs(J_signed)" in transformed  # explicit ablation remains available


def _write_front_table(path: Path, rows: list[tuple[float, ...]]) -> None:
    header = (
        "step,front_id,n_fire,J_signed_trial,J_effective_trial,J_sign_ref"
    )
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=",", header=header, comments="")


def test_reuse_audit_accepts_positive_signed_history(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    _write_front_table(
        case / "fronts_1000K.csv",
        [
            (1, 0, 0, -0.01, 0.0, 1.0),
            (2, 0, 0, 0.25, 0.25, 1.0),
            (3, 0, 1, 4.0, 4.0, 1.0),
        ],
    )
    audit = audit_positive_directional_J_history(case)
    assert audit["compatible"] is True
    assert audit["first_passage_step"] == 3
    assert audit["required_relation"] == "J_effective=max(J_signed,0)"


def test_reuse_audit_rejects_first_nonzero_negative_sign_latch(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    _write_front_table(
        case / "fronts_1000K.csv",
        [
            (1, 0, 0, -0.01, 0.01, -1.0),
            (2, 0, 0, 0.25, 0.0, -1.0),
            (3, 0, 1, 4.0, 0.0, -1.0),
        ],
    )
    with pytest.raises(ValueError, match="incompatible with positive signed directional J"):
        audit_positive_directional_J_history(case)
