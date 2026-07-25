from __future__ import annotations

import json
from pathlib import Path

import pytest

from arrhenius_fracture.sharp_front_v10_1_7_3 import (
    _rewrite_summary_event_semantics,
)


def _write_case(root: Path, *, n_advances: int) -> None:
    root.mkdir(parents=True)
    (root / "stochastic_avalanche_geometry_events.json").write_text("[]\n")
    (root / "summary.json").write_text(
        json.dumps([{"n_advances": n_advances}]) + "\n"
    )


def test_zero_event_summary_is_recorded_explicitly(tmp_path):
    case = tmp_path / "case"
    _write_case(case, n_advances=0)

    _rewrite_summary_event_semantics(
        ["--out", str(case), "--da-phys", "5e-6"]
    )

    row = json.loads((case / "summary.json").read_text())[0]
    assert row["n_geometry_events"] == 0
    assert row["n_equivalent_checkpoints_exact"] == 0.0
    assert row["n_equivalent_checkpoints_rounded"] == 0
    assert row["nominal_checkpoint_length_m"] == pytest.approx(5.0e-6)
    assert row["geometry_path_length_m"] == 0.0
    assert row["geometry_projected_extension_m"] == 0.0
    assert row["zero_geometry_events_validated_against_zero_advances"] is True


def test_empty_geometry_rejects_nonzero_advances(tmp_path):
    case = tmp_path / "case"
    _write_case(case, n_advances=1)

    with pytest.raises(RuntimeError, match="inconsistent"):
        _rewrite_summary_event_semantics(
            ["--out", str(case), "--da-phys", "5e-6"]
        )
