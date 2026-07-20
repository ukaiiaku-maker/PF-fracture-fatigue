from __future__ import annotations

import json

from arrhenius_fracture import sharp_front_v10_1_7_3 as entry73
from arrhenius_fracture import zero_event_summary_v10215  # noqa: F401


def test_zero_event_geometry_is_valid_right_censored_summary(tmp_path):
    (tmp_path / "stochastic_avalanche_geometry_events.json").write_text("[]\n")
    (tmp_path / "summary.json").write_text(
        json.dumps([{"T_K": 700.0, "n_advances": 0, "Kc_first": None}]) + "\n"
    )

    entry73._rewrite_summary_event_semantics(
        ["--out", str(tmp_path), "--da-phys", "5e-6"]
    )

    row = json.loads((tmp_path / "summary.json").read_text())[0]
    assert row["n_geometry_events"] == 0
    assert row["n_equivalent_checkpoints_exact"] == 0.0
    assert row["n_equivalent_checkpoints_rounded"] == 0
    assert row["nominal_checkpoint_length_m"] == 5.0e-6
    assert row["geometry_path_length_m"] == 0.0
    assert row["geometry_projected_extension_m"] == 0.0
    assert row["geometry_event_status"] == "no_accepted_events"
    assert row["zero_event_run_is_valid"] is True


def test_non_list_geometry_still_fails(tmp_path):
    (tmp_path / "stochastic_avalanche_geometry_events.json").write_text("{}\n")
    (tmp_path / "summary.json").write_text(json.dumps([{"T_K": 700.0}]) + "\n")
    try:
        entry73._rewrite_summary_event_semantics(
            ["--out", str(tmp_path), "--da-phys", "5e-6"]
        )
    except RuntimeError as exc:
        assert "must be a list" in str(exc)
    else:
        raise AssertionError("malformed geometry diagnostics must fail")
