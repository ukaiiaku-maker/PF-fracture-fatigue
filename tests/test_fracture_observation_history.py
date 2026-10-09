import csv
import json

import pytest

from arrhenius_fracture.fracture_observation_history import (
    FractureObservation, LoadDirection, ObservationHistory, ObservationStage,
    from_legacy_steps_csv,
)


def row(sequence, length_um, *, stage="reequilibrated", segment="load-1",
        pre_um=None, K=10.0, valid=True):
    pre_um = length_um if pre_um is None else pre_um
    return FractureObservation(
        sequence=sequence, time_s=float(sequence), cycle=float(sequence),
        segment_id=segment, stage=stage, front_id="primary",
        event_id=f"e{sequence}", load_direction=LoadDirection.LOADING,
        pre_length_m=pre_um * 1e-6, committed_length_m=length_um * 1e-6,
        K_I_Pa_sqrt_m=K * 1e6, K_effective_Pa_sqrt_m=K * 1e6,
        contour_valid=valid, observer_valid=valid,
    )


def test_repeated_lengths_and_load_segments_are_preserved():
    history = ObservationHistory([
        row(0, 10, segment="load-1", K=11),
        row(1, 10, segment="unload-1", K=5),
        row(2, 10, segment="reload-1", K=9),
    ])
    result = history.at_length(10e-6)
    assert result.status == "exact_repeated"
    assert [item.segment_id for item in result.exact] == ["load-1", "unload-1", "reload-1"]
    assert history.at_length(10e-6, segment_id="reload-1").exact[0].K_I_Pa_sqrt_m == 9e6


def test_discrete_jump_is_bracketed_and_never_interpolated():
    history = ObservationHistory([row(0, 8), row(1, 20, pre_um=8, K=14), row(2, 25, K=15)])
    result = history.at_length(12e-6)
    assert result.status == "discrete_jump_bracket"
    assert result.crossing[0].pre_length_m == pytest.approx(8e-6)
    assert result.crossing[0].committed_length_m == pytest.approx(20e-6)
    assert result.interpolation_performed is False


def test_stage_filter_prevents_cross_stage_blending():
    history = ObservationHistory([
        row(0, 10, stage=ObservationStage.PRE_EVENT, K=12),
        row(1, 12, stage=ObservationStage.COMMITTED, pre_um=10, K=12),
        row(2, 12, stage=ObservationStage.REEQUILIBRATED, K=11),
    ])
    assert history.at_length(12e-6, stage="committed").exact[0].K_I_Pa_sqrt_m == 12e6
    assert history.at_length(12e-6, stage="reequilibrated").exact[0].K_I_Pa_sqrt_m == 11e6


def test_invalid_contour_excluded_by_default_but_recoverable():
    history = ObservationHistory([row(0, 10, valid=False)])
    assert history.at_length(10e-6).status == "out_of_range"
    assert history.at_length(10e-6, valid_only=False).status == "exact"


def test_csv_and_jsonl_round_trip(tmp_path):
    source = ObservationHistory([row(0, 10), row(1, 12)])
    for suffix in ("csv", "jsonl"):
        path = tmp_path / f"history.{suffix}"
        getattr(source, f"write_{suffix}")(path)
        loaded = ObservationHistory.load(path)
        assert [item.sequence for item in loaded.rows] == [0, 1]
        assert [item.stage_length_m for item in loaded.rows] == pytest.approx([10e-6, 12e-6])
        assert [item.K_I_Pa_sqrt_m for item in loaded.rows] == pytest.approx([10e6, 10e6])


def test_legacy_steps_adapter_retains_event_interval(tmp_path):
    path = tmp_path / "steps_0300K.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "step", "KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire",
        ])
        writer.writeheader()
        writer.writerow({"step": 0, "KJ_Pa_sqrtm": 9e6, "crack_extension_m": 0, "da_block_m": 0, "n_fire": 0})
        writer.writerow({"step": 1, "KJ_Pa_sqrtm": 12e6, "crack_extension_m": 20e-6,
                         "da_block_m": 20e-6, "n_fire": 1})
    history = from_legacy_steps_csv(path)
    result = history.at_length(10e-6)
    assert result.status == "discrete_jump_bracket"
    assert result.crossing[0].K_I_Pa_sqrt_m == 12e6
