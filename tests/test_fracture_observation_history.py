import csv
import json
import math

import numpy as np
import pytest

from arrhenius_fracture.energy_ledger_output_v10227 import online_observation_history
from arrhenius_fracture.fracture_observation_history import (
    FractureObservation, LengthCoordinate, LoadDirection, ObservationHistory,
    ObservationStage, ValidityStatus, from_legacy_steps_csv,
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
    assert history.at_length(10e-6).status == "exact"
    assert history.at_length(10e-6, validity="valid").status == "invalid_or_unqualified_at_length"
    assert history.at_length(10e-6, validity="invalid").status == "exact"


def test_csv_and_jsonl_round_trip(tmp_path):
    source = ObservationHistory([row(0, 10), row(1, 12)])
    for suffix in ("csv", "jsonl"):
        path = tmp_path / f"history.{suffix}"
        getattr(source, f"write_{suffix}")(path)
        loaded = ObservationHistory.load(path)
        assert [item.sequence for item in loaded.rows] == [0, 1]
        assert [item.stage_length_m for item in loaded.rows] == pytest.approx([10e-6, 12e-6])
        assert [item.K_I_Pa_sqrt_m for item in loaded.rows] == pytest.approx([10e6, 10e6])


def test_legacy_steps_adapter_retains_all_rows_and_native_semantics(tmp_path):
    path = tmp_path / "steps_0300K.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "step", "Uapp_m", "KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire",
        ])
        writer.writeheader()
        writer.writerow({"step": 0, "Uapp_m": 1e-6, "KJ_Pa_sqrtm": 9e6,
                         "crack_extension_m": 0, "da_block_m": 0, "n_fire": 0})
        writer.writerow({"step": 1, "Uapp_m": 2e-6, "KJ_Pa_sqrtm": 12e6, "crack_extension_m": 20e-6,
                         "da_block_m": 20e-6, "n_fire": 1})
    history = from_legacy_steps_csv(path)
    assert len(history.rows) == 2
    assert history.rows[0].event_id == ""
    assert history.rows[0].stage is ObservationStage.UNKNOWN
    assert history.rows[0].validity_status is ValidityStatus.UNKNOWN
    assert history.rows[1].K_J_Pa_sqrt_m == 12e6
    assert math.isnan(history.rows[1].K_I_Pa_sqrt_m)
    result = history.at_length(10e-6)
    assert result.status == "discrete_jump_bracket"
    assert result.crossing[0].K_J_Pa_sqrt_m == 12e6


def test_legacy_unload_hold_and_repeated_length_are_chronological(tmp_path):
    path = tmp_path / "reload.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "step", "Uapp_m", "KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire",
        ])
        writer.writeheader()
        for step, u, k in ((0, 2e-6, 8e6), (1, 1e-6, 4e6),
                           (2, 1e-6, 4e6), (3, 2e-6, 8e6)):
            writer.writerow({"step": step, "Uapp_m": u, "KJ_Pa_sqrtm": k,
                             "crack_extension_m": 5e-6, "da_block_m": 0, "n_fire": 0})
    history = from_legacy_steps_csv(path, initial_length_m=100e-6)
    assert [item.load_direction for item in history.rows] == [
        LoadDirection.LOADING, LoadDirection.UNLOADING,
        LoadDirection.HOLD, LoadDirection.LOADING,
    ]
    assert history.at_length(5e-6).status == "exact_repeated"
    absolute = history.at_length(105e-6, coordinate=LengthCoordinate.ABSOLUTE_PROJECTED)
    assert absolute.status == "exact_repeated"


def test_unknown_validity_is_not_finite_equals_valid(tmp_path):
    path = tmp_path / "steps.csv"
    path.write_text("step,Uapp_m,KJ_Pa_sqrtm,crack_extension_m,da_block_m,n_fire\n"
                    "0,1e-6,9000000,1e-6,0,0\n")
    history = from_legacy_steps_csv(path)
    assert history.at_length(1e-6, validity="valid").status == "invalid_or_unqualified_at_length"


def test_online_history_separates_pre_event_and_committed_geometry():
    header = (
        "step,Uapp_m,KJ_Pa_sqrtm,a_tip_m,crack_extension_m,"
        "da_block_m,n_fire,dt_cur_s,fatigue_cycles"
    )
    steps = np.array([
        [0, 1e-6, 9e6, 101e-6, 1e-6, 0, 0, 2e-9, 0],
        [1, 2e-6, 12e6, 106e-6, 6e-6, 5e-6, 1, 3e-9, 0],
    ])
    history = online_observation_history(
        steps, header, np.array([40.0, 90.0]),
        j_signed=np.array([-40.0, 90.0]), source_file="steps.csv"
    )

    assert [record.stage for record in history.rows] == [
        ObservationStage.REEQUILIBRATED,
        ObservationStage.PRE_EVENT,
        ObservationStage.COMMITTED,
    ]
    assert history.rows[1].pre_length_m == pytest.approx(1e-6)
    assert history.rows[1].committed_length_m == pytest.approx(1e-6)
    assert history.rows[1].K_J_Pa_sqrt_m == pytest.approx(12e6)
    assert history.rows[1].J_J_per_m2 == pytest.approx(90.0)
    assert history.rows[1].J_signed_J_per_m2 == pytest.approx(90.0)
    assert math.isnan(history.rows[1].G_J_per_m2)
    assert history.rows[2].committed_length_m == pytest.approx(6e-6)
    assert math.isnan(history.rows[2].K_J_Pa_sqrt_m)
    assert math.isnan(history.rows[2].J_J_per_m2)
    query = history.at_length(3e-6)
    assert query.status == "discrete_jump_bracket"
    assert query.interpolation_performed is False


def test_online_history_distinguishes_reload_segments_and_event_multiplicity():
    header = (
        "step,Uapp_m,KJ_Pa_sqrtm,a_tip_m,crack_extension_m,"
        "da_block_m,n_fire,dt_cur_s,fatigue_cycles"
    )
    # load/event, unload, hold, reload/event; the final accepted step aggregates
    # two events without pretending that either has a separate post-event solve.
    steps = np.array([
        [0, 1e-6, 9e6, 101e-6, 1e-6, 0, 0, 1e-9, 0],
        [1, 2e-6, 12e6, 106e-6, 6e-6, 5e-6, 1, 1e-9, 0],
        [2, 1e-6, 7e6, 106e-6, 6e-6, 0, 0, 1e-9, 0],
        [3, 1e-6, 7e6, 106e-6, 6e-6, 0, 0, 1e-9, 0],
        [4, 3e-6, 14e6, 116e-6, 16e-6, 10e-6, 2, 1e-9, 0],
    ])
    history = online_observation_history(
        steps, header, np.array([30., 50., 20., 20., 70.]),
        j_signed=np.array([30., 50., -20., -20., 70.]),
        source_file="steps.csv",
    )
    assert [row.segment_id for row in history.rows] == [
        "native-000-loading", "native-000-loading", "native-000-loading",
        "native-001-unloading", "native-002-hold",
        "native-003-loading", "native-003-loading",
    ]
    assert history.rows[-2].event_id == "step-4-events-2-3"
    assert history.rows[-2].event_multiplicity == 2
    assert history.rows[-1].aggregate_event_count == 3
    assert math.isnan(history.rows[-1].G_J_per_m2)
