from pathlib import Path

import pytest

from arrhenius_fracture.fracture_observation_history import (
    FractureObservation, LoadDirection, ObservationHistory, ObservationStage,
)
from scripts.plot_fracture_observation_history import plot_history


def test_chronological_plot_writes_nonempty_png(tmp_path):
    rows = [
        FractureObservation(
            sequence=index, time_s=index, cycle=index, segment_id="load-1",
            stage=ObservationStage.REEQUILIBRATED, front_id="primary",
            event_id=str(index), load_direction=LoadDirection.LOADING,
            pre_length_m=index * 1e-6, committed_length_m=index * 1e-6,
            K_I_Pa_sqrt_m=(10 + index) * 1e6,
        )
        for index in range(3)
    ]
    output = tmp_path / "history.png"
    selected = plot_history(ObservationHistory(rows), output, quantity="K_I")
    assert [row.sequence for row in selected] == [0, 1, 2]
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_size > 1000


def test_plot_requires_one_named_quantity_and_never_falls_back(tmp_path):
    row = FractureObservation(
        sequence=0, time_s=0, cycle=0, segment_id="load-1",
        stage=ObservationStage.REEQUILIBRATED, front_id="primary",
        event_id="", load_direction=LoadDirection.LOADING,
        pre_length_m=0, committed_length_m=0, K_probe_Pa_sqrt_m=9e6,
    )
    with pytest.raises(ValueError, match="no finite K_J"):
        plot_history(ObservationHistory([row]), tmp_path / "wrong.png", quantity="K_J")
    selected = plot_history(
        ObservationHistory([row]), tmp_path / "probe.png", quantity="K_probe"
    )
    assert selected == (row,)
