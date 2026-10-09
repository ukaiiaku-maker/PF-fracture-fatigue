from pathlib import Path

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
    plot_history(ObservationHistory(rows), output)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_size > 1000
