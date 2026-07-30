import json
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.stochastic_avalanche_backend import (
    AvalancheSubsegmentBackend,
)


def test_avalanche_diagnostics_serialize_nested_numpy_values(tmp_path):
    backend = object.__new__(AvalancheSubsegmentBackend)
    backend.base_backend = SimpleNamespace(write_diagnostics=lambda out_dir: None)
    backend.advance_log = [
        {
            "event_index": np.int64(2),
            "direction_audit": {
                "direction": np.array([0.6, 0.8]),
                "direction_match_cosine": np.float64(1.0),
            },
        }
    ]

    backend.write_diagnostics(str(tmp_path))

    payload = json.loads(
        (tmp_path / "stochastic_avalanche_geometry_events.json").read_text()
    )
    assert payload == [
        {
            "event_index": 2,
            "direction_audit": {
                "direction": [0.6, 0.8],
                "direction_match_cosine": 1.0,
            },
        }
    ]
