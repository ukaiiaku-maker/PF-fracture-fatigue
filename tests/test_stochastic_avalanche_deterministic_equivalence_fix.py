import numpy as np

from arrhenius_fracture import stochastic_avalanche_tip as avalanche_tip
from arrhenius_fracture.crack_backend import CrackAdvanceResult
from arrhenius_fracture.stochastic_avalanche_backend import AvalancheSubsegmentBackend


class RecordingSharpWake:
    name = "sharp_wake"

    def __init__(self):
        self.cohesive_network = None
        self.advance_log = []
        self.received_p1 = None
        self.received_direction = None

    def advance(self, **kwargs):
        self.received_p1 = kwargs["p1"]
        self.received_direction = kwargs["direction"]
        p0 = np.asarray(kwargs["p0"], dtype=float)
        p1 = np.asarray(kwargs["p1"], dtype=float)
        moved = float(np.linalg.norm(p1 - p0))
        self.advance_log.append({"x1": float(p1[0]), "y1": float(p1[1])})
        return CrackAdvanceResult(
            mesh=kwargs["mesh"],
            boundary=kwargs["boundary"],
            damage=np.asarray(kwargs["damage"], dtype=float).copy(),
            displacement=np.asarray(kwargs["displacement"], dtype=float).copy(),
            moved=moved,
            inserted=True,
            selected_edge_length=moved,
            reason="ok",
        )


def _queue_event(length_m):
    avalanche_tip.clear_pending_geometry_events()
    avalanche_tip._PENDING_GEOMETRY_EVENTS.append({
        "event_advance_m": float(length_m),
        "event_length_factor": 1.0,
        "threshold_action": 1.0,
        "hazard_seed": 0,
        "hazard_event_index": 0,
        "geometry_subsegment_fraction": 0.1,
    })


def test_deterministic_fixed_mode_is_exact_driver_transaction(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    _queue_event(5.0e-6)

    p0 = np.array([1.25e-4, -2.0e-6])
    # Deliberately make the driver direction non-unit and the endpoint slightly
    # different from a normalized 5-um reconstruction. The wrapper must preserve
    # exactly what sharp_front requested.
    direction = np.array([0.999999999999, 2.0e-6])
    p1 = p0 + 5.0e-6 * direction
    p1_before = p1.copy()

    base = RecordingSharpWake()
    backend = AvalancheSubsegmentBackend(base)
    result = backend.advance(
        mesh=object(),
        boundary=object(),
        damage=np.zeros(1),
        displacement=np.zeros(1),
        p0=p0,
        p1=p1,
        direction=direction,
        front_id=0,
        kill_r=1.0e-6,
    )

    assert result.inserted
    assert base.received_p1 is p1
    assert base.received_direction is direction
    assert np.array_equal(p1, p1_before)
    row = backend.advance_log[-1]
    assert row["deterministic_geometry_passthrough"] is True
    assert row["geometry_transaction_mode"] == "exact_driver_passthrough"
    assert row["requested_endpoint_preserved"] is True
    assert row["endpoint_adjustment_m"] == 0.0
    avalanche_tip.clear_pending_geometry_events()


def test_stochastic_mode_still_replaces_and_synchronizes_endpoint(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
    _queue_event(8.0e-6)

    p0 = np.zeros(2)
    p1 = np.array([5.0e-6, 0.0])
    direction = np.array([2.0, 0.0])
    base = RecordingSharpWake()
    backend = AvalancheSubsegmentBackend(base)
    result = backend.advance(
        mesh=object(),
        boundary=object(),
        damage=np.zeros(1),
        displacement=np.zeros(1),
        p0=p0,
        p1=p1,
        direction=direction,
        front_id=0,
        kill_r=1.0e-6,
    )

    assert result.inserted
    assert np.allclose(p1, [8.0e-6, 0.0])
    row = backend.advance_log[-1]
    assert row["deterministic_geometry_passthrough"] is False
    assert row["geometry_transaction_mode"] == "variable_length_endpoint_replacement"
    assert np.isclose(row["driver_direction_norm"], 2.0)
    avalanche_tip.clear_pending_geometry_events()
