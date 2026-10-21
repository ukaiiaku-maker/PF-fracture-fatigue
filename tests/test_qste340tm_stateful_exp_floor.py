import numpy as np

from scripts.calibrate_qste340tm_stateful_exp_floor import (
    TEMPERATURE_K,
    build_records,
    evaluate,
    exp_floor_barrier,
)


def test_exp_floor_reparameterization_preserves_room_temperature_coordinates():
    barrier = exp_floor_barrier("test", 1.25, 2.4e9, 1.0e11)
    assert barrier.G0_eV(TEMPERATURE_K) == 1.25
    assert barrier.sigc_Pa(TEMPERATURE_K) == 2.4e9


def test_stateful_reduction_returns_finite_chronological_state():
    records = build_records(
        __import__("pathlib").Path("literature_data/fatigue_recovered_review").resolve()
    )[:4]
    parameters = np.asarray([
        16.0, 32.0, np.log(1.0), np.log(0.5),
        1.2, np.log(2.0), 2.0, np.log(100.0), 0.5, np.log(1.0e-2),
    ])
    out = evaluate(parameters, records)
    assert np.all(np.isfinite(out["prediction"]))
    assert np.all(out["prediction"] > 0.0)
    assert np.all(np.diff(out["retained"]) >= -1.0e-12)
    assert np.all(out["radius"] >= 1.0e-6)
