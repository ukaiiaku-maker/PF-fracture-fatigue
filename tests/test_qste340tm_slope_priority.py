import numpy as np

from arrhenius_fracture.qste340tm_slope_priority import SlopePriorityObjective


def records():
    rows = []
    for r_value, slope in zip((0.1, 0.3, 0.5, 0.7), (3.7, 3.4, 3.8, 2.9)):
        for order, delta_k in enumerate(np.geomspace(10.0, 30.0, 18)):
            rows.append({
                "R": r_value,
                "source_order": order,
                "DeltaK_MPa_sqrt_m": delta_k,
                "observed_dadn_m_per_cycle": 1.0e-12 * delta_k ** slope,
            })
    return rows


def test_offset_invariance_and_explicit_bias():
    objective = SlopePriorityObjective.from_records(records())
    observed = np.asarray([r["observed_dadn_m_per_cycle"] for r in records()])
    shifted = observed.copy()
    shifted[:18] *= 100.0
    assert np.allclose(objective.residuals(observed), 0.0)
    evaluation = objective.evaluate(shifted, barrier_pass=True)
    assert evaluation["blocks"]["0.1"]["shape_pass"]
    assert np.isclose(evaluation["blocks"]["0.1"]["bias_log10_rate"], 2.0)
    assert evaluation["primary_score"] < 1.0e-20
    assert evaluation["offset_score"] > 0.0


def test_wrong_slope_fails_each_history_gate():
    rows = records()
    objective = SlopePriorityObjective.from_records(rows)
    prediction = np.asarray([
        1.0e-10 * float(row["DeltaK_MPa_sqrt_m"]) for row in rows
    ])
    evaluation = objective.evaluate(prediction, barrier_pass=True)
    assert not evaluation["shape_pass"]
    assert not evaluation["eligible"]
    assert all(not block["shape_pass"] for block in evaluation["blocks"].values())


def test_barrier_status_is_required():
    rows = records()
    objective = SlopePriorityObjective.from_records(rows)
    observed = np.asarray([r["observed_dadn_m_per_cycle"] for r in rows])
    assert objective.evaluate(observed, barrier_pass=None)["shape_pass"]
    assert not objective.evaluate(observed, barrier_pass=None)["eligible"]
    assert objective.evaluate(observed, barrier_pass=True)["eligible"]
