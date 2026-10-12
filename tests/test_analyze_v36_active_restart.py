from scripts.analyze_v36_active_restart import numeric_differences


def test_numeric_differences_reports_exact_and_scaled_changes():
    assert numeric_differences({"a": [1.0, 2]}, {"a": [1.0, 2]}) == []
    rows = numeric_differences({"a": [1.0]}, {"a": [1.25]})
    assert rows == [{
        "path": "/a/0", "kind": "value", "left": 1.0, "right": 1.25,
        "absolute_difference": 0.25, "scaled_difference": 0.2,
    }]
