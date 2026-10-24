import time

from experimental_fits.ast2018 import campaign
from experimental_fits.ast2018.search_bounded import _score_bounded


def test_stalled_candidate_is_recorded_as_numerical_rejection(monkeypatch):
    class SlowObjective:
        case_calls = 0
        progress_callback = None

        def evaluate(self, _x, *, detail):
            self.case_calls += 1
            self.progress_callback("test_condition", "running", self.case_calls)
            time.sleep(.2)

    monkeypatch.setattr(campaign, "_WORKER", SlowObjective())
    row = _score_bounded([.5] * 7, .02)
    assert row["watchdog_timeout"]
    assert row["cost"] == 1e300
    assert "TimeoutError" in row["error"]
    assert row["completed_condition_progress"][0]["condition_id"] == "test_condition"
    assert campaign._WORKER.progress_callback is None
