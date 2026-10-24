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


def test_watchdog_escapes_native_exception_recovery(monkeypatch):
    class CatchingObjective:
        case_calls=0
        progress_callback=None

        def evaluate(self,_x,*,detail):
            self.case_calls+=1
            while True:
                try:
                    time.sleep(.01)
                except Exception:
                    # The native solver can legitimately catch ordinary
                    # numerical exceptions. The watchdog must escape it.
                    continue

    monkeypatch.setattr(campaign,"_WORKER",CatchingObjective())
    row=_score_bounded([.5]*7,.02)
    assert row["watchdog_timeout"]
    assert row["runtime_s"]<.5


def test_required_numerical_failure_rejects_candidate_immediately(monkeypatch):
    class FailedObjective:
        case_calls=0
        progress_callback=None

        def evaluate(self,_x,*,detail):
            self.case_calls+=1
            self.progress_callback("cold_required","numerical_maximum_steps",
                                   self.case_calls)
            raise AssertionError("scoring must stop at the failed condition")

    monkeypatch.setattr(campaign,"_WORKER",FailedObjective())
    row=_score_bounded([.5]*7,900)
    assert row["cost"]==1e300
    assert row["early_numerical_rejection"]
    assert not row["watchdog_timeout"]
    assert row["case_calls"]==1
    assert row["completed_condition_progress"]==[
        {"condition_id":"cold_required","status":"numerical_maximum_steps",
         "case_index":1}]
