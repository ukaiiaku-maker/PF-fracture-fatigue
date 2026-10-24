"""Direct spatial refinement must reject an incomplete required condition."""

from experimental_fits.ast2018 import local_spatial


def test_spatial_local_score_stops_after_required_numerical_failure(monkeypatch):
    class FailedObjective:
        case_calls=0
        progress_callback=None

        def evaluate(self,_x,*,detail):
            self.case_calls+=1
            self.progress_callback("required_condition","numerical_maximum_steps",
                                   self.case_calls)
            raise AssertionError("a failed 1-D case must end the candidate")

    monkeypatch.setattr(local_spatial,"_CTX",FailedObjective())
    row=local_spatial._score([.5]*7,100)
    assert row["cost"]==1e300
    assert row["error"]=="NumericalReject: required_condition: numerical_maximum_steps"
    assert not row["watchdog_timeout"]
    assert row["case_calls"]==1
    assert local_spatial._CTX.progress_callback is None
