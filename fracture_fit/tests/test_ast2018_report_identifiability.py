"""An optimized flat barrier coordinate must not be reported as identified."""
import numpy as np

from experimental_fits.ast2018.objective import PARAMETERS
from experimental_fits.ast2018.report import (_annotate_local_identifiability,
    _onset_metrics,_curve_support_metrics)
import pytest


def test_exact_zero_working_sensitivity_is_marked_in_parameter_table():
    rows=[{"scenario":"TABLE_PRIMARY","stage":"analytical","code_key":key,
           "status":"fitted","profile_status":"conditional profile",
           "local_jacobian_column_norm":""} for key,_,_ in PARAMETERS]
    jacobian=np.zeros((3,len(PARAMETERS)))
    jacobian[0,0]=1.0
    flat=_annotate_local_identifiability(rows,"TABLE_PRIMARY",jacobian)
    assert flat==[key for key,_,_ in PARAMETERS[1:]]
    assert rows[0]["status"]=="fitted"
    assert rows[0]["local_jacobian_column_norm"]==1.0
    assert all(row["status"]=="optimized; locally unidentifiable" for row in rows[1:])


def test_report_counts_onsets_inside_declared_working_tolerance():
    class Data:
        def toughness_scenario(self,_):
            return [{"observable":"Kq_Jcontinuous_onset","condition_id":"a",
                     "value":4.0},
                    {"observable":"Kq_Jcontinuous_onset","condition_id":"b",
                     "value":10.0}]
    rows=[]
    for level in ("reduced","spatial"):
        rows.extend([{"level":level,"condition_id":"a",
                      "onset_K_Jcontinuous_MPa_sqrt_m":4.15},
                     {"level":level,"condition_id":"b",
                      "onset_K_Jcontinuous_MPa_sqrt_m":10.6}])
    result=_onset_metrics(Data(),"TABLE_PRIMARY",rows)
    assert result["reduced"]["n_supported"]==2
    assert result["reduced"]["n_within_working_tolerance"]==1


def test_full_j_support_audit_requires_all_eight_curves_and_629_points():
    rows=[]
    for level in ("reduced","spatial"):
        for i in range(8):
            rows.append({"level":level,"all_points":79 if i<5 else 78,
                         "covered_all_points":40,"all_629_support_cost":1.0})
    result=_curve_support_metrics(rows,"TABLE_PRIMARY")
    assert result["spatial"]["all_points"]==629
    assert result["spatial"]["covered_points"]==320
    with pytest.raises(RuntimeError,match="incomplete full J support"):
        _curve_support_metrics(rows[:-1],"TABLE_PRIMARY")
