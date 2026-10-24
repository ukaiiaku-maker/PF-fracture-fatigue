"""An optimized flat barrier coordinate must not be reported as identified."""
import numpy as np

from experimental_fits.ast2018.objective import PARAMETERS
from experimental_fits.ast2018.report import _annotate_local_identifiability


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
