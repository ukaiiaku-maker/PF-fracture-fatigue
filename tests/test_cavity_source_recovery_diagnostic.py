from arrhenius_fracture.embryo_growth_v1 import CavityLoopSourceRecoveryQualificationError


def test_cavity_source_recovery_error_preserves_diagnostic():
    record = {"failed_checks": ("condition",), "checks": {"condition": {"value": 2, "limit": 1}}}
    error = CavityLoopSourceRecoveryQualificationError(record)
    assert str(error) == "resolved cavity loop source recovery failed qualification"
    assert error.diagnostic == record
