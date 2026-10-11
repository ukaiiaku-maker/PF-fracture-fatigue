from arrhenius_fracture.embryo_growth_v1 import CavityLoopSourceRecoveryQualificationError


def test_cavity_source_recovery_error_preserves_diagnostic():
    record = {"failed_checks": ("condition",), "checks": {"condition": {"value": 2, "limit": 1}}}
    error = CavityLoopSourceRecoveryQualificationError(record)
    assert str(error) == "resolved cavity loop source recovery failed qualification"
    assert error.diagnostic == record


def test_retriable_picard_error_contract_keeps_exact_message():
    from arrhenius_fracture.shared_v5_cavity_adapter import RETRIABLE_TRANSPORT_FAILURES
    assert "vacancy transport Picard iteration did not converge" in RETRIABLE_TRANSPORT_FAILURES
