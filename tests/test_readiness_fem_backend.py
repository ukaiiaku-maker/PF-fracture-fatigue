from arrhenius_fracture.readiness_fem_backend import inspect_current_companion


def test_current_companion_is_source_bound_without_overclaiming_generic_adapter():
    receipt = inspect_current_companion()
    assert receipt.source_bound
    assert receipt.exact_current_entry_status.startswith("SOURCE_BOUND")
    assert receipt.generic_parameter_loading_adapter_status.startswith("UNRESOLVED")
    assert receipt.opening_only_adapter_status.startswith("UNRESOLVED")
