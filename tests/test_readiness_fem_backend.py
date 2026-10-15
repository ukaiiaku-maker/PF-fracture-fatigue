from arrhenius_fracture.readiness_fem_backend import inspect_current_companion


def test_current_companion_is_source_bound_with_native_opening_only_owner():
    receipt = inspect_current_companion()
    assert receipt.source_bound
    assert receipt.exact_current_entry_status.startswith("SOURCE_BOUND")
    assert receipt.generic_parameter_loading_adapter_status.startswith("IMPLEMENTED")
    assert receipt.opening_only_adapter_status == "IMPLEMENTED_NATIVE_CHANNEL_OWNER_AND_EXECUTED"
    assert receipt.generic_entry.endswith("generic_current_pf_companion.py")
