from scripts.attest_r2_runtime_install_isolation import attest


def test_fresh_workers_resolve_identical_final_runtime_bindings():
    result = attest()
    assert result["fresh_process_count"] == 2
    assert result["receipts_exact"] is True
    worker = result["worker_receipts"][0]
    assert worker["persistent_invariants"]["advance_refresh"] is False
    assert worker["reversible_binding"]["all_expected_methods_bound"] is True
    assert worker["taylor_contract"]["finite_window_s"] is None
