from pathlib import Path

import pytest

from scripts.qualify_source_transport_axes import GEOGRAPHY, TIP_ONLY, qualify


pytestmark = pytest.mark.external_evidence


def test_source_transport_axes_use_active_native_histories():
    if not TIP_ONLY.is_dir() or not GEOGRAPHY.is_dir():
        pytest.skip("manifest-backed source geography evidence is absent")
    report = qualify()
    support = report["support_qualification"]
    assert support["tip_only"]["tip_process_active"] is True
    assert support["tip_only"]["bulk_process_active"] is False
    assert support["bulk_uniform"]["tip_process_active"] is False
    assert support["bulk_uniform"]["bulk_process_active"] is True
    assert support["tip_plus_bulk"]["tip_process_active"] is True
    assert support["tip_plus_bulk"]["bulk_process_active"] is True
    assert all(row["full_mechanics_history_present"] for row in support.values())


def test_missing_transfer_is_not_promoted_to_an_executable_model():
    if not TIP_ONLY.is_dir() or not GEOGRAPHY.is_dir():
        pytest.skip("manifest-backed source geography evidence is absent")
    report = qualify()
    transfer = report["capability_matrix"][-1]
    assert transfer["executable"] is False
    assert transfer["classification"].startswith("NOT_IMPLEMENTED")
