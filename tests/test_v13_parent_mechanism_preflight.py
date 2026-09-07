"""Narrow executable reproducers; these do not qualify a V13 implementation."""
from types import MethodType

import pytest

from arrhenius_fracture.current_source_multifront_hooks_v12 import restore_complete_current_source_engine
from arrhenius_fracture.general_multifront_v12 import ProcessEngineState
from arrhenius_fracture.persistent_site_audited_engine_v10221 import AuditedPersistentSiteStateResolvedTipEngine
from arrhenius_fracture.persistent_site_source_v10221 import SOURCE_MODEL, _persistent_emit
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.unified_mpz import UnifiedMPZState
from scripts.audit_v13_parent_mechanism import correlation_timing_counterexample


def test_current_correlation_policy_delays_a_completed_single_arm_clock():
    result = correlation_timing_counterexample()
    assert result["cleavage_clock_completion_s"] == 1.0
    assert result["proposal_count_at_clock_completion_with_correlation"] == 0
    assert result["one_arm_proposal_ready_s"] == 1.25
    assert result["runtime_unchanged"]


@pytest.mark.xfail(strict=True, reason=(
    "Demonstrated existing V12 restore defect: callable persistent emission "
    "binding is excluded from capture and not reinstalled by restore."
))
def test_persistent_emission_binding_must_survive_accepted_checkpoint_restore():
    # Minimal call-binding reproducer: no physical initialization, RNG draw,
    # mechanics, emission evolution, or fabricated material parameters.
    engine = AuditedPersistentSiteStateResolvedTipEngine.__new__(
        AuditedPersistentSiteStateResolvedTipEngine
    )
    engine.mpz = UnifiedMPZState.__new__(UnifiedMPZState)
    engine.mpz.source_model = SOURCE_MODEL
    engine.mpz._emit = MethodType(_persistent_emit, engine.mpz)
    payload = _capture_shared_engine(engine)
    saved = ProcessEngineState.from_v11_payload(
        engine_id="engine:binding-reproducer", source_state_id="source:binding-reproducer",
        payload=payload, active_ledgers={}, wake_ledgers={}, signed_system_ledgers={},
        update_count=0, event_renewal_count=0, local_process_coordinate_m=0.0,
    )
    restored = restore_complete_current_source_engine(saved)
    assert restored.mpz.source_model == SOURCE_MODEL
    assert restored.mpz._emit.__func__ is _persistent_emit
