"""Narrow executable reproducers; these do not qualify a V13 implementation."""
import pytest

from arrhenius_fracture.current_source_multifront_hooks_v12 import restore_complete_current_source_engine
from arrhenius_fracture.persistent_site_source_v10221 import SOURCE_MODEL, _persistent_emit
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from scripts.audit_v13_parent_mechanism import correlation_timing_counterexample
from scripts.v13_frozen_support import (
    saved_owner, initialized_engine, recapture, field_differences,
    frozen_process_step, rng_hash,
)
from arrhenius_fracture.current_source_runtime_bindings import (
    RuntimeBindingError, rehydrate_current_source_runtime_bindings,
    runtime_binding_inventory,
)


def test_current_correlation_policy_delays_a_completed_single_arm_clock():
    result = correlation_timing_counterexample()
    assert result["cleavage_clock_completion_s"] == 1.0
    assert result["proposal_count_at_clock_completion_with_correlation"] == 0
    assert result["one_arm_proposal_ready_s"] == 1.25
    assert result["runtime_unchanged"]


@pytest.fixture(scope="module")
def frozen_payload():
    return saved_owner()[1].complete_checkpoint_payload()


def test_persistent_emission_binding_must_survive_accepted_checkpoint_restore(frozen_payload):
    with initialized_engine(frozen_payload) as engine:
        before = _capture_shared_engine(engine)
        restored = restore_complete_current_source_engine(recapture(engine))
        assert restored.mpz.source_model == SOURCE_MODEL
        assert restored.mpz._emit.__func__ is _persistent_emit
        assert runtime_binding_inventory(engine) == runtime_binding_inventory(restored)
        assert len(runtime_binding_inventory(restored)["mpz"]) == 8
        assert field_differences(before, _capture_shared_engine(restored)) == []
        rehydrate_current_source_runtime_bindings(restored)
        assert field_differences(before, _capture_shared_engine(restored)) == []
        assert all(not callable(v) for group in ("engine_fields", "mpz_fields") for v in before[group].values())


@pytest.mark.parametrize("duration,active", [(0.0, True), (1e-10, False), (1e-10, True)])
def test_initialized_and_restored_execute_exactly(frozen_payload, duration, active):
    with initialized_engine(frozen_payload) as engine:
        restored = restore_complete_current_source_engine(recapture(engine))
        K = engine._signed_current_K_Pa_sqrt_m if active else 0.0
        rng = rng_hash(engine)
        a = frozen_process_step(engine, 300.0, duration, K)
        b = frozen_process_step(restored, 300.0, duration, K)
        assert field_differences(a, b) == []
        assert field_differences(_capture_shared_engine(engine), _capture_shared_engine(restored)) == []
        assert rng_hash(engine) == rng_hash(restored) == rng
        if duration and active:
            assert a["dN_emit"] > 1e-10
        elif not active:
            assert a["dN_emit"] == 0.0


@pytest.mark.parametrize("field", ["source_model", "state_model", "_signed_transport_mode"])
def test_unknown_model_ids_fail_without_state_mutation(frozen_payload, field):
    with initialized_engine(frozen_payload) as engine:
        setattr(engine.mpz, field, "unqualified")
        before = _capture_shared_engine(engine)
        with pytest.raises(RuntimeBindingError, match="model IDs"):
            rehydrate_current_source_runtime_bindings(engine)
        assert field_differences(before, _capture_shared_engine(engine)) == []


def test_unregistered_callable_fails_closed(frozen_payload):
    with initialized_engine(frozen_payload) as engine:
        engine.mpz.unrecognized_update = lambda: None
        with pytest.raises(RuntimeBindingError, match="unregistered"):
            rehydrate_current_source_runtime_bindings(engine)
