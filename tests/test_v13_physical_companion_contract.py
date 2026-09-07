import inspect
from pathlib import Path


def test_physical_companion_uses_actual_fields_and_exact_native_marginal_energy():
    from scripts.qualify_v13_physical_companions import evaluate_parent
    source = inspect.getsource(evaluate_parent)
    assert "assemble_mechanics(baseline.mesh" in source
    assert "probe_tensor_ahead(baseline.mesh, sigma, baseline.damage" in source
    assert "runtime.evaluate_trial(request)" in source
    assert "baseline.stored_energy_J_per_m) - pair_energy)/da" in source
    assert "preview.sigma_tip(K_comp/math.sqrt(companion.gamma_rel))" in source
    assert "trial_conditional_pair(" in source
    assert "apply_conditional_branch_mark(" in source
    assert "engine.step(" not in source
    assert "advance_accepted_step(" not in source
    assert "np.random" not in source


def test_physical_input_contract_rejects_historical_or_changed_parent():
    from scripts.qualify_v13_physical_companions import evaluate_parent
    source = inspect.getsource(evaluate_parent)
    assert 'not record["fresh_initialization"]' in source
    assert 'sha256(path) != record["event_context_sha256"]' in source
    assert "fp(baseline) == before_hash" in source
    assert "fp(_capture_shared_engine(engine)) == process_hash" in source
    assert "record[\"event_context_sha256\"]" in source
