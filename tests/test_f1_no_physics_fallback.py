import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

import pytest

from arrhenius_fracture.persistent_runtime_contract import require_persistent_binding
from arrhenius_fracture.persistent_site_source_v10221 import (
    SOURCE_MODEL, _coupled_persistent_source_transport, _persistent_advance,
    _persistent_emit,
)
from arrhenius_fracture.shared_execution import (
    build_execution_plan, construct_and_attest, execute_shared,
)
from arrhenius_fracture.shared_execution_worker import _verify_executed_physics
from arrhenius_fracture.shared_options import PRESETS, SharedOptions, TransportStrategy
from arrhenius_fracture.signed_burgers_shared_v1025 import (
    _signed_evolve_channel, _signed_evolve_scalar,
)


def _bound_state(transport="validated_scalar"):
    state = SimpleNamespace(
        source_model=SOURCE_MODEL, _signed_transport_mode=transport,
        _exchange_transport_mode=(
            "joint_ctmc" if transport == "validated_scalar" else "legacy_split"
        ),
        persistent_source_inventory_active=False,
    )
    state._emit = MethodType(_persistent_emit, state)
    state._coupled_persistent_source_transport = MethodType(
        _coupled_persistent_source_transport, state
    )
    state.evolve = MethodType(
        _signed_evolve_scalar if transport == "validated_scalar" else _signed_evolve_channel,
        state,
    )
    state.advance = MethodType(_persistent_advance, state)
    state._exchange = lambda *a: None
    state.emission_rate_per_site = lambda *a: None
    return state


def test_actual_bound_object_is_checked_and_corruption_fails_closed():
    state = _bound_state()
    assert require_persistent_binding(state, stage="test")["all_expected_methods_bound"]
    state._emit = MethodType(lambda self, *a: 0.0, state)
    with pytest.raises(RuntimeError, match="no substitute"):
        require_persistent_binding(state, stage="after_remesh")


def test_channel_resolved_bound_object_has_its_own_truthful_contract():
    state = _bound_state("channel_resolved")
    receipt = require_persistent_binding(state, stage="channel")
    assert receipt["all_expected_methods_bound"]
    assert receipt["signed_transport_mode"] == "channel_resolved"
    assert receipt["exchange_transport_mode"] == "legacy_split"
    assert "per-slip-system" in receipt["physical_contract"]["transport"]


def test_split_exchange_selection_fails_closed_even_when_outer_wrapper_matches():
    state = _bound_state()
    state._exchange_transport_mode = "legacy_split"
    with pytest.raises(RuntimeError, match="no substitute"):
        require_persistent_binding(state, stage="split_selection")


def test_coupled_inner_operator_binding_is_required():
    state = _bound_state()
    state._coupled_persistent_source_transport = MethodType(lambda self, *a: {}, state)
    with pytest.raises(RuntimeError, match="coupled_source_transport"):
        require_persistent_binding(state, stage="inner_corruption")


def test_missing_executed_binding_receipt_is_failure_not_fallback(tmp_path):
    options = PRESETS["pf_monotonic_tip_persistent"]
    plan = build_execution_plan(options)
    with pytest.raises(RuntimeError, match="no finite-source"):
        _verify_executed_physics(tmp_path, options, plan)
    (tmp_path / "persistent_native_runtime_binding.json").write_text(json.dumps({
        "all_expected_methods_bound": True,
        "finite_inventory_substitute_started": False,
        "signed_transport_mode": "validated_scalar",
    }))
    assert _verify_executed_physics(tmp_path, options, plan)["all_expected_methods_bound"]


def test_executed_transport_must_match_typed_request(tmp_path):
    options = PRESETS["pf_monotonic_tip_persistent_channel"]
    plan = build_execution_plan(options)
    (tmp_path / "persistent_native_runtime_binding.json").write_text(json.dumps({
        "all_expected_methods_bound": True,
        "finite_inventory_substitute_started": False,
        "signed_transport_mode": "validated_scalar",
    }))
    with pytest.raises(RuntimeError, match="transport does not match"):
        _verify_executed_physics(tmp_path, options, plan)
    payload = {
        "all_expected_methods_bound": True,
        "finite_inventory_substitute_started": False,
        "signed_transport_mode": TransportStrategy.CHANNEL_RESOLVED.value,
    }
    (tmp_path / "persistent_native_runtime_binding.json").write_text(json.dumps(payload))
    assert _verify_executed_physics(tmp_path, options, plan)["signed_transport_mode"] == "channel_resolved"


@pytest.mark.parametrize(
    ("preset", "geography", "tip_enabled"),
    [
        ("pf_monotonic_bulk", "bulk_uniform", False),
        ("pf_monotonic_tip_plus_bulk", "tip_plus_bulk", True),
    ],
)
def test_bulk_geography_requires_matching_native_post_run_audit(
    tmp_path, preset, geography, tip_enabled,
):
    options = PRESETS[preset]
    plan = build_execution_plan(options)
    with pytest.raises(RuntimeError, match="no native model audit"):
        _verify_executed_physics(tmp_path, options, plan)
    payload = {
        "source_geography": geography,
        "tip_emission_enabled": tip_enabled,
        "bulk_plasticity_mode": "full_field",
        "bulk_sources": "homogeneous_persistent_background",
        "bulk_finite_inventory": False,
        "bulk_density_transport": False,
        "direct_tip_bulk_population_exchange": False,
        "zero_stress_net_plastic_rate_exactly_zero": True,
    }
    path = tmp_path / "v10_4_bulk_coupled_model_audit.json"
    path.write_text(json.dumps(payload))
    receipt = _verify_executed_physics(tmp_path, options, plan)
    assert receipt["all_expected_bindings_match"] is True
    assert receipt["source_geography"] == geography
    payload["source_geography"] = (
        "tip_plus_bulk" if geography == "bulk_uniform" else "bulk_uniform"
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="does not match"):
        _verify_executed_physics(tmp_path, options, plan)


def test_provider_import_failure_never_imports_finite_source(monkeypatch):
    options = PRESETS["pf_monotonic_tip_persistent"]
    plan = build_execution_plan(options)
    imported = []
    original = __import__("importlib").import_module
    def fail_persistent(name, *args, **kwargs):
        imported.append(name)
        if name == "arrhenius_fracture.persistent_site_source_v10221":
            raise ImportError("simulated unavailable required provider")
        return original(name, *args, **kwargs)
    monkeypatch.setattr("arrhenius_fracture.shared_execution.importlib.import_module", fail_persistent)
    with pytest.raises(ImportError, match="simulated"):
        construct_and_attest(plan, options)
    assert "arrhenius_fracture.sharp_front_v10_1" not in imported


def test_missing_required_input_fails_before_worker(monkeypatch, tmp_path):
    options = SharedOptions(material_manifest=str(tmp_path / "missing.csv"))
    calls = []
    monkeypatch.setattr("arrhenius_fracture.shared_execution.subprocess.run",
                        lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="material_manifest is mapped only"):
        execute_shared(options, [], tmp_path / "out")
    assert calls == []


def test_simulated_scheduler_timeout_never_selects_alternate_executor(monkeypatch, tmp_path):
    options = PRESETS["pf_monotonic_tip_persistent"]
    commands = []
    def timeout(command, **kwargs):
        commands.append(command)
        raise subprocess.TimeoutExpired(command, 1.0)
    monkeypatch.setattr("arrhenius_fracture.shared_execution.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="same requested physics remains incomplete"):
        execute_shared(options, [], tmp_path / "out", source_policy="development_snapshot")
    assert len(commands) == 1
    assert "shared_execution_worker" in " ".join(commands[0])
    assert "sharp_front_v10_1" not in " ".join(commands[0])
