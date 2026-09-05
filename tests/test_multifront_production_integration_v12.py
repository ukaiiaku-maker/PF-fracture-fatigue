from types import SimpleNamespace

import pytest

from arrhenius_fracture.general_multifront_v12 import ResourcePolicy
from arrhenius_fracture.live_topology_kernel_v12 import (
    DynamicExactTopologyProviderV12, ProviderResourceLimitReached,
)
from arrhenius_fracture.production_multifront_v12 import (
    current_source_symbol_bindings, production_call_graph,
)
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    run_production_interval,
)


def test_dynamic_provider_reclassifies_legacy_metadata_as_operational_policy(monkeypatch):
    calls = []

    def fake(request):
        calls.append(len(request.crack_network.active_tip_ids))
        return {"maximum_fronts_supported": 16, "tips": []}

    monkeypatch.setattr(
        "arrhenius_fracture.live_topology_kernel_v11.evaluate_exact_topology", fake,
    )
    request = SimpleNamespace(
        crack_network=SimpleNamespace(active_tip_ids=tuple(f"f{i}" for i in range(32)))
    )
    provider = DynamicExactTopologyProviderV12(ResourcePolicy("mechanistic", None, None))
    result = provider.evaluate(request)
    assert calls == [32]
    assert result["maximum_fronts_supported"] is None
    assert result["operational_front_resource_limit"] is None
    assert result["inherited_v11_reported_limit_not_enforced"] == 16


def test_dynamic_provider_fails_before_lookup_at_configured_resource_bound(monkeypatch):
    called = []
    monkeypatch.setattr(
        "arrhenius_fracture.live_topology_kernel_v11.evaluate_exact_topology",
        lambda request: called.append(request),
    )
    request = SimpleNamespace(
        crack_network=SimpleNamespace(active_tip_ids=("a", "b", "c"))
    )
    provider = DynamicExactTopologyProviderV12(ResourcePolicy("mechanistic", 2, None))
    with pytest.raises(ProviderResourceLimitReached, match="configured_front"):
        provider.evaluate(request)
    assert not called


def test_production_call_graph_is_complete_and_ordered():
    graph = production_call_graph()
    assert [row["order"] for row in graph] == list(range(1, 12))
    assert [row["stage"] for row in graph] == [
        "generic_entrypoint", "adaptive_mesh", "accepted_fem_solve",
        "exact_topology_provider", "directional_local_J_and_tensor_probes",
        "correlated_same_tip_proposals",
        "isolated_atomic_energy_trials",
        "one_process_update_per_owner_and_global_scheduler",
        "atomic_topology_transaction_and_moving_frame_renewal",
        "canonical_output", "production_checkpoint",
    ]
    bindings = current_source_symbol_bindings()
    assert set(bindings) == {
        "generic_entrypoint", "adaptive_mesh", "accepted_fem_solve", "exact_topology_provider",
        "directional_local_J", "tensor_probe", "atomic_energy_trial",
        "process_engine_update", "scheduler", "moving_frame_renewal",
        "output", "checkpoint",
    }


def test_programmatic_production_entrypoint_is_wired_without_executing_mechanics(monkeypatch):
    calls = []

    class Provider:
        def __init__(self, policy):
            calls.append(("provider", policy))

    class Driver:
        def __init__(self, hooks, provider):
            calls.append(("driver", hooks, provider))

        def run_accepted_interval(self, runtime, accepted, *, duration_s):
            calls.append(("run", runtime, accepted, duration_s))
            return "wired"

    monkeypatch.setattr(
        "arrhenius_fracture.live_topology_kernel_v12.DynamicExactTopologyProviderV12",
        Provider,
    )
    monkeypatch.setattr(
        "arrhenius_fracture.production_multifront_v12.GenericMultiFrontProductionDriver",
        Driver,
    )
    runtime = SimpleNamespace(resource_policy="policy")
    assert run_production_interval(runtime, "accepted", "hooks", duration_s=0.5) == "wired"
    assert calls[0] == ("provider", "policy")
    assert calls[-1] == ("run", runtime, "accepted", 0.5)
