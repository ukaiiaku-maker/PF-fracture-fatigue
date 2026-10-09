import json

import pytest

from arrhenius_fracture.shared_options import (
    CavityMode, CrackPathMode, KineticsStrategy, LoadingMode,
    MechanicsBackend, PlasticityMode, PRESETS, SharedOptions,
    SourceLifecycle, SourceLocation,
)
from arrhenius_fracture.shared_runtime import (
    FEM_COMMIT, require_restart_compatible, resolve_runtime,
)


def test_every_preset_round_trips_and_resolves_exact_providers(tmp_path):
    for name, options in PRESETS.items():
        path = tmp_path / f"{name}.json"
        options.dump(path)
        loaded = SharedOptions.load(path)
        assert loaded == options
        assert loaded.fingerprint == options.fingerprint
        receipt = resolve_runtime(loaded)
        assert receipt.options_sha256 == options.fingerprint
        receipt.dump(tmp_path / f"{name}.receipt.json")
        assert json.loads((tmp_path / f"{name}.receipt.json").read_text())["schema"].endswith("/1")


def test_source_geography_and_lifecycle_are_independent_axes():
    persistent_tip = SharedOptions(
        source_location=SourceLocation.TIP_SURFACE,
        source_lifecycle=SourceLifecycle.PERSISTENT_REUSABLE,
    )
    finite_tip = SharedOptions(
        source_location=SourceLocation.TIP_SURFACE,
        source_lifecycle=SourceLifecycle.FINITE_INVENTORY,
    )
    finite_bulk = SharedOptions(
        source_location=SourceLocation.BULK_UNIFORM,
        source_lifecycle=SourceLifecycle.FINITE_INVENTORY,
        plasticity=PlasticityMode.FULL_FIELD_EVOLVING_RHO,
    )
    assert len({item.fingerprint for item in (persistent_tip, finite_tip, finite_bulk)}) == 3
    assert resolve_runtime(persistent_tip).providers[1].identity.startswith("v10.2.21")
    assert "finite" in resolve_runtime(finite_tip).providers[1].identity
    assert any(p.role == "bulk_plasticity" for p in resolve_runtime(finite_bulk).providers)


def test_unsupported_combinations_fail_before_solver_execution():
    with pytest.raises(ValueError, match="bulk source geography"):
        SharedOptions(source_location=SourceLocation.BULK_UNIFORM)
    with pytest.raises(ValueError, match="branching_multifront"):
        SharedOptions(cavity=CavityMode.FULL_ORDER_V5)
    with pytest.raises(ValueError, match="fixed crack front"):
        SharedOptions(
            kinetics=KineticsStrategy.OPERATOR_CONSISTENT_V3_6,
            crack_path=CrackPathMode.BRANCHING_MULTIFRONT,
        )
    with pytest.raises(ValueError, match="V5 cavity adapter"):
        SharedOptions(
            mechanics=MechanicsBackend.FULL_FEM_CZM,
            source_lifecycle=SourceLifecycle.CURRENT_SOURCE,
            crack_path=CrackPathMode.BRANCHING_MULTIFRONT,
            cavity=CavityMode.FULL_ORDER_V5,
        )


def test_pf_monotonic_cyclic_branch_void_and_bulk_bindings_are_distinct():
    identities = {
        key: tuple(provider.identity for provider in resolve_runtime(PRESETS[key]).providers)
        for key in (
            "pf_monotonic_tip_persistent", "pf_cyclic_tip_persistent",
            "pf_monotonic_bulk", "pf_branching_current", "pf_branching_void_v5",
        )
    }
    assert len(set(identities.values())) == len(identities)
    assert "v10.2.30_hazard_energy_gated_fatigue" in identities["pf_cyclic_tip_persistent"]
    assert "full_order_v5" in identities["pf_branching_void_v5"]


def test_external_full_fem_is_a_pinned_pair_not_a_pf_import():
    receipt = resolve_runtime(PRESETS["fem_czm_theta0"])
    assert receipt.providers == ()
    assert receipt.external_pair["commit"] == FEM_COMMIT
    assert receipt.external_pair["relationship"].startswith("comparison backend")


def test_unknown_option_fields_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        SharedOptions.from_dict({"loading": LoadingMode.MONOTONIC.value, "surprise": 1})


def test_restart_requires_same_options_and_provider_identities():
    options = PRESETS["pf_cyclic_tip_persistent"]
    receipt = resolve_runtime(options).to_dict()
    require_restart_compatible(receipt, options)
    changed = dict(receipt)
    changed["providers"] = [dict(item) for item in receipt["providers"]]
    changed["providers"][0]["identity"] = "changed"
    with pytest.raises(RuntimeError, match="provider identity"):
        require_restart_compatible(changed, options)
    with pytest.raises(RuntimeError, match="option fingerprint"):
        require_restart_compatible(receipt, PRESETS["pf_monotonic_tip_persistent"])
