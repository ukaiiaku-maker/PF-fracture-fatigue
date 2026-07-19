from types import SimpleNamespace

import pytest

import arrhenius_fracture.sharp_front_v10_2_13 as shared_entry


def _family(authorized=False):
    return SimpleNamespace(
        states=[object(), object()],
        metadata={"production_parameterization_allowed": authorized},
        audit_payload=lambda: {
            "schema": "v10.2.13_extension_only_real_signed_2d_shielding_atlas",
            "active_physical_kernel_axes": [
                "cumulative_crack_path_extension_m"
            ],
        },
    )


def _install_mocks(monkeypatch, family):
    monkeypatch.setattr(
        shared_entry.ExtensionOnlySigned2DShieldingKernelFamily,
        "from_json",
        classmethod(lambda cls, path: family),
    )
    monkeypatch.setattr(
        shared_entry.StateResolvedSignedBurgersTipEngine,
        "configure_state_resolved_physics",
        classmethod(lambda cls, supplied, mode, **kwargs: None),
    )
    monkeypatch.setattr(
        shared_entry._transport,
        "normalize_transport_mode",
        lambda value: "validated_scalar",
    )


def test_monotonic_and_fatigue_install_identical_v10213_engine(monkeypatch):
    family = _family()
    _install_mocks(monkeypatch, family)
    original_anisotropic = shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_entry = shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine
    seen = []

    def fake_main(args):
        seen.append(
            (
                shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.StateResolvedSignedBurgersTipEngine,
                shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.StateResolvedSignedBurgersTipEngine,
                "--fatigue-cycles" in args,
            )
        )
        return "ok"

    monkeypatch.setattr(shared_entry._transport, "main", fake_main)
    assert shared_entry.main(["--signed-kernel-family", "family.json"]) == "ok"
    assert shared_entry.main(
        ["--signed-kernel-family", "family.json", "--fatigue-cycles"]
    ) == "ok"
    assert seen == [(True, True, False), (True, True, True)]
    assert (
        shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
        is original_anisotropic
    )
    assert shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine is original_entry


def test_parameter_campaign_fails_closed_without_extension_only_authorization(monkeypatch):
    family = _family(authorized=False)
    _install_mocks(monkeypatch, family)
    monkeypatch.setenv("PARAMETER_CAMPAIGN", "1")
    with pytest.raises(SystemExit, match="has not authorized"):
        shared_entry.main(["--signed-kernel-family", "family.json"])
