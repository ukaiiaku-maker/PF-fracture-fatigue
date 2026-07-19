import arrhenius_fracture.sharp_front_v10_2_9 as shared_entry


class _FakeFamily:
    states = [object()]
    interpolation = {"method": "fixed_reference"}
    metadata = {"production_parameterization_allowed": False}

    def audit_payload(self):
        return {"schema": "fake", "n_states": 1}


def test_monotonic_and_fatigue_install_identical_v1029_engine(monkeypatch):
    family = _FakeFamily()
    monkeypatch.setattr(
        shared_entry.StateResolvedSignedShieldingKernelFamily,
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
