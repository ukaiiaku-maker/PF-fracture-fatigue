import pytest

import scripts.run_active_two_mechanism_natural_void_study as study


def _provider(status):
    return {
        "directional": [{
            "candidate_id": "cleave:test",
            "directional_drive_status": status,
        }]
    }


def _install_observers(monkeypatch, repaired_status):
    calls = []

    monkeypatch.setattr(
        study, "crack_tip_tensor", lambda state, branch_id: ([[1.0, 0.0], [0.0, 1.0]], [7])
    )

    def provider(state, **kwargs):
        calls.append(kwargs)
        return _provider("UNAVAILABLE" if len(calls) == 1 else repaired_status)

    monkeypatch.setattr(study, "sharp_front_load_provider", provider)
    monkeypatch.setattr(
        study,
        "directional_sharp_front_rates",
        lambda state, rows, **kwargs: [{
            "candidate_id": rows[0]["candidate_id"],
            "directional_drive_status": rows[0]["directional_drive_status"],
            "K_effective_Pa_sqrt_m": (
                12.0e6 if rows[0]["directional_drive_status"] == "AVAILABLE" else 0.0
            ),
        }],
    )
    return calls


def test_required_natural_drive_uses_bounded_same_geometry_retry(monkeypatch):
    calls = _install_observers(monkeypatch, "AVAILABLE")

    _, K, audit, rows = study.canonical_drive_check(object())

    assert K == 12.0e6
    assert audit["interpretation"] == "CANONICAL_DRIVE_AVAILABLE"
    assert len(calls) == 2
    assert calls[1]["marginal_delta_a_m"] == (3.0e-5, 2.0e-5)
    assert calls[1]["marginal_mesh_levels"] == (4, 5)
    assert calls[1]["mechanical_refinement_policy"] == "legacy_accumulated_generations"
    assert calls[1]["virtual_mechanical_trial_transport"] == "deferred_read_only"
    assert rows[0]["representation_retry"]["tolerance_changed"] is False
    assert rows[0]["representation_retry"]["physical_time_advanced"] is False


def test_required_natural_drive_rejects_interval_after_bounded_retry(monkeypatch):
    calls = _install_observers(monkeypatch, "UNAVAILABLE")

    with pytest.raises(
        RuntimeError,
        match="REQUIRED_FRACTURE_OBSERVATION_UNAVAILABLE_AFTER_BOUNDED_REFINEMENT",
    ):
        study.canonical_drive_check(object())

    assert len(calls) == 2
