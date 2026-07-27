import inspect

import pytest

from arrhenius_fracture import hazard_energy_event_gate_v10230 as gate
from arrhenius_fracture import sharp_front_v10_2_27 as paper
from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry


def test_hazard_resistance_uses_active_hazard_quantities_only():
    value = gate.hazard_resistance_J_per_m2(
        barrier_J=2.0,
        cooperative_hits=3.0,
        burgers_vector_m=2.0,
        gamma_relative=1.5,
    )
    assert value == pytest.approx(2.25)
    doubled = gate.hazard_resistance_J_per_m2(
        barrier_J=2.0,
        cooperative_hits=3.0,
        burgers_vector_m=2.0,
        gamma_relative=3.0,
    )
    assert doubled == pytest.approx(2.0 * value)
    source = inspect.getsource(gate.hazard_resistance_J_per_m2)
    assert "FractureResistanceConfig" not in source
    assert "Gc0_athermal" not in source


def test_four_canonical_persistent_site_options_are_preserved():
    assert list(paper.VALID_OPTIONS) == [
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
    ]


def test_waveform_observer_captures_probe_K_before_replacement(monkeypatch):
    captured = []
    monkeypatch.setattr(entry, "set_latest_probe_K", captured.append)

    def original(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    observed = entry._observed_waveform_factory(original)
    result = observed(Kmax=7.5, R=0.1, frequency_Hz=1000.0)
    assert captured == [7.5]
    assert result["kwargs"]["Kmax"] == 7.5
