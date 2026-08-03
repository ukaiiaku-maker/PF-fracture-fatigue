from arrhenius_fracture import persistent_site_high_cycle_dmd_v10230_v2 as dmd
from arrhenius_fracture import persistent_site_high_cycle_dmd_v10230_v3 as chained
from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high

from v10230_affine_dmd_fixture import (
    AffineEngine,
    Controller,
    Waveform,
    configure,
)


def _production_tolerances(monkeypatch):
    monkeypatch.setenv("V10230_DMD_STATE_VALIDATION_REL_TOL", "1e-3")
    monkeypatch.setenv("V10230_DMD_HAZARD_VALIDATION_REL_TOL", "1e-3")
    monkeypatch.setenv("V10230_DMD_LEDGER_VALIDATION_REL_TOL", "2e-3")


def test_affine_dmd_segment_propagates_linear_neutral_mode(monkeypatch):
    configure(monkeypatch)
    engine = AffineEngine(drift=2.0, hazard=1.0e-30)
    result = dmd.propagate_dmd_cycles(
        engine,
        Controller(),
        Waveform(),
        300.0,
        1.0e8,
        requested_project_cycles=1.0e8 - 12,
    )
    assert result.accepted is True
    assert result.cycles_consumed == 1.0e8
    assert abs(engine.mpz.mobile_count - 2.0e8) / 2.0e8 < 1.0e-10
    assert result.drift_relative_error < 1.0e-9
    assert result.hazard_relative_error < 1.0e-9


def test_chained_dmd_completes_1e12_in_one_projective_operation(monkeypatch):
    configure(monkeypatch)
    _production_tolerances(monkeypatch)
    monkeypatch.setenv("V10230_DMD_CHAIN_MAX_SEGMENTS", "256")
    engine = AffineEngine(drift=1.0, hazard=1.0e-30)
    result = chained.propagate_chained_dmd_cycles(
        engine,
        Controller(),
        Waveform(),
        300.0,
        1.0e12,
        requested_project_cycles=16.0,
    )
    assert result.accepted is True
    assert result.completed_requested_horizon is True
    assert result.cycles_consumed == 1.0e12
    assert result.accepted_segments <= 256
    assert result.drift_relative_error <= 1.0e-3
    assert result.hazard_relative_error <= 1.0e-3
    assert abs(engine.mpz.mobile_count - 1.0e12) / 1.0e12 < 1.0e-8


def test_production_alias_reaches_1e12_without_subcycle_fallback(monkeypatch):
    configure(monkeypatch)
    _production_tolerances(monkeypatch)
    monkeypatch.delenv("V10230_DMD_CHAIN_MAX_SEGMENTS", raising=False)
    engine = AffineEngine(drift=1.0, hazard=1.0e-30)

    def forbidden_subcycle(*args, **kwargs):
        raise AssertionError("ordinary high-cycle evolution entered subcycle fallback")

    wrapper_globals = high.integrate_state_coupled_waveform.__globals__
    bound = wrapper_globals["_bound_integrator"]
    monkeypatch.setattr(
        bound.__globals__["_transient"],
        "integrate_state_coupled_waveform",
        forbidden_subcycle,
    )
    result = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert result["coupled_hazard_cycles_consumed"] == 1.0e12
    assert result["coupled_hazard_partial_return"] is False
    projective = [
        row for row in result["coupled_hazard_modes"]
        if row["mode"] == "slow_projective"
    ]
    assert projective
    assert sum(row["cycles"] for row in projective) == 1.0e12
    assert "event_to_event" in high.MODEL_ID
    assert abs(engine.mpz.mobile_count - 1.0e12) / 1.0e12 < 1.0e-8
