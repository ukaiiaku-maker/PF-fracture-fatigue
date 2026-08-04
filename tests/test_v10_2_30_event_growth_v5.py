from __future__ import annotations

import csv
import copy
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from arrhenius_fracture import persistent_site_high_cycle_dmd_v10230_v4 as segment
from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint,
    write_checkpoint,
)
from arrhenius_fracture.stochastic_avalanche_tip import (
    StochasticAvalancheDiagnosticTipEngine,
    threshold_event_length_factor,
)
from arrhenius_fracture import sharp_front

from v10230_affine_dmd_fixture import AffineEngine, Controller, Waveform, configure


def _production(monkeypatch, tmp_path):
    configure(monkeypatch)
    monkeypatch.setenv("V10230_DMD_STATE_VALIDATION_REL_TOL", "1e-3")
    monkeypatch.setenv("V10230_DMD_HAZARD_VALIDATION_REL_TOL", "1e-3")
    monkeypatch.setenv("V10230_DMD_REUSE_MAX_SEGMENTS", "32")
    monkeypatch.setenv("V10230_DMD_CHAIN_MAX_SEGMENTS", "128")
    monkeypatch.setenv("V10230_HIGH_CYCLE_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS", "0")


def test_rate_separated_segment_reuses_map_and_writes_checkpoint(monkeypatch, tmp_path):
    _production(monkeypatch, tmp_path)
    engine = AffineEngine(drift=1.0, hazard=1.0e-30)
    result = segment.propagate_dmd_cycles(
        engine,
        Controller(),
        Waveform(),
        300.0,
        1.0e8,
        requested_project_cycles=16.0,
    )
    assert result.accepted is True
    assert result.completed_requested_horizon is True
    assert result.reused_local_map is True
    assert result.rate_separated_outputs is True
    assert result.positivity_preserving_coordinates is True
    assert result.ledger_delta["engine.N_em"] >= 0.0
    assert result.ledger_delta["mpz.emitted_total"] >= 0.0
    assert abs(engine.mpz.mobile_count - 1.0e8) / 1.0e8 < 1.0e-7
    assert (tmp_path / "high_cycle_live_checkpoint.json").is_file()
    assert (tmp_path / "high_cycle_live_state.npz").is_file()
    assert (tmp_path / "high_cycle_live_history.jsonl").is_file()


def test_signed_ledger_is_not_a_dmd_admission_gate(monkeypatch, tmp_path):
    _production(monkeypatch, tmp_path)

    class SignedLedgerEngine(AffineEngine):
        def _plastic_half_step(self, dt, temperature, sigma):
            before = self.mpz.signed_source_activations_total
            result = super()._plastic_half_step(dt, temperature, sigma)
            increment = self.mpz.signed_source_activations_total - before
            self.mpz.signed_source_activations_total -= 3.0 * increment
            return result

    engine = SignedLedgerEngine(drift=1.0, hazard=1.0e-30)
    result = segment.propagate_dmd_cycles(
        engine,
        Controller(),
        Waveform(),
        300.0,
        1.0e6,
        requested_project_cycles=16.0,
    )
    assert result.accepted is True
    assert result.ledger_delta["engine.N_em"] >= 0.0
    assert result.ledger_delta["mpz.emitted_total"] >= 0.0
    assert result.ledger_delta["mpz.signed_source_activations_total"] < 0.0


def test_checkpoint_restores_stationary_geometry_state(monkeypatch, tmp_path):
    _production(monkeypatch, tmp_path)
    engine = AffineEngine(drift=2.0, hazard=1.0e-30)
    result = segment.propagate_dmd_cycles(
        engine,
        Controller(),
        Waveform(),
        300.0,
        1.0e5,
        requested_project_cycles=16.0,
    )
    assert result.accepted
    restored = AffineEngine(drift=2.0, hazard=1.0e-30)
    restore_checkpoint(restored, tmp_path)
    assert abs(restored.mpz.mobile_count - engine.mpz.mobile_count) < 1.0e-8
    assert abs(restored.N_em - engine.N_em) < 1.0e-8
    assert restored.hazard_action_current == engine.hazard_action_current
    assert restored._hazard_rng.bit_generator.state == engine._hazard_rng.bit_generator.state


def test_production_alias_is_event_to_event_v5(monkeypatch, tmp_path):
    _production(monkeypatch, tmp_path)
    assert "event_to_event" in high.MODEL_ID
    assert "positive_state" in high.DMD_MODEL_ID
    engine = AffineEngine(drift=1.0, hazard=1.0e-30)
    result = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e8
    )
    assert result["coupled_hazard_cycles_consumed"] == 1.0e8
    assert result["coupled_hazard_rate_separated_ledgers"] is True
    assert result["coupled_hazard_positivity_preserving_coordinates"] is True
    assert result["coupled_hazard_live_checkpointing"] is True
    assert result["coupled_hazard_growth_objective"] == "event_to_event_100um"


def test_checkpoint_sync_event_guard_exact_fallback(monkeypatch, tmp_path):
    """The final 5 um driver checkpoint must precede all high-cycle evolution."""
    _production(monkeypatch, tmp_path)

    class DriverEngine(AffineEngine):
        _synchronize_driver_checkpoint_length = (
            StochasticAvalancheDiagnosticTipEngine.
            _synchronize_driver_checkpoint_length
        )
        _set_current_event_length = (
            StochasticAvalancheDiagnosticTipEngine._set_current_event_length
        )

        def __init__(self):
            super().__init__(drift=1.0, hazard=0.4)
            self.f = SimpleNamespace(da=20.0e-6)
            self.hazard_cfg = SimpleNamespace(mode="exponential", seed=2001726)
            self.avalanche_cfg = SimpleNamespace(
                mode="threshold_scaled",
                minimum_factor=0.5,
                maximum_factor=4.0,
            )
            self.hazard_threshold_action = 1.25
            self.avalanche_base_checkpoint_m = self.f.da
            self.avalanche_event_length_factor = 1.0
            self.avalanche_event_advance_m = self.f.da
            self.avalanche_event_length_history = []
            self.avalanche_checkpoint_synchronized = False
            self._set_current_event_length()

    engine = DriverEngine()
    threshold_before = engine.hazard_threshold_action
    rng_before = copy.deepcopy(engine._hazard_rng.bit_generator.state)
    factor_before = engine.avalanche_event_length_factor

    sharp_front._finalize_driver_physical_checkpoint(engine, 5.0e-6)

    assert engine.avalanche_checkpoint_synchronized is True
    assert engine.avalanche_base_checkpoint_m == 5.0e-6
    assert engine.B == 0.0
    assert engine.hazard_action_current == 0.0
    assert engine.hazard_threshold_action == threshold_before
    assert engine._hazard_rng.bit_generator.state == rng_before
    assert engine.avalanche_event_length_factor == factor_before
    assert engine.avalanche_event_advance_m == 5.0e-6 * factor_before
    assert factor_before == threshold_event_length_factor(
        threshold_before,
        mode="threshold_scaled",
        minimum_factor=0.5,
        maximum_factor=4.0,
        deterministic_threshold=False,
    )

    engine.avalanche_last_completed_advance_m = 9.0e-6
    engine.avalanche_last_completed_factor = 1.8
    engine.avalanche_event_length_history = [9.0e-6]
    write_checkpoint(engine, reason="post_event_restart")
    restored = DriverEngine()
    sharp_front._finalize_driver_physical_checkpoint(restored, 5.0e-6)
    restore_checkpoint(restored, tmp_path)
    assert restored.hazard_threshold_action == threshold_before
    assert restored.avalanche_event_length_factor == factor_before
    assert restored.avalanche_event_advance_m == 5.0e-6 * factor_before
    assert restored.avalanche_last_completed_advance_m == 9.0e-6
    assert restored.avalanche_last_completed_factor == 1.8
    assert restored.avalanche_event_length_history == [9.0e-6]

    base = high.integrate_state_coupled_waveform.__globals__["_bound_integrator"]
    globals_ = base.__globals__
    real_projective = globals_["_projective_with_requested_scale"]
    real_exact = globals_["_advance_exact_cycle_burst"]
    real_transient = globals_["_transient"].integrate_state_coupled_waveform

    def rejected_projective(*args, **kwargs):
        return SimpleNamespace(
            accepted=False,
            cycles_consumed=0.0,
            burst_cycles=8,
            projected_cycles=0.0,
            drift_relative_error=0.0,
            hazard_relative_error=0.0,
            attempts=1,
            failure_reason="dmd_event_guard",
        )

    exact_entered = []

    def exact_near_event(*args, **kwargs):
        exact_entered.append(True)
        return {
            "cycles": 0.0,
            "dB": 0.0,
            "dH": 0.0,
            "plastic": {},
            "event_near": True,
            "last_cycle": None,
        }

    def localized_transient(live, controller, waveform, temperature, cycles):
        assert live.avalanche_base_checkpoint_m == 5.0e-6
        assert live._hazard_rng.bit_generator.state == rng_before
        live.hazard_action_current = 0.5 * live.hazard_threshold_action
        live.B = live.hazard_action_current / live.hazard_threshold_action
        return {
            "fired": False,
            "coupled_hazard_cycles_consumed": cycles,
            "coupled_hazard_partial_return": False,
            "dB": 0.5,
            "physical_hazard_action_step": live.hazard_action_current,
            "da": 0.0,
            "packet_mean": 0.0,
            "packet_variance_m2": 0.0,
            "microsteps": 1,
            "plastic": {},
            "advance": {},
            "lambda_c": 0.0,
            "sigma_tip": 1.0e9,
        }

    monkeypatch.setitem(globals_, "_projective_with_requested_scale", rejected_projective)
    monkeypatch.setitem(globals_, "_advance_exact_cycle_burst", exact_near_event)
    monkeypatch.setattr(
        globals_["_transient"],
        "integrate_state_coupled_waveform",
        localized_transient,
    )
    try:
        result = high.integrate_state_coupled_waveform(
            engine, Controller(), Waveform(), 300.0, 2.0
        )
    finally:
        globals_["_projective_with_requested_scale"] = real_projective
        globals_["_advance_exact_cycle_burst"] = real_exact
        globals_["_transient"].integrate_state_coupled_waveform = real_transient

    assert exact_entered
    assert any(
        row.get("failure_reason") == "dmd_event_guard"
        for row in result["coupled_hazard_modes"]
    )
    assert any(
        row.get("mode") == "event_localization_transient"
        for row in result["coupled_hazard_modes"]
    )
    assert engine.B == engine.hazard_action_current / engine.hazard_threshold_action

    engine.f.da = 6.0e-6
    with pytest.raises(RuntimeError, match="after stochastic event evolution began"):
        engine._synchronize_driver_checkpoint_length()


def _load_growth_analyzer():
    path = Path(__file__).parents[1] / "scripts" / "analyze_v10_2_30_developed_fatigue_growth.py"
    spec = importlib.util.spec_from_file_location("growth_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_campaign_analyzer():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "analyze_v10_2_30_four_class_fatigue_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("campaign_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_developed_growth_analyzer_reports_stable_100um(tmp_path):
    fields = [
        "step", "fatigue_cycles", "crack_extension_m", "da_block_m", "n_fire",
        "B", "sigma_back_Pa", "mpz_mobile_count", "mpz_retained_count",
        "mpz_K_shield_Pa_sqrt_m", "lambda_c",
    ]
    with (tmp_path / "steps_0300K.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        extension = 0.0
        for index in range(20):
            extension += 5.0e-6
            writer.writerow(
                {
                    "step": index + 1,
                    "fatigue_cycles": 1000.0,
                    "crack_extension_m": extension,
                    "da_block_m": 5.0e-6,
                    "n_fire": 1,
                    "B": 0.1,
                    "sigma_back_Pa": 1.0e8,
                    "mpz_mobile_count": 10.0 + index,
                    "mpz_retained_count": 1.0,
                    "mpz_K_shield_Pa_sqrt_m": 0.0,
                    "lambda_c": 1.0,
                }
            )
    with (tmp_path / "crack_path_300K.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m", "y_m"])
        for index in range(21):
            writer.writerow([index * 5.0e-6, 0.0])

    control = {
        "parameter_option": "v913_paper_weakT01_0129902_persistent_sites",
        "target_deltaK_MPa_sqrt_m": 12.0,
        "target_Kmax_MPa_sqrt_m": 13.333333333333334,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "cleavage_hazard_seed": 2001726,
        "cycles_max": 1.0e12,
    }
    (tmp_path / "v10_2_30_fixed_deltaK_control.json").write_text(
        json.dumps(control)
    )
    geometry = []
    gates = []
    kinetic = []
    for index in range(20):
        threshold = 0.5 + index
        geometry.append(
            {
                "threshold_action": threshold,
                "event_length_factor": 1.0,
                "stochastic_proposed_event_length_m": 5.0e-6,
                "geometry_transaction_mode": "single_checked_outer_commit",
            }
        )
        gates.append(
            {
                "inserted": True,
                "arrest_reason": "stochastic_proposal_reached",
                "energy_admissible_event_length_m": 5.0e-6,
                "trial_rows": [
                    {
                        "trial_length_m": 5.0e-6,
                        "elastic_release_event_J_per_m": 2.0,
                        "hazard_dissipation_J_per_m": 1.0,
                        "energy_residual_J_per_m": 1.0,
                    }
                ],
            }
        )
        kinetic.append(
            {
                "physical_hazard_action_block": threshold,
                "coupled_hazard_modes": [
                    {"mode": "projective_rejected", "failure_reason": "dmd_event_guard"},
                    {"mode": "exact_cycle_burst"},
                    {"mode": "event_localization_transient"},
                ],
            }
        )
    (tmp_path / "stochastic_avalanche_geometry_events.json").write_text(
        json.dumps(geometry)
    )
    (tmp_path / "hazard_energy_gated_events_v10_2_30.json").write_text(
        json.dumps(gates)
    )
    (tmp_path / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": kinetic})
    )
    (tmp_path / "high_cycle_run_manifest.json").write_text(
        json.dumps({"git_head": "abc123", "generic_launcher": "test", "environment": {}})
    )

    analyzer = _load_growth_analyzer()
    rc = analyzer.main([str(tmp_path), "--target-extension-um", "100"])
    assert rc == 0
    payload = json.loads((tmp_path / "developed_fatigue_growth_summary.json").read_text())
    assert payload["event_count"] == 20
    assert payload["target_reached"] is True
    assert payload["stable_growth_provisional"] is True
    assert payload["developed_interval"]["da_dN"] > 0.0
    event = payload["event_measurements"][0]
    assert event["parameter_option"] == control["parameter_option"]
    assert event["threshold_action"] == 0.5
    assert event["physical_hazard_action"] == 0.5
    assert event["energy_available_J_per_m"] == 2.0
    assert event["energy_required_J_per_m"] == 1.0
    assert event["geometry_commit_inserted"] is True
    assert event["dmd_event_guard"] is True
    assert event["exact_fallback_entered"] is True
    assert event["transient_localization_entered"] is True
    assert event["private_trials_counted_as_cycles"] is False
    assert payload["provenance"]["git_head"] == "abc123"
    assert (tmp_path / "event_da_dN_vs_extension.png").is_file()


def test_four_class_campaign_analyzer_keeps_censors_and_provenance(tmp_path):
    roots = []
    for index, (option, rate, status) in enumerate(
        [
            ("v913_paper_peak01_0242980_persistent_sites", 2.0e-10, "growth_target_reached"),
            ("v913_paper_weakT01_0129902_persistent_sites", None, "no_committed_crack_event"),
        ]
    ):
        root = tmp_path / f"case_{index}"
        root.mkdir()
        roots.append(root)
        (root / "developed_fatigue_growth_summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "event_count": 3 if rate else 0,
                    "cycles_consumed": 1.0e6,
                    "final_projected_extension_um": 25.0 if rate else 0.0,
                    "target_reached": bool(rate),
                    "developed_interval": {"event_count": 2 if rate else 0, "da_dN": rate},
                    "event_measurements": ([{
                        "event_index": 1, "parameter_option": option,
                        "deltaK_MPa_sqrt_m": 10.0 + index, "hazard_seed": 1720,
                        "cycles_pre": 0.0, "cycles_post": 10.0, "cycles_between_events": 10.0,
                        "projected_extension_pre_m": 0.0, "projected_extension_post_m": 5e-6,
                        "projected_advance_m": 5e-6, "da_dN_m_per_cycle": rate,
                        "threshold_action": 0.5, "physical_hazard_action": 0.5,
                        "stochastic_proposed_advance_m": 5e-6,
                        "energy_gate_outcome": "stochastic_proposal_reached",
                        "geometry_commit_inserted": True, "acceleration_modes": "exact_cycle_burst",
                        "private_trials_counted_as_cycles": False,
                    }] if rate else []),
                    "provenance": {
                        "parameter_option": option,
                        "deltaK_MPa_sqrt_m": 10.0 + index,
                        "hazard_seed": 1720,
                        "git_head": "deadbeef",
                    },
                }
            )
        )
        (root / "v10_2_30_fixed_deltaK_control.json").write_text(
            json.dumps({"censor_status": "right_censored_no_event" if not rate else "propagated"})
        )
        (root / "exit_code.txt").write_text("0\n")
        if rate:
            (root / "stochastic_avalanche_geometry_events.json").write_text(json.dumps([{
                "event_index": 0, "x1": 0.001005, "y1": 0.0,
                "direction_audit": {"direction": [1.0, 0.0]},
            }]))

    out = tmp_path / "analysis"
    analyzer = _load_campaign_analyzer()
    assert analyzer.main([*(str(root) for root in roots), "--out", str(out)]) == 0
    payload = json.loads((out / "four_class_fatigue_summary.json").read_text())
    assert payload["case_count"] == 2
    assert payload["completed_count"] == 1
    assert payload["censored_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["cases"][0]["git_head"] == "deadbeef"
    assert (out / "four_class_fatigue_cases.csv").is_file()
    assert (out / "four_class_fatigue_censor_failure_table.csv").is_file()
    assert payload["event_interval_count"] == 1
    assert (out / "four_class_event_intervals.csv").is_file()
    assert (out / "four_class_da_dN_vs_deltaK.png").is_file()
