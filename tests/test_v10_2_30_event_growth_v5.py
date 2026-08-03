from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from arrhenius_fracture import persistent_site_high_cycle_dmd_v10230_v4 as segment
from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint,
)

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


def _load_growth_analyzer():
    path = Path(__file__).parents[1] / "scripts" / "analyze_v10_2_30_developed_fatigue_growth.py"
    spec = importlib.util.spec_from_file_location("growth_analyzer", path)
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

    analyzer = _load_growth_analyzer()
    rc = analyzer.main([str(tmp_path), "--target-extension-um", "100"])
    assert rc == 0
    payload = json.loads((tmp_path / "developed_fatigue_growth_summary.json").read_text())
    assert payload["event_count"] == 20
    assert payload["target_reached"] is True
    assert payload["stable_growth_provisional"] is True
    assert payload["developed_interval"]["da_dN"] > 0.0
    assert (tmp_path / "event_da_dN_vs_extension.png").is_file()
