from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "arrhenius_fracture" / "hazard_energy_gate_v10230.py"
OBSERVED = ROOT / "arrhenius_fracture" / "hazard_energy_observed_engine_v10230.py"
DIFFERENTIAL = ROOT / "arrhenius_fracture" / "hazard_energy_differential_engine_v10230.py"
OBSERVER = ROOT / "arrhenius_fracture" / "hazard_energy_observer_v10230.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_2_30_hazard_energy_gated.py"
AUDITED = ROOT / "arrhenius_fracture" / "sharp_front_v10_2_30_hazard_energy_gated_audited.py"
BACKEND = ROOT / "arrhenius_fracture" / "hazard_energy_backend_audit_v10230.py"


def test_no_absolute_Gc0_athermal_parameter_is_introduced():
    for path in (GATE, OBSERVED, DIFFERENTIAL, OBSERVER, ENTRY, AUDITED, BACKEND):
        assert "Gc0_athermal" not in path.read_text(), path


def test_dissipation_formula_uses_only_active_hazard_terms():
    source = GATE.read_text()
    assert "gamma * m_hits * deltaG / (b * b)" in source
    assert "engine.lambda_cleave" in source
    assert "engine.f.m_hits" in source
    assert "engine.b" in source


def test_fixed_deltaK_uses_quadratic_probe_scaling():
    source = GATE.read_text()
    assert "ratio = event / probe" in source
    assert "scale = ratio * ratio" in source
    assert "J_event = max(float(ctx.J_probe_J_per_m2), 0.0) * scale" in source


def test_existing_angular_dependence_enters_hazard_and_dissipation():
    observed = OBSERVED.read_text()
    gate = GATE.read_text()
    assert "sigma_scaled = sigma_physical / math.sqrt(gamma)" in observed
    assert "Gamma = gamma * m_hits * deltaG / (b * b)" in gate
    assert 'candidate.get("gamma_rel", candidate.get("gamma", 1.0))' in gate


def test_gate_is_inside_internal_microstep_before_mpz_translation():
    source = DIFFERENTIAL.read_text()
    loop = source.index("while remaining > 0.0:")
    gate = source.index("gate = self._current_gate", loop)
    proposed = source.index("da_proposed =", gate)
    accepted = source.index("da = da_proposed * gate_fraction", proposed)
    translation = source.index("self.mpz.advance(da)", accepted)
    assert loop < gate < proposed < accepted < translation


def test_microstep_and_integrated_work_inequalities_fail_closed():
    source = DIFFERENTIAL.read_text()
    assert "microstep work inequality failed" in source
    assert "integrated work inequality failed" in source
    assert "dissipated_step > available_step" in source
    assert "dissipated > available" in source


def test_geometry_event_uses_integrated_accepted_length():
    source = DIFFERENTIAL.read_text()
    assert '"event_advance_m": accepted_event' in source
    assert '"proposed_event_advance_m"' in source
    assert '"rejected_event_advance_m"' in source
    assert "sum_of_accepted_microstep_advances" in source


def test_entry_uses_differential_engine_and_root_signed_J():
    source = ENTRY.read_text()
    assert "DifferentialHazardEnergyGatedPersistentSiteCyclicTipEngine" in source
    assert '"gate_resolution": "every_internal_Strang_microstep"' in source
    assert "--allow-abs-directional-J is forbidden" in source
    assert "install_observer" in source
    assert "install_hazard_energy_backend_audit" in source


def test_adaptive_prediction_and_commit_share_observed_context():
    source = OBSERVED.read_text()
    assert "def predict_clock_increment" in source
    assert 'self._refresh_observed_context(float(K), "monotonic")' in source
    assert "def step" in source
    assert "def preview_cycle_waveform" in source
    assert "def cycle_step_waveform" in source


def test_geometry_veto_is_not_partially_rolled_back():
    source = OBSERVED.read_text()
    assert "def restore_geometry_veto" in source
    assert "Exact rollback requires replay" in source


def test_geometry_audit_records_probe_event_and_energy_fields():
    source = BACKEND.read_text()
    required = (
        "energy_gate_K_probe_Pa_sqrt_m",
        "energy_gate_K_event_Pa_sqrt_m",
        "energy_gate_probe_to_event_scale",
        "energy_gate_Gamma_haz_J_per_m2",
        "energy_gate_available_integrated_J_per_m",
        "energy_gate_dissipated_integrated_J_per_m",
    )
    for token in required:
        assert token in source
