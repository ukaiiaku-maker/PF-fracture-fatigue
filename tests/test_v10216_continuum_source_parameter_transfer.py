from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

from arrhenius_fracture import anisotropic_emission_v10174 as anisotropic
from arrhenius_fracture import sharp_front_v10_2_16 as entry
from arrhenius_fracture.anisotropic_continuum_source_v10216 import (
    SOURCE_MODEL, audit_payload, install_anisotropic_continuum_emission,
)
from arrhenius_fracture.continuum_source_tip import ContinuumSourceKineticTipEngine
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.unified_mpz import MPZConfig


def _state(velocity=1e-6):
    manifest = MaterialManifest.from_csv(default_manifest_path("DBTT"))
    fcfg = SimpleNamespace(r0=1e-6, sigma_cap=0.0, m_hits=1.0, tau_c=1e-6,
                           L_pz=1e-6, da=5e-6)
    ContinuumSourceKineticTipEngine.configure_default(KineticTipConfig())
    engine = ContinuumSourceKineticTipEngine(
        fcfg, None, None, 80e9, 0.3, 2.5e-10, manifest,
        MPZConfig(length_m=20e-6, n_bins=40, wake_length_m=20e-6),
    )
    state = engine.mpz
    install_anisotropic_continuum_emission(state, anisotropic.AnisotropicEmissionConfig())
    state._anisotropic_drive_factors = np.ones(state.n_systems)
    n = state.n_bins
    state._transport_rates = lambda *a, **k: {
        "peierls": np.ones(n), "taylor": np.zeros(n),
        "taylor_single": np.zeros(n), "m": np.ones(n),
        "jump": np.ones(n), "velocity": np.full(n, velocity),
        "encounter": np.zeros(n),
    }
    return state


def test_source_is_activity_not_finite_inventory():
    state = _state()
    assert state.source_model == SOURCE_MODEL
    assert state.continuum_source_finite_inventory is False
    assert state.continuum_source_multiplicity_consumed is False
    assert state.continuum_source_available_sites_semantics == "derived_M_ref_times_activity_proxy"


def test_repeated_throughput_exceeds_multiplicity_without_consuming_it():
    state = _state(1e-6)
    capacity = state.site_capacity.copy()
    reference = state.reference_source_multiplicity
    state.emission_rate_per_site = lambda stress, T: 1e9
    state._continuum_tip_radius_m = 1e-6
    for _ in range(50):
        state._emit(10.0, 1e9, 700.0)
    assert state.emitted_total > 2 * float(np.sum(capacity))
    assert state.reference_source_multiplicity == pytest.approx(reference)
    assert np.allclose(state.site_capacity, capacity)
    assert state.campaign_source_budget_consumed_total == pytest.approx(0.0)


def test_anisotropic_factor_enters_stress_before_hazard():
    state = _state(0.0)
    state._anisotropic_drive_factors = np.array([1.0, 0.25])
    state.emission_rate_per_site = lambda stress, T: max(float(stress), 0.0) / 1e9
    assert state._emit(1e-4, 1e9, 700.0) > 0
    assert state.anisotropic_last_sigma_emit_by_system_Pa[0] > state.anisotropic_last_sigma_emit_by_system_Pa[1]
    assert state.anisotropic_last_dN_emit_by_system[0] > state.anisotropic_last_dN_emit_by_system[1]


def test_advance_renews_activity_over_tip_radius_not_manifest_refresh_length():
    state = _state(0.0)
    state.tip_source_activity[:] = 0
    state.available_sites[:] = 0
    state._continuum_tip_radius_m = 1e-6
    result = state.advance(1e-6)
    expected = 1 - np.exp(-1)
    assert np.allclose(state.tip_source_activity, expected)
    assert result["tip_source_geometry_fraction"] == pytest.approx(expected)
    assert "campaign_source_refresh_length_m" not in result


def test_audit_fails_closed():
    payload = audit_payload()
    assert payload["finite_distributed_source_inventory"] is False
    assert payload["source_sites_per_system_role"] == "low_rate_arrhenius_hazard_multiplicity"
    assert payload["source_multiplicity_consumed"] is False
    assert payload["transport_operator_changed"] is False
    assert payload["shielding_law_changed"] is False
    assert payload["crack_geometry_changed"] is False


def test_entry_patches_only_source_installer(monkeypatch, tmp_path: Path):
    original = anisotropic.install_anisotropic_campaign_emission
    observed = {}
    selected = SimpleNamespace(
        option_key="dbtt_primary", candidate_id="candidate",
        mpz_length_um=50.0, mpz_n_bins=80,
        audit_payload=lambda: {"option_key": "dbtt_primary"},
    )
    manifest = tmp_path / "manifest.csv"; manifest.write_text("a\n1\n")
    selection = tmp_path / "selection.json"; selection.write_text("{}\n")
    monkeypatch.setattr(entry._base, "_prepare_parameter_option",
                        lambda args: (selected, manifest, selection))
    monkeypatch.setattr(entry._base, "_force_stage3_validity_envelope", lambda args: None)
    def fake_main(args):
        observed["installer"] = anisotropic.install_anisotropic_campaign_emission
        observed["args"] = list(args)
        return "ok"
    monkeypatch.setattr(entry._base._final_2d, "main", fake_main)
    assert entry.main(["--example", "1"]) == "ok"
    assert observed["installer"] is install_anisotropic_continuum_emission
    assert observed["args"] == ["--example", "1"]
    assert anisotropic.install_anisotropic_campaign_emission is original


def test_output_audit_records_parameter_overlay_only(tmp_path: Path):
    selected = SimpleNamespace(audit_payload=lambda: {"option_key": "dbtt_primary"})
    manifest = tmp_path / "manifest.csv"; manifest.write_text("a\n1\n")
    selection = tmp_path / "selection.json"; selection.write_text("{}\n")
    entry._write_v10216_audit(["--out", str(tmp_path)], selected, manifest, selection)
    payload = json.loads((tmp_path / "v10_2_16_continuum_source_parameter_overlay.json").read_text())
    assert payload["parameter_overlay_only"] is True
    assert payload["continuum_source"]["finite_distributed_source_inventory"] is False
    assert payload["preserved_physics"]["material_barrier_refit"] is False
