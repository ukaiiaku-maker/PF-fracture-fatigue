from arrhenius_fracture import crystal, fatigue_v1, fem
from arrhenius_fracture import hazard_energy_event_gate_v10230 as energy_gate
from arrhenius_fracture import sharp_front_v10_1_7_3 as avalanche
from arrhenius_fracture import sharp_front_v10_2_29_fatigue_audited as v10229
from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry
from arrhenius_fracture.hazard_energy_event_gate_mesh_consistent_v10230 import (
    energy_gate_event_length_mesh_consistent,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)


def test_v10230_entry_installs_and_restores_all_runtime_layers(monkeypatch, tmp_path):
    observed = {}
    original_engine = v10229.AuditedCoupledPersistentSiteCyclicTipEngine
    original_builder = avalanche.build_avalanche_backend
    original_assemble = fem.assemble_mechanics
    original_compete = crystal.cleave_direction_competition
    original_discrete = crystal.cleavage_branch_candidates
    original_energy_search = energy_gate.energy_gate_event_length
    original_waveform = fatigue_v1.FatigueWaveform

    def fake_main(args):
        observed["engine"] = v10229.AuditedCoupledPersistentSiteCyclicTipEngine
        observed["builder_changed"] = (
            avalanche.build_avalanche_backend is not original_builder
        )
        observed["assemble_changed"] = fem.assemble_mechanics is not original_assemble
        observed["compete_changed"] = (
            crystal.cleave_direction_competition is not original_compete
        )
        observed["discrete_changed"] = (
            crystal.cleavage_branch_candidates is not original_discrete
        )
        observed["energy_search"] = energy_gate.energy_gate_event_length
        observed["waveform_changed"] = fatigue_v1.FatigueWaveform is not original_waveform
        return "ok"

    monkeypatch.setattr(v10229, "main", fake_main)
    monkeypatch.setattr(entry, "write_last_energy_gate_diagnostics", lambda out: None)
    monkeypatch.setattr(entry, "_write_audit", lambda args: None)

    result = entry.main(["--fatigue-cycles", "--out", str(tmp_path)])
    assert result == "ok"
    assert (
        observed["engine"]
        is CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    assert observed["builder_changed"] is True
    assert observed["assemble_changed"] is True
    assert observed["compete_changed"] is True
    assert observed["discrete_changed"] is True
    assert observed["energy_search"] is energy_gate_event_length_mesh_consistent
    assert observed["waveform_changed"] is True

    assert v10229.AuditedCoupledPersistentSiteCyclicTipEngine is original_engine
    assert avalanche.build_avalanche_backend is original_builder
    assert energy_gate.energy_gate_event_length is original_energy_search
    assert fatigue_v1.FatigueWaveform is original_waveform
    assert fem.assemble_mechanics is original_assemble
    assert crystal.cleave_direction_competition is original_compete
    assert crystal.cleavage_branch_candidates is original_discrete
