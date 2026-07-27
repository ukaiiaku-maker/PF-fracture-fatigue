from arrhenius_fracture import crystal, fem
from arrhenius_fracture import sharp_front_v10_1_7_3 as avalanche
from arrhenius_fracture import sharp_front_v10_2_29_fatigue_audited as v10229
from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry
from arrhenius_fracture.persistent_site_cyclic_energy_gated_v10230 import (
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
)


def test_v10230_entry_installs_and_restores_all_runtime_layers(monkeypatch, tmp_path):
    observed = {}
    original_engine = v10229.AuditedCoupledPersistentSiteCyclicTipEngine
    original_builder = avalanche.build_avalanche_backend
    original_assemble = fem.assemble_mechanics
    original_compete = crystal.cleave_direction_competition
    original_discrete = crystal.cleavage_branch_candidates

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
        return "ok"

    monkeypatch.setattr(v10229, "main", fake_main)
    monkeypatch.setattr(entry, "write_last_energy_gate_diagnostics", lambda out: None)
    monkeypatch.setattr(entry, "_write_audit", lambda args: None)

    result = entry.main(["--fatigue-cycles", "--out", str(tmp_path)])
    assert result == "ok"
    assert observed["engine"] is HazardEnergyGatedPersistentSiteCyclicTipEngine
    assert observed["builder_changed"] is True
    assert observed["assemble_changed"] is True
    assert observed["compete_changed"] is True
    assert observed["discrete_changed"] is True

    assert v10229.AuditedCoupledPersistentSiteCyclicTipEngine is original_engine
    assert avalanche.build_avalanche_backend is original_builder
    assert fem.assemble_mechanics is original_assemble
    assert crystal.cleave_direction_competition is original_compete
    assert crystal.cleavage_branch_candidates is original_discrete
