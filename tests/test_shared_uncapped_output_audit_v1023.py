import json

from scripts.audit_v10_2_3_shared_uncapped_outputs import main as audit_main


def test_output_audit_accepts_shared_uncapped_metadata(tmp_path, monkeypatch, capsys):
    (tmp_path / "v10_1_driver_modes.json").write_text(json.dumps({
        "manifest_K_shield_cap_enabled": False,
        "legacy_manifest_K_shield_cap_reference_only": True,
        "shared_monotonic_and_fatigue_core": True,
    }))
    (tmp_path / "v10_1_1_source_model.json").write_text(json.dumps({
        "cleavage_shielding_bound": "none; signed raw elastic dislocation field",
    }))
    (tmp_path / "kinetic_tip_cell_audit_v101.json").write_text(json.dumps({
        "campaign_calibration": {
            "shielding_cap_from_manifest": False,
            "shielding_saturation": "population_dynamics_only",
        }
    }))

    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path)])
    audit_main()
    output = capsys.readouterr().out
    assert '"pass": true' in output
