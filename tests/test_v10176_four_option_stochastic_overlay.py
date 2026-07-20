from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import arrhenius_fracture.sharp_front_v10_1_7_6 as entry
from arrhenius_fracture.parameter_registry_v9111 import (
    CANONICAL_OPTIONS,
    select_option,
    write_material_manifest,
)


def test_exact_four_options_and_grids():
    selected = {key: select_option(key) for key in CANONICAL_OPTIONS}
    assert selected["ceramic_primary"].candidate_id == "ceramic_restart02_candidate00"
    assert selected["weakT_primary"].candidate_id == "weakT_restart00_candidate00"
    assert selected["dbtt_primary"].candidate_id == "DBTT_restart04_candidate03"
    assert selected["peak_primary"].candidate_id == "DBTT_restart05_candidate61"
    assert (selected["ceramic_primary"].mpz_length_um, selected["ceramic_primary"].mpz_n_bins) == (100.0, 200)
    assert (selected["weakT_primary"].mpz_length_um, selected["weakT_primary"].mpz_n_bins) == (100.0, 200)
    assert (selected["dbtt_primary"].mpz_length_um, selected["dbtt_primary"].mpz_n_bins) == (50.0, 80)
    assert (selected["peak_primary"].mpz_length_um, selected["peak_primary"].mpz_n_bins) == (50.0, 80)


def test_manifest_zero_cap_means_uncapped_input(tmp_path):
    selected = select_option("dbtt_primary")
    path = write_material_manifest(selected, tmp_path / "material.csv")
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["target_class"] == "dbtt_primary"
    assert float(row["max_K_shield_MPa_sqrt_m"]) == 0.0
    assert float(row["source_sites_per_system"]) == float(selected.row["source_sites_per_system"])


def test_entry_changes_only_manifest_and_mpz(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", "1720")
    seen = {}

    def fake_main(args):
        seen["args"] = list(args)
        return "ok"

    monkeypatch.setattr(entry._working_2d, "main", fake_main)
    args = [
        "--parameter-option", "ceramic_primary",
        "--mode", "2d",
        "--crystal-aniso",
        "--crystal-compete",
        "--out", str(tmp_path / "case"),
    ]
    assert entry.main(args) == "ok"
    forwarded = seen["args"]
    assert "--material-manifest" in forwarded
    assert forwarded[forwarded.index("--mpz-length-um") + 1] == "100.0"
    assert forwarded[forwarded.index("--mpz-n-bins") + 1] == "200"
    audit = json.loads((tmp_path / "case" / "v10_1_7_6_parameter_overlay_audit.json").read_text())
    assert audit["working_2d_entry"] == "arrhenius_fracture.sharp_front_v10_1_7_5"
    assert audit["parameter_overlay_only"] is True
    assert audit["atlas_used"] is False
    assert audit["source_model_changed"] is False
    assert audit["shielding_model_changed"] is False
    assert audit["cleavage_event_length_mode"] == "threshold_scaled"


def test_runner_has_no_atlas_or_mechanics_promotion_and_parses():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_v10_1_7_6_four_option_stochastic_sweep.sh"
    text = path.read_text()
    assert "sharp_front_v10_1_7_6" in text
    assert "CLEAVAGE_HAZARD_MODE=exponential" in text
    assert "CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled" in text
    assert "signed-active-shielding" in text
    assert "build_v10_2" not in text
    assert "kernel" not in text.lower()
    assert "atlas" in text.lower()  # status explicitly records atlas_used=False
    completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
