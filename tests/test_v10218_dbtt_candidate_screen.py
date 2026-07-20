from __future__ import annotations

from pathlib import Path
import subprocess

from arrhenius_fracture import sharp_front_v10_2_18 as entry
from arrhenius_fracture.parameter_registry_v9111 import select_option


def test_all_screen_options_are_saved_dbtt_rows():
    expected = {
        "dbtt_primary": "DBTT_restart04_candidate03",
        "dbtt_broad_shielding": "DBTT_restart01_candidate68",
        "dbtt_intrinsic_control": "DBTT_restart00_candidate103",
        "dbtt_moderate_shielding_reference": "DBTT_restart00_candidate04",
    }
    assert set(entry.DBTT_SCREEN_OPTIONS) == set(expected)
    for option, candidate in expected.items():
        selected = select_option(option, canonical_stage3_only=False)
        assert selected.material_class == "DBTT"
        assert selected.candidate_id == candidate
        assert (selected.mpz_length_um, selected.mpz_n_bins) == (50.0, 80)


def test_prepare_changes_only_exact_selected_row_and_mpz_grid(tmp_path: Path):
    out = tmp_path / "case"
    args = [
        "--parameter-option", "dbtt_intrinsic_control",
        "--parameter-registry", str(Path(entry.default_registry_path())),
        "--out", str(out),
        "--mode", "2d",
        "--target-crack-extension-um", "20",
    ]
    selected, manifest, audit = entry._prepare_dbtt_screen_option(args)
    assert selected.candidate_id == "DBTT_restart00_candidate103"
    assert manifest.is_file()
    assert audit.is_file()
    assert args[args.index("--mpz-length-um") + 1] == "50.0"
    assert args[args.index("--mpz-n-bins") + 1] == "80"
    assert args[args.index("--target-crack-extension-um") + 1] == "20"


def test_runner_and_analyzer_parse_and_preserve_v10217_controls():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_v10_2_18_dbtt_candidate_short_screen.sh"
    analyzer = root / "scripts" / "plot_v10_2_18_dbtt_candidate_screen.py"
    completed = subprocess.run(["bash", "-n", str(runner)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    text = runner.read_text()
    assert "sharp_front_v10_2_18" in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-20}" in text
    assert "seed = base + int(round(temperature))" in text
    assert "--front-state-model moving_pz" in text
    assert "--signed-active-shielding" in text
    assert "--no-wake-shielding" in text
    assert "--target-crack-extension-um \"$TARGET_EXT_UM\"" in text
    compiled = subprocess.run(
        ["python", "-m", "py_compile", str(analyzer), str(root / "arrhenius_fracture" / "sharp_front_v10_2_18.py")],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
