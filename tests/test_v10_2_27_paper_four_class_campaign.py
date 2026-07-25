from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OPTIONS = [
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
]
EXPECTED_CANDIDATES = [
    "v913_zeroD_sobol_0242980",
    "v913_zeroD_sobol_0202500",
    "v913_zeroD_sobol_0257068",
    "v913_zeroD_sobol_0189364",
]
EXPECTED_SOURCE_CLASSES = {
    "v913_paper_peak01_0242980_persistent_sites": "DBTT",
    "v913_paper_dbtt01_0202500_persistent_sites": "DBTT",
    "v913_paper_weakT01_0257068_persistent_sites": "weakT",
    "v913_paper_ceramic01_0189364_persistent_sites": "ceramic",
}
FORBIDDEN = {
    "weakT_primary",
    "weakT_restart00_candidate00",
    "ceramic_primary",
    "ceramic_restart02_candidate00",
}


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_registry_exact_order_and_candidates(tmp_path: Path) -> None:
    registry = tmp_path / "four_class.csv"
    selection = tmp_path / "four_class.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_v10_2_27_four_class_registry.py"),
            "--output-registry",
            str(registry),
            "--output-selection",
            str(selection),
        ],
        cwd=ROOT,
        check=True,
    )
    with registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["option_key"] for row in rows] == EXPECTED_OPTIONS
    assert [row["candidate_id"] for row in rows] == EXPECTED_CANDIDATES
    assert {row["option_key"]: row["material_class"] for row in rows} == (
        EXPECTED_SOURCE_CLASSES
    )
    assert not FORBIDDEN.intersection(
        value for row in rows for value in row.values()
    )

    metadata = json.loads(selection.read_text())
    assert metadata["canonical_option_order"] == EXPECTED_OPTIONS
    assert metadata["physics_contract"]["parameter_transfer_only"] is True
    assert metadata["physics_contract"]["mechanics_changed"] is False
    assert metadata["physics_contract"]["persistent_sites"] is True
    assert metadata["physics_contract"]["finite_source_inventory"] is False
    assert metadata["physics_contract"]["source_refresh_on_crack_advance"] is False
    assert metadata["physics_contract"]["explicit_recovery"] is False
    assert len(metadata["primary_candidates"]) == 4


def test_entry_has_stable_four_option_mapping() -> None:
    from arrhenius_fracture import sharp_front_v10_2_27 as entry

    assert list(entry.VALID_OPTIONS) == EXPECTED_OPTIONS
    assert list(entry.VALID_OPTIONS.values()) == EXPECTED_CANDIDATES
    assert entry.MODEL_ID.startswith("v10.2.27")
    assert entry.DEFAULT_REGISTRY.name == "v10_2_27_paper_four_class_registry.csv"


def test_entry_normalizes_only_legacy_loader_class(tmp_path: Path) -> None:
    from arrhenius_fracture import sharp_front_v10_2_27 as entry

    registry = tmp_path / "four_class.csv"
    selection = tmp_path / "four_class.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_v10_2_27_four_class_registry.py"),
            "--output-registry",
            str(registry),
            "--output-selection",
            str(selection),
        ],
        cwd=ROOT,
        check=True,
    )

    for option, source_class in EXPECTED_SOURCE_CLASSES.items():
        selected = entry._select_option_four_class(
            option,
            registry,
            canonical_stage3_only=False,
        )
        assert selected.material_class == "DBTT"
        assert selected.row["material_class"] == source_class
        assert selected.option_key == option
        assert selected.candidate_id == entry.VALID_OPTIONS[option]


def test_seed_contract_has_48_unique_seeds() -> None:
    temperatures = [300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
    seeds = [
        3621 + option_index * 1_000_000 + temperature_index * 1009
        for option_index in range(4)
        for temperature_index, _ in enumerate(temperatures)
    ]
    assert len(seeds) == 48
    assert len(set(seeds)) == 48
    assert seeds[0] == 3621
    assert seeds[12] == 1_003_621


def test_runner_production_defaults_and_physics_contract() -> None:
    text = (
        ROOT
        / "scripts"
        / "run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
    ).read_text()
    required_fragments = [
        "TARGET_EXT_UM=${TARGET_EXT_UM:-1000}",
        "THETA=${THETA:-30}",
        "STEPS=${STEPS:-2000000}",
        "SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-20}",
        "SNAPSHOT_COLS=${SNAPSHOT_COLS:-5}",
        "CLEAVAGE_HAZARD_MODE=exponential",
        "CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled",
        "--max-fronts 1",
        "--bulk-plasticity-mode tip_only",
        "--directional-j-mode root_signed",
        "--signed-active-shielding",
        "--mobile-shield-fraction 0",
        "--no-wake-shielding",
        "PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}",
    ]
    for fragment in required_fragments:
        assert fragment in text
    for option in EXPECTED_OPTIONS:
        assert option in text
    for old in FORBIDDEN:
        assert old not in text


def test_plotter_target_checkpoints_and_event_interval_semantics() -> None:
    module = _load_script(
        ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py",
        "plot_v10_2_27_paper_four_class_rcurves",
    )
    assert module.checkpoints_for_target(10.0) == (10.0,)
    assert module.checkpoints_for_target(1000.0)[-2:] == (750.0, 1000.0)

    pre = np.asarray([0.0, 250.0, 500.0])
    post = np.asarray([100.0, 400.0, 600.0])
    resistance = np.asarray([10.0, 40.0, 50.0])
    assert module._resistance_at_extension(
        pre, post, resistance, 300.0, 600.0
    ) == 40.0


def test_audited_entry_installs_required_corrections() -> None:
    text = (
        ROOT / "arrhenius_fracture" / "sharp_front_v10_2_27_audited.py"
    ).read_text()
    assert "install_backstress_complementarity_fix()" in text
    assert "install_physical_front_width()" in text
    assert "AuditedPersistentSiteStateResolvedTipEngine" in text
