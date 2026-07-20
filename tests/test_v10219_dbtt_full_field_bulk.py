from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from arrhenius_fracture.config import make_emergent_config
from arrhenius_fracture.emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorModel,
    config_from_dislocation_config,
    install_manifest_bulk_kinetics,
)
from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.parameter_registry_v9111 import (
    select_option,
    write_compatibility_manifest,
)
from arrhenius_fracture import sharp_front_v10_2_19 as entry


def _mapped_config(tmp_path: Path, option: str = "dbtt_intrinsic_control"):
    selected = select_option(option, canonical_stage3_only=False)
    path = write_compatibility_manifest(selected, tmp_path / "manifest.csv")
    manifest = MaterialManifest.from_csv(path)
    cfg = make_emergent_config()
    audit = install_manifest_bulk_kinetics(cfg.dislocations, manifest, selected.row)
    return selected, manifest, cfg, audit


def test_exact_registry_surfaces_are_installed_without_bulk_refit(tmp_path: Path):
    selected, manifest, cfg, audit = _mapped_config(tmp_path)
    mapped = config_from_dislocation_config(cfg.dislocations)
    p = manifest.peierls.as_surface(manifest.emission)
    t = manifest.taylor.as_surface(manifest.emission)

    assert mapped.exact_manifest_mapping is True
    assert mapped.candidate_id == selected.candidate_id
    assert mapped.peierls.G00_eV == p.G00_eV
    assert mapped.peierls.gT_eV_per_K == p.gT_eV_per_K
    assert mapped.peierls.alpha == p.alpha
    assert mapped.peierls.exponent == p.exponent
    assert mapped.taylor.G00_eV == t.G00_eV
    assert mapped.taylor.gT_eV_per_K == t.gT_eV_per_K
    assert mapped.taylor.alpha == t.alpha
    assert mapped.taylor.exponent == t.exponent
    assert mapped.peierls_stress_fraction == float(selected.row["peierls_stress_fraction"])
    assert mapped.taylor_stress_fraction == float(selected.row["taylor_stress_fraction"])
    assert mapped.taylor_corr_rho_c_m2 == manifest.taylor_corr_rho_c_m2
    assert mapped.taylor_corr_scale == manifest.taylor_corr_scale
    assert audit["exact_manifest_mapping"] is True
    assert audit["candidate_id"] == selected.candidate_id


def test_bulk_rate_adapter_is_finite_nonnegative_and_stress_activated(tmp_path: Path):
    _selected, _manifest, cfg, _audit = _mapped_config(tmp_path)
    model = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg.dislocations)
    )
    rho = np.array([5.0e12, 5.0e13, 5.0e14])
    low = model.rates(np.full(3, 0.5e9), rho, 900.0, cfg.material.b)
    high = model.rates(np.full(3, 5.0e9), rho, 900.0, cfg.material.b)
    required = {
        "equivalent_plastic_rate_s",
        "peierls_rate_s",
        "taylor_single_hit_rate_s",
        "taylor_completion_rate_s",
        "series_rate_s",
        "taylor_m_eff",
        "rho_mobile_m2",
        "G_peierls_eV",
        "G_taylor_eV",
    }
    assert required <= set(high)
    for payload in (low, high):
        for value in payload.values():
            array = np.asarray(value, dtype=float)
            assert np.all(np.isfinite(array))
            assert np.all(array >= 0.0)
    assert np.any(high["equivalent_plastic_rate_s"] > 0.0)
    assert np.all(high["peierls_rate_s"] >= low["peierls_rate_s"])
    assert np.all(high["taylor_single_hit_rate_s"] >= low["taylor_single_hit_rate_s"])
    assert np.all(high["taylor_m_eff"] >= 1.0)


def test_full_field_envelope_changes_only_bulk_mode(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    monkeypatch.setenv("ANISOTROPIC_USE_AVALANCHE_BACKEND", "1")
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", "2420")
    args = [
        "--mode", "2d",
        "--bulk-plasticity-mode", "full_field",
        "--directional-j-mode", "root_signed",
        "--tip-kinetics-mode", "moving_velocity",
        "--tip-source-model", "continuum",
        "--front-state-model", "moving_pz",
        "--max-fronts", "1",
        "--mobile-shield-fraction", "0",
        "--no-wake-shielding",
        "--crystal-aniso",
        "--crystal-compete",
    ]
    seed = entry._force_full_field_envelope(args)
    assert seed == 2420
    index = args.index("--bulk-plasticity-mode")
    assert args[index + 1] == "full_field"
    for name, expected in {
        "--directional-j-mode": "root_signed",
        "--tip-kinetics-mode": "moving_velocity",
        "--tip-source-model": "continuum",
        "--front-state-model": "moving_pz",
    }.items():
        assert args[args.index(name) + 1] == expected


def test_entry_prepares_candidate_and_exact_bulk_mapping(tmp_path: Path):
    args = [
        "--parameter-option", "dbtt_moderate_shielding_reference",
        "--parameter-registry", str(entry._screen.default_registry_path()),
        "--out", str(tmp_path / "case"),
        "--mode", "2d",
        "--target-crack-extension-um", "10",
    ]
    selected, manifest_path, _audit_path = entry._prepare_full_field_option(args)
    cfg = entry._make_manifest_bulk_config()
    mapped = config_from_dislocation_config(cfg.dislocations)
    assert selected.candidate_id == "DBTT_restart00_candidate04"
    assert Path(manifest_path).is_file()
    assert mapped.exact_manifest_mapping is True
    assert mapped.candidate_id == selected.candidate_id
    assert entry._ACTIVE_BULK_MAPPING["candidate_id"] == selected.candidate_id
    assert args[args.index("--target-crack-extension-um") + 1] == "10"


def test_runner_parses_and_preserves_matched_screen_controls():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_v10_2_19_dbtt_full_field_screen.sh"
    completed = subprocess.run(
        ["bash", "-n", str(runner)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    text = runner.read_text()
    assert "sharp_front_v10_2_19" in text
    assert "--bulk-plasticity-mode full_field" in text
    assert "--bulk-kinetics-model emission_derived_peierls_taylor_multihit" in text
    assert "--tip-plasticity" in text
    assert "--no-wake-shielding" in text
    assert "seed = base + int(round(temperature))" in text
    assert "DEFAULT_TARGET=50" in text
    assert "DEFAULT_TEMPS=\"300 400 500 600 700 800 900 1000 1100 1200\"" in text
    assert "v10_2_19_full_field_bulk_audit.json" in text


def test_python_entry_runner_and_comparison_scripts_execute_help():
    root = Path(__file__).resolve().parents[1]
    targets = (
        root / "arrhenius_fracture" / "emission_derived_plasticity.py",
        root / "arrhenius_fracture" / "sharp_front_v10_2_19.py",
        root / "scripts" / "compare_v10_2_18_tip_only_v10_2_19_full_field.py",
    )
    compiled = subprocess.run(
        ["python", "-m", "py_compile", *map(str, targets)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    comparison = subprocess.run(
        ["python", str(targets[-1]), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert comparison.returncode == 0, comparison.stderr
    assert "--tip-only-root" in comparison.stdout
    assert "--full-field-root" in comparison.stdout


def test_source_declares_real_update_audit_and_no_independent_fit():
    root = Path(__file__).resolve().parents[1]
    text = (root / "arrhenius_fracture" / "sharp_front_v10_2_19.py").read_text()
    assert "_plasticity.update_plasticity = audited_update" in text
    assert "result = original_update(*call_args, **call_kwargs)" in text
    assert '"bulk_independent_parameter_fit": False' in text
    assert '"bulk_and_tip_share_selected_arrhenius_surfaces": True' in text
    assert "_tip_only_update_plasticity" not in text
