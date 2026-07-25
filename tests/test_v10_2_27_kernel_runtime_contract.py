from __future__ import annotations

from pathlib import Path

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cubic_elasticity_is_part_of_the_kernel_identity():
    base = MechanicalKernelConfiguration()
    assert base.crystal_C11_Pa == 523.0e9
    assert base.crystal_C12_Pa == 203.0e9
    assert base.crystal_C44_Pa == 160.0e9
    changed = MechanicalKernelConfiguration(crystal_C44_Pa=240.0e9)
    assert changed.fingerprint() != base.fingerprint()


def test_production_entry_forces_runtime_mechanics_from_family():
    text = (
        ROOT / "arrhenius_fracture" / "sharp_front_v10_2_27.py"
    ).read_text()
    assert "_install_resolved_mechanics" in text
    for token in (
        '"--nx"',
        '"--ny"',
        '"--tip-h-fine"',
        '"--mpz-length-um"',
        '"--mpz-n-bins"',
        '"--da-phys"',
        '"--crystal-theta-deg"',
        '"--crystal-C11"',
        '"--crystal-C12"',
        '"--crystal-C44"',
    ):
        assert token in text
    assert "V10227_MECHANICAL_CONFIGURATION_FINGERPRINT" in text


def test_capture_builder_exports_resolved_cubic_elasticity():
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert 'payload["crystal_C11_Pa"]' in text
    assert 'payload["crystal_C12_Pa"]' in text
    assert 'payload["crystal_C44_Pa"]' in text
    assert 'export V10227_CRYSTAL_C11_PA="$C11"' in text
    assert 'export V10227_CRYSTAL_C12_PA="$C12"' in text
    assert 'export V10227_CRYSTAL_C44_PA="$C44"' in text


def test_explicit_mechanical_config_is_not_overwritten_by_runner_defaults():
    text = (
        ROOT / "scripts" / "resolve_v10_2_27_kernel_for_runner.sh"
    ).read_text()
    explicit, defaults = text.split("else\n  SOURCE_REGISTRY=", maxsplit=1)
    assert 'ARGS+=(--mechanical-config "$MECHANICAL_CONFIG")' in explicit
    assert 'REGISTRY_PZ_UM' not in explicit
    assert 'REGISTRY_PZ_UM' in defaults


def test_validated_launcher_runs_clean_final_four_class_adapter():
    validated = (
        ROOT
        / "scripts"
        / "run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh"
    ).read_text()
    assert "run_v10_2_27_full_current_kernel_four_class.sh" in validated

    adapter = (
        ROOT / "scripts" / "run_v10_2_27_full_current_kernel_four_class.sh"
    ).read_text()
    for token in (
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
    ):
        assert token in adapter
    assert "v913_zeroD_sobol_0129902" in adapter
    assert "v913_zeroD_sobol_0077080" in adapter


def test_partial_replacement_remains_fingerprint_gated():
    text = (
        ROOT / "scripts" / "run_v10_2_27_replace_weakT_ceramic_1000um.sh"
    ).read_text()
    assert "check_v10_2_27_retained_kernel_compatibility.py" in text
