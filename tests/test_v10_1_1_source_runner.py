from pathlib import Path


def test_source_model_gate_compares_continuum_and_finite_modes():
    text = Path("scripts/run_v10_1_1_source_model_gate_700K.sh").read_text()
    assert 'SOURCE_MODELS=${SOURCE_MODELS:-"continuum finite_sites"}' in text
    assert '--tip-source-model "$SOURCE_MODEL"' in text
    assert '--tip-plasticity --active-shielding' in text
    assert '--no-wake-shielding' in text
    assert 'KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}' in text
    assert 'KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}' in text
    assert 'v10_1_driver_modes.json' in text


def test_source_gate_fails_fast_on_no_advance_or_absurd_initiation_toughness():
    text = Path("scripts/run_v10_1_1_source_model_gate_700K.sh").read_text()
    assert 'STEPS=${STEPS:-500}' in text
    assert 'K_FIRST_MAX_MPA_SQRT_M=${K_FIRST_MAX_MPA_SQRT_M:-100}' in text
    assert 'row["n_advances"]' in text
    assert 'row["Kc_first_MPa_sqrt_m"]' in text
    assert 'did not reach the required crack advance' in text


def test_continuum_gate_audits_local_emission_backstress():
    text = Path("scripts/run_v10_1_1_source_model_gate_700K.sh").read_text()
    assert 'tip_source_local_density_m2' in text
    assert 'tip_source_backstress_shear_Pa' in text
    assert 'tip_source_backstress_equivalent_Pa' in text
    assert 'tip_source_effective_emission_stress_Pa' in text
    assert 'tip_source_backstress_equivalent_Pa"]) for r in records) > 0.0' in text
