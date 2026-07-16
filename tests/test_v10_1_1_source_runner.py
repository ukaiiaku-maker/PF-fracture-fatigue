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
