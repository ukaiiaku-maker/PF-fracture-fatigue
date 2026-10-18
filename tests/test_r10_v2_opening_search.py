import math

from scripts.run_r10_v2_opening_search import (
    access_load, consumed_parent, opening_arm, waveform_descriptor,
)


COHESIVE = {
    "Gamma_ref_J_m2": 3.0,
    "S_Gamma_J_m2_K": 0.0,
    "T_ref_K": 300.0,
    "activation_area_m2": 3e-19,
    "attempt_frequency_s": 1e12,
    "delta0_m": 3.0/(math.e*30e9),
    "t_hat_ref_Pa": 30e9,
}


def test_athermal_descriptor_uses_completion_not_branch_loss():
    low = waveform_descriptor(COHESIVE, "smooth_exponential", "cohesive_athermal", 1., 300.)
    root, high = access_load(COHESIVE, "smooth_exponential", "cohesive_athermal", 300.)
    assert not low["mechanical_completion"]
    assert math.isfinite(root)
    assert high["threshold_kind"] == "FINITE_COMPLIANCE_COMPLETION_COORDINATE"
    assert not high["mixed_thermal_advance_included"]


def test_mixed_descriptor_does_not_consume_static_mfpt_as_hazard():
    result = waveform_descriptor(COHESIVE, "smooth_exponential", "cohesive_mixed", 1., 300.)
    if "STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION" in result["kinetic_validity_counts"]:
        assert result["qualified_mixed_action"] == 0.0


def test_plasticity_hash_ignores_cleavage_and_metadata_only():
    row = {"candidate_id": "x", "cleave_G00_eV": 2., "emit_G00_eV": 3.,
           "peierls_H0_eV": 4., "role": "control"}
    assert consumed_parent(row) == {"emit_G00_eV": 3., "peierls_H0_eV": 4.}


def test_mode_binding_is_explicit():
    assert opening_arm("cohesive_athermal", "bilinear", COHESIVE).activation_area_m2 is None
    assert opening_arm("cohesive_mixed", "bilinear", COHESIVE).activation_area_m2 == 3e-19
