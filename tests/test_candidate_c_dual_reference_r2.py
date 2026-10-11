from arrhenius_fracture.candidate_c_dual_reference_r2 import recovered_candidate_c_receipt


def test_exact_candidate_c_donor_keeps_both_birth_owners_and_honest_no_event():
    receipt = recovered_candidate_c_receipt()
    assert receipt["tree"] == "4918243e7b9fce65331922dedb7fc10a49dc6211"
    assert receipt["birth_pathways"]["vacancy_diffusion"]["terminal_action_1p02"] > 0.0
    assert receipt["birth_pathways"]["direct_dislocation_interaction"]["terminal_action_1p02"] > 0.0
    assert receipt["birth_pathways"]["vacancy_diffusion"]["event"] is False
    assert receipt["forced_event"] is False
    assert receipt["birth_pathways"]["direct_dislocation_interaction"]["row_calibration"] == "REFERENCE_UNCALIBRATED_AT_300_K"
