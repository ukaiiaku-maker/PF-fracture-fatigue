from arrhenius_fracture.unit_slip_perturbation_v10212 import MODEL_ID


def test_v10214_stiffness_mask_model_id():
    assert MODEL_ID.startswith("v10.2.14_physical_signed_slip")
