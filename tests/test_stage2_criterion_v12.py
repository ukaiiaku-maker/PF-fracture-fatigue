from arrhenius_fracture.stage2_criterion_v12 import *
def test_required_matrices_are_complete():
    assert len(ROLLBACK_STAGES)==11 and "late_event_veto" in ROLLBACK_STAGES
    assert len(RESTART_FIELDS)==13 and len(V11_NEUTRALITY_FIELDS)==13
    assert {"kink","branch_capable","active_tip_refinement"}<=set(BOUNDED_CASES)
def test_missing_evidence_fails_closed():
    gates=qualify({REQUIRED_GATES[0]:True})
    assert gates[REQUIRED_GATES[0]]=="PASS"
    assert prerequisite(gates)=="FAIL"
