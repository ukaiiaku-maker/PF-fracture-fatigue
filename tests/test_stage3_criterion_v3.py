from arrhenius_fracture.stage3_criterion_v3 import *
def test_criteria_are_complete_and_fail_closed():
 assert len(TRANSACTION_CASES)==9 and len(DETERMINISTIC_SEQUENCE)==13
 gates=classify({REQUIRED_GATES[0]:True})
 assert gates[REQUIRED_GATES[0]]=="PASS" and gates[REQUIRED_GATES[-1]]=="FAIL"
