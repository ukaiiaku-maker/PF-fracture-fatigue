"""Prospective V12 one-void qualification criteria."""
CRITERION_ID="v12.stateful-voiding-qualification/1"
STATIC_LIMITS={"maximum_free_residual_relative":1e-6,"maximum_cavity_traction_relative":.15,"maximum_symmetry_error":.1,"maximum_refinement_change":.10}
TRANSACTION_CASES=("centered","positive_offset","negative_offset","miss","hit","downstream_nucleation","remesh_failure","equilibrium_failure","late_veto")
LIFECYCLE_CASES=("timestep_partition","competing_event","inventory","restart","rollback")
DETERMINISTIC_SEQUENCE=("AVAILABLE_SITE","MULTI_HIT_COMPLETE","EMBRYO","STABILIZED","SUBGRID_GROWTH","EXPLICIT_PROMOTION","RESOLVED_GROWTH","CRACK_VOID_INTERACTION","LIGAMENT_RUPTURE","CONNECTED_TOPOLOGY","DOWNSTREAM_FIRST_PASSAGE","NEW_SHARP_FRONT","CONTINUED_FRONT_EVENT")
REQUIRED_GATES=("V12_VOIDING_DISABLED_NEUTRALITY","V12_EXPLICIT_CRACK_VOID_STATIC_MECHANICS_QUALIFIED","V12_CRACK_VOID_TRANSACTION_QUALIFIED","V12_VOID_LIFECYCLE_QUALIFIED","V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED")
def classify(checks):
 return {gate:("PASS" if checks.get(gate) is True else "FAIL") for gate in REQUIRED_GATES}
