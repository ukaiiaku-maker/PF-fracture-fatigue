"""Prospective criteria for V12 production integration qualification."""
CRITERION_ID="v12.production-integration/2"
ROLLBACK_STAGES=("graph_edit","remesh","field_projection","support_rebuild","equilibrium","energy_gate","process_state_update","topology_verification","late_event_veto")
RESTART_FIELDS=("event_sequence","crack_graph","support","p0_damage","mesh_generation","front_state","process_state","hazards","thresholds","rng","reaction","energy","terminal_fingerprint")
V11_NEUTRALITY_FIELDS=("event_sequence","crack_graph","wake","active_fronts","process_state","source_state","hazards","thresholds","rng","reactions","energies","terminal_status","checkpoint_restart")
BOUNDED_CASES=("straight_repeated","sequential_events","kink","branch_capable","oblique_aligned","active_tip_refinement","rollback","restart")
REQUIRED_GATES=("V11_SELECTABLE_NEUTRALITY","V12_PRODUCTION_STATE_OWNERSHIP_QUALIFIED","PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED","V12_PRODUCTION_CHECKPOINT_RESTART_QUALIFIED","V12_BOUNDED_PRODUCTION_PROPAGATION_QUALIFIED")
def qualify(checks): return {gate:("PASS" if checks.get(gate) is True else "FAIL") for gate in REQUIRED_GATES}
def prerequisite(gates): return "PASS" if all(gates.get(g)=="PASS" for g in REQUIRED_GATES) else "FAIL"
