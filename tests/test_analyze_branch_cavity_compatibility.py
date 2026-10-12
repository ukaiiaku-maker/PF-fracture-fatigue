from scripts.analyze_branch_cavity_compatibility import build_report


def test_build_report_requires_exact_restart_and_distinct_owners():
    base = {
        "status": "PASS",
        "native_checkpoint_round_trip_exact": True,
        "reference_terminal_exact": None,
        "shared_fields": {"field_transfer_audit": {"projected_fields_nonzero": True}},
        "owners": {"branch_ids": ["root", "child"], "cavity_ids": ["void"],
                   "active_tip_ids": ["child"]},
        "execution": {"event_sequence": ["ligament", "child", "continue"]},
        "terminal": {"graph_length_m": 1.0, "cavity_radius_m": 0.1,
                     "reaction_N_per_m": 2.0, "energy_J_per_m": 3.0,
                     "void_phase": "ACTIVE", "active_crack_branch_ids": ["child"],
                     "event_counters": {"topology_actions": 3}},
        "cavity_feedback_switch": "enabled",
        "cavity_toggle_changes_branch_permission": False,
        "crack_void_connection_certificate": {"passed": True},
    }
    restarted = {**base, "reference_terminal_exact": True,
                 "execution": {"event_sequence": ["restore", "continue"]}}
    assert build_report(base, restarted)["status"] == "PASS"
    restarted["reference_terminal_exact"] = False
    assert build_report(base, restarted)["status"] == "FAIL"
