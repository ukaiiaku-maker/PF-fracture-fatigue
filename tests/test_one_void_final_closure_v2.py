from arrhenius_fracture.one_void_final_closure_v2 import REQUIRED_GATES, SCHEMA, validate


def record(status="PASS"):
    return {"schema": SCHEMA,
            "terminal_classification": "V5_ONE_VOID_SCIENTIFIC_CLOSURE_QUALIFIED",
            "r_tip_distinct_from_R_void": True,
            "gates": {name: {"status": status} for name in REQUIRED_GATES},
            "source_bound_ontology": {"row_count": 1, "rows": [
                {"dataset": "test", "case_identity": "one", "classification": "PASS"}]}}


def test_native_final_schema_accepts_only_consistent_terminal_decision():
    assert validate(record())["valid"]
    blocked = record()
    blocked["gates"]["r_tip_final_classification"] = {"status": "NOT_DEFINED"}
    blocked["terminal_classification"] = "V5_ONE_VOID_FINAL_SCIENTIFIC_CLOSURE_COMPLETE_BUT_BLOCKED"
    assert validate(blocked)["valid"]


def test_qualified_terminal_cannot_hide_a_nonpass_gate():
    value = record()
    value["gates"]["static_family_v2_final"] = {"status": "BLOCKED"}
    assert not validate(value)["valid"]
