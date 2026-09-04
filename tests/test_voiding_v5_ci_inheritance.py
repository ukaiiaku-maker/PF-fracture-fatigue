import json
from pathlib import Path


def test_machine_readable_general_ci_inheritance_is_exact():
    path = Path("artifacts/voiding_v5_semantic_hardening/general_ci_inheritance.json")
    record = json.loads(path.read_text())
    base = set(record["base"]["failure_ids"])
    head = set(record["v5_checkpoint"]["failure_ids"])
    hardened = set(record["semantic_hardening_head"]["failure_ids"])
    assert len(base) == len(head) == len(hardened) == 7
    assert base == head == hardened
    assert record["semantic_hardening_head"]["git_sha"] == "b786f7c9c5d2175803d50b88d4215853468d4157"
    assert record["semantic_hardening_head"]["run_id"] == 33907591022
    assert record["failure_identity_sets_equal"] is True
    assert record["v5_introduced_new_general_ci_failures"] is False
    assert record["general_repository_ci"] == "FAIL_INHERITED_BASELINE"
