import csv
import json

from arrhenius_fracture.branch_output_v11 import BRANCH_EVENT_FIELDS, TRIAL_FIELDS, BranchOutputWriter


def record(fields, **overrides):
    value = {field: 0 for field in fields}; value.update(overrides); return value


def test_outputs_have_stable_headers_are_restart_safe_and_complete_atomically(tmp_path):
    writer = BranchOutputWriter(tmp_path)
    trial = record(TRIAL_FIELDS, trial_id="trial-1", action_type="two_arm", accepted=True)
    writer.append_trial(trial); writer.append_trial({**trial, "trial_id": "trial-2"})
    assert [json.loads(line)["trial_id"] for line in (tmp_path / "branch_action_trials.jsonl").read_text().splitlines()] == ["trial-1", "trial-2"]
    event = record(BRANCH_EVENT_FIELDS, event_record_id="event-1")
    assert writer.branch_event(event); assert not writer.branch_event(event)
    with (tmp_path / "branch_events.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream)); assert tuple(rows[0]) == BRANCH_EVENT_FIELDS; assert len(rows) == 1
    writer.complete(status="target_reached", final_checkpoint="checkpoint/final.json", validation={"checkpoint": True})
    assert json.loads((tmp_path / "run_complete.json").read_text())["status"] == "target_reached"
    assert not list(tmp_path.glob("*.tmp"))
