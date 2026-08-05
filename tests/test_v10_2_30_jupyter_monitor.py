from pathlib import Path
import json

from scripts import v10230_jupyter_monitor as monitor


def dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload))


def test_invalid_json_missing_status_and_missing_growth_rate(tmp_path):
    case=tmp_path/"case"; out=case/"output"; out.mkdir(parents=True)
    (case/"qualification_status.json").write_text("{")
    row,warnings=monitor.case_row(case, {"label":"weakT","fraction":.8})
    assert row["status"] == "unknown" and row["developed_da_dN"] is None


def test_active_dead_pid_and_stale_checkpoint(tmp_path, monkeypatch):
    case=tmp_path/"case"; out=case/"output"; out.mkdir(parents=True)
    dump(case/"qualification_status.json", {"status":"running","pid":123})
    dump(out/"high_cycle_live_checkpoint.json", {"timestamp_unix_s":1,"cycles_from_engine_time":4})
    monkeypatch.setattr(monitor,"pid_alive",lambda _pid:False)
    row,warnings=monitor.case_row(case, now=1000, stale_seconds=10)
    assert row["status"] == "running" and any("dead" in w for w in warnings) and any("stale" in w for w in warnings)


def test_completed_and_censor_are_distinct_and_never_zero_rate(tmp_path):
    complete=tmp_path/"complete"; (complete/"output").mkdir(parents=True)
    dump(complete/"output/developed_fatigue_growth_summary.json", {"target_reached":True,"developed_interval":{"da_dN":2e-12}})
    censored=tmp_path/"censored"; (censored/"output").mkdir(parents=True)
    dump(censored/"output/v10_2_30_fixed_deltaK_control.json", {"censor_status":"right_censored_no_event"})
    assert monitor.case_row(complete)[0]["status"] == "completed_growth"
    row,_=monitor.case_row(censored); assert row["status"] == "right_censored_no_growth" and row["developed_da_dN"] is None


def test_valid_and_invalid_combined_checkpoint(tmp_path):
    from arrhenius_fracture.run_state_checkpoint_v10230 import write_combined_checkpoint
    write_combined_checkpoint(tmp_path, outer={"geometry":{}}, arrays={"a":[1]}, kinetic={"stochastic":{}}, kinetic_vector=[1])
    assert monitor.combined_checkpoint(tmp_path)["valid"]
    descriptor=json.loads((tmp_path/"run_state_checkpoint.json").read_text()); descriptor["files"]["outer.json"]="bad"
    dump(tmp_path/"run_state_checkpoint.json",descriptor)
    assert not monitor.combined_checkpoint(tmp_path)["valid"]


def test_path_validation_and_mixed_provenance_warning(tmp_path):
    repo=tmp_path/"repo"; campaign=repo/"runs"/"campaign"; campaign.mkdir(parents=True)
    assert monitor.validate_campaign_path(repo,campaign)==campaign.resolve()
    import pytest
    with pytest.raises(ValueError): monitor.validate_campaign_path(repo,tmp_path/"elsewhere")


def test_mixed_provenance_warning_and_bounded_event_tail(tmp_path):
    repo=tmp_path/"repo"; root=repo/"runs"/"campaign"; root.mkdir(parents=True)
    cases=[]
    for name,head in (("a","one"),("b","two")):
        case=root/name; (case/"output").mkdir(parents=True); cases.append(case)
        dump(case/"qualification_status.json",{"status":"completed"})
        dump(case/"output/developed_fatigue_growth_summary.json",{"target_reached":True,"provenance":{"git_head":head},"event_measurements":[{"event_index":i} for i in range(5)]})
    snap=monitor.campaign_snapshot(repo,root)
    assert any("mixed Git" in warning for warning in snap["warnings"])
    assert [row["event_index"] for row in monitor.recent_event_details(cases[0],2)]==[3,4]
