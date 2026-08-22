import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.analyze_v914_model_selection_screen import (
    REQUIRED_HEAD,
    audit_screen,
    summarize_models,
)


def _case(index: int) -> dict:
    family = "A" if index < 8 else "B"
    candidate = "v914_endurance_knee_0462" if family == "A" else "v914_endurance_knee_0658"
    local = index % 8
    R = 0.1 if local < 4 else -0.5
    kmax = (12.0, 15.0, 18.0, 21.0)[local % 4]
    return {
        "family": family, "candidate": candidate, "configuration_label": "LEGACY1D-V7",
        "R": R, "Kmax_MPa_sqrt_m": kmax, "deltaK_MPa_sqrt_m": kmax * (1 - R),
        "registry": "/registry.csv", "physics": "/physics.json",
    }


def _screen(tmp_path: Path, statuses: list[str] | None = None) -> Path:
    root = tmp_path / "screen"
    cases = [_case(i) for i in range(16)]
    # The production auditor requires the complete 64-case manifest.
    filler = []
    for i in range(48):
        case = _case(i % 16).copy()
        case["family"] = f"F{i}"
        case["candidate"] = f"candidate_{i}"
        filler.append(case)
    cases += filler
    root.mkdir(parents=True)
    (root / "screen_manifest.json").write_text(json.dumps({"cases": cases}))
    statuses = statuses or ["growth_target_reached"] * 16
    for index, status in enumerate(statuses):
        case = cases[index]
        family = case["family"].replace("-", "_")
        run = root / f"{family}_{case['candidate'].split('_')[-1]}" / (
            f"R{f'{case['R']:g}'.replace('-', 'm').replace('.', 'p')}_Kmax{case['Kmax_MPa_sqrt_m']:g}"
        )
        run.mkdir(parents=True)
        extension = 0.0 if status == "cycle_horizon_reached" else 40e-6
        final_time = 0.0005 if index == 0 else 10.25
        frequency = 1000.0
        exact_cycles = final_time * frequency
        events = [] if extension == 0 else [{"cumulative_cycles": exact_cycles}]
        result = {
            "status": status, "final_time_s": final_time,
            "completed_physical_cycles": math.ceil(exact_cycles),
            "cumulative_extension_m": extension, "event_count": len(events), "events": events,
            "restart_count": 0, "total_cycle_map_evaluations": 7,
        }
        contract = {
            "candidate": case["candidate"], "R": case["R"],
            "deltaK_MPa_sqrt_m": case["deltaK_MPa_sqrt_m"], "frequency_Hz": frequency,
            "temperature_K": 300, "seed": 1720, "configuration_label": "LEGACY1D-V7",
            "repository_head": REQUIRED_HEAD, "repository_clean": True,
            "repository_branch": "branch", "registry_sha256": "r", "physics_sha256": "p",
        }
        (run / "result.json").write_text(json.dumps(result))
        (run / "run_contract.json").write_text(json.dumps(contract))
    return root


def test_subcycle_rate_uses_exact_continuous_cycles(tmp_path):
    rows, _ = audit_screen(_screen(tmp_path), require_complete_ab=True)
    assert rows[0]["exact_cycles"] == pytest.approx(0.5)
    assert rows[0]["da_dN_observed_m_per_cycle"] == pytest.approx(80e-6)


def test_zero_censor_stays_zero_and_plot_ordinate_is_separate(tmp_path):
    statuses = ["cycle_horizon_reached"] + ["growth_target_reached"] * 15
    rows, _ = audit_screen(_screen(tmp_path, statuses), require_complete_ab=True)
    assert rows[0]["da_dN_observed_m_per_cycle"] == 0.0
    assert rows[0]["plot_rate_m_per_cycle"] > 0.0
    assert rows[0]["plot_rate_basis"].startswith("visualization_only")
    assert rows[0]["plot_deltaK_MPa_sqrt_m"] != rows[0]["deltaK_MPa_sqrt_m"]
    assert math.isnan(rows[0]["physical_censor_bound_m_per_cycle"])


def test_work_budget_is_numerical_partial(tmp_path):
    statuses = ["work_budget_exhausted"] + ["growth_target_reached"] * 15
    rows, _ = audit_screen(_screen(tmp_path, statuses), require_complete_ab=True)
    assert rows[0]["terminal_class"] == "numerical_partial"
    assert rows[0]["censor_type"] == "work_budget_exhausted"
    summary = summarize_models(rows)
    assert next(row for row in summary if row["family"] == "A" and row["R"] == 0.1)["interpretation"] == "numerically_unresolved"


def test_cycle_horizon_with_growth_is_not_zero_event_censor(tmp_path):
    statuses = ["cycle_horizon_reached"] + ["growth_target_reached"] * 15
    root = _screen(tmp_path, statuses)
    result_path = root / "A_0462/R0p1_Kmax12/result.json"
    result = json.loads(result_path.read_text())
    result["cumulative_extension_m"] = 2e-6
    result["event_count"] = 1
    result["events"] = [{"cumulative_cycles": 0.25}]
    result_path.write_text(json.dumps(result))
    rows, _ = audit_screen(root)
    assert rows[0]["terminal_class"] == "physical_horizon_with_growth"
    assert rows[0]["censor_type"] == "cycle_horizon_with_nonzero_growth"
    assert rows[0]["da_dN_observed_m_per_cycle"] > 0


@pytest.mark.parametrize("mutation,match", [
    (lambda r, c: r.update(status="mystery"), "unknown terminal status"),
    (lambda r, c: r.update(restart_count=1), "restart_count"),
    (lambda r, c: c.update(deltaK_MPa_sqrt_m=99), "DeltaK mismatch"),
])
def test_audit_fails_closed(tmp_path, mutation, match):
    root = _screen(tmp_path)
    result_path = next(root.rglob("result.json"))
    contract_path = result_path.with_name("run_contract.json")
    result, contract = json.loads(result_path.read_text()), json.loads(contract_path.read_text())
    mutation(result, contract)
    result_path.write_text(json.dumps(result)); contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=match):
        audit_screen(root)


def test_missing_contract_and_incomplete_ab_fail(tmp_path):
    root = _screen(tmp_path)
    path = next(root.rglob("run_contract.json")); path.unlink()
    with pytest.raises(ValueError, match="lacks run contract"):
        audit_screen(root)
    root = _screen(tmp_path / "other")
    result = next(root.rglob("result.json")); result.unlink()
    with pytest.raises(ValueError, match="A/B screen incomplete"):
        audit_screen(root)


def test_duplicate_and_event_cycle_mismatch_fail(tmp_path):
    root = _screen(tmp_path)
    manifest_path = root / "screen_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][1] = manifest["cases"][0]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit_screen(root)


def test_all_synthetic_terminal_rows_retained(tmp_path):
    rows, missing = audit_screen(
        _screen(tmp_path, ["growth_target_reached"] * 64),
        require_complete_ab=True,
    )
    assert len(rows) == 64
    assert not missing


def test_nonfinite_time_fails(tmp_path):
    root = _screen(tmp_path)
    result_path = next(root.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["final_time_s"] = float("nan")
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="final_time_s is not finite"):
        audit_screen(root)
