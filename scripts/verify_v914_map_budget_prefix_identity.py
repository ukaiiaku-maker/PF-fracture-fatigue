#!/usr/bin/env python3
"""Verify that raising --maximum-cycle-map-evaluations does not change the
physical trajectory already produced under a smaller budget.

Two (or more) fresh runs launched with identical candidate/DeltaK/seed/
physics/tolerances but different map budgets must share an exact common
prefix in both their committed events and their attempted-block history, up
to the point where the smaller budget was exhausted. Any divergence before
that point means changing the allowed work is silently changing an earlier
part of the trajectory, which must be investigated before the comparison is
used to pick a production map budget.
"""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path

# Columns compared for exact/near-exact prefix agreement. Anything not
# listed (e.g. wall-clock-derived or budget-bookkeeping fields) is allowed to
# differ even within the shared prefix.
EVENT_COLUMNS = [
    "cumulative_cycles", "cycle_index", "K_MPa_sqrt_m", "threshold_action",
    "physical_hazard_action", "committed_advance_m", "cumulative_extension_m",
    "tip_radius_m", "mobile_total_m2", "retained_total_m2",
]
BLOCK_COLUMNS = [
    "start_cycle", "end_cycle", "block_stride", "numerical_pass",
    "threshold_crossed", "hazard_clock_action_before", "hazard_threshold_action",
]
ANCHOR_COLUMNS = [
    "cycle_index", "hazard_action", "shielding_MPa_sqrt_m", "mobile_line_content",
    "retained_line_content", "returned_source_slip", "physical_return_fraction",
    "tip_radius_m", "cumulative_source_slip", "raw_return_fraction", "resolution",
    "hazard_clock_action", "hazard_threshold_action", "block_stride", "block_hazard_action",
]
# run_contract.json fields that are *expected* to differ between an old
# baseline launch and a fresh qualification launch, and must not be treated
# as a physics/numerical divergence.
CONTRACT_ALLOWED_TO_DIFFER = {"repository_head"}


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _as_float_or_str(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def compare_prefix(rows_a: list[dict], rows_b: list[dict], columns: list[str], rtol: float = 1.0e-9) -> dict:
    """Compare the shared prefix of two row sequences on the given columns.

    Returns a dict with `identical`, `compared_rows`, and, on the first
    divergence, `row_index`/`column`/`value_a`/`value_b`.
    """
    n = min(len(rows_a), len(rows_b))
    for i in range(n):
        for col in columns:
            if col not in rows_a[i] or col not in rows_b[i]:
                continue
            va = _as_float_or_str(rows_a[i][col])
            vb = _as_float_or_str(rows_b[i][col])
            if isinstance(va, float) and isinstance(vb, float):
                if va == vb:
                    continue
                denom = max(abs(va), abs(vb), 1.0e-300)
                if abs(va - vb) / denom > rtol:
                    return {"identical": False, "compared_rows": n, "row_index": i, "column": col, "value_a": va, "value_b": vb}
            elif va != vb:
                return {"identical": False, "compared_rows": n, "row_index": i, "column": col, "value_a": va, "value_b": vb}
    return {"identical": True, "compared_rows": n}


def compare_run_contracts(contract_a: dict, contract_b: dict) -> dict:
    """Every field must agree except CONTRACT_ALLOWED_TO_DIFFER."""
    keys = sorted(set(contract_a) | set(contract_b))
    for key in keys:
        if key in CONTRACT_ALLOWED_TO_DIFFER:
            continue
        va, vb = contract_a.get(key), contract_b.get(key)
        if va != vb:
            return {"identical": False, "field": key, "value_a": va, "value_b": vb}
    return {"identical": True}


def verify_pair(label_a: str, dir_a: Path, label_b: str, dir_b: Path) -> dict:
    result = {"pair": f"{label_a} vs {label_b}"}
    contract_a = json.loads((dir_a / "run_contract.json").read_text())
    contract_b = json.loads((dir_b / "run_contract.json").read_text())
    result["run_contract"] = compare_run_contracts(contract_a, contract_b)
    events_a, events_b = load_csv_rows(dir_a / "events.csv"), load_csv_rows(dir_b / "events.csv")
    result["events"] = compare_prefix(events_a, events_b, EVENT_COLUMNS)
    result["events"]["rows_a"] = len(events_a)
    result["events"]["rows_b"] = len(events_b)
    blocks_a, blocks_b = load_csv_rows(dir_a / "block_history.csv"), load_csv_rows(dir_b / "block_history.csv")
    result["blocks"] = compare_prefix(blocks_a, blocks_b, BLOCK_COLUMNS)
    result["blocks"]["rows_a"] = len(blocks_a)
    result["blocks"]["rows_b"] = len(blocks_b)
    anchors_a, anchors_b = load_csv_rows(dir_a / "anchor_history.csv"), load_csv_rows(dir_b / "anchor_history.csv")
    result["anchors"] = compare_prefix(anchors_a, anchors_b, ANCHOR_COLUMNS)
    result["anchors"]["rows_a"] = len(anchors_a)
    result["anchors"]["rows_b"] = len(anchors_b)
    result["prefix_identical"] = bool(
        result["run_contract"]["identical"]
        and result["events"]["identical"]
        and result["blocks"]["identical"]
        and result["anchors"]["identical"]
    )
    return result


def git_diff_is_empty(repo: Path, old_ref: str, new_ref: str, paths: list[str]) -> dict:
    """Confirm no engine-relevant file changed between two repository refs."""
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--stat", old_ref, new_ref, "--", *paths],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {"old_ref": old_ref, "new_ref": new_ref, "unchanged": diff == "", "diff_stat": diff}


def external_engine_repo_fingerprint(external_repo: Path) -> dict:
    """Fingerprint the out-of-repo V914_ROOT physics package the runner
    actually imports (`fatigue_v914`, `endurance_knee_v914`, ...), since the
    in-repo --expected-head contract says nothing about it."""
    head = subprocess.run(
        ["git", "-C", str(external_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(external_repo), "status", "--short"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {"repo": str(external_repo), "head": head, "clean": status.strip() == ""}


ENGINE_FILES = [
    "scripts/run_v914_v7_vhcf_event_to_event.py", "scripts/v914_adaptive_feedback_v6.py",
    "scripts/v914_intrinsic_reverse_glide_v7.py", "scripts/v914_signed_fatigue_loading.py",
    "scripts/v914_v7_adaptive_block_accelerator.py", "scripts/v914_v7_vhcf_event_engine.py",
    "scripts/v914_v7_rate_separated_dmd.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, help="label:directory, given in ascending map-budget order; e.g. 4096:/path/to/run 6144:/path/to/run")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--old-ref", action="append", default=[], help="a repository_head an older baseline was launched at; repeatable")
    ap.add_argument("--new-ref", default=None, help="the ref the qualification cases were launched at; defaults to current HEAD")
    ap.add_argument("--external-engine-repo", type=Path, default=None, help="the out-of-repo V914_ROOT physics package directory, if applicable")
    a = ap.parse_args()
    cases = []
    for spec in a.case:
        label, _, path = spec.partition(":")
        cases.append((label, Path(path)))

    report = {"schema": "v914_map_budget_prefix_identity_v1", "cases": [label for label, _ in cases]}

    new_ref = a.new_ref or subprocess.run(["git", "-C", str(a.repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    report["code_fingerprint"] = {
        "in_repo_engine_diffs": [git_diff_is_empty(a.repo, old, new_ref, ENGINE_FILES) for old in a.old_ref],
    }
    if a.external_engine_repo is not None:
        report["code_fingerprint"]["external_engine_repo"] = external_engine_repo_fingerprint(a.external_engine_repo)
    fingerprint_ok = all(d["unchanged"] for d in report["code_fingerprint"]["in_repo_engine_diffs"])
    if a.external_engine_repo is not None:
        fingerprint_ok = fingerprint_ok and report["code_fingerprint"]["external_engine_repo"]["clean"]
    report["code_fingerprint"]["ok"] = fingerprint_ok

    report["pairs"] = []
    prefix_ok = True
    for i in range(len(cases) - 1):
        label_a, dir_a = cases[i]
        label_b, dir_b = cases[i + 1]
        pair = verify_pair(label_a, dir_a, label_b, dir_b)
        report["pairs"].append(pair)
        prefix_ok = prefix_ok and pair["prefix_identical"]
    report["all_prefixes_identical"] = prefix_ok
    report["qualification_admissible"] = bool(fingerprint_ok and prefix_ok)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["qualification_admissible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
