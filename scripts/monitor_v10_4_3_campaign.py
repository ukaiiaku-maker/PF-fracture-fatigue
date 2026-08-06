#!/usr/bin/env python3
"""Monitor v10.4.3 fracture versus plastic-dominance campaigns."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


DEFAULT_OUTROOT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_"
    "four_class_1000um_reuse17_base3621_v1"
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _fmt(value: Any, spec: str = ".3g", missing: str = "-") -> str:
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _pid_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"pid": None, "alive": False, "state": "no_pid_file"}
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return {"pid": None, "alive": False, "state": "invalid_pid_file"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    else:
        alive = True
    return {
        "pid": pid,
        "alive": alive,
        "state": "running" if alive else "stale_pid_file",
    }


def _canonical_cases(root: Path) -> list[dict[str, Any]]:
    seed_map = root / "v10_2_27_case_seed_map.csv"
    manifest = _json(root / "v10_2_27_campaign_manifest.json")
    if not seed_map.is_file():
        cases = []
        for case_root in sorted(root.glob("*/T*K_th*_seed*")):
            if case_root.is_dir():
                cases.append({
                    "option": case_root.parent.name,
                    "temperature_K": None,
                    "seed": None,
                    "case_root": case_root,
                })
        return cases

    theta = float(manifest.get("crystal_theta_deg", 0.0))
    theta_tag = f"{theta:g}"
    cases = []
    with seed_map.open(newline="") as stream:
        for row in csv.DictReader(stream):
            temperature = float(row["temperature_K"])
            seed = int(row["seed"])
            option = row["option"]
            case_root = (
                root
                / option
                / f"T{temperature:g}K_th{theta_tag}_seed{seed}"
            )
            cases.append({
                "option": option,
                "temperature_K": temperature,
                "seed": seed,
                "case_root": case_root,
            })
    return cases


def _metric(payloads: list[dict[str, Any]], *names: str) -> Any:
    for payload in payloads:
        for name in names:
            if name in payload and payload[name] is not None:
                return payload[name]
    return None


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    root: Path = case["case_root"]
    status = _json(root / "stage3_case_status.json")
    terminal = _json(root / "plastic_flow_terminal_audit.json")
    candidate = _json(root / "plastic_flow_candidate_latest.json")
    payloads = [terminal, candidate, status]

    has_complete = (root / "COMPLETE").is_file()
    has_plastic = (root / "PLASTIC_FLOW").is_file()
    has_failed = (root / "RUN_FAILED").exists()
    has_reuse = (root / "v10_4_2_reuse_audit.json").is_file()
    has_log = (root / "run.log").is_file()

    if has_failed:
        state = "failed"
    elif has_plastic:
        state = "plastic_dominance"
    elif has_complete:
        state = "reused_fracture" if has_reuse else "fracture"
    elif has_log or candidate:
        state = "active_or_interrupted"
    elif root.exists():
        state = "created"
    else:
        state = "pending"

    criteria = _metric(payloads, "criteria")
    if not isinstance(criteria, dict):
        criteria = {}
    failed_criteria = _metric(payloads, "failed_criteria")
    if not isinstance(failed_criteria, list):
        failed_criteria = [name for name, passed in criteria.items() if not passed]

    energy_error = max(
        float(_metric(payloads, "window_energy_balance_relative_error") or 0.0),
        float(
            _metric(payloads, "cumulative_energy_balance_relative_error") or 0.0
        ),
        float(_metric(payloads, "energy_balance_relative_error") or 0.0),
    )

    log_path = root / "run.log"
    log_size = log_path.stat().st_size if log_path.is_file() else 0
    log_mtime = log_path.stat().st_mtime if log_path.is_file() else None

    return {
        **{key: case.get(key) for key in ("option", "temperature_K", "seed")},
        "case_root": str(root),
        "state": state,
        "reused": has_reuse,
        "terminal": has_complete or has_plastic,
        "status": status.get("status"),
        "projected_extension_um": status.get("projected_extension_um"),
        "Kc_first_MPa_sqrt_m": status.get("Kc_first_MPa_sqrt_m"),
        "plastic_accommodation_ratio": _metric(
            payloads,
            "plastic_accommodation_ratio_median",
            "plastic_accommodation_ratio",
        ),
        "elastic_accommodation_ratio": _metric(
            payloads,
            "elastic_accommodation_ratio_median",
            "elastic_accommodation_ratio",
        ),
        "active_plastic_area_fraction": _metric(
            payloads,
            "active_plastic_area_fraction_median",
            "active_plastic_area_fraction",
        ),
        "normalized_tangent_stiffness": _metric(
            payloads,
            "normalized_tangent_stiffness",
        ),
        "reaction_force_fraction_of_peak": _metric(
            payloads,
            "reaction_force_fraction_of_peak_window_median",
            "reaction_force_fraction_of_peak",
        ),
        "J_tip_positive_J_per_m2": _metric(
            payloads,
            "J_tip_positive_max_window_J_per_m2",
            "J_tip_positive_final_J_per_m2",
        ),
        "J_contour_shielding_J_per_m2": _metric(
            payloads,
            "J_contour_shielding_J_per_m2",
        ),
        "B_final": _metric(payloads, "B_final"),
        "energy_balance_relative_error": energy_error,
        "stagger_relative_change": _metric(
            payloads,
            "stagger_relative_change_max",
            "stagger_relative_change",
        ),
        "failed_criteria": failed_criteria,
        "log_bytes": log_size,
        "log_modified_unix_s": log_mtime,
    }


def snapshot(root: Path) -> dict[str, Any]:
    cases = [_case_record(case) for case in _canonical_cases(root)]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["state"]] = counts.get(case["state"], 0) + 1

    logs = sorted(
        (root / "v10_4_3_logs").glob("*.log"),
        key=lambda path: path.stat().st_mtime,
    ) if (root / "v10_4_3_logs").is_dir() else []
    latest_log = logs[-1] if logs else None
    tail: list[str] = []
    if latest_log is not None:
        try:
            tail = latest_log.read_text(errors="replace").splitlines()[-12:]
        except Exception:
            tail = []

    candidates = [
        case for case in cases
        if case["plastic_accommodation_ratio"] is not None
        and not case["terminal"]
    ]
    candidates.sort(
        key=lambda item: float(item["plastic_accommodation_ratio"]),
        reverse=True,
    )

    return {
        "schema": "v10.4.3_campaign_monitor_v1",
        "timestamp": datetime.now().astimezone().isoformat(),
        "outroot": str(root),
        "pid": _pid_status(root / "v10_4_3_campaign.pid"),
        "planned_cases": len(cases),
        "terminal_cases": sum(bool(case["terminal"]) for case in cases),
        "counts": counts,
        "cases": cases,
        "leading_plastic_candidates": candidates[:8],
        "latest_campaign_log": str(latest_log) if latest_log else None,
        "latest_campaign_log_tail": tail,
    }


def _print_case(case: dict[str, Any]) -> None:
    option = str(case["option"]).replace("v913_paper_", "").replace(
        "_persistent_sites", ""
    )
    failed = ",".join(case["failed_criteria"][:3]) or "-"
    print(
        f"  {option:30s} T={_fmt(case['temperature_K'], '.0f'):>4s} "
        f"state={case['state']:21s} "
        f"Phi_p={_fmt(case['plastic_accommodation_ratio']):>7s} "
        f"active={_fmt(case['active_plastic_area_fraction']):>7s} "
        f"Kt/K0={_fmt(case['normalized_tangent_stiffness']):>7s} "
        f"Eerr={_fmt(case['energy_balance_relative_error']):>7s} "
        f"failed={failed}"
    )


def print_snapshot(report: dict[str, Any]) -> None:
    pid = report["pid"]
    print(f"v10.4.3 monitor — {report['timestamp']}")
    print(f"Root: {report['outroot']}")
    print(
        f"Launcher: {pid['state']}"
        + (f" pid={pid['pid']}" if pid.get("pid") else "")
    )
    print(
        f"Cases: planned={report['planned_cases']} "
        f"terminal={report['terminal_cases']} "
        + " ".join(
            f"{name}={count}"
            for name, count in sorted(report["counts"].items())
        )
    )

    failed = [case for case in report["cases"] if case["state"] == "failed"]
    active = [
        case for case in report["cases"]
        if case["state"] in {"active_or_interrupted", "created"}
    ]
    if failed:
        print("\nFailures:")
        for case in failed[:12]:
            _print_case(case)
    if active:
        print("\nActive/interrupted cases:")
        for case in active[:12]:
            _print_case(case)
    if report["leading_plastic_candidates"]:
        print("\nLeading nonterminal plastic-dominance candidates:")
        for case in report["leading_plastic_candidates"]:
            _print_case(case)

    if report["latest_campaign_log"]:
        print(f"\nLatest supervisor log: {report['latest_campaign_log']}")
        for line in report["latest_campaign_log_tail"]:
            print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outroot",
        type=Path,
        default=Path(os.environ.get("OUTROOT", DEFAULT_OUTROOT)),
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.outroot.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"campaign root does not exist: {root}")
    if args.interval < 10.0:
        parser.error("--interval must be at least 10 seconds")

    while True:
        report = snapshot(root)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print_snapshot(report)
        if not args.watch:
            return 2 if report["counts"].get("failed", 0) else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
