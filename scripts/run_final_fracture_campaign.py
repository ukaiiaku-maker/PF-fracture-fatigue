#!/usr/bin/env python3
"""Audit or resume the finite original-PF fracture campaign.

The matrix is the durable source of case identity.  This runner deliberately
supports bounded batches and immutable case directories; it never overwrites a
previous attempt and never treats a failed attestation as a physical censor.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


MATRIX = ROOT / "finalization_control" / "CAMPAIGN_MATRIX.csv"
TEMPLATE_ROOT = ROOT / "integration_evidence" / "current_science_pilots_6abbf10" / "pf"
TEMPLATES = {
    "accommodation_enabled": TEMPLATE_ROOT / "DBTT_1200K" / "accommodation_enabled" / "shared_execution_request.json",
    "opening_only": TEMPLATE_ROOT / "DBTT_1200K" / "opening_only" / "shared_execution_request.json",
}
MATERIAL_OPTIONS = {
    "DBTT": "v913_paper_dbtt01_0202500_persistent_sites",
    "Peak": "v913_paper_peak01_0242980_persistent_sites",
    "Weak-T": "v913_paper_weakT01_0129902_persistent_sites",
    "Ceramic-like": "v913_paper_ceramic01_0077080_persistent_sites",
}
BASELINE_TEMPERATURES = {300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300}
ARMS = {"accommodation_enabled", "opening_only"}


def rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="") as handle:
        return list(csv.DictReader(handle))


def replace_argument(args: list[str], flag: str, value: str) -> None:
    if flag not in args:
        raise ValueError(f"template does not consume required flag {flag}")
    args[args.index(flag) + 1] = value


def validate_matrix(entries: list[dict[str, str]]) -> dict[str, object]:
    baseline = [row for row in entries if row["case_id"].startswith("baseline_")]
    identities = {(r["material_row"], int(r["temperature_K"]), r["arm"]) for r in baseline}
    expected = {(m, t, a) for m in MATERIAL_OPTIONS for t in BASELINE_TEMPERATURES for a in ARMS}
    duplicate_count = len(baseline) - len(identities)
    missing = sorted(expected - identities)
    extra = sorted(identities - expected)
    bad_status = sorted({r["status"] for r in entries} - {
        "running", "repairing", "ready", "complete_validated",
        "complete_physical_censor", "blocked_new_physics", "blocked_external_access",
    })
    reuse = []
    for row in baseline:
        if row["reuse_evidence"]:
            path = ROOT / row["reuse_evidence"]
            reuse.append({
                "case_id": row["case_id"],
                "path": str(path),
                "exists": path.is_dir(),
                "executed": (path / "shared_runtime_attestation.post.json").is_file(),
                "summary": (path / "summary.json").is_file(),
            })
    passed = not missing and not extra and not duplicate_count and not bad_status and all(
        r["exists"] and r["executed"] and r["summary"] for r in reuse
    )
    return {
        "schema": "final-fracture-campaign-audit/1",
        "matrix": str(MATRIX),
        "baseline_row_count": len(baseline),
        "unique_baseline_identities": len(identities),
        "duplicate_count": duplicate_count,
        "missing": missing,
        "extra": extra,
        "bad_status": bad_status,
        "reuse_checks": reuse,
        "passed": passed,
    }


def configured_case(row: dict[str, str]) -> tuple[SharedOptions, list[str]]:
    template = json.loads(TEMPLATES[row["arm"]].read_text())
    native = list(template["native_args"])
    replace_argument(native, "--parameter-option", MATERIAL_OPTIONS[row["material_row"]])
    replace_argument(native, "--temperatures", row["temperature_K"])
    replace_argument(native, "--steps", "500")
    replace_argument(native, "--target-crack-extension-um", "10")
    replace_argument(native, "--crystal-theta-deg", row["orientation_deg"])
    configured = dict(template["options"])
    configured["random_seed"] = int(row["seed"])
    configured["output"] = "compact_smoke"
    if row["source_geography"] != "tip_surface":
        raise ValueError("baseline runner currently accepts only the frozen tip_surface production geography")
    return SharedOptions(**configured), native


def attempt_path(row: dict[str, str]) -> Path:
    base = ROOT / row["output_root"]
    if not base.exists():
        return base
    index = 2
    while (base.parent / f"{base.name}_run{index}").exists():
        index += 1
    return base.parent / f"{base.name}_run{index}"


def successful_attempt(row: dict[str, str]) -> Path | None:
    """Return the newest verified immutable attempt, including suffixed retries."""
    base = ROOT / row["output_root"]
    candidates = [base]
    candidates.extend(sorted(
        base.parent.glob(f"{base.name}_run*"),
        key=lambda path: int(path.name.rsplit("_run", 1)[1])
        if path.name.rsplit("_run", 1)[1].isdigit() else -1,
    ))
    verified = []
    for candidate in candidates:
        attestation = candidate / "shared_runtime_attestation.post.json"
        summary = candidate / "summary.json"
        if not attestation.is_file() or not summary.is_file():
            continue
        payload = json.loads(attestation.read_text())
        if payload.get("execution_verified") is True:
            verified.append(candidate)
    return verified[-1] if verified else None


def close_matrix_row(row: dict[str, str], result: Path) -> None:
    summary = json.loads((result / "summary.json").read_text())
    record = summary[0] if isinstance(summary, list) else summary
    attestation = json.loads(
        (result / "shared_runtime_attestation.post.json").read_text()
    )
    fractured = record.get("Kc_first_MPa_sqrt_m") is not None
    row["status"] = "complete_validated" if fractured else "complete_physical_censor"
    row["reuse_evidence"] = str(result.relative_to(ROOT))
    row["producer_commit"] = str(attestation.get("source", {}).get("commit", ""))[:12]
    row["notes"] = (
        "matrix runner verified original spatial PF execution; physical first event"
        if fractured else
        "matrix runner verified original spatial PF execution; finite no-event censor"
    )


def write_matrix(entries: list[dict[str, str]]) -> None:
    fieldnames = list(entries[0])
    temporary = MATRIX.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    temporary.replace(MATRIX)


def execute_baseline(entries: list[dict[str, str]], *, resume: bool, maximum: int | None,
                     case_id: str | None) -> list[dict[str, object]]:
    selected = [r for r in entries if r["case_id"].startswith("baseline_")]
    if case_id:
        selected = [r for r in selected if r["case_id"] == case_id]
        if not selected:
            raise SystemExit(f"unknown baseline case: {case_id}")
    completed = []
    for row in selected:
        if row["status"] in {"complete_validated", "complete_physical_censor"}:
            completed.append({"case_id": row["case_id"], "action": "reused", "path": row["reuse_evidence"]})
            continue
        existing = successful_attempt(row) if resume else None
        if existing is not None:
            close_matrix_row(row, existing)
            write_matrix(entries)
            completed.append({"case_id": row["case_id"], "action": "existing_success", "path": str(existing)})
            continue
        if maximum is not None and sum(r["action"] == "executed" for r in completed) >= maximum:
            break
        target = attempt_path(row)
        target.mkdir(parents=True)
        options, native = configured_case(row)
        (target / "FINAL_CAMPAIGN_CASE.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        os.environ.update({
            "CLEAVAGE_HAZARD_MODE": "exponential",
            "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
            "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
            "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        })
        execute_shared(options, native, target, source_policy="qualified")
        close_matrix_row(row, target)
        write_matrix(entries)
        completed.append({"case_id": row["case_id"], "action": "executed", "path": str(target)})
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--group", choices=("baseline", "focused-seeds", "sensitivities"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case-id")
    args = parser.parse_args(argv)
    entries = rows()
    audit = validate_matrix(entries)
    if args.audit_only or args.group is None:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["passed"] else 2
    if not audit["passed"]:
        raise SystemExit("campaign matrix failed audit; refusing execution")
    if args.group != "baseline":
        raise SystemExit(f"{args.group} rows are not frozen in the matrix yet")
    result = execute_baseline(entries, resume=args.resume, maximum=args.max_cases, case_id=args.case_id)
    print(json.dumps({"schema": "final-fracture-campaign-batch/1", "results": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
