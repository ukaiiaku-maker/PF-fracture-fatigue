#!/usr/bin/env python3
"""Reconcile analytical allocation with native execution without label inflation."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "integration_evidence/r10_global_mechanism_search"
OUT = BASE / "information_scope_audit"


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def artifacts(candidate: str) -> dict:
    cases = sorted(BASE.glob(f"native_batch*/{candidate}"))
    preflights = [p for case in cases for p in case.glob("preflight.json")]
    fractures = [p for case in cases for p in case.glob("fracture_T_*.json")]
    fatigue = [p for case in cases for p in case.glob("fatigue_Kmax_*/result.json")]
    fdata = [json.loads(p.read_text()) for p in fatigue]
    kdata = [json.loads(p.read_text()) for p in fractures]
    developed = 0
    for d in fdata:
        lo = next((e for e in d.get("events", []) if e["extension_m"] >= 20e-6), None)
        hi = next((e for e in d.get("events", []) if e["extension_m"] >= 60e-6), None)
        developed += bool(lo and hi and hi["cycles"] > lo["cycles"])
    if developed:
        status, reason = "NATIVE_FATIGUE_DEVELOPED", f"{developed} developed load(s)"
    elif fatigue:
        status, reason = "NATIVE_FATIGUE_CENSORED_OR_STOPPED", "native results exist without a developed 20--60 um window"
    elif fractures:
        status, reason = "NATIVE_FRACTURE_ONLY", "fatigue tier pending"
    elif preflights:
        status, reason = "PREFLIGHT_ONLY", "native solve not completed"
    else:
        status, reason = "ALLOCATED_NOT_EXECUTED", "no native artifact"
    return {
        "status": status, "reason": reason, "native_roots": ";".join(p.parent.name for p in cases),
        "preflight_count": len(preflights), "fracture_result_count": len(fractures),
        "fatigue_result_count": len(fatigue), "developed_load_count": developed,
        "native_temperatures_K": ";".join(map(str, sorted({float(d["temperature_K"]) for d in fdata + kdata}))),
        "native_Kmax_MPa_sqrt_m": ";".join(map(str, sorted({float(d["Kmax_MPa_sqrt_m"]) for d in fdata}))),
        "pending": status != "NATIVE_FATIGUE_DEVELOPED", "replaced": False,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = list(csv.DictReader((BASE / "native_batch_registry.csv").open(newline="")))
    desc = {r["candidate_id"]: r for r in csv.DictReader((BASE / "native_batch_descriptors.csv").open(newline=""))}
    ledger = []
    for record in registry:
        d = desc[record["candidate_id"]]
        ledger.append({
            "candidate_id": record["candidate_id"], "family": record["family"],
            "thermal_proxy": d["thermal_acquisition_stratum"],
            "mechanism_acquisition_stratum": d["mechanism_acquisition_stratum"],
            "complete_row_sha256": record["complete_row_sha256"],
            **artifacts(record["candidate_id"]),
        })
    write_csv(OUT / "native_48_execution_ledger.csv", ledger)

    allocation = list(csv.DictReader((BASE / "state_reduction_allocation.csv").open(newline="")))
    selected = {r["candidate_id"] for r in registry}
    scope = []
    for row in allocation:
        a = artifacts(row["candidate_id"])
        evaluated = a["status"] not in ("ALLOCATED_NOT_EXECUTED", "PREFLIGHT_ONLY")
        scope.append({
            "candidate_id": row["candidate_id"],
            "mechanism_acquisition_stratum": row["mechanism_acquisition_stratum"],
            "thermal_proxy": row["thermal_acquisition_stratum"],
            "selected_in_native_48": row["candidate_id"] in selected,
            "native_evaluated": evaluated,
            "native_failed_or_stopped": a["status"] in ("PREFLIGHT_ONLY", "NATIVE_FATIGUE_CENSORED_OR_STOPPED"),
            "queued_unexecuted": not evaluated,
            "status": a["status"],
            "scope_note": "ANALYTICAL_DESCRIPTOR_ALLOCATION_NOT_STATE_AWARE_REDUCTION",
        })
    write_csv(OUT / "allocated_4096_execution_ledger.csv", scope)
    matrix = Counter(("NATIVE_48", r["mechanism_acquisition_stratum"],
                      r["thermal_proxy"], r["status"]) for r in ledger)
    matrix.update(("ALLOCATED_4096", r["mechanism_acquisition_stratum"],
                   r["thermal_proxy"], r["status"]) for r in scope)
    matrix_rows = [{"scope": k[0], "mechanism_acquisition_stratum": k[1],
                    "thermal_proxy": k[2], "native_status": k[3], "count": v}
                   for k, v in sorted(matrix.items())]
    write_csv(OUT / "class_by_mechanism_execution_matrix.csv", matrix_rows)
    payload = {
        "schema": "r10.information-scope-audit/1",
        "native_48": {"allocated": len(ledger), "status_counts": dict(Counter(r["status"] for r in ledger))},
        "allocated_4096": {"allocated": len(scope),
            "selected_in_native_48": sum(r["selected_in_native_48"] for r in scope),
            "native_evaluated": sum(r["native_evaluated"] for r in scope),
            "failed_or_stopped": sum(r["native_failed_or_stopped"] for r in scope),
            "queued_unexecuted": sum(r["queued_unexecuted"] for r in scope)},
        "interpretation": "4096 is an analytical descriptor allocation; it is not a completed state-aware reduction table",
    }
    (OUT / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
