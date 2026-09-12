#!/usr/bin/env python3
"""Independently audit topology/accounting in a trusted production bundle.

This restores local pickle checkpoints: never use an untrusted bundle. The
complete inventory and hashes are verified before any checkpoint is restored.
No FEM solve, refinement, or new physical execution is performed. Failed
scientific predicates remain in the output and produce a nonzero exit status.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.closure_lifecycle_evidence import conservation, stagewise_topology
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.finalization_v3_schema import canonical_hash
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint

LEVELS = ((32, 12), (64, 24), (128, 48), (512, 192))
SCHEMA = "v12.closure-production-derived-topology-diagnostic/1"


def verify_manifest(directory):
    root = Path(directory).resolve()
    manifest_path = root / "sha256_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("symlink manifest is not accepted")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an inventory mapping")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in production bundle")
        if path.is_file() and path != manifest_path:
            actual[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest:
        raise ValueError("production bundle inventory/hash mismatch")
    return hashlib.sha256(raw).hexdigest()


def auditor_identity():
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise RuntimeError("derived topology audit requires a clean committed implementation")
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def audit_bundle(directory):
    root = Path(directory).resolve()
    manifest_hash = verify_manifest(root)  # Must precede every pickle restore.
    auditor_sha = auditor_identity()
    transfer = json.loads((root / "transfer_manifest.json").read_text())
    if transfer.get("schema") != "v12.production-source-transfer/2":
        raise ValueError("unsupported production bundle schema")
    if transfer.get("physical_registry") != [list(level) for level in LEVELS]:
        raise ValueError("production resolution registry mismatch")
    producer_sha = transfer["executed_code_sha"]
    if (not isinstance(producer_sha, str) or len(producer_sha) != 40
            or any(ch not in "0123456789abcdef" for ch in producer_sha)):
        raise ValueError("invalid production implementation SHA")
    result = {"schema": SCHEMA,
        "classification": "DERIVED_EXISTING_CHECKPOINTS_NOT_NEW_PHYSICAL_EXECUTIONS",
        "audit_code_sha": auditor_sha, "producer_code_sha": producer_sha,
        "input_manifest_sha256": manifest_hash, "rows": []}
    for n, r in LEVELS:
        row = {"case_id": f"production:{n}:{r}", "passed": False}
        try:
            recorded = json.loads((root / f"production_{n}_{r}.json").read_text())
            if (recorded.get("case_id") != row["case_id"]
                    or recorded.get("executed_code_sha") != producer_sha
                    or recorded.get("connection_executed") is not True):
                raise ValueError("production connection identity/provenance mismatch")
            pre = restore_checkpoint(root / f"checkpoints/production_{n}_{r}_pre.json")
            connected = restore_checkpoint(root / f"checkpoints/production_{n}_{r}_connected.json")
            pre_fp, connected_fp = fingerprint(pre), fingerprint(connected)
            if (pre_fp != recorded.get("initial_state_fingerprint")
                    or connected_fp != recorded.get("connected_state_fingerprint")):
                raise ValueError("production checkpoint fingerprint mismatch")
            topology = stagewise_topology(connected)
            account = conservation(connected, pre)
            row.update({"pre_state_fingerprint": pre_fp,
                "connected_state_fingerprint": connected_fp,
                "topology_passed": bool(topology["passed"]), "conservation": account,
                "topology_source_fingerprint": topology["topology_source_fingerprint"],
                "certificate_fingerprints": {key: canonical_hash(canonical_data(topology[key]))
                    for key in ("independent_cut", "actual_cavity_cycle", "actual_component_incidence")},
                "independent_cut_no_intact_path": not topology["independent_cut"]["intact_cross_graph_path_exists"],
                "independent_cut_has_sufficient_seeds": not topology["independent_cut"]["insufficient_seed_segment_ids"],
                "cavity_cycle_passed": bool(topology["actual_cavity_cycle"]["passed"]),
                "component_incidence_passed": bool(topology["actual_component_incidence"]["passed"]),
                "pre_unchanged": fingerprint(pre) == pre_fp,
                "connected_unchanged": fingerprint(connected) == connected_fp})
            row["passed"] = bool(all(row[key] for key in ("topology_passed",
                "independent_cut_no_intact_path", "independent_cut_has_sufficient_seeds",
                "cavity_cycle_passed", "component_incidence_passed", "pre_unchanged",
                "connected_unchanged")) and account["passed"])
        except Exception as error:
            row["exception"] = {"type": type(error).__name__, "message": str(error)}
        result["rows"].append(row)
    # Recheck both evidence immutability and clean exact auditor identity.
    result["input_bundle_unchanged"] = verify_manifest(root) == manifest_hash
    result["audit_worker_still_clean"] = auditor_identity() == auditor_sha
    result["passed"] = bool(result["input_bundle_unchanged"]
        and result["audit_worker_still_clean"] and all(row["passed"] for row in result["rows"]))
    return canonical_data(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(args.bundle.resolve()):
        raise ValueError("derived output must be outside the source bundle")
    if output.exists():
        raise ValueError("refusing to overwrite derived output")
    result = audit_bundle(args.bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "physical_cases": len(result["rows"]),
        "classification": result["classification"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
