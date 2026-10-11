#!/usr/bin/env python3
"""Build the R2 runtime in two fresh workers and compare final bindings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


WORKER = r'''
import json
from arrhenius_fracture.reduced_persistent_reference_r2 import build_restored_engine, runtime_receipt
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig
engine = build_restored_engine(SharedReducedConfig(mpz_length_um=50.0, mpz_n_bins=80, wake_length_um=50.0, max_internal_steps=100000))
receipt = runtime_receipt(engine, stage="fresh_worker")
reversible = dict(receipt["reversible_binding"])
# A CPython object address is process-local, not part of the binding contract.
reversible.pop("object_id", None)
print(json.dumps({
    "engine_class": receipt["engine_class"],
    "state_class": receipt["state_class"],
    "methods": receipt["methods"],
    "reversible_binding": reversible,
    "persistent_invariants": receipt["persistent_invariants"],
    "source_geometry": receipt["source_geometry"],
    "taylor_contract": receipt["taylor_contract"],
}, sort_keys=True))
'''


def attest() -> dict:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    workers = []
    for _ in range(2):
        completed = subprocess.run(
            (sys.executable, "-c", WORKER), cwd=root, env=environment,
            check=True, capture_output=True, text=True,
        )
        workers.append(json.loads(completed.stdout))
    return {
        "schema": "r2-runtime-install-order-isolation/1",
        "fresh_process_count": 2,
        "receipts_exact": workers[0] == workers[1],
        "worker_receipts": workers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = attest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fresh_process_count": payload["fresh_process_count"],
        "receipts_exact": payload["receipts_exact"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if payload["receipts_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
