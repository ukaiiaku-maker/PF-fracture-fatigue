#!/usr/bin/env python3
"""Execute the frozen unique traction registry and its static closure ontology."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.closure_static_evidence import (
    SCHEMA, TRACTION_REGISTRY, source_fingerprints, environment_identity,
    recompute_measurements, derived_rows,
)
from arrhenius_fracture.finalization_v3_schema import canonical_hash
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise RuntimeError("evidence requires a clean committed implementation")
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite existing evidence")
    (out/"sources").mkdir(parents=True, exist_ok=True)
    bases, sources = {}, {}
    for key, cfg in TRACTION_REGISTRY.items():
        print("Executing unique physical solve "+key, flush=True)
        result = solve_crack_void_case(**cfg)
        raw = result["source_capture"]
        source_path = "sources/"+key.replace(":", "_")+".npz"
        np.savez_compressed(out/source_path, **raw)
        fingerprints = source_fingerprints(raw)
        bases[key] = {"case_id": key, "execution_id": key+":"+head,
            "executed_code_sha": head, "input_configuration": cfg,
            "input_hash": canonical_hash(cfg), "source_fingerprints": fingerprints,
            "realized_geometry_fingerprint": canonical_hash({k: fingerprints[k] for k in ("mesh", "support")}),
            "environment_identity": environment_identity(), "measurement_source": source_path,
            "operation_trace": ["build_mesh", "realize_V12_support", "assemble", "solve", "recover"],
            "measurements": recompute_measurements(raw, cfg)}
        sources[key] = raw
    payload = {"schema": SCHEMA, "executed_code_sha": head,
               "base_rows": list(bases.values()), "derived_rows": derived_rows(bases)}
    validation = validate_closure_evidence(payload, sources, executed_code_sha=head)
    write_json(out/"traction_convergence.json", payload)
    write_json(out/"ontology_validation.json", validation)
    write_json(out/"sha256_manifest.json", {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(validation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
