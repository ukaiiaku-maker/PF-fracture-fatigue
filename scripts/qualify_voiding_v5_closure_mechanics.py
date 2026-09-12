#!/usr/bin/env python3
"""Run the frozen unique static mechanics matrix without family aliases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture.closure_mechanics_evidence import SCHEMA, REGISTRY, GROUPS, measurements, predicates
from arrhenius_fracture.closure_static_evidence import source_fingerprints, environment_identity
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output", type=Path)
    args = parser.parse_args(); out = args.output
    if out.exists() and any(out.iterdir()): raise ValueError("refusing to overwrite evidence")
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise RuntimeError("mechanics evidence requires a clean committed implementation")
    sha = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    (out/"sources").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (key, cfg) in enumerate(REGISTRY.items()):
        print(f"Unique mechanics solve {index+1}/{len(REGISTRY)} {key[:12]}", flush=True)
        solved = solve_crack_void_case(**cfg); raw = solved["source_capture"]
        np.savez_compressed(out/"sources"/(key+".npz"), **raw)
        row = {"case_id": key, "input_configuration": cfg, "input_hash": key,
               "execution_id": key+":"+sha, "executed_code_sha": sha,
               "source_fingerprints": source_fingerprints(raw), "environment_identity": environment_identity(),
               "measurement_source": "sources/"+key+".npz", "measurements": measurements(raw, cfg)}
        rows.append(row)
        write_json(out/(key+".json"), row)
    payload = {"schema": SCHEMA, "executed_code_sha": sha, "base_rows": rows,
               "derived_rows": predicates({r["case_id"]: r for r in rows}), "family_source_references": GROUPS}
    # Load one source on demand; don't retain all fine sparse systems in RAM.
    class Sources:
        def __getitem__(self, key):
            with np.load(out/"sources"/(key+".npz"), allow_pickle=False) as source:
                return dict(source)
    validated = validate_closure_evidence(payload, Sources(), executed_code_sha=sha)
    write_json(out/"mechanics_matrix.json", payload)
    write_json(out/"ontology_validation.json", validated)
    write_json(out/"sha256_manifest.json", {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(validated), flush=True)


if __name__ == "__main__": main()
