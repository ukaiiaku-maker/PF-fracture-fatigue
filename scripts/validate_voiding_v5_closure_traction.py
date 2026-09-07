#!/usr/bin/env python3
"""Validate raw source manifests, static ontology, and optional canonical peer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence

SOURCE_FILES = (
    "arrhenius_fracture/closure_static_evidence.py", "arrhenius_fracture/explicit_cavity_v5.py",
    "arrhenius_fracture/crack_void_mechanics_v5.py", "arrhenius_fracture/fem.py",
    "arrhenius_fracture/mesh.py", "arrhenius_fracture/config.py",
    "arrhenius_fracture/mechanically_separating_sharp_wake_v12.py",
    "scripts/qualify_voiding_v5_closure_traction.py",
)


def validate_tree(root):
    manifest = json.loads((root/"sha256_manifest.json").read_text())
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and p.name != "sha256_manifest.json"}
    if manifest != actual:
        raise ValueError("exact manifest inventory/hash mismatch")
    payload = json.loads((root/"traction_convergence.json").read_text())
    sources = {}
    for row in payload["base_rows"]:
        path = (root/row["measurement_source"]).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError("source path escapes evidence tree")
        with np.load(path, allow_pickle=False) as loaded:
            sources[row["case_id"]] = dict(loaded)
    validation = validate_closure_evidence(payload, sources, executed_code_sha=payload["executed_code_sha"])
    if validation != json.loads((root/"ontology_validation.json").read_text()):
        raise ValueError("ontology validation record mismatch")
    return payload


def canonical_physics(payload):
    payload = json.loads(json.dumps(payload))
    del payload["executed_code_sha"]
    for row in payload["base_rows"]:
        del row["execution_id"]; del row["executed_code_sha"]
        # Both environments remain recorded and independently validated. No
        # numerical, geometry, raw-array, or solver fingerprints are ignored.
        del row["environment_identity"]
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("--canonical", type=Path)
    args = parser.parse_args()
    payload = validate_tree(args.tree)
    if args.canonical:
        canonical = validate_tree(args.canonical)
        for sha in (canonical["executed_code_sha"], payload["executed_code_sha"]):
            subprocess.run(("git", "cat-file", "-e", sha+"^{commit}"), cwd=ROOT, check=True)
        changed = subprocess.check_output(("git", "diff", "--name-only", canonical["executed_code_sha"],
            payload["executed_code_sha"], "--", *SOURCE_FILES), cwd=ROOT, text=True)
        if changed.strip(): raise ValueError("canonical implementation differs: "+changed)
        if canonical_physics(payload) != canonical_physics(canonical):
            raise ValueError("exact canonical physics/source comparison differs")
        print("EXACT_CANONICAL_PHYSICS_AND_SOURCE_COMPARISON=PASS")
    print("STATIC_CLOSURE_ONTOLOGY_AND_MANIFESTS=PASS")


if __name__ == "__main__": main()
