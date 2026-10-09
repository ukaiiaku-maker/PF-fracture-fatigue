#!/usr/bin/env python3
"""Validate a trusted locally generated mechanics/lifecycle/production tree.

Checkpoint files use the repository's pickle restart format. Never use this
command on an untrusted downloaded bundle. Verify inventory/hashes before any
checkpoint is restored. Failed scientific gates remain valid evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence


def validate_tree(root):
    root=Path(root)
    manifest=json.loads((root/"sha256_manifest.json").read_text())
    actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name!="sha256_manifest.json"}
    if actual!=manifest: raise ValueError("closure bundle inventory/hash mismatch")
    if (root/"mechanics_matrix.json").exists():
        payload=json.loads((root/"mechanics_matrix.json").read_text())
        class Sources:
            def __getitem__(self,key):
                with np.load(root/"sources"/(key+".npz"),allow_pickle=False) as source:
                    return dict(source)
    else:
        class Sources:
            def __getitem__(self,key):
                target=(root/key).resolve()
                if not target.is_relative_to(root.resolve()): raise ValueError("checkpoint escapes bundle")
                return restore_checkpoint(target)
        if (root/"lifecycle_rows.json").exists():
            payload=json.loads((root/"lifecycle_rows.json").read_text())
        else:
            transfer=json.loads((root/"transfer_manifest.json").read_text())
            payload={"schema":transfer["schema"],"executed_code_sha":transfer["executed_code_sha"],
                "transfer_manifest":transfer,"rows":[json.loads((root/f"production_{n}_{r}.json").read_text())
                    for n,r in transfer["physical_registry"]]}
    return validate_closure_evidence(payload,Sources(),executed_code_sha=payload["executed_code_sha"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("directory",type=Path)
    args=parser.parse_args(); print(json.dumps(validate_tree(args.directory),sort_keys=True))
