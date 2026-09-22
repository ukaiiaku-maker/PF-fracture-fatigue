#!/usr/bin/env python3
"""Execute the predeclared 98%-opening validation after OneD is frozen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from scripts.generate_load_aware_supports_v1 import RUN, START, run_case

PREDICTION = Path("/Volumes/Data/Data/Nanopillar_calculation/voiding/"
                  "oneD-v3-stateful-voiding-reduction/analysis_outputs/"
                  "oneD_v3_load_aware_98pct_prediction_v1/prediction.json")
OUT = ROOT/"artifacts/nearby_load_validation_98pct_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not PREDICTION.exists():
        raise RuntimeError("OneD validation prediction must be frozen first")
    prediction_sha = sha(PREDICTION)
    reference = restore_checkpoint(START)
    fingerprint = complete_accepted_state_fingerprint(reference)
    wall = time.perf_counter()
    final, row = run_case(reference, 0.98, "HELD_OUT_VALIDATION_ONLY")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = write_checkpoint(final, OUT/"validation_0p98.json", compression="gzip")
    result = {
        "schema": "unified-2d.nearby-load-validation/1",
        "classification": "INDEPENDENT_HELD_OUT_AFTER_ONED_PREDICTION_FROZEN",
        "prediction_path": str(PREDICTION),
        "prediction_sha256_before_validation": prediction_sha,
        "reference_checkpoint": str(START),
        "reference_state_fingerprint": fingerprint,
        "reference_state_preserved": (
            complete_accepted_state_fingerprint(reference) == fingerprint),
        "case": row, "checkpoint": manifest,
        "wall_time_s": time.perf_counter()-wall,
    }
    (OUT/"validation.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(OUT/"validation.json")
    print("HELD_OUT_98PCT_COMPLETE", row["endpoint_H"], row["scheduler_events"])


if __name__ == "__main__":
    main()
