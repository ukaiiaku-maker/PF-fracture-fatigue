#!/usr/bin/env python3
"""Create compact evidence from unchanged long-growth and cyclic histories."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from arrhenius_fracture.fracture_observation_history import from_legacy_steps_csv
from scripts.plot_fracture_observation_history import plot_history

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def summarize(path: Path, *, targets_um=()):
    history = from_legacy_steps_csv(path)
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    lengths = [row.committed_length_m for row in history.rows]
    repeated = len(lengths) - len(set(lengths))
    result = {
        "source": str(path), "source_sha256": sha256(path),
        "raw_rows": len(raw), "adapted_rows": len(history.rows),
        "event_rows": sum(row.aggregate_event_count > 0 for row in history.rows),
        "non_event_rows": sum(row.aggregate_event_count == 0 for row in history.rows),
        "repeated_committed_length_rows": repeated,
        "directions": dict(Counter(row.load_direction.value for row in history.rows)),
        "stages": dict(Counter(row.stage.value for row in history.rows)),
        "validity": dict(Counter(row.validity_status.value for row in history.rows)),
        "native_observers": {
            "K_J_finite_rows": sum(math.isfinite(row.K_J_Pa_sqrt_m) for row in history.rows),
            "K_probe_finite_rows": sum(math.isfinite(row.K_probe_Pa_sqrt_m) for row in history.rows),
            "K_I_finite_rows": sum(math.isfinite(row.K_I_Pa_sqrt_m) for row in history.rows),
        },
        "queries": {},
    }
    for target_um in targets_um:
        query = history.at_length(target_um * 1e-6)
        result["queries"][f"{target_um:g}_um_extension"] = query.to_dict()
    return history, result, raw

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--cyclic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    long_history, long_summary, _ = summarize(args.long, targets_um=(100, 500, 1000))
    cyclic_history, cyclic_summary, cyclic_raw = summarize(args.cyclic)
    header = set(cyclic_raw[0]) if cyclic_raw else set()
    cyclic_summary["cycle_envelope_columns"] = sorted(
        name for name in header if name.startswith("fatigue_K") or name in {"fatigue_cycles"})
    cyclic_summary["within_cycle_phase_samples_present"] = False
    cyclic_summary["qualification_note"] = (
        "accepted outer rows retain cyclic target envelopes and event cycles, but do not "
        "contain the within-cycle unload/reload samples needed for a phase-resolved K(a) curve"
    )
    plot_history(long_history, args.out / "long_growth_complete_history.png")
    plot_history(cyclic_history, args.out / "cyclic_accepted_outer_history.png")
    payload = {"schema": "corrected-k-history-reanalysis/1",
               "source_files_modified": False, "long_growth": long_summary,
               "cyclic_reload_case": cyclic_summary}
    (args.out / "corrected_k_history_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
