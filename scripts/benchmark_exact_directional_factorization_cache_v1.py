#!/usr/bin/env python3
"""Compare cached exact directional observation with saved qualified results."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.hybrid_directional_drive_v5 import _MARGINAL_FACTORIZATION_CACHE
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from scripts.run_active_two_mechanism_natural_void_study import canonical_drive_check

RUN = ROOT/"runs/managed_sustained_growth_v1_20260920/output"
OUT = ROOT/"runs/managed_sustained_growth_v1_20260920/directional_cache_benchmark_v1.json"
CASES = ("restart_000008475ms.json", "restart_000017475ms.json")


def main() -> None:
    progress = json.loads((RUN/"progress.json").read_text())
    history = {round(row["end_time_s"], 8): row for row in progress["history"]}
    rows = []
    for name in CASES:
        state = restore_checkpoint(RUN/name)
        fingerprint = complete_accepted_state_fingerprint(state)
        saved = history[round(state.junction_process_state["production_time_s"], 8)][
            "canonical_drive"]
        saved_row = next(row for row in saved["canonical_directional_rows"]
                         if row["directional_drive_status"] == "AVAILABLE")
        before = dict(_MARGINAL_FACTORIZATION_CACHE.audit())
        start = time.perf_counter()
        tensor, K, observation, _ = canonical_drive_check(state)
        elapsed = time.perf_counter()-start
        after = dict(_MARGINAL_FACTORIZATION_CACHE.audit())
        observed_row = next(row for row in observation["canonical_directional_rows"]
                            if row["directional_drive_status"] == "AVAILABLE")
        rows.append({
            "checkpoint": name,
            "accepted_time_s": state.junction_process_state["production_time_s"],
            "wall_time_s": elapsed,
            "accepted_state_immutable": (
                complete_accepted_state_fingerprint(state) == fingerprint),
            "G_saved_J_per_m2": saved_row["G_kinetic_used_J_per_m2"],
            "G_cached_J_per_m2": observed_row["G_kinetic_used_J_per_m2"],
            "G_relative_error": abs(observed_row["G_kinetic_used_J_per_m2"]
                                    -saved_row["G_kinetic_used_J_per_m2"])/abs(
                                        saved_row["G_kinetic_used_J_per_m2"]),
            "K_saved_Pa_sqrt_m": saved["state_bound_K_effective_Pa_sqrt_m"],
            "K_cached_Pa_sqrt_m": K,
            "K_relative_error": abs(K-saved["state_bound_K_effective_Pa_sqrt_m"])/abs(
                saved["state_bound_K_effective_Pa_sqrt_m"]),
            "source_tensor_saved_Pa": saved["adjacent_element_tensor_Pa"],
            "source_tensor_cached_Pa": np.asarray(tensor).tolist(),
            "source_tensor_relative_error": float(np.linalg.norm(
                np.asarray(tensor)-np.asarray(saved["adjacent_element_tensor_Pa"]))/max(
                    np.linalg.norm(saved["adjacent_element_tensor_Pa"]), 1e-300)),
            "cache_hits": after["hits"]-before["hits"],
            "cache_misses": after["misses"]-before["misses"],
            "cache_audit": after,
        })
    result = {
        "schema": "unified-2d.exact-directional-factorization-cache-benchmark/1",
        "cases": rows,
        "evolved_state_invalidation_control": (
            "CURRENT_RHS_FIELDS_REASSEMBLED_WHILE_IDENTICAL_OPERATOR_FACTORIZATION_REUSED"),
        "saved_reference_observer_wall_s_per_call": 94.53692308277823/2.0,
        "warm_observer_speedup": (94.53692308277823/2.0)/rows[-1]["wall_time_s"],
        "status": "PASS" if all(
            row["accepted_state_immutable"] and row["G_relative_error"] < 1e-6
            and row["K_relative_error"] < 1e-6
            and row["source_tensor_relative_error"] < 1e-6 for row in rows) else "FAIL",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(OUT)
    print(json.dumps({"status": result["status"], "times": [r["wall_time_s"] for r in rows],
                      "warm_speedup": result["warm_observer_speedup"]}, sort_keys=True))


if __name__ == "__main__":
    main()
