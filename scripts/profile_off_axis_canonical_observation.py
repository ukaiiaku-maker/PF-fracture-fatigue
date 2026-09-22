#!/usr/bin/env python3
"""Profile one saved post-event off-axis canonical mechanical observation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import sharp_front_load_provider
from scripts.run_off_axis_prescribed_void_benchmark_v2 import KINETIC_DELTA_A


DEFAULT_RESTART = ROOT / (
    "artifacts/informative_off_axis_active_trajectories_iterative/"
    "above/step_002_restart.json"
)
OUT = ROOT / "artifacts/informative_off_axis_canonical_profile"


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if hasattr(value, "to_dict"):
        return _finite(value.to_dict())
    if hasattr(value, "__dict__"):
        return _finite(vars(value))
    return value


def _instrument():
    import arrhenius_fracture.adaptive_multitip_mesh_v11 as adaptive
    import arrhenius_fracture.hybrid_directional_drive_v5 as hybrid
    import arrhenius_fracture.vacancy_transport_phase_field_v2 as transport
    import arrhenius_fracture.voiding_production_v5 as voiding

    records = defaultdict(list)

    def wrap(module, name, category, size_getter=None):
        original = getattr(module, name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            result = original(*args, **kwargs)
            row = {"wall_time_s": time.perf_counter() - started}
            if size_getter is not None:
                row.update(size_getter(result))
            records[category].append(row)
            return result

        setattr(module, name, measured)

    wrap(voiding, "_conform_bulk_point", "candidate_endpoint_conformity",
         lambda state: {"mechanics_nodes": state.mesh.nn,
                        "mechanics_elements": state.mesh.ne})
    wrap(voiding, "_project_fields", "accepted_field_transfer")
    wrap(adaptive, "refine_accepted_state", "mechanical_support_refinement",
         lambda pair: {"mechanics_nodes": pair[0].mesh.nn,
                       "mechanics_elements": pair[0].mesh.ne})
    # hybrid imported this symbol directly; patch the exact call site.
    wrap(hybrid, "equilibrate_fixed_load_with_production_fem", "mechanical_equilibrium",
         lambda state: {"mechanics_nodes": state.mesh.nn,
                        "mechanics_elements": state.mesh.ne})
    wrap(hybrid, "initialize_mechanically_separating_v12", "support_certificate")
    wrap(hybrid, "apply_mechanically_separating_v12_trial_geometry",
         "extended_support_and_geometry")
    wrap(transport, "build_exact_cut_transport_mesh_for_network",
         "cut_transport_mesh_construction",
         lambda cut: {"transport_nodes": cut.mesh.nn,
                      "transport_elements": cut.mesh.ne})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--restart", type=Path, default=DEFAULT_RESTART)
    parser.add_argument("--label")
    args = parser.parse_args()
    state = restore_checkpoint(args.restart)
    before = complete_accepted_state_fingerprint(state)
    records = _instrument()
    started = time.perf_counter()
    provider = sharp_front_load_provider(
        state, branch_id=state.crack_network.active_tip_ids[0],
        candidates=state.competition.candidates,
        evaluate_overlap_marginals=True,
        marginal_delta_a_m=(KINETIC_DELTA_A,), marginal_mesh_levels=(4, 5),
        finite_increment_mode=True,
        virtual_mechanical_trial_transport=(
            "full" if args.mode == "baseline" else "deferred_read_only"
        ),
        mechanical_refinement_policy=(
            "legacy_accumulated_generations"
            if args.mode == "baseline" else "physical_candidate_support"
        ),
    )
    elapsed = time.perf_counter() - started
    after = complete_accepted_state_fingerprint(state)
    summary = {}
    for category, rows in records.items():
        summary[category] = {
            "calls": len(rows),
            "inclusive_wall_time_s": sum(row["wall_time_s"] for row in rows),
            "maximum_wall_time_s": max(row["wall_time_s"] for row in rows),
            "last_sizes": {key: value for key, value in rows[-1].items()
                           if key != "wall_time_s"},
        }
    result = _finite({
        "schema": "informative.off-axis-canonical-profile/1",
        "mode": args.mode,
        "restart": str(args.restart.resolve().relative_to(ROOT)),
        "accepted_state_preserved": before == after,
        "accepted_mechanics_nodes": state.mesh.nn,
        "accepted_mechanics_elements": state.mesh.ne,
        "accepted_transport_nodes": state.vacancy_transport_state_v2.transport_mesh.nn,
        "accepted_transport_elements": state.vacancy_transport_state_v2.transport_mesh.ne,
        "total_wall_time_s": elapsed,
        "stage_timings_inclusive": summary,
        "stage_calls": records,
        "directional_rows": provider["directional"],
    })
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.label or args.mode}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "mode": args.mode, "wall_time_s": elapsed,
        "accepted_state_preserved": before == after,
        "output": str(path.relative_to(ROOT)), "stage_summary": summary,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
