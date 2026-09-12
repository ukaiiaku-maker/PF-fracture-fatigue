#!/usr/bin/env python3
"""Run one prospective V3 controlled history with complete bounded provenance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.closure_lifecycle_evidence import CFG, conservation, load_state, stagewise_topology
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.controlled_history_v3 import (
    DIFFUSION_OPENING_M, EXPECTED, dormant_interval_audit, growth_limiter_audit,
    mirrored_source_audit,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, _qualified_cavity_source, cavity_boundary_tensor,
    cavity_source_resolution_metrics, deterministic_trajectory, directional_clock_rates,
    downstream_front_transaction, refine_downstream_source,
)
from arrhenius_fracture.voiding_v5 import VoidPhase, arrhenius_rates, update_cavity_growth

CASES = (
    "accommodation_limited", "diffusion_limited", "delayed_downstream",
    "downstream_zero_drive", "local_remesh_refinement", "negative_offset",
)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(canonical_data(value), sort_keys=True, indent=2, allow_nan=False) + "\n")


def _source_tensor(state):
    cavity = state.void_state.cavities[0]
    node = int(np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - np.asarray(cavity.connection_exit_m), axis=1)))
    return cavity_boundary_tensor(state, boundary_node=node)


def _prepare_connected(center, *, segments, layers):
    trace = []
    path = ((0.0, 0.0), (0.0005725993004046688, 0.0))
    state, _ = deterministic_trajectory(
        stop_before_ligament=True, cavity_center_m=center, crack_path_m=path,
        state_trace=trace, boundary_segments=segments, radial_layers=layers,
    )
    from arrhenius_fracture.closure_lifecycle_evidence import advance_transition
    operations = []
    connected, _ = advance_transition(state, "ligament", 1, operations=operations)
    return state, connected, operations, trace


def _qualify(state, operations, *, levels=1):
    trial, audit = refine_downstream_source(
        state, max_refinement_levels=levels, refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    operations.append({"api": "refine_downstream_source", "audit": audit})
    tensor, _ = _source_tensor(trial)
    qualified = _qualified_cavity_source(trial, tensor)
    return trial, audit, qualified


def _growth_case(case, *, segments, layers):
    trace = []
    deterministic_trajectory(
        stop_before_ligament=True, state_trace=trace, boundary_segments=segments,
        radial_layers=layers, crack_path_m=((0.0, 0.0), (0.0005725993004046688, 0.0)),
    )
    before = dict(trace)["subgrid_void"]
    opening = DIFFUSION_OPENING_M if case == "diffusion_limited" else 1.0e-7
    state = load_state(before, opening)
    operations = []
    for _ in range(8):
        from arrhenius_fracture.voiding_production_v5 import local_site_tensor
        rates = arrhenius_rates(CFG, temperature_K=900.0, stress_tensor_Pa=local_site_tensor(state))
        voids = update_cavity_growth(
            state.void_state, state.void_state.cavities[0].cavity_id, rates=rates,
            dt_s=1.0e-10, radial_growth_scale_m=CFG.radial_growth_scale_m,
        )
        state = load_state(replace(state, void_state=voids), opening)
        operations.append({"api": "accepted_load_growth_interval", "duration_s": 1.0e-10,
                           "opening_m": opening, "rates": rates})
    target = "vacancy_transport_s" if case == "diffusion_limited" else "plastic_accommodation_s"
    audit = growth_limiter_audit(operations, target_channel=target)
    classification = audit["expected_terminal_classification"] if audit["passed"] else "RATE_REGIME_NOT_ESTABLISHED"
    passed = audit["passed"] and state.void_state.cavities[0].phase == VoidPhase.STABLE_SUBGRID_VOID
    return before, state, operations, audit, classification, passed, {}


def _source_case(case, *, segments, layers):
    center = (0.0007, -1.0e-5) if case == "negative_offset" else (0.0007, 0.0)
    before, state, operations, trace = _prepare_connected(center, segments=segments, layers=layers)
    extra = {"preparation_stages": [name for name, _ in trace]}
    if case == "local_remesh_refinement":
        coarse = cavity_source_resolution_metrics(state)
        tensor, _ = _source_tensor(state)
        coarse_qualified = _qualified_cavity_source(state, tensor)
        state, audit, qualified = _qualify(state, operations, levels=3)
        extra.update({"coarse_source_metrics": coarse, "coarse_source_qualified": coarse_qualified,
                      "refinement_audit": audit, "refined_source_qualified": qualified})
        if qualified:
            state, result, transaction, causal = downstream_front_transaction(state)
            operations.append({"api": "downstream_front_transaction", "operations": transaction,
                               "causal": causal, "accepted": result is not None})
        passed = qualified and state.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE
        classification = (EXPECTED[case] if passed else "REFINED_SOURCE_REMAINS_UNQUALIFIED")
        return before, state, operations, audit, classification, passed, extra

    if case == "negative_offset":
        _, positive, positive_ops, _ = _prepare_connected((0.0007, 1.0e-5), segments=segments, layers=layers)
        mirror = mirrored_source_audit(cavity_source_resolution_metrics(positive),
                                       cavity_source_resolution_metrics(state))
        state, audit, qualified = _qualify(state, operations, levels=1)
        extra.update({"positive_operation_trace": positive_ops, "mirror_source_audit": mirror,
                      "source_refinement_audit": audit, "source_qualified": qualified})
        tensor, _ = _source_tensor(state)
        rates = directional_clock_rates(state, tensor)
        zero = all(row["effective_rate_s"] == 0.0 for row in rates)
        passed = mirror["passed"] and qualified and zero
        classification = EXPECTED[case] if passed else "MIRRORED_SOURCE_QUALIFICATION_FAILED"
        return before, state, operations, mirror, classification, passed, extra

    state, source_audit, qualified = _qualify(state, operations, levels=1)
    extra.update({"initial_source_refinement_audit": source_audit,
                  "initial_source_qualified": qualified})
    if not qualified:
        return before, state, operations, source_audit, "CONNECTED_VOID_SOURCE_UNQUALIFIED", False, extra

    dormant_loaded = load_state(state, -4.0e-7)
    tensor, elements = _source_tensor(dormant_loaded)
    interval_start = dormant_loaded
    dormant_end, directional = _complete_next_clock(
        dormant_loaded, tensor, source_kind="cavity_surface",
        source_cavity_id=dormant_loaded.void_state.cavities[0].cavity_id,
        source_boundary_site_id="connection_exit",
        source_position_m=dormant_loaded.void_state.cavities[0].connection_exit_m,
        source_probe_identity={"kind": "direct_cavity_boundary_tensor", "element_ids": list(elements)},
        maximum_advance_duration_s=16.0e-6,
    )
    dormancy = dormant_interval_audit(interval_start, dormant_end, directional)
    operations.append({"api": "qualified_zero_drive_interval", "duration_s": 16.0e-6,
                       "audit": directional, "invariance": dormancy})
    extra["dormant_interval_audit"] = dormancy
    if case == "downstream_zero_drive":
        passed = dormancy["passed"] and not dormant_end.crack_network.active_tip_ids
        return before, dormant_end, operations, dormancy, (EXPECTED[case] if passed else
            "QUALIFIED_ZERO_DRIVE_INVARIANCE_FAILED"), passed, extra

    reloaded = load_state(dormant_end, 8.0e-7)
    reloaded, reload_audit, reload_qualified = _qualify(reloaded, operations, levels=1)
    extra.update({"reload_source_refinement_audit": reload_audit,
                  "reload_source_qualified": reload_qualified})
    result = None
    if reload_qualified:
        reloaded, result, transaction, causal = downstream_front_transaction(reloaded)
        operations.append({"api": "downstream_front_transaction_after_reload", "operations": transaction,
                           "causal": causal, "accepted": result is not None})
    passed = bool(dormancy["passed"] and reload_qualified and result is not None
                  and reloaded.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE)
    classification = EXPECTED[case] if passed else "DORMANT_RELOAD_FIRST_PASSAGE_NOT_ESTABLISHED"
    return before, reloaded, operations, reload_audit, classification, passed, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--boundary-segments", type=int, default=512)
    parser.add_argument("--radial-layers", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"), cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError("controlled-history evidence requires committed tracked implementation")
    sha = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    error = None
    try:
        result = (_growth_case(args.case, segments=args.boundary_segments, layers=args.radial_layers)
                  if args.case in ("accommodation_limited", "diffusion_limited")
                  else _source_case(args.case, segments=args.boundary_segments, layers=args.radial_layers))
        before, after, operations, audit, classification, passed, extra = result
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
        before = after = None; operations = []; audit = {}; classification = "EXECUTION_FAILED"; passed = False; extra = {}
    payload = {
        "schema": "v5.one-void-controlled-history-execution/3",
        "case_identity": args.case,
        "executed_code_sha": sha,
        "input_configuration": {"boundary_segments": args.boundary_segments,
                                "radial_layers": args.radial_layers},
        "prospectively_frozen_expected_classification": EXPECTED[args.case],
        "final_classification": classification,
        "passed": bool(passed),
        "failure": error,
        "initial_fingerprint": None if before is None else fingerprint(before),
        "terminal_fingerprint": None if after is None else fingerprint(after),
        "actual_operation_trace": operations,
        "classification_audit": audit,
        "stagewise_topology": None if after is None else stagewise_topology(after),
        "conservation": None if before is None or after is None else conservation(after, before),
        **extra,
    }
    args.output.mkdir(parents=True)
    report = args.output / "report.json"
    _write(report, payload)
    _write(args.output / "sha256_manifest.json", {
        "report.json": hashlib.sha256(report.read_bytes()).hexdigest(),
    })
    print(json.dumps({"case": args.case, "classification": classification, "passed": bool(passed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
