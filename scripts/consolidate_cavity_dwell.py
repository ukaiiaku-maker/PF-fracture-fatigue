#!/usr/bin/env python3
"""Consolidate the immutable suffixes of the declared 10 us cavity dwell."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = (
    "integration_evidence/active_fracture_comparisons_0a5d55e/"
    "cavity_post_ramp_dwell_10us_run4/shared_v5_cavity_audit.json",
    "integration_evidence/final_fracture_campaign/"
    "cavity_picard_diagnostic_run2/shared_v5_cavity_audit.json",
    "integration_evidence/active_fracture_comparisons_0a5d55e/"
    "cavity_post_ramp_dwell_10us_completion_run1/shared_v5_cavity_audit.json",
    "integration_evidence/final_fracture_campaign/"
    "cavity_picard_diagnostic_run4/shared_v5_cavity_audit.json",
    "integration_evidence/active_fracture_comparisons_0a5d55e/"
    "cavity_post_ramp_dwell_10us_completion_run2/shared_v5_cavity_audit.json",
)


def _delta(first: dict, last: dict, name: str):
    left, right = first.get(name), last.get(name)
    return None if left is None or right is None else float(right) - float(left)


def consolidate(root: Path = ROOT) -> dict:
    paths = [root / item for item in SEGMENTS]
    audits = [json.loads(path.read_text()) for path in paths]
    chain = []
    for index, (path, audit) in enumerate(zip(paths, audits)):
        handoff = True
        if index:
            handoff = all(
                audits[index - 1]["arms"][arm]["terminal_fingerprint"]
                == audit["arms"][arm]["history"][0]["input_fingerprint"]
                for arm in ("disabled", "enabled")
            )
        chain.append({
            "path": str(path.relative_to(root)),
            "status": audit["status"],
            "physical_time_start_s": (
                float(audit["physical_time_s"])
                - float(audit["physical_exposure_s_per_arm_this_call"])
            ),
            "physical_time_end_s": float(audit["physical_time_s"]),
            "accepted_exposure_s": float(
                audit["physical_exposure_s_per_arm_this_call"]
            ),
            "native_checkpoint_round_trip_exact": bool(
                audit["native_checkpoint_round_trip_exact"]
            ),
            "prior_terminal_to_input_fingerprint_exact": handoff,
        })
    first = {
        arm: audits[0]["arms"][arm]["history"][0]["observables"]
        for arm in ("disabled", "enabled")
    }
    last = {
        arm: audits[-1]["arms"][arm]["terminal"]
        for arm in ("disabled", "enabled")
    }
    exposure = sum(row["accepted_exposure_s"] for row in chain)
    target = 1.0e-5
    observables = {}
    for arm in ("disabled", "enabled"):
        observables[arm] = {
            "opening_start_m": first[arm]["applied_opening_m"],
            "opening_end_m": last[arm]["applied_opening_m"],
            "reaction_start_N_per_m": first[arm]["reaction_N_per_m"],
            "reaction_end_N_per_m": last[arm]["reaction_N_per_m"],
            "reaction_change_N_per_m": _delta(
                first[arm], last[arm], "reaction_N_per_m"
            ),
            "energy_change_J_per_m": _delta(
                first[arm], last[arm], "energy_J_per_m"
            ),
            "graph_length_change_m": _delta(
                first[arm], last[arm], "graph_length_m"
            ),
            "cavity_area_change_m2": _delta(
                first[arm], last[arm], "cavity_area_m2"
            ),
            "cavity_radius_change_m": _delta(
                first[arm], last[arm], "cavity_radius_m"
            ),
            "vacancy_inventory_change_m2": _delta(
                first[arm], last[arm], "vacancy_field_inventory_area_m2"
            ),
            "void_event_count_start": len(first[arm]["void_event_history"]),
            "void_event_count_end": len(last[arm]["void_event_history"]),
            "terminal_site_phase": last[arm]["site_phase"],
            "terminal_void_phase": last[arm]["void_phase"],
            "terminal_crack_void_connection": last[arm][
                "crack_void_connection_certificate"
            ],
        }
    return {
        "schema": "final-fracture-cavity-dwell-consolidation/1",
        "protocol": "original_declared_post_ramp_fixed_opening_dwell",
        "material_class": "DBTT",
        "temperature_K": 1200.0,
        "seed": 3621,
        "opening_m": 6.3e-6,
        "target_exposure_s": target,
        "accepted_exposure_s": exposure,
        "exposure_closure_error_s": exposure - target,
        "exposure_complete_within_binary_time_tolerance": math.isclose(
            exposure, target, rel_tol=0.0, abs_tol=2.0e-13
        ),
        "physical_time_start_s": chain[0]["physical_time_start_s"],
        "physical_time_end_s": chain[-1]["physical_time_end_s"],
        "all_checkpoint_round_trips_exact": all(
            row["native_checkpoint_round_trip_exact"] for row in chain
        ),
        "all_suffix_handoffs_exact": all(
            row["prior_terminal_to_input_fingerprint_exact"] for row in chain
        ),
        "terminal_status": audits[-1]["status"],
        "terminal_unused_requested_duration_s": audits[-1][
            "unused_requested_duration_s"
        ],
        "terminal_transport_rejection": audits[-1]["transport_rejection"],
        "segment_chain": chain,
        "observables": observables,
        "paired_terminal_differences": {
            "enabled_minus_disabled_reaction_N_per_m": (
                last["enabled"]["reaction_N_per_m"]
                - last["disabled"]["reaction_N_per_m"]
            ),
            "enabled_minus_disabled_energy_J_per_m": (
                last["enabled"]["energy_J_per_m"]
                - last["disabled"]["energy_J_per_m"]
            ),
        },
        "interpretation": (
            "The migration dwell now exercises accepted coupled evolution through "
            "the repaired recovery and transport path. At fixed opening it produces "
            "no new crack or topology event and only sub-femtometre cavity-radius "
            "growth. It is a weak-response finite exposure, not the required "
            "interaction-relevant loading continuation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = consolidate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
