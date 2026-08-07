#!/usr/bin/env python3
"""Reconstruct current and physical-scale v11 cluster handoff histories."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    selection = json.loads((run / "v10_2_22_parameter_selection.json").read_text())
    physical = float(selection["mpz_length_um"]) * 1.0e-6
    source_zone = float(selection["persistent_site_config"]["source_zone_length_m"])
    audit = json.loads((run / "v11_branching_model_audit.json").read_text())
    tokens = audit["argv"]
    def option(name, default):
        return float(tokens[tokens.index(name) + 1]) if name in tokens else float(default)
    requested_tip_h = option("--tip-h-fine", 1.0e-6)
    da = option("--da-phys", 5.0e-6)
    legacy_handoff = option("--L-pz", 1.0e-6)
    family = next((run.parent / "v11_direct_kernel_cache_causal").glob("*/mechanical_configuration.json"))
    mechanical = json.loads(family.read_text())
    interaction = float(mechanical["interaction_length_m"])

    fronts = {}
    for row in csv.DictReader((run / "fronts.csv").open()):
        fronts.setdefault(int(row["step"]), {})[row["front_id"]] = row
    rates = {}
    for row in _jsonl(run / "directional_rates.jsonl"):
        rates.setdefault(int(row["step"]), {}).setdefault(row["tip_id"], []).append(row)
    adaptations = _jsonl(run / "mesh_adaptations.jsonl")
    hbar_at = {}
    latest = requested_tip_h
    by_step = {int(row["step"]): row for row in adaptations}
    for step in range(max(fronts) + 1):
        if step in by_step:
            latest = float(by_step[step]["maximum_active_tip_hbar_m"])
        hbar_at[step] = latest

    definitions = (
        ("first", "b3f7dae78837e9ff", "b83112830cb89cf4", 60.0e-6, 295, 466),
        ("second", "b584969b46da50bd", "b65ee7fea748d53a", 215.0e-6, 466, max(fronts)),
    )
    histories = {}
    for label, arm1, arm2, junction_extension, start, stop in definitions:
        rows = []
        for step in range(start, stop + 1):
            if arm1 not in fronts.get(step, {}) or arm2 not in fronts[step]:
                continue
            first, second = fronts[step][arm1], fronts[step][arm2]
            p1 = (float(first["tip_x_m"]), float(first["tip_y_m"]))
            p2 = (float(second["tip_x_m"]), float(second["tip_y_m"]))
            lengths = (float(first["arclength_m"]), float(second["arclength_m"]))
            validity = tuple(
                bool(rates.get(step, {}).get(tip)) and
                all(bool(item["local_J_valid"]) for item in rates[step][tip])
                for tip in (arm1, arm2)
            )
            separation = math.dist(p1, p2)
            # The selected nested contour is no larger than the provider's
            # generated maximum radius max(request, 3*hbar).  Using that upper
            # bound makes the replay overlap test conservative.
            contour_upper = max(legacy_handoff, 3.0 * hbar_at[step])
            physical_ready = (
                min(lengths) >= physical and separation >= physical and
                separation >= 2.0 * contour_upper and all(validity)
            )
            rows.append({
                "step": step,
                "physical_extension_m": junction_extension + max(lengths),
                "arm_lengths_from_junction_m": list(lengths),
                "tip_separation_m": separation,
                "physical_process_zone_length_m": physical,
                "legacy_handoff_length_m": legacy_handoff,
                "selected_J_contour_radius_upper_bound_m": contour_upper,
                "actual_local_hbar_m": hbar_at[step],
                "independently_valid_local_J": list(validity),
                "contour_overlap_using_upper_bound": separation < 2.0 * contour_upper,
                "corrected_physical_scale_handoff_result": physical_ready,
            })
        histories[label] = rows

    first_ready = next((row for row in histories["first"] if row["corrected_physical_scale_handoff_result"]), None)
    step500 = next(row for row in histories["second"] if row["step"] == 500)
    payload = {
        "schema": "v11.branch-handoff-scale-audit/1",
        "source_run": str(run),
        "scale_trace": {
            "physical_process_zone_length_m": physical,
            "physical_process_zone_source": "v10_2_22_parameter_selection.json:mpz_length_um",
            "legacy_branch_handoff_length_m": legacy_handoff,
            "legacy_branch_handoff_source": "sharp_front parser --L-pz default passed directly to guard",
            "corrected_branch_handoff_length_m": physical,
            "corrected_branch_handoff_source": "mechanical configuration process_zone_length_m",
            "source_zone_length_m": source_zone,
            "source_zone_source": "persistent_site_config.source_zone_length_m",
            "requested_J_contour_radius_m": legacy_handoff,
            "interaction_integral_length_m": interaction,
            "tip_h_fine_m": requested_tip_h,
            "event_length_da_phys_m": da,
        },
        "first_cluster": {
            "current_resolution_step": 381,
            "current_resolution_extension_m": 105.0e-6,
            "corrected_accepted_geometry_replay_first_ready": first_ready,
            "causal_warning": "states after current step-381 handoff evolved with independent engines; replay timing is diagnostic, not a valid corrected trajectory",
            "history": histories["first"],
        },
        "second_cluster": {
            "former_step_500": step500,
            "history": histories["second"],
        },
        "ownership_history_changed": True,
        "source_305um_checkpoint_production_valid": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "first_replay_ready_step": None if first_ready is None else first_ready["step"],
        "first_replay_ready_extension_um": None if first_ready is None else first_ready["physical_extension_m"] * 1e6,
        "step500_corrected_ready": step500["corrected_physical_scale_handoff_result"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
