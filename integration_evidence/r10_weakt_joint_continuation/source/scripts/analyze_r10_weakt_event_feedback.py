#!/usr/bin/env python3
"""Extract event-conditioned Weak-T feedback and exact extension windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def endpoint(events: list[dict], threshold_m: float) -> dict:
    if threshold_m == 0.0:
        return {"event_index": 0, "cycles": 0.0, "extension_m": 0.0}
    return next(x for x in events if float(x["extension_m"]) >= threshold_m)


def window(events: list[dict], low_um: float, high_um: float) -> dict:
    left = endpoint(events, low_um * 1.0e-6)
    right = endpoint(events, high_um * 1.0e-6)
    dc = float(right["cycles"]) - float(left["cycles"])
    da = float(right["extension_m"]) - float(left["extension_m"])
    return {
        "requested_low_um": low_um, "requested_high_um": high_um,
        "left_event_index": left["event_index"],
        "right_event_index": right["event_index"],
        "left_fractional_cycles": float(left["cycles"]),
        "right_fractional_cycles": float(right["cycles"]),
        "left_actual_extension_um": 1.0e6 * float(left["extension_m"]),
        "right_actual_extension_um": 1.0e6 * float(right["extension_m"]),
        "events_in_endpoint_difference": int(right["event_index"] - left["event_index"]),
        "rate_m_per_cycle": da / dc,
        "endpoint_rule": "first accepted event with cumulative extension >= requested endpoint",
    }


def history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fatigue-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    event_rows, window_rows = [], []
    for result_path in sorted(args.fatigue_root.glob("Kmax_*/result.json")):
        result = json.loads(result_path.read_text())
        events = result["events"]
        logs = history(result_path.parent / "checkpoint" / "high_cycle_live_history.jsonl")
        accepted = [x for x in logs if x.get("reason") == "accepted_event"]
        if len(accepted) != len(events):
            raise RuntimeError(f"event/history mismatch for {result_path}")
        delta_k = float(result["DeltaK_MPa_sqrt_m"])
        for bounds, label in (((0.0, 20.0), "early_0_20"),
                              ((20.0, 60.0), "developed_20_60"),
                              ((60.0, 80.0), "late_60_80")):
            try:
                row = window(events, *bounds)
            except StopIteration:
                continue
            window_rows.append({"DeltaK_MPa_sqrt_m": delta_k, "window": label, **row})
        previous_gross = previous_return = 0.0
        for event, log in zip(events, accepted):
            diag = log["diagnostics"]; ledgers = log["ledgers"]
            gross = float(ledgers["mpz.cumulative_gross_source_activity"])
            returned = float(ledgers["mpz.cumulative_gross_return_activity"])
            mobile_signed = sum(float(v) for k, v in ledgers.items()
                                if "cumulative_returned_mobile" in k)
            physical_return = sum(float(v) for k, v in ledgers.items()
                                  if "cumulative_physical_returned_mobile" in k)
            wake_transfer = sum(float(v) for k, v in ledgers.items()
                                if "cumulative_source_slip_wake_transfer" in k)
            event_rows.append({
                "DeltaK_MPa_sqrt_m": delta_k,
                "event_index": event["event_index"],
                "fractional_cycles": event["cycles"],
                "extension_um": 1.0e6 * float(event["extension_m"]),
                "phase": event["phase"],
                "opening_stress_Pa": diag["sigma_tip_Pa"],
                "opening_rate_s-1": diag["lambda_cleave_s"],
                "effective_radius_m": diag["tip_radius_m"],
                "source_multiplicity_per_system": event["persistent_site_multiplicity_per_system"],
                "source_backstress_Pa": diag["sigma_back_Pa"],
                "mobile_count": diag["mobile_count"],
                "retained_count": diag["retained_count"],
                "gross_source_slip": gross,
                "gross_returned_slip": returned,
                "net_source_slip": gross - returned,
                "interval_gross_source_slip": gross - previous_gross,
                "interval_returned_slip": returned - previous_return,
                "K_shield_Pa_sqrt_m": diag["active_K_shield_Pa_sqrt_m"],
                "cumulative_returned_mobile": mobile_signed,
                "cumulative_physical_returned_mobile": physical_return,
                "post_advance_wake_transfer": wake_transfer,
                "continuous_mpz_progress_m": event["continuous_mpz_progress_m"],
                "remaining_waveform_continued": event["remaining_waveform_continued"],
            })
            previous_gross, previous_return = gross, returned

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    write(args.out_root / "event_conditioned_feedback.csv", event_rows)
    write(args.out_root / "extension_window_rates.csv", window_rows)

    # Condense the two intervals requested by the amendment.  Ratios are
    # descriptive sensitivities between complete trajectories, not ablations.
    summaries = []
    for low, high in ((7.0, 10.0), (15.0, 22.0)):
        pair = []
        for target in (low, high):
            rows = [x for x in event_rows if math.isclose(x["DeltaK_MPa_sqrt_m"], target,
                                                          rel_tol=0.0, abs_tol=1e-5)
                    and 20.0 <= float(x["extension_um"]) <= 60.0]
            pair.append({key: sum(float(x[key]) for x in rows) / len(rows)
                         for key in ("opening_stress_Pa", "opening_rate_s-1",
                                     "effective_radius_m", "source_backstress_Pa",
                                     "mobile_count", "retained_count",
                                     "interval_gross_source_slip", "K_shield_Pa_sqrt_m")})
        summaries.append({
            "interval_DeltaK": [low, high], "low_mean": pair[0], "high_mean": pair[1],
            "high_over_low": {key: pair[1][key] / pair[0][key]
                              if pair[0][key] != 0.0 else None for key in pair[0]},
        })
    result = {
        "schema": "r10.weakt-event-feedback/1",
        "candidate_id": "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1",
        "window_endpoint_rule": "first accepted event at or above each requested extension",
        "amplitude_interval_summaries": summaries,
        "interpretation": [
            "opening rate is the same-row exact complete-state diagnostic after each accepted advance",
            "event marks and rewards remain paired in the source result histories",
            "complete-trajectory contrasts are sensitivities, not channel ablations",
            "fatigue low_3/high_3 are amplitude subsets and are not temporal windows",
        ],
    }
    (args.out_root / "event_feedback_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
