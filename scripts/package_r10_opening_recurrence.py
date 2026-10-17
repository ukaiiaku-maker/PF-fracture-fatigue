#!/usr/bin/env python3
"""Package the R10 renewal/feedback continuation without rewriting prior evidence."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/r10_opening_recurrence"
OUT = ROOT / "integration_evidence/r10_opening_recurrence_continuation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    raw = {}
    for path in sorted(RUNS.glob("*/result.json")):
        data = json.loads(path.read_text())
        events = data["events"]
        cycles = data["cycle_history"]
        row = {
            "run": path.parent.name,
            "candidate_id": data["candidate_id"],
            "shape": data["shape"],
            "mode": data["mode"],
            "temperature_K": data["temperature_K"],
            "Kmax_MPa_sqrt_m": data["Kmax_MPa_sqrt_m"],
            "cycles_completed": len(cycles),
            "physical_exposure_s": cycles[-1]["physical_time_s"] if cycles else 0.,
            "event_count": len(events),
            "extension_m": cycles[-1]["extension_m"] if cycles else 0.,
            "first_event_cycle": events[0]["cycle"] if events else None,
            "last_event_cycle": events[-1]["cycle"] if events else None,
            "stop_category": data["stop_category"],
            "unsupported_crossover": data["unsupported_crossover_encountered"],
            "final_state_relative_change": (cycles[-1]["cycle_state_relative_change"]
                                             if cycles else None),
            "final_mobile_count": cycles[-1]["mobile_count"] if cycles else None,
            "final_retained_count": cycles[-1]["retained_count"] if cycles else None,
            "final_tip_radius_m": cycles[-1]["tip_radius_m"] if cycles else None,
            "final_K_shield_Pa_sqrt_m": cycles[-1]["K_shield_Pa_sqrt_m"] if cycles else None,
            "same_complete_material_sha256": data["same_complete_material_sha256"],
            "paired_plasticity_sha256": data["paired_plasticity_sha256"],
        }
        results.append(row)
        raw[path.parent.name] = data
    fields = list(results[0])
    with (OUT / "native_case_dispositions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    with gzip.open(OUT / "native_histories.json.gz", "wt") as stream:
        json.dump(raw, stream, sort_keys=True, separators=(",", ":"))

    feedback_source = RUNS / "feedback_2d_pl31630/feedback_qualification.json"
    state_source = RUNS / "feedback_2d_pl31630/active_states.npz"
    shutil.copy2(feedback_source, OUT / "finite_state_2d_feedback.json")
    shutil.copy2(state_source, OUT / "finite_state_2d_active_states.npz")
    shutil.copy2(RUNS / "crossed_registry.csv", OUT / "crossed_material_registry.csv")
    feedback = json.loads(feedback_source.read_text())

    prior_counts = {}
    prior_reclassified = []
    for name in ("native_12_case_dispositions.csv", "developed_followup_case_dispositions.csv"):
        source = ROOT / "integration_evidence/r10_opening_alternatives" / name
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                events = int(row["event_count"])
                cycles = int(row["cycles"])
                if row["status"] == "TARGET_COMPLETED" and cycles <= 1:
                    key = "TARGET_TRUNCATED_FIRST_EXCURSION_CASCADE"
                elif events > 0:
                    key = "FINITE_BURST_AT_DECLARED_EXPOSURE_ENDPOINT"
                else:
                    key = "DECLARED_EXPOSURE_NONADVANCE"
                prior_counts[key] = prior_counts.get(key, 0) + 1
                prior_reclassified.append({
                    "source_file": name, "candidate_id": row["candidate_id"],
                    "shape": row["shape"], "mode": row["mode"],
                    "Kmax_MPa_sqrt_m": row["Kmax_MPa_sqrt_m"],
                    "status": row["status"], "cycles": cycles,
                    "event_count": events,
                    "old_rate_disposition": row["rate_disposition"],
                    "old_developed_rate_m_per_cycle": row["developed_rate_m_per_cycle"],
                    "reconciled_stop_category": key,
                    "recurrent_growth_supported": False,
                })
    with (OUT / "prior_168_stop_reconciliation.csv").open("w", newline="") as stream:
        fields = list(prior_reclassified[0])
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(prior_reclassified)

    by_run = {row["run"]: row for row in results}
    summary = {
        "schema": "r10.opening-renewal-feedback-continuation/1",
        "source_parent_commit": "6a6dcf586fdcc74d2968f86652e65914d9b9317c",
        "branch": "codex/r10-athermal-mixed-czm-comparison",
        "prior_evidence_preserved": True,
        "prior_disposition_counts": prior_counts,
        "prior_saved_payload_restart_scope": (
            "Prior exported JSON lacks complete active arrays and RNG state; exact restart is unavailable. "
            "Deterministic same-material replay was used and new active-state vectors were saved bit-exactly."),
        "structural_process_zone_boundary_m": 50e-6,
        "initial_comparison": {
            "smooth": by_run["smooth_adv"],
            "bilinear": by_run["bilinear_adv"],
            "smooth_neighbor": by_run["smooth_neighbor"],
            "mixed_smooth": by_run["mixed_adv"],
            "conclusion": "smooth: one-cell burst then arrest; bilinear: first-excursion structural-boundary cascade; no recurrent growth",
        },
        "base_smooth_trigger_bracket_MPa_sqrt_m": [27.875, 28.0],
        "active_crossed_material": {
            "athermal_30": by_run["active30_a"],
            "mixed_30": by_run["active30_m"],
            "mixed_30_long": by_run["active30_m_long"],
            "classification": "FINITE_FIRST_EXCURSION_BURST; NO_REACTIVATION_THROUGH_128_CYCLES",
            "paris_rate_available": False,
        },
        "refinement": {
            "phase_base_extension_m": by_run["active30_m"]["extension_m"],
            "phase_refined_extension_m": by_run["active30_m_phase_refine"]["extension_m"],
            "cell_0p5um_extension_m": by_run["active30_m_cell_0p5"]["extension_m"],
            "cell_1um_extension_m": by_run["active30_m"]["extension_m"],
            "cell_2um_extension_m": by_run["active30_m_cell_2"]["extension_m"],
            "disposition": "PHASE_STABLE; CELL_SIZE_SENSITIVE_BOUNDED_FIRST_EXCURSION",
        },
        "mixed_crossover": {
            "implemented": "COMMON_COORDINATE_OVERDAMPED_MFPT",
            "friction_owner": "activation_area*initial_stiffness/declared_attempt_frequency",
            "adiabatic_gate": "harmonic_relaxation/phase_duration <= 1e-3",
            "unsupported_paths_remain_explicit": True,
            "demonstrated_case": "cross_coh00167_mixed",
        },
        "finite_state_2d": {
            "state_restore_bit_exact": feedback["state_restore_bit_exact"],
            "closure_magnitude_Pa": feedback["closure_magnitude_Pa"],
            "cases": [{"case": case["case"], "trigger": case["trigger"],
                       "bracket_delta0": case["bracket_delta0"]}
                      for case in feedback["cases"]],
            "old_evidence_disposition": feedback["old_2d_evidence_disposition"],
        },
        "objective": {
            "developed_recurrent_growth_found": False,
            "paris_slope_passes": 0,
            "next_acquisition_basis": "crossed cohesion/plasticity and measured opening/plastic exposure",
        },
    }
    (OUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    results_md = """# R10 opening renewal and finite-state feedback continuation

## Decision

The completed initial bank remains a zero-Paris-pass result. No trajectory in this continuation establishes recurrent cohesive fatigue, so no local Paris slope or developed rate is reported.

The retained 27231 smooth law has a 300 K athermal first-event bracket of **27.875–28.000 MPa√m**. At 28.770 MPa√m it advances one 1 µm cell and then reaches cyclic shakedown; the bilinear law instead reaches the geometry-owned 50 µm process-zone boundary in the first excursion. Those outcomes are an arrested increment and a target-truncated cascade, respectively—not recurrent growth.

Crossing the same 27231 cohesion with the active 31630 EXP_floor plasticity suppresses athermal advance through 35 MPa√m over the executed 48-cycle exposure. At 30 MPa√m the mixed law produces a three-cell common-coordinate first-passage burst during the first excursion, then no later event through 128 cycles. Phase refinement preserves 3 µm. Cell sizes 0.5, 1, and 2 µm give 2.5, 3, and 4 µm, so the event count is not a mesh-independent material rate and the result remains a bounded finite burst.

The low-barrier mixed crossover now uses an overdamped mean-first-passage calculation on the same finite-compliance cohesive coordinate. Its friction scale is fixed by the existing activation area, initial stiffness, and declared attempt frequency; it is accepted only when harmonic relaxation is at most 10^-3 of the opening phase. Unsupported nonadiabatic cases remain explicit.

The prior 2-D evidence evolved EXP_floor state but did not consume it in equilibrium. The corrected comparison restores the exact same saved active state in both arms. The retained A0-to-current mechanics gives 3.148 GPa of closing stress. Consuming it raises the smooth-interface FE branch-loss bracket from **1833.027 δ0** to **2455.044 δ0**; disabling feedback while holding state and geometry fixed recovers the lower bracket. This is a finite-state mechanical effect, not a Paris result.

## Stop semantics

- Physical non-advance, supported cyclic shakedown, unsupported first passage, finite first-excursion burst, structural-boundary cascade, numerical failure, and resource stop remain distinct.
- The 50 µm retained process-zone length is the structural boundary; the older 60 µm follow-up target is not reused.
- Prior exports did not contain complete active arrays plus RNG state, so they cannot support an exact checkpoint restart. They were deterministically replayed with identical material inputs; this continuation now exports the full active vectors and verifies bit-exact restore.
- No empirical cycle damage, per-cell delay, source suppression, opening-family substitution, or numerical advance cap was added.

## Verification

Focused cohesive/opening tests pass. The earlier full-suite run remains honestly recorded as interrupted after 39 passes; this continuation does not relabel it as a full pass.
"""
    write(OUT / "RESULTS.md", results_md)
    write(OUT / "README.md", "# Contents\n\nSee `RESULTS.md` for the decision, `results_summary.json` for machine-readable findings, `prior_168_stop_reconciliation.csv` for the closed initial/follow-up reclassification, `native_case_dispositions.csv` and `native_histories.json.gz` for new native evidence, and the finite-state 2-D files for the spatial comparison.\n")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.name != "SHA256SUMS.json" and path.is_file():
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "cases": len(results), "files": len(hashes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
