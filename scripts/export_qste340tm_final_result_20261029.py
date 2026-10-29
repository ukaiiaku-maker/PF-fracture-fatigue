#!/usr/bin/env python3
"""Create paper-ready tables and a hash manifest for the final QSTE340TM result."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
OUT = BASE / "FINAL_RESULT_20261029"
SIGNED = BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028"
NO_BEST = BASE / "resolved_increment_1d/no_bulk_matched_refit_20261028"
NO_REPEAT = BASE / "resolved_increment_1d/no_bulk_equal_effort_control_20261028"
NATIVE = BASE / "native_signed_bulk_refit_20261028"
EXTENDED = BASE / "native_signed_bulk_refit_20261028_extended_7events"

SOURCE_FILES = (
    "arrhenius_fracture/incremental_opening_advection.py",
    "arrhenius_fracture/persistent_site_high_cycle_propagation_v10230.py",
    "arrhenius_fracture/persistent_site_high_cycle_state_v10230.py",
    "arrhenius_fracture/persistent_site_poincare_v10230.py",
    "arrhenius_fracture/persistent_site_reversible_transport_v10230.py",
    "arrhenius_fracture/signed_gnd_projection.py",
    "arrhenius_fracture/physical_fem_capture_v10212.py",
    "arrhenius_fracture/qste340tm_restored_binding.py",
    "arrhenius_fracture/reduced_persistent_reference_r2.py",
    "arrhenius_fracture/sharp_front.py",
    "arrhenius_fracture/unified_front.py",
    "arrhenius_fracture/unified_mpz.py",
    "scripts/qste340tm_emission_blunting_reduction.py",
    "scripts/qste340tm_signed_bulk_1d.py",
    "scripts/refit_qste340tm_emission_owned_blunting.py",
    "scripts/refit_qste340tm_resolved_sloping_barrier.py",
    "scripts/refit_qste340tm_restored_native.py",
    "scripts/refit_qste340tm_signed_bulk_1d.py",
    "scripts/refit_qste340tm_no_bulk_matched.py",
    "scripts/refit_qste340tm_no_bulk_1d_control.py",
    "scripts/run_qste340tm_main_result_full_2d.py",
    "scripts/run_qste340tm_signed_bulk_2d_matrix.py",
    "scripts/analyze_qste340tm_signed_bulk_2d_matrix.py",
    "scripts/analyze_qste340tm_signed_bulk_extended_events.py",
    "scripts/plot_qste340tm_audit_barrier_comparison.py",
    "scripts/plot_qste340tm_audit_plastic_barriers.py",
    "scripts/export_qste340tm_final_result_20261029.py",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    signed = read_csv(SIGNED / "predictions_signed_bulk.csv")
    no_best = read_csv(NO_BEST / "predictions.csv")
    no_repeat = read_csv(NO_REPEAT / "predictions_no_bulk_refit.csv")
    if not (len(signed) == len(no_best) == len(no_repeat)):
        raise RuntimeError("1-D prediction tables do not have matching lengths")
    rows = []
    state_fields = [name for name in signed[0] if name not in {
        "series_id", "source_order", "block_index", "R", "DeltaK_MPa_sqrt_m",
        "observed_dadn_m_per_cycle", "cycles", "crack_length_m",
        "history_alignment", "history_index_exact", "premeasurement_exposure_included",
        "predicted_dadn_m_per_cycle", "residual_log10",
    }]
    for a, b, c in zip(signed, no_best, no_repeat):
        identity = (a["series_id"], a["source_order"], a["R"], a["DeltaK_MPa_sqrt_m"])
        if identity != (b["series_id"], b["source_order"], b["R"], b["DeltaK_MPa_sqrt_m"]):
            raise RuntimeError("best no-bulk table is not aligned")
        if identity != (c["series_id"], c["source_order"], c["R"], c["DeltaK_MPa_sqrt_m"]):
            raise RuntimeError("repeat no-bulk table is not aligned")
        row = {
            "series_id": a["series_id"], "source_order": a["source_order"],
            "R": a["R"], "DeltaK_MPa_sqrt_m": a["DeltaK_MPa_sqrt_m"],
            "observed_dadn_m_per_cycle": a["observed_dadn_m_per_cycle"],
            "signed_bulk_dadn_m_per_cycle": a["predicted_dadn_m_per_cycle"],
            "no_bulk_best_dadn_m_per_cycle": b["predicted_dadn_m_per_cycle"],
            "no_bulk_repeat_dadn_m_per_cycle": c["predicted_dadn_m_per_cycle"],
        }
        row.update({f"signed_bulk_{name}": a[name] for name in state_fields})
        rows.append(row)
    with (OUT / "paper_1d_curves_and_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    extended = json.loads((EXTENDED / "EXTENDED_ANALYSIS.json").read_text())
    native_rows = []
    for case in extended["cases"]:
        native_rows.append({
            "R": case["R"], "DeltaK_MPa_sqrt_m": case["DeltaK"],
            "event_cycles": json.dumps(case["event_cycles"]),
            "interevent_waits_cycles": json.dumps(case["waits"]),
            "early_waits_1_2_rate_m_per_cycle": case["accepted_waits_1_2"],
            "late_waits_4_6_rate_m_per_cycle": case["late_waits_4_6"],
            "late_over_early_log10": case["late_over_accepted_log10"],
            "first_three_events_match_historical_stale_Lpz": case["first_three_events_match_accepted"],
        })
    with (OUT / "paper_native_2d_event_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(native_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(native_rows)

    barrier_bundle = {
        "opening": json.loads((NATIVE / "audit_opening_barrier_comparison.json").read_text()),
        "plastic": json.loads((NATIVE / "audit_plastic_barrier_comparison.json").read_text()),
    }
    (OUT / "paper_barrier_parameters.json").write_text(
        json.dumps(barrier_bundle, indent=2, sort_keys=True) + "\n"
    )

    roots = (SIGNED, NO_BEST, NO_REPEAT, NATIVE, EXTENDED)
    evidence = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                evidence.append(path)
    evidence.extend(ROOT / path for path in SOURCE_FILES)
    evidence.extend((
        ROOT / "literature_data/fatigue_recovered_review/normalized/qste340tm_constant_amplitude.csv",
        OUT / "paper_1d_curves_and_state.csv",
        OUT / "paper_native_2d_event_windows.csv",
        OUT / "paper_barrier_parameters.json",
    ))
    unique = sorted(set(evidence))
    manifest = {
        "schema": "qste340tm-final-result-manifest/1",
        "identity": "QSTE340TM_FINAL_SIGNED_BULK_AUDITED_20261029",
        "result_scope": "accepted 1-D fit plus audit-qualified all-16-case seven-event native comparison",
        "historical_three_event_result_is_not_current_baseline": True,
        "current_native_driver_L_pz_m": 1.0e-6,
        "accepted_1d_RMSE_log10": 0.29523348393431903,
        "current_native_early_RMSE_log10": extended["windows"]["accepted_waits_1_2"]["rmse_log10_vs_experiment"],
        "current_native_late_RMSE_log10": extended["windows"]["late_waits_4_6"]["rmse_log10_vs_experiment"],
        "files": [{"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
                   "sha256": sha256(path)} for path in unique],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "manifest_files": len(unique),
                      "bytes": sum(path.stat().st_size for path in unique)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
