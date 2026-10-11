#!/usr/bin/env python3
"""Finalize the active-growth pilot and bounded cavity/FEM audits."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_current_science_pilots_v2 import (  # noqa: E402
    OUT, event_pairs, load_case, primary_state_rows, stage_query, state_record,
    write_csv,
)

HOT = (ROOT / "integration_evidence/current_science_pilots_6abbf10/pf/DBTT_1200K/"
       "developed_1000um_current_v2_run3")


def main() -> int:
    history, steps = load_case(HOT, "1200")
    pairs = event_pairs(history)
    states = [state_record(row, steps[row.source_row]) for row in primary_state_rows(history)]
    write_csv(OUT / "DBTT_1200K_MECHANISM_HISTORY_v3.csv", states)
    queries = {str(x): stage_query(history, x) for x in (100.0, 500.0, 1000.0)}
    (OUT / "DBTT_1200K_LENGTH_QUERIES_v3.json").write_text(json.dumps({
        "schema": "developed-length-queries/3", "coordinate": "projected extension",
        "interpolation_performed": False, "queries_um": queries,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")

    xs = [r["source_step"] for r in states]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    axes[0, 0].plot(xs, [r["opening_um"] for r in states], color="#3b528b")
    axes[0, 0].set_ylabel("Opening (µm)")
    axes[0, 1].plot(xs, [r["cleavage_clock_B"] for r in states], color="#5ec962")
    axes[0, 1].set_ylabel("Cleavage clock, B")
    axes[1, 0].plot(xs, [r["active_mobile_count"] for r in states], label="Mobile", color="#21918c")
    axes[1, 0].plot(xs, [r["active_retained_count"] for r in states], label="Retained", color="#440154")
    axes[1, 0].set_ylabel("Active line-content count")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(xs, [r["persistent_source_backstress_GPa"] for r in states], color="#d1495b")
    axes[1, 1].set_ylabel("Source backstress (GPa)")
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.02, 0.94, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.set_xlabel("Accepted source step")
    fig.savefig(OUT / "DBTT_1200K_mechanism_history_v3.png", dpi=240)
    plt.close(fig)

    final = history.rows[-1]
    payload = {
        "schema": "active-growth-pilot-completion/3",
        "hot_path": {
            "status": "COMPLETED_PHYSICAL_TARGET",
            "raw_root": str(HOT.relative_to(ROOT)),
            "temperature_K": 1200.0,
            "accepted_events": len(pairs),
            "terminal_stage": final.stage.value,
            "terminal_projected_extension_um": final.stage_extension_m * 1e6,
            "terminal_path_arclength_um": final.path_arclength_m * 1e6,
            "terminal_K_J_MPa_sqrt_m": None,
            "terminal_active_mobile_plus_retained_count": states[-1]["active_mobile_plus_retained_count"],
            "terminal_source_backstress_GPa": states[-1]["persistent_source_backstress_GPa"],
            "pilot_prefix_exact": True,
            "pilot_prefix": [[165, 52.37574061089843, 2.2973400956249022],
                             [186, 53.872930408603124, 10.564512816410554]],
            "native_combined_checkpoint": str((HOT / "checkpoint/run_state_checkpoint.json").relative_to(ROOT)),
            "queries": "DBTT_1200K_LENGTH_QUERIES_v3.json",
        },
        "cavity": {
            "status": "QUALIFIED_COMPLETE_LIGAMENT_OPERATION",
            "failed_arm": "cavity_enabled",
            "crack_tip_x_um": 592.5993004046689,
            "nearest_cavity_center_x_um": 600.0,
            "center_separation_um": 7.4006995953311,
            "solid_ligament_um": 7.358229034070524,
            "local_J_J_per_m2": 3932.2698590937002,
            "local_J_valid": False,
            "invalid_reason": "cavity_free_surface_intersects_contour",
            "supported_contour_radii_um": [26.591927524494184, 37.5, 50.0],
            "marginal_trial_advances_um": [20.0, 30.0],
            "marginal_trial_count_observed": 0,
            "complete_ligament_G_J_per_m2": 40.4943938223355,
            "complete_ligament_mesh_relative_error": 0.08771391032970505,
            "complete_ligament_mesh_relative_limit": 0.1,
            "candidate_effective_rate_s": 5.390928548202657e-58,
            "candidate_crossing_time_s": 2.233929559967089e56,
            "interpretation": "local J remains invalid, but the distinct original complete-ligament topology operation is qualified; its current hazard rate predicts no event in a practical bounded window",
        },
        "fem_czm": {
            "status": "EXACT_HISTORICAL_INPUT_UNAVAILABLE_AFTER_TARGETED_ROUTE",
            "backend_commit": "293491157063484bb10df6adc479f847a6a08ba6",
            "required_family_sha256": "a85b57ad9eee8331ef34ea222760ce7d4064f48ddef668f1e28a666d14946f3a",
            "required_mechanical_fingerprint": "e23971019aca58e8e3319a207153c4faa3ead76417e1d537ec00c2ed2986e815",
            "configuration_fingerprint": "1447653d199f0b43cb475951092d69444c9b785f6fdf518c723792abb3b1f5e5",
            "documented_resolution_path_exists": False,
            "exact_hash_in_present_local_git_family_blobs": False,
            "present_canonical_git_blob": "ccd4ef6b2b05ddaac1a1733b960f6f9a17ed4322",
            "present_canonical_family_sha256": "a15eada6a3d86a8f52e3ba6a921c5cf6c98d7f9bae082f2e09f1ad8c14307c24",
            "live_cache_family_sha256": "d41b08f69ae773009f65f4c0094ef5436ba7f0f290a94172e5b894d09a41f7c3",
            "accepted_full_FEM_steps": 0,
            "guard_weakened": False,
            "companion_branch": "codex/current-pf-companion",
            "companion_commit": "fa24f7b",
            "new_companion_run": "prepared and tested but not launched: the required durable independent backup destination is not yet available; the clean pinned checkout was not modified",
        },
        "storage": {"raw_hot_path_MiB": round(sum(p.stat().st_size for p in HOT.rglob('*') if p.is_file()) / 2**20, 3),
                    "free_space_GiB": round(shutil.disk_usage('/Volumes/Data').free / 2**30, 3),
                    "data_deleted": False, "external_data_modified": False},
    }
    (OUT / "ACTIVE_GROWTH_COMPLETION_v3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "ACTIVE_GROWTH_COMPLETION_v3.md").write_text(f"""# Active-growth pilot completion

The corrected DBTT 1200 K calculation reproduced the two-event pilot prefix exactly and then completed {payload['hot_path']['accepted_events']} accepted geometry events. It reached {payload['hot_path']['terminal_projected_extension_um']:.3f} µm projected extension ({payload['hot_path']['terminal_path_arclength_um']:.3f} µm path length). The terminal observation is a committed geometry state, so no post-event $K_J$ is invented.

The mechanism history uses the native chronological stage ledger. `N_em` is reported as active mobile-plus-retained line content, not cumulative emission; `sigma_back` is the persistent source-law Taylor backstress. Query results at 100, 500, and 1000 µm retain their measured pre-event coordinates and committed jump intervals without interpolation.

## Cavity boundary

The retained enabled snapshot places the crack tip at 592.599 µm and the first cavity center at 600 µm. Their 7.401 µm center separation is not the solid ligament. The actual polygonal ray intersection gives 7.358229 µm of intact solid. Local isolated-tip $J$ remains invalid because every supported contour crosses the cavity, but the distinct original complete-ligament topology operation is available: 40.494394 J m$^{{-2}}$ with an 8.771% level-3/4 difference, below the existing 10% limit. The accepted state and physical time are unchanged. At this load its original hazard route gives a $5.39\times10^{{-58}}$ s$^{{-1}}$ rate, so no practical finite-window connection event is claimed.

## FEM/CZM boundary

The exact historical `a85b57…` family remains unavailable and the pinned backend guard remains intact. The separate `codex/current-pf-companion` branch at `fa24f7b` binds the native full-order builder to the present `d41b08…` family and its `1447653…` mechanical configuration, explicitly distinct from historical `e239710…`. Its focused contract suite passes, but the one-step execution remains gated on a durable independent backup destination; the clean historical checkout was not modified.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
