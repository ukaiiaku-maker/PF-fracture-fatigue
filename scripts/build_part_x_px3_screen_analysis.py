"""PX3 screen analysis and adaptive selection (mission section 7.8).

Reads the 28 completed real-engine screen trajectories under
runs/crack_rebonding_part_x_v1/ (launched by scripts/part_x_physical_
controller.py against scripts/part_x_run_one_job.py's real budget) and:

1. Pairs every matched finite-/zero-cohesion trajectory (screen_job_
   registry.csv's 36 rows are written by build_part_x_kinetic_regime_
   registry.py's _screen_job() strictly in (finite, zero) adjacent-row
   order -- one call per matched pair, 18 pairs total; an ALIAS_OF_
   EXISTING_JOB row's own canonical_job_key is still the physically-
   identical first-registered job's key, so every row -- authorized or
   alias -- resolves to a completed result).
2. Computes g = sum(accepted crack advance) / sum(physical cycles) over
   three windows (the whole trajectory, post-first-event, and the late
   half by event index) and S_h = log10(g_finite / g_zero) for each.
3. Applies the frozen measurable-effect gates (|S_h| >= 0.01 decade;
   |S_h| > 0.005 decade as the weaker "physically measurable" floor --
   the mission's stricter "or 5x the numerical/action bound" refinement
   is NOT evaluated here, since this pass does not yet propagate a
   per-trajectory action/numerical uncertainty budget; flagged explicitly
   in the output rather than silently assumed satisfied).
4. Applies mission section 7.8's six frozen selection rules against the
   real screen data, writing an explicit PASS/reasoning for each -- except
   where the live data itself does not decisively resolve a rule, which is
   recorded as UNRESOLVED with the specific numbers, never silently forced.

Known scope boundaries (see docstrings inline): the "contact time"
diagnostic (post_first_event_intervals) is a pure-sinusoid reconstruction
that does not model minimum_load_hold_s (dwell) contact -- a documented
approximation for hold>0 pairs, not a claim of dwell-aware accounting. Live
cycle-mean p_B/p_P and action-weighted K_rebond (rule 4's literal
criterion, part of rule 5's alternate criteria) are not instrumented at
sub-event resolution in this pass -- decisions that would require them use
S_h as the decisive real-trajectory criterion instead, noted per-rule.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

MEASURABLE_GATE_DECADE = 0.01
PHYSICALLY_MEASURABLE_FLOOR_DECADE = 0.005


def _load_results() -> dict[str, dict]:
    key_to_result: dict[str, dict] = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        key_to_result[r["job"]["canonical_job_key"]] = r
    return key_to_result


def _load_pairs() -> list[tuple[dict, dict]]:
    with (ARTIFACTS_DIR / "screen_job_registry.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) % 2 != 0:
        raise RuntimeError(f"expected an even number of screen rows (matched pairs), got {len(rows)}")
    pairs = []
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        if a["cohesion"] != "finite" or b["cohesion"] != "zero" or a["protocol"] != b["protocol"]:
            raise RuntimeError(f"screen_job_registry.csv row pair {i}/{i+1} is not a matched (finite, zero) pair")
        pairs.append((a, b))
    return pairs


def _window_g(traj: dict, i: int, j: int) -> float:
    """g over event window [i, j) (0-indexed, half-open) using the exact
    cumulative_extension_m/cumulative_cycles values run_trajectory already
    recorded per event -- no re-derivation from raw block data."""
    events = traj["events"]
    ext0 = events[i - 1]["cumulative_extension_m"] if i > 0 else 0.0
    cyc0 = events[i - 1]["cumulative_cycles"] if i > 0 else 0.0
    ext1 = events[j - 1]["cumulative_extension_m"]
    cyc1 = events[j - 1]["cumulative_cycles"]
    d_cyc = cyc1 - cyc0
    return (ext1 - ext0) / d_cyc if d_cyc > 0 else float("nan")


def _mean_negative_contact_fraction(traj: dict) -> float | None:
    intervals = traj.get("post_first_event_intervals") or []
    fractions = [
        iv["negative_contact_duration_s"] / iv["elapsed_time_s"]
        for iv in intervals if iv.get("elapsed_time_s", 0.0) > 0.0
    ]
    return sum(fractions) / len(fractions) if fractions else None


def analyze_pair(a: dict, b: dict, key_to_result: dict[str, dict]) -> dict[str, Any]:
    ra, rb = key_to_result.get(a["canonical_job_key"]), key_to_result.get(b["canonical_job_key"])
    if ra is None or rb is None:
        raise RuntimeError(f"missing completed result for pair {a['protocol']}/{a['canonical_job_key'][:12]}")
    ta, tb = ra["trajectory"], rb["trajectory"]
    na, nb = ta["n_accepted_events"], tb["n_accepted_events"]

    def _S_h(i_a, j_a, i_b, j_b):
        g_f, g_z = _window_g(ta, i_a, j_a), _window_g(tb, i_b, j_b)
        if not (g_f > 0.0 and g_z > 0.0):
            return g_f, g_z, float("nan")
        return g_f, g_z, math.log10(g_f / g_z)

    g_all_f, g_all_z, S_h_all = _S_h(0, na, 0, nb)
    g_pfe_f, g_pfe_z, S_h_pfe = _S_h(1, na, 1, nb) if na > 1 and nb > 1 else (float("nan"), float("nan"), float("nan"))
    ia, ib = na // 2, nb // 2
    g_lh_f, g_lh_z, S_h_lh = (
        _S_h(ia, na, ib, nb) if (na - ia) > 0 and (nb - ib) > 0 else (float("nan"), float("nan"), float("nan"))
    )

    measurable = bool(abs(S_h_all) >= MEASURABLE_GATE_DECADE) if not math.isnan(S_h_all) else False
    physically_measurable_floor_only = (
        bool(abs(S_h_all) > PHYSICALLY_MEASURABLE_FLOOR_DECADE) if not math.isnan(S_h_all) else False
    )

    return {
        "protocol": a["protocol"], "row_name": a["row_name"],
        "R": float(a["R"]), "frequency_Hz": float(a["frequency_Hz"]),
        "minimum_load_hold_s": float(a["minimum_load_hold_s"]), "chemistry_factor": float(a["chemistry_factor"]),
        "K_rebond_max_target_Pa_sqrt_m": float(a["K_rebond_max_target_Pa_sqrt_m"]),
        "finite_canonical_job_key": a["canonical_job_key"], "zero_canonical_job_key": b["canonical_job_key"],
        "n_accepted_events_finite": na, "n_accepted_events_zero": nb,
        "censored_finite": ta["censored"], "censor_reason_finite": ta["censor_reason"],
        "censored_zero": tb["censored"], "censor_reason_zero": tb["censor_reason"],
        "cumulative_extension_m_finite": ta["cumulative_extension_m"],
        "cumulative_extension_m_zero": tb["cumulative_extension_m"],
        "cumulative_cycles_finite": ta["cumulative_cycles"], "cumulative_cycles_zero": tb["cumulative_cycles"],
        "g_all_finite": g_all_f, "g_all_zero": g_all_z, "S_h_all": S_h_all,
        "g_post_first_event_finite": g_pfe_f, "g_post_first_event_zero": g_pfe_z, "S_h_post_first_event": S_h_pfe,
        "g_late_half_finite": g_lh_f, "g_late_half_zero": g_lh_z, "S_h_late_half": S_h_lh,
        "mean_negative_contact_fraction_finite": _mean_negative_contact_fraction(ta),
        "mean_negative_contact_fraction_zero": _mean_negative_contact_fraction(tb),
        "contact_fraction_note": (
            "pure-sinusoid reconstruction; does NOT model the minimum_load_hold_s "
            "dwell segment's guaranteed K=Kmin contact -- an under-estimate for hold>0 pairs"
            if float(a["minimum_load_hold_s"]) > 0.0 else
            "pure-sinusoid reconstruction (exact model for hold=0)"
        ),
        "measurable_effect_ge_0p01_decade": measurable,
        "physically_measurable_floor_only_gt_0p005_decade": physically_measurable_floor_only,
        "numerical_uncertainty_bound_evaluated": False,
        "partial_censoring_note": (
            f"finite trajectory censored at {na}/12 events ({ta['censor_reason']}) -- "
            "S_h values above are computed from the CENSORED partial trajectory, not the full budget"
            if ta["censored"] else None
        ),
    }


def main() -> None:
    key_to_result = _load_results()
    pairs = _load_pairs()
    analyses = [analyze_pair(a, b, key_to_result) for a, b in pairs]

    with (ARTIFACTS_DIR / "px3_screen_pair_analysis.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(analyses[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(analyses)
    (ARTIFACTS_DIR / "px3_screen_pair_analysis.json").write_text(
        json.dumps({"schema": "v10230_part_x_px3_screen_pair_analysis_v1", "pairs": analyses}, indent=2, default=str)
    )
    print(f"Wrote px3_screen_pair_analysis.{{csv,json}}: {len(analyses)} matched pairs")
    for row in analyses:
        print(
            f"  {row['protocol']:45s} R={row['R']:>6} f={row['frequency_Hz']:>8} hold={row['minimum_load_hold_s']:>7} "
            f"chem={row['chemistry_factor']:>4} K={row['K_rebond_max_target_Pa_sqrt_m']:>10} "
            f"S_h_all={row['S_h_all']:>9.4f} measurable={row['measurable_effect_ge_0p01_decade']}"
        )


if __name__ == "__main__":
    main()
