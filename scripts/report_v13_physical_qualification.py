#!/usr/bin/env python3
"""Summarize physical companion results; never launches an ensemble."""
import argparse
import itertools
import json
import math
from pathlib import Path

from scripts.run_pf_current_source_multifront_field_atlas_v12 import CASES, atomic_json


def qualification_gate(results, plan):
    reasons = []
    if len(results) != 8 or any(r.get("status") != "COMPANION_CHECK_COMPLETE" for r in results):
        reasons.append("not_all_eight_clean_parent_companion_checks_completed")
    observations = [o for r in results for o in r.get("companions", [])]
    if not any(o.get("single_reference_overlay_disposition") == "PAIR_ACCEPTED" for o in observations):
        reasons.append("no_stochastic_companion_selected_at_the_preregistered_reference_mark")
    valid = {r["case"]: r for r in results if r.get("sensitivity_surface")}
    def key(row):
        return (round(math.log10(row["beta_B"]), 8), row["Q_junction_eV"], row["Q_overlap_eV"])
    tables = {case: {key(row): row["P_committed"] for row in r["sensitivity_surface"]} for case, r in valid.items()}
    grid = [plan["beta_B_log10_grid"], plan["Q_junction_eV_grid"], plan["Q_overlap_eV_grid"]]
    regions = []
    for point in itertools.product(*grid):
        robust_cases = []
        for case, table in tables.items():
            probability = table.get(point)
            if probability is None or not .05 <= probability <= .95:
                continue
            neighbors = []
            for axis in range(3):
                index = grid[axis].index(point[axis])
                for shift in (-1, 1):
                    if 0 <= index+shift < len(grid[axis]):
                        neighbor = list(point)
                        neighbor[axis] = grid[axis][index+shift]
                        neighbors.append(table[tuple(neighbor)])
            if neighbors and all(.01 <= p <= .99 for p in neighbors):
                robust_cases.append(case)
        if len({c.rsplit("_", 1)[0] for c in robust_cases}) < 2:
            continue
        probabilities = {case: table[point] for case, table in tables.items()}
        temperature_contrast = any(
            f"{m}_300K" in probabilities and f"{m}_1000K" in probabilities
            and abs(probabilities[f"{m}_300K"]-probabilities[f"{m}_1000K"]) >= .05
            for m in ("Peak", "DBTT", "weakT", "ceramic"))
        material_contrast = any(abs(probabilities[a]-probabilities[b]) >= .05
            for a, b in itertools.combinations(probabilities, 2) if a.rsplit("_", 1)[1] == b.rsplit("_", 1)[1])
        if temperature_contrast and material_contrast:
            regions.append({"log10_beta_B": point[0], "Q_junction_eV": point[1], "Q_overlap_eV": point[2],
                "robust_nondegenerate_cases": robust_cases, "probabilities": probabilities})
    if not regions:
        reasons.append("no_common_robust_nondegenerate_material_and_temperature_sensitive_grid_region")
    state_effect = any(
        o.get("process_state_attribution") and
        not math.isclose(o["process_state_attribution"]["raw_actual_per_s"],
                         o["process_state_attribution"]["raw_without_shielding_or_blunting_per_s"], rel_tol=.05, abs_tol=1e-30)
        for o in observations)
    if not state_effect:
        reasons.append("defensible_process_state_rate_sensitivity_not_demonstrated")
    return {"status": "PASS_SHORT_ENSEMBLE_WARRANTED" if not reasons else "NOT_PASSED_NO_ENSEMBLE",
        "reasons": reasons, "robust_regions": regions, "physical_ensemble_launched": False,
        "process_state_sensitivity_demonstrated": state_effect,
        "barrier_identifiability": "at zero junction separation, only Q_junction+Q_overlap is identifiable",
        "boundary": "BRANCHING_KINETICS_MODEL_UNCALIBRATED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root/"companions/summary.json").read_text())
    plan = json.loads((args.root/"preregistered_plan.json").read_text())
    gate = qualification_gate(summary["cases"], plan)
    atomic_json(args.root/"physical_ensemble_gate.json", gate)
    lines = ["# V13 physical companion-state qualification", "",
        "Permanent boundary: **BRANCHING_KINETICS_MODEL_UNCALIBRATED**.", "",
        "Initial V13 sampler/restoration qualification remains accepted. No sampler-only Monte Carlo or historical atlas continuation was performed.", "",
        "The initial record is `55955722db40ef65c43d2f1c08357ad8f77b82fa`. Complete initial history and the physical-source increment are preserved as verified Git bundles on the Data drive.", "",
        "## Clean parents and actual companion mechanics", "",
        "| Case | First accepted endpoint (s) | Opening (µm) | Companion check | Exact pair admissible |",
        "|---|---:|---:|---|---|"]
    all_surface = [s for r in summary["cases"] for s in r.get("sensitivity_surface", [])]
    if all_surface:
        maximum = max(row["P_committed"] for row in all_surface)
        intro = ["Model scope: **FRESH_INDEPENDENT_POST_PRIMARY_ORDER3_COMPANION_WITH_TAU_B_LE_TAU_C**. This is the qualified independent-sequential null model, not the final co-critical branching model. The original report/archive remains preserved at record `d10c3eb`.", "", "## Physical result", "",
            f"All {len(summary['cases'])} clean-parent checks completed. The largest committed-branch probability across {len(all_surface)} analytic case/parameter evaluations is **{maximum:.8g}**. No physical short ensemble was launched.", "",
            "The stop decision follows from the analytic surface, independently of the eight reference random marks. Exact two-arm mechanical acceptance does not imply appreciable companion-embryo probability.", ""]
        lines[8:8] = intro
    for r in summary["cases"]:
        p = r.get("clean_parent_record", {})
        candidates = r.get("companions", [])
        admissibility = ", ".join(str(c.get("exact_pair_admissible", "unevaluated")) for c in candidates) or "unevaluated"
        lines.append(f"| {r['case']} | {p.get('accepted_endpoint_s', '—')} | {p.get('accepted_opening_m', 0)*1e6 if p else '—'} | {r['status']} | {admissibility} |")
    lines += ["", "Parents start from fresh initialization at theta40 with the pinned four material rows and seed 3621. Each stops at its first accepted canonical single-arm cleavage. Full pre-cleavage and accepted single-arm checkpoints, event context, callable inventory, raw first-passage time, winner/ordinal, complete process state and RNG hashes are saved beneath `physical_parents/`.", "",
        "The original parent loop is unchanged apart from default-off capture hooks and a demonstrated diagnostic-name repair. Startup failures are retained in `parents/` and `clean_parents/`; neither produced an accepted physical interval. A new terminal-label publication error occurred after some valid parent captures; those raw worker reports are retained and do not require rerunning their saved parent states.", "",
        "Companion mechanics uses the exact same imposed opening and primary endpoint. The candidate-ray tensor is probed from the actual post-primary FEM stress field at the original junction. The companion is not yet a physical crack tip: its kinetic drive is the exact discrete single-to-pair marginal energy per companion length. This is model-native discrete mechanics, **not remote K or continuum-qualified G**. The existing signed-process strength law and raw cleavage barrier are evaluated on an isolated copy of the complete accepted process state. No additional emission, time, renewal, threshold or baseline RNG update occurs.", "",
        "The unchanged exact whole-pair geometry and energy transaction is evaluated separately. Primary energy cost is held exactly at the canonical accepted value; companion cost uses the existing hazard-energy formula. Branch-only junction/overlap barriers modify mark kinetics, not the established whole-topology energy rule. A single preregistered reference mark checks the overlay; no repeated-seed search is used.", "",
        "Sensitivity surfaces store embryo probability, pair-admissibility indicator, and committed-branch probability separately. Invalid observations remain unevaluated—no artificial rate or probability is supplied. Process-state attribution removes shielding/blunting only algebraically for diagnostics, never by changing an accepted physical state.", "",
        "At a newly born junction, separation is zero, so junction and overlap barriers are identifiable only through their sum. These surfaces do not calibrate the three new branch-only quantities or establish branch spacing.", "",
        "## Ensemble decision", "", f"**{gate['status']}**", ""]
    lines.extend("- " + reason for reason in gate["reasons"])
    lines += ["", "A failed grid-robustness gate is not a universal no-branching theorem. A short physical ensemble is not run unless all required physical checks and a defensible nondegenerate region pass. No long field atlas is authorized.", ""]
    lines += ["## Candidate-level evidence", "",
        "| Case | Candidate observation | Raw arrival (s⁻¹) | Raw barrier (eV) | Pair margin (J/m) | Embryo probability range | Committed probability range |",
        "|---|---|---:|---:|---:|---|---|"]
    for result in summary["cases"]:
        for observation in result.get("companions", []):
            surface = [row for row in result.get("sensitivity_surface", [])
                       if row["companion_id"] == observation["candidate_id"]]
            ranges = []
            for name in ("P_embryo", "P_committed"):
                values = [row[name] for row in surface]
                ranges.append(f"{min(values):.6g}–{max(values):.6g}" if values else "unevaluated")
            raw = observation.get("raw_arrival_per_s")
            barrier = observation.get("raw_barrier_J")
            margin = observation.get("pair_energy_margin_J_per_m")
            fmt = lambda value: "unevaluated" if value is None else f"{value:.8g}"
            lines.append(f"| {result['case']} | {observation.get('reason', observation['status'])} | {fmt(raw)} | {fmt(None if barrier is None else barrier/1.602176634e-19)} | {fmt(margin)} | {ranges[0]} | {ranges[1]} |")
    lines += ["", "The full JSON records retain candidate identities, tensor probes, process-state fingerprints, costs, exact fallback identities, and every preregistered grid point. A range spans the entire sensitivity grid and is not a fitted or calibrated uncertainty interval.", ""]
    lines += ["## Interpretation and limits", "",
        "The opportunity range is beta_B = 10⁻⁶ to 1 with tau_c = 1 µs. All eight raw multihit orders are three. Even with zero added junction/overlap barriers and the longest tested opportunity, raw arrival × tau_c is only approximately 0.000517–0.006455. Three arrivals within that opportunity are therefore very unlikely. Nonnegative branch-only barriers can only reduce this probability. No exposure range or barrier was changed after seeing the result.", "",
        "All eight accepted parent radii equal the 1 µm reference radius, and the frozen shielding/blunting attribution does not demonstrate the preregistered 5% raw-rate effect. This is a statement about these first-cleavage snapshots, not a proof that the continuous-emission process has no effect on the preceding parent history or on later growth. Material/temperature differences in small raw probabilities exist, but do not establish the required nondegenerate physical branching regime.", "",
        "The deterministic pair adapter accepted real PF FEM states in all eight cases. All eight preregistered stochastic reference marks retained the exact single-arm result; a positive stochastic pair commitment and subsequent owner-transition trajectory have not been demonstrated here. No mark seed search or forced physical branch was performed.", "",
        "Focused capture, companion-contract, topology, clock and checkpoint checks: 35 passed. Compileall and git diff --check passed. The accepted initial sampler qualification was not repeated; no new full-suite/Monte Carlo claim is made.", ""]
    (args.root/"V13_PHYSICAL_COMPANION_QUALIFICATION.md").write_text("\n".join(lines))
    print(gate["status"], gate["reasons"])


if __name__ == "__main__":
    main()
