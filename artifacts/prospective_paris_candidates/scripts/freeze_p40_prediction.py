"""Prediction freeze for the fresh prospective P40 pilot.

Emits the launcher-ready artifacts (candidate registry CSV + selection JSON),
the physical job registry with real job keys, and the freeze manifest holding
complete candidate rows, row hashes, target rates/slopes, barrier profiles,
every radius scenario, per-scenario predicted rates and slopes, selection
scores, and the acceptance gates.

Nothing here launches physics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest, PROV_PATH
from derive_p40_candidates import PILOT_K, R_LOAD, R0
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import barrier_from_vector

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SELECTED = "P40_RADIUS_SCALED_ENVELOPE_V1"
PARAM_NAMES = ["cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac"]
RESULT_ROOT = ROOT / "runs" / "prospective_paris_p40_pilot_v1"


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    stage2 = json.loads((OUT / "p40_stage2_selection.json").read_text())
    if stage2["selected_candidate"] != SELECTED:
        raise SystemExit(f"selection changed: {stage2['selected_candidate']!r} != {SELECTED!r}")
    cand = stage2["candidates"][SELECTED]
    vector = [float(cand["vector"][n]) for n in PARAM_NAMES]

    manifest, extra = load_a_native_manifest()
    prov = json.loads(PROV_PATH.read_text())
    base_row = dict(prov["complete_active_material_row"])

    # ---- complete candidate row: only the 5 cleavage fields change -------
    cand_row = dict(base_row)
    for name, value in zip(PARAM_NAMES, vector):
        cand_row[name] = repr(float(value))
    cand_row["option_key"] = SELECTED
    cand_row["candidate_id"] = SELECTED
    cand_row["role"] = "fresh prospective P40 fatigue-slope design candidate"
    cand_row["mechanism_summary"] = ("P40 target m_mid=4.0 at Kstar=18; opening-cleavage surface only; "
                                     "A_NATIVE background held fixed; transfer model NOT qualified")
    cand_row["validation_status"] = "PROSPECTIVE_FROZEN"

    changed = sorted(k for k in base_row if base_row[k] != cand_row.get(k))
    identity = {"option_key", "candidate_id", "role", "mechanism_summary", "validation_status"}
    constitutive_changed = [k for k in changed if k not in identity]
    if sorted(constitutive_changed) != sorted(PARAM_NAMES):
        raise SystemExit(f"candidate changed non-cleavage fields: {constitutive_changed}")

    # ---- launcher artifacts ---------------------------------------------
    fields = list(base_row.keys())
    registry_path = OUT / "p40_candidate_registry.csv"
    with registry_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerow(base_row)     # A_NATIVE control, unchanged
        w.writerow({k: cand_row.get(k, "") for k in fields})
    option_order = [base_row["option_key"], SELECTED]

    selection_path = OUT / "p40_candidate_selection.json"
    selection = dict(
        schema="v10.2.30_prospective_paris_p40_selection_v1",
        purpose="Fresh prospective P40 candidate plus the unchanged A_NATIVE control.",
        canonical_option_order=option_order,
        physics_contract=dict(base_model="v10.2.30 audited persistent-site sharp-front model",
                              parameter_transfer_only=True, stochastic_cleavage=True,
                              persistent_sites=True, source_refresh=False,
                              explicit_recovery=False, physical_front_width=True,
                              rebonding=False, PT_substitution=False),
        primary_candidates=[
            dict(paper_order=1, option_key=base_row["option_key"], candidate_id=base_row["candidate_id"],
                 role="unchanged A_NATIVE control"),
            dict(paper_order=2, option_key=SELECTED, candidate_id=SELECTED,
                 role="fresh prospective P40 design candidate"),
        ],
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    # ---- physical job registry ------------------------------------------
    preds = {s: {r["Kmax_MPa_sqrt_m"]: r for r in rows}
             for s, rows in _load_predictions().items()}
    jobs = []
    for K in PILOT_K:
        dk = round((1.0 - R_LOAD) * K, 6)
        result_path = RESULT_ROOT / SELECTED / f"K_{K:g}_R_0.1_seed_1720"
        jobs.append(dict(
            composite_id=f"{SELECTED}__DK_{dk:g}__R0p1__n80__seed1720",
            parameter_option=SELECTED, Kmax_MPa_sqrt_m=K, deltaK_MPa_sqrt_m=dk,
            R=R_LOAD, n_bins=80, seed=1720, temperature_K=300.0, frequency_Hz=1000.0,
            target_ext_um=100, cycles_max="1000000000000", max_wall_seconds=43200,
            fresh=True, resume=False, status="PENDING",
            result_path=str(result_path.resolve()),
            predicted_da_dN_OBSERVED_A_NATIVE_ENVELOPE=preds["OBSERVED_A_NATIVE_ENVELOPE"][K]["predicted_physical_da_dN"],
            predicted_da_dN_RAW_B1=preds["RAW_B1"][K]["predicted_physical_da_dN"],
            predicted_da_dN_RADIUS_SCALED_B1=preds["RADIUS_SCALED_B1"][K]["predicted_physical_da_dN"],
            predicted_local_slope_RADIUS_SCALED_B1=preds["RADIUS_SCALED_B1"][K]["local_slope"],
            target_da_dN=preds["RADIUS_SCALED_B1"][K]["target_da_dN"],
            target_local_slope=preds["RADIUS_SCALED_B1"][K]["target_slope"],
        ))
    job_path = OUT / "physical_job_registry.csv"
    with job_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jobs[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(jobs)

    # ---- barrier profile -------------------------------------------------
    barrier = barrier_from_vector(vector, manifest.cleavage)
    sig = np.linspace(1.0e9, 12.0e9, 45)
    profile = [dict(sigma_Pa=float(s), candidate_G_eV=float(g), a_native_G_eV=float(a))
               for s, g, a in zip(sig, barrier.values_eV(sig, 300.0),
                                  manifest.cleavage.values_eV(sig, 300.0))]
    with (OUT / "p40_barrier_profile.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sigma_Pa", "candidate_G_eV", "a_native_G_eV"], lineterminator="\n")
        w.writeheader(); w.writerows(profile)

    freeze = dict(
        schema="v10.2.30_prospective_paris_p40_prediction_freeze_v1",
        frozen_at_head=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"),
        worktree_clean_at_freeze=(git("status", "--porcelain") == ""),
        prediction_status="NOT_A_QUALIFIED_PHYSICAL_PREDICTION",
        transfer_model_status="B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION",
        selected_candidate=SELECTED,
        selection_rule=stage2["selection_rule"],
        candidates_considered=list(stage2["candidates"]),
        rejection_registry=stage2["rejection_registry"],
        bound_provenance=stage2["bound_provenance"],
        dual_exponential_family_required=stage2["dual_exponential_family_required"],
        candidate_vector=dict(zip(PARAM_NAMES, vector)),
        candidate_vector_sha256=sha256_obj(dict(zip(PARAM_NAMES, vector))),
        complete_candidate_row=cand_row,
        complete_candidate_row_sha256=sha256_obj(cand_row),
        a_native_background_row_sha256=prov["complete_row_sha256"],
        changed_constitutive_fields=sorted(constitutive_changed),
        changed_identity_fields=sorted(k for k in changed if k in identity),
        launcher_artifacts=dict(
            registry_csv=str(registry_path), registry_csv_sha256=sha256_file(registry_path),
            selection_json=str(selection_path), selection_json_sha256=sha256_file(selection_path),
            canonical_option_order=option_order,
            entry_module="arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
            launcher="scripts/run_v10_2_30_weakt_high_cycle_1e12.sh",
        ),
        rate_anchor=json.loads((OUT / "rate_anchor_gstar.json").read_text())["gstar_m_per_cycle"],
        target_profile_sha256=sha256_file(OUT / "target_profile_definition.json"),
        radius_scenarios=stage2["radius_scenarios"],
        alpha_r_frozen=stage2["alpha_r_frozen"],
        reduced_model_deficit=stage2["reduced_model_deficit"],
        scores_by_scenario=cand["scores_by_scenario"],
        minimax_J=cand["minimax_J"], minimax_scenario=cand["minimax_scenario"],
        admissibility=cand["admissibility"],
        physical_jobs=jobs,
        physical_contract=dict(temperature_K=300.0, R=R_LOAD, frequency_Hz=1000.0, mpz_n_bins=80,
                               seed=1720, target_extension_um=100, cycles_max=1e12,
                               max_workers=3, fresh_virgin_paths_only=True, resume_forbidden=True,
                               rebonding=False, PT_substitution=False,
                               explicit_evolution="primary"),
        acceptance_gates=dict(
            all_three_terminal_physical_results=True,
            developed_growth_gate=dict(transient_exclusion_um=20, min_developed_events=10,
                                       min_developed_growth_um=50,
                                       late_over_early_rate_ratio_range=[0.5, 2.0],
                                       requires_physical_target_or_qualified_censor=True),
            max_abs_log10_rate_error_decade=0.20,
            max_abs_adjacent_slope_error=1.0,
            physical_rate_ordering_monotonic=True,
            no_barrier_floor_stress_cap_or_renewal_ceiling_pathology=True,
            event_size_not_dominant_source_of_rate_difference=True,
            on_failure=("second-ranked eligible candidate is NOT run: P40_RAW_B1_ENVELOPE_V1 is "
                        "ineligible (E_rate=1.1366>1.0 under OBSERVED_A_NATIVE_ENVELOPE). A single "
                        "TRANSFER_CALIBRATED_GENERATION_2 row with a new id/hash may then be derived, "
                        "never overwriting P40_RADIUS_SCALED_ENVELOPE_V1."),
        ),
        second_candidate_not_launched=dict(
            candidate_id="P40_RAW_B1_ENVELOPE_V1",
            reason=("ineligible under the frozen minimax gates (E_rate=1.1366 > 1.0 in the "
                    "OBSERVED_A_NATIVE_ENVELOPE scenario) and its five parameters differ from the "
                    "selected row by <0.5% in every coordinate, so a physical run would be nearly "
                    "redundant while consuming a second 3-job x 12h wall-clock budget"),
        ),
    )
    (OUT / "prediction_freeze_manifest.json").write_text(json.dumps(freeze, indent=2, default=str) + "\n")

    print(json.dumps(dict(
        selected=SELECTED, candidate_vector=dict(zip(PARAM_NAMES, vector)),
        complete_candidate_row_sha256=freeze["complete_candidate_row_sha256"],
        changed_constitutive_fields=freeze["changed_constitutive_fields"],
        registry_sha256=freeze["launcher_artifacts"]["registry_csv_sha256"],
        jobs=[j["composite_id"] for j in jobs],
        predicted_rates={j["Kmax_MPa_sqrt_m"]: j["predicted_da_dN_RADIUS_SCALED_B1"] for j in jobs},
        target_rates={j["Kmax_MPa_sqrt_m"]: j["target_da_dN"] for j in jobs},
    ), indent=2, default=str))


def _load_predictions() -> dict:
    out: dict = {}
    with (OUT / "p40_stage2_predictions.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["candidate_id"] != SELECTED:
                continue
            out.setdefault(r["scenario"], []).append(
                {k: (float(v) if k not in ("candidate_id", "scenario") else v) for k, v in r.items()})
    return out


if __name__ == "__main__":
    main()
