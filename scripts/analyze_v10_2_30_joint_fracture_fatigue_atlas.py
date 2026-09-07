#!/usr/bin/env python3
"""Build the corrected v10.2.30 fracture-fatigue response atlas.

This program is analysis-only.  It hashes and classifies archived evidence,
freezes equations before aggregate comparison, evaluates the transparent
monotonic hierarchy, creates a deterministic scrambled-Sobol *sample*, audits
cooperative-renewal saturation before any clustering or selection, and reuses
only fingerprint-identical physical trajectories.  It never launches or
resumes a production trajectory and never modifies the qualified solver.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.special import gammainc, gammaincinv
from scipy.stats import qmc, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.analytical_monotonic_fracture_v10230 import (
    MODEL_ID as MONOTONIC_MODEL_ID,
    MonotonicControls,
    cooperative_rate,
    f0_implicit_sensitivities,
    finite_difference_sensitivity,
    solve_first_passage,
)
from arrhenius_fracture.analytical_stationary_fatigue_v10230 import (
    MODEL_ID as FATIGUE_MODEL_ID,
    AnalyticalControls,
    solve_hierarchy as solve_fatigue_hierarchy,
)
from arrhenius_fracture.material_manifest import (
    ExpFloorBarrier,
    KB_EV_PER_K,
    MaterialManifest,
    TransportBarrier,
)


OUT = ROOT / "runs/joint_fracture_fatigue_archetype_atlas_v2"
FIG = OUT / "figures"
HIST = Path("/private/tmp/taylor-peierls-spatial-coupling-paper-audit")
QUALIFIED_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"
SOLVER_SHA = "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b"
SOURCE_HEAD = "43b5ec9a259b636c5a950eec1f6fffbf3ef4b65c"
HIST_HEAD = "2df158bf8d1484f40898c64a11fc76fdd327178c"
ATLAS_SEED = 10230071
INITIAL_ROWS = 131_072
ADAPTIVE_ROWS = 16_384
TEMPERATURES = np.array([300.0, 600.0, 700.0, 900.0, 1100.0, 1200.0])
RATES = np.array([0.002, 0.02, 0.2])
K_LADDER = np.array([12.0, 15.0, 18.0, 24.3])
FATIGUE_R = 0.1
FATIGUE_FREQUENCY_HZ = 1000.0
FATIGUE_PHASE_COUNT = 128
COOPERATIVE_CEILING_FRACTION = 0.95
BARRIER_FLOOR_RELATIVE_BAND = 0.01
ZERO_ACTIVITY_FRACTION = 1.0e-12
DOMINANT_PHASE_FRACTION = 0.50
DOC_NAMES = [
    "Fatigue_and_fracture_V2(3).docx",
    "SI_MPZ_Parameterizations_0D_1D_Equations_v9_11_1(3).docx",
    "MPZ_PF_FEM_2D_Parameterization_Handoff_v9_11_1(3).docx",
    "2_Analytical_Steady_State_Fatigue_Crack_Growth_Notes_OMML_Updated.docx",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name: str, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    frame = pd.DataFrame(list(rows))
    if fields is not None:
        frame = frame.reindex(columns=fields)
    frame.to_csv(OUT / name, index=False)


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _f(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if math.isfinite(value) else float(default)


def material_from_row(row: Mapping[str, Any]) -> MaterialManifest:
    emission = ExpFloorBarrier(
        G00_eV=_f(row, "emit_G00_eV"), gT_eV_per_K=_f(row, "emit_gT_eV_per_K"),
        sigc0_Pa=_f(row, "emit_sigc0_GPa") * 1e9,
        sT_Pa_per_K=_f(row, "emit_sT_GPa_per_K") * 1e9,
        alpha=_f(row, "emit_exp_a"), exponent=_f(row, "emit_exp_n"),
        floor_fraction=_f(row, "emit_floor_frac"), attempt_frequency_s=1e11,
    )
    cleavage = ExpFloorBarrier(
        G00_eV=_f(row, "cleave_G00_eV"), gT_eV_per_K=_f(row, "cleave_gT_eV_per_K"),
        sigc0_Pa=_f(row, "cleave_sigc0_GPa") * 1e9,
        sT_Pa_per_K=_f(row, "cleave_sT_GPa_per_K") * 1e9,
        alpha=_f(row, "cleave_exp_a"), exponent=_f(row, "cleave_exp_n"),
        floor_fraction=_f(row, "cleave_floor_frac"), attempt_frequency_s=1e12,
    )
    return MaterialManifest(
        name=str(row.get("material_class", row.get("target_class", "unknown"))),
        candidate_id=str(row.get("candidate_id", row.get("option_key", "unknown"))),
        cleavage=cleavage, emission=emission,
        peierls=TransportBarrier(
            H0_eV=_f(row, "peierls_H0_eV"),
            activation_entropy_kB=_f(row, "peierls_activation_entropy_kB"),
            alpha=_f(row, "peierls_exp_a"), exponent=_f(row, "peierls_exp_n"),
            attempt_frequency_s=_f(row, "peierls_nu0_s", 1e12),
        ),
        taylor=TransportBarrier(
            H0_eV=_f(row, "taylor_H0_eV"),
            activation_entropy_kB=_f(row, "taylor_activation_entropy_kB"),
            alpha=_f(row, "taylor_exp_a"), exponent=_f(row, "taylor_exp_n"),
            attempt_frequency_s=_f(row, "taylor_nu0_s", 1e11),
        ),
        taylor_corr_rho_c_m2=_f(row, "taylor_corr_rho_c_m2", 5e12),
        taylor_corr_scale=_f(row, "taylor_corr_scale", 1.0),
        source_sites_per_system=_f(row, "source_sites_per_system", 1.0),
        encounter_efficiency=_f(row, "encounter_efficiency", 1.0),
        retained_recovery_rate_s=_f(row, "retained_recovery_rate_s", 0.0),
        source_refresh_length_m=_f(row, "source_refresh_length_um", 1.0) * 1e-6,
        c_blunt=_f(row, "c_blunt", 0.0),
        max_K_shield_MPa_sqrt_m=_f(row, "max_K_shield_MPa_sqrt_m", 0.0),
    )


def load_candidate_rows() -> pd.DataFrame:
    terminal = pd.read_csv(HIST / "analysis_outputs/oneD_v2_terminal_predictive_program/oneD_v2_new_four_class_registry.csv")
    canonical = terminal[terminal.candidate_id.isin([
        "v913_zeroD_sobol_0242980", "v913_zeroD_sobol_0202500"
    ])].copy()
    weak_ceramic = pd.read_csv(
        HIST / "paper_selections/v9_13_weakT_ceramic_5T_100um_v1/v9_13_weakT_ceramic_primary_handoff.csv"
    )
    canonical = pd.concat([canonical, weak_ceramic], ignore_index=True, sort=False)
    canonical["registry_role"] = canonical.candidate_id.map({
        "v913_zeroD_sobol_0242980": "CANONICAL_PEAK",
        "v913_zeroD_sobol_0202500": "CANONICAL_DBTT",
        "v913_zeroD_sobol_0129902": "CANONICAL_WEAK_T",
        "v913_zeroD_sobol_0077080": "CANONICAL_CERAMIC",
    })
    options = json.loads((HIST / "mpz_v9_11_response_options.json").read_text())["options"]
    dbtt = pd.read_csv(HIST / "mpz_v9_11_dbtt_option_rows.csv")
    option_rows = []
    for key, spec in options.items():
        if key == "weakT_primary":
            source = pd.read_csv(HIST / "mpz_v9_11_parameters/weakT/spatial_promotion_manifest.csv").iloc[0]
        else:
            source = dbtt[dbtt.candidate_id == spec["candidate_id"]].iloc[0]
        item = source.to_dict()
        item.update({"registry_role": key, "material_class": spec["material_class"],
                     "mechanism_summary": spec["mechanism_summary"]})
        option_rows.append(item)
    ceramic = pd.read_csv(HIST / "mpz_v9_11_parameters/ceramic/spatial_promotion_manifest.csv").iloc[0].to_dict()
    ceramic.update({"registry_role": "ceramic_primary", "material_class": "ceramic"})
    option_rows.append(ceramic)
    all_rows = pd.concat([canonical, pd.DataFrame(option_rows)], ignore_index=True, sort=False)
    all_rows["lineage"] = np.where(
        all_rows.registry_role.str.startswith("CANONICAL_"),
        "CURRENT_V913_PERSISTENT_SPATIAL_SELECTION",
        "LEGACY_V9_11_FINITE_SOURCE",
    )
    all_rows["row_sha256"] = [
        canonical_sha({k: None if pd.isna(v) else v for k, v in row.items()
                       if k not in {"row_sha256"}})
        for row in all_rows.to_dict("records")
    ]
    return all_rows


def source_audit(rows: pd.DataFrame) -> None:
    current_sources = [
        ROOT / "CODEX_HANDOFF.md", ROOT / "ANALYTICAL_STEADY_STATE_EQUATION_LINEAGE.md",
        ROOT / "arrhenius_fracture/material_manifest.py",
        ROOT / "arrhenius_fracture/analytical_stationary_fatigue_v10230.py",
        ROOT / "arrhenius_fracture/analytical_monotonic_fracture_v10230.py",
        ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv",
        ROOT / "runs/physical_slope_transfer_v1/physical_slope_transfer_final_decision.json",
    ]
    historical_sources = [
        HIST / "mpz_v9_13_zero_d_large_search_policy.json",
        HIST / "mpz_v9_11_response_options.json", HIST / "mpz_v9_11_dbtt_option_rows.csv",
        HIST / "analysis_outputs/oneD_four_class_audit/v2/oneD_four_class_temperature_summary_v2.csv",
        HIST / "analysis_outputs/oneD_v2_terminal_predictive_program/oneD_v2_final_four_class_results.csv",
        HIST / "analysis_outputs/oneD_v2_terminal_predictive_program/oneD_v2_pf_transfer_results.csv",
        HIST / "analysis_outputs/oneD_v2_taylor_peierls_spatial_transfer/pf_2d_spatial_transfer_onsets.csv",
        HIST / "analysis_outputs/oneD_v2_peak_dbtt_rcurve_search/pf_2d_peak_dbtt_R_onset_candidates.csv",
        HIST / "analysis_outputs/oneD_v2_taylor_peierls_rcurve_search/pf_2d_taylor_peierls_onset_candidates.csv",
    ]
    sources = []
    for path in current_sources + historical_sources:
        sources.append({
            "path": str(path), "available": path.exists(),
            "sha256": sha(path) if path.exists() else None,
            "repository": "PF-fracture-fatigue" if str(path).startswith(str(ROOT)) else "Arrhenius_FEM_CZM_MPZ",
            "branch": git(ROOT, "branch", "--show-current") if str(path).startswith(str(ROOT)) else "codex/taylor-peierls-spatial-coupling-paper-audit",
            "HEAD": git(ROOT, "rev-parse", "HEAD") if str(path).startswith(str(ROOT)) else HIST_HEAD,
        })
    for name in DOC_NAMES:
        sources.append({
            "path": name, "available": False, "sha256": None, "repository": "USER_DOCUMENT",
            "branch": None, "HEAD": None,
            "omission_explanation": "Exact filename absent after recursive project, home, CloudStorage, attachment, and Spotlight searches; equations were not reconstructed from selected rows.",
        })
    manifest = {
        "schema": "fracture_source_manifest_v2", "created_utc": now(),
        "current_repository": str(ROOT), "current_branch": git(ROOT, "branch", "--show-current"),
        "current_HEAD": git(ROOT, "rev-parse", "HEAD"),
        "qualified_solver_HEAD": QUALIFIED_HEAD, "qualified_solver_sha256": SOLVER_SHA,
        "historical_repository": str(HIST), "historical_remote": git(HIST, "remote", "get-url", "origin"),
        "historical_branch": "codex/taylor-peierls-spatial-coupling-paper-audit",
        "historical_HEAD": HIST_HEAD, "sources": sources,
        "missing_document_policy": "FAIL_CLOSED_DOCUMENT_UNAVAILABLE_USE_CODE_AND_HASHED_ARCHIVE_ONLY",
        "document_ingestion_complete": False,
    }
    write_json("fracture_source_manifest.json", manifest)
    write_json("source_hashes.json", {x["path"]: x["sha256"] for x in sources})
    rows.to_csv(OUT / "candidate_cross_lineage_registry.csv", index=False)
    lineages = [
        ("CURRENT_V10_2_30_PERSISTENT_SITE", "persistent sites; signed mobile/retained; no source refresh", True, False),
        ("LEGACY_V9_11_FINITE_SOURCE", "finite source capacity and crack-advance refresh", False, True),
        ("LEGACY_STORED_ENERGY_ABLATION", "stored-energy cleavage lowering ablation", False, True),
        ("REDUCED_PRESCRIBED_STATE", "archived prescribed reduced state", False, False),
        ("ONE_D_EVOLVING_MPZ", "v9.13 moving spatial persistent-GND state", False, False),
        ("PF_SHARP_FRONT", "2-D PF/sharp-front local mechanical transfer", True, False),
        ("FEM_CZM", "2-D FEM/CZM J-derived mechanical transfer", False, False),
    ]
    write_csv("constitutive_lineage_matrix.csv", [
        {"lineage": a, "semantics": b, "used_in_current_analytical_model": c,
         "legacy_inactive_controls_present": d} for a, b, c, d in lineages
    ])
    equations = [
        ("EXP_FLOOR", "arrhenius_fracture/material_manifest.py", "CURRENT_SOURCE", True),
        ("COOPERATIVE_GAMMA_RENEWAL", "arrhenius_fracture/analytical_monotonic_fracture_v10230.py", "CURRENT_SOURCE", True),
        ("F0_FIRST_PASSAGE", "mission phase 4 plus current source", "DERIVED_NO_FIT", True),
        ("F1_TRANSIENT_BLUNTING", "persistent_site_reversible_transport_v10230.py", "CURRENT_SOURCE_MOMENT", True),
        ("F2_SIGNED_TRANSPORT", "persistent_site_reversible_transport_v10230.py; unified_mpz.py", "CURRENT_SOURCE_MOMENT", True),
        ("FINITE_SOURCE_REFRESH", "v9.11 documentation", "LEGACY_INACTIVE", False),
        ("STORED_ENERGY_CLEAVAGE_LOWERING", "historical ablation", "LEGACY_INACTIVE", False),
        ("EMPIRICAL_TOUGHNESS_OR_PARIS", "none", "PROHIBITED_ABSENT", False),
    ]
    write_csv("equation_source_catalog.csv", [
        {"equation": a, "source": b, "classification": c, "active": d}
        for a, b, c, d in equations
    ])


def prospective_freeze(rows: pd.DataFrame) -> None:
    controls = MonotonicControls(dK_MPa_sqrt_m=0.1)
    payload = {
        "schema": "fracture_analytical_input_manifest_v1", "created_utc": now(),
        "source_HEAD": SOURCE_HEAD, "qualified_solver_HEAD": QUALIFIED_HEAD,
        "solver_sha256": SOLVER_SHA, "historical_HEAD": HIST_HEAD,
        "monotonic_model_id": MONOTONIC_MODEL_ID, "fatigue_model_id": FATIGUE_MODEL_ID,
        "candidate_row_hashes": dict(zip(rows.registry_role, rows.row_sha256)),
        "equations_frozen": ["bounded_EXP_floor", "exact_gamma_renewal", "transient_first_passage",
                              "persistent_emission_blunting", "signed_transport_moments"],
        "inactive_legacy_controls": ["finite_source_inventory", "source_refresh",
                                      "stored_energy_cleavage_lowering", "arbitrary_recovery"],
        "temperature_grid_K": TEMPERATURES.tolist(), "loading_rate_grid": RATES.tolist(),
        "controls": asdict(controls), "K_init_fit_performed": False,
        "state_only_calibration": False, "descriptive_error_bands": {
            "HIGH_ACCURACY": 0.05, "USEFUL": 0.10, "QUALITATIVE": 0.20,
        },
        "validation_splits": ["leave_one_temperature_out", "leave_one_rate_out",
                               "leave_one_response_class_out", "reduced_to_1D", "1D_to_PF",
                               "1D_to_FEM", "broad_class_vs_peak"],
        "prospective_blinding_note": "Source audit necessarily exposed archived onset fields before equation-freeze; no coefficient was fitted or state closure calibrated to them.",
    }
    write_json("fracture_analytical_input_manifest.json", payload)
    write_json("fracture_analytical_controls.json", asdict(controls))
    write_json("fracture_candidate_hashes.json", dict(zip(rows.registry_role, rows.row_sha256)))
    write_json("fracture_validation_plan.json", {
        "frozen": True, "K_init_fit_allowed": False,
        "training_subset": "NONE_NO_STATE_CALIBRATION", "held_out_tests": payload["validation_splits"],
        "accuracy_bands": payload["descriptive_error_bands"],
        "F2B_activation_gate": "activate only if F2 median state error >25% and F2 K_init error exceeds F1 by >2 percentage points",
    })
    (OUT / "fracture_analytical_equation_lineage.md").write_text(
        r"""# Analytical fracture and fatigue equation lineage

## Scope and source limitation

This is an analysis-only reduction of the current v10.2.30 implementation. The four
named DOCX sources were not available in the mounted worktree or attachment area and
are therefore not claimed as ingested. Document-only equations or interpretations are
not reconstructed from selected response rows. The executable code and hashed archive
listed in `fracture_source_manifest.json` are the auditable sources for this pass.

## Barrier and activated rate

For channel \(j\), the bounded EXP-floor barrier is

\[G_{0j}(T)=\max(G_{00,j}+g_{T,j}(T-T_{ref}),10^{-12})\;\mathrm{eV},\]
\[\sigma_{cj}(T)=\max(\sigma_{c0,j}+s_{T,j}(T-T_{ref}),1)\;\mathrm{Pa},\]
\[G_{fj}=\min(0.95G_{0j},\max(10^{-4}\;\mathrm{eV},f_jG_{0j})),\]
\[G_j(\sigma,T)=G_{fj}+(G_{0j}-G_{fj})
\exp[-a_j(\max(\sigma,0)/\sigma_{cj})^{n_j}],\]
\[\lambda_{j,raw}=\nu_j\exp[-G_j/(k_BT)]\;\mathrm{s^{-1}}.\]

Source: `arrhenius_fracture/material_manifest.py`, `ExpFloorBarrier.values_eV`
and `ExpFloorBarrier.rate`. Inputs use eV, K, Pa, and s\(^{-1}\).

## Cooperative cleavage renewal and saturation

With \(m_c=3\), \(\tau_c=10^{-6}\) s, and
\(x_c=\lambda_{c,raw}\tau_c\),

\[\Lambda_c=P(m_c,x_c)/\tau_c,\qquad 0\leq\Lambda_c\leq1/\tau_c.\]

Source: `arrhenius_fracture/analytical_monotonic_fracture_v10230.py`,
`cooperative_rate`. The stationary fatigue reduction integrates this rate over
the declared sinusoidal cycle. Its event-rate prediction is

\[da/dN=\bar\ell\langle\Lambda_c\rangle/f,\]

so \((da/dN)_{max}=\bar\ell/(f\tau_c)=5\times10^{-3}\) m/cycle for
\(\bar\ell=5\times10^{-6}\) m and \(f=1000\) Hz. This V2 atlas records
\(x_c\), \(\Lambda_c\tau_c\), the normalized \(da/dN\), and phase fractions
near the renewal ceiling, barrier floor, stress cap, and zero activity.

## Separate local stress channels

\[\sigma_{open}=K_{loc}/\sqrt{2\pi r_{eff}},\]
\[\sigma_{cleave}=\max(K_{loc}-K_{shield},0)/\sqrt{2\pi r_{eff}},\]
\[\sigma_{emit}=\max(w_e\sigma_{open}-\sigma_{back},0).\]

Source: `analytical_monotonic_fracture_v10230.py`, `stress_channels`. The
channels are never collapsed. Bulk PF/FEM redistribution is not added again as
explicit shielding.

## F0, F1, F2, and F2B

F0 integrates virgin cleavage action during the prescribed monotonic ramp until
\(\int\Lambda_c\,dt=1\), or right-censors at the declared maximum K. F1 adds a
pre-event persistent emission/blunting moment \(q\):

\[\dot q=R_e-k_bq,\qquad r_{eff}=r_0+c_{blunt}bq.\]

F2 adds nonnegative mobile and retained moments:

\[\dot M=R_e-(k_{enc}+k_{esc})M+k_TM_R,\]
\[\dot M_R=k_{enc}M-(k_T+k_{rec})M_R,\]

with Peierls transport, encounter, Taylor release, escape, recovery, back stress,
and signed retained shielding. F2B splits the same moments into near-tip and
outer-MPZ compartments with one tested exchange rate. Source:
`analytical_monotonic_fracture_v10230.py`, `_rhs`,
`_linear_positive_advance`, and `solve_first_passage`.

Initial conditions are virgin zero reduced populations and zero cleavage action.
Integration uses fixed K increments, positivity-preserving matrix exponentials
for linear compartment evolution, trapezoidal hazard action, and within-step
first-passage localization. Every reported state is pre-event/pre-translation;
no crack-advance translation occurs before first passage.

## Mechanical-transfer assumption and validation domains

F0/F1/F2 consume front-local K. Matched evolving-1D comparisons test the local
constitutive kernel. PF front-local J/K comparisons additionally test the PF
mechanical mapping. FEM/CZM applied-load comparisons require an independently
qualified mapping \(\mathcal M_{FEM}:K_{applied}\mapsto K_{local}\); V2 does not
fit or assume that map. Constitutive and mechanical-transfer errors are reported
separately in `monotonic_common_population_error.csv` and
`mechanical_transfer_decomposition.csv`.

## Inactive lineages and censor logic

Legacy finite source inventories, crack-advance refresh, stored-energy cleavage
lowering, empirical toughness laws, and Paris laws are inactive. F1/F2/F2B
nonconvergence or failure to reach unit action within the frozen K domain is a
right-censored analytical result, not a ductile law, toughness value, or physical
failure datum. The tested F2B closure did not improve the result; this does not
reject the broader class of two-compartment closures.
"""
    )


def exact_predictions(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, states, tderiv, rderiv, jacobian = [], [], [], [], []
    controls = MonotonicControls(dK_MPa_sqrt_m=0.1, Kmax_MPa_sqrt_m=100.0)
    sensitivity_controls = replace(controls, dK_MPa_sqrt_m=0.01)
    diagnostics = rows[rows.registry_role.isin([
        "CANONICAL_PEAK", "CANONICAL_DBTT", "CANONICAL_WEAK_T", "CANONICAL_CERAMIC",
        "dbtt_primary", "peak_primary", "dbtt_broad_shielding",
        "dbtt_intrinsic_control", "dbtt_moderate_shielding_reference",
        "weakT_primary", "ceramic_primary",
    ])]
    for _, row in diagnostics.iterrows():
        item = row.to_dict()
        manifest = material_from_row(item)
        for T in TEMPERATURES:
            level_results = {}
            for level in ("F0", "F1", "F2"):
                result = solve_first_passage(manifest, item, float(T), level, controls)
                level_results[level] = result
                predictions.append({
                    "registry_role": item["registry_role"], "candidate_id": item["candidate_id"],
                    "lineage": item["lineage"], "temperature_K": T, "level": level,
                    "K_init_MPa_sqrt_m": result["K_init_MPa_sqrt_m"],
                    "first_passage_reached": result["first_passage_reached"],
                    "right_censored": result["right_censored"],
                    "cleavage_action": result["cleavage_action"],
                    "emission_action": result["emission_action"],
                    "endpoint_K_init_MPa_sqrt_m": result["endpoint_K_init_MPa_sqrt_m"],
                    "endpoint_relative_error": result["endpoint_relative_error"],
                    "hazard_localization": result["hazard_localization"],
                })
                state = result["pre_event_state"]
                states.append({
                    "registry_role": item["registry_role"], "candidate_id": item["candidate_id"],
                    "temperature_K": T, "level": level, "state_semantics": "PRE_EVENT_PRE_TRANSLATION",
                    **{k: v for k, v in state.items() if isinstance(v, (int, float, np.number))},
                })
            if level_results["F0"]["first_passage_reached"]:
                tangent = f0_implicit_sensitivities(manifest, item, float(T), sensitivity_controls)
                dT = finite_difference_sensitivity(manifest, item, float(T), "F0", sensitivity_controls,
                                                   "temperature_K", 2e-3)
                dr = finite_difference_sensitivity(manifest, item, float(T), "F0", sensitivity_controls,
                                                   "loading_rate_MPa_sqrt_m_s", 2e-3)
                tderiv.append({
                    "registry_role": item["registry_role"], "candidate_id": item["candidate_id"],
                    "temperature_K": T, "level": "F0", "total_dKinit_dT": tangent["temperature_K"],
                    "intrinsic_free_energy_contribution": tangent["temperature_K"],
                    "cooperative_renewal_contribution": 0.0,
                    "radius_contribution": 0.0, "shielding_contribution": 0.0,
                    "transport_contribution": 0.0, "component_sum": tangent["temperature_K"],
                    "finite_difference": dT,
                    "relative_closure_error": abs(tangent["temperature_K"]-dT)/max(abs(dT),1e-30),
                })
                rderiv.append({
                    "registry_role": item["registry_role"], "candidate_id": item["candidate_id"],
                    "temperature_K": T, "level": "F0", "dKinit_dlnKdot": tangent["ln_loading_rate"],
                    "opening_action_contribution": tangent["ln_loading_rate"],
                    "state_contribution": 0.0, "finite_difference": dr * sensitivity_controls.loading_rate_MPa_sqrt_m_s,
                })
                if item["registry_role"].startswith("CANONICAL_") and T in {300.0, 900.0}:
                    steps = {
                        "cleave_G00_eV": 1e-4,
                        "cleave_gT_eV_per_K": 1e-7,
                        "cleave_sigc0_GPa": 1e-4,
                        "cleave_sT_GPa_per_K": 1e-7,
                        "cleave_exp_a": 1e-5,
                        "cleave_exp_n": 1e-5,
                        "cleave_floor_frac": 1e-6,
                    }
                    attr = {
                        "cleave_G00_eV":"G00_eV", "cleave_gT_eV_per_K":"gT_eV_per_K",
                        "cleave_sigc0_GPa":"sigc0_Pa", "cleave_sT_GPa_per_K":"sT_Pa_per_K",
                        "cleave_exp_a":"alpha", "cleave_exp_n":"exponent",
                        "cleave_floor_frac":"floor_fraction",
                    }
                    for parameter, step in steps.items():
                        key = attr[parameter]
                        base = getattr(manifest.cleavage, key)
                        native_step = step*1e9 if parameter in {"cleave_sigc0_GPa","cleave_sT_GPa_per_K"} else step
                        lo_m = replace(manifest, cleavage=replace(manifest.cleavage, **{key:base-native_step}))
                        hi_m = replace(manifest, cleavage=replace(manifest.cleavage, **{key:base+native_step}))
                        lo = solve_first_passage(lo_m,item,float(T),"F0",sensitivity_controls)["K_init_MPa_sqrt_m"]
                        hi = solve_first_passage(hi_m,item,float(T),"F0",sensitivity_controls)["K_init_MPa_sqrt_m"]
                        fd = (hi-lo)/(2*step)
                        jacobian.append({
                            "registry_role":item["registry_role"],"candidate_id":item["candidate_id"],
                            "temperature_K":T,"parameter":parameter,"tangent_derivative":tangent[parameter],
                            "centered_finite_difference":fd,
                            "relative_error":abs(tangent[parameter]-fd)/max(abs(fd),1e-30),
                        })
    pred = pd.DataFrame(predictions)
    state = pd.DataFrame(states)
    td = pd.DataFrame(tderiv)
    rd = pd.DataFrame(rderiv)
    for level in ("F0", "F1", "F2"):
        pred[pred.level == level].to_parquet(OUT / f"monotonic_{level}_predictions.parquet", index=False)
    state.to_parquet(OUT / "monotonic_state_validation.parquet", index=False)
    td.to_parquet(OUT / "temperature_sensitivity_decomposition.parquet", index=False)
    rd.to_parquet(OUT / "loading_rate_sensitivity_decomposition.parquet", index=False)
    jac = pd.DataFrame(jacobian)
    jac.to_parquet(OUT / "parameter_sensitivity_jacobian.parquet", index=False)
    return pred, state, td, rd


def archived_inventory() -> pd.DataFrame:
    specs = [
        ("analysis_outputs/oneD_four_class_audit/v2/oneD_four_class_temperature_summary_v2.csv",
         "ONE_D_EVOLVING_MPZ", "initial_onset_native_K_MPa_sqrt_m", "AUTONOMOUS_V913_ONE_D_DRIVER"),
        ("analysis_outputs/oneD_v2_terminal_predictive_program/oneD_v2_final_four_class_results.csv",
         "PF_OR_FEM_REDUCED_TRANSFER", "first_event_native_KJ_MPa_sqrt_m", "provider"),
        ("analysis_outputs/oneD_v2_taylor_peierls_spatial_transfer/pf_2d_spatial_transfer_onsets.csv",
         "PF_SHARP_FRONT", "pre_event_native_KJ_MPa_sqrt_m", "PF_2D_SPATIAL_TRANSFER"),
        ("analysis_outputs/oneD_v2_peak_dbtt_rcurve_search/pf_2d_peak_dbtt_R_onset_candidates.csv",
         "PF_SHARP_FRONT", "pre_event_native_KJ_MPa_sqrt_m", "PF_2D_PEAK_DBTT_R"),
        ("analysis_outputs/oneD_v2_taylor_peierls_rcurve_search/pf_2d_taylor_peierls_onset_candidates.csv",
         "PF_SHARP_FRONT", "pre_event_native_KJ_MPa_sqrt_m", "PF_2D_TP"),
    ]
    records = []
    for rel, default_fidelity, kfield, provider_field in specs:
        path = HIST / rel
        frame = pd.read_csv(path)
        if "event_transaction_index" in frame:
            frame = frame[frame.event_transaction_index == 0]
        for index, row in frame.iterrows():
            candidate = str(row.get("candidate_id", "UNKNOWN"))
            provider = str(row.get(provider_field, provider_field))
            fidelity = provider if provider in {"PF", "FEMCZM", "ONE_D"} else default_fidelity
            Kinit = _f(row, kfield, math.nan)
            right = bool(row.get("right_censored", row.get("right_censored_at_target", False)))
            status = str(row.get("status", "FIRST_PASSAGE_INITIATION"))
            terminal = "FIRST_PASSAGE_INITIATION" if math.isfinite(Kinit) else "RIGHT_CENSORED"
            if "NUMERICAL" in status.upper() or "FAILED" in status.upper():
                terminal = "NUMERICAL_FAILURE"
            key = canonical_sha({"candidate": candidate, "T": _f(row, "temperature_K", math.nan),
                                 "fidelity": fidelity, "K": Kinit, "source": rel, "row": int(index)})[:20]
            records.append({
                "inventory_id": key, "candidate_id": candidate,
                "material_class": str(row.get("material_class", "UNKNOWN")),
                "temperature_K": _f(row, "temperature_K", math.nan),
                "loading_rate_factor": _f(row, "loading_rate_factor", 1.0),
                "applied_loading_history": "ARCHIVED_MONOTONIC_RAMP",
                "crystal_orientation": str(row.get("theta_deg", "theta0_or_archived")),
                "mechanical_geometry": str(row.get("provider", provider)), "solver_fidelity": fidelity,
                "K_init_MPa_sqrt_m": Kinit, "first_event_time_s": _f(row, "pre_event_time_s", math.nan),
                "completion_state": terminal, "right_censored": right,
                "pre_event_r_eff_um": _f(row, "pre_event_tip_radius_um", math.nan),
                "pre_event_K_shield_MPa_sqrt_m": _f(row, "pre_event_K_shield_MPa_sqrt_m",
                                                    _f(row, "pre_event_signed_shielding_MPa_sqrt_m", math.nan)),
                "pre_event_backstress_GPa": _f(row, "pre_event_backstress_GPa", math.nan),
                "pre_event_mobile": _f(row, "pre_event_mobile_count", math.nan),
                "pre_event_retained": _f(row, "pre_event_retained_count", math.nan),
                "cleavage_action": _f(row, "cleavage_action", 1.0 if math.isfinite(Kinit) else math.nan),
                "emission_action": _f(row, "emission_action", math.nan),
                "event_topology": str(row.get("physical_avalanche_status", "FIRST_EVENT")),
                "source_file": str(path), "source_sha256": sha(path), "source_row": int(index),
                "lineage": "CURRENT_V913_PERSISTENT_SPATIAL_SELECTION",
                "physical_admission": terminal == "FIRST_PASSAGE_INITIATION",
            })
    # Inventory the current fatigue evidence separately and never reinterpret it as monotonic Kinit.
    fatigue_files = [
        ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_developed_fatigue_points.csv",
        ROOT / "runs/A_native_PT03_PT08_R_nominal_deltaK_v1/A_PT03_PT08_R_developed_points.csv",
        ROOT / "runs/physical_slope_transfer_v1/corrected_candidate_physical_points.csv",
    ]
    for path in fatigue_files:
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            candidate = str(row.get("parameter_option", row.get("candidate_id", row.get("option_key", "UNKNOWN"))))
            records.append({
                "inventory_id": canonical_sha({"path": str(path), "row": int(index)})[:20],
                "candidate_id": candidate, "material_class": "FATIGUE_MATCHING_ROW",
                "temperature_K": _f(row, "temperature_K", 300.0),
                "loading_rate_factor": _f(row, "frequency_Hz", 1000.0),
                "applied_loading_history": "FIXED_LOCAL_DELTAK_CYCLIC",
                "crystal_orientation": "theta0", "mechanical_geometry": "1D_SHARP_FRONT",
                "solver_fidelity": "ONE_D_FATIGUE", "K_init_MPa_sqrt_m": math.nan,
                "first_event_time_s": math.nan, "completion_state": "POST_INITIATION_PROPAGATION",
                "right_censored": bool(row.get("censored", False)), "pre_event_r_eff_um": math.nan,
                "pre_event_K_shield_MPa_sqrt_m": math.nan, "pre_event_backstress_GPa": math.nan,
                "pre_event_mobile": math.nan, "pre_event_retained": math.nan,
                "cleavage_action": math.nan, "emission_action": math.nan,
                "event_topology": "DEVELOPED_FATIGUE", "source_file": str(path),
                "source_sha256": sha(path), "source_row": int(index),
                "lineage": "CURRENT_V10_2_30_PERSISTENT_SITE", "physical_admission": True,
            })
    inventory = pd.DataFrame(records)
    if inventory.inventory_id.duplicated().any():
        raise RuntimeError("duplicate physical inventory id")
    inventory.to_csv(OUT / "fracture_result_inventory.csv", index=False)
    semantics = []
    for variable, timing, spatial, sign, population in [
        ("K_init_MPa_sqrt_m", "first accepted event", "front-local", "unsigned", "drive"),
        ("pre_event_r_eff_um", "pre-event pre-translation", "near-tip moment", "unsigned", "blunting"),
        ("pre_event_K_shield_MPa_sqrt_m", "pre-event pre-translation", "active MPZ", "signed", "retained-linked"),
        ("pre_event_backstress_GPa", "pre-event pre-translation", "source channel", "signed", "retained/local"),
        ("pre_event_mobile", "pre-event pre-translation", "full active MPZ", "signed where available", "mobile"),
        ("pre_event_retained", "pre-event pre-translation", "full active MPZ", "signed where available", "retained"),
        ("cleavage_action", "at first passage", "front-local", "unsigned", "hazard ledger"),
        ("emission_action", "at first passage", "source-local", "unsigned", "emission ledger"),
    ]:
        semantics.append({"variable": variable, "event_timing": timing, "spatial_semantics": spatial,
                          "signedness": sign, "population_semantics": population,
                          "comparison_policy": "MATCH_EXACT_SEMANTICS_OR_MARK_UNAVAILABLE"})
    write_csv("fracture_state_semantics.csv", semantics)
    return inventory


def cross_fidelity_validation(inventory: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    canonical = {
        "v913_zeroD_sobol_0242980": "CANONICAL_PEAK",
        "v913_zeroD_sobol_0202500": "CANONICAL_DBTT",
        "v913_zeroD_sobol_0129902": "CANONICAL_WEAK_T",
        "v913_zeroD_sobol_0077080": "CANONICAL_CERAMIC",
    }
    physical = inventory[
        inventory.candidate_id.isin(canonical) & inventory.physical_admission
        & inventory.K_init_MPa_sqrt_m.notna()
    ].copy()
    physical["registry_role"] = physical.candidate_id.map(canonical)
    pred = predictions.pivot_table(
        index=["registry_role", "candidate_id", "temperature_K"], columns="level",
        values="K_init_MPa_sqrt_m", aggfunc="first"
    ).reset_index()
    merged = physical.merge(pred, on=["registry_role", "candidate_id", "temperature_K"], how="left")
    for level in ("F0", "F1", "F2"):
        merged[f"relative_error_{level}"] = (merged[level] - merged.K_init_MPa_sqrt_m) / merged.K_init_MPa_sqrt_m
        error = merged[f"relative_error_{level}"].abs()
        merged[f"accuracy_{level}"] = np.select(
            [error <= .05, error <= .10, error <= .20],
            ["HIGH_ACCURACY", "USEFUL", "QUALITATIVE"], default="INSUFFICIENT"
        )
    merged["event_role"] = "FIRST_PASSAGE_INITIATION"
    merged["mechanical_transfer_residual_F2"] = merged.K_init_MPa_sqrt_m - merged.F2
    merged["comparison_domain"] = merged.solver_fidelity.map({
        "ONE_D_EVOLVING_MPZ": "MATCHED_LOCAL_1D",
        "PF": "PF_FRONT_LOCAL_J_DERIVED",
        "PF_SHARP_FRONT": "PF_SHARP_FRONT_MIXED_MECHANICAL_TRANSFER",
        "FEMCZM": "FEM_APPLIED_LOAD_REQUIRES_TRANSFER",
    }).fillna("UNCLASSIFIED_MECHANICAL_DOMAIN")
    merged["local_driving_force_equivalence_verified"] = (
        merged.comparison_domain == "MATCHED_LOCAL_1D"
    )
    merged["FEM_transfer_map_fitted"] = False
    merged.to_csv(OUT / "monotonic_cross_fidelity_validation.csv", index=False)

    common = merged.dropna(subset=["F0", "F1", "F2"]).copy()
    common_rows = []
    for domain, group in [("ALL_COMMON_FINITE", common), *common.groupby("comparison_domain")]:
        for level in ("F0", "F1", "F2"):
            error = group[f"relative_error_{level}"].abs()
            common_rows.append({
                "comparison_domain": domain,
                "level": level,
                "common_finite_count": int(len(group)),
                "median_absolute_relative_error": float(error.median()) if len(error) else math.nan,
                "mean_absolute_relative_error": float(error.mean()) if len(error) else math.nan,
                "population_policy": "SAME_ROWS_FINITE_FOR_F0_F1_F2",
            })
    write_csv("monotonic_common_population_error.csv", common_rows)

    transfer_rows = []
    for domain, group in merged.groupby("comparison_domain"):
        for level in ("F0", "F1", "F2"):
            finite = group.dropna(subset=[level, "K_init_MPa_sqrt_m"])
            ratio = finite.K_init_MPa_sqrt_m / finite[level]
            transfer_rows.append({
                "comparison_domain": domain,
                "solver_fidelity": ";".join(sorted(group.solver_fidelity.astype(str).unique())),
                "analytical_level": level,
                "finite_count": int(len(finite)),
                "archived_to_local_K_ratio_median": float(ratio.median()) if len(ratio) else math.nan,
                "median_absolute_relative_error": float(finite[f"relative_error_{level}"].abs().median()) if len(finite) else math.nan,
                "constitutive_kernel_test": domain == "MATCHED_LOCAL_1D",
                "mechanical_transfer_test": domain != "MATCHED_LOCAL_1D",
                "local_driving_force_equivalence_verified": domain == "MATCHED_LOCAL_1D",
                "transfer_map_status": (
                    "IDENTITY_BY_MATCHED_1D_PROTOCOL" if domain == "MATCHED_LOCAL_1D"
                    else "UNRESOLVED_NO_TRANSFER_MAP_FIT"
                ),
            })
    write_csv("mechanical_transfer_decomposition.csv", transfer_rows)
    write_json("mechanical_transfer_assumptions.json", {
        "schema": "mechanical_transfer_assumptions_v1",
        "analytical_input": "FRONT_LOCAL_K",
        "matched_1D_identity_assumption": True,
        "PF_front_local_J_or_K_requires_provider_specific_audit": True,
        "FEM_applied_to_local_mapping_qualified": False,
        "FEM_transfer_map_fitted": False,
        "no_nominal_K_equivalence_assumed_for_FEM": True,
        "conclusion": "LOCAL_KERNEL_VALIDATED_FOR_MATCHED_1D__CROSS_FIDELITY_MECHANICAL_TRANSFER_UNRESOLVED",
    })
    return merged


def search_bounds() -> dict[str, Any]:
    policy_path = HIST / "mpz_v9_13_zero_d_large_search_policy.json"
    policy = json.loads(policy_path.read_text())
    fixed = {
        "Tref_K": 481.33, "peierls_nu0_s": 1e12, "taylor_nu0_s": 1e11,
        "source_sites_per_system": "COMMON_PHYSICS", "encounter_efficiency": "COMMON_PHYSICS",
        "retained_recovery_rate_s": "COMMON_PHYSICS", "blunting_length_m": 0.5e-6,
        "mpz_length_m": 50e-6, "source_zone_length_m": 2e-6, "return_eligibility": False,
    }
    payload = {
        "schema": "historical_search_bounds_v1", "continuation_of_historical_search": True,
        "source_path": str(policy_path), "source_sha256": sha(policy_path),
        "source_HEAD": HIST_HEAD, "sampling": "scrambled_sobol_anchor_local_plus_global",
        "historical_search_dimensions": policy["search_dimensions"],
        "historical_anchor_candidate_ids": policy["anchor_candidate_ids"],
        "local_anchor_fraction": policy["local_anchor_fraction"], "fixed_current_coordinates": fixed,
        "rejection_rules": ["positive barrier surfaces over T/stress domain",
                            "finite characteristic stress", "no barrier-floor-only archetype",
                            "no numerical or controller coordinates"],
    }
    write_json("historical_search_bounds.json", payload)
    return payload


def _transform_sobol(u: np.ndarray, specs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    data = {}
    for j, (name, spec) in enumerate(specs.items()):
        low, high = float(spec["low"]), float(spec["high"])
        if "log10" in str(spec["mode"]):
            data[name] = np.power(10.0, np.log10(low) + u[:, j] * (np.log10(high)-np.log10(low)))
        else:
            data[name] = low + u[:, j] * (high-low)
    return pd.DataFrame(data)


def _surface_components(
    frame: pd.DataFrame, prefix: str, stress_Pa: np.ndarray, T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stress = np.asarray(stress_Pa, dtype=float)
    expand = (slice(None),) + (None,) * max(stress.ndim - 1, 0)
    G0 = np.maximum(
        frame[f"{prefix}_G00_eV"].to_numpy()[expand]
        + frame[f"{prefix}_gT_eV_per_K"].to_numpy()[expand] * (T - 481.33),
        1e-12,
    )
    floor = np.minimum(
        0.95 * G0,
        np.maximum(1e-4, frame[f"{prefix}_floor_frac"].to_numpy()[expand] * G0),
    )
    sigc = np.maximum(
        (
            frame[f"{prefix}_sigc0_GPa"].to_numpy()[expand]
            + frame[f"{prefix}_sT_GPa_per_K"].to_numpy()[expand] * (T - 481.33)
        ) * 1e9,
        1.0,
    )
    alpha = np.maximum(frame[f"{prefix}_exp_a"].to_numpy()[expand], 0.0)
    exponent = np.maximum(frame[f"{prefix}_exp_n"].to_numpy()[expand], 1e-9)
    G = floor + (G0 - floor) * np.exp(
        -alpha * np.power(np.maximum(stress, 0.0) / sigc, exponent)
    )
    return G, G0, floor, sigc


def _surface(frame: pd.DataFrame, prefix: str, stress_Pa: np.ndarray, T: float) -> np.ndarray:
    return _surface_components(frame, prefix, stress_Pa, T)[0]


def _atlas_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    r0, b, kdot = 1e-6, 2.74e-10, 0.02
    Kinit = np.full((n, len(TEMPERATURES)), np.nan)
    emit_action = np.zeros_like(Kinit)
    blunt = np.zeros_like(Kinit)
    shield_frac = np.zeros_like(Kinit)
    artifact = np.zeros(n, dtype=bool)
    Kgrid = np.linspace(0.02, 100.0, 180)
    for start in range(0, n, 4096):
        stop = min(start+4096, n)
        sub = frame.iloc[start:stop]
        for ti, T in enumerate(TEMPERATURES):
            cumulative = np.zeros(stop-start)
            found = np.zeros(stop-start, dtype=bool)
            kval = np.full(stop-start, np.nan)
            prev_rate = np.zeros(stop-start)
            for j, K in enumerate(Kgrid):
                stress = K*1e6 / math.sqrt(2*math.pi*r0)
                G = _surface(sub, "cleave", np.full(stop-start, stress), float(T))
                raw = 1e12*np.exp(np.clip(-G/(KB_EV_PER_K*T), -700, 0))
                rate = gammainc(3.0, np.minimum(raw*1e-6, 1e12))/1e-6
                if j:
                    cumulative += .5*(rate+prev_rate)*(K-Kgrid[j-1])/kdot
                new = (~found) & (cumulative >= 1.0)
                kval[new] = K
                found |= new
                prev_rate = rate
            Kinit[start:stop, ti] = kval
            safeK = np.where(np.isfinite(kval), kval, 100.0)
            sigma = safeK*1e6/math.sqrt(2*math.pi*r0)
            Ge = _surface(sub, "emit", sigma, float(T))
            eraw = 1e11*np.exp(np.clip(-Ge/(KB_EV_PER_K*T), -700, 0))
            multiplicity = sub.rho_source0_m2.to_numpy()*25e-12
            ea = 2*multiplicity*eraw*(safeK/kdot)
            emit_action[start:stop, ti] = ea
            peierls_G = np.maximum(sub.peierls_H0_eV.to_numpy() -
                sub.peierls_activation_entropy_kB.to_numpy()*KB_EV_PER_K*T, 1e-6)
            kp = 1e12*np.exp(np.clip(-peierls_G/(KB_EV_PER_K*T), -700, 0))
            residence = 1.0/(1.0+kp*np.maximum(safeK/kdot, 1e-12))
            q = np.minimum(ea*residence, 1e12)
            br = np.maximum(sub.c_blunt.to_numpy(), 0.0)*b*q/r0
            blunt[start:stop, ti] = br
            taylor_G = np.maximum(sub.taylor_H0_eV.to_numpy() -
                sub.taylor_activation_entropy_kB.to_numpy()*KB_EV_PER_K*T, 1e-6)
            kt = 1e11*np.exp(np.clip(-taylor_G/(KB_EV_PER_K*T), -700, 0))
            retained = q/(1.0+kt*np.maximum(safeK/kdot, 1e-12))
            kernel = 160e9*b/(1-.28)/math.sqrt(2*math.pi*.5e-6)
            shield_frac[start:stop, ti] = np.clip(kernel*retained/
                np.maximum(safeK*1e6, 1.0), -2.0, 2.0)
            artifact[start:stop] |= (br > 100.0) | (np.abs(shield_frac[start:stop, ti]) > .95)
    out = frame.copy()
    for ti, T in enumerate(TEMPERATURES.astype(int)):
        out[f"Kinit_F0_T{T}"] = Kinit[:, ti]
        out[f"Pi_eo_T{T}"] = emit_action[:, ti]
        out[f"Pi_b_T{T}"] = blunt[:, ti]
        out[f"Pi_sh_T{T}"] = shield_frac[:, ti]
        out[f"Kinit_F1_T{T}"] = Kinit[:, ti]*np.sqrt(1+np.minimum(blunt[:, ti], 100.0))
        out[f"Kinit_F2_T{T}"] = out[f"Kinit_F1_T{T}"] + shield_frac[:, ti]*Kinit[:, ti]
    F2 = np.column_stack([out[f"Kinit_F2_T{int(T)}"].to_numpy() for T in TEMPERATURES])
    diff = np.diff(F2, axis=1)/np.diff(TEMPERATURES)
    variation = np.nanmax(F2, axis=1)-np.nanmin(F2, axis=1)
    peak_index = np.nanargmax(np.where(np.isfinite(F2), F2, -np.inf), axis=1)
    peak = (peak_index > 0) & (peak_index < len(TEMPERATURES)-1) & (variation > 1.0)
    dbtt = (~peak) & (np.nanmax(diff, axis=1) > .006) & ((F2[:, -1]-F2[:, 0]) > 2.0)
    weak = (~peak) & (~dbtt) & (variation <= 2.0)
    klass = np.where(peak, "PEAK_LIKE", np.where(dbtt, "DBTT_LIKE",
                    np.where(weak, "WEAK_T", "CERAMIC_LIKE")))
    mechanism = np.where(np.nanmax(blunt, axis=1) > 1.0, "BLUNTING_MEDIATED",
                np.where(np.nanmax(np.abs(shield_frac), axis=1) > .05, "RETAINED_SHIELDING",
                         "INTRINSIC_OPENING"))
    out["response_class"] = klass
    out["mechanism_class"] = mechanism
    out["temperature_range_MPa_sqrt_m"] = variation
    out["max_dKinit_dT"] = np.nanmax(diff, axis=1)
    out["min_dKinit_dT"] = np.nanmin(diff, axis=1)
    out["peak_temperature_K"] = TEMPERATURES[peak_index]
    out["peak_amplitude_MPa_sqrt_m"] = np.nanmax(F2, axis=1)-.5*(F2[:,0]+F2[:,-1])
    out["monotonic_response_class"] = klass
    monotonic_artifact = artifact | ~np.isfinite(F2).all(axis=1)

    # Current event-conditioned fatigue screening at the only physically validated
    # temperature.  Every load is integrated over the declared waveform and carries
    # explicit renewal-ceiling, barrier-floor, stress-cap, and inactivity diagnostics.
    fatigue = np.empty((n, len(K_LADDER)))
    ceiling_ratio = np.empty_like(fatigue)
    near_ceiling_phase = np.empty_like(fatigue)
    near_floor_phase = np.empty_like(fatigue)
    stress_cap_phase = np.empty_like(fatigue)
    zero_activity_phase = np.empty_like(fatigue)
    xc_peak = np.empty_like(fatigue)
    xc_mean = np.empty_like(fatigue)
    lambda_tau_peak = np.empty_like(fatigue)
    phase = 2.0 * math.pi * (np.arange(FATIGUE_PHASE_COUNT) + 0.5) / FATIGUE_PHASE_COUNT
    waveform_ratio = 0.5 * (1.0 + FATIGUE_R) + 0.5 * (1.0 - FATIGUE_R) * np.cos(phase)
    max_da_dN = 5e-6 / (FATIGUE_FREQUENCY_HZ * 1e-6)
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        sub = frame.iloc[start:stop]
        radius = r0 * (1.0 + np.minimum(blunt[start:stop, 0], 100.0))
        for ki, K in enumerate(K_LADDER):
            K_phase = K * 1e6 * waveform_ratio[None, :]
            sigma_uncapped = K_phase / np.sqrt(2.0 * math.pi * radius[:, None])
            sigma = np.minimum(sigma_uncapped, 30.0e9)
            G, G0, floor, _ = _surface_components(sub, "cleave", sigma, 300.0)
            raw = 1e12 * np.exp(np.clip(-G / (KB_EV_PER_K * 300.0), -700.0, 0.0))
            xc = np.minimum(raw * 1e-6, 1e12)
            renewal_fraction = gammainc(3.0, xc)  # Lambda_c / (1/tau_c)
            cycle_fraction = np.mean(renewal_fraction, axis=1)
            fatigue[start:stop, ki] = max_da_dN * cycle_fraction
            ceiling_ratio[start:stop, ki] = cycle_fraction
            near_ceiling_phase[start:stop, ki] = np.mean(
                renewal_fraction >= COOPERATIVE_CEILING_FRACTION, axis=1
            )
            span = np.maximum(G0 - floor, 1e-300)
            near_floor_phase[start:stop, ki] = np.mean(
                (G - floor) / span <= BARRIER_FLOOR_RELATIVE_BAND, axis=1
            )
            stress_cap_phase[start:stop, ki] = np.mean(sigma_uncapped >= 30.0e9, axis=1)
            zero_activity_phase[start:stop, ki] = np.mean(
                renewal_fraction <= ZERO_ACTIVITY_FRACTION, axis=1
            )
            xc_peak[start:stop, ki] = np.max(xc, axis=1)
            xc_mean[start:stop, ki] = np.mean(xc, axis=1)
            lambda_tau_peak[start:stop, ki] = np.max(renewal_fraction, axis=1)
    for ki, K in enumerate(K_LADDER):
        tag = f"K{K:g}_R0p1_T300_f1000"
        out[f"fatigue_da_dN_{tag}"] = fatigue[:, ki]
        out[f"fatigue_xc_peak_{tag}"] = xc_peak[:, ki]
        out[f"fatigue_xc_cycle_mean_{tag}"] = xc_mean[:, ki]
        out[f"fatigue_Lambda_c_tau_peak_{tag}"] = lambda_tau_peak[:, ki]
        out[f"fatigue_Lambda_c_tau_cycle_mean_{tag}"] = ceiling_ratio[:, ki]
        out[f"fatigue_da_dN_ceiling_fraction_{tag}"] = ceiling_ratio[:, ki]
        out[f"fatigue_phase_fraction_near_cooperative_ceiling_{tag}"] = near_ceiling_phase[:, ki]
        out[f"fatigue_phase_fraction_near_barrier_floor_{tag}"] = near_floor_phase[:, ki]
        out[f"fatigue_phase_fraction_at_stress_cap_{tag}"] = stress_cap_phase[:, ki]
        out[f"fatigue_phase_fraction_effectively_zero_{tag}"] = zero_activity_phase[:, ki]
    slopes = np.diff(np.log(np.maximum(fatigue, 1e-300)), axis=1)/np.diff(np.log(K_LADDER))
    out["fatigue_local_slope_low"] = slopes[:,0]
    out["fatigue_local_slope_mid"] = slopes[:,1]
    out["fatigue_local_slope_high"] = slopes[:,2]
    out["fatigue_high_K_flattening"] = slopes[:,0]-slopes[:,-1]
    out["fatigue_max_da_dN_m_per_cycle"] = max_da_dN
    out["fatigue_ceiling_dominated"] = (
        np.mean(ceiling_ratio >= COOPERATIVE_CEILING_FRACTION, axis=1) >= DOMINANT_PHASE_FRACTION
    )
    out["fatigue_barrier_floor_dominated"] = np.max(near_floor_phase, axis=1) >= DOMINANT_PHASE_FRACTION
    out["fatigue_stress_cap_dominated"] = np.max(stress_cap_phase, axis=1) >= DOMINANT_PHASE_FRACTION
    out["fatigue_zero_activity_dominated"] = np.mean(
        zero_activity_phase >= COOPERATIVE_CEILING_FRACTION, axis=1
    ) >= DOMINANT_PHASE_FRACTION
    out["fatigue_saturation_state"] = np.select(
        [out.fatigue_ceiling_dominated, out.fatigue_zero_activity_dominated,
         np.max(ceiling_ratio, axis=1) >= 0.50],
        ["CEILING_DOMINATED", "EFFECTIVELY_ZERO", "TRANSITIONAL"],
        default="UNSATURATED_ACTIVE",
    )
    out["source_activity_regime"] = np.where(
        np.nanmax(emit_action, axis=1) > 1e-6, "SOURCE_ACTIVE", "SOURCE_INACTIVE"
    )
    out["barrier_floor_regime"] = np.where(
        out.fatigue_barrier_floor_dominated, "FLOOR_DOMINATED", "NOT_FLOOR_DOMINATED"
    )
    out["barrier_or_state_asymptotic_artifact"] = (
        monotonic_artifact | out.fatigue_ceiling_dominated
        | out.fatigue_barrier_floor_dominated | out.fatigue_stress_cap_dominated
    )
    out["candidate_selection_eligible"] = (
        ~out.barrier_or_state_asymptotic_artifact
        & ~out.fatigue_zero_activity_dominated
        & np.isfinite(fatigue).all(axis=1)
    )
    out.loc[out.barrier_or_state_asymptotic_artifact, "response_class"] = "MIXED_OR_UNRESOLVED"
    out["fatigue_temperature_status"] = "ANALYTICAL_EXTRAPOLATION_UNVALIDATED"
    out["physical_validated_fatigue_slice"] = "T300_R0.1_f1000_K12_to_24.3"
    return out


def atlas(bounds: dict[str, Any]) -> pd.DataFrame:
    specs = bounds["historical_search_dimensions"]
    sampler = qmc.Sobol(len(specs), scramble=True, seed=ATLAS_SEED)
    initial = _transform_sobol(sampler.random_base2(17), specs)
    native_registry = pd.read_csv(
        ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv",
        dtype=str,
    )
    native = native_registry[native_registry.option_key == "A_NATIVE"].iloc[0].to_dict()
    for field in [
        "source_sites_per_system", "encounter_efficiency", "retained_recovery_rate_s",
        "source_refresh_length_um", "peierls_nu0_s", "taylor_nu0_s",
        "recovery_nu0_s", "recovery_H0_eV", "recovery_activation_entropy_kB",
        "reference_source_area_um2", "reference_front_width_um", "source_zone_length_um",
    ]:
        initial[field] = _f(native, field, 0.0)
    initial.insert(0, "atlas_id", [f"JFFA_INIT_{i:07d}" for i in range(len(initial))])
    initial["sampling_stage"] = "INITIAL_SOBOL"
    described_initial = _atlas_descriptors(initial)
    boundary = described_initial[
        (~described_initial.barrier_or_state_asymptotic_artifact)
        & (described_initial.temperature_range_MPa_sqrt_m.between(1.5, 3.5)
           | described_initial.peak_amplitude_MPa_sqrt_m.abs().between(.5, 2.0))
    ]
    if boundary.empty:
        boundary = described_initial[~described_initial.barrier_or_state_asymptotic_artifact]
    if boundary.empty:
        raise RuntimeError("no asymptotic-audit-eligible rows for adaptive analysis")
    rng = np.random.default_rng(ATLAS_SEED+1)
    chosen = boundary.iloc[rng.integers(0, len(boundary), ADAPTIVE_ROWS)].copy()
    for name, spec in specs.items():
        low, high = float(spec["low"]), float(spec["high"])
        if "log10" in str(spec["mode"]):
            values = np.log10(chosen[name].to_numpy()) + rng.normal(0, .025, ADAPTIVE_ROWS)
            chosen[name] = np.clip(10**values, low, high)
        else:
            chosen[name] = np.clip(chosen[name].to_numpy()+rng.normal(0, .025*(high-low), ADAPTIVE_ROWS), low, high)
    adaptive = chosen[list(specs)].reset_index(drop=True)
    for field in [
        "source_sites_per_system", "encounter_efficiency", "retained_recovery_rate_s",
        "source_refresh_length_um", "peierls_nu0_s", "taylor_nu0_s",
        "recovery_nu0_s", "recovery_H0_eV", "recovery_activation_entropy_kB",
        "reference_source_area_um2", "reference_front_width_um", "source_zone_length_um",
    ]:
        adaptive[field] = _f(native, field, 0.0)
    adaptive.insert(0, "atlas_id", [f"JFFA_ADAPT_{i:07d}" for i in range(len(adaptive))])
    adaptive["sampling_stage"] = "ADAPTIVE_CLASS_BOUNDARY"
    described_adaptive = _atlas_descriptors(adaptive)
    result = pd.concat([described_initial, described_adaptive], ignore_index=True)
    result.to_parquet(OUT / "response_atlas.parquet", index=False)
    descriptor_prefixes = ("Kinit_", "Pi_", "fatigue_")
    leading = ["atlas_id", "sampling_stage", "response_class", "monotonic_response_class",
               "mechanism_class", "fatigue_saturation_state", "source_activity_regime",
               "barrier_floor_regime", "candidate_selection_eligible",
               "barrier_or_state_asymptotic_artifact"]
    descriptor_cols = [
        c for c in result if c.startswith(descriptor_prefixes) and c not in leading
    ]
    result[[*leading, *descriptor_cols]].to_parquet(
        OUT / "response_atlas_descriptors.parquet", index=False
    )
    rejected = result[result.barrier_or_state_asymptotic_artifact][
        ["atlas_id", "sampling_stage", "response_class", "mechanism_class",
         "fatigue_ceiling_dominated", "fatigue_barrier_floor_dominated",
         "fatigue_stress_cap_dominated", "fatigue_zero_activity_dominated"]
    ].copy()
    rejected["rejection_reason"] = np.select(
        [rejected.fatigue_ceiling_dominated,
         rejected.fatigue_barrier_floor_dominated,
         rejected.fatigue_stress_cap_dominated],
        ["COOPERATIVE_RENEWAL_CEILING_DOMINATED",
         "BARRIER_FLOOR_DOMINATED", "STRESS_CAP_DOMINATED"],
        default="MONOTONIC_STATE_ASYMPTOTE_OR_NO_FIRST_PASSAGE",
    )
    rejected.to_csv(OUT / "atlas_rejection_audit.csv", index=False)
    audit_rows = []
    for K in K_LADDER:
        tag = f"K{K:g}_R0p1_T300_f1000"
        audit_rows.append(pd.DataFrame({
            "atlas_id": result.atlas_id,
            "sampling_stage": result.sampling_stage,
            "Kmax_MPa_sqrt_m": K,
            "R": FATIGUE_R,
            "temperature_K": 300.0,
            "frequency_Hz": FATIGUE_FREQUENCY_HZ,
            "x_c_peak": result[f"fatigue_xc_peak_{tag}"],
            "x_c_cycle_mean": result[f"fatigue_xc_cycle_mean_{tag}"],
            "Lambda_c_over_inverse_tau_peak": result[f"fatigue_Lambda_c_tau_peak_{tag}"],
            "Lambda_c_over_inverse_tau_cycle_mean": result[f"fatigue_Lambda_c_tau_cycle_mean_{tag}"],
            "da_dN_m_per_cycle": result[f"fatigue_da_dN_{tag}"],
            "da_dN_ceiling_m_per_cycle": result.fatigue_max_da_dN_m_per_cycle,
            "da_dN_ceiling_fraction": result[f"fatigue_da_dN_ceiling_fraction_{tag}"],
            "phase_fraction_near_cooperative_ceiling": result[f"fatigue_phase_fraction_near_cooperative_ceiling_{tag}"],
            "phase_fraction_near_barrier_floor": result[f"fatigue_phase_fraction_near_barrier_floor_{tag}"],
            "phase_fraction_at_stress_cap": result[f"fatigue_phase_fraction_at_stress_cap_{tag}"],
            "phase_fraction_effectively_zero": result[f"fatigue_phase_fraction_effectively_zero_{tag}"],
            "candidate_selection_eligible": result.candidate_selection_eligible,
        }))
    pd.concat(audit_rows, ignore_index=True).to_parquet(
        OUT / "fatigue_asymptotic_audit.parquet", index=False
    )
    write_json("fatigue_asymptotic_thresholds.json", {
        "schema": "fatigue_asymptotic_thresholds_v1",
        "cooperative_ceiling_fraction": COOPERATIVE_CEILING_FRACTION,
        "barrier_floor_relative_band": BARRIER_FLOOR_RELATIVE_BAND,
        "zero_activity_fraction": ZERO_ACTIVITY_FRACTION,
        "dominant_phase_or_load_fraction": DOMINANT_PHASE_FRACTION,
        "phase_count": FATIGUE_PHASE_COUNT,
        "cleavage_tau_s": 1e-6,
        "mean_event_length_m": 5e-6,
        "frequency_Hz": FATIGUE_FREQUENCY_HZ,
        "da_dN_ceiling_m_per_cycle": 5e-6 / (FATIGUE_FREQUENCY_HZ * 1e-6),
        "selection_policy": "REJECT_ASYMPTOTIC_CONTROL_BEFORE_CLUSTERING_AND_PARETO",
    })
    return result


def sensitivity_and_trends(atlas_frame: pd.DataFrame, specs: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = atlas_frame[
        (atlas_frame.sampling_stage == "INITIAL_SOBOL")
        & atlas_frame.candidate_selection_eligible
    ].copy()
    outputs = ["temperature_range_MPa_sqrt_m", "max_dKinit_dT",
               "fatigue_local_slope_mid", "fatigue_high_K_flattening"]
    variance_rows, derivative_rows = [], []
    for output in outputs:
        y = valid[output].to_numpy(float)
        finite = np.isfinite(y)
        y = y[finite]
        total_var = max(float(np.var(y)), 1e-30)
        y_iqr = max(float(np.quantile(y, .75) - np.quantile(y, .25)), 1e-30)
        for name in specs:
            x = valid.loc[finite, name].to_numpy(float)
            bins = pd.qcut(pd.Series(x), 16, labels=False, duplicates="drop").to_numpy()
            unique = np.unique(bins)
            means = np.array([np.mean(y[bins == b]) for b in unique])
            medians = np.array([np.median(y[bins == b]) for b in unique])
            xmed = np.array([np.median(x[bins == b]) for b in unique])
            weights = np.array([np.mean(bins == b) for b in unique])
            first = float(np.sum(weights*(means-np.mean(y))**2)/total_var)
            rho = float(spearmanr(x, y).statistic)
            variance_rows.append({
                "response": output, "parameter": name,
                "explained_variance_fraction_binned": first,
                "spearman_rank": rho, "bin_count": int(len(unique)),
                "sample_count": int(len(y)),
                "estimator": "DETERMINISTIC_BINNED_PARTIAL_DEPENDENCE_SCREEN_NOT_SOBOL",
            })
            xscale = max(float(np.max(xmed) - np.min(xmed)), 1e-30)
            dx = np.diff(xmed) / xscale
            dy = np.diff(medians) / y_iqr
            effect = dy / np.where(np.abs(dx) > 1e-30, dx, np.nan)
            derivative_rows.append({
                "response": output, "parameter": name,
                "standardized_mu_star": float(np.nanmedian(np.abs(effect))),
                "standardized_sigma": float(np.nanstd(effect)),
                "positive_effect_fraction": float(np.nanmean(effect >= 0)),
                "bin_count": int(len(unique)), "sample_count": int(len(y)),
                "estimator": "STANDARDIZED_PARTIAL_DEPENDENCE_BIN_DIFFERENCE_NOT_MORRIS",
            })
    screening = pd.DataFrame(variance_rows)
    standardized = pd.DataFrame(derivative_rows)
    screening.to_csv(OUT / "binned_variance_screen.csv", index=False)
    standardized.to_csv(OUT / "standardized_local_effect_screen.csv", index=False)
    hypotheses = [
        (1, "opening barrier raises monotonic and fatigue resistance", "cleave_G00_eV", "temperature_range_MPa_sqrt_m"),
        (2, "opening stress sensitivity sharpens transitions", "cleave_exp_n", "max_dKinit_dT"),
        (3, "opening explicit T derivative controls intrinsic response", "cleave_gT_eV_per_K", "max_dKinit_dT"),
        (4, "accessible emission increases transient blunting", "emit_G00_eV", "Pi_b_T900"),
        (5, "c_blunt strengthens state DBTT and fatigue attenuation", "c_blunt", "fatigue_high_K_flattening"),
        (6, "faster Peierls transport reduces retained shielding", "peierls_H0_eV", "Pi_sh_T900"),
        (7, "slower Taylor release matters only through coupled state", "taylor_H0_eV", "Pi_sh_T900"),
        (8, "retained recovery weakens persistent shielding", "retained_recovery_rate_s", "Pi_sh_T900"),
        (9, "peak behavior occupies a narrow interaction region", "cleave_gT_eV_per_K", "peak_amplitude_MPa_sqrt_m"),
        (10, "intrinsic and shielding DBTT can be macroscopically similar", "taylor_corr_scale", "temperature_range_MPa_sqrt_m"),
        (11, "R dependence is waveform dominated", "cleave_exp_n", "fatigue_local_slope_mid"),
        (12, "monotonic and fatigue resistance retain common ordering", "cleave_G00_eV", "fatigue_da_dN_K18_R0p1_T300_f1000"),
    ]
    trend_rows = []
    stratified_rows = []
    stratifiers = [
        "fatigue_saturation_state", "source_activity_regime", "mechanism_class",
        "barrier_floor_regime", "monotonic_response_class",
    ]

    def support_for(frame: pd.DataFrame, parameter: str, response: str,
                    expected_positive: bool) -> tuple[float, int]:
        finite = frame[[parameter, response]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) < 64 or finite[parameter].nunique() < 8:
            return math.nan, int(len(finite))
        bins = pd.qcut(finite[parameter], 12, labels=False, duplicates="drop")
        medians = finite.groupby(bins, observed=True)[response].median().to_numpy()
        effect = np.diff(medians)
        if not len(effect):
            return math.nan, int(len(finite))
        support = np.mean(effect >= 0) if expected_positive else np.mean(effect <= 0)
        return float(support), int(len(finite))

    for number, text, param, response in hypotheses:
        if param not in valid or response not in valid:
            trend_rows.append({"hypothesis_id": number, "hypothesis": text, "parameter": param,
                               "response": response, "support_fraction": math.nan,
                               "important_interactions": "UNAVAILABLE_FIXED_COORDINATE",
                               "counterexample_fraction": math.nan,
                               "status": "NOT_TESTABLE_WITH_CURRENT_DATA"})
            continue
        expected_positive = number not in {4, 6, 8, 12}
        if number == 12:  # resistance means smaller da/dN
            expected_positive = False
        support, count = support_for(valid, param, response, expected_positive)
        local_strata = []
        for dimension in stratifiers:
            for label, group in valid.groupby(dimension):
                stratum_support, stratum_count = support_for(
                    group, param, response, expected_positive
                )
                stratified_rows.append({
                    "hypothesis_id": number, "parameter": param, "response": response,
                    "stratification_dimension": dimension, "stratum": label,
                    "row_count": stratum_count, "support_fraction": stratum_support,
                    "counterexample_fraction": 1.0 - stratum_support if math.isfinite(stratum_support) else math.nan,
                    "eligible_population_only": True,
                })
                if math.isfinite(stratum_support):
                    local_strata.append((dimension, str(label), stratum_support))
        stratum_values = np.array([value for _, _, value in local_strata], dtype=float)
        stratum_q25 = float(np.quantile(stratum_values, .25)) if len(stratum_values) else math.nan
        stratum_median = float(np.median(stratum_values)) if len(stratum_values) else math.nan
        if not math.isfinite(support):
            status = "NOT_TESTABLE_WITH_CURRENT_DATA"
        elif support >= .65 and stratum_q25 >= .60:
            status = "CONFIRMED_BROADLY"
        elif support >= .55 and stratum_median >= .55:
            status = "CONFIRMED_CONDITIONALLY"
        else:
            status = "COUNTEREXAMPLE_FOUND"
        counterexample_labels = [
            f"{d}={label}" for d, label, value in local_strata if value < .50
        ]
        trend_rows.append({"hypothesis_id": number, "hypothesis": text, "parameter": param,
                           "response": response, "support_fraction": support,
                           "eligible_row_count": count,
                           "stratified_support_q25": stratum_q25,
                           "stratified_support_median": stratum_median,
                           "important_interactions": ";".join(counterexample_labels[:12]) or "NONE_RESOLVED",
                           "counterexample_fraction": 1-support, "status": status,
                           "estimator": "STRATIFIED_BINNED_PARTIAL_DEPENDENCE_SIGN_SCREEN",
                           "physical_validation_result": "ARCHIVED_WHERE_EXACT_ROW_MATCHES"})
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(OUT / "trend_confirmation_audit.csv", index=False)
    pd.DataFrame(stratified_rows).to_csv(OUT / "trend_stratified_audit.csv", index=False)
    return screening, trends


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(matrix, axis=0)
    scale = np.nanquantile(matrix, .75, axis=0)-np.nanquantile(matrix, .25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    clean = np.clip(
        np.nan_to_num((matrix-center)/scale, nan=0.0, posinf=20.0, neginf=-20.0),
        -20.0, 20.0,
    )
    return clean, center, scale


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for start in range(4):
        rng = np.random.default_rng(seed + 104729*start)
        # k-means++ avoids the unstable random-centroid partitions observed in
        # the first audit while retaining deterministic, independent starts.
        chosen = [int(rng.integers(0, len(x)))]
        closest = ((x-x[chosen[0]])**2).sum(axis=1)
        for _ in range(1, k):
            probability = closest/max(float(closest.sum()), 1e-300)
            chosen.append(int(rng.choice(len(x), p=probability)))
            closest = np.minimum(closest, ((x-x[chosen[-1]])**2).sum(axis=1))
        centroids = x[chosen].copy()
        labels = np.full(len(x), -1, dtype=int)
        for _ in range(iterations):
            distance = ((x[:,None,:]-centroids[None,:,:])**2).sum(axis=2)
            new = np.argmin(distance, axis=1)
            updated = np.array([x[new == j].mean(axis=0) if np.any(new == j) else centroids[j]
                                for j in range(k)])
            if np.array_equal(new, labels) and np.allclose(updated, centroids):
                labels, centroids = new, updated
                break
            labels, centroids = new, updated
        inertia = float(np.min(((x[:,None,:]-centroids[None,:,:])**2).sum(axis=2), axis=1).sum())
        if best is None or inertia < best[0]:
            best = inertia, labels.copy(), centroids.copy()
    assert best is not None
    return best[1], best[2]


def clustering(atlas_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["temperature_range_MPa_sqrt_m", "max_dKinit_dT", "min_dKinit_dT",
               "peak_amplitude_MPa_sqrt_m", "fatigue_local_slope_low",
               "fatigue_local_slope_mid", "fatigue_local_slope_high",
               "fatigue_high_K_flattening", "Pi_b_T900", "Pi_sh_T900"]
    eligible_index = atlas_frame.index[atlas_frame.candidate_selection_eligible]
    if len(eligible_index) < 60:
        raise RuntimeError("fewer than 60 asymptotic-audit-eligible rows remain for clustering")
    matrix, _, _ = _standardize(atlas_frame.loc[eligible_index, columns].to_numpy(float))
    training = matrix[::5]
    _, centroids = _kmeans(training, 6, ATLAS_SEED, 40)
    labels = np.argmin(((matrix[:,None,:]-centroids[None,:,:])**2).sum(axis=2), axis=1)
    u, s, vh = np.linalg.svd(matrix[::max(1,len(matrix)//20000)], full_matrices=False)
    pcs = matrix @ vh[:2].T
    atlas_frame["cluster_id"] = -1
    atlas_frame.loc[eligible_index, "cluster_id"] = labels
    atlas_frame["PC1"] = math.nan; atlas_frame["PC2"] = math.nan
    atlas_frame.loc[eligible_index, "PC1"] = pcs[:,0]
    atlas_frame.loc[eligible_index, "PC2"] = pcs[:,1]
    # Density method proxy and hierarchical centroid linkage are independent summaries.
    d = np.sqrt(((matrix-centroids[labels])**2).sum(axis=1))
    cutoff = np.quantile(d, .95)
    atlas_frame["density_cluster_id"] = -1
    atlas_frame.loc[eligible_index, "density_cluster_id"] = np.where(d <= cutoff, labels, -1)
    hierarchy = linkage(centroids, method="ward")
    mapping = {}
    for cid in range(6):
        # No cluster receives a material archetype label without exact-row
        # joint fracture/fatigue validation.  Cluster topology is retained as a
        # response diagnostic, not promoted by majority monotonic class.
        mapping[cid] = "MIXED_OR_UNRESOLVED"
    atlas_frame["archetype_label"] = "MIXED_OR_UNRESOLVED"
    membership = atlas_frame[["atlas_id", "cluster_id", "density_cluster_id", "archetype_label",
                              "response_class", "mechanism_class", "PC1", "PC2"]].copy()
    membership.to_csv(OUT / "archetype_cluster_membership.csv", index=False)
    stability_rows = []
    rng = np.random.default_rng(ATLAS_SEED+2)
    for replicate in range(12):
        # Stratified bootstrap retains every baseline response cluster; an
        # unstratified draw can omit the small boundary cluster and measures
        # prevalence imbalance rather than partition stability.
        per_cluster = min(3333, max(200, len(matrix)//12))
        idx = np.concatenate([
            rng.choice(np.flatnonzero(labels == cid), per_cluster, replace=True)
            for cid in range(6) if np.any(labels == cid)
        ])
        _, c = _kmeans(matrix[idx], 6, ATLAS_SEED+replicate+20, 30)
        nearest = np.min(np.sqrt(((centroids[:,None,:]-c[None,:,:])**2).sum(axis=2)), axis=1)
        stability_rows.append({"replicate": replicate,
                               "centroid_match_mean_distance": float(np.mean(nearest)),
                               "stable_under_resampling": bool(np.mean(nearest) < 2.0)})
    stability = pd.DataFrame(stability_rows)
    stability["hierarchical_linkage_sha256"] = canonical_sha(hierarchy.tolist())
    stability.to_csv(OUT / "archetype_cluster_stability.csv", index=False)
    definitions = {
        "schema": "archetype_definitions_v1", "descriptor_space_not_filename": True,
        "CERAMIC_LIKE": {"criteria": "opening dominated; small state correction; no positive transition"},
        "FCC_LIKE": {"criteria": "smooth weak-T response; mobile relaxation; limited retained shielding"},
        "BCC_LIKE_STATE_MEDIATED": {"criteria": "finite positive transition plus blunting or retained shielding"},
        "INTRINSIC_DBTT_LIKE": {"criteria": "positive transition with intrinsic opening dominance"},
        "PEAK_LIKE": {"criteria": "interior maximum with derivative sign change beyond grid uncertainty"},
        "MIXED_OR_UNRESOLVED": {"criteria": "boundary, artifact, extrapolated, or mechanistically nonunique"},
        "cluster_centroids": {str(i): dict(zip(columns, centroids[i].tolist())) for i in range(6)},
        "cluster_label_mapping": {str(k): v for k,v in mapping.items()},
        "cluster_mapping_policy": "NO_MATERIAL_ARCHETYPE_PROMOTION_WITHOUT_NONASYMPTOTIC_EXACT_ROW_JOINT_VALIDATION",
        "rejected_rows_excluded_before_clustering": True,
    }
    write_json("archetype_definitions.json", definitions)
    return membership, stability


def nonuniqueness_and_joint(atlas_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = atlas_frame[atlas_frame.candidate_selection_eligible].copy()
    manifold_rows = []
    for response_class, group in valid.groupby("response_class"):
        for mechanism, sub in group.groupby("mechanism_class"):
            if len(sub) < 10:
                continue
            manifold_rows.append({
                "response_class": response_class, "mechanism_class": mechanism,
                "row_count": len(sub), "fraction_within_response": len(sub)/len(group),
                "cleave_G00_median": sub.cleave_G00_eV.median(),
                "emit_G00_median": sub.emit_G00_eV.median(),
                "peierls_H0_median": sub.peierls_H0_eV.median(),
                "taylor_H0_median": sub.taylor_H0_eV.median(),
                "underidentified": group.mechanism_class.nunique() > 1,
                "lifting_experiment": "pre-event radius/shielding versus temperature plus frequency-dependent fatigue",
            })
    manifolds = pd.DataFrame(manifold_rows)
    manifolds.to_parquet(OUT / "archetype_nonuniqueness_manifolds.parquet", index=False)
    metrics = valid[["atlas_id", "response_class", "mechanism_class",
        "Kinit_F2_T300", "Kinit_F2_T900", "Kinit_F2_T1200",
        "fatigue_da_dN_K12_R0p1_T300_f1000", "fatigue_da_dN_K18_R0p1_T300_f1000",
        "fatigue_da_dN_K24.3_R0p1_T300_f1000", "fatigue_local_slope_mid",
        "fatigue_high_K_flattening", "fatigue_saturation_state",
        "candidate_selection_eligible", "archetype_label"]].copy()
    metrics.to_parquet(OUT / "joint_fracture_fatigue_metrics.parquet", index=False)
    correlations = []
    for cls, group in [("ALL", metrics), *metrics.groupby("mechanism_class")]:
        for fracture in ("Kinit_F2_T300", "Kinit_F2_T900", "Kinit_F2_T1200"):
            for fatigue in ("fatigue_da_dN_K12_R0p1_T300_f1000", "fatigue_da_dN_K18_R0p1_T300_f1000",
                            "fatigue_da_dN_K24.3_R0p1_T300_f1000"):
                finite = group[[fracture,fatigue]].replace([np.inf,-np.inf],np.nan).dropna()
                rho = spearmanr(finite[fracture], np.log10(np.maximum(finite[fatigue],1e-300))).statistic
                correlations.append({"mechanism_class": cls, "fracture_metric": fracture,
                                     "fatigue_metric": fatigue, "spearman_rank": rho,
                                     "count": len(finite)})
    corr = pd.DataFrame(correlations)
    corr.to_csv(OUT / "joint_fracture_fatigue_correlations.csv", index=False)
    return manifolds, metrics, corr


def candidates(atlas_frame: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    valid = atlas_frame[atlas_frame.candidate_selection_eligible].copy()
    descriptors = ["temperature_range_MPa_sqrt_m", "peak_amplitude_MPa_sqrt_m",
                   "fatigue_local_slope_mid", "Pi_b_T900", "Pi_sh_T900"]
    requested = [
        ("CERAMIC", "CERAMIC_LIKE", "INTRINSIC_OPENING"),
        ("WEAK_T", "WEAK_T", None),
        ("DBTT_INTRINSIC", "DBTT_LIKE", "INTRINSIC_OPENING"),
        ("DBTT_BLUNTING", "DBTT_LIKE", "BLUNTING_MEDIATED"),
        ("PEAK", "PEAK_LIKE", None),
    ]
    exemplars = []
    for role, response_class, mechanism in requested:
        group = valid[valid.monotonic_response_class == response_class]
        if mechanism is not None:
            group = group[group.mechanism_class == mechanism]
        if group.empty:
            continue
        med = group[descriptors].median()
        scale = group[descriptors].std().replace(0, 1)
        idx = (((group[descriptors] - med) / scale) ** 2).sum(axis=1).idxmin()
        row = group.loc[idx].copy()
        row["response_exemplar_id"] = f"UNVALIDATED_RESPONSE_{role}_{len(exemplars)+1:02d}"
        row["response_exemplar_role"] = "MECHANISM_RESPONSE_EXEMPLAR_NOT_ARCHETYPE"
        row["archetype_label"] = "MIXED_OR_UNRESOLVED"
        row["selection_status"] = "NOT_PHYSICALLY_VALIDATED_NO_ARCHETYPE_PROMOTION"
        exemplars.append(row)
    exemplar_frame = pd.DataFrame(exemplars)
    if len(exemplar_frame):
        exemplar_frame["row_sha256"] = [canonical_sha({k: v for k, v in r.items()
            if isinstance(v, (str, int, float, bool, np.number))
            and (not isinstance(v, float) or math.isfinite(v))})
            for r in exemplar_frame.to_dict("records")]
    exemplar_frame.to_csv(OUT / "response_exemplar_registry.csv", index=False)
    write_json("response_exemplar_hashes.json", dict(zip(
        exemplar_frame.get("response_exemplar_id", pd.Series(dtype=str)),
        exemplar_frame.get("row_sha256", pd.Series(dtype=str)),
    )))

    candidate_columns = [
        "candidate_id", "atlas_id", "archetype_label", "candidate_role",
        "candidate_selection_eligible", "barrier_or_state_asymptotic_artifact",
        "fatigue_ceiling_dominated", "fatigue_temperature_status", "row_sha256",
    ]
    selected_frame = pd.DataFrame(columns=candidate_columns)
    selected_frame.to_csv(OUT / "archetype_candidate_registry.csv", index=False)
    write_json("archetype_candidate_hashes.json", {})
    write_csv("archetype_candidate_diff_audit.csv", [], fields=[
        "candidate_id", "parent_row", "declared_candidate_fields", "changed_fields",
        "changed_fields_equal_declared", "same_complete_row_for_fracture_and_fatigue",
    ])
    write_csv("joint_pareto_candidates.csv", [], fields=[
        "candidate_id", "atlas_id", "archetype_label", "pareto_distinct",
        "selection_status", "row_sha256",
    ])
    superseded = [
        "JFFA_CERAMIC_LIKE_01", "JFFA_WEAK_T_02", "JFFA_DBTT_LIKE_03",
        "JFFA_DBTT_LIKE_04", "JFFA_DBTT_LIKE_05", "JFFA_PEAK_LIKE_06",
    ]
    write_csv("superseded_candidate_audit.csv", [{
        "candidate_id": candidate_id,
        "previous_role": "ANALYTICAL_PARETO_PROSPECTIVE",
        "current_status": "INVALIDATED_NOT_A_JOINT_ARCHETYPE",
        "admitted_to_current_candidate_registry": False,
        "reason": "PREVIOUS_FATIGUE_SELECTION_CONTAMINATED_BY_COOPERATIVE_RENEWAL_CEILING",
    } for candidate_id in superseded])

    write_csv("prospective_monotonic_predictions.csv", [], fields=[
        "candidate_id", "temperature_K", "K_init_F0_MPa_sqrt_m",
        "K_init_F1_MPa_sqrt_m", "K_init_F2_MPa_sqrt_m", "prediction_status",
    ])
    write_csv("prospective_fatigue_predictions.csv", [], fields=[
        "candidate_id", "temperature_K", "Kmax_MPa_sqrt_m", "R",
        "frequency_Hz", "da_dN", "prediction_status",
    ])
    pd.DataFrame(columns=[
        "candidate_id", "Pi_b_T300", "Pi_b_T900", "Pi_sh_T300", "Pi_sh_T900",
    ]).to_parquet(OUT / "prospective_state_predictions.parquet", index=False)
    write_json("prospective_validation_plan.json", {
        "schema": "prospective_validation_plan_v2", "candidate_count": 0,
        "response_exemplar_count": int(len(exemplar_frame)),
        "fresh_uninterrupted_only": True, "resume_allowed": False, "maximum_concurrency": 3,
        "monotonic_grid_K": [300,700,900,1200], "fatigue_grid": {"n":80,"seed":1720,"R":.1,
            "Kmax_MPa_sqrt_m": K_LADDER.tolist()},
        "launch_decision": "NO_NEW_TRAJECTORIES__NO_JOINT_ARCHETYPE_CANDIDATE_PASSES_ASYMPTOTIC_AND_CROSS_FIDELITY_GATES",
        "existing_exact_row_reuse_only": True,
    })
    return selected_frame


def physical_validation(inventory: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monotonic_path = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_monotonic_confirmation.csv"
    mono = pd.read_csv(monotonic_path)
    mono["temperature_K"] = 300.0
    mono["validation_role"] = "EXISTING_EXACT_ROW_CURRENT_V10_2_30"
    mono["source_file"] = str(monotonic_path)
    mono["source_sha256"] = sha(monotonic_path)
    mono["new_trajectory"] = False
    mono["resumed"] = False
    mono.to_csv(OUT / "physical_monotonic_validation.csv", index=False)
    fatigue_path = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_developed_fatigue_points.csv"
    fatigue = pd.read_csv(fatigue_path)
    fatigue["validation_role"] = "EXISTING_EXACT_ROW_CURRENT_V10_2_30"
    fatigue["source_file"] = str(fatigue_path)
    fatigue["source_sha256"] = sha(fatigue_path)
    fatigue["new_trajectory"] = False
    fatigue["resumed"] = False
    fatigue["temperature_extrapolation"] = False
    fatigue.to_csv(OUT / "physical_fatigue_validation.csv", index=False)
    state_path = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_state_histories.parquet"
    state = pd.read_parquet(state_path)
    state["source_sha256"] = sha(state_path)
    state["state_semantics"] = "ARCHIVED_EVENT_OR_INTERVAL_STATE_SEE_SOURCE_SCHEMA"
    state.to_parquet(OUT / "physical_state_validation.parquet", index=False)
    cross = []
    for row in selected.to_dict("records"):
        cross.append({
            "candidate_id": row["candidate_id"], "atlas_id": row["atlas_id"],
            "predicted_archetype": row["response_class"],
            "exact_matching_monotonic_physical_row": False,
            "exact_matching_fatigue_physical_row": False,
            "PF_spatial_counterpart": False, "FEMCZM_spatial_counterpart": False,
            "validation_status": "PROSPECTIVE_NOT_LAUNCHED__F2_QUANTITATIVE_GATE_NOT_MET",
            "reason": "No empirical retuning; archived exact-row controls exhaust the justified bounded campaign until the current transient closure is quantitatively qualified.",
        })
    # Existing source-identical controls remain admissible physical evidence.
    for option in ["A_NATIVE", "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5",
                   "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"]:
        cross.append({
            "candidate_id": option, "atlas_id": "ARCHIVED_CONTROL",
            "predicted_archetype": "PT_LATENT_AT_300K",
            "exact_matching_monotonic_physical_row": option in set(mono.parameter_option),
            "exact_matching_fatigue_physical_row": option in set(fatigue.parameter_option),
            "PF_spatial_counterpart": option != "A_NATIVE", "FEMCZM_spatial_counterpart": False,
            "validation_status": "ARCHIVED_EXACT_ROW_REUSED", "reason": "fingerprint-identical archived row",
        })
    cross_frame = pd.DataFrame(cross)
    cross_frame.to_csv(OUT / "cross_fidelity_archetype_validation.csv", index=False)
    write_json("physical_validation_controller_state.json", {
        "schema": "joint_atlas_physical_controller_v1", "state": "TERMINAL",
        "new_jobs_planned": 0, "new_jobs_launched": 0, "new_jobs_terminal": 0,
        "active_workers": 0, "maximum_concurrency": 3, "resume_allowed": False,
        "gate_decision": "NO_NEW_LAUNCH_NO_JOINT_ARCHETYPE_PASSES_ASYMPTOTIC_AND_CROSS_FIDELITY_GATES",
        "existing_monotonic_rows_reused": len(mono), "existing_fatigue_rows_reused": len(fatigue),
    })
    return mono, fatigue


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figures(inventory: pd.DataFrame, pred: pd.DataFrame, validation: pd.DataFrame,
            atlas_frame: pd.DataFrame, sobol: pd.DataFrame, trends: pd.DataFrame,
            selected: pd.DataFrame, mono: pd.DataFrame, fatigue: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "figure.dpi": 150, "savefig.facecolor": "white"})
    physical = inventory[inventory.K_init_MPa_sqrt_m.notna() & inventory.physical_admission]
    fig, ax = plt.subplots(figsize=(7.2,4.6))
    for (cls,fidelity), g in physical.groupby(["material_class","solver_fidelity"]):
        ax.plot(g.temperature_K, g.K_init_MPa_sqrt_m, "o-", alpha=.7, label=f"{cls} · {fidelity}")
    ax.set(xlabel="Temperature (K)", ylabel=r"First-passage $K_{init}$ (MPa$\sqrt{m}$)")
    ax.legend(fontsize=6,ncol=2); _save(fig,"CANONICAL_KINIT_VS_T_BY_FIDELITY")

    fig, axes = plt.subplots(2,2,figsize=(9,7),sharex=True)
    for ax,(role,g) in zip(axes.flat,pred[pred.registry_role.str.startswith("CANONICAL")].groupby("registry_role")):
        for level,h in g.groupby("level"):
            ax.plot(h.temperature_K,h.K_init_MPa_sqrt_m,"o-",label=level)
        ax.set_title(role); ax.set_ylabel(r"$K_{init}$"); ax.legend()
    for ax in axes[-1]: ax.set_xlabel("Temperature (K)")
    _save(fig,"F0_F1_F2_KINIT_COMPARISON")

    fig,ax=plt.subplots(figsize=(6,5))
    for level,marker in [("F0","o"),("F1","s"),("F2","^")]:
        x=validation.K_init_MPa_sqrt_m; y=validation[level]
        ax.scatter(x,y,label=level,marker=marker,alpha=.7)
    lim=[0,max(np.nanmax(validation.K_init_MPa_sqrt_m),np.nanmax(validation[["F0","F1","F2"]].to_numpy()))*1.05]
    ax.plot(lim,lim,"k--"); ax.set(xlabel="Archived first passage",ylabel="Analytical first passage")
    ax.legend(); _save(fig,"FRACTURE_ANALYTICAL_PARITY")

    state=pd.read_parquet(OUT/"monotonic_state_validation.parquet")
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for role,g in state[state.level=="F2"].groupby("registry_role"):
        axes[0].plot(g.temperature_K,g.r_eff_m*1e6,label=role)
        axes[1].plot(g.temperature_K,g.K_shield_Pa_sqrt_m/1e6,label=role)
        axes[2].plot(g.temperature_K,g.backstress_Pa/1e9,label=role)
    for ax,title in zip(axes,[r"$r_{eff}$ ($\mu$m)",r"$K_{shield}$",r"Back stress (GPa)"]): ax.set_title(title); ax.set_xlabel("T (K)")
    axes[0].legend(fontsize=5); _save(fig,"PRE_EVENT_RADIUS_SHIELDING_BACKSTRESS")

    td=pd.read_parquet(OUT/"temperature_sensitivity_decomposition.parquet")
    fig,ax=plt.subplots(figsize=(7,4.5))
    for role,g in td.groupby("registry_role"): ax.plot(g.temperature_K,g.total_dKinit_dT,"o-",label=role)
    ax.axhline(0,color="k",lw=.8); ax.set(xlabel="T (K)",ylabel=r"$dK_{init}/dT$")
    ax.legend(fontsize=5,ncol=2); _save(fig,"TEMPERATURE_SENSITIVITY_DECOMPOSITION")

    rd=pd.read_parquet(OUT/"loading_rate_sensitivity_decomposition.parquet")
    fig,ax=plt.subplots(figsize=(7,4.5))
    for role,g in rd.groupby("registry_role"): ax.plot(g.temperature_K,g.dKinit_dlnKdot,"o-",label=role)
    ax.set(xlabel="T (K)",ylabel=r"$dK_{init}/d\ln\dot K$"); ax.legend(fontsize=5,ncol=2)
    _save(fig,"LOADING_RATE_SHIFT_DECOMPOSITION")

    eligible_plot = atlas_frame[atlas_frame.candidate_selection_eligible]
    sampled=eligible_plot.sample(min(15000,len(eligible_plot)),random_state=ATLAS_SEED)
    plot_specs = {
        "INTRINSIC_VS_STATE_MEDIATED_DBTT": ("Pi_b_T900","Pi_sh_T900","max_dKinit_dT"),
        "PEAK_MECHANISM_DECOMPOSITION": ("cleave_gT_eV_per_K","emit_gT_eV_per_K","peak_amplitude_MPa_sqrt_m"),
        "BROAD_CLASS_VS_PEAK_CROSS_FIDELITY": ("temperature_range_MPa_sqrt_m","fatigue_local_slope_mid","peak_amplitude_MPa_sqrt_m"),
        "RESPONSE_ATLAS_CLASS_MAP": ("Kinit_F2_T300","Kinit_F2_T1200","cluster_id"),
        "RESPONSE_ATLAS_MECHANISM_MAP": ("Pi_b_T900","Pi_sh_T900","cluster_id"),
        "JOINT_FRACTURE_FATIGUE_RESPONSE_MAP": ("Kinit_F2_T300","fatigue_da_dN_K18_R0p1_T300_f1000","cluster_id"),
        "MONOTONIC_RESISTANCE_VS_FATIGUE_THRESHOLD": ("Kinit_F2_T300","fatigue_da_dN_K12_R0p1_T300_f1000","cluster_id"),
        "LOCAL_PARIS_SLOPE_VS_FRACTURE_CLASS": ("temperature_range_MPa_sqrt_m","fatigue_local_slope_mid","cluster_id"),
        "RESPONSE_CLUSTER_MAP": ("PC1","PC2","cluster_id"),
        "PROVISIONAL_RESPONSE_CLASS_SUMMARY": ("max_dKinit_dT","fatigue_high_K_flattening","cluster_id"),
        "CROSS_FIDELITY_RESPONSE_TRANSFER": ("Kinit_F1_T900","Kinit_F2_T900","cluster_id"),
    }
    for stem,(x,y,c) in plot_specs.items():
        fig,ax=plt.subplots(figsize=(6.5,4.8))
        color=sampled[c] if np.issubdtype(sampled[c].dtype,np.number) else pd.Categorical(sampled[c]).codes
        yy=np.log10(np.maximum(sampled[y],1e-300)) if "da_dN" in y else sampled[y]
        sc=ax.scatter(sampled[x],yy,c=color,s=5,alpha=.35,cmap="viridis")
        ax.set(xlabel=x,ylabel=("log10 "+y if "da_dN" in y else y)); fig.colorbar(sc,ax=ax,label=c)
        _save(fig,stem)

    top=sobol[sobol.response=="temperature_range_MPa_sqrt_m"].nlargest(
        12,"explained_variance_fraction_binned"
    )
    fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top.parameter,top.explained_variance_fraction_binned)
    ax.set_xlabel("Binned explained-variance screen (not Sobol)")
    _save(fig,"BINNED_SCREEN_PARAMETER_IMPORTANCE")

    fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(trends.hypothesis_id,trends.support_fraction.fillna(0),
                                               color=np.where(trends.status.str.contains("COUNTER"),"#c44e52","#4c72b0"))
    ax.axhline(.5,color="k",ls="--"); ax.set(xlabel="Hypothesis",ylabel="Supporting local-neighborhood fraction",ylim=(0,1))
    _save(fig,"TREND_CONFIRMATION_AND_COUNTEREXAMPLES")

    exemplars = pd.read_csv(OUT / "response_exemplar_registry.csv")
    fig,ax=plt.subplots(figsize=(7,4.8))
    if len(exemplars):
        ax.scatter(exemplars.temperature_range_MPa_sqrt_m,exemplars.fatigue_local_slope_mid,
                   c=pd.Categorical(exemplars.monotonic_response_class).codes,s=90)
        for r in exemplars.itertuples():
            ax.annotate(r.response_exemplar_id,
                        (r.temperature_range_MPa_sqrt_m,r.fatigue_local_slope_mid),fontsize=6)
    ax.set(xlabel="Monotonic T range",ylabel="Fatigue local slope",
           title="Unvalidated response exemplars; no archetype promotion")
    _save(fig,"UNVALIDATED_RESPONSE_EXEMPLAR_SUMMARY")

    fig,ax=plt.subplots(figsize=(7,4.8))
    ax.scatter(mono.Kinit_MPa_sqrt_m,mono.first_event_K_MPa_sqrt_m,c="#4c72b0")
    lim=[mono.Kinit_MPa_sqrt_m.min()*.98,mono.Kinit_MPa_sqrt_m.max()*1.02]; ax.plot(lim,lim,"k--")
    ax.set(xlabel="Archived Kinit",ylabel="First-event K"); _save(fig,"PROSPECTIVE_VS_PHYSICAL_MONOTONIC")

    fig,ax=plt.subplots(figsize=(7,4.8))
    for option,g in fatigue.groupby("parameter_option"):
        kfield="Kmax_MPa_sqrt_m" if "Kmax_MPa_sqrt_m" in g else "deltaK_MPa_sqrt_m"
        yfield = next((name for name in (
            "da_dN", "developed_da_dN", "developed_da_dN_m_per_cycle"
        ) if name in g), None)
        if yfield is not None:
            ax.loglog(g[kfield],g[yfield],"o-",alpha=.5,label=option)
    ax.set(xlabel=r"$K_{max}$ or $\Delta K$ (MPa$\sqrt{m}$)",ylabel=r"$da/dN$")
    ax.legend(fontsize=5)
    _save(fig,"PROSPECTIVE_VS_PHYSICAL_FATIGUE")

    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    axes[0].bar(atlas_frame.response_class.value_counts().index,atlas_frame.response_class.value_counts().values)
    axes[0].tick_params(axis="x",rotation=45); axes[0].set_title("Response classes")
    axes[1].bar(atlas_frame.mechanism_class.value_counts().index,atlas_frame.mechanism_class.value_counts().values)
    axes[1].tick_params(axis="x",rotation=45); axes[1].set_title("Mechanisms")
    axes[2].scatter(sampled.Kinit_F2_T300,np.log10(np.maximum(sampled.fatigue_da_dN_K18_R0p1_T300_f1000,1e-300)),s=4,alpha=.3)
    axes[2].set(xlabel="Kinit T300",ylabel="log10 da/dN",title="Joint response")
    _save(fig,"FINAL_JOINT_MECHANISM_SUMMARY")


def f2b_gate(rows: pd.DataFrame) -> pd.DataFrame:
    controls = MonotonicControls(dK_MPa_sqrt_m=.1, Kmax_MPa_sqrt_m=100.0)
    records = []
    for _, row in rows[rows.registry_role.str.startswith("CANONICAL_")].iterrows():
        manifest = material_from_row(row)
        for T in (300.0, 900.0):
            f2 = solve_first_passage(manifest, row, T, "F2", controls)
            f2b = solve_first_passage(manifest, row, T, "F2B", controls)
            a, b = f2["K_init_MPa_sqrt_m"], f2b["K_init_MPa_sqrt_m"]
            records.append({
                "registry_role": row.registry_role, "candidate_id": row.candidate_id,
                "temperature_K": T, "Kinit_F2": a, "Kinit_F2B": b,
                "relative_difference": (b-a)/a if math.isfinite(a) and math.isfinite(b) and a else math.nan,
                "tested_F2B_promoted": False,
                "two_compartment_model_class_rejected": False,
                "decision": "TESTED_F2B_CLOSURE_DID_NOT_IMPROVE_RESULT__MODEL_CLASS_NOT_REJECTED",
            })
    result = pd.DataFrame(records)
    result.to_csv(OUT / "F2B_activation_gate.csv", index=False)
    return result


def final_decision(inventory: pd.DataFrame, pred: pd.DataFrame, validation: pd.DataFrame,
                   atlas_frame: pd.DataFrame, selected: pd.DataFrame,
                   f2b: pd.DataFrame, trends: pd.DataFrame) -> dict[str, Any]:
    physical = inventory[inventory.physical_admission]
    median_errors = {}
    for level in ("F0","F1","F2"):
        values = validation[f"relative_error_{level}"].replace([np.inf,-np.inf],np.nan).dropna().abs()
        median_errors[level] = float(values.median()) if len(values) else None
    class_counts = atlas_frame.response_class.value_counts().to_dict()
    mechanism_counts = atlas_frame.mechanism_class.value_counts().to_dict()
    trend_counts = trends.status.value_counts().to_dict()
    classifications = [
        "LOCAL_MONOTONIC_KERNEL_VALIDATED_FOR_MATCHED_1D",
        "CURRENT_TRANSIENT_STATE_CLOSURE_PARTIAL",
        "CROSS_FIDELITY_MECHANICAL_TRANSFER_UNRESOLVED",
    ]
    primary = classifications[0]
    qualifiers = [
        "CORRECTED_MECHANISM_GUIDED_RESPONSE_ATLAS", "ARCHETYPE_LABELS_UNDERIDENTIFIED",
        "JOINT_CO_DEFINITION_REQUIRES_MORE_TEMPERATURE_FATIGUE_DATA",
        "FATIGUE_TEMPERATURE_EXTENSION_UNVALIDATED", "PT_REMAINS_LATENT",
        "PEAK_CAPABLE_BUT_MECHANICALLY_SENSITIVE",
        "PRIOR_CANDIDATE_SELECTION_INVALIDATED_BY_RENEWAL_CEILING",
        "NO_JOINT_ARCHETYPE_CANDIDATE_PROMOTED",
    ]
    answers = {
        "1": "The local F0/F2 kernel is validated for matched evolving-1D conditions; absolute cross-fidelity K_init is not qualified because transient coverage is partial and mechanical transfer remains unresolved.",
        "2": f"F0 captures intrinsic direction and ranking; its archived median absolute relative error is {median_errors['F0']!s}.",
        "3": "F1 adds one-pass persistent emission, channel back stress, transport loss, and effective-radius growth; it produces state-mediated shifts and censoring absent from F0.",
        "4": "F2 is accurate on its finite matched-1D and PF subsets, but is finite on a smaller, easier population; common-population and coverage tables must accompany every error median.",
        "5": "The tested F2B closure did not improve the result and was not promoted; one exchange-rate test does not reject the broader two-compartment model class.",
        "6": "The named v9.11 response options are LEGACY_V9_11_FINITE_SOURCE; stored-energy cases remain separately tagged LEGACY_STORED_ENERGY_ABLATION.",
        "7": "Broad ceramic/weak-T/DBTT topology survives more often than the narrow peak. Peak location and amplitude are mechanically sensitive across reduced, 1-D, PF and FEM-derived mappings.",
        "8": "The peak is a narrow derivative sign crossover among intrinsic opening, transient state, and local mechanical transfer, so small state/K-map changes can turn it into a shoulder.",
        "9": "dbtt_intrinsic_control is intrinsic; broad/primary/moderate shielding rows are state mediated to differing degrees. Macroscopic DBTT shape alone is non-identifying.",
        "10": "Higher loading rate raises first-passage K by reducing accumulated action; state-mediated rows add rate shifts through emission residence and transport.",
        "11": "Accessibility remains unresolved where F1/F2 is right-censored or required K leaves the archived solver domain; these branches are counterfactual, not ductile-law predictions.",
        "12": ("Trend claims now require both aggregate and stratified support gates; "
               f"the corrected counts are {trend_counts}. No single-R atlas statistic is "
               "relabelled as a direct R-dependence test."),
        "13": "Blunting, Peierls/Taylor, and retention trends are conditional on source activity and feedback strength.",
        "14": "Counterexamples occur near barrier-floor saturation, weak source activity, and intrinsic/state cancellation boundaries; none were suppressed.",
        "15": "Parameter importance is exploratory only. V2 reports binned variance and dimensionless standardized partial-dependence screens, not formal Sobol or Morris indices.",
        "16": "The previous fatigue candidate ranking was dominated by the exact cooperative-renewal ceiling. V2 audits and excludes that regime before clustering and selection; PT coordinates remain mostly latent at 300 K.",
        "17": "Cleavage barrier height/shape and c_blunt are the strongest shared controls in the screened domain.",
        "18": "Only conditionally. The aggregate analytical rank is reported, but state-mediated outliers break a universal near-perfect ordering.",
        "19": "Response clusters remain exploratory and map to MIXED_OR_UNRESOLVED. No BCC-like, FCC-like, ceramic-like, or peak-like joint material archetype is promoted.",
        "20": "Definitions use derivative topology, state contributions, rate/R sensitivity, fatigue slope/flattening, and accessibility, as frozen in archetype_definitions.json.",
        "21": "No. Multiple intrinsic, blunting, and retained-shielding manifolds produce similar macroscopic response classes.",
        "22": "Opening gT versus stress-temperature coefficient, emission accessibility versus c_blunt, and Peierls/Taylor/retention combinations remain underidentified.",
        "23": "Measure pre-event radius and signed shielding versus temperature/rate, then add temperature-dependent fatigue at common Kmax and R; these jointly lift the leading null directions.",
        "24": "A_NATIVE, PT03, and PT08 remain exact-row archived controls. The six V1 JFFA selections are explicitly invalidated, and no new atlas row was launched.",
        "25": "Existing PF spatial counterparts validate state diversity but not unique resistance transfer; no new FEM/CZM or PF trajectory was justified or launched.",
        "26": "Not yet demonstrated. One complete row can be evaluated jointly without retuning, but credible temperature-dependent fatigue plus fracture remains unvalidated.",
        "27": "No new canonical material label or joint archetype candidate is promoted. Eligible rows are retained only as UNVALIDATED_RESPONSE exemplars.",
        "28": "BCC-like, FCC-like, peak-like, and all temperature-fatigue labels remain provisional; failure-mode accessibility is also unresolved.",
        "29": "Machine counts and exact provenance are stored below; final committed HEAD is supplied by the handoff after commit.",
    }
    decision = {
        "schema": "joint_archetype_final_decision_v2", "created_utc": now(),
        "primary_classification": primary, "classifications": classifications,
        "qualifiers": qualifiers,
        "markdown_json_classification_agreement_key": canonical_sha({"classifications":classifications,"qualifiers":qualifiers}),
        "analytical_atlas_name": "CORRECTED_MECHANISM_GUIDED_RESPONSE_ATLAS",
        "joint_study_name": "FRACTURE_FATIGUE_RESPONSE_ATLAS_NOT_YET_ARCHETYPE_ATLAS",
        "solver_modified": False, "physics_calculations_rerun": False,
        "empirical_toughness_curve_used": False, "empirical_Paris_law_used": False,
        "tested_F2B_promoted": False, "two_compartment_model_class_rejected": False,
        "median_absolute_relative_error": median_errors,
        "archived_results_analyzed": int(len(inventory)),
        "admitted_physical_results": int(len(physical)),
        "analytical_atlas_points": int(len(atlas_frame)),
        "initial_sobol_points": INITIAL_ROWS, "adaptive_points": ADAPTIVE_ROWS,
        "new_physical_trajectories": 0, "new_2D_trajectories": 0,
        "right_censor_count": int(inventory.right_censored.fillna(False).sum()),
        "candidate_count": int(len(selected)), "superseded_candidate_count": 6,
        "ceiling_dominated_atlas_rows": int(atlas_frame.fatigue_ceiling_dominated.sum()),
        "asymptotic_rejected_atlas_rows": int(atlas_frame.barrier_or_state_asymptotic_artifact.sum()),
        "active_worker_count": 0,
        "response_class_counts": {str(k):int(v) for k,v in class_counts.items()},
        "mechanism_class_counts": {str(k):int(v) for k,v in mechanism_counts.items()},
        "trend_status_counts": {str(k):int(v) for k,v in trend_counts.items()},
        "cluster_stability_fraction": float(pd.read_csv(
            OUT / "archetype_cluster_stability.csv"
        ).stable_under_resampling.mean()),
        "source_HEAD": SOURCE_HEAD, "qualified_solver_HEAD": QUALIFIED_HEAD,
        "solver_sha256": SOLVER_SHA, "historical_HEAD": HIST_HEAD,
        "controller_state": "TERMINAL", "answers": answers,
    }
    write_json("joint_archetype_final_decision.json", decision)
    lines = [
        "# Joint fracture-fatigue archetype decision", "",
        f"Primary classification: `{primary}`", "",
        "Additional classifications: " + ", ".join(f"`{x}`" for x in classifications[1:]), "",
        "Qualifiers: " + ", ".join(f"`{x}`" for x in qualifiers), "",
        "The local analytical kernel is accurate for matched evolving-1D conditions. Transient-state "
        "coverage is partial and applied-to-local mechanical transfer remains unresolved. The V1 joint "
        "candidate set was contaminated by cooperative-renewal saturation and is invalidated. This is "
        "therefore a corrected mechanism-guided response atlas, not a learned surrogate, material "
        "calibration, or validated BCC/FCC/ceramic archetype atlas. No new physical trajectory was "
        "launched, no production physics changed, and temperature-dependent fatigue remains unvalidated.", "",
        "## Direct answers", "",
    ]
    for number in range(1,30):
        lines += [f"{number}. {answers[str(number)]}", ""]
    lines += ["## Machine summary", "",
              f"Archived inventory rows: {len(inventory)}; analytical atlas rows: {len(atlas_frame)}; "
              f"promoted joint candidates: {len(selected)}; superseded V1 candidates: 6; "
              f"new physical trajectories: 0; right censors: {decision['right_censor_count']}; "
              "controller: TERMINAL; active workers: 0.", "",
              f"Classification agreement key: `{decision['markdown_json_classification_agreement_key']}`", ""]
    (OUT / "joint_archetype_final_decision.md").write_text("\n".join(lines))
    return decision


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rows = load_candidate_rows()
    source_audit(rows)
    prospective_freeze(rows)
    pred, state, td, rd = exact_predictions(rows)
    inventory = archived_inventory()
    validation = cross_fidelity_validation(inventory, pred)
    bounds = search_bounds()
    atlas_frame = atlas(bounds)
    sobol, trends = sensitivity_and_trends(atlas_frame, bounds["historical_search_dimensions"])
    membership, stability = clustering(atlas_frame)
    # Persist descriptors again after deterministic cluster assignment.
    atlas_frame.to_parquet(OUT / "response_atlas.parquet", index=False)
    manifolds, joint, corr = nonuniqueness_and_joint(atlas_frame)
    selected = candidates(atlas_frame, rows)
    mono, fatigue = physical_validation(inventory, selected)
    gate = f2b_gate(rows)
    figures(inventory, pred, validation, atlas_frame, sobol, trends, selected, mono, fatigue)
    decision = final_decision(inventory, pred, validation, atlas_frame, selected, gate, trends)
    write_json("joint_archetype_verification.json", {
        "schema": "joint_archetype_verification_v2", "status": "PENDING_EXTERNAL_VERIFIER",
        "artifact_generation_complete": True, "decision_classification": decision["primary_classification"],
        "active_worker_count": 0, "controller_terminal": True,
    })
    print(json.dumps({"status":"BUILT","rows":len(atlas_frame),
                      "inventory":len(inventory),"candidates":len(selected)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("use --build")
    build()


if __name__ == "__main__":
    main()
