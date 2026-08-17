#!/usr/bin/env python3
"""Fail-closed acceptance verifier for the final joint slope study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


CORE = (
    "paris_slope_master.csv", "local_paris_slope_curves.csv",
    "fracture_barrier_detailed_descriptors.csv", "fracture_hazard_sensitivity.csv",
    "fatigue_hazard_sensitivity.csv", "fracture_fatigue_hazard_sensitivity_master.csv",
    "prospective_slope_design_registry.csv", "prospective_slope_design_audit.csv",
    "prospective_slope_fracture_results.csv", "prospective_slope_fatigue_results.csv",
    "joint_slope_pareto_front.csv",
)
FIGURES = (
    "paris_slopes_all_shared_candidates", "local_m_vs_deltaK",
    "m_HCF_vs_cleavage_barrier_derivative", "delta_m_vs_cleavage_curvature",
    "measured_vs_hazard_predicted_paris_slope", "fracture_dKdT_measured_vs_hazard_predicted",
    "fracture_fatigue_common_sensitivity_map", "paris_slope_design_chart",
    "fracture_design_chart", "prospective_slope_candidate_fatigue_curves",
    "prospective_slope_candidate_fracture_curves", "joint_slope_pareto_map",
)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args();root=args.root
    failures=[]
    for name in CORE:
        path=root/name
        if not path.exists() or path.stat().st_size==0: failures.append(f"missing core {name}")
    for stem in FIGURES:
        for suffix in (".png",".pdf","_plot_data.csv"):
            path=root/f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size==0: failures.append(f"missing figure artifact {path.name}")
    for name in ("representative_joint_slope_six_panel.png","representative_joint_slope_six_panel.pdf","representative_joint_slope_six_panel_plot_data.csv","JOINT_FRACTURE_FATIGUE_PARIS_SLOPE_REPORT.md","joint_paris_slope_final_manifest.json"):
        if not (root/name).exists() or (root/name).stat().st_size==0: failures.append(f"missing {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    fracture=pd.read_csv(root/"prospective_slope_fracture_results.csv")
    fatigue=pd.read_csv(root/"prospective_slope_fatigue_results.csv")
    status=pd.read_csv(root/"prospective_slope_all_run_status.csv")
    hazard=pd.read_csv(root/"fracture_hazard_sensitivity.csv")
    prospective_hazard=hazard[hazard.sensitivity_operator.eq("EXACT_V913_FIXED_PEAK_COUPLED_TRAJECTORY_REPLAY_CENTERED_DIFFERENCE")] if "sensitivity_operator" in hazard else hazard.iloc[0:0]
    overlap=pd.read_csv(root/"prospective_slope_accelerated_explicit_overlap.csv")
    if len(fracture)!=30 or not fracture.fracture_qualified_for_fatigue.astype(bool).all(): failures.append("not all 30 fracture rows qualified")
    if len(fatigue)!=30 or set(fatigue.candidate_id)!=set(fracture.candidate_id): failures.append("fatigue population is not exact 30-row transfer")
    if not fatigue.accelerated_finite_points.ge(4).all(): failures.append("fewer than four finite HCF points")
    if not fatigue.explicit_finite_points.ge(1).all(): failures.append("missing explicit LCF point")
    if fatigue.partial_or_numerical_unresolved.sum()!=0: failures.append("partial/numerical results remain")
    if status.developed_da_dN_m_per_cycle[~status.status_class.eq("developed_target_reached")].notna().any(): failures.append("censor or partial assigned artificial rate")
    if len(prospective_hazard)!=330 or not prospective_hazard.baseline_state_replay_valid.astype(bool).all(): failures.append("exact monotonic hazard grid incomplete or invalid")
    if prospective_hazard.baseline_hazard_relative_error.max()>1e-4: failures.append("monotonic hazard replay error exceeds gate")
    overlap_attempts=status[status.integration_mode.eq("explicit")].candidate_id.nunique()
    if overlap_attempts!=30: failures.append("accelerated/explicit overlap attempts incomplete")
    if overlap.candidate_id.nunique()<24 or not np.isfinite(overlap.explicit_over_accelerated_rate_ratio).all(): failures.append("finite accelerated/explicit parity evidence incomplete")
    report=(root/"JOINT_FRACTURE_FATIGUE_PARIS_SLOPE_REPORT.md").read_text()
    numbered = [int(value) for value in re.findall(r"(?m)^(\d+)\. \*\*", report)]
    if numbered != list(range(1, 17)): failures.append("report does not answer all 16 questions")
    manifest=json.loads((root/"joint_paris_slope_final_manifest.json").read_text())
    if manifest.get("physics_changed") is not False or manifest.get("experimental_reference_status")!="NO_DEFENSIBLE_QUANTITATIVE_LOCAL_ENVELOPE": failures.append("manifest semantics invalid")
    if failures: raise SystemExit("\n".join(failures))
    print(f"V913_JOINT_PARIS_SLOPE_VERIFIED candidates=30 fracture_cases=330 finite_overlap_candidates={overlap.candidate_id.nunique()} figures=12 physics_changed=false")
    return 0


if __name__=="__main__": raise SystemExit(main())
