#!/usr/bin/env python3
"""Select 6--8 prospective fracture rows for exact fatigue transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-summary", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--joint-master", type=Path, required=True)
    parser.add_argument("--joint-pareto", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = pd.read_csv(args.response_summary)
    registry = pd.read_csv(args.registry)
    joint = pd.read_csv(args.joint_master)
    pareto = pd.read_csv(args.joint_pareto)
    data = response.merge(
        registry,
        left_on="candidate_id",
        right_on="prospective_candidate_id",
        suffixes=("", "__registry"),
        validate="one_to_one",
    )
    selected: list[dict[str, object]] = []
    used: set[str] = set()

    def add(row: pd.Series, category: str, reason: str) -> None:
        candidate_id = str(row.candidate_id)
        if candidate_id in used:
            return
        used.add(candidate_id)
        selected.append(
            {
                "candidate_id": candidate_id,
                "parameter_fingerprint": row.parameter_fingerprint,
                "design_family": row.design_family,
                "design_role": row.design_role,
                "selection_category": category,
                "selection_reason": reason,
                "morphology_class": row.morphology_class,
                "DBTT_magnitude_MPa_sqrt_m": row.DBTT_magnitude_MPa_sqrt_m,
                "peak_prominence_MPa_sqrt_m": row.peak_prominence_MPa_sqrt_m,
                "K_span_MPa_sqrt_m": row.K_span_MPa_sqrt_m,
                "K300_MPa_sqrt_m": row.K300_MPa_sqrt_m,
                "F1": row.F1,
                "F2": row.F2,
                "F3": row.F3,
                "F4": row.F4,
            }
        )

    for family in ("DBTT", "Peak-T"):
        center = data[data.design_family.eq(family) & data.design_role.eq("EXACT_CANONICAL_CENTER_CONTROL")]
        if len(center) != 1:
            raise RuntimeError(f"missing exact {family} control")
        add(center.iloc[0], "MECHANISTIC_CONTROL", f"exact canonical {family} control")

    dbtt = data[data.design_family.eq("DBTT") & data.design_role.ne("EXACT_CANONICAL_CENTER_CONTROL")]
    peak = data[data.design_family.eq("Peak-T") & data.design_role.ne("EXACT_CANONICAL_CENTER_CONTROL")]
    add(dbtt.sort_values("DBTT_magnitude_MPa_sqrt_m", ascending=False).iloc[0], "BEST_DBTT_CAUSAL_SUCCESS", "largest prospective DBTT magnitude at qualified K300")
    add(dbtt.sort_values("K_span_MPa_sqrt_m").iloc[0], "DBTT_TO_WEAK_BOUNDARY", "smallest temperature-response span in DBTT-centered design")
    add(peak.sort_values("peak_prominence_MPa_sqrt_m", ascending=False).iloc[0], "BEST_PEAKT_CAUSAL_SUCCESS", "largest retained prospective Peak-T prominence")
    add(peak.sort_values("peak_prominence_MPa_sqrt_m").iloc[0], "PEAKT_REMOVAL_CONTROL", "smallest Peak-T prominence after controlled perturbation")

    noncontrol = data[data.design_role.ne("EXACT_CANONICAL_CENTER_CONTROL")]
    add(noncontrol.sort_values("F4").iloc[0], "LOW_PLASTIC_BOTTLENECK_CONTROL", "extreme low F4 control within qualified design")

    pareto_ids = set(pareto.loc[pareto.pareto_nondominated.astype(bool), "candidate_id"].astype(str))
    q = joint[joint.candidate_id.astype(str).isin(pareto_ids)].dropna(
        subset=["delta_mu_emit_minus_cleave", "activation_window_overlap_Oce", "delta_Theta_sigma_900"]
    )
    if len(q):
        target = q[["delta_mu_emit_minus_cleave", "activation_window_overlap_Oce", "delta_Theta_sigma_900"]].to_numpy(float)
        pool = data[~data.candidate_id.isin(used)].copy()
        values = pool[["F1", "F2", "F3"]].to_numpy(float)
        scale = np.nanstd(joint[["delta_mu_emit_minus_cleave", "activation_window_overlap_Oce", "delta_Theta_sigma_900"]].to_numpy(float), axis=0)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        distance = np.min(np.linalg.norm((values[:, None, :] - target[None, :, :]) / scale, axis=2), axis=1)
        row = pool.iloc[int(np.argmin(distance))]
        add(row, "BEST_JOINT_BALANCE_PREDICTED", "nearest prospective F1-F3 geometry to the model-internal existing Pareto set")

    if not 6 <= len(selected) <= 8:
        raise RuntimeError(f"selection count outside requested range: {len(selected)}")
    output = pd.DataFrame(selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    (args.out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "schema": "v913_prospective_joint_candidate_selection_v1",
                "selected_count": len(output),
                "selection_is_pareto_categorical": True,
                "single_scalar_score_used": False,
                "experimental_agreement_claimed": False,
                "realism_basis": "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"V913_PROSPECTIVE_JOINT_SELECTION_COMPLETE selected={len(output)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
