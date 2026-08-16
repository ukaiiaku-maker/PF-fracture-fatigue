#!/usr/bin/env python3
"""Select inverse-designed information-gain confirmation fracture rows."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scripts.design_v913_prospective_fracture_causality as design
except ModuleNotFoundError:  # Support direct ``python scripts/...py`` execution.
    import design_v913_prospective_fracture_causality as design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-registry", type=Path, required=True)
    parser.add_argument("--response-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(args.primary_registry)
    response = pd.read_csv(args.response_summary)
    v1 = design.load_module(design.V1_SCRIPT, "_v913_confirmation_v1")
    focused_mod = design.load_module(design.FOCUSED_SCRIPT, "_v913_confirmation_focused")
    ExpFloorSurface, PTMechanism = v1.load_production_types(design.SOURCE)
    candidates, _, _, _ = v1.load_population(design.SOURCE)
    focused = pd.read_csv(design.FOCUSED / "focused_model_master.csv", low_memory=False)
    plastic = pd.read_csv(design.FOCUSED / "plastic_bottleneck_descriptors.csv", low_memory=False)
    focused = design.with_low_temperature_bottleneck(focused, plastic)
    pstats = design.robust_parameter_stats(candidates, design.VARY_FIELDS)
    cstats = design.coordinate_stats(focused)
    all_stats = design.robust_parameter_stats(
        candidates,
        tuple(
            dict.fromkeys(
                (
                    *design.VARY_FIELDS,
                    "cleave_sigc0_GPa",
                    "cleave_sT_GPa_per_K",
                    "emit_sigc0_GPa",
                )
            )
        ),
    ).reset_index()

    merged = primary.merge(response, left_on="prospective_candidate_id", right_on="candidate_id", suffixes=("", "__response"))
    coordinate_columns = [f"achieved__{name}" for name in design.COORDS]
    response_columns = [
        "DBTT_magnitude_MPa_sqrt_m",
        "peak_prominence_MPa_sqrt_m",
        "K_span_MPa_sqrt_m",
        "temperature_at_max_K",
    ]
    accepted: list[pd.Series] = []
    audits: list[dict[str, object]] = []
    for family, parent_id in design.CANONICAL.items():
        group = merged[
            merged.design_family.eq(family)
            & merged.design_role.ne("EXACT_CANONICAL_CENTER_CONTROL")
        ].copy()
        x = group[coordinate_columns].to_numpy(float)
        xscale = cstats.loc[list(design.COORDS), "robust_scale"].to_numpy(float)
        y = group[response_columns].to_numpy(float)
        yscale = np.nanstd(y, axis=0)
        yscale[~np.isfinite(yscale) | (yscale <= 1e-12)] = 1.0
        pairs = []
        for left, right in itertools.combinations(range(len(group)), 2):
            dx = float(np.linalg.norm((x[left] - x[right]) / xscale))
            dy = float(np.linalg.norm((y[left] - y[right]) / yscale))
            different = group.iloc[left].morphology_class != group.iloc[right].morphology_class
            score = dy / max(dx, 0.15) + (4.0 if different else 0.0)
            pairs.append((score, different, left, right, dx, dy))
        pairs.sort(reverse=True)
        parent = candidates[candidates.candidate_id.eq(parent_id)].iloc[0]
        family_targets: list[np.ndarray] = []
        for score, different, left, right, dx, dy in pairs:
            if len(family_targets) >= args.per_family:
                break
            target_vector = 0.5 * (x[left] + x[right])
            if any(np.linalg.norm((target_vector - old) / xscale) < 0.20 for old in family_targets):
                continue
            target = dict(zip(design.COORDS, target_vector))
            fitted, achieved, result = design.fit_target(
                parent,
                target,
                pstats,
                cstats,
                v1,
                ExpFloorSurface,
                PTMechanism,
            )
            rms, max_abs = design.target_quality(target, achieved, cstats)
            within = all(
                cstats.loc[name, "min"] <= achieved[name] <= cstats.loc[name, "max"]
                for name in design.COORDS
            )
            feasible = bool(result.success and rms <= 0.08 and max_abs <= 0.16 and within)
            left_id = str(group.iloc[left].prospective_candidate_id)
            right_id = str(group.iloc[right].prospective_candidate_id)
            number = len(family_targets) + 1
            candidate_id = f"v913_prospective_{family.lower().replace('-', '')}_confirm_{number:02d}"
            audits.append(
                {
                    "prospective_candidate_id": candidate_id,
                    "design_family": family,
                    "source_pair_left": left_id,
                    "source_pair_right": right_id,
                    "source_pair_morphology_left": group.iloc[left].morphology_class,
                    "source_pair_morphology_right": group.iloc[right].morphology_class,
                    "different_morphology_boundary": bool(different),
                    "information_gain_score": score,
                    "coordinate_distance_robust": dx,
                    "response_distance_standardized": dy,
                    "selection_rationale": (
                        "MIDPOINT_OF_MORPHOLOGY_BOUNDARY_WITH_HIGH_RESPONSE_DISAGREEMENT"
                        if different
                        else "MIDPOINT_OF_HIGH_RESPONSE_GRADIENT_PAIR"
                    ),
                    "inverse_design_feasible": feasible,
                    "inverse_design_residual_rms_robust": rms,
                    "inverse_design_residual_max_abs_robust": max_abs,
                    **{f"requested__{name}": target[name] for name in design.COORDS},
                    **{f"achieved__{name}": achieved[name] for name in design.COORDS},
                }
            )
            if not feasible:
                continue
            row = fitted.copy()
            row["prospective_candidate_id"] = candidate_id
            row["design_family"] = family
            row["design_role"] = "INFORMATION_GAIN_CONFIRMATION"
            row["parent_candidate_id"] = parent_id
            row["target_code"] = "CONFIRM_BOUNDARY"
            row["accepted_design_attempt_id"] = f"confirmation:{family}:{number}"
            row["parent_parameter_fingerprint"] = design.parameter_fingerprint(parent, v1.ACTIVE_FIELDS)
            row["parameter_fingerprint"] = design.parameter_fingerprint(row, v1.ACTIVE_FIELDS)
            row["design_fingerprint"] = design.stable_json_sha(
                {
                    "candidate": candidate_id,
                    "source_pair": [left_id, right_id],
                    "target": target,
                    "parameter_fingerprint": row.parameter_fingerprint,
                }
            )
            row["design_residual_rms_robust_units"] = rms
            row["design_residual_max_abs_robust_units"] = max_abs
            for name in design.COORDS:
                row[f"requested__{name}"] = target[name]
                row[f"achieved__{name}"] = achieved[name]
            row["simulation_status"] = "CONFIRMATION_DESIGNED_NOT_RUN"
            row["simulation_git_sha"] = design.SIM_SHA
            row["design_analysis_git_sha"] = design.git("rev-parse", "HEAD")
            row["historical_temperature_grid_K"] = "700;800;900;950;1000;1050;1100;1200;1300;1400"
            accepted.append(row)
            family_targets.append(target_vector)
    if len(accepted) != 2 * args.per_family:
        raise RuntimeError(f"only {len(accepted)} confirmation rows were feasible")
    registry = pd.DataFrame(accepted)
    base = [
        "prospective_candidate_id",
        "design_family",
        "design_role",
        "parent_candidate_id",
        "target_code",
        "accepted_design_attempt_id",
        "parent_parameter_fingerprint",
        "parameter_fingerprint",
        "design_fingerprint",
        "design_residual_rms_robust_units",
        "design_residual_max_abs_robust_units",
    ]
    coordinate_fields = [f"{prefix}__{name}" for prefix in ("requested", "achieved") for name in design.COORDS]
    metadata = ["simulation_status", "simulation_git_sha", "design_analysis_git_sha", "historical_temperature_grid_K"]
    registry = registry[[column for column in base + v1.ACTIVE_FIELDS + coordinate_fields + metadata if column in registry]]
    if registry.parameter_fingerprint.duplicated().any():
        raise RuntimeError("duplicate confirmation parameter fingerprint")
    registry.to_csv(args.out / "prospective_fracture_confirmation_registry.csv", index=False)
    pd.DataFrame(audits).to_csv(args.out / "prospective_fracture_confirmation_design_audit.csv", index=False)
    anchor_input = registry.copy()
    anchor_input["design_role"] = "FEASIBLE_PRIMARY"
    anchors = design.anchor_plan(anchor_input, candidates, all_stats)
    anchors.to_csv(args.out / "prospective_fracture_confirmation_K300_anchor_plan.csv", index=False)
    manifest = {
        "schema": "v913_prospective_information_gain_confirmation_design_v1",
        "confirmation_count": len(registry),
        "per_family": args.per_family,
        "selection_basis": "response-gradient and morphology-boundary information gain",
        "inverse_design_used": True,
        "physics_changed": False,
    }
    (args.out / "confirmation_design_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"V913_CONFIRMATION_DESIGN_COMPLETE rows={len(registry)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
