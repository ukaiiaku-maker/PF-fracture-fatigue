#!/usr/bin/env python3
"""Apply the single-coordinate K300 anchor to prospective fracture rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


STRESS_FIELDS = (
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--k300-results", type=Path, required=True)
    parser.add_argument(
        "--parent-k300-results",
        type=Path,
        help=(
            "Optional previously qualified K300 result table containing the "
            "canonical parent controls. This permits confirmation-only batches "
            "without rerunning the controls."
        ),
    )
    parser.add_argument("--anchor-plan", type=Path, required=True)
    parser.add_argument("--out-registry", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite value: {value!r}")
    return result


def finite_or(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def fingerprint(row: pd.Series, active_fields: list[str]) -> str:
    payload = {field: finite(row[field]) for field in active_fields}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.relative_tolerance < 0.25:
        raise ValueError("relative tolerance must lie in (0, 0.25)")
    registry = pd.read_csv(args.registry)
    results = pd.read_csv(args.k300_results)
    if args.parent_k300_results is not None:
        parent_results = pd.read_csv(args.parent_k300_results)
        results = pd.concat([results, parent_results], ignore_index=True)
    plan = pd.read_csv(args.anchor_plan).set_index("prospective_candidate_id")
    if not results.temperature_K.eq(300.0).all():
        raise RuntimeError("K300 results contain a non-300 K case")
    duplicate = results[results.candidate_id.duplicated(keep=False)]
    if not duplicate.empty:
        inconsistent = duplicate.groupby("candidate_id")["K_50um_MPa_sqrt_m"].nunique()
        if (inconsistent > 1).any():
            raise RuntimeError("K300 results contain inconsistent duplicate candidates")
        results = results.drop_duplicates("candidate_id", keep="first")
    k300 = results.set_index("candidate_id")["K_50um_MPa_sqrt_m"].astype(float)
    active_fields = [
        field
        for field in registry.columns
        if field
        in {
            "Tref_K",
            "cleave_G00_eV",
            "cleave_gT_eV_per_K",
            "cleave_sigc0_GPa",
            "cleave_sT_GPa_per_K",
            "cleave_exp_a",
            "cleave_exp_n",
            "cleave_floor_frac",
            "emit_G00_eV",
            "emit_gT_eV_per_K",
            "emit_sigc0_GPa",
            "emit_sT_GPa_per_K",
            "emit_exp_a",
            "emit_exp_n",
            "emit_floor_frac",
            "peierls_H0_eV",
            "peierls_activation_entropy_kB",
            "peierls_exp_a",
            "peierls_exp_n",
            "peierls_nu0_s",
            "taylor_H0_eV",
            "taylor_activation_entropy_kB",
            "taylor_exp_a",
            "taylor_exp_n",
            "taylor_nu0_s",
            "rho_source0_m2",
            "taylor_corr_rho_c_m2",
            "taylor_corr_scale",
            "c_blunt",
        }
    ]
    if len(active_fields) != 29:
        raise RuntimeError(f"expected 29 active fields, found {len(active_fields)}")

    parent_control = {
        "DBTT": "v913_prospective_dbtt_CENTER",
        "Peak-T": "v913_prospective_peakt_CENTER",
    }
    accepted: list[pd.Series] = []
    audits: list[dict[str, object]] = []
    for _, source_row in registry.iterrows():
        row = source_row.copy()
        cid = str(row.prospective_candidate_id)
        family = str(row.design_family)
        observed = finite(k300.loc[cid])
        target = finite(k300.loc[parent_control[family]])
        relative_error = observed / target - 1.0
        current_total = finite_or(row.get("anchor_lambda_total", 1.0), 1.0)
        required_step = target / observed
        proposed_total = current_total * required_step
        is_control = str(row.design_role) == "EXACT_CANONICAL_CENTER_CONTROL"
        if is_control or abs(relative_error) <= args.relative_tolerance:
            status = "PASS_UNCHANGED" if is_control else "PASS_WITHIN_TOLERANCE"
            step = 1.0
            accepted_flag = True
        else:
            bounds = plan.loc[cid]
            lower = finite(bounds.historical_envelope_lambda_min)
            upper = finite(bounds.historical_envelope_lambda_max)
            if proposed_total < lower or proposed_total > upper:
                status = "INFEASIBLE_K300_WITHIN_HISTORICAL_ENVELOPE"
                step = 1.0
                accepted_flag = False
            else:
                status = "ANCHOR_APPLIED_REQUIRES_POSTCHECK"
                step = required_step
                accepted_flag = True
                for field in STRESS_FIELDS:
                    row[field] = finite(row[field]) * step
                row["anchor_lambda_total"] = proposed_total
                row["parameter_fingerprint"] = fingerprint(row, active_fields)
                row["design_fingerprint"] = hashlib.sha256(
                    json.dumps(
                        {
                            "prospective_candidate_id": cid,
                            "parent_candidate_id": row.parent_candidate_id,
                            "parameter_fingerprint": row.parameter_fingerprint,
                            "anchor_lambda_total": proposed_total,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
        row["K300_anchor_status"] = status
        row["K300_parent_MPa_sqrt_m"] = target
        row["K300_input_MPa_sqrt_m"] = observed
        row["K300_input_relative_error"] = relative_error
        row["anchor_lambda_step"] = step
        row["anchor_lambda_total"] = finite_or(
            row.get("anchor_lambda_total", current_total), current_total
        )
        row["simulation_status"] = "K300_ANCHORED_NOT_GRID_RUN"
        if accepted_flag:
            accepted.append(row)
        audits.append(
            {
                "prospective_candidate_id": cid,
                "design_family": family,
                "design_role": row.design_role,
                "K300_parent_MPa_sqrt_m": target,
                "K300_input_MPa_sqrt_m": observed,
                "K300_input_relative_error": relative_error,
                "relative_tolerance": args.relative_tolerance,
                "anchor_lambda_before": current_total,
                "anchor_lambda_required_step": required_step,
                "anchor_lambda_proposed_total": proposed_total,
                "anchor_lambda_applied_step": step,
                "anchor_lambda_after": finite_or(
                    row.get("anchor_lambda_total", current_total), current_total
                ),
                "anchor_status": status,
                "accepted_for_next_iteration": accepted_flag,
                "analytic_change_F1": 0.0,
                "analytic_change_F2": 0.0,
                "analytic_change_F3": 0.0,
                "analytic_change_F4_frozen_intrinsic": 0.0,
                "physical_model_changed": False,
            }
        )
    output = pd.DataFrame(accepted)
    if output.parameter_fingerprint.duplicated().any():
        raise RuntimeError("anchoring produced duplicate physical fingerprints")
    args.out_registry.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_registry, index=False)
    pd.DataFrame(audits).to_csv(args.out_audit, index=False)
    counts = pd.Series([row["anchor_status"] for row in audits]).value_counts()
    print(
        "V913_K300_ANCHOR_COMPLETE "
        f"accepted={len(output)}/{len(registry)} statuses={counts.to_dict()} "
        f"out={args.out_registry}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
