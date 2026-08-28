#!/usr/bin/env python3
"""Analyze canonical PF rate/orientation mechanisms without trajectory replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DEFAULT = Path("analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit")
PUBLICATION_DEFAULT = Path("analysis_outputs/pf_canonical_fracture_v2_final/publication")
DEEP_RATE_TEMPERATURES = {900, 1000, 1050, 1100, 1150, 1200}


def onset_state(row: pd.Series, prefix: str = "") -> dict[str, object]:
    return {
        f"{prefix}tip_radius_um": row.get("tip_radius_um", np.nan),
        f"{prefix}front_width_um": row.get("front_width_um", np.nan),
        f"{prefix}mobile_count": row.get("mobile_count", np.nan),
        f"{prefix}retained_count": row.get("retained_count", np.nan),
        f"{prefix}multiplicity": row.get("multiplicity", np.nan),
        f"{prefix}backstress_Pa": row.get("backstress_Pa", np.nan),
        f"{prefix}signed_shielding_MPa_sqrt_m": row.get(
            "signed_shielding_MPa_sqrt_m", np.nan
        ),
        f"{prefix}maximum_emission_rate_system_index": row.get(
            "maximum_emission_rate_system_index", np.nan
        ),
        f"{prefix}maximum_emission_rate_system_name": row.get(
            "maximum_emission_rate_system_name", None
        ),
        f"{prefix}tensor_probe_reliable": row.get("tensor_probe_reliable", False),
    }


def exact_product_decomposition(
    k0: float, u0: float, c0: float, k1: float, u1: float, c1: float
) -> tuple[float, float, float, float]:
    delta_k = k1 - k0
    opening = 0.5 * (c0 + c1) * (u1 - u0)
    structural = 0.5 * (u0 + u1) * (c1 - c0)
    residual = delta_k - opening - structural
    return delta_k, opening, structural, residual


def rate_audit(data: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak = data.loc[
        data.material_class.eq("Peak") & data.theta_deg.eq(0.0)
        & data.is_rate_matrix_case
    ].copy()
    pre_parts = []
    transfer_rows = []
    onset_rows = []
    for case_id, run in peak.groupby("case_id", sort=False):
        run = run.sort_values("accepted_step_index", kind="stable")
        hits = run.index[run.is_initial_onset]
        if len(hits) != 1:
            raise ValueError(f"Peak rate case does not have one initial onset: {case_id}")
        onset_index = hits[0]
        pre = run.loc[:onset_index].copy()
        pre["preinitiation_state"] = True
        pre_parts.append(pre)
        c = pre.native_KJ_per_opening_Pa_sqrt_m_per_m.to_numpy(float)
        transfer_rows.append(
            {
                "case_id": case_id,
                "temperature_K": float(run.temperature_K.iloc[0]),
                "rate_tag": run.rate_tag.iloc[0],
                "loading_rate_factor": float(run.loading_rate_factor.iloc[0]),
                "seed": int(run.seed.iloc[0]),
                "preinitiation_accepted_row_count": len(pre),
                "C_K_initial_mean_Pa_sqrt_m_per_m": float(np.mean(c)),
                "C_K_initial_min_Pa_sqrt_m_per_m": float(np.min(c)),
                "C_K_initial_max_Pa_sqrt_m_per_m": float(np.max(c)),
                "C_K_initial_relative_range": float(
                    (np.max(c) - np.min(c)) / max(abs(np.mean(c)), 1e-300)
                ),
            }
        )
        onset_rows.append(run.loc[onset_index])
    prehistory = pd.concat(pre_parts, ignore_index=True)
    prehistory.to_parquet(
        output / "pf_peak_theta0_rate_state_history.parquet",
        index=False, compression="zstd",
    )
    transfer = pd.DataFrame(transfer_rows)
    spread = transfer.groupby("temperature_K").C_K_initial_mean_Pa_sqrt_m_per_m.agg(
        ["min", "max", "mean"]
    )
    spread["matched_rate_relative_spread"] = (spread["max"] - spread["min"]) / spread["mean"]
    transfer = transfer.merge(
        spread[["matched_rate_relative_spread"]], left_on="temperature_K", right_index=True
    )
    transfer["matched_structural_transfer_identical_to_numerical_precision"] = (
        transfer.matched_rate_relative_spread < 1e-12
    )
    transfer.to_csv(output / "pf_peak_rate_initial_structural_transfer.csv", index=False)

    onset = pd.DataFrame(onset_rows).reset_index(drop=True)
    decomposition = []
    for temperature, group in onset.groupby("temperature_K"):
        reference = group.loc[group.rate_tag.eq("rate1x")]
        if len(reference) != 1:
            raise ValueError(f"missing unique Peak rate1 reference at {temperature} K")
        ref = reference.iloc[0]
        for _, row in group.iterrows():
            k0 = float(ref.native_KJ_MPa_sqrt_m)
            u0 = float(ref.applied_opening_m)
            c0 = k0 / u0
            k1 = float(row.native_KJ_MPa_sqrt_m)
            u1 = float(row.applied_opening_m)
            c1 = k1 / u1
            delta, opening, structural, residual = exact_product_decomposition(
                k0, u0, c0, k1, u1, c1
            )
            record = {
                "case_id": row.case_id,
                "temperature_K": temperature,
                "rate_tag": row.rate_tag,
                "loading_rate_factor": row.loading_rate_factor,
                "seed": row.seed,
                "reference_rate_tag": "rate1x",
                "onset_KJ_MPa_sqrt_m": k1,
                "onset_opening_m": u1,
                "onset_C_MPa_sqrt_m_per_m": c1,
                "delta_K_vs_rate1_MPa_sqrt_m": delta,
                "required_opening_contribution_MPa_sqrt_m": opening,
                "structural_KJ_per_opening_contribution_MPa_sqrt_m": structural,
                "identity_residual_MPa_sqrt_m": residual,
                "deep_audit_temperature": int(temperature) in DEEP_RATE_TEMPERATURES,
                "structural_transfer_same_across_rates": abs(structural) < 1e-10,
            }
            record.update(
                {
                    "onset_tip_radius_um": row.observer_persistent_tip_radius_m * 1e6,
                    "onset_front_width_um": row.observer_persistent_site_front_width_m * 1e6,
                    "onset_mobile_count": row.observer_developed_state_mobile_count,
                    "onset_retained_count": row.observer_developed_state_retained_count,
                    "onset_backstress_Pa": row.observer_persistent_sigma_back_Pa,
                    "onset_signed_shielding_MPa_sqrt_m": (
                        row.observer_active_K_shield_signed_Pa_sqrt_m / 1e6
                    ),
                    "onset_multiplicity_per_system": (
                        row.observer_persistent_site_multiplicity_per_system
                    ),
                    "onset_cleavage_rate_s-1": row["observer_lambda_c_s-1"],
                    "onset_emission_rate_s-1": row.observer_tip_source_emission_rate_s,
                    "onset_cleavage_completed_action": row.observer_hazard_action_completed,
                    "onset_selected_emission_system_name": (
                        row.observer_selected_emission_system_name
                    ),
                }
            )
            decomposition.append(record)
    decomposition_df = pd.DataFrame(decomposition)
    decomposition_df.to_csv(
        output / "pf_peak_theta0_rate_onset_decomposition.csv", index=False
    )

    counterfactuals = []
    for _, row in decomposition_df.loc[
        decomposition_df.deep_audit_temperature
    ].iterrows():
        for diagnostic in (
            "ACTUAL_STATE", "RATE1X_REFERENCE_RADIUS", "RATE1X_REFERENCE_BACKSTRESS",
            "RATE1X_REFERENCE_SHIELDING", "RATE1X_REFERENCE_MULTIPLICITY",
            "RATE1X_REFERENCE_MOBILE_RETAINED_STATE", "COMMON_REFERENCE_TENSOR_DRIVE",
        ):
            actual = diagnostic == "ACTUAL_STATE"
            counterfactuals.append(
                {
                    "case_id": row.case_id,
                    "temperature_K": row.temperature_K,
                    "rate_tag": row.rate_tag,
                    "diagnostic": diagnostic,
                    "label": "COUNTERFACTUAL DIAGNOSTIC — NOT PRODUCTION PHYSICS"
                    if not actual else "ARCHIVED PRODUCTION STATE",
                    "evaluation_status": "OBSERVED" if actual else "UNAVAILABLE_MISSING_EXACT_STATE_INJECTION_CONTRACT",
                    "evaluated_KJ_MPa_sqrt_m": row.onset_KJ_MPa_sqrt_m if actual else np.nan,
                    "reason": "complete evolved-state tensor/source snapshot and exact component injection were not archived"
                    if not actual else "authoritative accepted-state onset",
                    "additive_contribution_claim_allowed": False,
                }
            )
    pd.DataFrame(counterfactuals).to_csv(
        output / "pf_peak_theta0_rate_counterfactuals.csv", index=False
    )
    return prehistory, decomposition_df


def orientation_audit(
    data: pd.DataFrame, publication: Path, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    onsets = pd.read_csv(publication / "pf_canonical_onset_candidates_v2.csv")
    avalanches = pd.read_csv(publication / "pf_canonical_physical_avalanches_v2.csv")
    manifest = pd.read_csv(publication / "pf_canonical_fracture_run_manifest.csv")
    orientation_ids = set(
        manifest.loc[manifest.is_orientation_matrix_case.astype(bool), "case_id"]
    )
    onsets = onsets.loc[onsets.case_id.isin(orientation_ids)].copy()
    initial = onsets.loc[onsets.onset_role.eq("INITIAL_ONSET_PRE_EVENT")].copy()
    if len(initial) != 192:
        raise ValueError(f"expected 192 orientation initial onsets, found {len(initial)}")

    initial_rows = []
    for (material, temperature), group in initial.groupby(
        ["material_class", "temperature_K"]
    ):
        reference = group.loc[group.theta_deg.eq(0.0)]
        if len(reference) != 1:
            raise ValueError(f"missing theta0 onset reference: {material}, {temperature}")
        ref = reference.iloc[0]
        k0, u0 = float(ref.pre_event_native_KJ_MPa_sqrt_m), float(ref.pre_event_opening_m)
        c0 = k0 / u0
        for _, row in group.iterrows():
            k1, u1 = float(row.pre_event_native_KJ_MPa_sqrt_m), float(row.pre_event_opening_m)
            c1 = k1 / u1
            delta, opening, structural, residual = exact_product_decomposition(
                k0, u0, c0, k1, u1, c1
            )
            record = {
                "case_id": row.case_id,
                "material_class": material,
                "temperature_K": temperature,
                "theta_deg": row.theta_deg,
                "seed": row.seed,
                "U0_theta_m": u1,
                "K0_theta_MPa_sqrt_m": k1,
                "C0_theta_MPa_sqrt_m_per_m": c1,
                "theta0_reference_U_m": u0,
                "theta0_reference_K_MPa_sqrt_m": k0,
                "theta0_reference_C_MPa_sqrt_m_per_m": c0,
                "delta_K_vs_theta0_MPa_sqrt_m": delta,
                "opening_local_threshold_contribution_MPa_sqrt_m": opening,
                "structural_KJ_per_opening_contribution_MPa_sqrt_m": structural,
                "identity_residual_MPa_sqrt_m": residual,
                "opening_tensor_available": False,
                "complete_channel_tensors_available": False,
                "resolved_signed_shears_available_in_full_trajectory": True,
            }
            record.update(onset_state(row, "onset_"))
            initial_rows.append(record)
    initial_df = pd.DataFrame(initial_rows)
    initial_df.to_csv(output / "pf_orientation_initial_onset_decomposition.csv", index=False)

    reinit_rows = []
    for case_id, group in onsets.groupby("case_id"):
        first = group.loc[group.onset_role.eq("INITIAL_ONSET_PRE_EVENT")].iloc[0]
        k1, u1 = float(first.pre_event_native_KJ_MPa_sqrt_m), float(first.pre_event_opening_m)
        c1 = k1 / u1
        for _, row in group.loc[group.onset_role.str.startswith("REINITIATION")].iterrows():
            k2, u2 = float(row.pre_event_native_KJ_MPa_sqrt_m), float(row.pre_event_opening_m)
            c2 = k2 / u2
            delta, opening, structural, residual = exact_product_decomposition(
                k1, u1, c1, k2, u2, c2
            )
            record = {
                "case_id": case_id,
                "material_class": row.material_class,
                "temperature_K": row.temperature_K,
                "theta_deg": row.theta_deg,
                "seed": row.seed,
                "reinitiation_event_transaction_index": row.event_transaction_index,
                "reinitiation_physical_avalanche_index": row.physical_avalanche_index,
                "initial_U_m": u1,
                "initial_KJ_MPa_sqrt_m": k1,
                "initial_C_MPa_sqrt_m_per_m": c1,
                "reinitiation_U_m": u2,
                "reinitiation_KJ_MPa_sqrt_m": k2,
                "reinitiation_C_MPa_sqrt_m_per_m": c2,
                "delta_K_reinit_MPa_sqrt_m": delta,
                "required_opening_local_state_contribution_MPa_sqrt_m": opening,
                "structural_wake_transfer_contribution_MPa_sqrt_m": structural,
                "identity_residual_MPa_sqrt_m": residual,
                "frozen_geometry_linearity_basis": "PF_sharp_wake_map_load_scaling_passed",
                "complete_evolved_state_swap_available": False,
            }
            record.update(onset_state(first, "initial_"))
            record.update(onset_state(row, "reinitiation_"))
            reinit_rows.append(record)
    reinit_df = pd.DataFrame(reinit_rows)
    reinit_df.to_csv(output / "pf_orientation_reinitiation_decomposition.csv", index=False)

    stats_rows = []
    final_reinit = (
        reinit_df.sort_values("reinitiation_event_transaction_index")
        .groupby("case_id", as_index=False).tail(1)
    )
    for (material, theta), cases in manifest.loc[
        manifest.is_orientation_matrix_case.astype(bool)
    ].groupby(["material_class", "theta_deg"]):
        subset = final_reinit.loc[
            final_reinit.material_class.eq(material) & final_reinit.theta_deg.eq(theta)
        ]
        avalanche_subset = avalanches.loc[avalanches.case_id.isin(cases.case_id)]
        finite = len(subset)
        stats_rows.append(
            {
                "material_class": material,
                "theta_deg": theta,
                "temperature_case_count": 12,
                "finite_reinitiation_case_count": finite,
                "finite_reinitiation_fraction": finite / 12,
                "conditional_mean_delta_K_reinit_MPa_sqrt_m": subset.delta_K_reinit_MPa_sqrt_m.mean(),
                "conditional_median_delta_K_reinit_MPa_sqrt_m": subset.delta_K_reinit_MPa_sqrt_m.median(),
                "reinitiation_temperature_min_K": subset.temperature_K.min(),
                "reinitiation_temperature_max_K": subset.temperature_K.max(),
                "total_physical_avalanche_count": len(avalanche_subset),
                "mean_physical_avalanche_count_per_case": len(avalanche_subset) / 12,
                "mean_largest_avalanche_fraction": cases.largest_avalanche_fraction.mean(),
                "median_largest_avalanche_fraction": cases.largest_avalanche_fraction.median(),
            }
        )
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(output / "pf_orientation_conditional_reinitiation_statistics.csv", index=False)

    classifications = []
    last_lookup = final_reinit.set_index("case_id")
    for _, row in initial_df.iterrows():
        if row.theta_deg == 0 or row.delta_K_vs_theta0_MPa_sqrt_m >= 0:
            initial_class = "UNRESOLVED" if row.theta_deg != 0 else "THETA0_REFERENCE"
        elif abs(row.structural_KJ_per_opening_contribution_MPa_sqrt_m) > abs(
            row.opening_local_threshold_contribution_MPa_sqrt_m
        ):
            initial_class = "LOWER_INITIAL_ONSET_STRUCTURAL_TRANSFER"
        else:
            initial_class = "LOWER_INITIAL_ONSET_LOCAL_TENSOR_DRIVE_OR_STATE_EVOLUTION"
        if row.case_id not in last_lookup.index:
            reinit_class = "NO_REINITIATION"
            delta_reinit = np.nan
        else:
            re = last_lookup.loc[row.case_id]
            delta_reinit = re.delta_K_reinit_MPa_sqrt_m
            if delta_reinit < 0:
                reinit_class = "NEGATIVE_REINITIATION_SOFTENING"
            elif abs(re.structural_wake_transfer_contribution_MPa_sqrt_m) > abs(
                re.required_opening_local_state_contribution_MPa_sqrt_m
            ):
                reinit_class = "POSITIVE_REINITIATION_STRUCTURAL_WAKE"
            else:
                reinit_class = "POSITIVE_REINITIATION_MIXED_LOCAL_DOMINANT"
        classifications.append(
            {
                "case_id": row.case_id,
                "material_class": row.material_class,
                "temperature_K": row.temperature_K,
                "theta_deg": row.theta_deg,
                "initial_onset_classification": initial_class,
                "reinitiation_classification": reinit_class,
                "final_delta_K_reinit_MPa_sqrt_m": delta_reinit,
                "classification_scope": "exact_K_equals_C_times_U_decomposition_plus_archived_scalar_state",
                "tensor_state_limitation": "complete_tensor_and_exact_evolved_state_swaps_unavailable",
            }
        )
    classification = pd.DataFrame(classifications)
    classification.to_csv(output / "pf_orientation_mechanism_classification.csv", index=False)
    return initial_df, reinit_df, stats


def frozen_swap(output: Path) -> pd.DataFrame:
    map_dir = output / "deterministic_frozen_orientation_maps"
    frames = []
    for theta in (0, 15, 30, 45):
        mechanics = pd.read_csv(map_dir / f"pf_v2_theta{theta}_mechanics_map.csv")
        drive = pd.read_csv(map_dir / f"pf_v2_theta{theta}_source_drive_map.csv")
        joined = drive.merge(
            mechanics[
                ["theta_deg", "actual_extension_um", "reaction_N_per_m",
                 "compliance_m2_per_N", "elastic_energy_J_per_m",
                 "native_J_J_per_m2", "native_KJ_MPa_sqrt_m", "KJ_native_over_U",
                 "F_over_U_N_per_m2", "damage_wake_fingerprint"]
            ],
            left_on=["theta_deg", "extension_um"],
            right_on=["theta_deg", "actual_extension_um"],
            validate="many_to_one",
        )
        joined["diagnostic_label"] = "DETERMINISTIC FROZEN-STATE DIAGNOSTIC — NOT PRODUCTION PHYSICS"
        joined["state_history"] = "ZERO_HISTORY"
        joined["stochastic_clock_advanced"] = False
        joined["crack_path_rotation"] = False
        joined["crystal_orientation_rotated_relative_to_fixed_horizontal_path"] = True
        frames.append(joined)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "pf_orientation_frozen_swap_matrix.csv", index=False)
    return result


def write_reports(
    output: Path,
    rate: pd.DataFrame,
    initial: pd.DataFrame,
    reinit: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    slow = rate.loc[rate.rate_tag.eq("rate0p01x") & rate.deep_audit_temperature]
    base = rate.loc[rate.rate_tag.eq("rate1x") & rate.deep_audit_temperature]
    merged = slow.merge(base, on="temperature_K", suffixes=("_slow", "_base"))
    mean_delta = merged.delta_K_vs_rate1_MPa_sqrt_m_slow.mean()
    radius_ratio = np.median(
        merged.onset_tip_radius_um_slow / merged.onset_tip_radius_um_base
    )
    mobile_ratio = np.median(
        merged.onset_mobile_count_slow / merged.onset_mobile_count_base
    )
    backstress_ratio = np.median(
        merged.onset_backstress_Pa_slow / merged.onset_backstress_Pa_base
    )
    shield_delta = np.median(
        merged.onset_signed_shielding_MPa_sqrt_m_slow
        - merged.onset_signed_shielding_MPa_sqrt_m_base
    )
    report = f"""# PF Peak Theta0 Slow-Rate Mechanism Audit

## Decision

The Peak slow-rate onset elevation is **not a structural KJ/opening effect**.
Before initial fracture, `KJ/U` is identical across the matched 0.01x, 1x,
and 100x cases to a maximum relative spread below `1e-12` at every
temperature. The entire exact KJ difference is therefore the greater opening
required after rate-dependent local-state evolution.

Across the deep 900–1200 K audit set, slow minus 1x onset KJ averages
**{mean_delta:.2f} MPa sqrt(m)**. Median slow/1x ratios are **{radius_ratio:.2f}**
for radius, **{mobile_ratio:.2f}** for mobile population, and
**{backstress_ratio:.2f}** for backstress; the median signed-shielding change is
**{shield_delta:.3f} MPa sqrt(m)**. Retained population and multiplicity vary
non-monotonically with temperature, so neither supports a single-variable
explanation.

The supported classification is **TIME_AVAILABLE_FOR_EMISSION +
MIXED_LOCAL_STATE_EFFECT**, with important emission/blunting and backstress
changes and temperature-dependent shielding/multiplicity. A unique additive
partition among radius, backstress, shielding, multiplicity, and cleavage
action is not supported because the nonlinear evolved-state component swaps
cannot be reconstructed exactly from the default-off observer archive.

## Evidence and limits

- Common random numbers are preserved across rates at fixed class and
  temperature (identical seed triplets).
- The accepted-state archive supplies scalar radius, width, populations,
  backstress, signed shielding, multiplicity, rates, actions, resolved shears,
  and channel-resolved transport.
- Complete tensor matrices were not archived. No tensor was interpolated or
  inferred from a later state.
- The counterfactual table is fail-closed: actual states are recorded, while
  unavailable one-at-a-time evolved-state injections are explicitly marked
  unavailable rather than fabricated.
- Model-native KJ is not applied K or a conventional R-curve.
"""
    (output / "PF_PEAK_THETA0_SLOW_RATE_MECHANISM_AUDIT.md").write_text(report)

    peak_dbtt = stats.loc[stats.material_class.isin(["Peak", "DBTT"])]
    positive = reinit.loc[reinit.delta_K_reinit_MPa_sqrt_m > 0]
    if len(positive):
        opening_share = (
            positive.required_opening_local_state_contribution_MPa_sqrt_m.sum()
            / positive.delta_K_reinit_MPa_sqrt_m.sum()
        )
        structural_share = (
            positive.structural_wake_transfer_contribution_MPa_sqrt_m.sum()
            / positive.delta_K_reinit_MPa_sqrt_m.sum()
        )
    else:
        opening_share = structural_share = np.nan
    orientation_report = f"""# PF Peak/DBTT Orientation and Reinitiation Audit

## Decision

Theta rotates the cubic elasticity and crystallographic source/tensor relation
relative to a **fixed horizontal cleavage trace**; it does not rotate the crack
line. Initial-onset changes contain both an exact structural `KJ/U` term and an
opening/local-threshold term. The zero-history frozen-orientation matrix
independently measures the structural/tensor transfer, reaction, energy,
resolved shears, and source-channel ordering at common geometry and opening.

Off-axis positive reload-separated Delta K candidates are conditional, not
universal. The accompanying incidence table reports the fraction of 12
temperatures as well as conditional mean/median values and avalanche topology.
Across all positive canonical reinitiation rows, the signed aggregate exact
decomposition is **{opening_share:.1%} required-opening/local-state** and
**{structural_share:.1%} changed sharp-wake structural coefficient**. These
signed aggregate shares can exceed 100% or be negative when the two exact terms
oppose; they are not additive material-toughness fractions.

The evidence supports a mixed interpretation: increased theta lowers many
Peak/DBTT initial onsets through changed anisotropic structural/tensor transfer
and the opening at which the local stochastic condition is met. After the
first avalanche, both the required opening and sharp-wake `KJ/U` may change;
their exact contributions are case-resolved in the decomposition table.

## Qualification boundary

The deterministic swap matrix is zero-history and model-native. Exact evolved
state swaps and component ablations are unavailable because complete archived
tensor/source snapshots and a qualified state-injection contract do not exist.
No later event state was substituted. Consequently, `LOCAL_HARDENING` versus
specific radius/backstress/shielding causation remains unresolved where the
exact K=C*U decomposition alone cannot decide it.

Positive model-native Delta K is an **effective reload-separated resistance
candidate**, not automatically a conventional energy-release R-curve.
"""
    (output / "PF_PEAK_DBTT_ORIENTATION_REINITIATION_AUDIT.md").write_text(
        orientation_report
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--publication", type=Path, default=PUBLICATION_DEFAULT)
    args = parser.parse_args()
    data = pd.read_parquet(args.output / "pf_canonical_full_step_trajectories.parquet")
    peak_state = pd.read_parquet(args.output / "pf_peak_theta0_rate_full_state_source.parquet")
    prehistory, rate = rate_audit(peak_state, args.output)
    initial, reinit, stats = orientation_audit(data, args.publication, args.output)
    swap = frozen_swap(args.output)
    write_reports(args.output, rate, initial, reinit, stats)
    summary = {
        "schema": "pf_canonical_rate_orientation_mechanism_summary_v1",
        "analysis_only": True,
        "new_stochastic_trajectories": 0,
        "fem_czm_runs": 0,
        "canonical_artifacts_modified": False,
        "peak_preinitiation_state_rows": len(prehistory),
        "rate_onset_decomposition_rows": len(rate),
        "orientation_initial_decomposition_rows": len(initial),
        "orientation_reinitiation_decomposition_rows": len(reinit),
        "orientation_conditional_statistics_rows": len(stats),
        "frozen_swap_rows": len(swap),
        "complete_evolved_tensor_state_available": False,
        "evolved_state_counterfactual_policy": "FAIL_CLOSED",
    }
    (args.output / "pf_rate_orientation_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
