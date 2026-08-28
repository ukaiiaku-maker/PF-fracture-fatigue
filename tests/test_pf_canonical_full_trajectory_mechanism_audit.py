from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit"
PUB = ROOT / "analysis_outputs/pf_canonical_fracture_v2_final/publication"
RAW = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "canonical_pf_fracture_v2_20260826"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@pytest.fixture(scope="module")
def trajectories() -> pd.DataFrame:
    return pd.read_parquet(OUT / "pf_canonical_full_step_trajectories.parquet")


def test_all_288_cases_and_steps_hashes_are_verified():
    record = json.loads((OUT / "pf_canonical_full_trajectory_manifest.json").read_text())
    index = pd.read_csv(OUT / "pf_canonical_full_step_trajectories_index.csv")
    assert record["canonical_case_count"] == record["unique_case_count"] == 288
    assert record["all_steps_sha256_verified"] is True
    assert len(index) == index.case_id.nunique() == 288
    for row in index.itertuples(index=False):
        assert digest(RAW / row.raw_steps_relative_path) == row.raw_steps_sha256


def test_canonical_plot_temperature_counts_and_shared_case_identity():
    atlas = json.loads((OUT / "pf_canonical_full_KJ_atlas_manifest.json").read_text())
    orientation = [r for r in atlas["figure_records"]
                   if r["category"] == "canonical_orientation_individual"]
    rates = [r for r in atlas["figure_records"]
             if r["category"] == "canonical_rate_individual"]
    assert len(orientation) == 32  # full + early
    assert len(rates) == 24
    assert all(row["temperature_count"] == 12 for row in orientation + rates)
    assert atlas["canonical_case_count"] == 288


def test_trajectory_chronology_and_duplicate_extension_are_preserved(trajectories):
    for _, run in trajectories.groupby("case_id", sort=False):
        assert np.array_equal(run.accepted_step_index.to_numpy(), np.arange(len(run)))
        assert np.all(np.diff(run.physical_time_s.to_numpy(float)) >= -1e-12)
    index = pd.read_csv(OUT / "pf_canonical_full_step_trajectories_index.csv")
    assert index.duplicate_extension_row_count.sum() > 0
    assert trajectories.duplicated(["case_id", "projected_crack_extension_m"], keep=False).any()


def test_no_kdot_time_applied_k_or_r_curve_plot_labels():
    atlas_source = (ROOT / "scripts/plot_pf_canonical_full_trajectory_atlas.py").read_text()
    mechanism_source = (ROOT / "scripts/plot_pf_canonical_mechanism_figures.py").read_text()
    assert "Kdot" not in atlas_source + mechanism_source
    atlas = json.loads((OUT / "pf_canonical_full_KJ_atlas_manifest.json").read_text())
    assert atlas["quantity"] == "PF_MODEL_NATIVE_KJ_MPa_sqrt_m"
    assert atlas["quantity_is_applied_K"] is False
    assert atlas["quantity_is_conventional_R_curve"] is False
    assert all("R_CURVE" not in row["stem"] for row in atlas["figure_records"])


def test_onset_markers_exactly_match_published_onset_table(trajectories):
    onset = pd.read_csv(PUB / "pf_canonical_onset_candidates_v2.csv")
    marked = trajectories.loc[
        trajectories.is_initial_onset | trajectories.is_reload_separated_reinitiation_onset
    ]
    assert len(marked) == len(onset) == 450
    joined = onset.merge(
        marked,
        left_on=["case_id", "event_transaction_index", "pre_event_step"],
        right_on=["case_id", "crack_event_transaction_index", "raw_step"],
        validate="one_to_one",
    )
    assert np.allclose(joined.pre_event_native_KJ_MPa_sqrt_m,
                       joined.native_KJ_MPa_sqrt_m, rtol=2e-12)


def test_physical_avalanche_membership_matches_transaction_ranges(trajectories):
    avalanches = pd.read_csv(PUB / "pf_canonical_physical_avalanches_v2.csv")
    expected_events = int(avalanches.event_transaction_count.sum())
    event_rows = trajectories.loc[trajectories.is_crack_event_row]
    assert len(event_rows) == expected_events
    assert event_rows.physical_avalanche_index.notna().all()


def test_supplemental_and_branching_cases_are_excluded(trajectories):
    supplemental = pd.read_parquet(
        OUT / "pf_theta45_rate0p01x_supplemental_full_trajectories.parquet"
    )
    assert supplemental.case_id.nunique() == 42
    assert set(supplemental.case_id).isdisjoint(set(trajectories.case_id))
    assert supplemental.theta_deg.eq(45.0).all()
    assert supplemental.rate_tag.eq("rate0p01x").all()
    assert not trajectories.case_id.str.contains("branch", case=False).any()


def test_conditional_reinitiation_incidence_and_magnitude_are_both_reported():
    stats = pd.read_csv(OUT / "pf_orientation_conditional_reinitiation_statistics.csv")
    assert len(stats) == 16
    assert {"finite_reinitiation_fraction",
            "conditional_mean_delta_K_reinit_MPa_sqrt_m",
            "conditional_median_delta_K_reinit_MPa_sqrt_m"}.issubset(stats.columns)


def test_rate_comparisons_use_common_random_numbers():
    manifest = pd.read_csv(PUB / "pf_canonical_fracture_run_manifest.csv")
    rate = manifest.loc[manifest.is_rate_matrix_case.astype(bool)]
    assert rate.groupby(["material_class", "temperature_K"]).seed.nunique().eq(1).all()
    assert rate.groupby(["material_class", "temperature_K"]).rate_tag.nunique().eq(3).all()


def test_frozen_swaps_are_diagnostic_and_do_not_mutate_production():
    swap = pd.read_csv(OUT / "pf_orientation_frozen_swap_matrix.csv")
    assert not swap.stochastic_clock_advanced.astype(bool).any()
    assert swap.state_history.eq("ZERO_HISTORY").all()
    assert swap.crack_path_rotation.astype(bool).eq(False).all()
    summary = json.loads((OUT / "pf_rate_orientation_mechanism_summary.json").read_text())
    assert summary["new_stochastic_trajectories"] == summary["fem_czm_runs"] == 0


def test_unavailable_state_fails_closed_and_no_production_runner_is_imported():
    counter = pd.read_csv(OUT / "pf_peak_theta0_rate_counterfactuals.csv")
    unavailable = counter.loc[counter.diagnostic.ne("ACTUAL_STATE")]
    assert unavailable.evaluation_status.eq(
        "UNAVAILABLE_MISSING_EXACT_STATE_INJECTION_CONTRACT"
    ).all()
    assert unavailable.evaluated_KJ_MPa_sqrt_m.isna().all()
    builder = (ROOT / "scripts/build_pf_canonical_full_trajectory_atlas.py").read_text()
    analyzer = (ROOT / "scripts/analyze_pf_canonical_rate_orientation_mechanisms.py").read_text()
    assert "run_tip_only" not in builder + analyzer
    assert "stochastic_avalanche_runner" not in builder + analyzer


def test_every_figure_has_pdf_svg_png_and_source_data_with_recorded_hashes():
    manifests = [
        json.loads((OUT / "pf_canonical_full_KJ_atlas_manifest.json").read_text()),
        json.loads((OUT / "pf_canonical_mechanism_figure_manifest.json").read_text()),
    ]
    records = manifests[0]["figure_records"] + manifests[1]["records"]
    assert len(records) == 91
    for row in records:
        source = Path(row["source_data"])
        assert source.is_file() and digest(source) == row["source_data_sha256"]
        assert set(row["outputs"]) == {"pdf", "svg", "png"}
        for artifact in row["outputs"].values():
            path = Path(artifact["path"])
            assert path.is_file() and digest(path) == artifact["sha256"]


def test_product_decomposition_identities_close():
    rate = pd.read_csv(OUT / "pf_peak_theta0_rate_onset_decomposition.csv")
    initial = pd.read_csv(OUT / "pf_orientation_initial_onset_decomposition.csv")
    reinit = pd.read_csv(OUT / "pf_orientation_reinitiation_decomposition.csv")
    assert rate.identity_residual_MPa_sqrt_m.abs().max() < 1e-10
    assert initial.identity_residual_MPa_sqrt_m.abs().max() < 1e-10
    assert reinit.identity_residual_MPa_sqrt_m.abs().max() < 1e-10


def test_peak_initial_structural_transfer_is_rate_invariant():
    transfer = pd.read_csv(OUT / "pf_peak_rate_initial_structural_transfer.csv")
    assert len(transfer) == 36
    assert transfer.matched_structural_transfer_identical_to_numerical_precision.astype(bool).all()
    assert transfer.matched_rate_relative_spread.max() < 1e-12


def test_provenance_declares_analysis_only_and_preserves_canonical_sources():
    provenance = json.loads(
        (OUT / "pf_canonical_full_trajectory_and_mechanism_provenance.json").read_text()
    )
    assert provenance["new_stochastic_pf_trajectories"] == 0
    assert provenance["fem_czm_runs"] == 0
    assert provenance["material_rows_changed"] == 0
    assert provenance["physical_equations_changed"] == 0
    assert provenance["canonical_raw_artifacts_modified"] is False
