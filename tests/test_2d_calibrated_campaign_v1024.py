from __future__ import annotations

import csv
from pathlib import Path

from arrhenius_fracture.mechanical_closure_v1024 import (
    AtlasSample,
    TensorDriveAtlas,
    build_atlas_from_trace_roots,
)
from arrhenius_fracture.reduced_campaign_v1024 import (
    DEFAULT_TEMPERATURES_K,
    generate_candidate_rows,
    score_candidate,
)


def test_tensor_factor_closure_is_load_scale_invariant():
    atlas = TensorDriveAtlas(
        [
            AtlasSample(5.0, 0.0, 0.10, 0.01),
            AtlasSample(8.0, 0.5, 0.20, 0.02),
            AtlasSample(12.0, 1.0, 0.30, 0.03),
        ],
        neighbors=2,
    )
    low = atlas.evaluate(2.0, 0.5)
    high = atlas.evaluate(80.0, 0.5)
    assert low.factors == high.factors
    assert low.outside_support is False
    assert high.outside_support is False
    assert atlas.audit()["K_used_as_interpolation_coordinate"] is False


def test_atlas_builder_reads_v1023_schedule(tmp_path: Path):
    root = tmp_path / "trace" / "two_d"
    root.mkdir(parents=True)
    schedule = root / "v10_2_3_2d_replay_schedule.csv"
    with schedule.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "K_Pa_sqrt_m",
                "expected_micro_advance_total_m",
                "drive_factor_0",
                "drive_factor_1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "K_Pa_sqrt_m": 5.0e6,
                "expected_micro_advance_total_m": 0.0,
                "drive_factor_0": 0.12,
                "drive_factor_1": 0.01,
            }
        )
        writer.writerow(
            {
                "K_Pa_sqrt_m": 7.0e6,
                "expected_micro_advance_total_m": 5.0e-6,
                "drive_factor_0": 0.15,
                "drive_factor_1": 0.02,
            }
        )
    output = tmp_path / "atlas.csv"
    payload = build_atlas_from_trace_roots([tmp_path / "trace"], output)
    assert payload["sample_count"] == 2
    assert output.is_file()
    assert Path(str(output) + ".json").is_file()


def test_generated_candidates_preserve_fallbacks_and_disable_cap():
    rows = generate_candidate_rows(8, seed=7)
    assert len(rows) == 8
    assert len({row["candidate_id"] for row in rows}) == 8
    for row in rows:
        assert row["max_K_shield_MPa_sqrt_m"] == 0.0
        total = (
            row["anchor_weight_A0002333"]
            + row["anchor_weight_A0003837"]
            + row["anchor_weight_A0002277"]
        )
        assert abs(total - 1.0) < 1.0e-12
        assert row["source_sites_per_system"] > 0.0
        assert row["c_blunt"] > 0.0


def _result(K: float, outside: float = 0.0):
    return {
        "K_first_MPa_sqrt_m": K,
        "closure_outside_support_fraction": outside,
    }


def test_scoring_selects_shielding_history_coupled_dbtt():
    results = {}
    for T, K in zip(DEFAULT_TEMPERATURES_K, [10.0, 13.0, 17.0, 20.0]):
        results[("full", T)] = _result(K)
    results[("plasticity_off", 300.0)] = _result(10.0)
    results[("plasticity_off", 1200.0)] = _result(11.0)
    results[("shielding_off", 300.0)] = _result(10.0)
    results[("shielding_off", 1200.0)] = _result(13.0)
    results[("backstress_off", 300.0)] = _result(10.0)
    results[("backstress_off", 1200.0)] = _result(18.0)
    score = score_candidate(results)
    assert score["strict_reduced_pass"] is True
    assert score["full_endpoint_ratio"] == 2.0
    assert score["plasticity_off_endpoint_ratio"] == 1.1
    assert score["shielding_fraction_of_full_rise"] >= 0.5
