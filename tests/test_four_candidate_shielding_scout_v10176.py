from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_v10_1_7_6_shielding_scout import (
    _candidate_summary,
    _max_abs_array,
    _positive_emission_drive_fraction,
)
from scripts.prepare_v10_1_7_5_candidate_transfer import REQUIRED_SOURCE_FIELDS
from scripts.prepare_v10_1_7_6_shielding_scout import (
    DEFAULT_CANDIDATES,
    SCOUT_MODES,
    prepare_shielding_scout,
)


def _candidate_row(candidate: str, low: float, high: float) -> dict[str, object]:
    row: dict[str, object] = {name: 1.0 for name in REQUIRED_SOURCE_FIELDS}
    row.update(
        {
            "candidate_id": candidate,
            "transition_bracket": f"T{low:04.0f}_{high:04.0f}K",
            "refinement_transition_temperatures_K": json.dumps(
                [low, low + (high - low) / 3.0, low + 2.0 * (high - low) / 3.0, high]
            ),
            "c_blunt": 2.0,
        }
    )
    return row


def test_prepares_four_candidates_two_endpoints_four_modes(tmp_path: Path):
    brackets = [(400.0, 500.0), (600.0, 700.0), (800.0, 900.0), (900.0, 1000.0)]
    source = pd.DataFrame(
        [_candidate_row(candidate, *bracket) for candidate, bracket in zip(DEFAULT_CANDIDATES, brackets)]
    )
    cases = prepare_shielding_scout(source, list(DEFAULT_CANDIDATES), tmp_path)
    assert len(cases) == 32
    assert set(cases["mode"]) == set(SCOUT_MODES)
    assert cases.groupby("candidate_id")["T_K"].nunique().eq(2).all()
    assert not cases["mode"].isin(["blunting_off", "background_field_off"]).any()


def test_history_helpers_use_entire_trajectory_not_only_fired_state():
    records = [
        {
            "anisotropic_sigma_back_by_system_Pa": [1.7e9, -1.2e9],
            "anisotropic_sigma_emit_by_system_Pa": [2.0e8, 0.0],
            "fired": False,
        },
        {
            "anisotropic_sigma_back_by_system_Pa": [2.0e7, -1.0e7],
            "anisotropic_sigma_emit_by_system_Pa": [0.0, 0.0],
            "fired": True,
        },
    ]
    assert _max_abs_array(records, ("anisotropic_sigma_back_by_system_Pa",)) == 1.7e9
    assert _positive_emission_drive_fraction(records) == 0.25


def test_shielding_history_priority_identifies_desired_response():
    rows = []
    values = {
        "full": (10.0, 18.0, 8.0),
        "plasticity_off": (10.0, 10.5, 0.5),
        "shielding_off": (10.0, 12.0, 2.0),
        "backstress_off": (10.0, 19.0, 9.0),
    }
    for mode, (low, high, rise) in values.items():
        rows.append(
            {
                "candidate_id": "candidate",
                "transition_bracket": "T0800_0900K",
                "mode": mode,
                "low_K_init_MPa_sqrt_m": low,
                "high_K_init_MPa_sqrt_m": high,
                "rise_MPa_sqrt_m": rise,
                "endpoint_ratio": high / low,
                "maximum_abs_K_shield_history_MPa_sqrt_m": 1.0 if mode == "full" else 0.0,
                "maximum_abs_sigma_back_history_Pa": 1.5e9,
                "low_positive_emission_drive_fraction": 0.05,
                "high_positive_emission_drive_fraction": 0.20,
                "low_max_cumulative_emitted": 100.0,
                "high_max_cumulative_emitted": 500.0,
                "cumulative_emission_growth": 400.0,
                "low_max_mobile": 10.0,
                "high_max_mobile": 30.0,
                "low_max_retained": 5.0,
                "high_max_retained": 20.0,
                "low_max_blunted_radius_um": 1.0,
                "high_max_blunted_radius_um": 1.2,
                "minimum_tensor_reliable_fraction": 1.0,
                "post_hazard_weighting_count": 0,
            }
        )
    result = _candidate_summary("candidate", pd.DataFrame(rows))
    assert result["shielding_history_scout_priority"] is True
    assert result["shielding_history_fraction_of_full_rise"] >= 0.5
    assert result["plasticity_off_endpoint_ratio"] <= 1.25
