from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_2_26_paper_top2_500um_temperature_sweep.sh"
PLOTTER = ROOT / "scripts" / "plot_v10_2_26_paper_top2_rcurves.py"


def test_runner_contract() -> None:
    text = RUNNER.read_text()
    assert "arrhenius_fracture.sharp_front_v10_2_25_audited" in text
    assert (
        'OPTIONS=${OPTIONS:-"v913_paper_peak01_0242980_persistent_sites '
        'v913_paper_dbtt01_0202500_persistent_sites"}'
    ) in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-500}' in text
    assert 'STEPS=${STEPS:-1000000}' in text
    assert 'SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-12}' in text
    assert 'SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}' in text
    assert 'SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}' in text
    assert '"stochastic_cleavage_hazard": True' in text
    assert '"common_random_numbers": False' in text
    assert 'CLEAVAGE_HAZARD_SEED="$case_seed"' in text
    assert 'v10_2_26_case_seed_map.csv' in text
    assert '--save-snapshots "$SAVE_SNAPSHOTS"' in text
    assert '--snapshot-cols "$SNAPSHOT_COLS"' in text
    assert '--target-extension-um "$TARGET_EXT_UM"' in text
    assert "--no-plots" not in text
    for temperature in (
        "300",
        "600",
        "800",
        "900",
        "950",
        "1000",
        "1050",
        "1100",
        "1150",
        "1200",
        "1250",
        "1300",
    ):
        assert temperature in text
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def _write_case(
    outroot: Path,
    option: str,
    candidate: str,
    response_class: str,
    temperature: int,
    seed: int,
) -> None:
    case = outroot / option / f"T{temperature}K_th45_seed{seed}"
    case.mkdir(parents=True)
    (case / "COMPLETE").write_text("\n")
    fields = [
        "KJ_Pa_sqrtm",
        "crack_extension_m",
        "da_block_m",
        "n_fire",
        "sigma_back_Pa",
        "mpz_available_site_fraction",
        "mpz_K_shield_Pa_sqrt_m",
    ]
    rows = [
        [20.0e6, 20.0e-6, 20.0e-6, 1.0, 0.5e9, 1.0, 0.01e6],
        [30.0e6, 100.0e-6, 80.0e-6, 1.0, 1.0e9, 1.0, -0.02e6],
        [35.0e6, 250.0e-6, 150.0e-6, 1.0, 1.5e9, 1.0, 0.02e6],
        [40.0e6, 400.0e-6, 150.0e-6, 1.0, 1.8e9, 1.0, -0.03e6],
        [45.0e6, 505.0e-6, 105.0e-6, 1.0, 2.0e9, 1.0, 0.03e6],
    ]
    with (case / f"steps_{temperature}K.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)

    (case / "v10_2_25_v913_paper_campaign_parameter_transfer.json").write_text(
        json.dumps(
            {
                "selected_candidate": candidate,
                "paper_campaign_selection": {
                    "candidate_id": candidate,
                    "response_class": response_class,
                    "interpretation": f"synthetic {response_class}",
                },
            }
        )
    )
    (case / "anisotropic_emission_audit_v10174.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "persistent_site_front_width_m": 2.0e-6,
                        "persistent_tip_radius_m": 1.5e-6,
                        "persistent_aggregate_emission_hazard_s": 3.0,
                    },
                    {
                        "persistent_site_front_width_m": 1.0e-6,
                        "persistent_tip_radius_m": 2.5e-6,
                        "persistent_aggregate_emission_hazard_s": 5.0,
                    },
                ]
            }
        )
    )


def test_plotter_generates_event_resolved_500um_outputs(tmp_path: Path) -> None:
    outroot = tmp_path / "campaign"
    outroot.mkdir()
    (outroot / "v10_2_26_campaign_manifest.json").write_text(
        json.dumps(
            {
                "target_crack_extension_um": 500.0,
                "stochastic_cleavage_hazard": True,
                "common_random_numbers": False,
            }
        )
    )
    cases = (
        (
            "v913_paper_peak01_0242980_persistent_sites",
            "v913_zeroD_sobol_0242980",
            "peak_like",
        ),
        (
            "v913_paper_dbtt01_0202500_persistent_sites",
            "v913_zeroD_sobol_0202500",
            "classic_dbtt_upper_shelf",
        ),
    )
    expected_seeds: set[int] = set()
    for option_index, (option, candidate, response_class) in enumerate(cases):
        for temperature_index, temperature in enumerate((900, 1100)):
            seed = 3621 + option_index * 1_000_000 + temperature_index * 1009
            expected_seeds.add(seed)
            _write_case(
                outroot,
                option,
                candidate,
                response_class,
                temperature,
                seed,
            )

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(PLOTTER), "--outroot", str(outroot)],
        check=True,
        env=env,
    )

    summary = list(
        csv.DictReader(
            (outroot / "v10_2_26_paper_top2_500um_summary.csv").open(newline="")
        )
    )
    assert len(summary) == 4
    assert {int(row["seed"]) for row in summary} == expected_seeds
    assert {row["candidate_id"] for row in summary} == {
        "v913_zeroD_sobol_0242980",
        "v913_zeroD_sobol_0202500",
    }
    for row in summary:
        assert np.isclose(float(row["campaign_target_extension_um"]), 500.0)
        assert np.isclose(float(row["achieved_extension_um"]), 505.0)
        assert np.isclose(float(row["K_100um_MPa_sqrt_m"]), 30.0)
        assert np.isclose(float(row["K_300um_MPa_sqrt_m"]), 35.0 + 5.0 / 3.0)
        assert np.isclose(float(row["K_500um_MPa_sqrt_m"]), 45.0)
        assert np.isclose(
            float(row["deltaK_500um_from_first_MPa_sqrt_m"]), 25.0
        )
        assert np.isclose(float(row["min_available_site_fraction"]), 1.0)
        assert np.isclose(float(row["min_front_width_um"]), 1.0)
        assert np.isclose(float(row["max_tip_radius_um"]), 2.5)

    pngs = list((outroot / "plots").rglob("*.png"))
    pdfs = list((outroot / "plots").rglob("*.pdf"))
    # Four individual plots, two candidate overlays, and two temperature overlays.
    assert len(pngs) == 8
    assert len(pdfs) == 8
