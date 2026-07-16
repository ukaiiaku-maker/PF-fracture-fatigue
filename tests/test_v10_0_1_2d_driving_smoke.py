from __future__ import annotations

import re

from arrhenius_fracture import sharp_front_v10_1


def test_tip_only_abs_forward_keeps_2d_drive_nonzero(tmp_path, capsys):
    out = tmp_path / "drive"
    sharp_front_v10_1.main([
        "--mode", "2d",
        "--material-class", "ceramic",
        "--temperatures", "700",
        "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "abs_forward",
        "--steps", "12",
        "--nx", "16",
        "--ny", "32",
        "--dU", "2e-6",
        "--dt", "0.01",
        "--n-stagger", "1",
        "--tip-h-fine", "2e-6",
        "--tip-ratio", "1.2",
        "--da-phys", "5e-6",
        "--target-crack-extension-um", "10",
        "--mpz-length-um", "100",
        "--mpz-n-bins", "200",
        "--wake-length-um", "100",
        "--wake-n-bins", "0",
        "--wake-shielding",
        "--crystal-aniso",
        "--crystal-compete",
        "--crystal-theta-deg", "45",
        "--crystal-material", "w",
        "--j-decomposition", "cluster",
        "--max-fronts", "1",
        "--print-every", "1",
        "--save-snapshots", "0",
        "--no-plots",
        "--out", str(out),
    ])
    text = capsys.readouterr().out
    values = [float(v) for v in re.findall(r"KJ=\s*([0-9.eE+-]+)", text)]
    assert len(values) >= 6, text
    positive = [v for v in values if v > 0.0]
    assert len(positive) >= 4, text
    first_positive = next(i for i, v in enumerate(values) if v > 0.0)
    assert all(v > 0.0 for v in values[first_positive:]), text
    assert values[-1] > values[first_positive], text
