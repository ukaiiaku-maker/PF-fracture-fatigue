from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_spatial_pf_matrix_is_two_worker_dbtt_only_and_reuses_control() -> None:
    path = ROOT / "scripts" / "run_oneD_v2_taylor_peierls_spatial_pf_matrix.py"
    ast.parse(path.read_text())
    text = path.read_text()
    assert 'default=2' in text
    assert 'args.maximum_workers != 2' in text
    assert '"TEMPERATURE_K": "1100"' in text
    assert '"HAZARD_SEED": "1008666"' in text
    assert '"TARGET_EXTENSION_UM"' in text
    assert '"v913_zeroD_sobol_0202500"' in text
    assert 'new_FEMCZM_runs' in text


def test_transfer_entry_accepts_a_unique_bounded_option_set() -> None:
    text = (ROOT / "scripts" / "run_oneD_v2_terminal_pf_transfer.py").read_text()
    assert "len(options) != 4" not in text
    assert "len(options) != len(rows)" in text
