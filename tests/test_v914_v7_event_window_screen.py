from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from screen_v914_v7_event_windows import _safe_screen_evaluation


def test_high_load_exact_refinement_failure_is_recorded_not_reinterpreted():
    def rejected():
        raise RuntimeError("adaptive cycle refinement failed closed")

    result = _safe_screen_evaluation(rejected, Kmax=80.0, R=0.1)
    assert result["status"] == "numerical_rejection"
    assert result["deltaK_MPa_sqrt_m"] == 72.0
    assert math.isnan(result["expected_waiting_cycles"])
    assert "failed closed" in result["failure"]
