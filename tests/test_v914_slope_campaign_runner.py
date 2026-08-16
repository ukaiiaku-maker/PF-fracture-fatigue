from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_v914_slope_fatigue_campaign.py")
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def args():
    return SimpleNamespace(
        python=Path("/python"), v914_root=Path("/v914"), registry=Path("r.csv"),
        physics=Path("p.json"), expected_head="abc",
    )


def test_accelerated_command_pins_numerical_contract():
    command = runner.accelerated_command(args(), "candidate", [1.0, 1.1], Path("out"))
    assert command[command.index("--maximum-explicit-cycles") + 1] == "4096"
    assert command[command.index("--phase-steps") + 1] == "32"
    assert "--force" not in command and "--restart" not in command


def test_explicit_command_pins_head_seed_and_state_history():
    row = SimpleNamespace(candidate_id="candidate", deltaK_MPa_sqrt_m=24.0, normalized_f=1.1)
    command = runner.explicit_command(args(), row, Path("out"))
    assert command[command.index("--expected-head") + 1] == "abc"
    assert command[command.index("--seed") + 1] == "1720"
    assert command[command.index("--state-history-cycle-interval") + 1] == "10"
    assert runner.fraction_label(1.125) == "1p125"
