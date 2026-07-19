from pathlib import Path

from arrhenius_fracture import sharp_front


def test_active_solver_retains_anisotropy_branching_and_fatigue_options():
    parser = sharp_front._build_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    required = {
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg",
        "--max-fronts", "--branch-ratio", "--fatigue-cycles",
        "--j-decomposition", "--adaptive-events", "--tip-h-fine",
        "--material-class", "--wake-shielding",
    }
    assert not (required - option_strings)
