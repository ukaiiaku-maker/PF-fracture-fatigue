from pathlib import Path

from scripts.qualify_native_source_geography import GEOGRAPHIES, native_args


def test_geography_qualification_matrix_is_exactly_the_two_v104_paths():
    assert GEOGRAPHIES == {
        "bulk_uniform": "pf_monotonic_bulk",
        "tip_plus_bulk": "pf_monotonic_tip_plus_bulk",
    }


def test_native_arguments_keep_geography_owned_by_shared_worker():
    args = native_args(Path("family.json"), 2)
    assert args[args.index("--steps") + 1] == "2"
    assert "--tip-plasticity" not in args
    assert "--no-tip-plasticity" not in args
    assert args[args.index("--bulk-plasticity-mode") + 1] == "full_field"
    assert args[args.index("--temperatures") + 1] == "900"
