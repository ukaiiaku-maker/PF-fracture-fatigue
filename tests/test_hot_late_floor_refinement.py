from scripts.run_hot_late_floor_refinement import FLOORS, native_args_for_floor


def test_floor_matrix_has_predeclared_resolution_levels():
    assert FLOORS == (1e-8, 1e-9, 1e-10)


def test_native_suffix_has_common_physical_endpoint_and_selected_floor():
    original = [
        "--steps", "2000000", "--target-crack-extension-um", "1000",
        "--print-every", "400", "--adaptive-events",
    ]
    refined = native_args_for_floor(original, 1e-10)
    assert refined[refined.index("--steps") + 1] == "1710"
    assert refined[refined.index("--target-crack-extension-um") + 1] == "1000"
    assert refined[refined.index("--print-every") + 1] == "1"
    assert refined[refined.index("--adaptive-min-frac") + 1] == "1e-10"
    assert original[-1] == "--adaptive-events"
