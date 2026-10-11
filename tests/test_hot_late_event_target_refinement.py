from scripts.run_hot_late_event_target_refinement import (
    TARGETS, native_args_for_target,
)


def test_event_target_matrix_is_two_successive_refinements():
    assert TARGETS == (0.15, 0.075, 0.0375)


def test_target_refinement_fixes_floor_and_physical_endpoint():
    original = [
        "--steps", "2000000", "--target-crack-extension-um", "1000",
        "--adaptive-event-target", "0.15", "--adaptive-events",
    ]
    refined = native_args_for_target(original, 0.0375)
    assert refined[refined.index("--steps") + 1] == "1710"
    assert refined[refined.index("--target-crack-extension-um") + 1] == "1000"
    assert refined[refined.index("--adaptive-min-frac") + 1] == "1e-10"
    assert refined[refined.index("--adaptive-event-target") + 1] == "0.037499999999999999"
