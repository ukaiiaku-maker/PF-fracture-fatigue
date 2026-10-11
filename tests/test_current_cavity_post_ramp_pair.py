from scripts.run_current_cavity_post_ramp_pair import NATIVE_ARGS, matrix, options


def test_cavity_suffix_is_explicit_post_ramp_dwell():
    plan = matrix()
    assert plan["protocol"] == "explicit_post_ramp_fixed_opening_dwell"
    assert plan["additional_opening_m"] == 0.0
    assert plan["additional_time_s"] == 1e-5
    assert plan["hazard_seed"] == 3621


def test_cavity_suffix_preserves_pair_physics_and_seed():
    configured = options()
    assert configured.random_seed == 3621
    assert configured.cavity.value == "full_order_v5"
    assert configured.plasticity.value == "tip_plus_full_field"
    assert configured.mechanics.value == "spatial_pf"
    assert NATIVE_ARGS[NATIVE_ARGS.index("--duration-s") + 1] == "1e-5"
