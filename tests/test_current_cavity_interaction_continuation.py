from scripts.run_current_cavity_interaction_continuation import (
    HISTORICAL_OPENING_RATE_M_PER_S,
    options,
)


def test_interaction_continuation_uses_historical_physical_ramp():
    assert HISTORICAL_OPENING_RATE_M_PER_S == 1.0e-6
    configured = options()
    assert configured.mechanics.value == "spatial_pf"
    assert configured.loading.value == "monotonic"
    assert configured.random_seed == 3621
    assert configured.cavity.value == "full_order_v5"
