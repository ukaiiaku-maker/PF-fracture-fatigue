from scripts.run_current_cavity_interaction_continuation import (
    HISTORICAL_OPENING_RATE_M_PER_S,
    options,
)
from arrhenius_fracture.shared_v5_cavity_adapter import _finite
from arrhenius_fracture.material_manifest import ExpFloorBarrier


def test_interaction_continuation_uses_historical_physical_ramp():
    assert HISTORICAL_OPENING_RATE_M_PER_S == 1.0e-6
    configured = options()
    assert configured.mechanics.value == "spatial_pf"
    assert configured.loading.value == "monotonic"
    assert configured.random_seed == 3621
    assert configured.cavity.value == "full_order_v5"


def test_event_rich_audit_serializes_native_barrier_dataclass():
    barrier = ExpFloorBarrier(
        G00_eV=2.0, gT_eV_per_K=0.0, sigc0_Pa=1.0,
        sT_Pa_per_K=0.0, alpha=1.0, exponent=2.0,
        floor_fraction=0.25,
    )
    converted = _finite({"barrier": barrier})
    assert converted["barrier"]["G00_eV"] == 2.0
    assert converted["barrier"]["floor_fraction"] == 0.25
