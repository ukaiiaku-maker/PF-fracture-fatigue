from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fatigue_v1 import (
    ExpFloorBarrierParams, ScaledExpFloorBarrier, FatigueControllerConfig,
    FatigueCycleHazardController, FatigueWaveform,
)
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front import FrontConfig, default_cleavage_barrier, default_emission_barrier
from arrhenius_fracture.unified_front import UnifiedMPZFrontEngine
from arrhenius_fracture.unified_mpz import MPZConfig


def test_fatigue_controller_delegates_to_unified_state():
    mat = ElasticProperties(); f = FrontConfig(); f.da = 5e-6; f.sigma_cap = 0; f.L_pz = 100e-6
    front = UnifiedMPZFrontEngine(
        f, default_cleavage_barrier(), default_emission_barrier(mat.b), mat.G, mat.nu, mat.b,
        MaterialManifest.from_csv(default_manifest_path("weakT")),
        MPZConfig(length_m=100e-6, n_bins=100),
    )
    dummy = ScaledExpFloorBarrier(ExpFloorBarrierParams(), 1.0, 1.0)
    cfg = FatigueControllerConfig(block_cycles=1.0, max_block_cycles=1.0, min_block_cycles=1e-6)
    controller = FatigueCycleHazardController(cfg, dummy, dummy, dummy)
    out = controller.cycle_step_front(front, FatigueWaveform(Kmax=17e6, R=0.1, frequency_Hz=1000), 700.0, requested_cycles=1.0)
    assert out["cycle_limiter"] == "unified_hazard_state"
    assert out["mpz_state_model"] == front.mpz.state_model
    assert front.mpz.time_s > 0.0
