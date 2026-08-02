import inspect
from pathlib import Path

from arrhenius_fracture import fatigue_controller_delegate_v10229 as delegate
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_v10230 import (
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
)


ROOT = Path(__file__).resolve().parents[1]


def test_continuum_comparison_cannot_suppress_first_passage_in_base_engine():
    assert (
        HazardEnergyGatedPersistentSiteCyclicTipEngine.
        hazard_energy_gate_continuum_affects_hazard
        is False
    )
    source = inspect.getsource(
        HazardEnergyGatedPersistentSiteCyclicTipEngine._integrate_coupled
    )
    assert 'effective_lambda = 0.0' not in source
    assert 'energy_gate_continuum_open' not in source.split(
        'effective_lambda = lambda_override', 1
    )[0].split('continuum =', 1)[-1]


def test_corrected_engine_does_not_monkey_patch_module_global_gate():
    assert (
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.
        hazard_energy_gate_continuum_affects_hazard
        is False
    )
    source = inspect.getsource(
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine._integrate_coupled
    )
    assert '_base.continuum_gate_diagnostics =' not in source
    assert 'finally:' not in source


def test_force_cycles_is_a_pure_engine_commit_path():
    source = inspect.getsource(delegate.install_engine_native_cycle_preview)
    assert 'force_cycles=cap' in source
    assert 'self.cfg.max_block_cycles =' not in source
    assert 'force_cycles=None' not in source.split(
        'and force_cycles is not None', 1
    )[-1].split('return original_step', 1)[0]


def test_feedback_state_production_launcher_is_hard_disabled():
    text = (
        ROOT / 'scripts' /
        'run_v10_2_30_300K_four_class_fatigue_feedback_state.sh'
    ).read_text()
    assert 'intentionally disabled' in text
    assert 'exit 2' in text
    assert 'run_v10_2_30_300K_four_class_fatigue.sh' not in text
