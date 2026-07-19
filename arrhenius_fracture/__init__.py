"""Arrhenius sharp-front PF fracture and fatigue with unified MPZ state.

This repository currently shares the historical top-level package name
``arrhenius_fracture`` with the separate Arrhenius_FEM_CZM_MPZ distribution.
The explicit identity constants allow launchers to fail closed instead of
silently importing the other editable installation.
"""

__version__ = "10.2.14"
PROJECT_ID = "PF-fracture-fatigue"
PROJECT_REPOSITORY = "ukaiiaku-maker/PF-fracture-fatigue"
PACKAGE_NAMESPACE = "arrhenius_fracture"

from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState
from .unified_front import UnifiedMPZFrontEngine
from .kinetic_tip_cell import KineticMovingTipFrontEngine, KineticTipConfig
from . import continuum_source_tip as _continuum_source_tip
from .separated_source_tip import SeparatedSourceKineticTipEngine
from .campaign_calibrated_tip import CampaignCalibratedTipEngine
from .developed_state_diagnostic_tip import DevelopedStateDiagnosticTipEngine
from .stochastic_hazard_tip import (
    HazardThresholdConfig,
    StochasticHazardDiagnosticTipEngine,
)
from .stochastic_avalanche_tip import (
    AvalancheLengthConfig,
    StochasticAvalancheDiagnosticTipEngine,
)

# Preserve the separated public continuum class by default. Protected versioned
# entry points switch only their own process to campaign, diagnostic, or pilot
# engines.
_continuum_source_tip.ContinuumSourceKineticTipEngine = (
    SeparatedSourceKineticTipEngine
)
ContinuumSourceKineticTipEngine = SeparatedSourceKineticTipEngine

__all__ = [
    "__version__",
    "PROJECT_ID",
    "PROJECT_REPOSITORY",
    "PACKAGE_NAMESPACE",
    "MaterialManifest",
    "MPZConfig",
    "UnifiedMPZState",
    "UnifiedMPZFrontEngine",
    "KineticMovingTipFrontEngine",
    "KineticTipConfig",
    "ContinuumSourceKineticTipEngine",
    "SeparatedSourceKineticTipEngine",
    "CampaignCalibratedTipEngine",
    "DevelopedStateDiagnosticTipEngine",
    "HazardThresholdConfig",
    "StochasticHazardDiagnosticTipEngine",
    "AvalancheLengthConfig",
    "StochasticAvalancheDiagnosticTipEngine",
]
