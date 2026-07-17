"""Arrhenius sharp-front fracture and fatigue with unified MPZ state."""

__version__ = "10.1.5"

from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState
from .unified_front import UnifiedMPZFrontEngine
from .kinetic_tip_cell import KineticMovingTipFrontEngine, KineticTipConfig
from . import continuum_source_tip as _continuum_source_tip
from .separated_source_tip import SeparatedSourceKineticTipEngine
from .campaign_calibrated_tip import CampaignCalibratedTipEngine

# Preserve the v10.1.4 public continuum class by default. The protected
# sharp_front_v10_1_5 entry point switches only its own process to the
# campaign-calibrated implementation.
_continuum_source_tip.ContinuumSourceKineticTipEngine = (
    SeparatedSourceKineticTipEngine
)
ContinuumSourceKineticTipEngine = SeparatedSourceKineticTipEngine

__all__ = [
    "MaterialManifest",
    "MPZConfig",
    "UnifiedMPZState",
    "UnifiedMPZFrontEngine",
    "KineticMovingTipFrontEngine",
    "KineticTipConfig",
    "ContinuumSourceKineticTipEngine",
    "SeparatedSourceKineticTipEngine",
    "CampaignCalibratedTipEngine",
]
