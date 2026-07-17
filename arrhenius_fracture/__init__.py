"""Arrhenius sharp-front fracture and fatigue with unified MPZ state."""

__version__ = "10.1.9.1"

from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState
from .unified_front import UnifiedMPZFrontEngine
from .kinetic_tip_cell import KineticMovingTipFrontEngine, KineticTipConfig
from . import continuum_source_tip as _continuum_source_tip
from .separated_source_tip import SeparatedSourceKineticTipEngine
from .campaign_calibrated_tip import CampaignCalibratedTipEngine
from .developed_state_diagnostic_tip import DevelopedStateDiagnosticTipEngine
from .geometry_source_feedback_tip import GeometrySourceFeedbackTipEngine

# Preserve the separated public continuum class by default. Protected versioned
# entry points switch only their own process to campaign or diagnostic engines.
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
    "DevelopedStateDiagnosticTipEngine",
    "GeometrySourceFeedbackTipEngine",
]
