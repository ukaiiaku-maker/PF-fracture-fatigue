"""Arrhenius sharp-front fracture and fatigue with unified MPZ state."""

__version__ = "10.1.3"

from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState
from .unified_front import UnifiedMPZFrontEngine
from .kinetic_tip_cell import KineticMovingTipFrontEngine, KineticTipConfig
from .continuum_source_tip import ContinuumSourceKineticTipEngine

__all__ = [
    "MaterialManifest",
    "MPZConfig",
    "UnifiedMPZState",
    "UnifiedMPZFrontEngine",
    "KineticMovingTipFrontEngine",
    "KineticTipConfig",
    "ContinuumSourceKineticTipEngine",
]
