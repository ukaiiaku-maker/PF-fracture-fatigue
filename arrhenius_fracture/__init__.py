"""Arrhenius sharp-front fracture and fatigue with unified MPZ state."""

__version__ = "10.0.2.1"

from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState
from .unified_front import UnifiedMPZFrontEngine

__all__ = ["MaterialManifest", "MPZConfig", "UnifiedMPZState", "UnifiedMPZFrontEngine"]
