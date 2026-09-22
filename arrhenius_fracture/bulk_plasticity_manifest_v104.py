"""Compatibility surface for the immutable production v10.4 manifest."""
from .bulk_plasticity_v104 import (
    BulkPlasticityManifestV104, PlasticityScope, production_bulk_configuration,
)

__all__ = ["BulkPlasticityManifestV104", "PlasticityScope",
           "production_bulk_configuration"]
