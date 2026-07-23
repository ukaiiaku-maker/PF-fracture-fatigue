"""Mesh-independent along-front source-width closure for v10.2.22."""
from __future__ import annotations

import numpy as np

from . import persistent_site_source_v10221 as _source


MODEL_ID = "v10.2.22_physical_along_front_width"


def physical_source_geometry(state) -> dict[str, float]:
    """Return source geometry without tying along-front width to MPZ ``dx``.

    ``state.dx`` resolves distance ahead of the crack tip. The persistent-site
    front width is an orthogonal along-front correlation length and therefore
    must not be bounded by that discretization.
    """
    cfg = state._persistent_site_cfg
    rho_by_system = _source._campaign_local_density_m2(state)
    rho_width = max(
        float(state.cfg.forest_density_floor_m2) + float(np.sum(rho_by_system)),
        cfg.reference_density_m2,
    )
    minimum = max(
        float(cfg.minimum_front_width_m),
        abs(float(state._persistent_b)),
        1.0e-30,
    )
    maximum = (
        float(cfg.maximum_front_width_m)
        if cfg.maximum_front_width_m > 0.0
        else float(state.length_m)
    )
    width = _source.effective_front_width_m(
        rho_width,
        reference_width_m=cfg.reference_front_width_m,
        reference_density_m2=cfg.reference_density_m2,
        minimum_width_m=minimum,
        maximum_width_m=maximum,
    )
    radius = float(state.blunted_radius(state._persistent_r0_m, state._persistent_b))
    multiplicity = _source.persistent_site_multiplicity(
        cfg.rho_site0_m2,
        radius,
        width,
        state._persistent_active_arc_factor,
    )
    return {
        "tip_radius_m": radius,
        "front_width_m": width,
        "rho_width_m2": rho_width,
        "source_area_m2": state._persistent_active_arc_factor * radius * width,
        "multiplicity_per_system": multiplicity,
        "minimum_front_width_m": minimum,
        "front_width_grid_independent": True,
    }


def install_physical_front_width() -> None:
    """Install the mesh-independent geometry in the v10.2.21 source module."""
    _source._source_geometry = physical_source_geometry


__all__ = ["MODEL_ID", "physical_source_geometry", "install_physical_front_width"]
