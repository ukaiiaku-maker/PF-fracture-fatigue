"""Resolver for one exact accepted v11 crack-network FEM provider state."""
from __future__ import annotations

from pathlib import Path
import pickle

from .live_topology_kernel_cache_v11 import ExactTopologyCache
from .live_topology_kernel_v11 import (
    LiveTopologyRequest, evaluate_exact_topology, topology_fingerprint,
)


def resolve_live_topology_request(
    request: LiveTopologyRequest, *, cache_root: str | Path, accepted: bool,
) -> tuple[dict, bool]:
    if len(request.crack_network.active_tip_ids) > 2:
        raise ValueError("v11 exact-topology provider supports at most two active fronts")
    kwargs = dict(
        network=request.crack_network, mesh=request.mesh, damage=request.damage,
        mechanical_configuration_fingerprint=request.mechanical_configuration_fingerprint,
        specimen_geometry=request.specimen_geometry,
        boundary_condition_identity=request.boundary_condition_identity,
        elastic_constants=request.elastic_constants, cluster_frame=request.cluster_frame,
        mpz_station_coordinates_m=request.mpz_station_coordinates_m,
        wake_station_coordinates_m=request.wake_station_coordinates_m,
        contour_definitions={"radius_m": request.contour_radius_m, "exclude_radius_m": request.exclude_radius_m},
    )
    fingerprint = topology_fingerprint(**kwargs)
    if not accepted:
        # Ephemeral trial states are deliberately never persisted.
        return evaluate_exact_topology(request), False
    cache = ExactTopologyCache(cache_root)
    return cache.get_or_evaluate_accepted(
        request.mechanical_configuration_fingerprint, fingerprint,
        lambda: evaluate_exact_topology(request),
    )


def resolve_pickled_request(
    request_path: str | Path, *, cache_root: str | Path, accepted: bool,
) -> tuple[dict, bool]:
    request = pickle.loads(Path(request_path).read_bytes())
    if not isinstance(request, LiveTopologyRequest):
        raise ValueError("v11 resolver input is not a LiveTopologyRequest")
    return resolve_live_topology_request(request, cache_root=cache_root, accepted=accepted)


__all__ = ["resolve_live_topology_request", "resolve_pickled_request"]
