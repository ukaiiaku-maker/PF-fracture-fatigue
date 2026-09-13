"""Read-only, edge-resolved diagnosis of accepted CST cavity traction."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .explicit_cavity_v5 import cavity_edge_traction_geometry


SCHEMA = "unified-2d-one-void.arc-resolved-cavity-traction/1"


def _stress_rows(stress, element_count: int) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    if values.shape == (3, element_count):
        return np.column_stack((values[0], values[1], values[2]))
    if values.shape == (element_count, 3):
        return values
    if values.shape == (element_count, 2, 2):
        return np.column_stack((values[:, 0, 0], values[:, 1, 1], values[:, 0, 1]))
    raise ValueError("stress has an unsupported shape")


def _element_quality(points: np.ndarray) -> np.ndarray:
    sides2 = np.sum((points - points[:, [1, 2, 0]]) ** 2, axis=(1, 2))
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    twice_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return 2.0 * np.sqrt(3.0) * twice_area / np.maximum(sides2, 1.0e-300)


def _summary(rows: list[dict[str, Any]], remote: float, radius: float) -> dict[str, Any]:
    length = float(sum(row["edge_length_m"] for row in rows))
    t2 = float(sum(row["traction_l2_numerator_Pa2_m"] for row in rows))
    tn2 = float(sum(row["normal_l2_numerator_Pa2_m"] for row in rows))
    tt2 = float(sum(row["tangential_l2_numerator_Pa2_m"] for row in rows))
    resultant = np.sum([row["traction_resultant_contribution_N_per_m"] for row in rows], axis=0) if rows else np.zeros(2)
    moment = float(sum(row["traction_moment_contribution_N"] for row in rows))
    denominator = remote * math.sqrt(length) if length else math.nan
    return {
        "edge_count": len(rows), "boundary_length_m": length,
        "traction_l2_dimensional_Pa_sqrt_m": math.sqrt(t2),
        "normal_l2_dimensional_Pa_sqrt_m": math.sqrt(tn2),
        "tangential_l2_dimensional_Pa_sqrt_m": math.sqrt(tt2),
        "normalized_traction_l2": math.sqrt(t2) / denominator if length else math.nan,
        "normalized_normal_traction_l2": math.sqrt(tn2) / denominator if length else math.nan,
        "normalized_tangential_traction_l2": math.sqrt(tt2) / denominator if length else math.nan,
        "traction_resultant_N_per_m": np.asarray(resultant, dtype=float).tolist(),
        "traction_resultant_normalized": (
            np.asarray(resultant, dtype=float) / (remote * length)
        ).tolist() if length else [math.nan, math.nan],
        "traction_moment_N": moment,
        "traction_moment_normalized": moment / (remote * length * radius) if length else math.nan,
        "element_ids": sorted({row["adjacent_element_id"] for row in rows}),
    }


def diagnose_arc_resolved_cavity_traction_v1(
    *, nodes, elements, stress_Pa, owned_boundary_edges, cavity_center_m,
    cavity_radius_m: float, source_position_m, junction_position_m,
    process_length_m: float, remote_traction_scale_Pa: float,
    plastic_strain=None, density_m2=None,
) -> dict[str, Any]:
    """Export one deterministic row per owned edge without changing the state."""
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    center = np.asarray(cavity_center_m, dtype=float)
    source = np.asarray(source_position_m, dtype=float)
    junction = np.asarray(junction_position_m, dtype=float)
    radius = float(cavity_radius_m)
    remote = float(remote_traction_scale_Pa)
    half_width = 0.5 * min(radius, float(process_length_m))
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a triangular plane mesh")
    if center.shape != source.shape or center.shape != junction.shape or center.shape != (2,):
        raise ValueError("cavity positions must be two-dimensional")
    if not np.isfinite((radius, remote, half_width)).all() or min(radius, remote, half_width) <= 0:
        raise ValueError("radius, process length, and remote traction must be positive")
    supplied_edges = [tuple(sorted(map(int, edge))) for edge in owned_boundary_edges]
    edges = sorted(set(supplied_edges))
    if len(edges) != len(supplied_edges):
        raise ValueError("owned cavity boundary contains a duplicate edge")
    owners: dict[tuple[int, int], list[int]] = {}
    for eid, element in enumerate(tri):
        for edge in ((element[0], element[1]), (element[1], element[2]), (element[2], element[0])):
            owners.setdefault(tuple(sorted(map(int, edge))), []).append(eid)
    stresses = _stress_rows(stress_Pa, len(tri))
    quality = _element_quality(xy[tri])
    ep = np.zeros((len(tri), 3)) if plastic_strain is None else np.asarray(plastic_strain, dtype=float)
    if ep.shape == (3, len(tri)):
        ep = ep.T
    if ep.shape != (len(tri), 3):
        raise ValueError("plastic strain must have shape (3, ne) or (ne, 3)")
    density = np.zeros(len(tri)) if density_m2 is None else np.asarray(density_m2, dtype=float)
    if density.shape != (len(tri),):
        raise ValueError("density must have one value per element")
    source_normal = (source - center) / np.linalg.norm(source - center)
    rows = []
    for edge in edges:
        if len(owners.get(edge, ())) != 1:
            raise ValueError("each owned cavity edge must have exactly one adjacent element")
        eid = owners[edge][0]
        endpoints = xy[list(edge)]
        midpoint = endpoints.mean(axis=0)
        tensor = np.array(((stresses[eid, 0], stresses[eid, 2]),
                           (stresses[eid, 2], stresses[eid, 1])))
        geometry = cavity_edge_traction_geometry(endpoints[0], endpoints[1], center, tensor)
        tangent = np.asarray(geometry["canonical_edge_tangent"])
        normal = np.asarray(geometry["cavity_outward_into_solid_normal"])
        traction = np.asarray(geometry["traction"])
        length = float(geometry["edge_length_m"])
        arm = midpoint - center
        angle = math.atan2(
            source_normal[0] * arm[1] - source_normal[1] * arm[0],
            float(arm @ source_normal),
        )
        arc = radius * angle
        force = traction * length
        normal_value = float(geometry["normal_traction"])
        tangential_value = float(geometry["tangential_traction"])
        rows.append({
            "edge_node_ids": list(edge), "edge_midpoint_m": midpoint.tolist(),
            "edge_length_m": length, "arc_coordinate_from_source_m": arc,
            "actual_tangent_xy": tangent.tolist(), "actual_outward_normal_xy": normal.tolist(),
            "nominal_circle_normal_xy": (arm / np.linalg.norm(arm)).tolist(),
            "polygon_nominal_normal_dot": float(normal @ (arm / np.linalg.norm(arm))),
            "adjacent_element_id": int(eid), "adjacent_element_quality": float(quality[eid]),
            "raw_sigma_nn_Pa": normal_value, "raw_sigma_nt_Pa": tangential_value,
            "normalized_normal_traction": normal_value / remote,
            "normalized_tangential_traction": tangential_value / remote,
            "traction_vector_Pa": traction.tolist(),
            "traction_l2_numerator_Pa2_m": float(traction @ traction) * length,
            "normal_l2_numerator_Pa2_m": normal_value**2 * length,
            "tangential_l2_numerator_Pa2_m": tangential_value**2 * length,
            "traction_resultant_contribution_N_per_m": force.tolist(),
            "normal_resultant_contribution_N_per_m": normal_value * length,
            "tangential_resultant_contribution_N_per_m": tangential_value * length,
            "traction_moment_contribution_N": float(arm[0] * force[1] - arm[1] * force[0]),
            "plastic_strain": ep[eid].tolist(), "plastic_strain_l2": float(np.linalg.norm(ep[eid])),
            "density_m2": float(density[eid]),
            "distance_from_source_m": float(np.linalg.norm(midpoint - source)),
            "distance_from_crack_cavity_junction_m": float(np.linalg.norm(midpoint - junction)),
        })
    source_rows = [row for row in rows if abs(row["arc_coordinate_from_source_m"]) <= half_width]
    junction_rows = [row for row in rows if row["distance_from_crack_cavity_junction_m"] <= half_width]
    source_ids = {tuple(row["edge_node_ids"]) for row in source_rows}
    junction_ids = {tuple(row["edge_node_ids"]) for row in junction_rows}
    smooth_rows = [
        row for row in rows
        if tuple(row["edge_node_ids"]) not in source_ids | junction_ids
        and float((np.asarray(row["edge_midpoint_m"]) - center) @ source_normal) >= 0.0
    ]
    summaries = {
        "fixed_source_window": _summary(source_rows, remote, radius),
        "smooth_far_side_cavity_arc": _summary(smooth_rows, remote, radius),
        "crack_cavity_junction_neighborhood": _summary(junction_rows, remote, radius),
        "complete_cavity_boundary": _summary(rows, remote, radius),
    }
    complete_l2_squared = summaries["complete_cavity_boundary"][
        "traction_l2_dimensional_Pa_sqrt_m"
    ] ** 2
    for summary in summaries.values():
        summary["fraction_of_complete_l2_squared"] = (
            summary["traction_l2_dimensional_Pa_sqrt_m"] ** 2
            / max(complete_l2_squared, 1.0e-300)
        )
    ranked = sorted(rows, key=lambda row: row["traction_l2_numerator_Pa2_m"], reverse=True)
    row_by_edge = {tuple(row["edge_node_ids"]): row for row in rows}
    supplied_order_rows = [row_by_edge[edge] for edge in supplied_edges]
    supplied_length = sum(row["edge_length_m"] for row in supplied_order_rows)
    supplied_t2 = sum(row["traction_l2_numerator_Pa2_m"] for row in supplied_order_rows)
    total = math.sqrt(supplied_t2 / supplied_length) / remote
    dominant = max(
        ("fixed_source_window", "smooth_far_side_cavity_arc", "crack_cavity_junction_neighborhood"),
        key=lambda name: summaries[name]["traction_l2_dimensional_Pa_sqrt_m"],
    )
    return {
        "schema": SCHEMA, "operator": "RAW_ADJACENT_CST_ARC_DIAGNOSIS_V1",
        "zero_boundary_traction_imposed": False,
        "edge_count": len(rows), "one_owner_boundary_identity": True,
        "source_window_half_width_m": half_width,
        "region_definitions": {
            "fixed_source_window": "absolute signed source arc coordinate <= half width",
            "crack_cavity_junction_neighborhood": "Euclidean edge-midpoint distance to connection entry <= half width",
            "smooth_far_side_cavity_arc": "source-facing half-cycle excluding the fixed source and junction windows",
            "complete_cavity_boundary": "all actual owned cavity edges",
        },
        "remote_traction_scale_Pa": remote, "edge_rows": rows, "region_summaries": summaries,
        "recomputed_raw_adjacent_element_traction_normalized": total,
        "dominant_partition_by_dimensional_l2": dominant,
        "largest_edge_localization": {
            "top_5_fraction_of_complete_l2_squared": sum(
                row["traction_l2_numerator_Pa2_m"] for row in ranked[:5]
            ) / max(complete_l2_squared, 1.0e-300),
            "top_10_fraction_of_complete_l2_squared": sum(
                row["traction_l2_numerator_Pa2_m"] for row in ranked[:10]
            ) / max(complete_l2_squared, 1.0e-300),
            "top_10_edge_node_ids": [row["edge_node_ids"] for row in ranked[:10]],
        },
    }


__all__ = ["SCHEMA", "diagnose_arc_resolved_cavity_traction_v1"]
