#!/usr/bin/env python3
"""Generate candidate-independent PF mechanics/source maps for 15/30/45 degrees.

These are production-discrete sharp-wake maps, not continuum G.  The wake is
advanced in the physical crack direction by the production 5 micrometre event
length; the reduced coordinate is projected x extension.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture.anisotropic_emission_v10174 import AnisotropicEmissionConfig, build_front_drive
from arrhenius_fracture.config import ElasticProperties, GeometryConfig, JIntegralConfig, MeshConfig
from arrhenius_fracture.crack_backend import SharpWakeBackend
from arrhenius_fracture.crystal import cubic_plane_strain_D
from arrhenius_fracture.fem import assemble_mechanics, elastic_energy_densities, solve_dirichlet
from arrhenius_fracture.j_integral import compute_J_integral
from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh


REFERENCE_OPENING_M = 1.0e-5
PHYSICAL_EVENT_LENGTH_M = 5.0e-6
RADII_UM = (1.0, 2.0, 4.0, 8.0, 12.0, 25.0, 50.0)
PHYSICAL_SOURCE_COMMIT = "9e884fb0b0845da621d2612bdf1042e481b8df49"


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def initial_damage(mesh, geometry) -> np.ndarray:
    x, y = np.asarray(mesh.nodes).T
    damage = np.zeros(mesh.nn)
    damage[(x <= geometry.a0) & (np.abs(y) <= geometry.notch_half_thickness)] = 1.0
    return damage


def state(mesh, boundary, damage, D, material, tip, tangent, segments, kill_radius,
          opening_m=REFERENCE_OPENING_M):
    u0 = np.zeros(mesh.ndof); ep = np.zeros((3, mesh.ne)); rho = np.zeros(mesh.ne)
    K, R0, *_ = assemble_mechanics(mesh, u0, ep, rho, damage, D, material, kappa=1e-6)
    u, reaction = solve_dirichlet(K, R0, u0, boundary, .5 * opening_m, -.5 * opening_m)
    _, _, sigma, *_ = assemble_mechanics(mesh, u, ep, rho, damage, D, material, kappa=1e-6)
    stored, _ = elastic_energy_densities(mesh, u, ep, sigma, D)
    energy = float(np.sum(stored * mesh.area_e))
    _, _, info = compute_J_integral(
        mesh, u, sigma, stored, damage, tip, tangent, material,
        ell=80e-6, cfg=JIntegralConfig(), crack_segments=segments,
        exclude_radius=2.0 * kill_radius,
    )
    signed = float(info.get("J_signed", info.get("J", 0.0)) or 0.0)
    effective = max(signed, 0.0)
    return u, sigma, {
        "reaction_N_per_m": float(reaction),
        "compliance_m2_per_N": float(opening_m / reaction),
        "elastic_energy_J_per_m": energy,
        "native_J_J_per_m2": effective,
        "native_J_signed_J_per_m2": signed,
        "native_KJ_MPa_sqrt_m": float(np.sqrt(effective * material.Eprime) / 1e6),
        "domain_metadata_json": json.dumps({key: float(value) for key, value in info.items()
                                             if isinstance(value, (int, float, np.integer, np.floating))}, sort_keys=True),
    }


def generate(theta_deg: float, target_projected_um: float, out: Path) -> dict[str, Any]:
    theta = math.radians(theta_deg); tangent = np.array([math.cos(theta), math.sin(theta)])
    geometry = GeometryConfig(); config = MeshConfig(nx=36, ny=72, jitter=0.0, tip_h_fine=1e-6, tip_ratio=1.2)
    mesh = make_tri_mesh(geometry, config, seed=42, tip_center=np.array([geometry.a0, 0.0]))
    boundary = make_boundary_data(mesh, geometry); material = ElasticProperties()
    D = cubic_plane_strain_D(523e9, 203e9, 160e9, theta_deg)
    damage = initial_damage(mesh, geometry); displacement = np.zeros(mesh.ndof)
    backend = SharpWakeBackend(); kill = max(float(mesh.hbar_tip), .5e-6)
    tip0 = np.array([geometry.a0, 0.0]); tip = tip0.copy()
    segments = [(np.array([0.0, 0.0]), tip0.copy())]
    projected_step_m = PHYSICAL_EVENT_LENGTH_M * tangent[0]
    event_count = int(math.ceil(target_projected_um * 1e-6 / projected_step_m - 1e-14))
    mechanics: list[dict[str, Any]] = []; drives: list[dict[str, Any]] = []
    linearity_checks: list[dict[str, Any]] = []
    for count in range(event_count + 1):
        if count:
            p0 = tip.copy(); p1 = p0 + PHYSICAL_EVENT_LENGTH_M * tangent
            result = backend.advance(mesh=mesh, boundary=boundary, damage=damage,
                                     displacement=displacement, p0=p0, p1=p1,
                                     direction=tangent, front_id=0, kill_r=kill)
            if not result.inserted:
                raise RuntimeError(f"theta={theta_deg:g} wake event {count} failed: {result.reason}")
            damage = result.damage; tip = p1; segments.append((p0, p1))
        u, sigma, metrics = state(mesh, boundary, damage, D, material, tip, tangent, segments, kill)
        projected_um = count * projected_step_m * 1e6
        mechanics.append({
            "theta_deg": theta_deg, "target_extension_um": projected_um,
            "actual_extension_um": projected_um, "physical_path_extension_um": count * 5.0,
            "geometry_event_count": count, "reference_opening_m": REFERENCE_OPENING_M,
            **metrics, "F_over_U_N_per_m2": metrics["reaction_N_per_m"] / REFERENCE_OPENING_M,
            "J_native_over_U2": metrics["native_J_J_per_m2"] / REFERENCE_OPENING_M**2,
            "KJ_native_over_U": metrics["native_KJ_MPa_sqrt_m"] / REFERENCE_OPENING_M,
            "wake_width_m": 2.0 * kill, "mesh_scale_m": float(mesh.hbar_tip),
            "tip_x_m": float(tip[0]), "tip_y_m": float(tip[1]),
            "mesh_fingerprint": digest_arrays(mesh.nodes, mesh.elems),
            "damage_wake_fingerprint": digest_arrays(damage),
        })
        for radius_um in RADII_UM:
            probe = build_front_drive(mesh, sigma, damage, tip, AnisotropicEmissionConfig(
                crystal_theta_deg=theta_deg, probe_radius_m=radius_um * 1e-6,
            ))
            opening_stress = float(probe["sigma_amplitude_Pa"])
            drives.append({
                "backend": "PF", "theta_deg": theta_deg, "extension_um": projected_um,
                "physical_path_extension_um": count * 5.0, "tip_radius_um": radius_um,
                "reference_opening_m": REFERENCE_OPENING_M,
                "drive_factor_system_0": float(probe["drive_factors"][0]),
                "drive_factor_system_1": float(probe["drive_factors"][1]),
                "tau_signed_system_0_Pa": float(probe["tau_signed_Pa"][0]),
                "tau_signed_system_1_Pa": float(probe["tau_signed_Pa"][1]),
                "opening_stress_Pa": opening_stress,
                "sigma_nn_probe_Pa": float(probe["sigma_nn_probe_Pa"]),
                "sigma1_probe_Pa": float(probe["sigma1_probe_Pa"]),
                "tau_over_opening_system_0": float(probe["tau_signed_Pa"][0] / max(abs(opening_stress), 1e-300)),
                "tau_over_opening_system_1": float(probe["tau_signed_Pa"][1] / max(abs(opening_stress), 1e-300)),
                "reliable": bool(probe["reliable"]),
                "field_snapshot_hash": digest_arrays(u, sigma),
                "topology_fingerprint": digest_arrays(damage),
                "source_commit": PHYSICAL_SOURCE_COMMIT,
            })
        if count in {0, event_count // 2, event_count}:
            for scale in (0.5, 1.5):
                _, sigma_scaled, scaled = state(
                    mesh, boundary, damage, D, material, tip, tangent, segments,
                    kill, REFERENCE_OPENING_M * scale,
                )
                scaled_probe = build_front_drive(
                    mesh, sigma_scaled, damage, tip,
                    AnisotropicEmissionConfig(crystal_theta_deg=theta_deg,
                                              probe_radius_m=4.0e-6),
                )
                expected = {
                    "reaction_N_per_m": metrics["reaction_N_per_m"] * scale,
                    "elastic_energy_J_per_m": metrics["elastic_energy_J_per_m"] * scale**2,
                    "native_J_J_per_m2": metrics["native_J_J_per_m2"] * scale**2,
                    "native_KJ_MPa_sqrt_m": metrics["native_KJ_MPa_sqrt_m"] * scale,
                    "source_sigma_amplitude_Pa": next(
                        row["opening_stress_Pa"] for row in drives
                        if row["extension_um"] == projected_um and row["tip_radius_um"] == 4.0
                    ) * scale,
                }
                observed = {
                    **{key: scaled[key] for key in (
                        "reaction_N_per_m", "elastic_energy_J_per_m",
                        "native_J_J_per_m2", "native_KJ_MPa_sqrt_m")},
                    "source_sigma_amplitude_Pa": float(scaled_probe["sigma_amplitude_Pa"]),
                }
                for quantity in expected:
                    denominator = max(abs(expected[quantity]), 1e-30)
                    linearity_checks.append({
                        "extension_um": projected_um,
                        "opening_scale": scale,
                        "quantity": quantity,
                        "expected": expected[quantity],
                        "observed": observed[quantity],
                        "relative_error": abs(observed[quantity] - expected[quantity]) / denominator,
                    })
    mechanics_path = out / f"pf_v2_theta{theta_deg:g}_mechanics_map.csv"
    drive_path = out / f"pf_v2_theta{theta_deg:g}_source_drive_map.csv"
    write_csv(mechanics_path, mechanics); write_csv(drive_path, drives)
    if not all(row["reliable"] for row in drives):
        raise RuntimeError(f"theta={theta_deg:g} has unreliable source probes")
    interpolation_errors: dict[str, float] = {}
    for actual_key in ("reaction_N_per_m", "elastic_energy_J_per_m",
                       "native_J_J_per_m2", "native_KJ_MPa_sqrt_m"):
        errors = []
        for index in range(1, len(mechanics) - 1):
            estimate = 0.5 * (float(mechanics[index - 1][actual_key]) + float(mechanics[index + 1][actual_key]))
            actual = float(mechanics[index][actual_key])
            errors.append(abs(estimate - actual) / max(abs(actual), 1e-30))
        interpolation_errors[actual_key] = max(errors, default=0.0)
    max_linearity_error = max(row["relative_error"] for row in linearity_checks)
    if max_linearity_error > 1e-8:
        raise RuntimeError(f"theta={theta_deg:g} load linearity failed: {max_linearity_error}")
    return {
        "theta_deg": theta_deg, "projected_event_increment_um": projected_step_m * 1e6,
        "event_count": event_count, "maximum_projected_extension_um": mechanics[-1]["actual_extension_um"],
        "mechanics_map": str(mechanics_path), "mechanics_map_sha256": hashlib.sha256(mechanics_path.read_bytes()).hexdigest(),
        "source_drive_map": str(drive_path), "source_drive_map_sha256": hashlib.sha256(drive_path.read_bytes()).hexdigest(),
        "mechanics_semantics": "PF_MODEL_NATIVE_PRODUCTION_DISCRETE_SHARP_WAKE_NOT_CONTINUUM_G",
        "candidate_independent": True, "no_extrapolation": True,
        "tip_radius_bounds_um": [min(RADII_UM), max(RADII_UM)],
        "load_linearity_check_count": len(linearity_checks),
        "maximum_load_scaling_relative_error": max_linearity_error,
        "load_scaling_passed": True,
        "interpolation_uncertainty_method": "leave_one_geometry_event_out_linear_midpoint_relative_error",
        "maximum_interpolation_relative_error_by_quantity": interpolation_errors,
        "bounds_policy": "FAIL_CLOSED_NO_EXTRAPOLATION_BEYOND_RECORDED_EXTENSION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thetas", type=float, nargs="+", default=(15.0, 30.0, 45.0))
    parser.add_argument("--target-projected-um", type=float, default=1000.0)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    records = [generate(theta, args.target_projected_um, args.out) for theta in args.thetas]
    manifest = {"schema": "pf_canonical_angle_provider_maps_v1", "physical_source_commit": PHYSICAL_SOURCE_COMMIT,
                "target_projected_extension_um": args.target_projected_um, "maps": records}
    (args.out / "pf_canonical_angle_provider_map_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
