"""Operator-consistent fixed-support mechanics for the V3.6 fatigue lane.

The object in this module is deliberately explicit about its continuum
measure.  Plastic distortion, elastic response, deformation diagnostics and
mechanism observations all use the same element supports.  Equal-element
averages are retained only as historical V3.5 audit observables.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from arrhenius_fracture.config import ElasticProperties, GeometryConfig, MeshConfig
from arrhenius_fracture.crystal import bcc_slip_traces, cubic_plane_strain_D
from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh
from arrhenius_fracture.swept_slip_ledger_v10230 import fixed_width_slip_patch
from arrhenius_fracture.unit_slip_perturbation_v1026 import solve_fixed_crack_state

MODEL_ID = "v10.2.30_v3_6_operator_consistent_fixed_support_v1"


def area_weighted(values: np.ndarray, areas: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(areas, float)[mask]
    v = np.asarray(values, float)[mask]
    if not a.size or float(a.sum()) <= 0.0:
        raise ValueError("observation support has zero represented area")
    return float(np.dot(a, v) / a.sum())


def triangle_displacement_gradients(nodes: np.ndarray, elems: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return grad(u)[element, component, coordinate] for linear triangles."""
    xy = np.asarray(nodes, float)[np.asarray(elems, int)]
    uv = np.asarray(u, float).reshape((-1, 2))[np.asarray(elems, int)]
    out = np.zeros((len(elems), 2, 2), float)
    for e, (p, q) in enumerate(zip(xy, uv)):
        M = np.column_stack((np.ones(3), p))
        coeff = np.linalg.solve(M, q)
        out[e] = coeff[1:, :].T
    return out


@dataclass(frozen=True)
class MaterialSlipPatch:
    patch_id: int
    material_anchor_m: float
    gamma: tuple[float, float]


class OperatorConsistentProjection:
    """Linear fixed-crack projection with material-coordinate slip patches."""

    def __init__(self, *, nx: int = 80, ny: int = 160, support_width_m: float = 2e-6,
                 patch_length_m: float = 5e-6, reference_K: float = 50.0,
                 reference_displacement_m: float = 4.552e-6, seed: int = 1733,
                 orientation_signs=(1.0, 1.0), tip_h_fine_m: float = .125e-6):
        self.geom = GeometryConfig(Lx=2e-3, Ly=4e-3, a0=.5e-3, notch_half_thickness=1e-6)
        self.mat = ElasticProperties()
        self.D = cubic_plane_strain_D(523e9, 203e9, 160e9, 0.0)
        self.traces = bcc_slip_traces(0.0)
        self.support_width_m = float(support_width_m)
        self.patch_length_m = float(patch_length_m)
        self.reference_K = float(reference_K)
        self.reference_displacement_m = float(reference_displacement_m)
        self.orientation_signs = np.asarray(orientation_signs, float)
        if self.orientation_signs.shape != (2,) or np.any(abs(self.orientation_signs) != 1.0):
            raise ValueError("orientation_signs must contain two +/-1 entries")
        self.mesh = make_tri_mesh(
            self.geom,
            MeshConfig(nx=int(nx), ny=int(ny), jitter=0.0, tip_h_fine=float(tip_h_fine_m), tip_ratio=1.15),
            seed=int(seed), tip_center=np.array([self.geom.a0, 0.0]),
        )
        self.boundary = make_boundary_data(self.mesh, self.geom)
        damage = np.zeros(self.mesh.nn); damage[self.boundary.notch_nodes] = 1.0
        self.damage = damage
        self.base = solve_fixed_crack_state(
            mesh=self.mesh, boundary=self.boundary, u=np.zeros(self.mesh.ndof),
            ep_gp=np.zeros((3, self.mesh.ne)), rho_gp=np.zeros(self.mesh.ne), d=damage,
            D=self.D, mat=self.mat, Uy_top=self.reference_displacement_m,
            Uy_bot=-self.reference_displacement_m,
        )
        self.centroids = self.mesh.nodes[self.mesh.elems].mean(axis=1)
        xrel = self.centroids[:, 0] - self.geom.a0
        self.observation_masks = {
            "activation_8_16": (xrel >= 8e-6) & (xrel <= 16e-6) & (abs(self.centroids[:, 1]) <= 10e-6),
            "diagnostic_2_8": (xrel >= 2e-6) & (xrel <= 8e-6) & (abs(self.centroids[:, 1]) <= 5e-6),
        }
        self._unit_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def _patch_ep_mask(self, system: int, local_anchor_m: float, gamma: float = 1.0):
        trace = self.traces[int(system)]
        d = np.asarray(trace["t"], float)
        start = np.array([self.geom.a0 + float(local_anchor_m), 0.0])
        end = start + self.patch_length_m / abs(d[0]) * d
        ep, meta = fixed_width_slip_patch(
            self.mesh, start_xy_m=start, end_xy_m=end, direction=d, normal=trace["n"],
            slip_displacement_m=float(gamma) * self.support_width_m * self.orientation_signs[int(system)],
            physical_band_width_m=self.support_width_m,
        )
        return ep, np.any(abs(ep) > 0.0, axis=0), meta

    def unit_response(self, system: int, local_anchor_m: float) -> dict[str, np.ndarray]:
        # Nanometre quantization only stabilizes cache keys; it is far below the mesh scale.
        key = (int(system), int(round(float(local_anchor_m) / 1e-9)))
        if key not in self._unit_cache:
            ep, mask, meta = self._patch_ep_mask(system, local_anchor_m)
            q = solve_fixed_crack_state(
                mesh=self.mesh, boundary=self.boundary, u=self.base["u"], ep_gp=ep,
                rho_gp=np.zeros(self.mesh.ne), d=self.damage, D=self.D, mat=self.mat,
                Uy_top=self.reference_displacement_m, Uy_bot=-self.reference_displacement_m,
            )
            self._unit_cache[key] = {
                "ep": ep, "mask": mask, "u": q["u"] - self.base["u"],
                "sigma": q["sigma_gp"] - self.base["sigma_gp"], "meta": meta,
            }
        return self._unit_cache[key]

    @staticmethod
    def _resolved_shear(stress: np.ndarray, direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
        sx, sy, txy = np.asarray(stress, float)
        d = np.asarray(direction, float); n = np.asarray(normal, float)
        return d[0] * (sx * n[0] + txy * n[1]) + d[1] * (txy * n[0] + sy * n[1])

    def project(self, patches: list[MaterialSlipPatch], *, crack_extension_m: float, K_MPa_sqrt_m: float,
                compute_kinematics: bool = True):
        scale = float(K_MPa_sqrt_m) / self.reference_K
        ep = np.zeros((3, self.mesh.ne)); u = scale * self.base["u"].copy(); sigma = scale * self.base["sigma_gp"].copy()
        support_masks: dict[tuple[int, int], np.ndarray] = {}
        for patch in patches:
            local = float(patch.material_anchor_m) - float(crack_extension_m)
            for system, gamma in enumerate(patch.gamma):
                response = self.unit_response(system, local)
                ep += float(gamma) * response["ep"]
                u += float(gamma) * response["u"]
                sigma += float(gamma) * response["sigma"]
                support_masks[(patch.patch_id, system)] = response["mask"]
        beta = np.zeros((self.mesh.ne, 2, 2))
        beta[:, 0, 0] = ep[0]; beta[:, 1, 1] = ep[1]
        beta[:, 0, 1] = beta[:, 1, 0] = .5 * ep[2]
        if compute_kinematics:
            grad_u = triangle_displacement_gradients(self.mesh.nodes, self.mesh.elems, u)
            strain_total = .5 * (grad_u + grad_u.transpose(0, 2, 1))
            rotation = .5 * (grad_u - grad_u.transpose(0, 2, 1))
            elastic = strain_total - beta
        else:
            grad_u = strain_total = rotation = elastic = None
        opening = {}
        opening_equal = {}
        for name, mask in self.observation_masks.items():
            opening[name] = area_weighted(sigma[1], self.mesh.area_e, mask)
            opening_equal[name] = float(np.mean(sigma[1, mask]))
        shear = np.zeros(2)
        if patches:
            active = max(patches, key=lambda x: x.material_anchor_m)
            for system, trace in enumerate(self.traces):
                mask = support_masks[(active.patch_id, system)]
                tau = self._resolved_shear(sigma, trace["t"], trace["n"])
                shear[system] = self.orientation_signs[system] * area_weighted(tau, self.mesh.area_e, mask)
        return {
            "beta_p": beta, "engineering_eigenstrain": ep, "u": u, "stress": sigma,
            "grad_u": grad_u, "total_strain": strain_total, "rotation": rotation,
            "elastic_strain": elastic, "opening_Pa": opening,
            "opening_equal_element_Pa": opening_equal, "resolved_shear_Pa": shear,
            "support_masks": support_masks,
        }

    def virgin_correspondence(self) -> dict[str, np.ndarray | float]:
        opening = area_weighted(self.base["sigma_gp"][1], self.mesh.area_e, self.observation_masks["activation_8_16"])
        shear = np.zeros(2)
        for system, trace in enumerate(self.traces):
            _, mask, _ = self._patch_ep_mask(system, 0.0, 0.0)
            # gamma=0 creates a zero ep field and therefore no mask; obtain it
            # from the cached unit support instead.
            mask = self.unit_response(system, 0.0)["mask"]
            tau = self._resolved_shear(self.base["sigma_gp"], trace["t"], trace["n"])
            shear[system] = self.orientation_signs[system] * area_weighted(tau, self.mesh.area_e, mask)
        return {"opening_Pa_at_reference_K": opening, "resolved_shear_Pa_at_reference_K": shear}

    def diagnostics(self, result: dict, core_mask: np.ndarray | None = None) -> dict[str, float]:
        mask = np.ones(self.mesh.ne, dtype=bool) if core_mask is None else np.asarray(core_mask, bool)
        def mx(name):
            value = np.asarray(result[name])
            selected = value[:, mask] if name == "engineering_eigenstrain" else value[mask]
            return float(np.max(np.abs(selected)))
        return {
            "projected_beta_p_max_abs": mx("beta_p"),
            "projected_engineering_eigenstrain_max_abs": mx("engineering_eigenstrain"),
            "projected_displacement_gradient_max_abs": mx("grad_u"),
            "projected_rotation_max_abs": mx("rotation"),
            "projected_elastic_strain_max_abs": mx("elastic_strain"),
        }

    def save_reference_fields(self, path: Path) -> None:
        np.savez_compressed(
            path, mesh_nodes=self.mesh.nodes, mesh_elems=self.mesh.elems,
            mesh_area_e=self.mesh.area_e, base_u_reference=self.base["u"],
            base_sigma_reference=self.base["sigma_gp"],
        )


def exact_regularized_inverse_sqrt_integral(left: float, right: float, core: float) -> float:
    """Integral of 1/sqrt(max(x, core)) over one bin."""
    left, right, core = float(left), float(right), float(core)
    if not (0.0 <= left < right and core > 0.0):
        raise ValueError("invalid bin or core")
    if right <= core:
        return (right - left) / math.sqrt(core)
    if left < core:
        return (core - left) / math.sqrt(core) + 2.0 * (math.sqrt(right) - math.sqrt(core))
    return 2.0 * (math.sqrt(right) - math.sqrt(left))


def bin_exact_shielding(state) -> np.ndarray:
    signed = np.asarray(state.signed_gnd_m2(), float)
    factors = np.asarray(state.c.shielding_orientation_factors, float)
    core = float(state.c.core_regularization_b * state.c.b_m)
    dx = float(state.dx)
    integrals = np.array([exact_regularized_inverse_sqrt_integral(i * dx, (i + 1) * dx, core) for i in range(state.c.n_bins)])
    prefactor = state.c.G_Pa * state.c.b_m / (1.0 - state.c.nu) / math.sqrt(2.0 * math.pi)
    # Piecewise-constant density times physical strip width; convert Pa sqrt(m) to MPa sqrt(m).
    kernel = prefactor * float(state.state_strip_width_m) * integrals / 1e6
    return np.sum(factors[:, None] * signed * kernel[None, :], axis=1)


__all__ = ["MODEL_ID", "MaterialSlipPatch", "OperatorConsistentProjection", "area_weighted",
           "triangle_displacement_gradients", "exact_regularized_inverse_sqrt_integral", "bin_exact_shielding"]
