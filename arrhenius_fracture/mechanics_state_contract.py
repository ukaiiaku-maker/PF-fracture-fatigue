"""Explicit in-plane loading and out-of-plane constraint contracts.

The production solvers remain two-dimensional.  This module therefore does
not advertise mode III or a finite-thickness 3-D solution.  It supplies the
full symmetric stress tensor required by constitutive consumers and makes the
transverse assumption mechanically explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import numpy as np


class PlaneState(str, Enum):
    PLANE_STRAIN = "plane_strain"
    PLANE_STRESS = "plane_stress"
    GENERALIZED_PLANE_STRAIN = "generalized_plane_strain"


class InPlaneMode(str, Enum):
    MODE_I = "mode_I"
    MODE_II = "mode_II"
    MIXED_I_II = "mixed_I_II"


class ConstraintControl(str, Enum):
    FREE_TRANSVERSE = "free_transverse_resultant"
    PRESCRIBED_EPS_ZZ = "prescribed_eps_zz"
    PRESCRIBED_SIGMA_ZZ = "prescribed_sigma_zz"


@dataclass(frozen=True)
class MechanicsStateOptions:
    plane_state: PlaneState = PlaneState.PLANE_STRAIN
    in_plane_mode: InPlaneMode = InPlaneMode.MODE_I
    constraint_control: ConstraintControl = ConstraintControl.PRESCRIBED_EPS_ZZ
    eps_zz: float | None = 0.0
    sigma_zz_Pa: float | None = None
    mode_I_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name, enum_type in (("plane_state", PlaneState),
                                ("in_plane_mode", InPlaneMode),
                                ("constraint_control", ConstraintControl)):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        f = float(self.mode_I_fraction)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError("mode_I_fraction must lie in [0, 1]")
        object.__setattr__(self, "mode_I_fraction", f)
        if self.in_plane_mode is InPlaneMode.MODE_I and f != 1.0:
            raise ValueError("mode_I requires mode_I_fraction=1")
        if self.in_plane_mode is InPlaneMode.MODE_II and f != 0.0:
            raise ValueError("mode_II requires mode_I_fraction=0")
        if self.in_plane_mode is InPlaneMode.MIXED_I_II and f in (0.0, 1.0):
            raise ValueError("mixed_I_II requires a strictly mixed fraction")
        if self.plane_state is PlaneState.PLANE_STRESS:
            if self.constraint_control is not ConstraintControl.FREE_TRANSVERSE:
                raise ValueError("plane stress requires a free transverse resultant")
            if self.sigma_zz_Pa not in (None, 0.0):
                raise ValueError("plane stress requires sigma_zz=0")
        elif self.plane_state is PlaneState.PLANE_STRAIN:
            if self.constraint_control is not ConstraintControl.PRESCRIBED_EPS_ZZ:
                raise ValueError("plane strain requires prescribed eps_zz")
            if self.eps_zz not in (0, 0.0):
                raise ValueError("plane strain requires eps_zz=0")
        else:
            prescribed = (self.eps_zz is not None, self.sigma_zz_Pa is not None)
            if prescribed == (True, True) or prescribed == (False, False):
                raise ValueError("generalized plane strain requires exactly one transverse control")
            if (self.constraint_control is ConstraintControl.PRESCRIBED_EPS_ZZ) != prescribed[0]:
                raise ValueError("constraint_control does not match the prescribed value")

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MechanicsStateOptions":
        return cls(**payload)


def isotropic_stress_tensor(strain: np.ndarray, E_Pa: float, nu: float, *,
                            transverse_sigma_Pa: float | None = None) -> np.ndarray:
    """Return a 3-D Cauchy tensor from a symmetric small-strain tensor.

    When ``transverse_sigma_Pa`` is specified, eps_zz is solved so that the
    requested resultant is satisfied while all other strain components are
    retained.
    """
    eps = np.asarray(strain, dtype=float).copy()
    if eps.shape != (3, 3) or not np.allclose(eps, eps.T):
        raise ValueError("strain must be a symmetric 3x3 tensor")
    if E_Pa <= 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("invalid isotropic elastic constants")
    lam = E_Pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_Pa / (2.0 * (1.0 + nu))
    if transverse_sigma_Pa is not None:
        eps[2, 2] = (float(transverse_sigma_Pa) - lam * (eps[0, 0] + eps[1, 1])) / (lam + 2.0 * mu)
    return lam * np.trace(eps) * np.eye(3) + 2.0 * mu * eps


def full_tensor_from_in_plane(strain_xx: float, strain_yy: float,
                              engineering_gamma_xy: float, E_Pa: float, nu: float,
                              options: MechanicsStateOptions) -> tuple[np.ndarray, np.ndarray]:
    lam = E_Pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_Pa / (2.0 * (1.0 + nu))
    if options.plane_state is PlaneState.PLANE_STRESS:
        eps_zz = -lam * (strain_xx + strain_yy) / (lam + 2.0 * mu)
    elif options.sigma_zz_Pa is not None:
        eps_zz = (float(options.sigma_zz_Pa)
                  - lam * (strain_xx + strain_yy)) / (lam + 2.0 * mu)
    else:
        eps_zz = float(options.eps_zz or 0.0)
    eps = np.array([[strain_xx, 0.5 * engineering_gamma_xy, 0.0],
                    [0.5 * engineering_gamma_xy, strain_yy, 0.0],
                    [0.0, 0.0, eps_zz]])
    return eps, isotropic_stress_tensor(eps, E_Pa, nu)


def rotate_tensor(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if tensor.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("tensor and rotation must be 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) or np.linalg.det(rotation) < 0.0:
        raise ValueError("rotation must be proper orthogonal")
    return rotation @ tensor @ rotation.T


def resolved_traction(stress: np.ndarray, normal: np.ndarray,
                      direction: np.ndarray) -> tuple[float, float]:
    stress = np.asarray(stress, dtype=float)
    n = np.asarray(normal, dtype=float)
    s = np.asarray(direction, dtype=float)
    n /= np.linalg.norm(n); s /= np.linalg.norm(s)
    if abs(float(n @ s)) > 1e-12:
        raise ValueError("normal and direction must be orthogonal")
    traction = stress @ n
    return float(n @ traction), float(s @ traction)


def elastic_work_density(stress: np.ndarray, strain: np.ndarray) -> float:
    """Small-strain work with tensor shear counted exactly twice by contraction."""
    return 0.5 * float(np.einsum("ij,ij->", stress, strain))


def constrained_elastic_potential_density(stress: np.ndarray, strain: np.ndarray,
                                           options: MechanicsStateOptions) -> float:
    """Reduced elastic potential for the declared transverse control."""
    potential = elastic_work_density(stress, strain)
    if options.sigma_zz_Pa is not None:
        potential -= float(options.sigma_zz_Pa) * float(strain[2, 2])
    return potential


__all__ = ["ConstraintControl", "InPlaneMode", "MechanicsStateOptions", "PlaneState",
           "constrained_elastic_potential_density", "elastic_work_density",
           "full_tensor_from_in_plane", "isotropic_stress_tensor", "resolved_traction",
           "rotate_tensor"]
