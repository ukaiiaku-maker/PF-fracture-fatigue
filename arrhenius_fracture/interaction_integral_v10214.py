"""Signed interaction integral with intrinsic isotropy verification.

The production tungsten stiffness is assembled from cubic constants.  For the
baseline constants ``2*C44 == C11-C12`` exactly, so the resulting plane-strain
matrix is isotropic.  The legacy guard compared that matrix against separately
rounded ``E`` and ``nu`` metadata and incorrectly classified the approximately
0.2 percent metadata mismatch as material anisotropy.

This wrapper determines isotropy from the supplied stiffness matrix itself,
derives the consistent Lamé/Young/Poisson constants, and then evaluates the
reviewed analytic Williams interaction integral with those constants.  A truly
anisotropic matrix still fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ElasticProperties, JIntegralConfig
from .interaction_integral_v1026 import InteractionIntegralResult
from .interaction_integral_v1029 import (
    MODEL_ID as UNDERLYING_MODEL_ID,
    compute_signed_interaction_integral as _compute_isotropic,
    isotropic_plane_strain_D,
)

MODEL_ID = "v10.2.14_signed_intrinsic_isotropy_interaction_integral"


@dataclass(frozen=True)
class IntrinsicIsotropicFit:
    lame_lambda_Pa: float
    lame_mu_Pa: float
    E_Pa: float
    poisson: float
    reconstructed_D: np.ndarray
    maximum_relative_residual: float
    metadata_D_maximum_relative_difference: float

    def audit_payload(self) -> dict[str, Any]:
        return {
            "model": "plane_strain_isotropic_lame_least_squares",
            "lame_lambda_Pa": float(self.lame_lambda_Pa),
            "lame_mu_Pa": float(self.lame_mu_Pa),
            "derived_E_Pa": float(self.E_Pa),
            "derived_poisson": float(self.poisson),
            "maximum_relative_intrinsic_isotropy_residual": float(
                self.maximum_relative_residual
            ),
            "metadata_D_maximum_relative_difference": float(
                self.metadata_D_maximum_relative_difference
            ),
            "elastic_constants_for_auxiliary_field": "derived_from_supplied_D",
        }


def fit_intrinsic_isotropic_plane_strain(
    D: np.ndarray,
    mat: ElasticProperties,
    *,
    relative_tolerance: float = 1.0e-8,
) -> IntrinsicIsotropicFit:
    """Fit ``D=lambda*A+mu*B`` and reject a genuinely anisotropic matrix.

    Engineering-shear plane-strain isotropy has

    ``D=[[lambda+2mu, lambda, 0],
        [lambda, lambda+2mu, 0],
        [0, 0, mu]]``.

    The fit tests the supplied matrix against that constitutive subspace.  It does
    not compare against independently rounded material metadata.
    """
    supplied = np.asarray(D, dtype=float)
    if supplied.shape != (3, 3) or not np.all(np.isfinite(supplied)):
        raise ValueError("interaction-integral stiffness must be a finite 3x3 matrix")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("intrinsic isotropy tolerance must be positive and finite")

    A = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    B = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    design = np.column_stack([A.ravel(), B.ravel()])
    coefficients, _, rank, _ = np.linalg.lstsq(design, supplied.ravel(), rcond=None)
    if int(rank) != 2:
        raise RuntimeError("isotropic plane-strain stiffness fit is rank deficient")
    lame_lambda = float(coefficients[0])
    lame_mu = float(coefficients[1])
    reconstructed = lame_lambda * A + lame_mu * B
    scale = max(float(np.max(np.abs(supplied))), 1.0)
    residual = float(np.max(np.abs(supplied - reconstructed))) / scale
    if residual > tolerance:
        raise ValueError(
            "supplied stiffness is materially anisotropic for the Williams "
            f"auxiliary field: intrinsic isotropy residual={residual:.3e}, "
            f"tolerance={tolerance:.3e}"
        )
    if not (lame_mu > 0.0 and lame_lambda + 2.0 * lame_mu > 0.0):
        raise ValueError("supplied isotropic stiffness is not mechanically stable")
    denominator = lame_lambda + lame_mu
    if denominator <= 0.0:
        raise ValueError("supplied isotropic stiffness has invalid Lamé constants")
    poisson = lame_lambda / (2.0 * denominator)
    E = lame_mu * (3.0 * lame_lambda + 2.0 * lame_mu) / denominator
    if not (np.isfinite(E) and E > 0.0 and np.isfinite(poisson) and -1.0 < poisson < 0.5):
        raise ValueError("failed to derive admissible isotropic elastic constants")

    metadata_reference = isotropic_plane_strain_D(mat.E, mat.nu)
    metadata_scale = max(float(np.max(np.abs(metadata_reference))), 1.0)
    metadata_difference = float(
        np.max(np.abs(supplied - metadata_reference)) / metadata_scale
    )
    return IntrinsicIsotropicFit(
        lame_lambda_Pa=lame_lambda,
        lame_mu_Pa=lame_mu,
        E_Pa=E,
        poisson=poisson,
        reconstructed_D=reconstructed,
        maximum_relative_residual=residual,
        metadata_D_maximum_relative_difference=metadata_difference,
    )


def compute_signed_interaction_integral(
    mesh,
    u: np.ndarray,
    sigma_gp: np.ndarray,
    d: np.ndarray,
    crack_tip: np.ndarray,
    crack_direction: np.ndarray,
    mat: ElasticProperties,
    ell: float,
    *,
    cfg: JIntegralConfig | None = None,
    crack_segments=None,
    exclude_radius: float = 0.0,
    D: np.ndarray | None = None,
    isotropy_relative_tolerance: float = 1.0e-8,
) -> InteractionIntegralResult:
    """Evaluate signed KI/KII using elastic constants consistent with ``D``."""
    if D is None:
        fit = fit_intrinsic_isotropic_plane_strain(
            isotropic_plane_strain_D(mat.E, mat.nu),
            mat,
            relative_tolerance=isotropy_relative_tolerance,
        )
    else:
        fit = fit_intrinsic_isotropic_plane_strain(
            D,
            mat,
            relative_tolerance=isotropy_relative_tolerance,
        )
    effective_mat = ElasticProperties(
        E=float(fit.E_Pa),
        nu=float(fit.poisson),
        b=float(mat.b),
        Tm=float(mat.Tm),
    )
    underlying = _compute_isotropic(
        mesh,
        u,
        sigma_gp,
        d,
        crack_tip,
        crack_direction,
        effective_mat,
        ell,
        cfg=cfg,
        crack_segments=crack_segments,
        exclude_radius=exclude_radius,
        D=fit.reconstructed_D,
        isotropy_relative_tolerance=isotropy_relative_tolerance,
    )
    diagnostics = {
        **underlying.diagnostics,
        "schema": MODEL_ID,
        "underlying_schema": UNDERLYING_MODEL_ID,
        "intrinsic_isotropy_verified": True,
        "rounded_metadata_mismatch_does_not_define_anisotropy": True,
        "intrinsic_isotropic_fit": fit.audit_payload(),
    }
    return InteractionIntegralResult(
        K_I_Pa_sqrt_m=float(underlying.K_I_Pa_sqrt_m),
        K_II_Pa_sqrt_m=float(underlying.K_II_Pa_sqrt_m),
        M_I_m_per_Pa=float(underlying.M_I_m_per_Pa),
        M_II_m_per_Pa=float(underlying.M_II_m_per_Pa),
        diagnostics=diagnostics,
    )


__all__ = [
    "MODEL_ID",
    "UNDERLYING_MODEL_ID",
    "IntrinsicIsotropicFit",
    "fit_intrinsic_isotropic_plane_strain",
    "compute_signed_interaction_integral",
]
