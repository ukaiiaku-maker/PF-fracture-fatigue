"""Experiment-facing compact-tension mappings for v10.2.30 fatigue studies.

These functions are passive LEFM reporting/control utilities.  They do not
change the sharp-front constitutive law or infer parameters from crack growth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CompactTensionGeometry:
    width_m: float
    thickness_m: float
    initial_crack_m: float

    def validate(self) -> "CompactTensionGeometry":
        if self.width_m <= 0.0 or self.thickness_m <= 0.0:
            raise ValueError("C(T) width and thickness must be positive")
        x = self.initial_crack_m / self.width_m
        if not 0.0 < x < 1.0:
            raise ValueError("C(T) initial a/W must lie strictly between zero and one")
        return self


def ct_geometry_factor(x: float) -> float:
    """Return the standard dimensionless C(T) geometry function F(a/W)."""
    x = float(x)
    if not 0.0 < x < 1.0:
        raise ValueError("C(T) a/W must lie strictly between zero and one")
    polynomial = 0.886 + 4.64*x - 13.32*x*x + 14.72*x**3 - 5.60*x**4
    return (2.0 + x) * polynomial / (1.0 - x)**1.5


def ct_k_from_load_pa_sqrt_m(load_n: float, geometry: CompactTensionGeometry,
                             crack_m: float) -> float:
    geometry.validate()
    return float(load_n) * ct_geometry_factor(float(crack_m) / geometry.width_m) / (
        geometry.thickness_m * math.sqrt(geometry.width_m)
    )


def ct_load_from_k_pa_sqrt_m(k_pa_sqrt_m: float, geometry: CompactTensionGeometry,
                             crack_m: float) -> float:
    geometry.validate()
    return float(k_pa_sqrt_m) * geometry.thickness_m * math.sqrt(geometry.width_m) / (
        ct_geometry_factor(float(crack_m) / geometry.width_m)
    )


def projected_macro_crack_m(geometry: CompactTensionGeometry,
                            projected_extension_m: float) -> float:
    crack_m = geometry.initial_crack_m + float(projected_extension_m)
    if crack_m >= geometry.width_m:
        raise ValueError("projected macroscopic crack reaches C(T) ligament boundary")
    return crack_m


def nominal_ranges(kmax_pa_sqrt_m: float, R: float) -> dict[str, float]:
    kmax = float(kmax_pa_sqrt_m); ratio = float(R); kmin = ratio*kmax
    return {
        "Kmax_Pa_sqrt_m": kmax,
        "Kmin_Pa_sqrt_m": kmin,
        "deltaK_full_Pa_sqrt_m": kmax-kmin,
        "deltaK_tensile_Pa_sqrt_m": kmax-max(kmin, 0.0),
    }


def energy_equivalent_driver_k_pa_sqrt_m(k_nominal_pa_sqrt_m: float,
                                          macro_Eprime_pa: float,
                                          local_Eprime_pa: float) -> float:
    """Map nominal K to local-driver K by J=K^2/E' equivalence."""
    if macro_Eprime_pa <= 0.0 or local_Eprime_pa <= 0.0:
        raise ValueError("positive effective moduli are required")
    return float(k_nominal_pa_sqrt_m) * math.sqrt(local_Eprime_pa/macro_Eprime_pa)


PRIMARY_CT = CompactTensionGeometry(10.0e-3, 2.5e-3, 5.0e-3)
SENSITIVITY_CT = CompactTensionGeometry(25.0e-3, 6.25e-3, 12.5e-3)

