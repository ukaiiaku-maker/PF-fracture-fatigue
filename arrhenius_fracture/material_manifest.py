"""Portable material manifests for the unified sharp-front MPZ solver.

The CSV records are the exact v9.10.2/v9.10.3 promoted ceramic, weakT and
DBTT candidates.  This module performs no fitting and introduces no legacy
back-stress, saturation or cohesive-zone parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

KB_EV_PER_K = 8.617333262145e-5
TREF_K = 481.33


def _f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"manifest field {key!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"manifest field {key!r} is not finite")
    return value


@dataclass(frozen=True)
class ExpFloorBarrier:
    G00_eV: float
    gT_eV_per_K: float
    sigc0_Pa: float
    sT_Pa_per_K: float
    alpha: float
    exponent: float
    floor_fraction: float
    floor_min_eV: float = 1.0e-4
    floor_max_fraction: float = 0.95
    Tref_K: float = TREF_K
    attempt_frequency_s: float = 1.0e12

    def values_eV(self, stress_Pa: np.ndarray | float, T_K: float) -> np.ndarray:
        sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        dT = float(T_K) - self.Tref_K
        G0 = max(self.G00_eV + self.gT_eV_per_K * dT, 1.0e-12)
        sigc = max(self.sigc0_Pa + self.sT_Pa_per_K * dT, 1.0)
        raw_floor = max(self.floor_min_eV, self.floor_fraction * G0)
        floor = min(self.floor_max_fraction * G0, raw_floor)
        return np.maximum(
            floor + (G0 - floor) * np.exp(
                -max(self.alpha, 0.0)
                * np.power(sigma / sigc, max(self.exponent, 1.0e-9))
            ),
            0.0,
        )

    def rate(self, stress_Pa: np.ndarray | float, T_K: float) -> np.ndarray:
        G = self.values_eV(stress_Pa, T_K)
        return self.attempt_frequency_s * np.exp(
            np.clip(-G / max(KB_EV_PER_K * float(T_K), 1.0e-30), -700.0, 0.0)
        )

    def diagnostic(self, stress_Pa: float, T_K: float) -> dict[str, Any]:
        """Evaluate the executed surface and its local piecewise derivatives.

        Thermal derivatives hold physical stress and all external state fixed.
        A separate fixed-normalized-stress derivative is reported because it
        is not the same thermodynamic quantity when ``sigc`` varies with T.
        At executed max/min kinks the corresponding derivative is ``None``;
        no fictitious smooth value is supplied.
        """
        sigma_input = float(stress_Pa)
        T = float(T_K)
        if not math.isfinite(sigma_input) or not math.isfinite(T) or T <= 0.0:
            raise ValueError("stress and positive temperature must be finite")
        sigma = max(sigma_input, 0.0)
        raw_G0 = self.G00_eV + self.gT_eV_per_K * (T - self.Tref_K)
        raw_sigc = self.sigc0_Pa + self.sT_Pa_per_K * (T - self.Tref_K)
        G0 = max(raw_G0, 1.0e-12)
        sigc = max(raw_sigc, 1.0)
        kink_G0 = math.isclose(raw_G0, 1.0e-12, rel_tol=0.0, abs_tol=1.0e-14)
        kink_sigc = math.isclose(raw_sigc, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
        dG0 = None if kink_G0 else (self.gT_eV_per_K if raw_G0 > 1.0e-12 else 0.0)
        dsigc = None if kink_sigc else (self.sT_Pa_per_K if raw_sigc > 1.0 else 0.0)

        fractional = self.floor_fraction * G0
        lower_kink = math.isclose(fractional, self.floor_min_eV, rel_tol=1e-12, abs_tol=1e-15)
        lower = max(self.floor_min_eV, fractional)
        upper = self.floor_max_fraction * G0
        upper_kink = math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-15)
        floor = min(upper, lower)
        if lower_kink or upper_kink or dG0 is None:
            dfloor = None
            floor_branch = "kink"
        elif upper < lower:
            dfloor = self.floor_max_fraction * dG0
            floor_branch = "maximum_fraction_cap"
        elif fractional > self.floor_min_eV:
            dfloor = self.floor_fraction * dG0
            floor_branch = "fractional_floor"
        else:
            dfloor = 0.0
            floor_branch = "absolute_floor"

        alpha = max(self.alpha, 0.0)
        n = max(self.exponent, 1.0e-9)
        q = (sigma / sigc) ** n
        attenuation = math.exp(-alpha * q)
        barrier = floor + (G0 - floor) * attenuation
        if dG0 is None or dsigc is None or dfloor is None:
            dT_fixed_stress = None
            dT_fixed_normalized = None
        else:
            d_attenuation = attenuation * alpha * n * q * dsigc / sigc
            dT_fixed_stress = (dfloor + (dG0 - dfloor) * attenuation
                               + (G0 - floor) * d_attenuation)
            dT_fixed_normalized = dfloor + (dG0 - dfloor) * attenuation

        if sigma_input < 0.0:
            dsigma = 0.0
            d2sigma = 0.0
            stress_branch = "negative_stress_gate"
        elif sigma == 0.0:
            stress_branch = "zero_stress_one_sided"
            if n > 2.0:
                dsigma, d2sigma = 0.0, 0.0
            elif math.isclose(n, 2.0):
                dsigma = 0.0
                d2sigma = -(G0 - floor) * alpha * n / sigc**2
            elif n > 1.0:
                dsigma, d2sigma = 0.0, None
            elif math.isclose(n, 1.0):
                dsigma = -(G0 - floor) * alpha / sigc
                d2sigma = (G0 - floor) * alpha**2 / sigc**2
            else:
                dsigma, d2sigma = None, None
        else:
            stress_branch = "positive_stress_smooth"
            c = alpha / sigc**n
            slope_factor = c * n * sigma ** (n - 1.0)
            dsigma = -(G0 - floor) * attenuation * slope_factor
            d2sigma = ((G0 - floor) * attenuation
                       * (slope_factor**2 - c*n*(n-1.0)*sigma**(n-2.0)))
        return {
            "barrier_eV": barrier, "raw_G0_eV": raw_G0, "executed_G0_eV": G0,
            "raw_sigc_Pa": raw_sigc, "executed_sigc_Pa": sigc,
            "floor_eV": floor, "floor_branch": floor_branch,
            "stress_gate_branch": stress_branch, "attenuation": attenuation,
            "dG0_dT_eV_K": self.gT_eV_per_K,
            "partial_G_partial_T_fixed_stress_state_eV_K": dT_fixed_stress,
            "partial_G_partial_T_fixed_sigma_over_sigc_eV_K": dT_fixed_normalized,
            "activation_entropy_fixed_stress_state_eV_K": (
                None if dT_fixed_stress is None else -dT_fixed_stress),
            "partial_G_partial_sigma_eV_Pa": dsigma,
            "partial2_G_partial_sigma2_eV_Pa2": d2sigma,
            "activation_volume_m3": None if dsigma is None else -dsigma * 1.602176634e-19,
            "derivative_convention": "partial derivative at fixed physical stress and frozen external state",
            "temperature_kink": bool(kink_G0 or kink_sigc or lower_kink or upper_kink),
        }


@dataclass(frozen=True)
class TransportBarrier:
    H0_eV: float
    activation_entropy_kB: float
    alpha: float
    exponent: float
    attempt_frequency_s: float
    stress_ratio: float = 1.0

    def as_surface(self, parent: ExpFloorBarrier) -> ExpFloorBarrier:
        emit0 = max(parent.G00_eV, 1.0e-12)
        return ExpFloorBarrier(
            G00_eV=self.H0_eV,
            gT_eV_per_K=-self.activation_entropy_kB * KB_EV_PER_K,
            sigc0_Pa=self.stress_ratio * parent.sigc0_Pa,
            sT_Pa_per_K=self.stress_ratio * parent.sT_Pa_per_K,
            alpha=self.alpha,
            exponent=self.exponent,
            floor_fraction=parent.floor_fraction,
            floor_min_eV=parent.floor_min_eV * self.H0_eV / emit0,
            floor_max_fraction=parent.floor_max_fraction,
            Tref_K=parent.Tref_K,
            attempt_frequency_s=self.attempt_frequency_s,
        )


@dataclass(frozen=True)
class MaterialManifest:
    name: str
    candidate_id: str
    cleavage: ExpFloorBarrier
    emission: ExpFloorBarrier
    peierls: TransportBarrier
    taylor: TransportBarrier
    taylor_corr_rho_c_m2: float
    taylor_corr_scale: float
    source_sites_per_system: float
    encounter_efficiency: float
    retained_recovery_rate_s: float
    source_refresh_length_m: float
    c_blunt: float
    max_K_shield_MPa_sqrt_m: float

    @classmethod
    def from_csv(cls, path: str | Path) -> "MaterialManifest":
        path = Path(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected exactly one manifest row in {path}; found {len(rows)}")
        row = rows[0]
        emission = ExpFloorBarrier(
            G00_eV=_f(row, "emit_G00_eV"),
            gT_eV_per_K=_f(row, "emit_gT_eV_per_K"),
            sigc0_Pa=_f(row, "emit_sigc0_GPa") * 1.0e9,
            sT_Pa_per_K=_f(row, "emit_sT_GPa_per_K") * 1.0e9,
            alpha=_f(row, "emit_exp_a"),
            exponent=_f(row, "emit_exp_n"),
            floor_fraction=_f(row, "emit_floor_frac"),
            attempt_frequency_s=1.0e11,
        )
        cleavage = ExpFloorBarrier(
            G00_eV=_f(row, "cleave_G00_eV"),
            gT_eV_per_K=_f(row, "cleave_gT_eV_per_K"),
            sigc0_Pa=_f(row, "cleave_sigc0_GPa") * 1.0e9,
            sT_Pa_per_K=_f(row, "cleave_sT_GPa_per_K") * 1.0e9,
            alpha=_f(row, "cleave_exp_a"),
            exponent=_f(row, "cleave_exp_n"),
            floor_fraction=_f(row, "cleave_floor_frac"),
            attempt_frequency_s=1.0e12,
        )
        return cls(
            name=str(row.get("target_class", path.parent.name)),
            candidate_id=str(row.get("candidate_id", "UNKNOWN")),
            cleavage=cleavage,
            emission=emission,
            peierls=TransportBarrier(
                H0_eV=_f(row, "peierls_H0_eV"),
                activation_entropy_kB=_f(row, "peierls_activation_entropy_kB"),
                alpha=_f(row, "peierls_exp_a"),
                exponent=_f(row, "peierls_exp_n"),
                attempt_frequency_s=_f(row, "peierls_nu0_s"),
            ),
            taylor=TransportBarrier(
                H0_eV=_f(row, "taylor_H0_eV"),
                activation_entropy_kB=_f(row, "taylor_activation_entropy_kB"),
                alpha=_f(row, "taylor_exp_a"),
                exponent=_f(row, "taylor_exp_n"),
                attempt_frequency_s=_f(row, "taylor_nu0_s"),
            ),
            taylor_corr_rho_c_m2=_f(row, "taylor_corr_rho_c_m2"),
            taylor_corr_scale=_f(row, "taylor_corr_scale"),
            source_sites_per_system=_f(row, "source_sites_per_system"),
            encounter_efficiency=_f(row, "encounter_efficiency"),
            retained_recovery_rate_s=_f(row, "retained_recovery_rate_s"),
            source_refresh_length_m=_f(row, "source_refresh_length_um") * 1.0e-6,
            c_blunt=_f(row, "c_blunt"),
            max_K_shield_MPa_sqrt_m=_f(row, "max_K_shield_MPa_sqrt_m"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidate_id": self.candidate_id,
            "cleavage": self.cleavage.__dict__,
            "emission": self.emission.__dict__,
            "peierls": self.peierls.__dict__,
            "taylor": self.taylor.__dict__,
            "taylor_corr_rho_c_m2": self.taylor_corr_rho_c_m2,
            "taylor_corr_scale": self.taylor_corr_scale,
            "source_sites_per_system": self.source_sites_per_system,
            "encounter_efficiency": self.encounter_efficiency,
            "retained_recovery_rate_s": self.retained_recovery_rate_s,
            "source_refresh_length_m": self.source_refresh_length_m,
            "c_blunt": self.c_blunt,
            "max_K_shield_MPa_sqrt_m": self.max_K_shield_MPa_sqrt_m,
        }


def default_manifest_path(material_class: str) -> Path:
    root = Path(__file__).resolve().parent / "data" / "materials"
    key = str(material_class).strip()
    aliases = {"weakt": "weakT", "weak_t": "weakT", "dbtt": "DBTT", "ceramic": "ceramic"}
    canonical = aliases.get(key.lower(), key)
    path = root / canonical / "spatial_promotion_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path
