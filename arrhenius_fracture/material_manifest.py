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
