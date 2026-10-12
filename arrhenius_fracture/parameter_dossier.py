"""Auditable parent-material domain dossier; no parameter fitting or promotion."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .material_manifest import ExpFloorBarrier, KB_EV_PER_K, TransportBarrier
from .prospective_campaign import REGISTRY


def _barrier(row: dict[str, str], prefix: str) -> ExpFloorBarrier:
    return ExpFloorBarrier(float(row[f"{prefix}_G00_eV"]), float(row[f"{prefix}_gT_eV_per_K"]),
                           float(row[f"{prefix}_sigc0_GPa"]) * 1e9,
                           float(row[f"{prefix}_sT_GPa_per_K"]) * 1e9,
                           float(row[f"{prefix}_exp_a"]), float(row[f"{prefix}_exp_n"]),
                           float(row[f"{prefix}_floor_frac"]),
                           attempt_frequency_s=1e12 if prefix == "cleave" else 1e11,
                           Tref_K=float(row["Tref_K"]))


def _mechanism(name: str, barrier: ExpFloorBarrier, T: float, stress_fraction: float) -> dict[str, Any]:
    raw_G0 = barrier.G00_eV + barrier.gT_eV_per_K * (T - barrier.Tref_K)
    raw_sigc = barrier.sigc0_Pa + barrier.sT_Pa_per_K * (T - barrier.Tref_K)
    if raw_G0 <= 0.0 or raw_sigc <= 0.0:
        return {"mechanism": name, "T_K": T, "raw_G0_eV": raw_G0,
                "raw_sigc_Pa": raw_sigc, "domain_status": "UNSUPPORTED_RAW_SCALE",
                "barrier_eV": None, "dG_dsigma_eV_Pa": None, "d2G_dsigma2_eV_Pa2": None,
                "dG_dT_eV_K": barrier.gT_eV_per_K, "rate_s": None, "timescale_s": None}
    sigma = stress_fraction * raw_sigc
    h = max(raw_sigc * 1e-5, 1.0)
    gm = float(barrier.values_eV(sigma - h, T)); g0 = float(barrier.values_eV(sigma, T)); gp = float(barrier.values_eV(sigma + h, T))
    rate = float(barrier.rate(sigma, T))
    floor = min(barrier.floor_max_fraction * max(raw_G0, 1e-12),
                max(barrier.floor_min_eV, barrier.floor_fraction * max(raw_G0, 1e-12)))
    return {"mechanism": name, "T_K": T, "stress_fraction": stress_fraction,
            "stress_Pa": sigma, "raw_G0_eV": raw_G0, "raw_sigc_Pa": raw_sigc,
            "barrier_eV": g0, "floor_eV": floor,
            "dG_dsigma_eV_Pa": (gp-gm)/(2*h),
            "d2G_dsigma2_eV_Pa2": (gp-2*g0+gm)/(h*h),
            "dG_dT_eV_K": barrier.gT_eV_per_K, "attempt_frequency_s": barrier.attempt_frequency_s,
            "rate_s": rate, "timescale_s": 1.0/rate if rate > 0 else None,
            "activation_support": "physical mechanism support; supplied by native spatial consumer",
            "completion_law": "integrated rate reaches dimensionless exponential mark",
            "time_unit": "s", "domain_status": "ADMISSIBLE_PARENT_EXAMPLE"}


def build_parameter_dossier(root: str | Path, temperatures=(300.0, 600.0, 800.0, 950.0, 1000.0, 1050.0, 1300.0),
                            stress_fractions=(0.0, 0.5, 1.0)) -> dict[str, Any]:
    path = Path(root) / REGISTRY
    parents = []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        mechanisms = []
        emit = _barrier(row, "emit")
        for T in temperatures:
            cleave = _barrier(row, "cleave")
            peierls = TransportBarrier(float(row["peierls_H0_eV"]), float(row["peierls_activation_entropy_kB"]),
                                       float(row["peierls_exp_a"]), float(row["peierls_exp_n"]),
                                       float(row["peierls_nu0_s"]), float(row["peierls_stress_fraction"])).as_surface(emit)
            taylor = TransportBarrier(float(row["taylor_H0_eV"]), float(row["taylor_activation_entropy_kB"]),
                                      float(row["taylor_exp_a"]), float(row["taylor_exp_n"]),
                                      float(row["taylor_nu0_s"]), float(row["taylor_stress_fraction"])).as_surface(emit)
            for f in stress_fractions:
                mechanisms.extend((_mechanism("cleavage", cleave, T, f),
                                   _mechanism("emission", emit, T, f),
                                   _mechanism("peierls_completion", peierls, T, f),
                                   _mechanism("taylor_release", taylor, T, f)))
        invalid = [m for m in mechanisms if m["domain_status"] != "ADMISSIBLE_PARENT_EXAMPLE"]
        parents.append({"family": row["material_class"], "option_key": row["option_key"],
                        "candidate_id": row["candidate_id"], "classification": "PRESERVED_PARENT_NOT_NEWLY_VALIDATED",
                        "physical_multiplicity": {"n_slip_channels": int(row["n_slip_channels"]),
                                                  "source_sites_per_system": float(row["source_sites_per_system"])},
                        "state_limits": {"rho_forest_floor_m2": float(row["rho_forest_floor_m2"]),
                                         "source_refresh_length_m": float(row["source_refresh_length_um"])*1e-6,
                                         "retained_recovery_rate_s": float(row["retained_recovery_rate_s"])},
                        "domain_status": "UNSUPPORTED_POINTS_PRESENT" if invalid else "GRID_ADMISSIBLE",
                        "mechanisms": mechanisms})
    return {"schema": "common-fracture-parameter-dossier/1",
            "selection_status": "parent examples only; no candidate promoted",
            "temperature_law_changes": "PHYSICAL_DECISION_REQUIRED",
            "parents": parents}


__all__ = ["build_parameter_dossier"]
