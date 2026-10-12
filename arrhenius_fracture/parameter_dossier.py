"""Auditable parent-material domain dossier; no parameter fitting or promotion."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .material_manifest import ExpFloorBarrier, TransportBarrier
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
    native = barrier.diagnostic(sigma, T)
    g0 = native["barrier_eV"]
    rate = float(barrier.rate(sigma, T))
    semantics = {
        "cleavage": {
            "elementary_rate_meaning": "local Arrhenius cleavage attempt rate",
            "effective_quantity": "aggregate physical-support hazard requires geometry/site support and state",
            "completion_law": "candidate hazard compared with its owned stochastic mark",
        },
        "emission": {
            "elementary_rate_meaning": "per persistent source/channel injection propensity",
            "effective_quantity": "simultaneous injection depends on active persistent source state and physical support",
            "completion_law": "coupled source-population update; not a generic one-mark mechanism",
        },
        "peierls_completion": {
            "elementary_rate_meaning": "signed mobile transport/mobility coefficient",
            "effective_quantity": "advance depends on mobile population, direction, slip distance and local state",
            "completion_law": "transported population/velocity update in physical seconds",
        },
        "taylor_release": {
            "elementary_rate_meaning": "single-hit retained-to-mobile release coefficient",
            "effective_quantity": "cooperative completion depends on retained population and Taylor correlation state",
            "completion_law": "native cooperative release mapping; not an independent exponential event mark",
        },
    }[name]
    return {"mechanism": name, "T_K": T, "stress_fraction": stress_fraction,
            "stress_Pa": sigma, "raw_G0_eV": raw_G0, "raw_sigc_Pa": raw_sigc,
            "barrier_eV": g0, "floor_eV": native["floor_eV"],
            "floor_branch": native["floor_branch"],
            "stress_gate_branch": native["stress_gate_branch"],
            "dG0_dT_eV_K": native["dG0_dT_eV_K"],
            "partial_G_partial_T_fixed_stress_state_eV_K": native["partial_G_partial_T_fixed_stress_state_eV_K"],
            "partial_G_partial_T_fixed_sigma_over_sigc_eV_K": native["partial_G_partial_T_fixed_sigma_over_sigc_eV_K"],
            "activation_entropy_fixed_stress_state_eV_K": native["activation_entropy_fixed_stress_state_eV_K"],
            "dG_dsigma_eV_Pa": native["partial_G_partial_sigma_eV_Pa"],
            "d2G_dsigma2_eV_Pa2": native["partial2_G_partial_sigma2_eV_Pa2"],
            "activation_volume_m3": native["activation_volume_m3"],
            "derivative_convention": native["derivative_convention"],
            "temperature_kink": native["temperature_kink"],
            "native_callable": "arrhenius_fracture.material_manifest.ExpFloorBarrier.values_eV/rate/diagnostic",
            "attempt_frequency_s": barrier.attempt_frequency_s,
            "rate_s": rate, "timescale_s": 1.0/rate if rate > 0 else None,
            "activation_support": "state/geometry dependent; no effective multiplicity fabricated in frozen scalar diagnostic",
            **semantics, "time_unit": "s", "domain_status": "ADMISSIBLE_PARENT_EXAMPLE"}


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
    return {"schema": "common-fracture-parameter-dossier/2",
            "supersedes_diagnostic": "common_framework_readiness/PARAMETER_PARENT_DOSSIER.json (historical v1 retained)",
            "selection_status": "parent examples only; no candidate promoted",
            "temperature_law_changes": "PHYSICAL_DECISION_REQUIRED",
            "parents": parents}


__all__ = ["build_parameter_dossier"]
