"""Exact manifest-to-bulk Peierls--Taylor coupling for v10.4.

The crack-tip MPZ and the surrounding FEM bulk remain distinct state
populations. They interact mechanically through the shared elastic-plastic
solve and directional J. No tip-emitted density is deposited into the bulk in
this first implementation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

KB_EV_PER_K = 8.617333262145e-5
MODEL_ID = "v10.4.0_bulk_peierls_taylor_manifest_coupling"

_REQUIRED = (
    "option_key",
    "candidate_id",
    "Tref_K",
    "rho_forest_floor_m2",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_stress_fraction",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_stress_fraction",
    "taylor_nu0_s",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
)


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v10.4 manifest field {key!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"v10.4 manifest field {key!r} must be finite")
    return value


def _positive(row: dict[str, str], key: str) -> float:
    value = _number(row, key)
    if value <= 0.0:
        raise ValueError(f"v10.4 manifest field {key!r} must be positive")
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return an audit-compatible ratio; direct exact fields remain authoritative."""
    if abs(denominator) <= 1.0e-30:
        return 0.0 if abs(numerator) <= 1.0e-30 else 1.0
    value = numerator / denominator
    return value if math.isfinite(value) else 1.0


@dataclass(frozen=True)
class BulkManifestParameters:
    option_key: str
    candidate_id: str
    Tref_K: float
    rho0_m2: float
    emit_G00_eV: float
    emit_gT_eV_per_K: float
    emit_sigc0_Pa: float
    emit_sT_Pa_per_K: float
    emit_exp_a: float
    emit_exp_n: float
    emit_floor_frac: float
    peierls_H0_eV: float
    peierls_entropy_kB: float
    peierls_exp_a: float
    peierls_exp_n: float
    peierls_stress_fraction: float
    peierls_nu0_s: float
    taylor_H0_eV: float
    taylor_entropy_kB: float
    taylor_exp_a: float
    taylor_exp_n: float
    taylor_stress_fraction: float
    taylor_nu0_s: float
    taylor_corr_rho_c_m2: float
    taylor_corr_scale: float
    source_path: str
    exact_row: dict[str, str]

    @classmethod
    def from_csv(cls, path: str | Path) -> "BulkManifestParameters":
        source = Path(path).expanduser().resolve()
        with source.open(newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if len(rows) != 1:
            raise ValueError(
                f"v10.4 requires exactly one selected manifest row; found {len(rows)}"
            )
        return cls.from_row(rows[0], source_path=str(source))

    @classmethod
    def from_row(
        cls,
        row: dict[str, str],
        *,
        source_path: str = "<memory>",
    ) -> "BulkManifestParameters":
        missing = [key for key in _REQUIRED if key not in row]
        if missing:
            raise ValueError(f"v10.4 selected manifest is missing fields: {missing}")
        return cls(
            option_key=str(row["option_key"]).strip(),
            candidate_id=str(row["candidate_id"]).strip(),
            Tref_K=_positive(row, "Tref_K"),
            rho0_m2=_positive(row, "rho_forest_floor_m2"),
            emit_G00_eV=_positive(row, "emit_G00_eV"),
            emit_gT_eV_per_K=_number(row, "emit_gT_eV_per_K"),
            emit_sigc0_Pa=_positive(row, "emit_sigc0_GPa") * 1.0e9,
            emit_sT_Pa_per_K=_number(row, "emit_sT_GPa_per_K") * 1.0e9,
            emit_exp_a=_positive(row, "emit_exp_a"),
            emit_exp_n=_positive(row, "emit_exp_n"),
            emit_floor_frac=_number(row, "emit_floor_frac"),
            peierls_H0_eV=_positive(row, "peierls_H0_eV"),
            peierls_entropy_kB=_number(row, "peierls_activation_entropy_kB"),
            peierls_exp_a=_positive(row, "peierls_exp_a"),
            peierls_exp_n=_positive(row, "peierls_exp_n"),
            peierls_stress_fraction=_positive(row, "peierls_stress_fraction"),
            peierls_nu0_s=_positive(row, "peierls_nu0_s"),
            taylor_H0_eV=_positive(row, "taylor_H0_eV"),
            taylor_entropy_kB=_number(row, "taylor_activation_entropy_kB"),
            taylor_exp_a=_positive(row, "taylor_exp_a"),
            taylor_exp_n=_positive(row, "taylor_exp_n"),
            taylor_stress_fraction=_positive(row, "taylor_stress_fraction"),
            taylor_nu0_s=_positive(row, "taylor_nu0_s"),
            taylor_corr_rho_c_m2=_positive(row, "taylor_corr_rho_c_m2"),
            taylor_corr_scale=_positive(row, "taylor_corr_scale"),
            source_path=source_path,
            exact_row=dict(row),
        )

    @property
    def peierls_gT_eV_per_K(self) -> float:
        return -self.peierls_entropy_kB * KB_EV_PER_K

    @property
    def taylor_gT_eV_per_K(self) -> float:
        return -self.taylor_entropy_kB * KB_EV_PER_K

    def configure(self, disl_cfg: Any) -> dict[str, Any]:
        """Map the exact selected row onto the existing production bulk model."""
        mapping = {
            "enable_plasticity": True,
            "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
            "plastic_update_mode": "explicit_rate",
            "thermo_consistency_mode": "time_cone",
            "bulk_mult_frac": 1.0,
            "tip_source_rho_per_emit": 0.0,
            "rho_transport_c": 0.0,
            "exhaustion_enabled": False,
            "freeze_rho": False,
            "use_static_recovery": False,
            "mobile_rho_floor": self.rho0_m2,
            "pt_emit_G00_eV": self.emit_G00_eV,
            "pt_emit_gT_eV_per_K": self.emit_gT_eV_per_K,
            "pt_emit_sigc0_Pa": self.emit_sigc0_Pa,
            "pt_emit_sT_Pa_per_K": self.emit_sT_Pa_per_K,
            "pt_emit_exp_a": self.emit_exp_a,
            "pt_emit_exp_n": self.emit_exp_n,
            "pt_emit_floor_frac": self.emit_floor_frac,
            "pt_emit_floor_min_eV": 1.0e-4,
            "pt_emit_floor_max_frac": 0.95,
            "pt_emit_Tref_K": self.Tref_K,
            "pt_peierls_energy_ratio": self.peierls_H0_eV / self.emit_G00_eV,
            "pt_peierls_entropy_ratio": _safe_ratio(
                self.peierls_gT_eV_per_K, self.emit_gT_eV_per_K
            ),
            "pt_peierls_stress_ratio": self.peierls_stress_fraction,
            "pt_peierls_nu0_s": self.peierls_nu0_s,
            "pt_peierls_G00_eV": self.peierls_H0_eV,
            "pt_peierls_gT_eV_per_K": self.peierls_gT_eV_per_K,
            "pt_peierls_exp_a": self.peierls_exp_a,
            "pt_peierls_exp_n": self.peierls_exp_n,
            "pt_peierls_Tref_K": self.Tref_K,
            "pt_taylor_energy_ratio": self.taylor_H0_eV / self.emit_G00_eV,
            "pt_taylor_entropy_ratio": _safe_ratio(
                self.taylor_gT_eV_per_K, self.emit_gT_eV_per_K
            ),
            "pt_taylor_stress_ratio": self.taylor_stress_fraction,
            "pt_taylor_nu0_s": self.taylor_nu0_s,
            "pt_taylor_G00_eV": self.taylor_H0_eV,
            "pt_taylor_gT_eV_per_K": self.taylor_gT_eV_per_K,
            "pt_taylor_exp_a": self.taylor_exp_a,
            "pt_taylor_exp_n": self.taylor_exp_n,
            "pt_taylor_Tref_K": self.Tref_K,
            "pt_taylor_corr_rho_c": self.taylor_corr_rho_c_m2,
            "pt_taylor_m_exponent": 1.0,
            "pt_taylor_m_scale": self.taylor_corr_scale,
            "pt_taylor_m_cap": float("inf"),
            "pt_mobile_density_floor_m2": self.rho0_m2,
        }
        for name, value in mapping.items():
            setattr(disl_cfg, name, value)
        return mapping

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ID,
            "selected_option": self.option_key,
            "selected_candidate": self.candidate_id,
            "selected_manifest": self.source_path,
            "reference_temperature_K": self.Tref_K,
            "initial_bulk_density_m2": self.rho0_m2,
            "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
            "bulk_source_population": "homogeneous_persistent_background",
            "tip_source_population": "persistent_site_moving_MPZ",
            "tip_and_bulk_populations_distinct": True,
            "direct_tip_to_bulk_density_transfer": False,
            "tip_source_rho_per_emit": 0.0,
            "bulk_density_transport": False,
            "finite_bulk_source_inventory": False,
            "bulk_multiplication_fraction": 1.0,
            "bulk_static_recovery": False,
            "thermodynamic_update": "local_time_cone",
            "mechanical_coupling": (
                "bulk_ep_and_rho_change_FEM_stress_and_directional_J; "
                "tip_MPZ_changes_shielding_blunting_and_hazards"
            ),
            "exact_registry_row": dict(self.exact_row),
        }


@dataclass
class BulkUpdateDiagnostics:
    calls: int = 0
    element_updates: int = 0
    max_dot_ep_s: float = 0.0
    max_dep_eq_accepted: float = 0.0
    max_rho_m2: float = 0.0
    min_rho_m2: float = float("inf")
    final_mean_rho_m2: float = 0.0
    max_pt_peierls_rate_s: float = 0.0
    max_pt_taylor_completion_rate_s: float = 0.0
    max_pt_series_rate_s: float = 0.0
    negative_accepted_work_count: int = 0
    limited_element_updates: int = 0
    thermo_modes: set[str] = field(default_factory=set)

    def observe(self, rho_out: Any, dot_ep: Any, info: dict[str, Any] | None) -> None:
        rho = np.asarray(rho_out, dtype=float)
        rate = np.asarray(dot_ep, dtype=float)
        self.calls += 1
        self.element_updates += int(rho.size)
        if rho.size:
            self.max_rho_m2 = max(self.max_rho_m2, float(np.nanmax(rho)))
            self.min_rho_m2 = min(self.min_rho_m2, float(np.nanmin(rho)))
            self.final_mean_rho_m2 = float(np.nanmean(rho))
        if rate.size:
            self.max_dot_ep_s = max(self.max_dot_ep_s, float(np.nanmax(rate)))
        if not isinstance(info, dict):
            return

        def _max(key: str) -> float:
            values = np.asarray(info.get(key, []), dtype=float)
            return float(np.nanmax(values)) if values.size else 0.0

        self.max_dep_eq_accepted = max(
            self.max_dep_eq_accepted, _max("dep_eq_accepted_gp")
        )
        self.max_pt_peierls_rate_s = max(
            self.max_pt_peierls_rate_s, _max("pt_peierls_rate_gp")
        )
        self.max_pt_taylor_completion_rate_s = max(
            self.max_pt_taylor_completion_rate_s,
            _max("pt_taylor_completion_rate_gp"),
        )
        self.max_pt_series_rate_s = max(
            self.max_pt_series_rate_s, _max("pt_series_rate_gp")
        )
        accepted_work = np.asarray(info.get("dWp_accepted_gp", []), dtype=float)
        if accepted_work.size:
            self.negative_accepted_work_count += int(
                np.count_nonzero(accepted_work < -1.0e-12)
            )
        limited = np.asarray(info.get("dep_eq_limited_gp", []), dtype=float)
        if limited.size:
            self.limited_element_updates += int(np.count_nonzero(limited > 0.5))
        mode = info.get("thermo_mode")
        if mode is not None:
            self.thermo_modes.add(str(mode))

    def payload(self) -> dict[str, Any]:
        minimum = self.min_rho_m2 if math.isfinite(self.min_rho_m2) else 0.0
        return {
            "plasticity_update_calls": self.calls,
            "element_updates": self.element_updates,
            "maximum_equivalent_plastic_rate_s": self.max_dot_ep_s,
            "maximum_accepted_equivalent_strain_increment": self.max_dep_eq_accepted,
            "minimum_bulk_density_m2": minimum,
            "maximum_bulk_density_m2": self.max_rho_m2,
            "final_mean_bulk_density_m2": self.final_mean_rho_m2,
            "maximum_peierls_rate_s": self.max_pt_peierls_rate_s,
            "maximum_taylor_completion_rate_s": self.max_pt_taylor_completion_rate_s,
            "maximum_series_rate_s": self.max_pt_series_rate_s,
            "negative_accepted_plastic_work_count": self.negative_accepted_work_count,
            "plastic_increment_limited_element_updates": self.limited_element_updates,
            "thermodynamic_modes": sorted(self.thermo_modes),
            "local_plastic_work_nonnegative": self.negative_accepted_work_count == 0,
        }


class BulkPlasticityCoupling:
    def __init__(self, parameters: BulkManifestParameters):
        self.parameters = parameters
        self.diagnostics = BulkUpdateDiagnostics()
        self.last_mapping: dict[str, Any] = {}

    def wrap(self, original: Callable) -> Callable:
        def coupled_update(*args, **kwargs):
            # update_plasticity(ep, rho, sigma, mat, T, dt, plast_model,
            #                   disl_cfg, return_info=False)
            if len(args) >= 8:
                disl_cfg = args[7]
            elif "disl_cfg" in kwargs:
                disl_cfg = kwargs["disl_cfg"]
            else:
                raise TypeError("v10.4 could not locate disl_cfg in plasticity update")
            self.last_mapping = self.parameters.configure(disl_cfg)
            result = original(*args, **kwargs)
            wants_info = bool(kwargs.get("return_info", False))
            if len(args) >= 9:
                wants_info = bool(args[8])
            if wants_info:
                _, rho_out, dot_ep, info = result
            else:
                _, rho_out, dot_ep = result
                info = None
            self.diagnostics.observe(rho_out, dot_ep, info)
            return result

        coupled_update.__name__ = "v104_bulk_manifest_coupled_update"
        coupled_update.__doc__ = (
            "Exact selected-row bulk Peierls--Taylor update with local "
            "thermodynamic projection."
        )
        return coupled_update

    def write_audit(self, output_root: str | Path) -> Path:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        payload = self.parameters.audit_payload()
        payload["resolved_dislocation_config"] = dict(self.last_mapping)
        payload["runtime_diagnostics"] = self.diagnostics.payload()
        path = root / "v10_4_bulk_peierls_taylor_coupling_audit.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


__all__ = [
    "BulkManifestParameters",
    "BulkPlasticityCoupling",
    "BulkUpdateDiagnostics",
    "KB_EV_PER_K",
    "MODEL_ID",
]
