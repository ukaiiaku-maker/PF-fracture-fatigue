"""v10.2.3 shared-state reduced model consistent with the v10.2.2 2-D front.

This module is deliberately not a new phenomenological zero-dimensional model.
It executes the same production material manifest, spatial ``UnifiedMPZState``,
finite campaign source budget, anisotropic emission law, validated transport
operator, Taylor back stress, accumulated-slip blunting, recovery, escape, and
moving-frame translation used by the v10.2.2 2-D solver.

Only the mechanical closure is reduced:

* ``replay`` consumes an externally recorded 2-D schedule and is intended for
  exact constitutive-state equivalence tests;
* ``monotonic`` prescribes a scalar K ramp and supplied channel drive factors.
  This is a cheap calibration surrogate, but its mechanics approximation is
  explicit and audited rather than hidden in a separate constitutive model.

The legacy manifest shielding cap is never used in kinetics. Effective shielding
is always the raw signed elastic field, matching v10.2.2.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from .anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    install_anisotropic_campaign_emission,
)
from .campaign_calibrated_tip import (
    CampaignCalibratedTipEngine,
    install_campaign_calibrated_source,
)
from .kinetic_tip_cell import KineticTipConfig
from .material_manifest import MaterialManifest
from .physical_shielding_v1022 import install_uncapped_physical_shielding
from .unified_mpz import MPZConfig


MODEL_ID = "v10.2.3_shared_reduced_state_equivalence"
VALIDATED_SCALAR_TRANSPORT = "validated_scalar"
CHANNEL_RESOLVED_TRANSPORT = "channel_resolved"
VALID_MODES = {
    "full",
    "plasticity_off",
    "shielding_off",
    "backstress_off",
}

# Mean tensor-derived factors observed in the theta=45 degree deterministic
# transfer scouts. They are a mechanical closure input, not a fitted material
# parameter. Replay mode should supply the actual 2-D factor history instead.
DEFAULT_THETA45_DRIVE_FACTORS = (0.132886, 0.008596)


@dataclass
class SharedReducedConfig:
    Kdot_MPa_sqrt_m_s: float = 0.005
    Kmax_MPa_sqrt_m: float = 80.0
    max_dK_step_MPa_sqrt_m: float = 0.01
    target_extension_um: float = 5.0
    checkpoint_da_um: float = 5.0
    r0_m: float = 1.0e-6
    mpz_length_um: float = 100.0
    mpz_n_bins: int = 200
    wake_length_um: float = 100.0
    wake_n_bins: int = 0
    source_bin_count: int = 2
    blunting_length_um: float = 0.5
    shielding_core_m: float = 2.5e-10
    forest_density_floor_m2: float = 5.0e12
    mobile_shield_fraction: float = 1.0
    G_Pa: float = 160.0e9
    poisson: float = 0.28
    b_m: float = 2.74e-10
    cleavage_hits: float = 3.0
    cleavage_tau_s: float = 1.0e-6
    drive_factors: tuple[float, float] = DEFAULT_THETA45_DRIVE_FACTORS
    crystal_theta_deg: float = 45.0
    transport_mode: str = VALIDATED_SCALAR_TRANSPORT
    max_action_substep: float = 0.01
    max_translation_substep_m: float = 5.0e-8
    max_internal_steps: int = 20000

    def validate(self) -> "SharedReducedConfig":
        if self.Kdot_MPa_sqrt_m_s <= 0.0:
            raise ValueError("Kdot must be positive")
        if self.Kmax_MPa_sqrt_m <= 0.0:
            raise ValueError("Kmax must be positive")
        if self.max_dK_step_MPa_sqrt_m <= 0.0:
            raise ValueError("max dK step must be positive")
        if self.target_extension_um <= 0.0 or self.checkpoint_da_um <= 0.0:
            raise ValueError("extension and checkpoint length must be positive")
        if self.r0_m <= 0.0 or self.mpz_length_um <= 0.0:
            raise ValueError("r0 and MPZ length must be positive")
        if self.mpz_n_bins < 4:
            raise ValueError("mpz_n_bins must be at least four")
        if self.source_bin_count < 1:
            raise ValueError("source_bin_count must be positive")
        if self.G_Pa <= 0.0 or self.b_m <= 0.0:
            raise ValueError("G and b must be positive")
        factors = tuple(float(value) for value in self.drive_factors)
        if len(factors) != 2 or any(value < 0.0 or not math.isfinite(value) for value in factors):
            raise ValueError("exactly two finite nonnegative drive factors are required")
        self.drive_factors = factors
        mode = str(self.transport_mode).strip().lower().replace("-", "_")
        aliases = {
            "scalar": VALIDATED_SCALAR_TRANSPORT,
            "validated": VALIDATED_SCALAR_TRANSPORT,
            "channel": CHANNEL_RESOLVED_TRANSPORT,
            "resolved": CHANNEL_RESOLVED_TRANSPORT,
        }
        mode = aliases.get(mode, mode)
        if mode not in {VALIDATED_SCALAR_TRANSPORT, CHANNEL_RESOLVED_TRANSPORT}:
            raise ValueError(f"invalid transport mode: {self.transport_mode}")
        self.transport_mode = mode
        return self


FALLBACK_ROLES = {
    "DBTT_A0002333": "large_DBTT_rise",
    "DBTT_A0003837": "strong_shielding_sensitivity",
    "DBTT_A0002277": "non_cap_limited_shielding_state",
}


def fallback_manifest_path(candidate_id: str) -> Path:
    candidate = str(candidate_id).strip()
    if candidate not in FALLBACK_ROLES:
        raise KeyError(f"unknown fallback candidate {candidate!r}")
    path = (
        Path(__file__).resolve().parent
        / "data"
        / "materials"
        / "fallback_dbtt"
        / f"{candidate}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _front_config(cfg: SharedReducedConfig) -> SimpleNamespace:
    return SimpleNamespace(
        r0=float(cfg.r0_m),
        L_pz=float(cfg.mpz_length_um) * 1.0e-6,
        da=float(cfg.checkpoint_da_um) * 1.0e-6,
        sigma_cap=0.0,
        m_hits=float(cfg.cleavage_hits),
        tau_c=float(cfg.cleavage_tau_s),
    )


def _mpz_config(cfg: SharedReducedConfig) -> MPZConfig:
    return MPZConfig(
        length_m=float(cfg.mpz_length_um) * 1.0e-6,
        n_bins=int(cfg.mpz_n_bins),
        n_systems=2,
        source_bin_count=int(cfg.source_bin_count),
        shielding_orientation_factors=(1.0, 1.0),
        mobile_shield_fraction=float(cfg.mobile_shield_fraction),
        shielding_core_m=float(cfg.shielding_core_m),
        blunting_length_m=float(cfg.blunting_length_um) * 1.0e-6,
        forest_density_floor_m2=float(cfg.forest_density_floor_m2),
        wake_length_m=float(cfg.wake_length_um) * 1.0e-6,
        wake_n_bins=int(cfg.wake_n_bins),
        wake_shielding=False,
    )


def _tip_config(cfg: SharedReducedConfig, mode: str) -> KineticTipConfig:
    return KineticTipConfig(
        enabled=True,
        plasticity_enabled=mode != "plasticity_off",
        active_shielding=mode not in {"plasticity_off", "shielding_off"},
        signed_active_shielding=True,
        mobile_shield_fraction=float(cfg.mobile_shield_fraction),
        packet_length_m=float(cfg.b_m),
        velocity_scale=1.0,
        max_action_substep=float(cfg.max_action_substep),
        max_translation_substep_m=float(cfg.max_translation_substep_m),
        min_substep_s=1.0e-15,
        max_internal_steps=int(cfg.max_internal_steps),
        coupling_scheme="strang",
    ).validate()


def build_shared_engine(
    manifest: MaterialManifest,
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
    drive_factors: Iterable[float] | None = None,
) -> CampaignCalibratedTipEngine:
    """Construct the production moving-MPZ engine without a FEM object."""
    cfg = cfg.validate()
    mode = str(mode).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"invalid reduced mode {mode!r}; choose {sorted(VALID_MODES)}")

    engine = CampaignCalibratedTipEngine(
        _front_config(cfg),
        manifest.cleavage,
        manifest.emission,
        float(cfg.G_Pa),
        float(cfg.poisson),
        float(cfg.b_m),
        manifest,
        _mpz_config(cfg),
    )
    engine.tip_cfg = _tip_config(cfg, mode)
    engine.mpz.cfg.mobile_shield_fraction = float(cfg.mobile_shield_fraction)

    # Reinstall explicitly so the mode-specific backstress scale is auditable and
    # independent of any process-global class default inherited from a prior run.
    install_campaign_calibrated_source(
        engine.mpz,
        float(cfg.b_m),
        float(cfg.G_Pa),
        backstress_scale=0.0 if mode == "backstress_off" else 1.0,
        refresh_scale=1.0,
    )

    inherited_evolve = engine.mpz.evolve
    anisotropic_cfg = AnisotropicEmissionConfig(
        enabled=True,
        crystal_theta_deg=float(cfg.crystal_theta_deg),
        shared_forest_density=True,
        require_reliable_probe=True,
    )
    install_anisotropic_campaign_emission(engine.mpz, anisotropic_cfg)
    if cfg.transport_mode == VALIDATED_SCALAR_TRANSPORT:
        # Exactly mirror v10.2.2's promoted transport mode: anisotropic emission
        # remains installed, while the inherited common transport operator is used.
        engine.mpz.evolve = inherited_evolve

    factors = np.asarray(
        tuple(cfg.drive_factors) if drive_factors is None else tuple(drive_factors),
        dtype=float,
    ).reshape(-1)
    if factors.size != engine.mpz.n_systems:
        raise ValueError(
            f"drive factor count {factors.size} does not match n_systems={engine.mpz.n_systems}"
        )
    if np.any(~np.isfinite(factors)) or np.any(factors < 0.0):
        raise ValueError("drive factors must be finite and nonnegative")
    engine.mpz._anisotropic_drive_factors = factors.copy()
    engine.mpz._anisotropic_drive_reliable = True
    engine.mpz._anisotropic_drive_serial = 0
    engine._shared_reduced_mode = mode
    engine._shared_reduced_mechanics_closure = "prescribed_K_and_channel_drive_factors"
    engine._shared_reduced_transport_mode = cfg.transport_mode
    return engine


def _field_snapshot(engine: CampaignCalibratedTipEngine) -> dict[str, Any]:
    state = engine.mpz
    active_raw = float(engine._active_shielding_raw_uncapped())
    active_effective = float(engine._active_shielding_signed())
    factors = np.asarray(state._anisotropic_drive_factors, dtype=float)
    sigma_back = np.asarray(
        getattr(state, "anisotropic_last_sigma_back_by_system_Pa", np.zeros(state.n_systems)),
        dtype=float,
    )
    sigma_emit = np.asarray(
        getattr(state, "anisotropic_last_sigma_emit_by_system_Pa", np.zeros(state.n_systems)),
        dtype=float,
    )
    return {
        "mobile_count": float(state.mobile_count),
        "retained_count": float(state.retained_count),
        "emitted_total": float(state.emitted_total),
        "escaped_total": float(state.escaped_total),
        "recovered_total": float(state.recovered_total),
        "available_sites_total": float(np.sum(state.available_sites)),
        "source_capacity_total": float(np.sum(state.site_capacity)),
        "local_slip_count": float(state.local_slip_count()),
        "r_eff_m": float(engine.r_eff()),
        "K_shield_raw_Pa_sqrt_m": active_raw,
        "K_shield_effective_Pa_sqrt_m": active_effective,
        "raw_minus_effective_Pa_sqrt_m": active_raw - active_effective,
        "drive_factor_0": float(factors[0]),
        "drive_factor_1": float(factors[1]),
        "sigma_back_0_Pa": float(sigma_back[0]) if sigma_back.size else 0.0,
        "sigma_back_1_Pa": float(sigma_back[1]) if sigma_back.size > 1 else 0.0,
        "sigma_emit_0_Pa": float(sigma_emit[0]) if sigma_emit.size else 0.0,
        "sigma_emit_1_Pa": float(sigma_emit[1]) if sigma_emit.size > 1 else 0.0,
    }


def run_monotonic_shared_front(
    manifest: MaterialManifest,
    temperature_K: float,
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Run a cheap prescribed-K first-passage calculation with shared state code."""
    cfg = cfg.validate()
    engine = build_shared_engine(manifest, cfg, mode=mode)
    target_advances = max(
        int(math.ceil(float(cfg.target_extension_um) / float(cfg.checkpoint_da_um))),
        1,
    )
    K_left = 0.0
    history: list[dict[str, Any]] = []
    internal_steps = 0

    with install_uncapped_physical_shielding():
        while K_left < cfg.Kmax_MPa_sqrt_m and engine.n_adv < target_advances:
            internal_steps += 1
            if internal_steps > 2_000_000:
                raise RuntimeError("shared reduced front exceeded outer-step limit")
            dK = min(
                float(cfg.max_dK_step_MPa_sqrt_m),
                float(cfg.Kmax_MPa_sqrt_m) - K_left,
            )
            dt = dK / float(cfg.Kdot_MPa_sqrt_m_s)
            K_mid = K_left + 0.5 * dK
            result = engine.step(K_mid * 1.0e6, float(temperature_K), dt)
            consumed = float(result.get("kinetic_dt_consumed_s", dt))
            consumed = min(max(consumed, 0.0), dt)
            K_event = K_left + float(cfg.Kdot_MPa_sqrt_m_s) * consumed
            if not bool(result.get("fired", False)):
                K_event = K_left + dK
            snapshot = {
                "outer_step": internal_steps,
                "time_s": float(engine.t),
                "K_MPa_sqrt_m": float(K_event),
                "temperature_K": float(temperature_K),
                "mode": mode,
                "fired": bool(result.get("fired", False)),
                "checkpoint_progress_action": float(engine.B),
                "checkpoint_advances": int(engine.n_adv),
                "micro_advance_total_m": float(engine.micro_advance_total_m),
                "sigma_opening_Pa": float(engine.sigma_opening_tip(K_mid * 1.0e6)),
                "sigma_cleave_Pa": float(
                    result.get("sigma_cleave_eff_Pa", result.get("sigma_tip", 0.0))
                ),
                "lambda_cleave_s": float(result.get("lambda_c", 0.0)),
                "dN_emit": float(result.get("dN_emit", result.get("dN_emit_raw", 0.0))),
                "dN_trapped": float(result.get("dN_trapped", 0.0)),
                "dN_recovered": float(result.get("dN_recovered", 0.0)),
                "dN_escaped": float(result.get("dN_escaped", 0.0)),
                **_field_snapshot(engine),
            }
            history.append(snapshot)
            K_left = float(K_event)
            if bool(result.get("fired", False)):
                # The engine localizes the event and may leave part of the outer
                # increment unused. Continue only when more checkpoints are needed.
                if engine.n_adv >= target_advances:
                    break
                if consumed <= 0.0:
                    K_left += min(dK, 1.0e-12)

    final = history[-1] if history else {}
    reached = engine.n_adv >= target_advances
    result = {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "material_class": manifest.name,
        "fallback_role": FALLBACK_ROLES.get(manifest.candidate_id),
        "mode": mode,
        "temperature_K": float(temperature_K),
        "status": "complete" if reached else "incomplete",
        "K_first_MPa_sqrt_m": (
            next(
                (float(row["K_MPa_sqrt_m"]) for row in history if row["fired"]),
                None,
            )
        ),
        "checkpoint_advances": int(engine.n_adv),
        "target_checkpoint_advances": int(target_advances),
        "micro_advance_total_m": float(engine.micro_advance_total_m),
        "outer_steps": int(internal_steps),
        "history": history,
        "final_state": final,
        "config": asdict(cfg),
        "mechanics_closure": "prescribed scalar K ramp plus supplied channel drive factors",
        "state_evolution_source": "production v10.2.2 moving MPZ classes",
        "constitutive_K_shield_clip_applied": False,
        "legacy_manifest_cap_reference_MPa_sqrt_m": float(
            manifest.max_K_shield_MPa_sqrt_m
        ),
        "legacy_manifest_cap_used_in_kinetics": False,
        "raw_equals_effective": bool(
            all(
                abs(float(row["raw_minus_effective_Pa_sqrt_m"]))
                <= max(1.0e-6, 1.0e-12 * max(abs(float(row["K_shield_raw_Pa_sqrt_m"])), 1.0))
                for row in history
            )
        ),
    }
    return result


def _read_replay_schedule(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty replay schedule: {path}")
    required = {"dt_s", "temperature_K", "K_Pa_sqrt_m", "advance_m"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"replay schedule is missing columns: {sorted(missing)}")
    parsed: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        item = {key: float(row[key]) for key in required}
        item["drive_factor_0"] = float(row.get("drive_factor_0", DEFAULT_THETA45_DRIVE_FACTORS[0]))
        item["drive_factor_1"] = float(row.get("drive_factor_1", DEFAULT_THETA45_DRIVE_FACTORS[1]))
        item["row_index"] = float(index)
        parsed.append(item)
    return parsed


def replay_shared_state(
    manifest: MaterialManifest,
    schedule: str | Path | list[dict[str, float]],
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Replay a recorded 2-D K/drive/advance schedule through shared state code."""
    cfg = cfg.validate()
    rows = _read_replay_schedule(schedule) if isinstance(schedule, (str, Path)) else list(schedule)
    engine = build_shared_engine(manifest, cfg, mode=mode)
    history: list[dict[str, Any]] = []

    with install_uncapped_physical_shielding():
        for index, row in enumerate(rows):
            dt = max(float(row["dt_s"]), 0.0)
            T = float(row["temperature_K"])
            K = max(float(row["K_Pa_sqrt_m"]), 0.0)
            advance = max(float(row.get("advance_m", 0.0)), 0.0)
            factors = np.asarray(
                [row.get("drive_factor_0", cfg.drive_factors[0]), row.get("drive_factor_1", cfg.drive_factors[1])],
                dtype=float,
            )
            if np.any(factors < 0.0) or np.any(~np.isfinite(factors)):
                raise ValueError(f"invalid drive factors at replay row {index}: {factors}")
            engine.mpz._anisotropic_drive_factors = factors.copy()
            engine._separated_current_K_Pa_sqrt_m = K
            opening_stress = engine.sigma_opening_tip(K)
            plastic = (
                engine.mpz.evolve(dt, T, opening_stress, cfg.b_m)
                if mode != "plasticity_off" and dt > 0.0
                else {}
            )
            moved = engine.mpz.advance(advance) if advance > 0.0 else {}
            history.append(
                {
                    "row_index": index,
                    "dt_s": dt,
                    "temperature_K": T,
                    "K_Pa_sqrt_m": K,
                    "advance_m": advance,
                    "sigma_opening_Pa": float(opening_stress),
                    "dN_emit": float(plastic.get("dN_emit", 0.0)),
                    "dN_trapped": float(plastic.get("dN_trapped", 0.0)),
                    "dN_recovered": float(plastic.get("dN_recovered", 0.0)),
                    "dN_escaped": float(plastic.get("dN_escaped", 0.0)),
                    "source_sites_refreshed": float(moved.get("source_sites_refreshed", 0.0)),
                    **_field_snapshot(engine),
                }
            )

    return {
        "schema": MODEL_ID,
        "replay_mode": True,
        "candidate_id": manifest.candidate_id,
        "mode": mode,
        "n_schedule_rows": len(rows),
        "history": history,
        "config": asdict(cfg),
        "state_evolution_source": "production v10.2.2 moving MPZ classes",
        "constitutive_K_shield_clip_applied": False,
        "legacy_manifest_cap_used_in_kinetics": False,
        "raw_equals_effective": bool(
            all(
                abs(float(row["raw_minus_effective_Pa_sqrt_m"]))
                <= max(1.0e-6, 1.0e-12 * max(abs(float(row["K_shield_raw_Pa_sqrt_m"])), 1.0))
                for row in history
            )
        ),
    }


def write_shared_result(result: dict[str, Any], out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    history = list(result.get("history", []))
    payload = dict(result)
    payload.pop("history", None)
    (root / "shared_reduced_summary.json").write_text(json.dumps(payload, indent=2))
    (root / "shared_reduced_audit.json").write_text(
        json.dumps(
            {
                "schema": MODEL_ID,
                "candidate_id": result.get("candidate_id"),
                "mode": result.get("mode"),
                "state_evolution_source": result.get("state_evolution_source"),
                "mechanics_closure": result.get("mechanics_closure", "2-D replay schedule"),
                "constitutive_K_shield_clip_applied": False,
                "legacy_manifest_cap_used_in_kinetics": False,
                "raw_equals_effective": bool(result.get("raw_equals_effective", False)),
                "fallback_role": result.get("fallback_role"),
                "n_history_rows": len(history),
            },
            indent=2,
        )
    )
    if history:
        fields = list(history[0])
        with (root / "shared_reduced_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


def fallback_registry() -> dict[str, Any]:
    return {
        "schema": "v10.2.3_fallback_DBTT_parameterizations",
        "parameterizations": [
            {
                "candidate_id": candidate,
                "role": role,
                "manifest": str(fallback_manifest_path(candidate)),
                "status": "preserved_for_fallback_and_equivalence_tests",
            }
            for candidate, role in FALLBACK_ROLES.items()
        ],
    }


__all__ = [
    "CHANNEL_RESOLVED_TRANSPORT",
    "DEFAULT_THETA45_DRIVE_FACTORS",
    "FALLBACK_ROLES",
    "MODEL_ID",
    "SharedReducedConfig",
    "VALIDATED_SCALAR_TRANSPORT",
    "build_shared_engine",
    "fallback_manifest_path",
    "fallback_registry",
    "replay_shared_state",
    "run_monotonic_shared_front",
    "write_shared_result",
]
