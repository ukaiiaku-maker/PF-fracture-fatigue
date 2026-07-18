"""v10.2.3 shared-state reduced model consistent with the v10.2.2 2-D front.

This is not a second phenomenological one-dimensional constitutive model.  It
executes the production ``MaterialManifest``, spatial ``UnifiedMPZState``, finite
campaign source budget, anisotropic emission law, promoted transport operator,
Taylor back stress, accumulated-slip blunting, recovery, escape, moving-frame
translation, and uncapped signed shielding used by v10.2.2.

Only the mechanical closure is reduced:

* ``monotonic`` prescribes a scalar K ramp and externally supplied channel drive
  factors;
* ``replay`` consumes a recorded 2-D sequence of K, temperature, timestep, and
  channel factors and calls the same production coupled ``engine.step`` method.

The legacy manifest shielding cap is retained only as provenance.  It is never
used in the kinetics or cleavage stress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
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

# Mean factors observed in the theta=45-degree deterministic 2-D transfer scouts.
# They are part of the reduced mechanical closure, not fitted material parameters.
# Exact replay supplies the actual factor history from the corresponding 2-D run.
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
    max_outer_steps: int = 2_000_000

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
        if self.max_internal_steps < 1 or self.max_outer_steps < 1:
            raise ValueError("internal and outer step limits must be positive")
        factors = tuple(float(value) for value in self.drive_factors)
        if len(factors) != 2 or any(
            value < 0.0 or not math.isfinite(value) for value in factors
        ):
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


def load_manifest(
    *,
    candidate_id: str | None = None,
    manifest_path: str | Path | None = None,
) -> MaterialManifest:
    if bool(candidate_id) == bool(manifest_path):
        raise ValueError("provide exactly one of candidate_id or manifest_path")
    path = fallback_manifest_path(str(candidate_id)) if candidate_id else Path(manifest_path)
    return MaterialManifest.from_csv(path)


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


def _zero_shielding(self) -> float:
    return 0.0


def build_shared_engine(
    manifest: MaterialManifest,
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
    drive_factors: Iterable[float] | None = None,
) -> CampaignCalibratedTipEngine:
    """Construct the production moving-MPZ engine without constructing a FEM."""
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

    # Reinstall explicitly so the mode-specific backstress scale is independent
    # of process-global class defaults from any preceding calculation.
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
        # Mirror v10.2.2's promoted path: retain the anisotropic finite-source
        # emitter but use the inherited validated common transport operator.
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

    # CampaignCalibratedTipEngine historically overrode the base method without
    # consulting tip_cfg.active_shielding.  Bind the diagnostic ablation directly
    # to this instance so shielding_off is an actual mechanics ablation, not a
    # parser label.  The raw field is still recorded for comparison.
    if mode in {"plasticity_off", "shielding_off"}:
        engine._active_shielding_signed = MethodType(_zero_shielding, engine)
        engine._wake_shielding_signed = MethodType(_zero_shielding, engine)

    engine._shared_reduced_mode = mode
    engine._shared_reduced_mechanics_closure = (
        "prescribed_K_and_channel_drive_factors"
    )
    engine._shared_reduced_transport_mode = cfg.transport_mode
    return engine


def _state_arrays(engine: CampaignCalibratedTipEngine) -> dict[str, np.ndarray]:
    state = engine.mpz
    return {
        "mobile": np.asarray(state.mobile, dtype=float).copy(),
        "retained": np.asarray(state.retained, dtype=float).copy(),
        "accumulated_slip": np.asarray(state.accumulated_slip, dtype=float).copy(),
        "available_sites": np.asarray(state.available_sites, dtype=float).copy(),
        "site_capacity": np.asarray(state.site_capacity, dtype=float).copy(),
        "wake_mobile": np.asarray(state.wake_mobile, dtype=float).copy(),
        "wake_retained": np.asarray(state.wake_retained, dtype=float).copy(),
        "wake_slip": np.asarray(state.wake_slip, dtype=float).copy(),
    }


def _field_snapshot(engine: CampaignCalibratedTipEngine) -> dict[str, Any]:
    state = engine.mpz
    active_raw = float(engine._active_shielding_raw_uncapped())
    active_effective = float(engine._active_shielding_signed())
    factors = np.asarray(state._anisotropic_drive_factors, dtype=float)
    sigma_back = np.asarray(
        getattr(
            state,
            "anisotropic_last_sigma_back_by_system_Pa",
            np.zeros(state.n_systems),
        ),
        dtype=float,
    )
    sigma_emit = np.asarray(
        getattr(
            state,
            "anisotropic_last_sigma_emit_by_system_Pa",
            np.zeros(state.n_systems),
        ),
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


def _raw_effective_consistent(history: list[dict[str, Any]], mode: str) -> bool:
    if mode == "shielding_off":
        return True
    return bool(
        all(
            abs(float(row["raw_minus_effective_Pa_sqrt_m"]))
            <= max(
                1.0e-6,
                1.0e-12
                * max(abs(float(row["K_shield_raw_Pa_sqrt_m"])), 1.0),
            )
            for row in history
        )
    )


def run_monotonic_shared_front(
    manifest: MaterialManifest,
    temperature_K: float,
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Run prescribed-K first passage while using the production state engine."""
    cfg = cfg.validate()
    engine = build_shared_engine(manifest, cfg, mode=mode)
    target_advances = max(
        int(math.ceil(float(cfg.target_extension_um) / float(cfg.checkpoint_da_um))),
        1,
    )
    K_left = 0.0
    history: list[dict[str, Any]] = []
    outer_step = 0

    with install_uncapped_physical_shielding():
        while K_left < cfg.Kmax_MPa_sqrt_m and engine.n_adv < target_advances:
            outer_step += 1
            if outer_step > int(cfg.max_outer_steps):
                raise RuntimeError("shared reduced front exceeded outer-step limit")
            dK = min(
                float(cfg.max_dK_step_MPa_sqrt_m),
                float(cfg.Kmax_MPa_sqrt_m) - K_left,
            )
            dt_requested = dK / float(cfg.Kdot_MPa_sqrt_m_s)
            K_mid = K_left + 0.5 * dK
            result = engine.step(K_mid * 1.0e6, float(temperature_K), dt_requested)
            consumed = min(
                max(float(result.get("kinetic_dt_consumed_s", dt_requested)), 0.0),
                dt_requested,
            )
            fired = bool(result.get("fired", False))
            K_right = (
                K_left + float(cfg.Kdot_MPa_sqrt_m_s) * consumed
                if fired
                else K_left + dK
            )
            history.append(
                {
                    "outer_step": outer_step,
                    "time_s": float(engine.t),
                    "dt_requested_s": float(dt_requested),
                    "dt_consumed_s": float(consumed),
                    "K_input_MPa_sqrt_m": float(K_mid),
                    "K_MPa_sqrt_m": float(K_right),
                    "temperature_K": float(temperature_K),
                    "mode": mode,
                    "fired": fired,
                    "checkpoint_progress_action": float(engine.B),
                    "checkpoint_advances": int(engine.n_adv),
                    "micro_advance_step_m": float(
                        result.get("kinetic_micro_advance_step_m", 0.0)
                    ),
                    "micro_advance_total_m": float(engine.micro_advance_total_m),
                    "sigma_opening_Pa": float(
                        engine.sigma_opening_tip(K_mid * 1.0e6)
                    ),
                    "sigma_cleave_Pa": float(
                        result.get(
                            "sigma_cleave_eff_Pa",
                            result.get("sigma_tip", 0.0),
                        )
                    ),
                    "lambda_cleave_s": float(result.get("lambda_c", 0.0)),
                    "dN_emit": float(
                        result.get("dN_emit", result.get("dN_emit_raw", 0.0))
                    ),
                    "dN_trapped": float(result.get("dN_trapped", 0.0)),
                    "dN_recovered": float(result.get("dN_recovered", 0.0)),
                    "dN_escaped": float(result.get("dN_escaped", 0.0)),
                    **_field_snapshot(engine),
                }
            )
            K_left = float(K_right)
            if fired and engine.n_adv >= target_advances:
                break
            if fired and consumed <= 0.0:
                K_left += min(dK, 1.0e-12)

    reached = engine.n_adv >= target_advances
    first = next(
        (float(row["K_MPa_sqrt_m"]) for row in history if row["fired"]),
        None,
    )
    return {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "material_class": manifest.name,
        "fallback_role": FALLBACK_ROLES.get(manifest.candidate_id),
        "mode": mode,
        "temperature_K": float(temperature_K),
        "status": "complete" if reached else "incomplete",
        "K_first_MPa_sqrt_m": first,
        "checkpoint_advances": int(engine.n_adv),
        "target_checkpoint_advances": int(target_advances),
        "micro_advance_total_m": float(engine.micro_advance_total_m),
        "outer_steps": int(outer_step),
        "history": history,
        "_final_arrays": _state_arrays(engine),
        "config": asdict(cfg),
        "mechanics_closure": (
            "prescribed scalar K ramp plus supplied channel drive factors"
        ),
        "state_evolution_source": "production v10.2.2 moving MPZ classes",
        "constitutive_K_shield_clip_applied": False,
        "shielding_ablation": mode == "shielding_off",
        "legacy_manifest_cap_reference_MPa_sqrt_m": float(
            manifest.max_K_shield_MPa_sqrt_m
        ),
        "legacy_manifest_cap_used_in_kinetics": False,
        "raw_equals_effective_when_shielding_active": _raw_effective_consistent(
            history, mode
        ),
    }


def _read_replay_schedule(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty replay schedule: {path}")
    required = {"dt_s", "temperature_K", "K_Pa_sqrt_m"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"replay schedule is missing columns: {sorted(missing)}")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item: dict[str, Any] = {
            "row_index": index,
            "dt_s": float(row["dt_s"]),
            "temperature_K": float(row["temperature_K"]),
            "K_Pa_sqrt_m": float(row["K_Pa_sqrt_m"]),
            "drive_factor_0": float(
                row.get("drive_factor_0", DEFAULT_THETA45_DRIVE_FACTORS[0])
            ),
            "drive_factor_1": float(
                row.get("drive_factor_1", DEFAULT_THETA45_DRIVE_FACTORS[1])
            ),
        }
        for name in (
            "expected_micro_advance_step_m",
            "expected_micro_advance_total_m",
            "expected_K_shield_effective_Pa_sqrt_m",
            "expected_mobile_count",
            "expected_retained_count",
        ):
            raw = row.get(name)
            item[name] = (
                float(raw) if raw not in {None, "", "nan", "NaN"} else math.nan
            )
        expected_fired = row.get("expected_fired")
        item["expected_fired"] = (
            None
            if expected_fired in {None, ""}
            else str(expected_fired).strip().lower() in {"1", "true", "yes"}
        )
        parsed.append(item)
    return parsed


def replay_shared_state(
    manifest: MaterialManifest,
    schedule: str | Path | list[dict[str, Any]],
    cfg: SharedReducedConfig,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Replay recorded 2-D outer engine calls through the production step method."""
    cfg = cfg.validate()
    rows = (
        _read_replay_schedule(schedule)
        if isinstance(schedule, (str, Path))
        else list(schedule)
    )
    engine = build_shared_engine(manifest, cfg, mode=mode)
    history: list[dict[str, Any]] = []

    with install_uncapped_physical_shielding():
        for index, row in enumerate(rows):
            dt = max(float(row["dt_s"]), 0.0)
            T = float(row["temperature_K"])
            K = max(float(row["K_Pa_sqrt_m"]), 0.0)
            factors = np.asarray(
                [
                    row.get("drive_factor_0", cfg.drive_factors[0]),
                    row.get("drive_factor_1", cfg.drive_factors[1]),
                ],
                dtype=float,
            )
            if np.any(factors < 0.0) or np.any(~np.isfinite(factors)):
                raise ValueError(
                    f"invalid drive factors at replay row {index}: {factors}"
                )
            engine.mpz._anisotropic_drive_factors = factors.copy()
            result = engine.step(K, T, dt)
            snapshot = {
                "row_index": index,
                "dt_s": dt,
                "temperature_K": T,
                "K_Pa_sqrt_m": K,
                "fired": bool(result.get("fired", False)),
                "micro_advance_step_m": float(
                    result.get("kinetic_micro_advance_step_m", 0.0)
                ),
                "micro_advance_total_m": float(engine.micro_advance_total_m),
                "sigma_opening_Pa": float(engine.sigma_opening_tip(K)),
                "sigma_cleave_Pa": float(
                    result.get("sigma_cleave_eff_Pa", result.get("sigma_tip", 0.0))
                ),
                "lambda_cleave_s": float(result.get("lambda_c", 0.0)),
                "dN_emit": float(
                    result.get("dN_emit", result.get("dN_emit_raw", 0.0))
                ),
                "dN_trapped": float(result.get("dN_trapped", 0.0)),
                "dN_recovered": float(result.get("dN_recovered", 0.0)),
                "dN_escaped": float(result.get("dN_escaped", 0.0)),
                **_field_snapshot(engine),
            }
            for expected_name, actual_name in (
                ("expected_micro_advance_step_m", "micro_advance_step_m"),
                ("expected_micro_advance_total_m", "micro_advance_total_m"),
                (
                    "expected_K_shield_effective_Pa_sqrt_m",
                    "K_shield_effective_Pa_sqrt_m",
                ),
                ("expected_mobile_count", "mobile_count"),
                ("expected_retained_count", "retained_count"),
            ):
                expected = float(row.get(expected_name, math.nan))
                snapshot[f"error_{actual_name}"] = (
                    float(snapshot[actual_name]) - expected
                    if math.isfinite(expected)
                    else math.nan
                )
            expected_fired = row.get("expected_fired")
            snapshot["fired_matches_expected"] = (
                True
                if expected_fired is None
                else bool(snapshot["fired"]) == bool(expected_fired)
            )
            history.append(snapshot)

    finite_errors = [
        abs(float(value))
        for row in history
        for key, value in row.items()
        if key.startswith("error_") and math.isfinite(float(value))
    ]
    return {
        "schema": MODEL_ID,
        "replay_mode": True,
        "candidate_id": manifest.candidate_id,
        "mode": mode,
        "n_schedule_rows": len(rows),
        "history": history,
        "_final_arrays": _state_arrays(engine),
        "config": asdict(cfg),
        "mechanics_closure": "recorded 2-D K, dt, T, and channel-factor schedule",
        "state_evolution_source": "production v10.2.2 moving MPZ classes",
        "constitutive_K_shield_clip_applied": False,
        "shielding_ablation": mode == "shielding_off",
        "legacy_manifest_cap_reference_MPa_sqrt_m": float(
            manifest.max_K_shield_MPa_sqrt_m
        ),
        "legacy_manifest_cap_used_in_kinetics": False,
        "raw_equals_effective_when_shielding_active": _raw_effective_consistent(
            history, mode
        ),
        "all_fired_flags_match": all(
            bool(row["fired_matches_expected"]) for row in history
        ),
        "maximum_abs_scalar_replay_error": max(finite_errors, default=0.0),
    }


def write_shared_result(result: dict[str, Any], out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    history = list(result.get("history", []))
    arrays = dict(result.get("_final_arrays", {}))
    payload = dict(result)
    payload.pop("history", None)
    payload.pop("_final_arrays", None)
    (root / "shared_reduced_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    (root / "shared_reduced_audit.json").write_text(
        json.dumps(
            {
                "schema": MODEL_ID,
                "candidate_id": result.get("candidate_id"),
                "mode": result.get("mode"),
                "state_evolution_source": result.get("state_evolution_source"),
                "mechanics_closure": result.get("mechanics_closure"),
                "constitutive_K_shield_clip_applied": False,
                "shielding_ablation": bool(result.get("shielding_ablation", False)),
                "legacy_manifest_cap_used_in_kinetics": False,
                "raw_equals_effective_when_shielding_active": bool(
                    result.get("raw_equals_effective_when_shielding_active", False)
                ),
                "fallback_role": result.get("fallback_role"),
                "n_history_rows": len(history),
                "spatial_final_state_saved": bool(arrays),
            },
            indent=2,
        )
    )
    if arrays:
        np.savez_compressed(root / "shared_reduced_final_state.npz", **arrays)
    if history:
        fields: list[str] = []
        for row in history:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (root / "shared_reduced_history.csv").open(
            "w", newline=""
        ) as handle:
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
    "load_manifest",
    "replay_shared_state",
    "run_monotonic_shared_front",
    "write_shared_result",
]
