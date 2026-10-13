"""Native current-law front owner for the full-spatial scheduler."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .front_constitutive_strategy_r8 import CURRENT_STRATEGY


@dataclass(frozen=True)
class CurrentFrontAdvance:
    advanced_front_state: Mapping[str, Any]
    cleavage_event_occurred: bool
    cleavage_event_time_s: float | None
    geometry_events: tuple[Mapping[str, Any], ...]
    updated_r_tip_m: float
    updated_shielding_Pa_sqrt_m: float
    audit: Mapping[str, Any]


def bind_signed_fem_drive(engine: Any, stress_tensor_Pa: Any, *,
                          tangent_xy=(1.0, 0.0), serial=0) -> dict[str, Any]:
    """Bind signed Schmid channels from an accepted FEM tensor.

    Crystal traces are rotated with the accepted crack tangent/material frame;
    scalar populations are not rotated or copied between channels.
    """
    from .anisotropic_emission_v10174 import resolve_channel_drives

    tensor = np.asarray(stress_tensor_Pa, float).reshape(2, 2)
    tangent = np.asarray(tangent_xy, float).reshape(2)
    norm = float(np.linalg.norm(tangent))
    if not np.all(np.isfinite(tensor)) or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("signed spatial drive requires a finite tensor and tangent")
    tangent /= norm
    tangent_angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    crystal_theta = float(engine.mpz._anisotropic_emission_config.crystal_theta_deg)
    drive = resolve_channel_drives(
        tensor, (tensor, tensor), crystal_theta+tangent_angle,
        float(engine.mpz._anisotropic_emission_config.schmid_reference),
    )
    if not drive["tensor_drive_admissible"]:
        raise RuntimeError("accepted FEM tensor is not admissible for signed current drive")
    engine._r2_synthetic_drive_setup = False
    engine.mpz._anisotropic_drive_factors = np.asarray(drive["drive_factors"], float)
    engine.mpz._anisotropic_tau_signed_Pa = np.asarray(drive["tau_signed_Pa"], float)
    engine.mpz._anisotropic_drive_reliable = True
    engine.mpz._anisotropic_drive_serial = int(serial)
    return {
        "tensor_Pa": tensor.tolist(), "tangent_xy": tangent.tolist(),
        "crystal_theta_deg": crystal_theta,
        "material_frame_theta_deg": crystal_theta+tangent_angle,
        "drive_factors": list(map(float, drive["drive_factors"])),
        "tau_signed_Pa": list(map(float, drive["tau_signed_Pa"])),
        "normalization": "ACTUAL_ACCEPTED_FEM_TENSOR_SCHMID_RESOLUTION",
    }


class CurrentFrontProcessAdapter:
    """Advance the native current clock and continuous moving source once."""

    def __init__(self, material: Any, material_bundle: Any):
        self.material = material
        self.bundle = material_bundle

    def advance_front_process(self, accepted_front_state: Mapping[str, Any],
                              accepted_mechanical_drive: Mapping[str, Any],
                              temperature: float, duration: float) -> CurrentFrontAdvance:
        from .accepted_update_controller_r6 import advance_current_source_to_earliest_event
        from .voiding_production_v5 import capture_sharp_front_engine, restore_sharp_front_engine

        engine = restore_sharp_front_engine(
            self.material, self.bundle, accepted_front_state)
        if getattr(engine, "_front_constitutive_strategy", None) != CURRENT_STRATEGY:
            raise RuntimeError("current adapter received a non-current front payload")
        K = float(accepted_mechanical_drive["K_effective_Pa_sqrt_m"])
        tensor = accepted_mechanical_drive.get("stress_tensor_Pa")
        if tensor is None:
            raise RuntimeError("current spatial front requires its accepted FEM tensor")
        drive = bind_signed_fem_drive(
            engine, tensor,
            tangent_xy=accepted_mechanical_drive.get("tangent_xy", (1.0, 0.0)),
            serial=accepted_mechanical_drive.get("mechanics_serial", 0),
        )
        before_micro = float(getattr(engine, "micro_advance_total_m", 0.0))
        before_n = int(engine.n_adv)
        before_threshold = float(engine.hazard_threshold_action)
        update = advance_current_source_to_earliest_event(
            engine, applied_K_Pa_sqrt_m=K, temperature_K=float(temperature),
            duration_s=float(duration),
        )
        result_engine = update.engine
        fired = bool(update.result["fired"])
        consumed = float(update.consumed_s)
        payload = capture_sharp_front_engine(result_engine)
        audit = MappingProxyType({
            "constitutive_strategy": CURRENT_STRATEGY,
            "clock_authority": "NATIVE_CURRENT_PERSISTENT_REVERSIBLE",
            "translation_authority": "NATIVE_CURRENT_MOVING_MPZ",
            "accepted_drive": copy.deepcopy(dict(accepted_mechanical_drive)),
            "signed_drive": drive,
            "requested_duration_s": float(duration),
            "consumed_duration_s": consumed,
            "unused_duration_s": max(float(duration)-consumed, 0.0),
            "pre_threshold_action": before_threshold,
            "post_threshold_action": float(result_engine.hazard_threshold_action),
            "pre_n_adv": before_n, "post_n_adv": int(result_engine.n_adv),
            "continuous_micro_advance_m": (
                float(getattr(result_engine, "micro_advance_total_m", 0.0))-before_micro),
            "step_info": copy.deepcopy(update.result),
            "controller": copy.deepcopy(update.audit),
            "geometry_events": copy.deepcopy(update.geometry_events),
            "artificial_threshold_used": False,
        })
        return CurrentFrontAdvance(
            advanced_front_state=payload,
            cleavage_event_occurred=fired,
            cleavage_event_time_s=consumed if fired else None,
            geometry_events=tuple(copy.deepcopy(update.geometry_events)),
            updated_r_tip_m=float(result_engine.r_eff()),
            updated_shielding_Pa_sqrt_m=float(result_engine.K_shield()),
            audit=audit,
        )


__all__ = ["CurrentFrontAdvance", "CurrentFrontProcessAdapter", "bind_signed_fem_drive"]
