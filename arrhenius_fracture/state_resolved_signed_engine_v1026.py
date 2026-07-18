"""One state-resolved signed-dislocation engine for monotonic and fatigue paths."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .signed_burgers_shared_v1025 import (
    CHANNEL_RESOLVED_TRANSPORT,
    VALIDATED_SCALAR_TRANSPORT,
    SignedBurgersAnisotropicTipEngine,
    _active_K,
    _total_K,
    _wake_K,
)
from .signed_kernel_family_v1026 import (
    SCHEMA as FAMILY_SCHEMA,
    StateResolvedSignedShieldingKernelFamily,
)

MODEL_ID = "v10.2.6_shared_state_resolved_signed_burgers_engine"


class StateResolvedSignedBurgersTipEngine(SignedBurgersAnisotropicTipEngine):
    """Shared monotonic/fatigue engine with state-envelope kernel resolution."""

    state_resolved_signed_kernel_active = True
    _state_family_default: StateResolvedSignedShieldingKernelFamily | None = None

    @classmethod
    def configure_state_resolved_physics(
        cls,
        family: StateResolvedSignedShieldingKernelFamily | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
    ) -> None:
        resolved = (
            family
            if isinstance(family, StateResolvedSignedShieldingKernelFamily)
            else StateResolvedSignedShieldingKernelFamily.from_json(family)
        )
        cls._state_family_default = resolved
        # The parent initializer installs the object through the same signed-state
        # path.  It is cloned per engine immediately after construction.
        cls._signed_kernel_default = resolved
        selected = str(transport_mode).strip().lower().replace("-", "_")
        if selected not in {
            VALIDATED_SCALAR_TRANSPORT,
            CHANNEL_RESOLVED_TRANSPORT,
        }:
            raise ValueError(f"invalid signed transport mode {transport_mode!r}")
        cls._signed_transport_mode_default = selected

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        family = cls._state_family_default
        payload["state_resolved_signed_kernel"] = {
            "model_id": MODEL_ID,
            "schema": FAMILY_SCHEMA,
            "same_engine_for_monotonic_and_fatigue": True,
            "family": family.audit_payload() if family is not None else None,
            "state_axes": [
                "r_eff_over_r0",
                "opening_strength_fraction",
                "crack_extension_m",
            ],
            "out_of_envelope_policy": "fail_closed",
            "constitutive_K_shield_cap": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        template = type(self)._state_family_default
        if template is None:
            raise RuntimeError(
                "v10.2.6 requires a validated state-resolved signed-kernel family"
            )
        super().__init__(*args, **kwargs)
        family = template.clone_for_engine()
        family.validate_state(self.mpz)
        self.mpz._signed_kernel = family
        self._state_kernel_family = family
        self._signed_current_K_Pa_sqrt_m = 0.0
        self._signed_last_state_coordinates = {
            "r_eff_over_r0": 1.0,
            "opening_strength_fraction": 0.0,
            "crack_extension_m": 0.0,
        }
        self._resolve_state_kernel(0.0)

    def _kernel_state_coordinates(self, K_Pa_sqrt_m: float) -> dict[str, float]:
        r0 = max(float(self.f.r0), 1.0e-30)
        r_eff = max(float(self.r_eff()), r0)
        sigma_uncapped = max(float(K_Pa_sqrt_m), 0.0) / math.sqrt(
            2.0 * math.pi * r_eff
        )
        sigma_cap = float(self.f.sigma_cap)
        if sigma_cap <= 0.0:
            raise RuntimeError(
                "state-resolved kernel uses opening_strength_fraction and therefore "
                "requires the serialized local cohesive/strength limit sigma_cap"
            )
        sigma_local = min(sigma_uncapped, sigma_cap)
        extension = max(
            float(getattr(self, "micro_advance_total_m", 0.0)),
            float(getattr(self.mpz, "advance_total_m", 0.0)),
            0.0,
        )
        return {
            "r_eff_over_r0": r_eff / r0,
            "opening_strength_fraction": sigma_local / sigma_cap,
            "crack_extension_m": extension,
        }

    def _resolve_state_kernel(self, K_Pa_sqrt_m: float) -> None:
        coordinates = self._kernel_state_coordinates(K_Pa_sqrt_m)
        self._state_kernel_family.resolve(**coordinates)
        self._signed_last_state_coordinates = coordinates
        self._signed_current_K_Pa_sqrt_m = max(float(K_Pa_sqrt_m), 0.0)

    def sigma_tip(self, K):
        # Every monotonic and cyclic phase-point stress evaluation resolves the
        # same mechanical kernel family before the inherited constitutive law is
        # evaluated.  This prevents fatigue from using a separate stale kernel.
        self._resolve_state_kernel(float(K))
        return super().sigma_tip(K)

    def _active_shielding_raw_uncapped(self) -> float:
        self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _active_K(self.mpz)

    def _active_shielding_signed(self) -> float:
        self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _active_K(self.mpz)

    def _wake_shielding_signed(self) -> float:
        self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _wake_K(self.mpz)

    def K_shield(self):
        self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _total_K(self.mpz)

    def _state_resolved_diagnostics(self) -> dict[str, Any]:
        family = self._state_kernel_family
        return {
            "state_resolved_signed_kernel_model_id": MODEL_ID,
            "state_resolved_same_monotonic_fatigue_engine": True,
            "state_resolved_r_eff_over_r0": float(
                self._signed_last_state_coordinates["r_eff_over_r0"]
            ),
            "state_resolved_opening_strength_fraction": float(
                self._signed_last_state_coordinates[
                    "opening_strength_fraction"
                ]
            ),
            "state_resolved_crack_extension_m": float(
                self._signed_last_state_coordinates["crack_extension_m"]
            ),
            "state_resolved_kernel_state_ids": list(family._last_state_ids),
            "state_resolved_kernel_weights": family._last_weights.tolist(),
            "state_resolved_mode_II_active_operator_norm": float(
                (family.active_kernel_II ** 2).sum() ** 0.5
            ),
            "state_resolved_mode_II_wake_operator_norm": float(
                (family.wake_kernel_II ** 2).sum() ** 0.5
            ),
            "state_resolved_out_of_envelope_policy": "fail_closed",
            "state_resolved_K_shield_cap_applied": False,
        }

    def step(self, K, T, dt):
        self._signed_current_K_Pa_sqrt_m = max(float(K), 0.0)
        result = super().step(K, T, dt)
        result.update(self._state_resolved_diagnostics())
        return result

    def cycle_step_waveform(self, controller, waveform, T_K: float,
                            requested_cycles=None, force_cycles=None):
        self._signed_current_K_Pa_sqrt_m = max(float(waveform.Kmax), 0.0)
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        result.update(self._state_resolved_diagnostics())
        return result


__all__ = [
    "MODEL_ID",
    "StateResolvedSignedBurgersTipEngine",
]
