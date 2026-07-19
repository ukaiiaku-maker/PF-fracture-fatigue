"""Shared monotonic/fatigue engine for v10.2.14 active-only signed kernels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .signed_burgers_shared_v1025 import VALIDATED_SCALAR_TRANSPORT
from .signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as FAMILY_SCHEMA,
)
from .state_resolved_signed_engine_v10212 import (
    StateResolvedSignedBurgersTipEngine as _V10212Engine,
)

MODEL_ID = "v10.2.14_shared_active_only_signed_burgers_engine"


class StateResolvedSignedBurgersTipEngine(_V10212Engine):
    """Use the measured active operator and fail closed on wake shielding."""

    _state_family_default: ActiveOnlySigned2DShieldingKernelFamily | None = None

    @classmethod
    def configure_state_resolved_physics(
        cls,
        family: ActiveOnlySigned2DShieldingKernelFamily | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
        **kwargs,
    ) -> None:
        resolved = (
            family
            if isinstance(family, ActiveOnlySigned2DShieldingKernelFamily)
            else ActiveOnlySigned2DShieldingKernelFamily.from_json(family)
        )
        super().configure_state_resolved_physics(
            resolved,
            transport_mode,
            **kwargs,
        )
        cls._state_family_default = resolved
        cls._signed_kernel_default = resolved

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpz.cfg.wake_shielding = False

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        family = cls._state_family_default
        payload["state_resolved_signed_kernel_v10214"] = {
            "model_id": MODEL_ID,
            "schema": FAMILY_SCHEMA,
            "same_engine_for_monotonic_and_fatigue": True,
            "physical_kernel_axes": ["cumulative_crack_path_extension_m"],
            "opening_strength_fraction_used_for_interpolation": False,
            "analytical_r_eff_used_for_interpolation": False,
            "constitutive_K_shield_cap": False,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_enabled": False,
            "family": family.audit_payload() if family is not None else None,
        }
        return payload

    def _state_resolved_diagnostics(self) -> dict[str, Any]:
        payload = super()._state_resolved_diagnostics()
        payload.update(
            {
                "state_resolved_signed_kernel_model_id": MODEL_ID,
                "state_resolved_physical_kernel_axes": [
                    "cumulative_crack_path_extension_m"
                ],
                "state_resolved_active_kernel_mechanically_measured": True,
                "state_resolved_wake_kernel_mechanically_measured": False,
                "state_resolved_wake_shielding_enabled": False,
            }
        )
        return payload


__all__ = ["MODEL_ID", "StateResolvedSignedBurgersTipEngine"]
