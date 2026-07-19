"""Shared monotonic/fatigue engine for the v10.2.12 real signed 2-D atlas."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .signed_burgers_shared_v1025 import VALIDATED_SCALAR_TRANSPORT
from .signed_kernel_family_v10212 import (
    SCHEMA as FAMILY_SCHEMA,
    RealSigned2DShieldingKernelFamily,
)
from .state_resolved_signed_engine_v1029 import (
    StateResolvedSignedBurgersTipEngine as _V1029Engine,
)

MODEL_ID = "v10.2.12_shared_real_2d_signed_burgers_engine"


class StateResolvedSignedBurgersTipEngine(_V1029Engine):
    """v10.2.9 effective-opening fixed point with v10.2.12 atlas semantics."""

    _state_family_default: RealSigned2DShieldingKernelFamily | None = None

    @classmethod
    def configure_state_resolved_physics(
        cls,
        family: RealSigned2DShieldingKernelFamily | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
        **kwargs,
    ) -> None:
        resolved = (
            family
            if isinstance(family, RealSigned2DShieldingKernelFamily)
            else RealSigned2DShieldingKernelFamily.from_json(family)
        )
        super().configure_state_resolved_physics(
            resolved,
            transport_mode,
            **kwargs,
        )
        cls._state_family_default = resolved
        cls._signed_kernel_default = resolved

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        family = cls._state_family_default
        payload["state_resolved_signed_kernel_v10212"] = {
            "model_id": MODEL_ID,
            "schema": FAMILY_SCHEMA,
            "same_engine_for_monotonic_and_fatigue": True,
            "kernel_state_opening_drive": "effective_K_after_signed_shielding",
            "physical_kernel_axes": [
                "opening_strength_fraction",
                "crack_extension_m",
            ],
            "analytical_r_eff_used_for_interpolation": False,
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "constitutive_K_shield_cap": False,
            "family": family.audit_payload() if family is not None else None,
        }
        return payload

    def _state_resolved_diagnostics(self) -> dict[str, Any]:
        payload = super()._state_resolved_diagnostics()
        payload.update(
            {
                "state_resolved_signed_kernel_model_id": MODEL_ID,
                "state_resolved_kernel_radius_axis_policy": (
                    "disabled_constant_compatibility"
                ),
                "state_resolved_observed_analytical_r_eff_over_r0": float(
                    getattr(
                        self._state_kernel_family,
                        "_last_observed_analytical_r_eff_over_r0",
                        1.0,
                    )
                ),
                "state_resolved_analytical_r_eff_used_for_interpolation": False,
            }
        )
        return payload


__all__ = ["MODEL_ID", "StateResolvedSignedBurgersTipEngine"]
