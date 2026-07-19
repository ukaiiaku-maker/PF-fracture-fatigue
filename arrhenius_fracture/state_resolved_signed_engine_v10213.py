"""Shared monotonic/fatigue engine for v10.2.13 extension-only atlases."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .signed_burgers_shared_v1025 import VALIDATED_SCALAR_TRANSPORT
from .signed_kernel_family_v10213 import (
    SCHEMA as FAMILY_SCHEMA,
    ExtensionOnlySigned2DShieldingKernelFamily,
)
from .state_resolved_signed_engine_v10212 import (
    StateResolvedSignedBurgersTipEngine as _V10212Engine,
)

MODEL_ID = "v10.2.13_shared_extension_only_signed_burgers_engine"


class StateResolvedSignedBurgersTipEngine(_V10212Engine):
    """Use path-extension kernels while retaining local opening diagnostics."""

    _state_family_default: ExtensionOnlySigned2DShieldingKernelFamily | None = None

    @classmethod
    def configure_state_resolved_physics(
        cls,
        family: ExtensionOnlySigned2DShieldingKernelFamily | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
        **kwargs,
    ) -> None:
        resolved = (
            family
            if isinstance(family, ExtensionOnlySigned2DShieldingKernelFamily)
            else ExtensionOnlySigned2DShieldingKernelFamily.from_json(family)
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
        payload["state_resolved_signed_kernel_v10213"] = {
            "model_id": MODEL_ID,
            "schema": FAMILY_SCHEMA,
            "same_engine_for_monotonic_and_fatigue": True,
            "physical_kernel_axes": ["cumulative_crack_path_extension_m"],
            "opening_strength_fraction_used_for_interpolation": False,
            "opening_strength_fraction_retained_as_diagnostic": True,
            "analytical_r_eff_used_for_interpolation": False,
            "constitutive_K_shield_cap": False,
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
                "state_resolved_opening_used_for_kernel_interpolation": False,
                "state_resolved_observed_opening_strength_fraction": float(
                    getattr(
                        self._state_kernel_family,
                        "_last_observed_opening_strength_fraction",
                        0.0,
                    )
                ),
            }
        )
        return payload


__all__ = ["MODEL_ID", "StateResolvedSignedBurgersTipEngine"]
