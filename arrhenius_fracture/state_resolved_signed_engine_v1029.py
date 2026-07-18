"""v10.2.9 shared signed engine with effective-opening kernel selection.

The kernel state cannot be selected from applied K because the selected kernel
changes signed shielding and therefore changes the local opening state.  This
module solves that constitutive loop as a damped, bounded fixed point using the
same implementation for monotonic fracture and every fatigue phase point.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .signed_burgers_shared_v1025 import (
    CHANNEL_RESOLVED_TRANSPORT,
    VALIDATED_SCALAR_TRANSPORT,
    _active_K,
    _total_K,
    _wake_K,
)
from .signed_kernel_family_v1029 import (
    SCHEMA as FAMILY_SCHEMA,
    StateResolvedSignedShieldingKernelFamily,
)
from .state_resolved_signed_engine_v1026 import (
    StateResolvedSignedBurgersTipEngine as _V1026Engine,
)

MODEL_ID = "v10.2.9_shared_effective_opening_signed_burgers_engine"


class StateResolvedSignedBurgersTipEngine(_V1026Engine):
    """Shared monotonic/fatigue engine with a self-consistent kernel state."""

    effective_opening_fixed_point_active = True
    _state_family_default: StateResolvedSignedShieldingKernelFamily | None = None
    _fixed_point_tolerance_default = 1.0e-8
    _fixed_point_max_iterations_default = 80
    _fixed_point_damping_default = 0.5

    @classmethod
    def configure_state_resolved_physics(
        cls,
        family: StateResolvedSignedShieldingKernelFamily | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
        *,
        fixed_point_tolerance: float = 1.0e-8,
        fixed_point_max_iterations: int = 80,
        fixed_point_damping: float = 0.5,
    ) -> None:
        resolved = (
            family
            if isinstance(family, StateResolvedSignedShieldingKernelFamily)
            else StateResolvedSignedShieldingKernelFamily.from_json(family)
        )
        selected = str(transport_mode).strip().lower().replace("-", "_")
        if selected not in {
            VALIDATED_SCALAR_TRANSPORT,
            CHANNEL_RESOLVED_TRANSPORT,
        }:
            raise ValueError(f"invalid signed transport mode {transport_mode!r}")
        tolerance = float(fixed_point_tolerance)
        iterations = int(fixed_point_max_iterations)
        damping = float(fixed_point_damping)
        if tolerance <= 0.0 or not math.isfinite(tolerance):
            raise ValueError("fixed-point tolerance must be positive and finite")
        if iterations < 2:
            raise ValueError("fixed-point iteration limit must be at least two")
        if not (0.0 < damping <= 1.0):
            raise ValueError("fixed-point damping must lie in (0,1]")
        cls._state_family_default = resolved
        cls._signed_kernel_default = resolved
        cls._signed_transport_mode_default = selected
        cls._fixed_point_tolerance_default = tolerance
        cls._fixed_point_max_iterations_default = iterations
        cls._fixed_point_damping_default = damping

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        family = cls._state_family_default
        payload["state_resolved_signed_kernel_v1029"] = {
            "model_id": MODEL_ID,
            "schema": FAMILY_SCHEMA,
            "same_engine_for_monotonic_and_fatigue": True,
            "kernel_state_opening_drive": "effective_K_after_signed_shielding",
            "fixed_point_tolerance": cls._fixed_point_tolerance_default,
            "fixed_point_max_iterations": cls._fixed_point_max_iterations_default,
            "fixed_point_damping": cls._fixed_point_damping_default,
            "out_of_envelope_policy": "axis_specific_fail_closed",
            "constitutive_K_shield_cap": False,
            "family": family.audit_payload() if family is not None else None,
        }
        return payload

    def __init__(self, *args, **kwargs):
        self._kernel_resolution_active = False
        self._fixed_point_iterations = 0
        self._fixed_point_residual = 0.0
        self._fixed_point_converged = False
        self._effective_K_tip_Pa_sqrt_m = 0.0
        self._opening_sigma_uncapped_Pa = 0.0
        self._opening_sigma_local_Pa = 0.0
        super().__init__(*args, **kwargs)

    def _geometric_state(self) -> tuple[float, float, float, float]:
        r0 = max(float(self.f.r0), 1.0e-30)
        r_eff = max(float(self.r_eff()), r0)
        extension = max(
            float(getattr(self, "micro_advance_total_m", 0.0)),
            float(getattr(self.mpz, "advance_total_m", 0.0)),
            0.0,
        )
        sigma_cap = float(self.f.sigma_cap)
        if sigma_cap <= 0.0:
            raise RuntimeError(
                "effective-opening kernel state requires the serialized local "
                "cohesive/strength limit sigma_cap"
            )
        return r0, r_eff, extension, sigma_cap

    def _resolve_state_kernel(self, K_Pa_sqrt_m: float) -> None:
        if self._kernel_resolution_active:
            return
        self._kernel_resolution_active = True
        try:
            applied = max(float(K_Pa_sqrt_m), 0.0)
            r0, r_eff, extension, sigma_cap = self._geometric_state()
            previous = getattr(self, "_signed_last_state_coordinates", {})
            opening = float(previous.get("opening_strength_fraction", 0.0))
            if not math.isfinite(opening):
                opening = 0.0
            opening = min(max(opening, 0.0), 1.0)
            tolerance = type(self)._fixed_point_tolerance_default
            damping = type(self)._fixed_point_damping_default
            maximum = type(self)._fixed_point_max_iterations_default
            converged = False
            residual = math.inf
            effective_K = 0.0
            sigma_uncapped = 0.0
            sigma_local = 0.0
            iterations = 0

            for iterations in range(1, maximum + 1):
                self._state_kernel_family.resolve(
                    r_eff_over_r0=r_eff / r0,
                    opening_strength_fraction=opening,
                    crack_extension_m=extension,
                )
                K_shield = float(_total_K(self.mpz))
                effective_K = max(applied - K_shield, 0.0)
                sigma_uncapped = effective_K / math.sqrt(
                    2.0 * math.pi * r_eff
                )
                sigma_local = min(max(sigma_uncapped, 0.0), sigma_cap)
                target = sigma_local / sigma_cap
                residual = target - opening
                if abs(residual) <= tolerance:
                    converged = True
                    opening = target
                    break
                opening = min(max(opening + damping * residual, 0.0), 1.0)

            if not converged:
                raise RuntimeError(
                    "state-resolved signed-kernel effective-opening fixed point "
                    f"did not converge after {maximum} iterations; "
                    f"K_applied={applied:.9e}, residual={residual:.9e}"
                )

            # Leave the operator resolved at the converged coordinate and verify
            # one final constitutive residual after that exact operator is installed.
            self._state_kernel_family.resolve(
                r_eff_over_r0=r_eff / r0,
                opening_strength_fraction=opening,
                crack_extension_m=extension,
            )
            K_shield = float(_total_K(self.mpz))
            effective_K = max(applied - K_shield, 0.0)
            sigma_uncapped = effective_K / math.sqrt(2.0 * math.pi * r_eff)
            sigma_local = min(max(sigma_uncapped, 0.0), sigma_cap)
            final_target = sigma_local / sigma_cap
            final_residual = final_target - opening
            if abs(final_residual) > 5.0 * tolerance:
                raise RuntimeError(
                    "effective-opening kernel resolution lost convergence after "
                    f"final operator installation; residual={final_residual:.9e}"
                )

            self._signed_current_K_Pa_sqrt_m = applied
            self._signed_last_state_coordinates = {
                "r_eff_over_r0": r_eff / r0,
                "opening_strength_fraction": opening,
                "crack_extension_m": extension,
            }
            self._fixed_point_iterations = int(iterations)
            self._fixed_point_residual = float(final_residual)
            self._fixed_point_converged = True
            self._effective_K_tip_Pa_sqrt_m = float(effective_K)
            self._opening_sigma_uncapped_Pa = float(sigma_uncapped)
            self._opening_sigma_local_Pa = float(sigma_local)
        finally:
            self._kernel_resolution_active = False

    def sigma_tip(self, K):
        applied = max(float(K), 0.0)
        self._signed_current_K_Pa_sqrt_m = applied
        self._resolve_state_kernel(applied)
        # Use the converged effective K directly.  This is algebraically the same
        # local strength law as the parent implementation without re-entering the
        # kernel resolver through K_shield().
        return float(self._opening_sigma_local_Pa)

    def _active_shielding_raw_uncapped(self) -> float:
        if not self._kernel_resolution_active:
            self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _active_K(self.mpz)

    def _active_shielding_signed(self) -> float:
        if not self._kernel_resolution_active:
            self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _active_K(self.mpz)

    def _wake_shielding_signed(self) -> float:
        if not self._kernel_resolution_active:
            self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _wake_K(self.mpz)

    def K_shield(self):
        if not self._kernel_resolution_active:
            self._resolve_state_kernel(self._signed_current_K_Pa_sqrt_m)
        return _total_K(self.mpz)

    def _state_resolved_diagnostics(self) -> dict[str, Any]:
        payload = super()._state_resolved_diagnostics()
        payload.update(
            {
                "state_resolved_signed_kernel_model_id": MODEL_ID,
                "state_resolved_opening_drive": "effective_K_after_signed_shielding",
                "state_resolved_effective_K_tip_Pa_sqrt_m": float(
                    self._effective_K_tip_Pa_sqrt_m
                ),
                "state_resolved_opening_sigma_uncapped_Pa": float(
                    self._opening_sigma_uncapped_Pa
                ),
                "state_resolved_opening_sigma_local_Pa": float(
                    self._opening_sigma_local_Pa
                ),
                "state_resolved_fixed_point_iterations": int(
                    self._fixed_point_iterations
                ),
                "state_resolved_fixed_point_residual": float(
                    self._fixed_point_residual
                ),
                "state_resolved_fixed_point_converged": bool(
                    self._fixed_point_converged
                ),
                "state_resolved_opening_boundary_action": getattr(
                    self._state_kernel_family, "_last_boundary_action", "none"
                ),
            }
        )
        return payload


__all__ = ["MODEL_ID", "StateResolvedSignedBurgersTipEngine"]
