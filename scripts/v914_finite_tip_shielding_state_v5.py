"""v5 finite-tip shielding audit for minimal reversible fatigue.

The qualified v4 reversible model retains the historical sharp-crack shielding
kernel W(x) proportional to 1/sqrt(x), even after the source-slip ledger has
produced a micron-scale crack-tip radius.  The amplified negative-R branch is
strongly dominated by retained GND in the first MPZ cell, so this module tests
whether replacing the sub-tip sharp singularity by the *existing* evolving tip
radius removes that grid sensitivity.

This is an explicit model-form audit, not a replacement of v4.  Two parameter-
free geometric regularizations are provided:

* radius_floor: x_eff = max(x, r_tip, r_core)
* radius_shift: x_eff = max(x + r_tip, r_core)

Both preserve the qualified retained-GND sign/orientation law, material
constants, transport/storage kinetics, return semantics, barriers, stochastic
law, and source/blunting law.  No fitted shielding factor is introduced.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(_SCRIPTS), str(_DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_DEFAULT_V914))
sys.path.insert(0, str(_SCRIPTS))

from v914_minimal_reversible_state_v4 import (
    MinimalReversibleEmergentGNDState as _V4State,
)


FLOOR_MODEL_ID = "v9.14_reversible_v5_finite_tip_shield_radius_floor"
SHIFT_MODEL_ID = "v9.14_reversible_v5_finite_tip_shield_radius_shift"
FLOOR_SCHEMA = "v914_minimal_reversible_state_extension_v5_finite_tip_floor"
SHIFT_SCHEMA = "v914_minimal_reversible_state_extension_v5_finite_tip_shift"


def finite_tip_shield_kernel(
    x_m: np.ndarray,
    *,
    G_Pa: float,
    b_m: float,
    nu: float,
    cell_area_m2: float,
    tip_radius_m: float,
    core_radius_m: float,
    mode: str,
) -> np.ndarray:
    """Return the finite-tip shielding quadrature kernel.

    ``radius_floor`` is the minimal singularity cutoff implied by a finite tip:
    material inside the current tip radius cannot be assigned a sharper
    1/sqrt(x) weight than material at r_tip.  ``radius_shift`` provides a smooth
    companion audit using x+r_tip.  Neither mode contains an adjustable scale.
    """
    x = np.asarray(x_m, dtype=float)
    if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("x_m must be a finite nonnegative one-dimensional array")
    radius = max(float(tip_radius_m), float(core_radius_m), 1.0e-30)
    core = max(float(core_radius_m), 1.0e-30)
    if mode == "radius_floor":
        x_eff = np.maximum(x, radius)
    elif mode == "radius_shift":
        x_eff = np.maximum(x + radius, core)
    else:
        raise ValueError(f"unsupported finite-tip shielding mode: {mode}")
    prefactor = float(G_Pa) * float(b_m) / max(1.0 - float(nu), 1.0e-12)
    per_line = prefactor / np.sqrt(2.0 * math.pi * x_eff)
    return per_line * float(cell_area_m2) / 1.0e6


class _FiniteTipShieldState(_V4State):
    SHIELD_MODE = ""
    MODEL_ID = ""
    STATE_EXTENSION_SCHEMA = ""

    def _finite_tip_shield_kernel(self) -> np.ndarray:
        return finite_tip_shield_kernel(
            np.asarray(self.x, dtype=float),
            G_Pa=self.c.G_Pa,
            b_m=self.c.b_m,
            nu=self.c.nu,
            cell_area_m2=self.cell_area_m2,
            tip_radius_m=self.tip_radius_m(),
            core_radius_m=self.c.core_regularization_b * self.c.b_m,
            mode=self.SHIELD_MODE,
        )

    def K_shield_MPa_sqrt_m(self) -> float:
        factors = np.asarray(self.c.shielding_orientation_factors, dtype=float)
        kernel = self._finite_tip_shield_kernel()
        return float(
            np.sum(
                factors[:, None]
                * self.signed_gnd_m2()
                * kernel[None, :]
            )
        )

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": self.MODEL_ID,
                "finite_tip_shielding_audit": True,
                "finite_tip_shielding_mode": self.SHIELD_MODE,
                "finite_tip_shielding_radius_source": "existing_dynamic_tip_radius_m",
                "finite_tip_shielding_fitted_parameter": False,
                "retained_gnd_sign_law_changed_from_v4": False,
                "transport_storage_physics_changed_from_v4": False,
                "surface_return_semantics_changed_from_v4": False,
                "cleavage_barrier_changed_from_v4": False,
                "emission_barrier_changed_from_v4": False,
            }
        )
        return metadata

    def reversibility_diagnostics(self) -> dict[str, float]:
        data = dict(super().reversibility_diagnostics())
        finite = self._finite_tip_shield_kernel()
        sharp = np.asarray(self._shield_kernel, dtype=float)
        ratio0 = (
            float(finite[0] / sharp[0])
            if sharp.size and finite.size and sharp[0] > 0.0
            else math.nan
        )
        data.update(
            {
                "reversible_finite_tip_radius_m": float(self.tip_radius_m()),
                "reversible_finite_tip_shield_kernel0_ratio_to_sharp": ratio0,
            }
        )
        return data

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        payload = dict(super().reversible_checkpoint_payload())
        payload["schema"] = self.STATE_EXTENSION_SCHEMA
        payload["finite_tip_shielding_mode"] = self.SHIELD_MODE
        return payload

    def restore_reversible_checkpoint_payload(
        self,
        payload: Mapping[str, Any] | None,
    ) -> None:
        self._ensure_reversible_fields()
        if not payload:
            return
        if payload.get("schema") != self.STATE_EXTENSION_SCHEMA:
            raise ValueError(
                "finite-tip v5 checkpoint schema mismatch; v4/sharp and the two "
                "v5 regularizations are not cross-promoted"
            )
        if payload.get("finite_tip_shielding_mode") != self.SHIELD_MODE:
            raise ValueError("finite-tip v5 checkpoint shielding mode mismatch")
        parent = dict(payload)
        parent["schema"] = "v914_minimal_reversible_state_extension_v4"
        parent.pop("finite_tip_shielding_mode", None)
        super().restore_reversible_checkpoint_payload(parent)


class FiniteTipFloorReversibleState(_FiniteTipShieldState):
    SHIELD_MODE = "radius_floor"
    MODEL_ID = FLOOR_MODEL_ID
    STATE_EXTENSION_SCHEMA = FLOOR_SCHEMA


class FiniteTipShiftReversibleState(_FiniteTipShieldState):
    SHIELD_MODE = "radius_shift"
    MODEL_ID = SHIFT_MODEL_ID
    STATE_EXTENSION_SCHEMA = SHIFT_SCHEMA


__all__ = [
    "FLOOR_MODEL_ID",
    "SHIFT_MODEL_ID",
    "FLOOR_SCHEMA",
    "SHIFT_SCHEMA",
    "finite_tip_shield_kernel",
    "FiniteTipFloorReversibleState",
    "FiniteTipShiftReversibleState",
]
