"""v10.2.9 reduced campaign using the effective-opening signed engine.

The reduced mechanics and scoring remain those validated in v10.2.7/v10.2.8;
only the kernel-family schema and shared constitutive engine are upgraded.
"""
from __future__ import annotations

from pathlib import Path

from . import state_resolved_reduced_campaign_v1027 as _base
from .signed_kernel_family_v1029 import StateResolvedSignedShieldingKernelFamily
from .state_resolved_signed_engine_v1029 import StateResolvedSignedBurgersTipEngine

MODEL_ID = "v10.2.9_exact_effective_opening_reduced_campaign"


class StateResolvedProductionConfig(_base.StateResolvedProductionConfig):
    @classmethod
    def from_trace(
        cls,
        payload,
        *,
        kernel_family_path: str | Path,
        drive_family_path: str | Path,
    ):
        saved_family = _base.StateResolvedSignedShieldingKernelFamily
        try:
            _base.StateResolvedSignedShieldingKernelFamily = (
                StateResolvedSignedShieldingKernelFamily
            )
            return super().from_trace(
                payload,
                kernel_family_path=kernel_family_path,
                drive_family_path=drive_family_path,
            )
        finally:
            _base.StateResolvedSignedShieldingKernelFamily = saved_family

    def load_families(self):
        kernel = StateResolvedSignedShieldingKernelFamily.from_json(
            self.kernel_family_path
        )
        drive = _base.StateResolvedSignedDriveFamily.from_json(self.drive_family_path)
        drive.validate_against_kernel_family(kernel)
        return kernel, drive

    def build_engine(self, manifest, *, mode: str = "full"):
        saved_family = _base.StateResolvedSignedShieldingKernelFamily
        saved_engine = _base.StateResolvedSignedBurgersTipEngine
        try:
            _base.StateResolvedSignedShieldingKernelFamily = (
                StateResolvedSignedShieldingKernelFamily
            )
            _base.StateResolvedSignedBurgersTipEngine = (
                StateResolvedSignedBurgersTipEngine
            )
            return super().build_engine(manifest, mode=mode)
        finally:
            _base.StateResolvedSignedShieldingKernelFamily = saved_family
            _base.StateResolvedSignedBurgersTipEngine = saved_engine

    def parity_report(self, engine, *, mode: str):
        report = super().parity_report(engine, mode=mode)
        report.update(
            {
                "schema": MODEL_ID,
                "same_v10_2_9_engine_class": isinstance(
                    engine, StateResolvedSignedBurgersTipEngine
                ),
                "effective_opening_fixed_point_active": bool(
                    getattr(engine, "effective_opening_fixed_point_active", False)
                ),
                "kernel_state_opening_drive": "effective_K_after_signed_shielding",
                "K_shield_cap_present": False,
            }
        )
        report.pop("same_v10_2_6_engine_class", None)
        if not report["same_v10_2_9_engine_class"]:
            report["differences"].append(
                {
                    "path": "engine_class",
                    "expected": "StateResolvedSignedBurgersTipEngine_v10.2.9",
                    "actual": type(engine).__name__,
                }
            )
        if not report["effective_opening_fixed_point_active"]:
            report["differences"].append(
                {
                    "path": "effective_opening_fixed_point_active",
                    "expected": True,
                    "actual": False,
                }
            )
        report["passed"] = not report["differences"]
        return report


ReducedCampaignControl = _base.ReducedCampaignControl
run_reduced_r_curve = _base.run_reduced_r_curve
score_ceramic_reference = _base.score_ceramic_reference
VALID_MODES = _base.VALID_MODES
DEFAULT_TEMPERATURES_K = _base.DEFAULT_TEMPERATURES_K

__all__ = [
    "MODEL_ID",
    "ReducedCampaignControl",
    "StateResolvedProductionConfig",
    "run_reduced_r_curve",
    "score_ceramic_reference",
    "VALID_MODES",
    "DEFAULT_TEMPERATURES_K",
]
