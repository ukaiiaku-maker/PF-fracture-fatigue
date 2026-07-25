"""Environment-driven mechanical geometry and elasticity override for v10.2.27.

The low-level sharp-front driver historically obtained specimen dimensions and
cubic tungsten constants from module defaults. This installer makes those values
follow the same resolved mechanical configuration used to construct the signed
kernel, without changing material kinetics or the fracture law.
"""
from __future__ import annotations

import os
from typing import Callable

from . import crystal
from . import sharp_front

MODEL_ID = "v10.2.27_explicit_mechanical_geometry_override"
_ORIGINAL: Callable | None = None
_ORIGINAL_ELASTICITY: tuple[float, float, float] | None = None
_INSTALLED = False


def _positive_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = float(default if raw is None or not raw.strip() else raw)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _apply_elasticity_environment() -> None:
    c11 = _positive_env("V10227_CRYSTAL_C11_PA", crystal.W_C11)
    c12 = _positive_env("V10227_CRYSTAL_C12_PA", crystal.W_C12)
    c44 = _positive_env("V10227_CRYSTAL_C44_PA", crystal.W_C44)
    if c11 <= c12:
        raise ValueError("resolved cubic elasticity requires C11 > C12")
    crystal.W_C11 = c11
    crystal.W_C12 = c12
    crystal.W_C44 = c44


def install_geometry_override() -> None:
    global _ORIGINAL, _ORIGINAL_ELASTICITY, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL = sharp_front.make_emergent_config
    _ORIGINAL_ELASTICITY = (crystal.W_C11, crystal.W_C12, crystal.W_C44)

    def configured():
        assert _ORIGINAL is not None
        _apply_elasticity_environment()
        cfg = _ORIGINAL()
        cfg.geometry.Lx = _positive_env("V10227_SPECIMEN_LX_M", cfg.geometry.Lx)
        cfg.geometry.Ly = _positive_env("V10227_SPECIMEN_LY_M", cfg.geometry.Ly)
        cfg.geometry.a0 = _positive_env("V10227_INITIAL_CRACK_LENGTH_M", cfg.geometry.a0)
        cfg.geometry.notch_half_thickness = _positive_env(
            "V10227_NOTCH_HALF_THICKNESS_M",
            cfg.geometry.notch_half_thickness,
        )
        if cfg.geometry.a0 >= cfg.geometry.Lx:
            raise ValueError("initial crack length must be smaller than specimen Lx")
        if 2.0 * cfg.geometry.notch_half_thickness >= cfg.geometry.Ly:
            raise ValueError("notch thickness must be smaller than specimen Ly")
        return cfg

    sharp_front.make_emergent_config = configured
    _INSTALLED = True


def restore_geometry_override() -> None:
    global _ORIGINAL, _ORIGINAL_ELASTICITY, _INSTALLED
    if not _INSTALLED:
        return
    assert _ORIGINAL is not None
    sharp_front.make_emergent_config = _ORIGINAL
    if _ORIGINAL_ELASTICITY is not None:
        crystal.W_C11, crystal.W_C12, crystal.W_C44 = _ORIGINAL_ELASTICITY
    _ORIGINAL = None
    _ORIGINAL_ELASTICITY = None
    _INSTALLED = False


def audit_payload() -> dict[str, object]:
    return {
        "schema": MODEL_ID,
        "installed": bool(_INSTALLED),
        "specimen_Lx_m": float(os.environ.get("V10227_SPECIMEN_LX_M", "0") or 0.0),
        "specimen_Ly_m": float(os.environ.get("V10227_SPECIMEN_LY_M", "0") or 0.0),
        "initial_crack_length_m": float(
            os.environ.get("V10227_INITIAL_CRACK_LENGTH_M", "0") or 0.0
        ),
        "notch_half_thickness_m": float(
            os.environ.get("V10227_NOTCH_HALF_THICKNESS_M", "0") or 0.0
        ),
        "crystal_C11_Pa": float(os.environ.get("V10227_CRYSTAL_C11_PA", "0") or 0.0),
        "crystal_C12_Pa": float(os.environ.get("V10227_CRYSTAL_C12_PA", "0") or 0.0),
        "crystal_C44_Pa": float(os.environ.get("V10227_CRYSTAL_C44_PA", "0") or 0.0),
    }


__all__ = [
    "MODEL_ID",
    "install_geometry_override",
    "restore_geometry_override",
    "audit_payload",
]
