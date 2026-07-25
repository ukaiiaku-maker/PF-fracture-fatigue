"""Environment-driven specimen geometry override for v10.2.27 mechanics.

The low-level sharp-front driver historically hard-coded the default specimen in
``make_emergent_config``. This installer makes specimen dimensions part of the
same explicit mechanical configuration used by the kernel resolver, without
changing material kinetics or the fracture law.
"""
from __future__ import annotations

import os
from typing import Callable

from . import sharp_front

MODEL_ID = "v10.2.27_explicit_specimen_geometry_override"
_ORIGINAL: Callable | None = None
_INSTALLED = False


def _positive_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = float(default if raw is None or not raw.strip() else raw)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def install_geometry_override() -> None:
    global _ORIGINAL, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL = sharp_front.make_emergent_config

    def configured():
        assert _ORIGINAL is not None
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
    global _ORIGINAL, _INSTALLED
    if not _INSTALLED:
        return
    assert _ORIGINAL is not None
    sharp_front.make_emergent_config = _ORIGINAL
    _ORIGINAL = None
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
    }


__all__ = [
    "MODEL_ID",
    "install_geometry_override",
    "restore_geometry_override",
    "audit_payload",
]
