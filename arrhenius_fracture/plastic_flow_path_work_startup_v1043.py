"""Startup-safe endpoint-path work overlay for v10.4.3.

The endpoint-path ledger needs the equilibrated stress at the beginning of each
accepted step.  After the first accepted step that field is the previous final
``sigma_gp``.  Before the first mechanics solve, however, ``sigma_gp`` has not
been bound yet.  The physical beginning state is unloaded with ``u=0`` and
``ep=0``, so its exact stress is zero.

This overlay replaces the unconditional beginning-stress copy with an explicit
shape-checked prior-state reuse and a zero-stress initialization only when no
compatible prior accepted stress exists.  It changes no constitutive state,
fracture law, hazard, event gate, timestep decision, or work formula.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_path_work_v1043 import transform_source as _path_transform

MODEL_ID = "v10.4.3_startup_safe_equilibrated_endpoint_path_plastic_work"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_startup_safe_path_work"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _path_transform(source)

    old = """            sigma_gp_step0_path_v1043 = np.asarray(
                sigma_gp, dtype=float
            ).copy()
            ep_gp_step0_path_v1043 = np.asarray(ep_gp, dtype=float).copy()
"""

    new = """            _v1043_sigma_step0_candidate = locals().get('sigma_gp')
            if (
                _v1043_sigma_step0_candidate is not None
                and np.asarray(_v1043_sigma_step0_candidate).shape
                == (3, mesh.ne)
            ):
                sigma_gp_step0_path_v1043 = np.asarray(
                    _v1043_sigma_step0_candidate, dtype=float
                ).copy()
                endpoint_stress_start_source_v1043 = (
                    'previous_accepted_equilibrated_stress'
                )
            else:
                # Before the first mechanics solve the accepted beginning state
                # is u=0, ep=0, hence sigma=0 exactly.  A shape mismatch is also
                # treated as no reusable prior field rather than dereferencing a
                # stale mesh-sized array.
                sigma_gp_step0_path_v1043 = np.zeros(
                    (3, mesh.ne), dtype=float
                )
                endpoint_stress_start_source_v1043 = (
                    'unloaded_zero_stress_initialization'
                )
            ep_gp_step0_path_v1043 = np.asarray(ep_gp, dtype=float).copy()
"""

    return _replace_once(
        text,
        old,
        new,
        "v10.4.3 startup-safe endpoint stress snapshot",
    )


def load_transformed_sharp_front() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source_path = Path(__file__).with_name("sharp_front.py")
    transformed = transform_source(source_path.read_text())
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader=None)
    if spec is None:
        raise RuntimeError(
            "could not allocate v10.4.3 startup-safe endpoint-path module"
        )
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-startup-safe-endpoint-path-work]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
