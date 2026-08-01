"""v10.4.2 directional-J sign convention repair.

The domain-integral implementation uses q=1 on the inner contour and q=0 on
the outer contour, with the candidate crack-extension vector supplied as e1.
Under that fixed convention, positive raw signed J is forward configurational
work and negative raw signed J is non-driving.

The inherited sharp-front path latched a global sign from the first nonzero J.
That is unsafe because a small startup value can be negative before the loaded
root field becomes positive. The latch can then convert every later positive
root J to zero. This overlay removes that latch. It does not use abs(J): a
backward or otherwise inadmissible negative directional J remains non-driving.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_accepted_work_v1042 import transform_source as _accepted_transform

MODEL_ID = "v10.4.2_positive_raw_directional_J_convention"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1042_positive_directional_J"


def effective_directional_J(J_signed: float, *, allow_abs: bool = False) -> float:
    """Return the admissible directional configurational work."""
    value = float(J_signed)
    return abs(value) if allow_abs else max(value, 0.0)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _accepted_transform(source)

    old = """            _J_SIGN_REF = {'sign': 0.0, 'step': -1, 'J_root_signed': 0.0}

            def _effective_JK_from_info(Jinfo):
                J_signed = float(Jinfo.get('J_signed', Jinfo.get('J', 0.0)) or 0.0)
                if bool(getattr(args, 'allow_abs_directional_J', False)):
                    J_eff = abs(J_signed)
                    sign_ref = 0.0
                else:
                    if _J_SIGN_REF['sign'] == 0.0 and abs(J_signed) > 1e-30:
                        _J_SIGN_REF['sign'] = 1.0 if J_signed > 0.0 else -1.0
                        _J_SIGN_REF['step'] = int(step)
                        _J_SIGN_REF['J_root_signed'] = J_signed
                    sign_ref = _J_SIGN_REF['sign'] if _J_SIGN_REF['sign'] != 0.0 else 1.0
                    J_eff = max(sign_ref * J_signed, 0.0)
                K_eff = float(np.sqrt(max(J_eff, 0.0) * mat.Eprime))
                try:
                    Jinfo['J_effective_signed_positive'] = float(J_eff)
                    Jinfo['KJ_effective_signed_positive'] = float(K_eff)
                    Jinfo['J_sign_ref'] = float(sign_ref)
                    Jinfo['J_sign_ref_step'] = int(_J_SIGN_REF['step'])
                    Jinfo['J_root_signed_ref'] = float(_J_SIGN_REF['J_root_signed'])
                except Exception:
                    pass
                return float(J_eff), K_eff, J_signed
"""

    new = """            _J_SIGN_REF = {
                'sign': 1.0,
                'step': 0,
                'J_root_signed': 0.0,
                'convention': 'positive_raw_signed_J_is_forward_configurational_work',
            }

            def _effective_JK_from_info(Jinfo):
                J_signed = float(Jinfo.get('J_signed', Jinfo.get('J', 0.0)) or 0.0)
                allow_abs = bool(getattr(args, 'allow_abs_directional_J', False))
                if allow_abs:
                    J_eff = abs(J_signed)
                    sign_ref = 0.0
                    convention = 'explicit_abs_directional_J_ablation'
                else:
                    J_eff = max(J_signed, 0.0)
                    sign_ref = 1.0
                    convention = 'positive_raw_signed_J_is_forward_configurational_work'
                K_eff = float(np.sqrt(max(J_eff, 0.0) * mat.Eprime))
                try:
                    Jinfo['J_effective_signed_positive'] = float(J_eff)
                    Jinfo['KJ_effective_signed_positive'] = float(K_eff)
                    Jinfo['J_sign_ref'] = float(sign_ref)
                    Jinfo['J_sign_ref_step'] = 0
                    Jinfo['J_root_signed_ref'] = 0.0
                    Jinfo['J_sign_convention'] = convention
                    Jinfo['first_nonzero_sign_latch_used'] = False
                    Jinfo['absolute_value_used_for_production_directional_J'] = bool(allow_abs)
                except Exception:
                    pass
                return float(J_eff), K_eff, J_signed
"""

    text = _replace_once(
        text,
        old,
        new,
        "v10.4.2 positive directional-J convention",
    )
    return text


def load_transformed_sharp_front() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source_path = Path(__file__).with_name("sharp_front.py")
    transformed = transform_source(source_path.read_text())
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader=None)
    if spec is None:
        raise RuntimeError("could not allocate v10.4.2 positive-J module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(compile(transformed, str(source_path) + "[v10.4.2-positive-J]", "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = [
    "MODEL_ID",
    "effective_directional_J",
    "load_transformed_sharp_front",
    "transform_source",
]
