"""Runtime-safe v10.4.3 plastic-dominance transform and loader.

The primary overlay composes several historical source transforms.  This final
adapter repairs string-literal escaping in the generated candidate-audit writer
before compiling the transformed production module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from . import plastic_dominance_v1043 as _base

MODEL_ID = _base.MODEL_ID
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_plastic_dominance_runtime"
terminal_metrics_v1043 = _base.terminal_metrics_v1043
contour_scan_v1043 = _base.contour_scan_v1043


def transform_source(source: str) -> str:
    transformed = _base.transform_source(source)
    broken_write = "_v1043_fp.write('" + "\n" + "')"
    fixed_write = "_v1043_fp.write('\\n')"
    broken_concat = "+ '" + "\n" + "'"
    fixed_concat = "+ '\\n'"
    write_count = transformed.count(broken_write)
    concat_count = transformed.count(broken_concat)
    if write_count != 1 or concat_count != 1:
        raise RuntimeError(
            "v10.4.3 generated diagnostic newline pattern changed: "
            f"write={write_count} concat={concat_count}"
        )
    transformed = transformed.replace(broken_write, fixed_write, 1)
    transformed = transformed.replace(broken_concat, fixed_concat, 1)
    return transformed


def load_transformed_sharp_front() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source_path = Path(__file__).with_name("sharp_front.py")
    transformed = transform_source(source_path.read_text())
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader=None)
    if spec is None:
        raise RuntimeError("could not allocate v10.4.3 runtime module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-plastic-dominance-runtime]",
                "exec",
            ),
            module.__dict__,
        )
        module._v1042_terminal_metrics = terminal_metrics_v1043
        module._v1042_contour_scan = contour_scan_v1043
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = [
    "MODEL_ID",
    "MODULE_NAME",
    "contour_scan_v1043",
    "load_transformed_sharp_front",
    "terminal_metrics_v1043",
    "transform_source",
]
