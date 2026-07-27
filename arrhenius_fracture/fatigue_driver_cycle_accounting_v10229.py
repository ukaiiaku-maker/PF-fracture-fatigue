"""Narrow runtime patch for consumed-cycle accounting in the legacy 2-D driver.

The v10.2.29 engine may localize a crack event inside a proposed cycle block.
The inherited driver must accumulate the engine-reported consumed cycles rather
than the originally proposed cycles. This installer replaces exactly one known
source statement and refuses to run if the inherited implementation changes.
"""
from __future__ import annotations

import inspect

from . import sharp_front

_OLD = "fatigue_cycles_total_accepted += max(float(fatigue_cycles_accepted), 0.0)"
_NEW = "fatigue_cycles_total_accepted += max(float(info.get('cycles', fatigue_cycles_accepted)), 0.0)"
_ORIGINAL_RUN_2D = None


def install_consumed_cycle_accounting() -> None:
    global _ORIGINAL_RUN_2D
    if _ORIGINAL_RUN_2D is not None:
        return
    source = inspect.getsource(sharp_front.run_2d)
    if source.count(_OLD) != 1:
        raise RuntimeError(
            "v10.2.29 consumed-cycle patch expected exactly one inherited counter statement"
        )
    patched = source.replace(_OLD, _NEW)
    namespace: dict[str, object] = {}
    exec(compile(patched, sharp_front.__file__, "exec"), sharp_front.__dict__, namespace)
    replacement = namespace.get("run_2d")
    if replacement is None:
        raise RuntimeError("v10.2.29 failed to compile patched run_2d")
    _ORIGINAL_RUN_2D = sharp_front.run_2d
    replacement.__module__ = sharp_front.__name__
    replacement.__qualname__ = sharp_front.run_2d.__qualname__
    sharp_front.run_2d = replacement


def restore_consumed_cycle_accounting() -> None:
    global _ORIGINAL_RUN_2D
    if _ORIGINAL_RUN_2D is not None:
        sharp_front.run_2d = _ORIGINAL_RUN_2D
        _ORIGINAL_RUN_2D = None


__all__ = ["install_consumed_cycle_accounting", "restore_consumed_cycle_accounting"]
