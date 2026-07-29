"""Fast, state-exact trial cloning for the v10.2.30 cyclic engine."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)


MODEL_ID = "v10.2.30_fast_independent_rng_trial_clone_v1"
_ORIGINAL_DEEPCOPY = None


def _clone_generator(rng: np.random.Generator) -> np.random.Generator:
    """Clone a Generator without invoking NumPy's SeedSequence reduce path."""
    bit_generator_type = type(rng.bit_generator)
    try:
        bit_generator = bit_generator_type()
    except Exception:
        return copy.deepcopy(rng)
    bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return np.random.Generator(bit_generator)


def fast_trial_deepcopy(self, memo: dict[int, Any]):
    cls = type(self)
    result = cls.__new__(cls)
    memo[id(self)] = result
    for key, value in self.__dict__.items():
        if isinstance(value, np.random.Generator):
            cloned = _clone_generator(value)
            memo[id(value)] = cloned
        else:
            cloned = copy.deepcopy(value, memo)
        setattr(result, key, cloned)
    result._energy_gate_provisional = True
    result._energy_gate_pending = None
    return result


def install_fast_trial_clone() -> None:
    global _ORIGINAL_DEEPCOPY
    if _ORIGINAL_DEEPCOPY is not None:
        return
    _ORIGINAL_DEEPCOPY = getattr(
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
        "__deepcopy__",
    )
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__ = (
        fast_trial_deepcopy
    )


def restore_fast_trial_clone() -> None:
    global _ORIGINAL_DEEPCOPY
    if _ORIGINAL_DEEPCOPY is None:
        return
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__ = (
        _ORIGINAL_DEEPCOPY
    )
    _ORIGINAL_DEEPCOPY = None


__all__ = [
    "MODEL_ID",
    "fast_trial_deepcopy",
    "install_fast_trial_clone",
    "restore_fast_trial_clone",
]
